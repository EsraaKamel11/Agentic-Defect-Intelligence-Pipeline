"""
Track test outcomes and trigger Bayesian weight updates.
Subscribes to test outcome events from AQAF (or mock).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from adip.config.settings import settings
from adip.scoring.bayesian_updater import bulk_update_from_outcome, persist_weights

logger = logging.getLogger(__name__)

# In-memory outcome store (fallback)
_outcomes: List[Dict[str, Any]] = []


class OutcomeTracker:
    """Track test outcomes and update scoring weights."""

    def __init__(self, db=None):
        self._db = db
        self._subscriber = None
        self._use_redis = False
        self._try_subscribe()

    def _try_subscribe(self):
        if settings.mock_mode:
            return
        try:
            import redis as redis_lib
            client = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=3)
            client.ping()
            self._subscriber = client.pubsub()
            self._subscriber.subscribe("adip:test_outcomes")
            self._use_redis = True
            logger.info("Subscribed to adip:test_outcomes")
        except Exception as exc:
            logger.warning("Redis subscribe failed (%s); outcomes tracked in-memory only", exc)

    async def record_outcome(
        self,
        directive_id: str,
        defect_found: bool,
        test_result: str,
        risk_score: float = 0.5,
    ) -> Dict[str, float]:
        """
        Record a test outcome and trigger Bayesian weight update.
        Returns updated weights.
        """
        outcome = {
            "directive_id": directive_id,
            "defect_found": defect_found,
            "test_result": test_result,
            "risk_score": risk_score,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        _outcomes.append(outcome)

        # Bayesian update
        updated_weights = bulk_update_from_outcome(
            defect_found=defect_found,
            risk_score=risk_score,
        )

        # Persist to DB if available
        if self._db:
            try:
                await self._db.store_test_outcome(outcome)
                await persist_weights(self._db)
            except Exception as exc:
                logger.warning("Failed to persist outcome: %s", exc)

        logger.info(
            "Recorded outcome for directive %s: defect_found=%s, updated weights=%s",
            directive_id, defect_found, updated_weights,
        )
        return updated_weights

    def poll_outcomes(self) -> List[Dict[str, Any]]:
        """Poll Redis for new test outcome messages."""
        if not self._use_redis or not self._subscriber:
            return []
        messages = []
        try:
            msg = self._subscriber.get_message(timeout=1.0)
            while msg:
                if msg["type"] == "message":
                    messages.append(json.loads(msg["data"]))
                msg = self._subscriber.get_message(timeout=0.1)
        except Exception as exc:
            logger.error("Redis poll error: %s", exc)
        return messages

    @staticmethod
    def get_all_outcomes() -> List[Dict[str, Any]]:
        """Return all tracked outcomes (in-memory)."""
        return list(_outcomes)
