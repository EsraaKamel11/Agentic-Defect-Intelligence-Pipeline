"""
Publish TestGenerationDirective to Redis channel.
Falls back to in-memory queue if Redis is unavailable.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Dict, List

from adip.config.settings import settings

logger = logging.getLogger(__name__)

# In-memory fallback queue
_memory_queue: deque = deque(maxlen=1000)


class RedisPublisher:
    """Publish directives to Redis pub/sub or in-memory fallback."""

    def __init__(self):
        self._client = None
        self._use_redis = False
        self._try_connect()

    def _try_connect(self):
        if settings.mock_mode:
            logger.info("Mock mode — using in-memory queue instead of Redis")
            return
        try:
            import redis as redis_lib
            self._client = redis_lib.Redis.from_url(
                settings.redis_url, decode_responses=True, socket_timeout=3
            )
            self._client.ping()
            self._use_redis = True
            logger.info("Connected to Redis at %s", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); using in-memory queue", exc)

    def publish(self, directive: Dict[str, Any]) -> bool:
        """Publish a single directive."""
        payload = json.dumps(directive, default=str)
        if self._use_redis:
            try:
                self._client.publish(settings.redis_directives_channel, payload)
                logger.debug("Published to Redis channel %s", settings.redis_directives_channel)
                return True
            except Exception as exc:
                logger.error("Redis publish failed: %s", exc)
                # Fall through to memory queue

        _memory_queue.append(directive)
        logger.debug("Queued directive in memory (queue size: %d)", len(_memory_queue))
        return True

    def publish_batch(self, directives: List[Dict[str, Any]]) -> int:
        """Publish multiple directives. Returns count published."""
        count = 0
        for d in directives:
            if self.publish(d):
                count += 1
        return count

    @staticmethod
    def get_memory_queue() -> List[Dict[str, Any]]:
        """Access the in-memory fallback queue (for testing / mock mode)."""
        return list(_memory_queue)

    def close(self):
        if self._client:
            self._client.close()
