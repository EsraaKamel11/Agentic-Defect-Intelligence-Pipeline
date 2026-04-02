"""
Supervisor — orchestration logic shared by all agents.
Provides error handling, logging, and state management helpers.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict

from adip.graph.state import ADIPState

logger = logging.getLogger(__name__)


def init_run(trigger_type: str) -> Dict[str, Any]:
    """Create initial state for a new pipeline run."""
    return {
        "trigger_type": trigger_type,
        "run_id": str(uuid.uuid4()),
        "triggered_at": datetime.utcnow().isoformat(),
        "raw_events": [],
        "defect_events": [],
        "clusters": [],
        "risk_scores": [],
        "test_directives": [],
        "notifications": [],
        "errors": [],
        "clustering_skipped": False,
        "human_review_required": False,
        "indexed_count": 0,
    }


def append_error(state: ADIPState, error: str) -> ADIPState:
    """Safely append an error to state."""
    errors = list(state.get("errors", []))
    errors.append(f"[{datetime.utcnow().isoformat()}] {error}")
    return {**state, "errors": errors}


class AgentTimer:
    """Context manager for timing agent execution."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        logger.info("▶ %s started", self.agent_name)
        return self

    def __exit__(self, *exc_info):
        elapsed = time.monotonic() - self._start
        if exc_info[0]:
            logger.error("✖ %s failed after %.2fs: %s", self.agent_name, elapsed, exc_info[1])
        else:
            logger.info("✔ %s completed in %.2fs", self.agent_name, elapsed)
