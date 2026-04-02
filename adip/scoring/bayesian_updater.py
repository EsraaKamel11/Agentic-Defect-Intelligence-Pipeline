"""
Bayesian running-average weight update from test outcome feedback.
Persists updated weights to PostgreSQL weight_history table.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

from adip.config.settings import settings

logger = logging.getLogger(__name__)

# Default feature weights — matches the specification
DEFAULT_WEIGHTS: Dict[str, float] = {
    "defect_frequency": 0.35,
    "git_churn": 0.25,
    "coverage_gap": 0.20,
    "cluster_severity": 0.15,
    "recency_decay": 0.05,
}

# In-memory current weights (loaded from DB on startup)
_current_weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
_update_count: int = 0


def get_current_weights() -> Dict[str, float]:
    """Return the current (possibly Bayesian-updated) weights."""
    return dict(_current_weights)


def bayesian_update(
    feature: str,
    prediction_correct: bool,
    learning_rate: float = 0.01,
) -> Dict[str, float]:
    """
    Bayesian running-average weight update.

    If the prediction using this feature was correct, slightly increase its weight.
    If incorrect, slightly decrease and redistribute.
    Weights always sum to 1.0.
    """
    global _current_weights, _update_count

    if feature not in _current_weights:
        logger.warning("Unknown feature '%s'; skipping update", feature)
        return get_current_weights()

    delta = learning_rate if prediction_correct else -learning_rate
    _current_weights[feature] = max(0.01, _current_weights[feature] + delta)

    # Re-normalize so weights sum to 1.0
    total = sum(_current_weights.values())
    _current_weights = {k: v / total for k, v in _current_weights.items()}
    _update_count += 1

    logger.info(
        "Bayesian update #%d: feature=%s correct=%s → weights=%s",
        _update_count, feature, prediction_correct, _current_weights,
    )
    return get_current_weights()


def bulk_update_from_outcome(
    defect_found: bool,
    risk_score: float,
    feature_contributions: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Update all weights based on a test outcome.
    If risk was high and defect was found → correct prediction → boost all.
    If risk was high and no defect → false positive → reduce top contributors.
    """
    high_risk = risk_score >= settings.conditional_threshold
    correct = (high_risk and defect_found) or (not high_risk and not defect_found)

    if feature_contributions:
        # Update only the features that contributed most
        sorted_features = sorted(
            feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for feat, _ in sorted_features[:3]:
            bayesian_update(feat, correct)
    else:
        # Update all features equally
        for feat in list(_current_weights.keys()):
            bayesian_update(feat, correct, learning_rate=0.005)

    return get_current_weights()


async def persist_weights(db) -> None:
    """Save current weights to the weight_history table."""
    try:
        await db.store_weight_history(
            weights=get_current_weights(),
            update_count=_update_count,
            timestamp=datetime.utcnow(),
        )
        logger.info("Persisted weight snapshot #%d", _update_count)
    except Exception as exc:
        logger.warning("Failed to persist weights: %s", exc)


async def load_weights(db) -> None:
    """Load latest weights from DB on startup."""
    global _current_weights, _update_count
    try:
        record = await db.get_latest_weights()
        if record:
            _current_weights = record["weights"]
            _update_count = record.get("update_count", 0)
            logger.info("Loaded weights from DB: %s (update #%d)", _current_weights, _update_count)
    except Exception as exc:
        logger.warning("Could not load weights from DB (%s); using defaults", exc)
        _current_weights = dict(DEFAULT_WEIGHTS)
