"""
Deterministic PROCEED / CONDITIONAL / HOLD rule engine.
No LLM involvement — pure logic. Runs BEFORE any LLM call.
"""
from __future__ import annotations

import logging
from typing import List

from adip.config.settings import settings
from adip.graph.state import (
    ClusterResult,
    FileRiskScore,
    ReleaseRecommendation,
    DefectSeverity,
)

logger = logging.getLogger(__name__)


def evaluate(
    risk_scores: List[FileRiskScore],
    clusters: List[ClusterResult],
) -> ReleaseRecommendation:
    """
    Rule engine — deterministic release recommendation.

    HOLD:
      - Any P0 cluster with recurrence_count > 3
      - Any file with risk_score > 0.8

    CONDITIONAL:
      - Any file with risk_score 0.5–0.8
      - Any P1 cluster with recurrence_count > 5

    PROCEED:
      - All other cases
    """
    reasons: List[str] = []

    # ── HOLD checks ──────────────────────────────────────────────────────
    for cluster in clusters:
        sev = cluster.root_cause_category  # We check severity via member events
        if _is_p0_cluster(cluster) and cluster.recurrence_count > settings.recurrence_hold_count:
            reasons.append(
                f"HOLD: P0 cluster '{cluster.label}' has recurrence_count={cluster.recurrence_count}"
            )
            logger.warning(reasons[-1])
            return ReleaseRecommendation.HOLD

    for score in risk_scores:
        if score.risk_score > settings.hold_threshold:
            reasons.append(
                f"HOLD: {score.file_path} risk_score={score.risk_score:.3f}"
            )
            logger.warning(reasons[-1])
            return ReleaseRecommendation.HOLD

    # ── CONDITIONAL checks ───────────────────────────────────────────────
    for score in risk_scores:
        if settings.conditional_threshold <= score.risk_score <= settings.hold_threshold:
            reasons.append(
                f"CONDITIONAL: {score.file_path} risk_score={score.risk_score:.3f}"
            )
            logger.info(reasons[-1])
            return ReleaseRecommendation.CONDITIONAL

    for cluster in clusters:
        if _is_p1_cluster(cluster) and cluster.recurrence_count > 5:
            reasons.append(
                f"CONDITIONAL: P1 cluster '{cluster.label}' recurrence={cluster.recurrence_count}"
            )
            logger.info(reasons[-1])
            return ReleaseRecommendation.CONDITIONAL

    # ── PROCEED ──────────────────────────────────────────────────────────
    logger.info("Rule engine: PROCEED — no hold/conditional triggers")
    return ReleaseRecommendation.PROCEED


def _is_p0_cluster(cluster: ClusterResult) -> bool:
    """Check if cluster is P0 severity — uses weight as proxy or metadata."""
    # In a full implementation, we'd check member event severities.
    # Here we use weight > 0.9 as a heuristic for critical clusters.
    return cluster.weight >= 0.9


def _is_p1_cluster(cluster: ClusterResult) -> bool:
    """Check if cluster is P1 severity."""
    return 0.7 <= cluster.weight < 0.9
