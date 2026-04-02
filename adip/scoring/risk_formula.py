"""
Five-feature risk scoring formula with exact weights.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from adip.config.settings import settings
from adip.graph.state import FileRiskScore, RiskTier


def _normalize(value: float, max_val: float) -> float:
    """Min-max normalize to [0, 1]."""
    if max_val <= 0:
        return 0.0
    return min(max(value / max_val, 0.0), 1.0)


def compute_risk_score(
    file_path: str,
    component: str,
    defect_frequency_30d: float,
    git_churn: float,
    coverage_gap: float,
    cluster_severity_weight: float,
    recency_decay: float,
    max_defect_freq: float = 20.0,
    max_git_churn: float = 100.0,
    weights: Optional[Dict[str, float]] = None,
) -> FileRiskScore:
    """
    Exact five-feature formula:
        risk_score = (
            defect_frequency_30d_normalized * 0.35 +
            git_churn_normalized          * 0.25 +
            coverage_gap_normalized       * 0.20 +
            cluster_severity_weight       * 0.15 +
            recency_decay                 * 0.05
        )
    """
    w = weights or {
        "defect_frequency": settings.weight_defect_frequency,
        "git_churn": settings.weight_git_churn,
        "coverage_gap": settings.weight_coverage_gap,
        "cluster_severity": settings.weight_cluster_severity,
        "recency_decay": settings.weight_recency_decay,
    }

    freq_norm = _normalize(defect_frequency_30d, max_defect_freq)
    churn_norm = _normalize(git_churn, max_git_churn)
    cov_norm = min(max(coverage_gap, 0.0), 1.0)       # already 0-1
    sev_norm = min(max(cluster_severity_weight, 0.0), 1.0)
    rec_norm = min(max(recency_decay, 0.0), 1.0)

    risk_score = (
        freq_norm * w["defect_frequency"]
        + churn_norm * w["git_churn"]
        + cov_norm * w["coverage_gap"]
        + sev_norm * w["cluster_severity"]
        + rec_norm * w["recency_decay"]
    )
    risk_score = min(max(risk_score, 0.0), 1.0)

    # Determine tier
    if risk_score >= settings.hold_threshold:
        tier = RiskTier.CRITICAL
    elif risk_score >= settings.conditional_threshold:
        tier = RiskTier.HIGH
    elif risk_score >= 0.3:
        tier = RiskTier.MEDIUM
    else:
        tier = RiskTier.LOW

    return FileRiskScore(
        file_path=file_path,
        component=component,
        risk_score=round(risk_score, 4),
        risk_tier=tier,
        defect_frequency_30d=defect_frequency_30d,
        git_churn_score=git_churn,
        coverage_gap_score=coverage_gap,
        cluster_severity_weight=cluster_severity_weight,
        recency_decay=recency_decay,
        confidence=_confidence(defect_frequency_30d),
    )


def _confidence(freq: float) -> float:
    """Higher data volume → higher confidence, capped at 0.95."""
    return min(0.95, 0.3 + 0.1 * math.log1p(freq))
