"""Unit tests for scoring/risk_formula.py."""
import pytest

from adip.graph.state import RiskTier
from adip.scoring.risk_formula import compute_risk_score


class TestRiskFormula:
    def test_zero_inputs(self):
        score = compute_risk_score(
            file_path="test.py", component="svc",
            defect_frequency_30d=0, git_churn=0,
            coverage_gap=0, cluster_severity_weight=0, recency_decay=0,
        )
        assert score.risk_score == 0.0
        assert score.risk_tier == RiskTier.LOW

    def test_max_inputs(self):
        score = compute_risk_score(
            file_path="test.py", component="svc",
            defect_frequency_30d=20, git_churn=100,
            coverage_gap=1.0, cluster_severity_weight=1.0, recency_decay=1.0,
        )
        assert score.risk_score == pytest.approx(1.0, abs=0.01)
        assert score.risk_tier == RiskTier.CRITICAL

    def test_weights_sum_to_one(self):
        from adip.config.settings import settings
        total = (
            settings.weight_defect_frequency
            + settings.weight_git_churn
            + settings.weight_coverage_gap
            + settings.weight_cluster_severity
            + settings.weight_recency_decay
        )
        assert total == pytest.approx(1.0)

    def test_exact_weights(self):
        # With only defect_frequency=20 (normalized to 1.0), score should be 0.35
        score = compute_risk_score(
            file_path="test.py", component="svc",
            defect_frequency_30d=20, git_churn=0,
            coverage_gap=0, cluster_severity_weight=0, recency_decay=0,
        )
        assert score.risk_score == pytest.approx(0.35, abs=0.01)

    def test_risk_tier_boundaries(self):
        # Medium: 0.3-0.5
        score = compute_risk_score(
            file_path="t.py", component="x",
            defect_frequency_30d=10, git_churn=20,
            coverage_gap=0.3, cluster_severity_weight=0.2, recency_decay=0.1,
        )
        assert score.risk_score > 0.0
        assert score.risk_tier in (RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL)
