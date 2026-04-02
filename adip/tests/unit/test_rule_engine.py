"""Unit tests for scoring/rule_engine.py."""
from adip.graph.state import (
    ClusterResult,
    FileRiskScore,
    ReleaseRecommendation,
    RiskTier,
    RootCauseCategory,
)
from adip.scoring.rule_engine import evaluate


class TestRuleEngine:
    def test_proceed_when_all_clear(self):
        scores = [
            FileRiskScore(file_path="a.py", risk_score=0.2, risk_tier=RiskTier.LOW),
        ]
        clusters = [
            ClusterResult(label="minor", weight=0.3, recurrence_count=1),
        ]
        assert evaluate(scores, clusters) == ReleaseRecommendation.PROCEED

    def test_hold_on_high_risk_score(self):
        scores = [
            FileRiskScore(file_path="critical.py", risk_score=0.85, risk_tier=RiskTier.CRITICAL),
        ]
        assert evaluate(scores, []) == ReleaseRecommendation.HOLD

    def test_hold_on_p0_recurrence(self):
        scores = []
        clusters = [
            ClusterResult(label="P0 cluster", weight=0.95, recurrence_count=5),
        ]
        assert evaluate(scores, clusters) == ReleaseRecommendation.HOLD

    def test_conditional_on_medium_risk(self):
        scores = [
            FileRiskScore(file_path="risky.py", risk_score=0.65, risk_tier=RiskTier.HIGH),
        ]
        assert evaluate(scores, []) == ReleaseRecommendation.CONDITIONAL

    def test_empty_inputs_proceed(self):
        assert evaluate([], []) == ReleaseRecommendation.PROCEED
