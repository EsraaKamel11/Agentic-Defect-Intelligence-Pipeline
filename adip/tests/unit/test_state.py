"""Unit tests for graph/state.py — all Pydantic models and enums."""
import pytest
from datetime import datetime

from adip.graph.state import (
    ClusterResult,
    DataFreshnessStatus,
    DefectEvent,
    DefectSeverity,
    FileRiskScore,
    IndexingResult,
    NotificationResult,
    ReleaseRecommendation,
    RiskReport,
    RiskTier,
    RootCauseCategory,
    TestGenerationDirective,
    TriggerType,
)


class TestEnums:
    def test_trigger_types(self):
        assert len(TriggerType) == 3
        assert TriggerType.STREAM_EVENT.value == "STREAM_EVENT"

    def test_severity_levels(self):
        assert len(DefectSeverity) == 4

    def test_risk_tiers(self):
        assert len(RiskTier) == 4

    def test_release_recommendations(self):
        assert len(ReleaseRecommendation) == 3

    def test_root_cause_categories_exactly_11(self):
        assert len(RootCauseCategory) == 11
        expected = {
            "NULL_REF", "RACE_CONDITION", "MEMORY_LEAK", "AUTH_FAILURE",
            "SCHEMA_MISMATCH", "TIMEOUT", "CONFIGURATION_ERROR",
            "DEPENDENCY_FAILURE", "INPUT_VALIDATION", "LOGIC_ERROR", "UNKNOWN",
        }
        assert {e.value for e in RootCauseCategory} == expected


class TestDefectEvent:
    def test_create_minimal(self):
        event = DefectEvent(source="cicd", raw_content="test")
        assert event.source == "cicd"
        assert event.severity == DefectSeverity.P3

    def test_create_full(self):
        event = DefectEvent(
            source="sentry",
            raw_content="Error occurred",
            normalized_content="[SENTRY] Error occurred",
            component="auth-service",
            file_path="src/auth.py",
            severity=DefectSeverity.P0,
            stack_trace="Traceback...",
        )
        assert event.severity == DefectSeverity.P0
        assert event.file_path == "src/auth.py"


class TestFileRiskScore:
    def test_score_bounds(self):
        score = FileRiskScore(file_path="test.py", risk_score=0.75)
        assert 0.0 <= score.risk_score <= 1.0

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(Exception):
            FileRiskScore(file_path="test.py", risk_score=1.5)


class TestRiskReport:
    def test_create_report(self):
        report = RiskReport(trigger_type=TriggerType.SCHEDULED_BATCH)
        assert report.release_recommendation == ReleaseRecommendation.PROCEED
        assert isinstance(report.generated_at, datetime)


class TestClusterResult:
    def test_create_cluster(self):
        cluster = ClusterResult(label="NullRef in auth")
        assert cluster.root_cause_category == RootCauseCategory.UNKNOWN
        assert cluster.weight == 1.0


class TestDataFreshness:
    def test_fresh_by_default(self):
        status = DataFreshnessStatus()
        assert status.all_fresh is True
        assert status.stale_sources == []
