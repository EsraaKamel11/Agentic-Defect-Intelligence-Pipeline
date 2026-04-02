"""
ADIP LangGraph State Schema.

All Pydantic models that flow through the graph plus the
top-level ADIPState TypedDict consumed by LangGraph nodes.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class TriggerType(str, enum.Enum):
    STREAM_EVENT = "STREAM_EVENT"
    SCHEDULED_BATCH = "SCHEDULED_BATCH"
    PR_WEBHOOK = "PR_WEBHOOK"


class DefectSeverity(str, enum.Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class RiskTier(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReleaseRecommendation(str, enum.Enum):
    PROCEED = "PROCEED"
    CONDITIONAL = "CONDITIONAL"
    HOLD = "HOLD"


class RootCauseCategory(str, enum.Enum):
    """Exactly 11 root-cause categories — the LLM taxonomy is constrained to these."""
    NULL_REF = "NULL_REF"
    RACE_CONDITION = "RACE_CONDITION"
    MEMORY_LEAK = "MEMORY_LEAK"
    AUTH_FAILURE = "AUTH_FAILURE"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    TIMEOUT = "TIMEOUT"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    INPUT_VALIDATION = "INPUT_VALIDATION"
    LOGIC_ERROR = "LOGIC_ERROR"
    UNKNOWN = "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════
# Core domain models
# ═══════════════════════════════════════════════════════════════════════════

class DefectEvent(BaseModel):
    """A single defect event normalised from any source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(
        ...,
        description="Origin system: cicd | jira | sentry | datadog",
    )
    raw_content: str = Field(..., description="Verbatim payload from source")
    normalized_content: str = Field(
        default="", description="Cleaned / structured text"
    )
    component: str = Field(default="unknown", description="Service / module tag")
    file_path: Optional[str] = Field(
        default=None, description="Primary file associated with the defect"
    )
    severity: DefectSeverity = Field(default=DefectSeverity.P3)
    stack_trace: Optional[str] = Field(
        default=None, description="Extracted stack trace if available"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding_id: Optional[str] = Field(
        default=None, description="ID in vector store after indexing"
    )

    class Config:
        use_enum_values = True


class ClusterResult(BaseModel):
    """A single defect cluster produced by HDBSCAN + LLM labeling."""
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(..., description="Human-readable cluster label")
    root_cause_category: RootCauseCategory = Field(default=RootCauseCategory.UNKNOWN)
    member_count: int = Field(default=0)
    centroid_embedding: Optional[List[float]] = Field(
        default=None, description="Mean embedding of cluster members"
    )
    recurrence_count: int = Field(
        default=0, description="Times this pattern has appeared"
    )
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    weight: float = Field(
        default=1.0,
        description="Exponential decay weight — decreases with age",
    )

    class Config:
        use_enum_values = True


class FileRiskScore(BaseModel):
    """Risk score for a single file / component."""
    file_path: str
    component: str = "unknown"
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite risk 0–1"
    )
    risk_tier: RiskTier = Field(default=RiskTier.LOW)
    defect_frequency_30d: float = Field(default=0.0)
    git_churn_score: float = Field(default=0.0)
    coverage_gap_score: float = Field(default=0.0)
    cluster_severity_weight: float = Field(default=0.0)
    recency_decay: float = Field(default=0.0)
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in this score"
    )

    class Config:
        use_enum_values = True


class RiskReport(BaseModel):
    """Top-level risk report produced once per pipeline run."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: TriggerType
    release_recommendation: ReleaseRecommendation = Field(
        default=ReleaseRecommendation.PROCEED
    )
    executive_summary: str = Field(
        default="", description="3-sentence max executive summary"
    )
    high_risk_files: List[FileRiskScore] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    test_coverage_gaps: List[str] = Field(default_factory=list)
    cluster_summary: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class TestGenerationDirective(BaseModel):
    """Instruction for downstream test-generation tooling (e.g. AQAF)."""
    directive_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_file: str
    test_type: str = Field(
        ...,
        description="unit | integration | regression | fuzzing",
    )
    scenarios: List[str] = Field(default_factory=list)
    priority: str = Field(
        default="medium",
        description="urgent | high | medium | low",
    )
    cluster_id: Optional[str] = Field(
        default=None, description="Originating cluster if applicable"
    )
    published_at: datetime = Field(default_factory=datetime.utcnow)


class DataFreshnessStatus(BaseModel):
    """Staleness check across all data sources."""
    kafka_stream_age_seconds: float = Field(default=0.0)
    jira_poll_age_seconds: float = Field(default=0.0)
    vector_store_upsert_lag_seconds: float = Field(default=0.0)
    all_fresh: bool = Field(default=True)
    stale_sources: List[str] = Field(default_factory=list)


class IndexingResult(BaseModel):
    """Result from the RAG indexer agent."""
    indexed_count: int = 0
    skipped_duplicates: int = 0
    errors: List[str] = Field(default_factory=list)


class NotificationResult(BaseModel):
    """Result from a single alert dispatch action."""
    channel: str  # jira | slack | email
    success: bool
    detail: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph State — TypedDict consumed by every node
# ═══════════════════════════════════════════════════════════════════════════

class ADIPState(TypedDict, total=False):
    """
    Top-level state flowing through the LangGraph graph.
    Every key is optional (total=False) so nodes only write what they own.
    """
    # ── Trigger metadata ─────────────────────────────────────────────────
    trigger_type: str                        # TriggerType value
    run_id: str                              # unique pipeline run id
    triggered_at: str                        # ISO-8601

    # ── Ingestion ────────────────────────────────────────────────────────
    raw_events: List[Dict[str, Any]]         # raw payloads before parsing
    defect_events: List[Dict[str, Any]]      # serialised DefectEvent dicts

    # ── RAG indexing ─────────────────────────────────────────────────────
    indexing_result: Dict[str, Any]          # serialised IndexingResult
    indexed_count: int

    # ── Clustering ───────────────────────────────────────────────────────
    clusters: List[Dict[str, Any]]           # serialised ClusterResult dicts
    clustering_skipped: bool

    # ── Risk scoring ─────────────────────────────────────────────────────
    risk_scores: List[Dict[str, Any]]        # serialised FileRiskScore dicts

    # ── Report ───────────────────────────────────────────────────────────
    risk_report: Dict[str, Any]              # serialised RiskReport
    release_recommendation: str              # ReleaseRecommendation value

    # ── Test feedback ────────────────────────────────────────────────────
    test_directives: List[Dict[str, Any]]    # serialised TestGenerationDirective

    # ── Alerts ───────────────────────────────────────────────────────────
    notifications: List[Dict[str, Any]]      # serialised NotificationResult

    # ── Freshness ────────────────────────────────────────────────────────
    freshness: Dict[str, Any]                # serialised DataFreshnessStatus

    # ── Human-in-the-loop ────────────────────────────────────────────────
    human_review_required: bool
    human_decision: Optional[str]            # approve | reject | override

    # ── Errors / diagnostics ─────────────────────────────────────────────
    errors: List[str]
