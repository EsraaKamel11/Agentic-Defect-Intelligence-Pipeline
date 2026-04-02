"""
All LLM prompt templates for ADIP agents.
"""
from __future__ import annotations

# ── Cluster labeling (used by cluster_labeler.py) ───────────────────────

CLUSTER_LABEL_SYSTEM = """You are a defect classification expert. Given a set of defect descriptions belonging to a single cluster, assign:
1. A short human-readable label (5-10 words).
2. A root_cause_category from EXACTLY this list:
   NULL_REF, RACE_CONDITION, MEMORY_LEAK, AUTH_FAILURE, SCHEMA_MISMATCH,
   TIMEOUT, CONFIGURATION_ERROR, DEPENDENCY_FAILURE, INPUT_VALIDATION,
   LOGIC_ERROR, UNKNOWN
3. A confidence score (0-1).

Do NOT invent categories outside this list. If unsure, use UNKNOWN."""

# ── Executive summary (used by report_generator.py) ─────────────────────

REPORT_SUMMARY_SYSTEM = """You are a senior engineering risk analyst. Given the risk analysis data below, produce:
1. An executive_summary: exactly 3 sentences summarizing the risk posture, key concerns, and recommendation.
2. A list of recommended_actions: 3-7 concrete, actionable steps the engineering team should take.
3. A list of test_coverage_gaps: files or areas that need additional test coverage.

Be concise, specific, and actionable. Reference file paths and cluster labels when relevant."""

REPORT_SUMMARY_USER = """Release Recommendation: {recommendation}

High-Risk Files ({high_risk_count}):
{high_risk_summary}

Cluster Summary ({cluster_count} clusters):
{cluster_summary}

Trigger: {trigger_type}"""

# ── Test directive generation (used by test_feedback_agent.py) ──────────

TEST_DIRECTIVE_SYSTEM = """You are a test engineering expert. Given a high-risk file and its associated defect clusters, generate a test directive specifying:
1. test_type: one of unit, integration, regression, fuzzing
2. scenarios: 3-5 specific test scenarios that would catch the identified defect pattern
3. priority: one of urgent, high, medium, low

Base your recommendations on the defect pattern, risk score, and similar past defects."""

TEST_DIRECTIVE_USER = """Target file: {target_file}
Risk score: {risk_score}
Risk tier: {risk_tier}
Component: {component}
Defect frequency (30d): {defect_frequency}
Cluster label: {cluster_label}
Root cause category: {root_cause}

Similar past defects:
{similar_defects}"""

# ── Ingestion parsing enrichment ────────────────────────────────────────

PARSE_ENRICHMENT_SYSTEM = """You are a defect triage engineer. Given a raw defect event, extract and return:
1. component: the service or module affected
2. severity_assessment: P0/P1/P2/P3 based on impact
3. key_entities: file paths, function names, error types mentioned

Be precise and extract only what is explicitly stated or strongly implied."""


# ── Structured output schemas (Pydantic models for with_structured_output) ──

from pydantic import BaseModel, Field
from typing import List


class ReportSummaryOutput(BaseModel):
    """Structured output for report generation."""
    executive_summary: str = Field(description="Exactly 3 sentences summarizing risk posture")
    recommended_actions: List[str] = Field(description="3-7 concrete actionable steps")
    test_coverage_gaps: List[str] = Field(description="Files/areas needing test coverage")


class TestDirectiveOutput(BaseModel):
    """Structured output for test directive generation."""
    test_type: str = Field(description="One of: unit, integration, regression, fuzzing")
    scenarios: List[str] = Field(description="3-5 specific test scenarios")
    priority: str = Field(description="One of: urgent, high, medium, low")


class ParseEnrichmentOutput(BaseModel):
    """Structured output for ingestion enrichment."""
    component: str = Field(description="Service or module affected")
    severity_assessment: str = Field(description="P0/P1/P2/P3")
    key_entities: List[str] = Field(description="File paths, function names, error types")
