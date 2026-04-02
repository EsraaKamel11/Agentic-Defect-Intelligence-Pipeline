"""
LLM-assisted cluster labeling constrained to the 11-category RootCauseCategory taxonomy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from adip.graph.state import RootCauseCategory

logger = logging.getLogger(__name__)

# ── Structured output schema for LLM ────────────────────────────────────

class ClusterLabel(BaseModel):
    """LLM output schema — constrained to RootCauseCategory enum."""
    label: str = Field(description="Short human-readable cluster label (5-10 words)")
    root_cause_category: RootCauseCategory = Field(
        description="One of the 11 root-cause categories"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this classification (0-1)",
    )


LABELING_SYSTEM_PROMPT = """You are a defect classification expert. Given a set of defect descriptions belonging to a single cluster, assign:
1. A short human-readable label (5-10 words).
2. A root_cause_category from EXACTLY this list:
   NULL_REF, RACE_CONDITION, MEMORY_LEAK, AUTH_FAILURE, SCHEMA_MISMATCH,
   TIMEOUT, CONFIGURATION_ERROR, DEPENDENCY_FAILURE, INPUT_VALIDATION,
   LOGIC_ERROR, UNKNOWN
3. A confidence score (0-1).

Do NOT invent categories outside this list. If unsure, use UNKNOWN."""


async def label_cluster(
    llm,
    member_texts: List[str],
    max_samples: int = 10,
) -> ClusterLabel:
    """
    Use LLM with structured output to label a single cluster.
    """
    sample = member_texts[:max_samples]
    content = "Cluster member defects:\n" + "\n---\n".join(sample)

    try:
        structured_llm = llm.with_structured_output(ClusterLabel)
        result = await structured_llm.ainvoke(
            [
                {"role": "system", "content": LABELING_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )
        return result
    except Exception as exc:
        logger.warning("LLM labeling failed (%s); defaulting to UNKNOWN", exc)
        return ClusterLabel(
            label="Unclassified cluster",
            root_cause_category=RootCauseCategory.UNKNOWN,
            confidence=0.0,
        )


def should_relabel(
    old_member_ids: set,
    new_member_ids: set,
    threshold: float = 0.20,
) -> bool:
    """Re-label only when cluster membership changes >20%."""
    if not old_member_ids:
        return True
    overlap = len(old_member_ids & new_member_ids)
    total = len(old_member_ids | new_member_ids)
    change_ratio = 1.0 - (overlap / max(total, 1))
    return change_ratio > threshold
