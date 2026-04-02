"""
LangGraph routing functions — conditional edges in the graph.
"""
from __future__ import annotations

from typing import List, Literal

from langgraph.types import Send

from adip.config.settings import settings
from adip.graph.state import ADIPState, ReleaseRecommendation, TriggerType


def route_after_indexing(state: ADIPState) -> str:
    """
    Decide whether to cluster or skip straight to scoring.

    → "cluster_defects" if:
      - trigger is SCHEDULED_BATCH
      - indexed_count >= recluster_threshold (default 5)
      - any P0 event detected
    → "score_risks" otherwise (use existing clusters)
    """
    trigger = state.get("trigger_type", "")

    # Batch mode always re-clusters
    if trigger == TriggerType.SCHEDULED_BATCH or trigger == "SCHEDULED_BATCH":
        return "cluster_defects"

    # Enough new data to warrant re-clustering
    indexed_count = state.get("indexed_count", 0)
    if indexed_count >= settings.recluster_threshold:
        return "cluster_defects"

    # Any P0 event forces re-clustering
    for event in state.get("defect_events", []):
        if event.get("severity") in ("P0", "CRITICAL"):
            return "cluster_defects"

    return "score_risks"


def route_after_report(state: ADIPState) -> List[Send]:
    """
    Parallel fan-out: alert_dispatch + test_feedback run concurrently.
    """
    return [
        Send("alert_dispatch", state),
        Send("test_feedback", state),
    ]


def route_after_actions(state: ADIPState) -> Literal["human_review", "__end__"]:
    """
    Route to human review if recommendation is HOLD, else end.
    """
    recommendation = state.get("release_recommendation", "PROCEED")

    if recommendation in (ReleaseRecommendation.HOLD, "HOLD"):
        return "human_review"

    return "__end__"
