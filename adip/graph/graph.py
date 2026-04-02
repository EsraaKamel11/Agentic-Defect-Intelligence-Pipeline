"""
LangGraph graph definition — the main ADIP pipeline.

START → ingest_and_parse → index_to_rag → [conditional] cluster_defects
→ score_risks → generate_report → [parallel] alert_dispatch + test_feedback
→ [conditional] human_review → END
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from adip.agents.alert_dispatcher import alert_dispatch
from adip.agents.defect_clusterer import cluster_defects
from adip.agents.ingestion_parser import ingest_and_parse
from adip.agents.rag_indexer import index_to_rag
from adip.agents.report_generator import generate_report
from adip.agents.risk_scorer import score_risks
from adip.agents.test_feedback_agent import test_feedback
from adip.graph.routers import (
    route_after_actions,
    route_after_indexing,
    route_after_report,
)
from adip.graph.state import ADIPState

logger = logging.getLogger(__name__)


async def human_review(state: ADIPState) -> Dict[str, Any]:
    """
    Human-in-the-loop node — pauses for human decision.
    In production, interrupt_before is set so the graph pauses here.
    """
    logger.info(
        "HUMAN REVIEW REQUIRED — recommendation: %s",
        state.get("release_recommendation", "?"),
    )
    # In real usage, the graph is interrupted before this node.
    # When resumed, human_decision is set in state.
    decision = state.get("human_decision", "pending")
    logger.info("Human decision: %s", decision)
    return {"human_decision": decision}


def _merge_parallel_results(state: ADIPState) -> Dict[str, Any]:
    """
    Merge node after parallel alert_dispatch + test_feedback.
    LangGraph handles merging automatically for Send() patterns,
    but we provide an explicit merge point for routing.
    """
    return {}


def build_graph() -> StateGraph:
    """
    Construct and compile the ADIP LangGraph pipeline.
    Returns compiled graph ready for invocation.
    """
    graph = StateGraph(ADIPState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("ingest_and_parse", ingest_and_parse)
    graph.add_node("index_to_rag", index_to_rag)
    graph.add_node("cluster_defects", cluster_defects)
    graph.add_node("score_risks", score_risks)
    graph.add_node("generate_report", generate_report)
    graph.add_node("alert_dispatch", alert_dispatch)
    graph.add_node("test_feedback", test_feedback)
    graph.add_node("merge_actions", _merge_parallel_results)
    graph.add_node("human_review", human_review)

    # ── Edges ────────────────────────────────────────────────────────────
    # START → ingest
    graph.add_edge(START, "ingest_and_parse")

    # ingest → index
    graph.add_edge("ingest_and_parse", "index_to_rag")

    # index → conditional: cluster or score
    graph.add_conditional_edges(
        "index_to_rag",
        route_after_indexing,
        {
            "cluster_defects": "cluster_defects",
            "score_risks": "score_risks",
        },
    )

    # cluster → score
    graph.add_edge("cluster_defects", "score_risks")

    # score → report
    graph.add_edge("score_risks", "generate_report")

    # report → parallel fan-out
    graph.add_conditional_edges(
        "generate_report",
        route_after_report,
        ["alert_dispatch", "test_feedback"],
    )

    # parallel → merge
    graph.add_edge("alert_dispatch", "merge_actions")
    graph.add_edge("test_feedback", "merge_actions")

    # merge → conditional: human review or end
    graph.add_conditional_edges(
        "merge_actions",
        route_after_actions,
        {
            "human_review": "human_review",
            "__end__": END,
        },
    )

    # human_review → end
    graph.add_edge("human_review", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with optional checkpointer and HITL interrupt."""
    graph = build_graph()

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    # HITL: interrupt before human_review so operator can approve/reject
    compile_kwargs["interrupt_before"] = ["human_review"]

    return graph.compile(**compile_kwargs)


def get_checkpointer():
    """Get a checkpointer — PostgreSQL preferred, InMemorySaver fallback."""
    from adip.config.settings import settings

    # Try PostgreSQL checkpointer
    if not settings.mock_mode:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            checkpointer = PostgresSaver.from_conn_string(settings.database_url)
            logger.info("Using PostgreSQL checkpointer")
            return checkpointer
        except Exception as exc:
            logger.warning("PostgreSQL checkpointer unavailable (%s)", exc)

    # InMemorySaver fallback (always works, no context-manager issues)
    try:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        logger.info("Using InMemorySaver checkpointer")
        return checkpointer
    except Exception as exc:
        logger.warning("MemorySaver failed (%s); running without checkpointer", exc)
        return None
