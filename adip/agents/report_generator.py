"""
Report Generator Agent — LangGraph node.

Rule engine FIRST (PROCEED/CONDITIONAL/HOLD) → then LLM for summary.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from adip.agents.supervisor import AgentTimer, append_error
from adip.graph.state import (
    ADIPState, ClusterResult, FileRiskScore,
    ReleaseRecommendation, RiskReport, TriggerType,
)
from adip.llm.factory import get_llm
from adip.llm.prompts import (
    REPORT_SUMMARY_SYSTEM, REPORT_SUMMARY_USER, ReportSummaryOutput,
)
from adip.scoring.rule_engine import evaluate

logger = logging.getLogger(__name__)


async def generate_report(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: determine release recommendation via rule engine,
    then generate executive summary via LLM.
    """
    with AgentTimer("report_generator"):
        try:
            risk_score_dicts = state.get("risk_scores", [])
            cluster_dicts = state.get("clusters", [])
            trigger = state.get("trigger_type", TriggerType.STREAM_EVENT)

            # Deserialize into Pydantic models for rule engine
            risk_scores = [FileRiskScore(**s) for s in risk_score_dicts]
            clusters = [ClusterResult(**c) for c in cluster_dicts]

            # ── RULE ENGINE FIRST — no LLM involved ─────────────────────
            recommendation = evaluate(risk_scores, clusters)

            # ── LLM call for executive summary ───────────────────────────
            high_risk = [s for s in risk_scores if s.risk_score >= 0.5]
            high_risk_summary = "\n".join(
                f"  - {s.file_path}: score={s.risk_score:.3f} tier={s.risk_tier}"
                for s in high_risk[:10]
            ) or "  (none)"

            cluster_summary_text = "\n".join(
                f"  - {c.label} ({c.root_cause_category}): "
                f"{c.member_count} members, recurrence={c.recurrence_count}"
                for c in clusters[:10]
            ) or "  (none)"

            user_msg = REPORT_SUMMARY_USER.format(
                recommendation=recommendation,
                high_risk_count=len(high_risk),
                high_risk_summary=high_risk_summary,
                cluster_count=len(clusters),
                cluster_summary=cluster_summary_text,
                trigger_type=trigger,
            )

            try:
                llm = get_llm()
                structured_llm = llm.with_structured_output(ReportSummaryOutput)
                llm_result = await structured_llm.ainvoke(
                    [
                        {"role": "system", "content": REPORT_SUMMARY_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ]
                )
                executive_summary = llm_result.executive_summary
                recommended_actions = llm_result.recommended_actions
                test_coverage_gaps = llm_result.test_coverage_gaps
            except Exception as llm_exc:
                logger.warning("LLM report generation failed (%s); using fallback", llm_exc)
                executive_summary = (
                    f"Release recommendation: {recommendation}. "
                    f"{len(high_risk)} high-risk files identified across {len(clusters)} defect clusters. "
                    f"Review recommended before proceeding."
                )
                recommended_actions = [
                    f"Review {s.file_path}" for s in high_risk[:5]
                ]
                test_coverage_gaps = [
                    s.file_path for s in high_risk if s.coverage_gap_score > 0.5
                ]

            # Build report
            report = RiskReport(
                trigger_type=TriggerType(trigger),
                release_recommendation=ReleaseRecommendation(recommendation),
                executive_summary=executive_summary,
                high_risk_files=risk_scores,
                recommended_actions=recommended_actions,
                test_coverage_gaps=test_coverage_gaps,
                cluster_summary=[c.model_dump() for c in clusters],
            )

            report_dict = report.model_dump()

            # Persist report
            try:
                from adip.agents.rag_indexer import get_db
                db = await get_db()
                await db.store_report(report_dict)
            except Exception as db_exc:
                logger.warning("Failed to persist report: %s", db_exc)

            logger.info("Report generated: recommendation=%s", recommendation)

            return {
                "risk_report": report_dict,
                "release_recommendation": recommendation.value if hasattr(recommendation, 'value') else str(recommendation),
                "human_review_required": recommendation == ReleaseRecommendation.HOLD,
            }

        except Exception as exc:
            logger.error("Report generation failed: %s", exc, exc_info=True)
            return append_error(state, f"report_generator: {exc}")
