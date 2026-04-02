"""
Test Feedback Agent — LangGraph node.

- Checks existing test coverage for high-risk files
- RAG retrieval: similar past defects + their test directives
- Generates TestGenerationDirective per high-risk file
- Publishes to Redis channel + writes to PostgreSQL
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from adip.agents.supervisor import AgentTimer, append_error
from adip.agents.rag_indexer import get_vector_store, get_db
from adip.feedback.redis_publisher import RedisPublisher
from adip.graph.state import ADIPState, FileRiskScore, TestGenerationDirective
from adip.llm.factory import get_llm
from adip.llm.prompts import (
    TEST_DIRECTIVE_SYSTEM, TEST_DIRECTIVE_USER, TestDirectiveOutput,
)
from adip.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_publisher: RedisPublisher | None = None


def _get_publisher() -> RedisPublisher:
    global _publisher
    if _publisher is None:
        _publisher = RedisPublisher()
    return _publisher


async def test_feedback(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: generate test directives for high-risk files.
    """
    with AgentTimer("test_feedback_agent"):
        try:
            report = state.get("risk_report", {})
            risk_scores = state.get("risk_scores", [])

            # Filter to high-risk files
            high_risk = [
                FileRiskScore(**s) for s in risk_scores
                if s.get("risk_score", 0) >= 0.3
            ]

            if not high_risk:
                logger.info("No high-risk files; skipping test directive generation")
                return {"test_directives": []}

            store = get_vector_store()
            retriever = HybridRetriever(store)
            db = await get_db()
            publisher = _get_publisher()
            llm = get_llm()

            directives: List[Dict[str, Any]] = []

            for score in high_risk[:10]:  # Cap at 10 directives per run
                # RAG retrieval: find similar past defects
                similar = retriever.retrieve(
                    query=f"defect in {score.file_path} {score.component}",
                    top_k=3,
                    filters={"component": score.component} if score.component != "unknown" else None,
                    rerank=False,  # Speed over precision here
                )
                similar_text = "\n".join(
                    f"  - {r.text[:200]}" for r in similar
                ) or "  (no similar past defects found)"

                # Find cluster info
                clusters = state.get("clusters", [])
                cluster_label = "Unknown"
                root_cause = "UNKNOWN"
                cluster_id = None
                for c in clusters:
                    cluster_label = c.get("label", "Unknown")
                    root_cause = c.get("root_cause_category", "UNKNOWN")
                    cluster_id = c.get("cluster_id")
                    break  # Use first cluster as primary context

                # LLM-generated directive
                user_msg = TEST_DIRECTIVE_USER.format(
                    target_file=score.file_path,
                    risk_score=score.risk_score,
                    risk_tier=score.risk_tier,
                    component=score.component,
                    defect_frequency=score.defect_frequency_30d,
                    cluster_label=cluster_label,
                    root_cause=root_cause,
                    similar_defects=similar_text,
                )

                try:
                    structured_llm = llm.with_structured_output(TestDirectiveOutput)
                    llm_result = await structured_llm.ainvoke(
                        [
                            {"role": "system", "content": TEST_DIRECTIVE_SYSTEM},
                            {"role": "user", "content": user_msg},
                        ]
                    )
                    directive = TestGenerationDirective(
                        target_file=score.file_path,
                        test_type=llm_result.test_type,
                        scenarios=llm_result.scenarios,
                        priority=llm_result.priority,
                        cluster_id=cluster_id,
                    )
                except Exception as llm_exc:
                    logger.warning("LLM directive gen failed for %s: %s", score.file_path, llm_exc)
                    directive = TestGenerationDirective(
                        target_file=score.file_path,
                        test_type="regression",
                        scenarios=[
                            f"Verify {score.file_path} handles error cases",
                            f"Test {score.component} integration points",
                            f"Check edge cases for {root_cause} pattern",
                        ],
                        priority="high" if score.risk_score >= 0.5 else "medium",
                        cluster_id=cluster_id,
                    )

                directive_dict = directive.model_dump()
                directives.append(directive_dict)

            # Publish to Redis
            publisher.publish_batch(directives)

            # Write to PostgreSQL
            await db.store_test_directives(directives)

            logger.info("Generated %d test directives", len(directives))

            return {"test_directives": directives}

        except Exception as exc:
            logger.error("Test feedback failed: %s", exc, exc_info=True)
            return {
                **append_error(state, f"test_feedback_agent: {exc}"),
                "test_directives": [],
            }
