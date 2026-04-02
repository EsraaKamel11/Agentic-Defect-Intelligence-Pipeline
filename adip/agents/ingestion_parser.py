"""
Ingestion & Parser Agent — LangGraph node.

- Consumes raw events from Kafka/Jira/mock sources
- Normalizes into DefectEvent objects
- Enriches with git blame, file path, component tag
- Structure-aware parsing for stack traces, Jira bodies, CI logs
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from adip.agents.supervisor import AgentTimer, append_error
from adip.graph.state import ADIPState
from adip.ingestion.kafka_consumer import KafkaEventConsumer
from adip.ingestion.jira_poller import JiraPoller
from adip.ingestion.normalizer import normalize_batch

logger = logging.getLogger(__name__)

# Module-level singletons (created once, reused)
_kafka_consumer: KafkaEventConsumer | None = None
_jira_poller: JiraPoller | None = None


def _get_kafka() -> KafkaEventConsumer:
    global _kafka_consumer
    if _kafka_consumer is None:
        _kafka_consumer = KafkaEventConsumer()
    return _kafka_consumer


def _get_jira() -> JiraPoller:
    global _jira_poller
    if _jira_poller is None:
        _jira_poller = JiraPoller()
    return _jira_poller


async def ingest_and_parse(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: ingest raw events and parse into DefectEvents.
    """
    with AgentTimer("ingestion_parser"):
        try:
            # Collect raw events from all sources
            raw_events = list(state.get("raw_events", []))

            # Kafka / mock stream
            kafka = _get_kafka()
            kafka_events = kafka.consume_batch(max_events=30)
            raw_events.extend(kafka_events)

            # Jira poll
            jira = _get_jira()
            jira_events = await jira.poll()
            raw_events.extend(jira_events)

            if not raw_events:
                logger.warning("No events ingested from any source")
                return {"defect_events": [], "raw_events": [], "indexed_count": 0}

            # Normalize all events into DefectEvent objects
            defect_events = normalize_batch(raw_events)

            logger.info(
                "Ingested %d raw events → %d normalized DefectEvents",
                len(raw_events), len(defect_events),
            )

            return {
                "raw_events": [e if isinstance(e, dict) else e for e in raw_events],
                "defect_events": [e.model_dump() for e in defect_events],
            }

        except Exception as exc:
            logger.error("Ingestion failed: %s", exc, exc_info=True)
            return append_error(state, f"ingestion_parser: {exc}")
