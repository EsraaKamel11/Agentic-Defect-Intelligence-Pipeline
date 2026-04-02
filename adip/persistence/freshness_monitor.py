"""
Three-tier data freshness monitoring.
Alerts via Slack if any source exceeds staleness threshold.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from adip.config.settings import settings
from adip.graph.state import DataFreshnessStatus

logger = logging.getLogger(__name__)


class FreshnessMonitor:
    """Monitor staleness of Kafka, Jira, and vector store data sources."""

    def __init__(self):
        self._last_kafka_event: Optional[datetime] = None
        self._last_jira_poll: Optional[datetime] = None
        self._last_vector_upsert: Optional[datetime] = None

    def record_kafka_event(self):
        self._last_kafka_event = datetime.utcnow()

    def record_jira_poll(self):
        self._last_jira_poll = datetime.utcnow()

    def record_vector_upsert(self):
        self._last_vector_upsert = datetime.utcnow()

    def check(self) -> DataFreshnessStatus:
        """Evaluate freshness across all three tiers."""
        now = datetime.utcnow()
        stale_sources = []

        kafka_age = (
            (now - self._last_kafka_event).total_seconds()
            if self._last_kafka_event else float(settings.kafka_max_age_seconds + 1)
        )
        jira_age = (
            (now - self._last_jira_poll).total_seconds()
            if self._last_jira_poll else float(settings.jira_max_age_seconds + 1)
        )
        vector_lag = (
            (now - self._last_vector_upsert).total_seconds()
            if self._last_vector_upsert else float(settings.vector_store_max_lag_seconds + 1)
        )

        if kafka_age > settings.kafka_max_age_seconds:
            stale_sources.append("kafka")
        if jira_age > settings.jira_max_age_seconds:
            stale_sources.append("jira")
        if vector_lag > settings.vector_store_max_lag_seconds:
            stale_sources.append("vector_store")

        status = DataFreshnessStatus(
            kafka_stream_age_seconds=kafka_age,
            jira_poll_age_seconds=jira_age,
            vector_store_upsert_lag_seconds=vector_lag,
            all_fresh=len(stale_sources) == 0,
            stale_sources=stale_sources,
        )

        if stale_sources:
            logger.warning("Stale data sources detected: %s", stale_sources)
            self._alert_slack(status)

        return status

    def _alert_slack(self, status: DataFreshnessStatus):
        """Post staleness alert to Slack (mock if no webhook)."""
        if not settings.slack_webhook_url:
            logger.info(
                "[MOCK SLACK] Freshness alert: stale_sources=%s, kafka_age=%.0fs, jira_age=%.0fs, vector_lag=%.0fs",
                status.stale_sources,
                status.kafka_stream_age_seconds,
                status.jira_poll_age_seconds,
                status.vector_store_upsert_lag_seconds,
            )
            return

        try:
            import requests
            payload = {
                "text": (
                    f":warning: *ADIP Data Freshness Alert*\n"
                    f"Stale sources: {', '.join(status.stale_sources)}\n"
                    f"Kafka age: {status.kafka_stream_age_seconds:.0f}s\n"
                    f"Jira age: {status.jira_poll_age_seconds:.0f}s\n"
                    f"Vector lag: {status.vector_store_upsert_lag_seconds:.0f}s"
                )
            }
            requests.post(settings.slack_webhook_url, json=payload, timeout=5)
        except Exception as exc:
            logger.error("Slack alert failed: %s", exc)
