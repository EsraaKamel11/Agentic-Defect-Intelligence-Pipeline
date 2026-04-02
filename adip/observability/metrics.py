"""
Prometheus metrics for ADIP pipeline observability.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    # ── Counters ─────────────────────────────────────────────────────────
    EVENTS_INGESTED = Counter(
        "adip_events_ingested_total",
        "Total defect events ingested",
        ["source"],
    )
    EVENTS_INDEXED = Counter(
        "adip_events_indexed_total",
        "Total events indexed to vector store",
    )
    CLUSTERS_PRODUCED = Counter(
        "adip_clusters_produced_total",
        "Total clusters produced",
    )
    REPORTS_GENERATED = Counter(
        "adip_reports_generated_total",
        "Total risk reports generated",
        ["recommendation"],
    )
    DIRECTIVES_PUBLISHED = Counter(
        "adip_directives_published_total",
        "Total test directives published",
    )
    ALERTS_DISPATCHED = Counter(
        "adip_alerts_dispatched_total",
        "Total alerts dispatched",
        ["channel"],
    )

    # ── Gauges ───────────────────────────────────────────────────────────
    VECTOR_STORE_SIZE = Gauge(
        "adip_vector_store_size",
        "Number of vectors in the store",
    )
    ACTIVE_CLUSTERS = Gauge(
        "adip_active_clusters",
        "Number of active clusters",
    )
    KAFKA_STREAM_AGE = Gauge(
        "adip_kafka_stream_age_seconds",
        "Age of most recent Kafka event",
    )

    # ── Histograms ───────────────────────────────────────────────────────
    PIPELINE_DURATION = Histogram(
        "adip_pipeline_duration_seconds",
        "End-to-end pipeline execution time",
        ["trigger_type"],
        buckets=[1, 5, 10, 30, 60, 120, 180, 300],
    )
    AGENT_DURATION = Histogram(
        "adip_agent_duration_seconds",
        "Per-agent execution time",
        ["agent"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    logger.warning("prometheus_client not installed; metrics disabled")
    _PROMETHEUS_AVAILABLE = False

    # Stubs
    class _Stub:
        def labels(self, *a, **kw): return self
        def inc(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass

    EVENTS_INGESTED = _Stub()
    EVENTS_INDEXED = _Stub()
    CLUSTERS_PRODUCED = _Stub()
    REPORTS_GENERATED = _Stub()
    DIRECTIVES_PUBLISHED = _Stub()
    ALERTS_DISPATCHED = _Stub()
    VECTOR_STORE_SIZE = _Stub()
    ACTIVE_CLUSTERS = _Stub()
    KAFKA_STREAM_AGE = _Stub()
    PIPELINE_DURATION = _Stub()
    AGENT_DURATION = _Stub()


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics HTTP endpoint."""
    if _PROMETHEUS_AVAILABLE:
        try:
            start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)
        except Exception as exc:
            logger.warning("Failed to start metrics server: %s", exc)
    else:
        logger.info("Metrics server not started (prometheus_client not installed)")
