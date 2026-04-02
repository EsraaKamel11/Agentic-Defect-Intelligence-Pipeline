"""
ADIP Configuration — single source of truth for all settings.
Falls back to environment variables → .env file → defaults.
"""
from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="Vocareum proxy API key")
    openai_base_url: str = Field(
        default="https://openai.vocareum.com/v1",
        description="Vocareum OpenAI-compatible endpoint",
    )
    primary_model: str = Field(default="gpt-4o", description="Primary LLM model")

    # ── Kafka ────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topics: List[str] = ["cicd-logs", "sentry-errors", "datadog-alerts"]
    kafka_consumer_group: str = "adip-pipeline"
    kafka_max_age_seconds: int = 300

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "defect_events"

    # ── PostgreSQL / SQLite fallback ─────────────────────────────────────
    database_url: str = "postgresql://localhost/adip"
    sqlite_fallback_path: str = "adip_fallback.db"

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_directives_channel: str = "adip:test_directives"

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: str = "all-mpnet-base-v2"
    embedding_batch_size: int = 64

    # ── Clustering ───────────────────────────────────────────────────────
    min_cluster_size: int = 5
    cluster_decay_days: int = 60
    umap_n_components: int = 50
    recluster_threshold: int = 5

    # ── Risk scoring ─────────────────────────────────────────────────────
    hold_threshold: float = 0.8
    conditional_threshold: float = 0.5
    recurrence_hold_count: int = 3
    weight_defect_frequency: float = 0.35
    weight_git_churn: float = 0.25
    weight_coverage_gap: float = 0.20
    weight_cluster_severity: float = 0.15
    weight_recency_decay: float = 0.05

    # ── Freshness monitoring ─────────────────────────────────────────────
    jira_max_age_seconds: int = 900
    vector_store_max_lag_seconds: int = 120

    # ── Jira (mock-friendly) ─────────────────────────────────────────────
    jira_base_url: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "DEFECT"
    jira_poll_interval_seconds: int = 900  # 15 minutes

    # ── Slack (mock-friendly) ─────────────────────────────────────────────
    slack_webhook_url: str = ""

    # ── Scheduler ────────────────────────────────────────────────────────
    batch_interval_hours: int = 6

    # ── Observability ────────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "adip-pipeline"
    prometheus_port: int = 9090

    # ── Runtime flags ────────────────────────────────────────────────────
    mock_mode: bool = False  # --mock flag sets this to True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "ADIP_",
        "case_sensitive": False,
    }


# Singleton — import this everywhere
settings = Settings()
