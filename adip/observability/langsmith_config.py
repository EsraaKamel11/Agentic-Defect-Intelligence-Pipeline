"""
LangSmith tracing configuration.
"""
from __future__ import annotations

import logging
import os

from adip.config.settings import settings

logger = logging.getLogger(__name__)


def configure_langsmith():
    """Set up LangSmith tracing via environment variables."""
    if not settings.langsmith_api_key:
        logger.info("LangSmith API key not set; tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)
