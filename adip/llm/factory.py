"""
LLM factory — creates LangChain ChatOpenAI instances pointed at Vocareum proxy.
All agents share a single LLM configuration.
"""
from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from adip.config.settings import settings

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance configured for the Vocareum proxy.
    Caches the default instance for reuse.
    """
    global _llm_instance

    model = model or settings.primary_model
    is_default = (model == settings.primary_model and temperature == 0.1 and max_tokens == 2048)

    if is_default and _llm_instance is not None:
        return _llm_instance

    llm = ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.info("Created LLM: model=%s base_url=%s", model, settings.openai_base_url)

    if is_default:
        _llm_instance = llm
    return llm
