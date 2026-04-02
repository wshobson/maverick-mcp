"""LLM factory for creating language model instances.

This module provides a factory function to create LLM instances with intelligent
model selection. Configuration is read from ``LLMSettings`` (wired into
``Settings.llm``) rather than raw ``os.getenv`` calls.
"""

import logging
from typing import Any

from langchain_community.llms import FakeListLLM

from maverick_mcp.config.settings import LLMSettings, get_settings
from maverick_mcp.providers.openrouter_provider import (
    TaskType,
    get_openrouter_llm,
)

logger = logging.getLogger(__name__)


def _get_llm_settings() -> LLMSettings:
    """Return the current LLMSettings from the global Settings singleton."""
    return get_settings().llm


def get_llm(
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,  # Default to cost-effective
    prefer_quality: bool = False,
    model_override: str | None = None,
) -> Any:
    """Create and return an LLM instance with intelligent model selection.

    Args:
        task_type: Type of task to optimize model selection for
        prefer_fast: Prioritize speed over quality
        prefer_cheap: Prioritize cost over quality (default True)
        prefer_quality: Use premium models regardless of cost
        model_override: Override automatic model selection

    Returns:
        An LLM instance optimized for the task.

    Priority order:
    1. OpenRouter API if OPENROUTER_API_KEY is available (with smart model selection)
    2. OpenAI ChatOpenAI if OPENAI_API_KEY is available (fallback)
    3. Anthropic ChatAnthropic if ANTHROPIC_API_KEY is available (fallback)
    4. FakeListLLM as fallback for testing
    """
    settings = _get_llm_settings()
    provider = settings.provider
    temperature = settings.temperature

    # --- Explicit provider selection -------------------------------------------
    if provider == "openrouter":
        api_key = settings.get_openrouter_api_key()
        if api_key:
            logger.info("Using OpenRouter (explicit) for task: %s", task_type)
            return get_openrouter_llm(
                api_key=api_key,
                task_type=task_type,
                prefer_fast=prefer_fast,
                prefer_cheap=prefer_cheap,
                prefer_quality=prefer_quality,
                model_override=model_override,
                base_url=settings.openrouter_base_url,
            )
        logger.warning("LLM_PROVIDER=openrouter but no OPENROUTER_API_KEY set")

    elif provider == "openai":
        api_key = settings.get_openai_api_key()
        if api_key:
            logger.info("Using OpenAI (explicit)")
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": settings.openai_default_model,
                "temperature": temperature,
                "streaming": False,
            }
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            return ChatOpenAI(openai_api_key=api_key, **kwargs)
        logger.warning("LLM_PROVIDER=openai but no OPENAI_API_KEY set")

    elif provider == "anthropic":
        api_key = settings.get_anthropic_api_key()
        if api_key:
            logger.info("Using Anthropic (explicit)")
            from langchain_anthropic import ChatAnthropic

            kwargs: dict[str, Any] = {
                "model": settings.anthropic_default_model,
                "temperature": temperature,
            }
            if settings.anthropic_base_url:
                kwargs["anthropic_api_url"] = settings.anthropic_base_url
            return ChatAnthropic(anthropic_api_key=api_key, **kwargs)
        logger.warning("LLM_PROVIDER=anthropic but no ANTHROPIC_API_KEY set")

    # --- Auto-detection chain (provider == "auto") ----------------------------
    openrouter_api_key = settings.get_openrouter_api_key()
    if openrouter_api_key:
        logger.info(
            "Using OpenRouter with intelligent model selection for task: %s",
            task_type,
        )
        return get_openrouter_llm(
            api_key=openrouter_api_key,
            task_type=task_type,
            prefer_fast=prefer_fast,
            prefer_cheap=prefer_cheap,
            prefer_quality=prefer_quality,
            model_override=model_override,
            base_url=settings.openrouter_base_url,
        )

    openai_api_key = settings.get_openai_api_key()
    if openai_api_key:
        logger.info("Falling back to OpenAI API")
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": settings.openai_default_model,
            "temperature": temperature,
            "streaming": False,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(openai_api_key=openai_api_key, **kwargs)

    anthropic_api_key = settings.get_anthropic_api_key()
    if anthropic_api_key:
        logger.info("Falling back to Anthropic API")
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {
            "model": settings.anthropic_default_model,
            "temperature": temperature,
        }
        if settings.anthropic_base_url:
            kwargs["anthropic_api_url"] = settings.anthropic_base_url
        return ChatAnthropic(anthropic_api_key=anthropic_api_key, **kwargs)

    # Final fallback to fake LLM for testing
    logger.warning("No LLM API keys found - using FakeListLLM for testing")
    return FakeListLLM(
        responses=[
            "Mock analysis response for testing purposes.",
            "This is a simulated LLM response.",
            "Market analysis: Moderate bullish sentiment detected.",
        ]
    )
