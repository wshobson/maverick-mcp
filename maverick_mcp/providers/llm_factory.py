"""LLM factory for creating language model instances.

This module provides a factory function to create LLM instances with intelligent model selection.
"""

import logging
import os
from typing import Any

from langchain_community.llms import FakeListLLM

from maverick_mcp.providers.openrouter_provider import (
    OpenRouterProvider,
    TaskType,
    get_openrouter_llm,
)

logger = logging.getLogger(__name__)


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
    # Check for OpenRouter first (preferred)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        logger.info(f"Using OpenRouter with intelligent model selection for task: {task_type}")
        return get_openrouter_llm(
            api_key=openrouter_api_key,
            task_type=task_type,
            prefer_fast=prefer_fast,
            prefer_cheap=prefer_cheap,
            prefer_quality=prefer_quality,
            model_override=model_override,
        )

    # Fallback to OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        logger.info("Falling back to OpenAI API")
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=False)
        except ImportError:
            pass

    # Fallback to Anthropic
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        logger.info("Falling back to Anthropic API")
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
        except ImportError:
            pass

    # Final fallback to fake LLM for testing
    logger.warning("No LLM API keys found - using FakeListLLM for testing")
    return FakeListLLM(
        responses=[
            "Mock analysis response for testing purposes.",
            "This is a simulated LLM response.",
            "Market analysis: Moderate bullish sentiment detected.",
        ]
    )
