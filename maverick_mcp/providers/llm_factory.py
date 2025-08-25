"""LLM factory for creating language model instances.

This module provides a factory function to create LLM instances based on available API keys.
"""

import os
from typing import Any

from langchain_community.llms import FakeListLLM


def get_llm() -> Any:
    """Create and return an LLM instance based on available API keys.

    Returns:
        An LLM instance - either OpenAI, Anthropic, or a fake LLM for testing.

    Priority order:
    1. OpenAI ChatOpenAI if OPENAI_API_KEY is available
    2. Anthropic ChatAnthropic if ANTHROPIC_API_KEY is available
    3. FakeListLLM as fallback for testing
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
        except ImportError:
            pass

    if anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
        except ImportError:
            pass

    # Fallback to fake LLM for testing
    return FakeListLLM(
        responses=[
            "Mock analysis response for testing purposes.",
            "This is a simulated LLM response.",
            "Market analysis: Moderate bullish sentiment detected.",
        ]
    )
