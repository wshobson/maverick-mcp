"""OpenRouter LLM provider with intelligent model selection.

This module provides integration with OpenRouter API for accessing various LLMs
with automatic model selection based on task requirements.
"""

import logging
from enum import StrEnum
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskType(StrEnum):
    """Task types for model selection."""

    # Analysis tasks
    DEEP_RESEARCH = "deep_research"
    MARKET_ANALYSIS = "market_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"

    # Synthesis tasks
    RESULT_SYNTHESIS = "result_synthesis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"

    # Query processing
    QUERY_CLASSIFICATION = "query_classification"
    QUICK_ANSWER = "quick_answer"

    # Complex reasoning
    COMPLEX_REASONING = "complex_reasoning"
    MULTI_AGENT_ORCHESTRATION = "multi_agent_orchestration"

    # Default
    GENERAL = "general"


class ModelProfile(BaseModel):
    """Profile for an LLM model with capabilities and costs."""

    model_id: str = Field(description="OpenRouter model identifier")
    name: str = Field(description="Human-readable model name")
    provider: str = Field(description="Model provider (e.g., anthropic, openai)")
    context_length: int = Field(description="Maximum context length in tokens")
    cost_per_million_input: float = Field(
        description="Cost per million input tokens in USD"
    )
    cost_per_million_output: float = Field(
        description="Cost per million output tokens in USD"
    )
    speed_rating: int = Field(description="Speed rating 1-10 (10 being fastest)")
    quality_rating: int = Field(description="Quality rating 1-10 (10 being best)")
    best_for: list[TaskType] = Field(description="Task types this model excels at")
    temperature: float = Field(
        default=0.3, description="Default temperature for this model"
    )


# Model profiles for intelligent selection
MODEL_PROFILES = {
    # Premium models (use sparingly for critical tasks)
    "anthropic/claude-opus-4.1": ModelProfile(
        model_id="anthropic/claude-opus-4.1",
        name="Claude Opus 4.1",
        provider="anthropic",
        context_length=200000,
        cost_per_million_input=15.0,
        cost_per_million_output=75.0,
        speed_rating=7,
        quality_rating=10,
        best_for=[
            TaskType.COMPLEX_REASONING,  # Only for the most complex tasks
        ],
        temperature=0.3,
    ),
    # Cost-effective high-quality models (primary workhorses)
    "anthropic/claude-sonnet-4": ModelProfile(
        model_id="anthropic/claude-sonnet-4",
        name="Claude Sonnet 4",
        provider="anthropic",
        context_length=1000000,  # 1M token context capability!
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
        speed_rating=8,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.MULTI_AGENT_ORCHESTRATION,
            TaskType.RESULT_SYNTHESIS,
            TaskType.PORTFOLIO_OPTIMIZATION,
        ],
        temperature=0.3,
    ),
    "openai/gpt-5": ModelProfile(
        model_id="openai/gpt-5",
        name="GPT-5",
        provider="openai",
        context_length=400000,
        cost_per_million_input=1.25,
        cost_per_million_output=10.0,
        speed_rating=8,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
        ],
        temperature=0.3,
    ),
    # Excellent cost-performance ratio models
    "google/gemini-2.5-pro": ModelProfile(
        model_id="google/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="google",
        context_length=1000000,  # 1M token context!
        cost_per_million_input=2.0,
        cost_per_million_output=8.0,
        speed_rating=8,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
        ],
        temperature=0.3,
    ),
    "deepseek/deepseek-r1": ModelProfile(
        model_id="deepseek/deepseek-r1",
        name="DeepSeek R1",
        provider="deepseek",
        context_length=128000,
        cost_per_million_input=0.5,
        cost_per_million_output=1.0,
        speed_rating=8,
        quality_rating=9,
        best_for=[
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.RISK_ASSESSMENT,
        ],
        temperature=0.3,
    ),
    # Fast, cost-effective models for simpler tasks
    # Speed-optimized models for research timeouts
    "google/gemini-2.5-flash": ModelProfile(
        model_id="google/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        context_length=1000000,
        cost_per_million_input=0.075,  # Ultra low cost
        cost_per_million_output=0.30,
        speed_rating=10,  # 199 tokens/sec - FASTEST available
        quality_rating=8,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.QUICK_ANSWER,
            TaskType.SENTIMENT_ANALYSIS,
        ],
        temperature=0.2,
    ),
    "openai/gpt-4o-mini": ModelProfile(
        model_id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_length=128000,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
        speed_rating=9,  # 126 tokens/sec - Excellent speed/cost balance
        quality_rating=8,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.QUICK_ANSWER,
        ],
        temperature=0.2,
    ),
    "anthropic/claude-3.5-haiku": ModelProfile(
        model_id="anthropic/claude-3.5-haiku",
        name="Claude 3.5 Haiku",
        provider="anthropic",
        context_length=200000,
        cost_per_million_input=0.25,
        cost_per_million_output=1.25,
        speed_rating=7,  # 65.6 tokens/sec - Updated with actual speed rating
        quality_rating=8,
        best_for=[
            TaskType.QUERY_CLASSIFICATION,
            TaskType.QUICK_ANSWER,
            TaskType.SENTIMENT_ANALYSIS,
        ],
        temperature=0.2,
    ),
    "openai/gpt-5-nano": ModelProfile(
        model_id="openai/gpt-5-nano",
        name="GPT-5 Nano",
        provider="openai",
        context_length=400000,
        cost_per_million_input=0.05,
        cost_per_million_output=0.40,
        speed_rating=9,  # 180 tokens/sec - Very fast
        quality_rating=7,
        best_for=[
            TaskType.QUICK_ANSWER,
            TaskType.QUERY_CLASSIFICATION,
            TaskType.DEEP_RESEARCH,  # Added for emergency research
        ],
        temperature=0.2,
    ),
    # Specialized models
    "xai/grok-4": ModelProfile(
        model_id="xai/grok-4",
        name="Grok 4",
        provider="xai",
        context_length=128000,
        cost_per_million_input=3.0,
        cost_per_million_output=12.0,
        speed_rating=7,
        quality_rating=9,
        best_for=[
            TaskType.MARKET_ANALYSIS,
            TaskType.SENTIMENT_ANALYSIS,
            TaskType.PORTFOLIO_OPTIMIZATION,
        ],
        temperature=0.3,
    ),
}


class OpenRouterProvider:
    """Provider for OpenRouter API with intelligent model selection."""

    def __init__(self, api_key: str):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self._model_usage_stats: dict[str, dict[str, int]] = {}

    def get_llm(
        self,
        task_type: TaskType = TaskType.GENERAL,
        prefer_fast: bool = False,
        prefer_cheap: bool = True,  # Default to cost-effective
        prefer_quality: bool = False,  # Override for premium models
        model_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout_budget: float | None = None,  # Emergency mode for timeouts
    ) -> ChatOpenAI:
        """Get an LLM instance optimized for the task.

        Args:
            task_type: Type of task to optimize for
            prefer_fast: Prioritize speed over quality
            prefer_cheap: Prioritize cost over quality (default True)
            prefer_quality: Use premium models regardless of cost
            model_override: Override model selection
            temperature: Override default temperature
            max_tokens: Maximum tokens for response
            timeout_budget: Available time budget - triggers emergency mode if < 30s

        Returns:
            Configured ChatOpenAI instance
        """
        # Use override if provided
        if model_override:
            model_id = model_override
            model_profile = MODEL_PROFILES.get(
                model_id,
                ModelProfile(
                    model_id=model_id,
                    name=model_id,
                    provider="unknown",
                    context_length=128000,
                    cost_per_million_input=1.0,
                    cost_per_million_output=1.0,
                    speed_rating=5,
                    quality_rating=5,
                    best_for=[TaskType.GENERAL],
                    temperature=0.3,
                ),
            )
        # Emergency mode for tight timeout budgets
        elif timeout_budget is not None and timeout_budget < 30:
            model_profile = self._select_emergency_model(task_type, timeout_budget)
            model_id = model_profile.model_id
            logger.warning(
                f"EMERGENCY MODE: Selected ultra-fast model '{model_profile.name}' "
                f"for {timeout_budget}s timeout budget"
            )
        else:
            model_profile = self._select_model(
                task_type, prefer_fast, prefer_cheap, prefer_quality
            )
            model_id = model_profile.model_id

        # Use provided temperature or model default
        final_temperature = (
            temperature if temperature is not None else model_profile.temperature
        )

        # Log model selection
        logger.info(
            f"Selected model '{model_profile.name}' for task '{task_type}' "
            f"(speed={model_profile.speed_rating}/10, quality={model_profile.quality_rating}/10, "
            f"cost=${model_profile.cost_per_million_input}/{model_profile.cost_per_million_output} per 1M tokens)"
        )

        # Track usage
        self._track_usage(model_id, task_type)

        # Create LangChain ChatOpenAI instance
        return ChatOpenAI(
            model=model_id,
            temperature=final_temperature,
            max_tokens=max_tokens,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/wshobson/maverick-mcp",
                "X-Title": "Maverick MCP",
            },
            streaming=True,
        )

    def _select_model(
        self,
        task_type: TaskType,
        prefer_fast: bool = False,
        prefer_cheap: bool = True,
        prefer_quality: bool = False,
    ) -> ModelProfile:
        """Select the best model for the task with cost-efficiency in mind.

        Args:
            task_type: Type of task
            prefer_fast: Prioritize speed
            prefer_cheap: Prioritize cost (default True)
            prefer_quality: Use premium models regardless of cost

        Returns:
            Selected model profile
        """
        candidates = []

        # Find models suitable for this task
        for profile in MODEL_PROFILES.values():
            if task_type in profile.best_for or task_type == TaskType.GENERAL:
                candidates.append(profile)

        if not candidates:
            # Fallback to GPT-5 Nano for general tasks
            return MODEL_PROFILES["openai/gpt-5-nano"]

        # Score and rank candidates
        scored_candidates = []
        for profile in candidates:
            score = 0

            # Calculate average cost for this model
            avg_cost = (
                profile.cost_per_million_input + profile.cost_per_million_output
            ) / 2

            # Quality preference overrides cost considerations
            if prefer_quality:
                # Heavily weight quality for premium mode
                score += profile.quality_rating * 20
                # Task fitness is critical
                if task_type in profile.best_for:
                    score += 40
                # Minimal cost consideration
                score += max(0, 20 - avg_cost)
            else:
                # Cost-efficiency focused scoring (default)
                # Calculate cost-efficiency ratio
                cost_efficiency = profile.quality_rating / max(1, avg_cost)
                score += cost_efficiency * 30

                # Task fitness bonus
                if task_type in profile.best_for:
                    score += 25

                # Base quality (reduced weight)
                score += profile.quality_rating * 5

                # Speed preference
                if prefer_fast:
                    score += profile.speed_rating * 5
                else:
                    score += profile.speed_rating * 2

                # Cost preference adjustment
                if prefer_cheap:
                    # Strong cost preference
                    cost_score = max(0, 100 - avg_cost * 5)
                    score += cost_score
                else:
                    # Balanced cost consideration (default)
                    cost_score = max(0, 60 - avg_cost * 3)
                    score += cost_score

            scored_candidates.append((score, profile))

        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

    def _select_emergency_model(
        self, task_type: TaskType, timeout_budget: float
    ) -> ModelProfile:
        """Select the fastest model available for emergency timeout situations.

        Emergency mode prioritizes speed above all other considerations.
        Used when timeout_budget < 30 seconds.

        Args:
            task_type: Type of task
            timeout_budget: Available time in seconds (< 30s)

        Returns:
            Fastest available model profile
        """
        # Emergency model priority (by actual tokens per second)

        # For ultra-tight budgets (< 15s), use only the absolute fastest
        if timeout_budget < 15:
            return MODEL_PROFILES["google/gemini-2.5-flash"]

        # For tight budgets (< 25s), use fastest available models
        if timeout_budget < 25:
            if task_type in [TaskType.SENTIMENT_ANALYSIS, TaskType.QUICK_ANSWER]:
                return MODEL_PROFILES[
                    "google/gemini-2.5-flash"
                ]  # Fastest for all tasks
            return MODEL_PROFILES["openai/gpt-4o-mini"]  # Speed + quality balance

        # For moderate emergency (< 30s), use speed-optimized models for complex tasks
        if task_type in [
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
        ]:
            return MODEL_PROFILES[
                "openai/gpt-4o-mini"
            ]  # Best speed/quality for research

        # Default to fastest model
        return MODEL_PROFILES["google/gemini-2.5-flash"]

    def _track_usage(self, model_id: str, task_type: TaskType):
        """Track model usage for analytics.

        Args:
            model_id: Model identifier
            task_type: Task type
        """
        if model_id not in self._model_usage_stats:
            self._model_usage_stats[model_id] = {}

        task_key = task_type.value
        if task_key not in self._model_usage_stats[model_id]:
            self._model_usage_stats[model_id][task_key] = 0

        self._model_usage_stats[model_id][task_key] += 1

    def get_usage_stats(self) -> dict[str, dict[str, int]]:
        """Get model usage statistics.

        Returns:
            Dictionary of model usage by task type
        """
        return self._model_usage_stats.copy()

    def recommend_models_for_workload(
        self, workload: dict[TaskType, int]
    ) -> dict[str, Any]:
        """Recommend optimal model mix for a given workload.

        Args:
            workload: Dictionary of task types and their frequencies

        Returns:
            Recommendations including models and estimated costs
        """
        recommendations = {}
        total_cost = 0.0

        for task_type, frequency in workload.items():
            # Select best model for this task
            model = self._select_model(task_type)

            # Estimate tokens (rough approximation)
            avg_input_tokens = 2000
            avg_output_tokens = 1000

            # Calculate cost
            input_cost = (
                avg_input_tokens * frequency * model.cost_per_million_input
            ) / 1_000_000
            output_cost = (
                avg_output_tokens * frequency * model.cost_per_million_output
            ) / 1_000_000
            task_cost = input_cost + output_cost

            recommendations[task_type.value] = {
                "model": model.name,
                "model_id": model.model_id,
                "frequency": frequency,
                "estimated_cost": task_cost,
            }

            total_cost += task_cost

        return {
            "recommendations": recommendations,
            "total_estimated_cost": total_cost,
            "cost_per_request": total_cost / sum(workload.values()) if workload else 0,
        }


# Convenience function for backward compatibility
def get_openrouter_llm(
    api_key: str,
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,
    prefer_quality: bool = False,
    **kwargs,
) -> ChatOpenAI:
    """Get an OpenRouter LLM instance with cost-efficiency by default.

    Args:
        api_key: OpenRouter API key
        task_type: Task type for model selection
        prefer_fast: Prioritize speed
        prefer_cheap: Prioritize cost (default True)
        prefer_quality: Use premium models regardless of cost
        **kwargs: Additional arguments for get_llm

    Returns:
        Configured ChatOpenAI instance
    """
    provider = OpenRouterProvider(api_key)
    return provider.get_llm(
        task_type=task_type,
        prefer_fast=prefer_fast,
        prefer_cheap=prefer_cheap,
        prefer_quality=prefer_quality,
        **kwargs,
    )
