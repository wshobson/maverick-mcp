"""OpenRouter LLM provider with intelligent model selection.

This module provides integration with OpenRouter API for accessing various LLMs
with automatic model selection based on task requirements.

Model profiles are loaded from maverick_mcp/config/model_profiles.yaml at startup,
with hardcoded defaults as fallback.
"""

from __future__ import annotations

import logging
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from maverick_mcp.providers.cost_tracking import BudgetExceededError, CostAccumulator

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


def _get_default_model_profiles() -> dict[str, ModelProfile]:
    """Return hardcoded default model profiles as fallback.

    These defaults are used when the YAML config file is missing or
    cannot be parsed. They should stay in sync with model_profiles.yaml.
    """
    return {
        "anthropic/claude-opus-4.1": ModelProfile(
            model_id="anthropic/claude-opus-4.1",
            name="Claude Opus 4.1",
            provider="anthropic",
            context_length=200000,
            cost_per_million_input=15.0,
            cost_per_million_output=75.0,
            speed_rating=7,
            quality_rating=10,
            best_for=[TaskType.COMPLEX_REASONING],
            temperature=0.3,
        ),
        "anthropic/claude-sonnet-4": ModelProfile(
            model_id="anthropic/claude-sonnet-4",
            name="Claude Sonnet 4",
            provider="anthropic",
            context_length=1000000,
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
            best_for=[TaskType.DEEP_RESEARCH, TaskType.MARKET_ANALYSIS],
            temperature=0.3,
        ),
        "google/gemini-2.5-pro": ModelProfile(
            model_id="google/gemini-2.5-pro",
            name="Gemini 2.5 Pro",
            provider="google",
            context_length=1000000,
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
        "google/gemini-2.5-flash": ModelProfile(
            model_id="google/gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider="google",
            context_length=1000000,
            cost_per_million_input=0.075,
            cost_per_million_output=0.30,
            speed_rating=10,
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
            speed_rating=9,
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
            speed_rating=7,
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
            speed_rating=9,
            quality_rating=7,
            best_for=[
                TaskType.QUICK_ANSWER,
                TaskType.QUERY_CLASSIFICATION,
                TaskType.DEEP_RESEARCH,
            ],
            temperature=0.2,
        ),
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


def _load_model_profiles_from_yaml(
    yaml_path: Path | None = None,
) -> dict[str, ModelProfile]:
    """Load model profiles from a YAML config file.

    Searches for model_profiles.yaml in the config directory. Falls back
    to hardcoded defaults if the file is missing or unparseable.

    Args:
        yaml_path: Optional explicit path to the YAML file.

    Returns:
        Dictionary mapping model_id to ModelProfile.
    """
    if yaml_path is None:
        # Look relative to this file: ../config/model_profiles.yaml
        yaml_path = Path(__file__).parent.parent / "config" / "model_profiles.yaml"

    if not yaml_path.exists():
        logger.warning(
            f"Model profiles YAML not found at {yaml_path}, using hardcoded defaults"
        )
        return _get_default_model_profiles()

    try:
        with open(yaml_path) as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict):
            logger.error(
                "Model profiles YAML has unexpected format, using hardcoded defaults"
            )
            return _get_default_model_profiles()

        profiles: dict[str, ModelProfile] = {}
        for model_id, profile_data in raw_data.items():
            try:
                # Convert best_for strings to TaskType enums
                best_for_raw = profile_data.get("best_for", [])
                best_for_enums = [TaskType(t) for t in best_for_raw]

                profiles[model_id] = ModelProfile(
                    model_id=model_id,
                    name=profile_data["name"],
                    provider=profile_data["provider"],
                    context_length=profile_data["context_length"],
                    cost_per_million_input=profile_data["cost_per_million_input"],
                    cost_per_million_output=profile_data["cost_per_million_output"],
                    speed_rating=profile_data["speed_rating"],
                    quality_rating=profile_data["quality_rating"],
                    best_for=best_for_enums,
                    temperature=profile_data.get("temperature", 0.3),
                )
            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Skipping invalid model profile '{model_id}' in YAML: {e}"
                )
                continue

        if not profiles:
            logger.error(
                "No valid model profiles loaded from YAML, using hardcoded defaults"
            )
            return _get_default_model_profiles()

        logger.info(f"Loaded {len(profiles)} model profiles from {yaml_path}")
        return profiles

    except Exception as e:
        logger.error(
            f"Failed to load model profiles from {yaml_path}: {e}, "
            "using hardcoded defaults"
        )
        return _get_default_model_profiles()


# Load model profiles from YAML config (falls back to hardcoded defaults)
MODEL_PROFILES = _load_model_profiles_from_yaml()


class CostTrackingLLM:
    """Wraps a ChatOpenAI instance to enforce cost budgets on every LLM call.

    Before each ``ainvoke``/``invoke``/``agenerate``/``generate`` call the wrapper
    estimates cost and checks the budget via :class:`CostAccumulator`.  After the
    call it records actual token usage from the response metadata.

    * ``BudgetExceededError`` is intentionally allowed to propagate so that
      callers can handle budget exhaustion.
    * All *other* cost-tracking errors are logged and swallowed so that cost
      tracking never crashes the main LLM call path.
    """

    # Default token estimates used for the pre-call budget check when we
    # cannot know the exact token count in advance.
    _DEFAULT_ESTIMATED_INPUT_TOKENS = 2000
    _DEFAULT_ESTIMATED_OUTPUT_TOKENS = 1000

    def __init__(
        self,
        llm: ChatOpenAI,
        model_id: str,
        provider: OpenRouterProvider,
        request_id: str | None = None,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._provider = provider
        # Store explicit request_id; generate per-call IDs when None
        self._request_id = request_id

    @property
    def __class__(self):
        """Report the wrapped LLM's class so isinstance() checks pass.

        LangGraph helpers like create_react_agent use
        ``isinstance(model, BaseChatModel)`` to select code paths.
        """
        return self._llm.__class__

    def _get_request_id(self) -> str:
        """Return the explicit request_id or generate a fresh one per call."""
        return self._request_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Async call wrappers
    # ------------------------------------------------------------------

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Async invoke with pre-call budget check and post-call cost recording."""
        rid = await self._pre_call_budget_check()
        response = await self._llm.ainvoke(*args, **kwargs)
        await self._post_call_record(response, request_id=rid)
        return response

    async def agenerate(self, *args: Any, **kwargs: Any) -> Any:
        """Async generate with pre-call budget check and post-call cost recording."""
        rid = await self._pre_call_budget_check()
        response = await self._llm.agenerate(*args, **kwargs)
        await self._post_call_record_generation(response, request_id=rid)
        return response

    # ------------------------------------------------------------------
    # Sync call wrappers
    # ------------------------------------------------------------------

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Sync invoke with post-call cost recording (budget check is async-only)."""
        response = self._llm.invoke(*args, **kwargs)
        self._post_call_record_sync(response)
        return response

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Sync generate with post-call cost recording."""
        response = self._llm.generate(*args, **kwargs)
        self._post_call_record_generation_sync(response)
        return response

    # ------------------------------------------------------------------
    # bind_tools returns a new wrapper so cost tracking persists
    # ------------------------------------------------------------------

    def bind_tools(self, *args: Any, **kwargs: Any) -> CostTrackingLLM:
        """Bind tools and return a new CostTrackingLLM wrapping the result."""
        bound = self._llm.bind_tools(*args, **kwargs)
        return CostTrackingLLM(
            llm=bound,
            model_id=self._model_id,
            provider=self._provider,
            request_id=self._request_id,  # Preserve explicit ID if set
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _pre_call_budget_check(self, request_id: str | None = None) -> str:
        """Estimate cost and check budget before an LLM call.

        Raises ``BudgetExceededError`` if the budget would be exceeded.
        All other errors are logged and swallowed.

        Returns the request_id used (generated if not provided).
        """
        rid = request_id or self._get_request_id()
        try:
            await self._provider.check_and_record_cost(
                model_id=self._model_id,
                estimated_input_tokens=self._DEFAULT_ESTIMATED_INPUT_TOKENS,
                estimated_output_tokens=self._DEFAULT_ESTIMATED_OUTPUT_TOKENS,
                request_id=rid,
            )
        except BudgetExceededError:
            raise
        except Exception:
            logger.debug(
                "Cost tracking pre-call check failed for model %s; proceeding with call",
                self._model_id,
                exc_info=True,
            )
        return rid

    async def _post_call_record(self, response: Any, request_id: str | None = None) -> None:
        """Record actual token usage from an AIMessage response."""
        rid = request_id or self._get_request_id()
        try:
            input_tokens, output_tokens = self._extract_token_usage(response)
            if input_tokens or output_tokens:
                await self._provider.record_actual_cost(
                    model_id=self._model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_id=rid,
                )
        except Exception:
            logger.debug(
                "Cost tracking post-call record failed for model %s",
                self._model_id,
                exc_info=True,
            )

    async def _post_call_record_generation(self, response: Any, request_id: str | None = None) -> None:
        """Record actual token usage from a generate() LLMResult response."""
        rid = request_id or self._get_request_id()
        try:
            input_tokens, output_tokens = self._extract_generation_token_usage(response)
            if input_tokens or output_tokens:
                await self._provider.record_actual_cost(
                    model_id=self._model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_id=rid,
                )
        except Exception:
            logger.debug(
                "Cost tracking post-call generation record failed for model %s",
                self._model_id,
                exc_info=True,
            )

    def _post_call_record_sync(self, response: Any) -> None:
        """Best-effort sync recording -- logs usage but cannot await."""
        try:
            input_tokens, output_tokens = self._extract_token_usage(response)
            if input_tokens or output_tokens:
                logger.info(
                    "Sync LLM call for %s: %d input / %d output tokens (async recording skipped)",
                    self._model_id,
                    input_tokens,
                    output_tokens,
                )
        except Exception:
            logger.debug(
                "Cost tracking sync record failed for model %s",
                self._model_id,
                exc_info=True,
            )

    def _post_call_record_generation_sync(self, response: Any) -> None:
        """Best-effort sync recording for generate() responses."""
        try:
            input_tokens, output_tokens = self._extract_generation_token_usage(response)
            if input_tokens or output_tokens:
                logger.info(
                    "Sync generate call for %s: %d input / %d output tokens (async recording skipped)",
                    self._model_id,
                    input_tokens,
                    output_tokens,
                )
        except Exception:
            logger.debug(
                "Cost tracking sync generation record failed for model %s",
                self._model_id,
                exc_info=True,
            )

    @staticmethod
    def _extract_token_usage(response: Any) -> tuple[int, int]:
        """Extract input/output token counts from a LangChain AIMessage.

        LangChain >= 0.3 AIMessage exposes ``usage_metadata`` with keys
        ``input_tokens`` and ``output_tokens``.  Falls back to
        ``response_metadata`` -> ``token_usage`` for older providers.
        """
        input_tokens = 0
        output_tokens = 0

        # Primary: usage_metadata (LangChain 0.3+)
        usage = getattr(response, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return input_tokens, output_tokens

        # Fallback: response_metadata.token_usage (OpenAI-style)
        resp_meta = getattr(response, "response_metadata", None)
        if resp_meta and isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage", {})
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

        return input_tokens, output_tokens

    @staticmethod
    def _extract_generation_token_usage(response: Any) -> tuple[int, int]:
        """Extract token counts from a LangChain LLMResult (from generate()).

        ``LLMResult.llm_output`` often contains ``token_usage`` with
        ``prompt_tokens`` and ``completion_tokens``.
        """
        input_tokens = 0
        output_tokens = 0

        llm_output = getattr(response, "llm_output", None)
        if llm_output and isinstance(llm_output, dict):
            token_usage = llm_output.get("token_usage", {})
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

        return input_tokens, output_tokens

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped LLM instance."""
        return getattr(self._llm, name)


class OpenRouterProvider:
    """Provider for OpenRouter API with intelligent model selection and cost tracking."""

    # Shared cost accumulator across all provider instances
    _shared_cost_accumulator: CostAccumulator | None = None

    def __init__(
        self,
        api_key: str,
        cost_accumulator: CostAccumulator | None = None,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            cost_accumulator: Optional CostAccumulator instance. If not provided,
                a shared singleton is created automatically.
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self._model_usage_stats: dict[str, dict[str, int]] = {}

        # Use provided accumulator, shared singleton, or create new one
        if cost_accumulator is not None:
            self.cost_accumulator = cost_accumulator
        else:
            if OpenRouterProvider._shared_cost_accumulator is None:
                OpenRouterProvider._shared_cost_accumulator = CostAccumulator()
            self.cost_accumulator = OpenRouterProvider._shared_cost_accumulator

    def get_model_profile(self, model_id: str) -> ModelProfile:
        """Get the model profile for a given model ID.

        Args:
            model_id: The model identifier.

        Returns:
            The ModelProfile, or a default profile for unknown models.
        """
        return MODEL_PROFILES.get(
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
        request_id: str | None = None,
        track_cost: bool = True,
    ) -> ChatOpenAI | CostTrackingLLM:
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
            request_id: Optional request ID for per-request budget tracking.
                If not provided a UUID is generated automatically.
            track_cost: Whether to wrap the LLM with cost tracking (default True).
                Set to False to get a raw ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance, optionally wrapped with cost tracking
        """
        # Use override if provided
        if model_override:
            model_id = model_override
            model_profile = self.get_model_profile(model_id)
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
        llm = ChatOpenAI(
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

        # Wrap with cost tracking when accumulator is available
        if track_cost and self.cost_accumulator is not None:
            return CostTrackingLLM(
                llm=llm,
                model_id=model_id,
                provider=self,
                request_id=request_id,
            )

        return llm

    async def check_and_record_cost(
        self,
        model_id: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        request_id: str | None = None,
    ) -> float:
        """Check budget and pre-authorize cost for an upcoming LLM call.

        Call this before invoking the LLM. It estimates the cost using the
        model's pricing profile and checks both per-request and daily budgets.

        Args:
            model_id: The model identifier to use.
            estimated_input_tokens: Estimated number of input tokens.
            estimated_output_tokens: Estimated number of output tokens.
            request_id: Optional request ID for per-request budget tracking.

        Returns:
            The estimated cost in USD.

        Raises:
            BudgetExceededError: If the call would exceed budget limits.
        """
        profile = self.get_model_profile(model_id)
        estimated_cost = self.cost_accumulator.estimate_cost(
            model_id=model_id,
            cost_per_million_input=profile.cost_per_million_input,
            cost_per_million_output=profile.cost_per_million_output,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )
        await self.cost_accumulator.check_budget(estimated_cost, request_id)
        return estimated_cost

    async def record_actual_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        request_id: str | None = None,
    ) -> float:
        """Record actual token usage and cost after an LLM call completes.

        Args:
            model_id: The model identifier used.
            input_tokens: Actual input tokens consumed.
            output_tokens: Actual output tokens consumed.
            request_id: Optional request ID for per-request tracking.

        Returns:
            The actual cost in USD.
        """
        profile = self.get_model_profile(model_id)
        record = await self.cost_accumulator.record_cost(
            model_id=model_id,
            cost_per_million_input=profile.cost_per_million_input,
            cost_per_million_output=profile.cost_per_million_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_id=request_id,
        )
        return record.actual_cost

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

    async def get_cost_summary(self) -> dict:
        """Get a summary of cost tracking state.

        Delegates to the underlying :class:`CostAccumulator`.

        Returns:
            Dictionary with daily total, request totals, limits, and record count.
        """
        return await self.cost_accumulator.get_summary()

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
    **kwargs: Any,
) -> ChatOpenAI | CostTrackingLLM:
    """Get an OpenRouter LLM instance with cost-efficiency by default.

    Args:
        api_key: OpenRouter API key
        task_type: Task type for model selection
        prefer_fast: Prioritize speed
        prefer_cheap: Prioritize cost (default True)
        prefer_quality: Use premium models regardless of cost
        **kwargs: Additional arguments for get_llm (including ``request_id``
            and ``track_cost``)

    Returns:
        Configured ChatOpenAI instance, optionally wrapped with cost tracking
    """
    provider = OpenRouterProvider(api_key)
    return provider.get_llm(
        task_type=task_type,
        prefer_fast=prefer_fast,
        prefer_cheap=prefer_cheap,
        prefer_quality=prefer_quality,
        **kwargs,
    )
