"""Centralised tool usage estimation configuration."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EstimationBasis(str, Enum):
    """Describes how a tool estimate was derived."""

    EMPIRICAL = "empirical"
    CONSERVATIVE = "conservative"
    HEURISTIC = "heuristic"
    SIMULATED = "simulated"


class ToolComplexity(str, Enum):
    """Qualitative complexity buckets used for monitoring and reporting."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    PREMIUM = "premium"


class ToolEstimate(BaseModel):
    """Static estimate describing expected LLM usage for a tool."""

    model_config = ConfigDict(frozen=True)

    llm_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    based_on: EstimationBasis
    complexity: ToolComplexity
    notes: str | None = None

    @field_validator("llm_calls", "total_tokens")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Estimates must be non-negative")
        return value


class MonitoringThresholds(BaseModel):
    """Thresholds for triggering alerting logic."""

    llm_calls_warning: int = 15
    llm_calls_critical: int = 25
    tokens_warning: int = 20_000
    tokens_critical: int = 35_000
    variance_warning: float = 0.5
    variance_critical: float = 1.0

    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "llm_calls_warning",
        "llm_calls_critical",
        "tokens_warning",
        "tokens_critical",
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Monitoring thresholds must be positive")
        return value


class ToolEstimationConfig(BaseModel):
    """Container for all tool estimates used across the service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_confidence: float = 0.75
    monitoring: MonitoringThresholds = Field(default_factory=MonitoringThresholds)
    simple_default: ToolEstimate = Field(
        default_factory=lambda: ToolEstimate(
            llm_calls=1,
            total_tokens=600,
            confidence=0.85,
            based_on=EstimationBasis.EMPIRICAL,
            complexity=ToolComplexity.SIMPLE,
            notes="Baseline simple operation",
        )
    )
    standard_default: ToolEstimate = Field(
        default_factory=lambda: ToolEstimate(
            llm_calls=3,
            total_tokens=4000,
            confidence=0.75,
            based_on=EstimationBasis.HEURISTIC,
            complexity=ToolComplexity.STANDARD,
            notes="Baseline standard analysis",
        )
    )
    complex_default: ToolEstimate = Field(
        default_factory=lambda: ToolEstimate(
            llm_calls=6,
            total_tokens=9000,
            confidence=0.7,
            based_on=EstimationBasis.SIMULATED,
            complexity=ToolComplexity.COMPLEX,
            notes="Baseline complex workflow",
        )
    )
    premium_default: ToolEstimate = Field(
        default_factory=lambda: ToolEstimate(
            llm_calls=10,
            total_tokens=15000,
            confidence=0.65,
            based_on=EstimationBasis.CONSERVATIVE,
            complexity=ToolComplexity.PREMIUM,
            notes="Baseline premium orchestration",
        )
    )
    unknown_tool_estimate: ToolEstimate = Field(
        default_factory=lambda: ToolEstimate(
            llm_calls=3,
            total_tokens=5000,
            confidence=0.3,
            based_on=EstimationBasis.CONSERVATIVE,
            complexity=ToolComplexity.STANDARD,
            notes="Fallback estimate for unknown tools",
        )
    )
    tool_estimates: dict[str, ToolEstimate] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:  # noqa: D401
        if not self.tool_estimates:
            self.tool_estimates = _build_default_estimates(self)
        else:
            normalised: dict[str, ToolEstimate] = {}
            for key, estimate in self.tool_estimates.items():
                normalised[key.lower()] = estimate
            self.tool_estimates = normalised

    def get_estimate(self, tool_name: str) -> ToolEstimate:
        key = tool_name.lower()
        return self.tool_estimates.get(key, self.unknown_tool_estimate)

    def get_default_for_complexity(self, complexity: ToolComplexity) -> ToolEstimate:
        mapping = {
            ToolComplexity.SIMPLE: self.simple_default,
            ToolComplexity.STANDARD: self.standard_default,
            ToolComplexity.COMPLEX: self.complex_default,
            ToolComplexity.PREMIUM: self.premium_default,
        }
        return mapping[complexity]

    def get_tools_by_complexity(self, complexity: ToolComplexity) -> list[str]:
        return sorted(
            [name for name, estimate in self.tool_estimates.items() if estimate.complexity == complexity]
        )

    def get_summary_stats(self) -> dict[str, Any]:
        if not self.tool_estimates:
            return {}

        total_tools = len(self.tool_estimates)
        by_complexity: dict[str, int] = {c.value: 0 for c in ToolComplexity}
        basis_distribution: dict[str, int] = {b.value: 0 for b in EstimationBasis}
        llm_total = 0
        token_total = 0
        confidence_total = 0.0

        for estimate in self.tool_estimates.values():
            by_complexity[estimate.complexity.value] += 1
            basis_distribution[estimate.based_on.value] += 1
            llm_total += estimate.llm_calls
            token_total += estimate.total_tokens
            confidence_total += estimate.confidence

        return {
            "total_tools": total_tools,
            "by_complexity": by_complexity,
            "avg_llm_calls": llm_total / total_tools,
            "avg_tokens": token_total / total_tools,
            "avg_confidence": confidence_total / total_tools,
            "basis_distribution": basis_distribution,
        }

    def should_alert(self, tool_name: str, actual_llm_calls: int, actual_tokens: int) -> tuple[bool, str]:
        estimate = self.get_estimate(tool_name)
        thresholds = self.monitoring
        alerts: list[str] = []

        if actual_llm_calls >= thresholds.llm_calls_critical:
            alerts.append(
                f"Critical: LLM calls ({actual_llm_calls}) exceeded threshold ({thresholds.llm_calls_critical})"
            )
        elif actual_llm_calls >= thresholds.llm_calls_warning:
            alerts.append(
                f"Warning: LLM calls ({actual_llm_calls}) exceeded threshold ({thresholds.llm_calls_warning})"
            )

        if actual_tokens >= thresholds.tokens_critical:
            alerts.append(
                f"Critical: Token usage ({actual_tokens}) exceeded threshold ({thresholds.tokens_critical})"
            )
        elif actual_tokens >= thresholds.tokens_warning:
            alerts.append(
                f"Warning: Token usage ({actual_tokens}) exceeded threshold ({thresholds.tokens_warning})"
            )

        expected_llm = estimate.llm_calls
        expected_tokens = estimate.total_tokens

        llm_variance = float("inf") if expected_llm == 0 and actual_llm_calls > 0 else (
            (actual_llm_calls - expected_llm) / max(expected_llm, 1)
        )
        token_variance = float("inf") if expected_tokens == 0 and actual_tokens > 0 else (
            (actual_tokens - expected_tokens) / max(expected_tokens, 1)
        )

        if llm_variance == float("inf") or llm_variance > thresholds.variance_critical:
            alerts.append("Critical: LLM call variance exceeded acceptable range")
        elif llm_variance > thresholds.variance_warning:
            alerts.append("Warning: LLM call variance elevated")

        if token_variance == float("inf") or token_variance > thresholds.variance_critical:
            alerts.append("Critical: Token variance exceeded acceptable range")
        elif token_variance > thresholds.variance_warning:
            alerts.append("Warning: Token variance elevated")

        message = "; ".join(alerts)
        return (bool(alerts), message)


def _build_default_estimates(config: ToolEstimationConfig) -> dict[str, ToolEstimate]:
    data: dict[str, dict[str, Any]] = {
        "get_stock_price": {
            "llm_calls": 0,
            "total_tokens": 200,
            "confidence": 0.92,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Direct market data lookup",
        },
        "get_company_info": {
            "llm_calls": 1,
            "total_tokens": 600,
            "confidence": 0.88,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Cached profile summary",
        },
        "get_stock_info": {
            "llm_calls": 1,
            "total_tokens": 550,
            "confidence": 0.87,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Quote lookup",
        },
        "calculate_sma": {
            "llm_calls": 0,
            "total_tokens": 180,
            "confidence": 0.9,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Local technical calculation",
        },
        "get_market_hours": {
            "llm_calls": 0,
            "total_tokens": 120,
            "confidence": 0.95,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Static schedule lookup",
        },
        "get_chart_links": {
            "llm_calls": 1,
            "total_tokens": 500,
            "confidence": 0.85,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Generates chart URLs",
        },
        "list_available_agents": {
            "llm_calls": 1,
            "total_tokens": 800,
            "confidence": 0.82,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Lists registered AI agents",
        },
        "clear_cache": {
            "llm_calls": 0,
            "total_tokens": 100,
            "confidence": 0.9,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Invalidates cache entries",
        },
        "get_cached_price_data": {
            "llm_calls": 0,
            "total_tokens": 150,
            "confidence": 0.86,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Reads cached OHLC data",
        },
        "get_watchlist": {
            "llm_calls": 1,
            "total_tokens": 650,
            "confidence": 0.84,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Fetches saved watchlists",
        },
        "generate_dev_token": {
            "llm_calls": 1,
            "total_tokens": 700,
            "confidence": 0.82,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.SIMPLE,
            "notes": "Generates development API token",
        },
        "get_rsi_analysis": {
            "llm_calls": 2,
            "total_tokens": 3000,
            "confidence": 0.78,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.STANDARD,
            "notes": "RSI interpretation",
        },
        "get_macd_analysis": {
            "llm_calls": 3,
            "total_tokens": 3200,
            "confidence": 0.74,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.STANDARD,
            "notes": "MACD indicator narrative",
        },
        "get_support_resistance": {
            "llm_calls": 4,
            "total_tokens": 3400,
            "confidence": 0.72,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.STANDARD,
            "notes": "Support/resistance summary",
        },
        "fetch_stock_data": {
            "llm_calls": 1,
            "total_tokens": 2600,
            "confidence": 0.8,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.STANDARD,
            "notes": "Aggregates OHLC data",
        },
        "get_maverick_stocks": {
            "llm_calls": 4,
            "total_tokens": 4500,
            "confidence": 0.73,
            "based_on": EstimationBasis.SIMULATED,
            "complexity": ToolComplexity.STANDARD,
            "notes": "Retrieves screening candidates",
        },
        "get_news_sentiment": {
            "llm_calls": 3,
            "total_tokens": 4800,
            "confidence": 0.76,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.STANDARD,
            "notes": "Summarises latest news sentiment",
        },
        "get_economic_calendar": {
            "llm_calls": 2,
            "total_tokens": 2800,
            "confidence": 0.79,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.STANDARD,
            "notes": "Economic calendar summary",
        },
        "get_full_technical_analysis": {
            "llm_calls": 6,
            "total_tokens": 9200,
            "confidence": 0.72,
            "based_on": EstimationBasis.EMPIRICAL,
            "complexity": ToolComplexity.COMPLEX,
            "notes": "Comprehensive technical package",
        },
        "risk_adjusted_analysis": {
            "llm_calls": 5,
            "total_tokens": 8800,
            "confidence": 0.7,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.COMPLEX,
            "notes": "Risk-adjusted metrics",
        },
        "compare_tickers": {
            "llm_calls": 6,
            "total_tokens": 9400,
            "confidence": 0.71,
            "based_on": EstimationBasis.SIMULATED,
            "complexity": ToolComplexity.COMPLEX,
            "notes": "Ticker comparison",
        },
        "portfolio_correlation_analysis": {
            "llm_calls": 5,
            "total_tokens": 8700,
            "confidence": 0.72,
            "based_on": EstimationBasis.SIMULATED,
            "complexity": ToolComplexity.COMPLEX,
            "notes": "Portfolio correlation study",
        },
        "get_market_overview": {
            "llm_calls": 4,
            "total_tokens": 7800,
            "confidence": 0.74,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.COMPLEX,
            "notes": "Market breadth overview",
        },
        "get_all_screening_recommendations": {
            "llm_calls": 5,
            "total_tokens": 8200,
            "confidence": 0.7,
            "based_on": EstimationBasis.SIMULATED,
            "complexity": ToolComplexity.COMPLEX,
            "notes": "Bulk screening results",
        },
        "analyze_market_with_agent": {
            "llm_calls": 10,
            "total_tokens": 14000,
            "confidence": 0.65,
            "based_on": EstimationBasis.CONSERVATIVE,
            "complexity": ToolComplexity.PREMIUM,
            "notes": "Multi-agent orchestration",
        },
        "get_agent_streaming_analysis": {
            "llm_calls": 12,
            "total_tokens": 16000,
            "confidence": 0.6,
            "based_on": EstimationBasis.CONSERVATIVE,
            "complexity": ToolComplexity.PREMIUM,
            "notes": "Streaming agent analysis",
        },
        "compare_personas_analysis": {
            "llm_calls": 9,
            "total_tokens": 12000,
            "confidence": 0.62,
            "based_on": EstimationBasis.HEURISTIC,
            "complexity": ToolComplexity.PREMIUM,
            "notes": "Persona comparison",
        },
    }

    estimates = {
        name: ToolEstimate(**details)
        for name, details in data.items()
    }
    return estimates


_config: ToolEstimationConfig | None = None


def get_tool_estimation_config() -> ToolEstimationConfig:
    """Return the singleton tool estimation configuration."""

    global _config
    if _config is None:
        _config = ToolEstimationConfig()
    return _config


def get_tool_estimate(tool_name: str) -> ToolEstimate:
    """Convenience helper returning the estimate for ``tool_name``."""

    return get_tool_estimation_config().get_estimate(tool_name)


def should_alert_for_usage(tool_name: str, llm_calls: int, total_tokens: int) -> tuple[bool, str]:
    """Check whether actual usage deviates enough to raise an alert."""

    return get_tool_estimation_config().should_alert(tool_name, llm_calls, total_tokens)


class ToolCostEstimator:
    """Legacy cost estimator retained for backwards compatibility."""

    BASE_COSTS = {
        "search": {"simple": 1, "moderate": 3, "complex": 5, "very_complex": 8},
        "analysis": {"simple": 2, "moderate": 4, "complex": 7, "very_complex": 12},
        "data": {"simple": 1, "moderate": 2, "complex": 4, "very_complex": 6},
        "research": {"simple": 3, "moderate": 6, "complex": 10, "very_complex": 15},
    }

    MULTIPLIERS = {
        "batch_size": {"small": 1.0, "medium": 1.5, "large": 2.0},
        "time_sensitivity": {"normal": 1.0, "urgent": 1.3, "real_time": 1.5},
    }

    @classmethod
    def estimate_tool_cost(
        cls,
        tool_name: str,
        category: str,
        complexity: str = "moderate",
        additional_params: dict[str, Any] | None = None,
    ) -> int:
        additional_params = additional_params or {}
        base_cost = cls.BASE_COSTS.get(category, {}).get(complexity, 3)

        batch_size = additional_params.get("batch_size", 1)
        if batch_size <= 10:
            batch_multiplier = cls.MULTIPLIERS["batch_size"]["small"]
        elif batch_size <= 50:
            batch_multiplier = cls.MULTIPLIERS["batch_size"]["medium"]
        else:
            batch_multiplier = cls.MULTIPLIERS["batch_size"]["large"]

        time_sensitivity = additional_params.get("time_sensitivity", "normal")
        time_multiplier = cls.MULTIPLIERS["time_sensitivity"].get(time_sensitivity, 1.0)

        total_cost = base_cost * batch_multiplier * time_multiplier

        if "portfolio" in tool_name.lower():
            total_cost *= 1.2
        elif "screening" in tool_name.lower():
            total_cost *= 1.1
        elif "real_time" in tool_name.lower():
            total_cost *= 1.3

        return max(1, int(total_cost))


tool_cost_estimator = ToolCostEstimator()


def estimate_tool_cost(
    tool_name: str,
    category: str = "analysis",
    complexity: str = "moderate",
    **kwargs: Any,
) -> int:
    """Convenience wrapper around :class:`ToolCostEstimator`."""

    return tool_cost_estimator.estimate_tool_cost(tool_name, category, complexity, kwargs)
