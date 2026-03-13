"""
Tool registry to register router tools directly on main server.
This avoids Claude Desktop's issue with mounted router tool names.

Includes per-category rate limiting, enforced timeouts, and error classification.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from fastmcp import FastMCP

from maverick_mcp.utils.decision_logger import decision_logger as _decision_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Category & Error Classification Enums
# ---------------------------------------------------------------------------


class ToolCategory(StrEnum):
    """Tool categories for access control and rate limiting."""

    DATA = "data"  # Stock data fetching
    TECHNICAL = "technical"  # Technical analysis
    SCREENING = "screening"  # Stock screening
    PORTFOLIO = "portfolio"  # Portfolio management
    RESEARCH = "research"  # AI-powered research (expensive)
    AGENT = "agent"  # Agent-based analysis (expensive)
    BACKTESTING = "backtesting"  # Backtesting (compute-intensive)
    SYSTEM = "system"  # Health, status tools


class ErrorCategory(StrEnum):
    """Error classification for tool failures."""

    TRANSIENT = "transient"  # Retry-able (network timeout, rate limit)
    PERMANENT = "permanent"  # Don't retry (bad input, not found)
    RATE_LIMITED = "rate_limited"  # Upstream rate limit hit
    UPSTREAM_FAILURE = "upstream"  # External service down
    INTERNAL = "internal"  # Bug in our code


class DecisionStatus(StrEnum):
    """Status values for decision log entries."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Per-tool Configuration
# ---------------------------------------------------------------------------


@dataclass
class ToolConfig:
    """Per-tool configuration for rate limiting, timeouts, and access control."""

    category: ToolCategory
    timeout_seconds: float = 30.0
    max_calls_per_minute: int = 10
    max_calls_per_hour: int = 100
    requires_api_key: list[str] = field(default_factory=list)  # Which API keys needed
    description: str = ""


# Tool configuration registry - defines limits for each tool category
TOOL_CATEGORY_CONFIGS: dict[ToolCategory, ToolConfig] = {
    ToolCategory.DATA: ToolConfig(
        category=ToolCategory.DATA,
        timeout_seconds=15.0,
        max_calls_per_minute=20,
        max_calls_per_hour=500,
        requires_api_key=["TIINGO_API_KEY"],
        description="Stock data fetching tools",
    ),
    ToolCategory.TECHNICAL: ToolConfig(
        category=ToolCategory.TECHNICAL,
        timeout_seconds=20.0,
        max_calls_per_minute=15,
        max_calls_per_hour=300,
        description="Technical analysis tools",
    ),
    ToolCategory.SCREENING: ToolConfig(
        category=ToolCategory.SCREENING,
        timeout_seconds=25.0,
        max_calls_per_minute=10,
        max_calls_per_hour=200,
        description="Stock screening tools",
    ),
    ToolCategory.PORTFOLIO: ToolConfig(
        category=ToolCategory.PORTFOLIO,
        timeout_seconds=15.0,
        max_calls_per_minute=20,
        max_calls_per_hour=500,
        description="Portfolio management tools",
    ),
    ToolCategory.RESEARCH: ToolConfig(
        category=ToolCategory.RESEARCH,
        timeout_seconds=300.0,
        max_calls_per_minute=2,
        max_calls_per_hour=20,
        requires_api_key=["OPENROUTER_API_KEY"],
        description="AI-powered research tools (expensive)",
    ),
    ToolCategory.AGENT: ToolConfig(
        category=ToolCategory.AGENT,
        timeout_seconds=120.0,
        max_calls_per_minute=3,
        max_calls_per_hour=30,
        requires_api_key=["OPENROUTER_API_KEY"],
        description="Agent-based analysis tools (expensive)",
    ),
    ToolCategory.BACKTESTING: ToolConfig(
        category=ToolCategory.BACKTESTING,
        timeout_seconds=180.0,
        max_calls_per_minute=3,
        max_calls_per_hour=20,
        description="Backtesting tools (compute-intensive)",
    ),
    ToolCategory.SYSTEM: ToolConfig(
        category=ToolCategory.SYSTEM,
        timeout_seconds=5.0,
        max_calls_per_minute=60,
        max_calls_per_hour=3600,
        description="System health and status tools",
    ),
}


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class ToolRateLimiter:
    """Per-category rate limiter using sliding window counters."""

    def __init__(self):
        self._minute_counts: dict[str, list[float]] = defaultdict(list)
        self._hour_counts: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check_rate_limit(self, category: ToolCategory) -> bool:
        """Check if a tool call is within rate limits. Returns True if allowed."""
        config = TOOL_CATEGORY_CONFIGS.get(category)
        if not config:
            return True

        now = time.time()
        key = category.value

        with self._lock:
            # Clean old entries and check minute limit
            self._minute_counts[key] = [
                t for t in self._minute_counts[key] if now - t < 60
            ]
            if len(self._minute_counts[key]) >= config.max_calls_per_minute:
                return False

            # Check hour limit
            self._hour_counts[key] = [
                t for t in self._hour_counts[key] if now - t < 3600
            ]
            if len(self._hour_counts[key]) >= config.max_calls_per_hour:
                return False

            # Record this call
            self._minute_counts[key].append(now)
            self._hour_counts[key].append(now)
            return True

    def get_status(self) -> dict[str, Any]:
        """Get current rate limit status for all categories."""
        now = time.time()
        status: dict[str, Any] = {}
        with self._lock:
            for cat in ToolCategory:
                key = cat.value
                config = TOOL_CATEGORY_CONFIGS.get(cat)
                if not config:
                    continue
                # Prune stale entries while computing counts
                if key in self._minute_counts:
                    self._minute_counts[key] = [t for t in self._minute_counts[key] if now - t < 60]
                if key in self._hour_counts:
                    self._hour_counts[key] = [t for t in self._hour_counts[key] if now - t < 3600]
                status[key] = {
                    "calls_this_minute": len(self._minute_counts.get(key, [])),
                    "limit_per_minute": config.max_calls_per_minute,
                    "calls_this_hour": len(self._hour_counts.get(key, [])),
                    "limit_per_hour": config.max_calls_per_hour,
                }
        return status


# Module-level singleton
_rate_limiter = ToolRateLimiter()


# ---------------------------------------------------------------------------
# Error Classification
# ---------------------------------------------------------------------------


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an exception into an error category.

    This function never raises -- always returns at least INTERNAL.
    """
    try:
        error_str = str(error).lower()

        # Rate limiting
        if any(kw in error_str for kw in ["rate limit", "429", "too many requests"]):
            return ErrorCategory.RATE_LIMITED

        # Transient errors (retry-able)
        if any(
            kw in error_str
            for kw in [
                "timeout",
                "connection",
                "temporary",
                "unavailable",
                "503",
                "502",
            ]
        ):
            return ErrorCategory.TRANSIENT
        if isinstance(
            error, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)
        ):
            return ErrorCategory.TRANSIENT

        # Permanent errors (don't retry)
        if any(
            kw in error_str
            for kw in [
                "not found",
                "invalid",
                "bad request",
                "400",
                "404",
                "validation",
            ]
        ):
            return ErrorCategory.PERMANENT
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorCategory.PERMANENT

        # Upstream failure
        if any(kw in error_str for kw in ["500", "internal server error", "service"]):
            return ErrorCategory.UPSTREAM_FAILURE

        return ErrorCategory.INTERNAL
    except Exception:
        return ErrorCategory.INTERNAL


# ---------------------------------------------------------------------------
# Tool Registration Functions
# ---------------------------------------------------------------------------


def register_technical_tools(mcp: FastMCP) -> None:
    """Register technical analysis tools directly on main server"""
    from maverick_mcp.api.routers.technical import (
        get_macd_analysis,
        get_rsi_analysis,
        get_support_resistance,
    )

    # Import enhanced versions with proper timeout handling and logging
    from maverick_mcp.api.routers.technical_enhanced import (
        get_full_technical_analysis_enhanced,
        get_stock_chart_analysis_enhanced,
    )
    from maverick_mcp.validation.technical import TechnicalAnalysisRequest

    # Register with prefixed names to maintain organization
    mcp.tool(name="technical_get_rsi_analysis")(get_rsi_analysis)
    mcp.tool(name="technical_get_macd_analysis")(get_macd_analysis)
    mcp.tool(name="technical_get_support_resistance")(get_support_resistance)

    # Use enhanced versions with timeout handling and comprehensive logging
    @mcp.tool(name="technical_get_full_technical_analysis")
    async def technical_get_full_technical_analysis(ticker: str, days: int = 365):
        """
        Get comprehensive technical analysis for a given ticker with enhanced logging and timeout handling.

        This enhanced version provides:
        - Step-by-step logging for debugging
        - 25-second timeout to prevent hangs
        - Comprehensive error handling
        - Guaranteed JSON-RPC responses

        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data to analyze (default: 365)

        Returns:
            Dictionary containing complete technical analysis or error information
        """
        if not _rate_limiter.check_rate_limit(ToolCategory.TECHNICAL):
            return {
                "error": "Rate limit exceeded for technical tools. Please try again shortly."
            }
        config = TOOL_CATEGORY_CONFIGS[ToolCategory.TECHNICAL]
        try:
            async with asyncio.timeout(config.timeout_seconds):
                request = TechnicalAnalysisRequest(ticker=ticker, days=days)
                return await get_full_technical_analysis_enhanced(request)
        except TimeoutError:
            return {
                "error": f"Technical analysis timed out after {config.timeout_seconds}s for {ticker}"
            }

    @mcp.tool(name="technical_get_stock_chart_analysis")
    async def technical_get_stock_chart_analysis(ticker: str):
        """
        Generate a comprehensive technical analysis chart with enhanced error handling.

        This enhanced version provides:
        - 15-second timeout for chart generation
        - Progressive chart sizing for Claude Desktop compatibility
        - Detailed logging for debugging
        - Graceful fallback on errors

        Args:
            ticker: The ticker symbol of the stock to analyze

        Returns:
            Dictionary containing chart data or error information
        """
        if not _rate_limiter.check_rate_limit(ToolCategory.TECHNICAL):
            return {
                "error": "Rate limit exceeded for technical tools. Please try again shortly."
            }
        config = TOOL_CATEGORY_CONFIGS[ToolCategory.TECHNICAL]
        try:
            async with asyncio.timeout(config.timeout_seconds):
                return await get_stock_chart_analysis_enhanced(ticker)
        except TimeoutError:
            return {
                "error": f"Chart analysis timed out after {config.timeout_seconds}s for {ticker}"
            }


def register_screening_tools(mcp: FastMCP) -> None:
    """Register screening tools directly on main server"""
    from maverick_mcp.api.routers.screening import (
        get_all_screening_recommendations,
        get_maverick_bear_stocks,
        get_maverick_stocks,
        get_screening_by_criteria,
        get_supply_demand_breakouts,
    )

    mcp.tool(name="screening_get_maverick_stocks")(get_maverick_stocks)
    mcp.tool(name="screening_get_maverick_bear_stocks")(get_maverick_bear_stocks)
    mcp.tool(name="screening_get_supply_demand_breakouts")(get_supply_demand_breakouts)
    mcp.tool(name="screening_get_all_screening_recommendations")(
        get_all_screening_recommendations
    )
    mcp.tool(name="screening_get_screening_by_criteria")(get_screening_by_criteria)


def register_portfolio_tools(mcp: FastMCP) -> None:
    """Register portfolio tools directly on main server"""
    from maverick_mcp.api.routers.portfolio import (
        add_portfolio_position,
        clear_my_portfolio,
        compare_tickers,
        get_my_portfolio,
        portfolio_correlation_analysis,
        remove_portfolio_position,
        risk_adjusted_analysis,
    )

    # Portfolio management tools
    mcp.tool(name="portfolio_add_position")(add_portfolio_position)
    mcp.tool(name="portfolio_get_my_portfolio")(get_my_portfolio)
    mcp.tool(name="portfolio_remove_position")(remove_portfolio_position)
    mcp.tool(name="portfolio_clear_portfolio")(clear_my_portfolio)

    # Portfolio analysis tools
    mcp.tool(name="portfolio_risk_adjusted_analysis")(risk_adjusted_analysis)
    mcp.tool(name="portfolio_compare_tickers")(compare_tickers)
    mcp.tool(name="portfolio_portfolio_correlation_analysis")(
        portfolio_correlation_analysis
    )


def register_data_tools(mcp: FastMCP) -> None:
    """Register data tools directly on main server"""
    from maverick_mcp.api.routers.data import (
        clear_cache,
        fetch_stock_data,
        fetch_stock_data_batch,
        get_cached_price_data,
        get_chart_links,
        get_stock_info,
    )

    # Import enhanced news sentiment that uses Tiingo or LLM
    from maverick_mcp.api.routers.news_sentiment_enhanced import (
        get_news_sentiment_enhanced,
    )

    mcp.tool(name="data_fetch_stock_data")(fetch_stock_data)
    mcp.tool(name="data_fetch_stock_data_batch")(fetch_stock_data_batch)
    mcp.tool(name="data_get_stock_info")(get_stock_info)

    # Use enhanced news sentiment that doesn't rely on EXTERNAL_DATA_API_KEY
    @mcp.tool(name="data_get_news_sentiment")
    async def get_news_sentiment(ticker: str, timeframe: str = "7d", limit: int = 10):
        """
        Get news sentiment analysis for a stock using Tiingo News API or LLM analysis.

        This enhanced tool provides reliable sentiment analysis by:
        - Using Tiingo's news API if available (requires paid plan)
        - Analyzing sentiment with LLM (Claude/GPT)
        - Falling back to research-based sentiment
        - Never failing due to missing EXTERNAL_DATA_API_KEY

        Args:
            ticker: Stock ticker symbol
            timeframe: Time frame for news (1d, 7d, 30d, etc.)
            limit: Maximum number of news articles to analyze

        Returns:
            Dictionary containing sentiment analysis with confidence scores
        """
        if not _rate_limiter.check_rate_limit(ToolCategory.DATA):
            return {
                "error": "Rate limit exceeded for data tools. Please try again shortly."
            }
        config = TOOL_CATEGORY_CONFIGS[ToolCategory.DATA]
        try:
            async with asyncio.timeout(config.timeout_seconds):
                return await get_news_sentiment_enhanced(ticker, timeframe, limit)
        except TimeoutError:
            return {
                "error": f"News sentiment timed out after {config.timeout_seconds}s for {ticker}"
            }

    mcp.tool(name="data_get_cached_price_data")(get_cached_price_data)
    mcp.tool(name="data_get_chart_links")(get_chart_links)
    mcp.tool(name="data_clear_cache")(clear_cache)


def register_performance_tools(mcp: FastMCP) -> None:
    """Register performance tools directly on main server"""
    from maverick_mcp.api.routers.performance import (
        analyze_database_index_usage,
        clear_system_caches,
        get_cache_performance_status,
        get_database_performance_status,
        get_redis_health_status,
        get_system_performance_health,
        optimize_cache_configuration,
    )

    mcp.tool(name="performance_get_system_performance_health")(
        get_system_performance_health
    )
    mcp.tool(name="performance_get_redis_health_status")(get_redis_health_status)
    mcp.tool(name="performance_get_cache_performance_status")(
        get_cache_performance_status
    )
    mcp.tool(name="performance_get_database_performance_status")(
        get_database_performance_status
    )
    mcp.tool(name="performance_analyze_database_index_usage")(
        analyze_database_index_usage
    )
    mcp.tool(name="performance_optimize_cache_configuration")(
        optimize_cache_configuration
    )
    mcp.tool(name="performance_clear_system_caches")(clear_system_caches)


def register_agent_tools(mcp: FastMCP) -> None:
    """Register agent tools directly on main server if available"""
    try:
        from maverick_mcp.api.routers.agents import (
            analyze_market_with_agent,
            compare_multi_agent_analysis,
            compare_personas_analysis,
            deep_research_financial,
            get_agent_streaming_analysis,
            list_available_agents,
            orchestrated_analysis,
        )

        # Non-agent tools (no decision logging needed)
        mcp.tool(name="agents_list_available_agents")(list_available_agents)

        # Agent-invoking tools wrapped with decision logging

        @mcp.tool(name="agents_analyze_market_with_agent")
        async def _agents_analyze_market_with_agent(
            query: str,
            persona: str = "moderate",
            screening_strategy: str = "momentum",
            max_results: int = 20,
            session_id: str | None = None,
        ) -> dict:
            """Analyze market conditions using an AI agent with specified persona and strategy."""
            if not _rate_limiter.check_rate_limit(ToolCategory.AGENT):
                return {
                    "error": "Rate limit exceeded for agent tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.AGENT]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await analyze_market_with_agent(
                        query=query,
                        persona=persona,
                        screening_strategy=screening_strategy,
                        max_results=max_results,
                        session_id=session_id,
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="market_screening",
                        routing_decision=["market_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="market_screening",
                        routing_decision=["market_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        @mcp.tool(name="agents_get_agent_streaming_analysis")
        async def _agents_get_agent_streaming_analysis(
            query: str,
            persona: str = "moderate",
            stream_mode: str = "updates",
            session_id: str | None = None,
        ) -> dict:
            """Get streaming analysis from an AI agent for real-time updates."""
            if not _rate_limiter.check_rate_limit(ToolCategory.AGENT):
                return {
                    "error": "Rate limit exceeded for agent tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.AGENT]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await get_agent_streaming_analysis(
                        query=query,
                        persona=persona,
                        stream_mode=stream_mode,
                        session_id=session_id,
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="streaming_analysis",
                        routing_decision=["streaming_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="streaming_analysis",
                        routing_decision=["streaming_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        @mcp.tool(name="agents_compare_personas_analysis")
        async def _agents_compare_personas_analysis(
            query: str, session_id: str | None = None
        ) -> dict:
            """Compare analysis results across different investor personas."""
            if not _rate_limiter.check_rate_limit(ToolCategory.AGENT):
                return {
                    "error": "Rate limit exceeded for agent tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.AGENT]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await compare_personas_analysis(
                        query=query, session_id=session_id
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="persona_comparison",
                        routing_decision=["multi_persona_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="persona_comparison",
                        routing_decision=["multi_persona_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        @mcp.tool(name="agents_orchestrated_analysis")
        async def _agents_orchestrated_analysis(
            query: str,
            persona: str = "moderate",
            routing_strategy: str = "llm_powered",
            max_agents: int = 3,
            parallel_execution: bool = True,
            session_id: str | None = None,
        ) -> dict:
            """Perform orchestrated multi-agent analysis with intelligent routing."""
            if not _rate_limiter.check_rate_limit(ToolCategory.AGENT):
                return {
                    "error": "Rate limit exceeded for agent tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.AGENT]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await orchestrated_analysis(
                        query=query,
                        persona=persona,
                        routing_strategy=routing_strategy,
                        max_agents=max_agents,
                        parallel_execution=parallel_execution,
                        session_id=session_id,
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="orchestrated_analysis",
                        routing_decision=["supervisor_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="orchestrated_analysis",
                        routing_decision=["supervisor_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        @mcp.tool(name="agents_deep_research_financial")
        async def _agents_deep_research_financial(
            research_topic: str,
            persona: str = "moderate",
            research_depth: str = "comprehensive",
            focus_areas: list[str] | None = None,
            timeframe: str = "30d",
            session_id: str | None = None,
        ) -> dict:
            """Perform deep financial research on a topic with AI agents."""
            if not _rate_limiter.check_rate_limit(ToolCategory.AGENT):
                return {
                    "error": "Rate limit exceeded for agent tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.AGENT]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await deep_research_financial(
                        research_topic=research_topic,
                        persona=persona,
                        research_depth=research_depth,
                        focus_areas=focus_areas,
                        timeframe=timeframe,
                        session_id=session_id,
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=research_topic[:500] if research_topic else None,
                        query_classification="deep_research",
                        routing_decision=["deep_research_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=research_topic[:500] if research_topic else None,
                        query_classification="deep_research",
                        routing_decision=["deep_research_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        @mcp.tool(name="agents_compare_multi_agent_analysis")
        async def _agents_compare_multi_agent_analysis(
            query: str,
            agent_types: list[str] | None = None,
            persona: str = "moderate",
            session_id: str | None = None,
        ) -> dict:
            """Compare analysis results from multiple agent types side by side."""
            if not _rate_limiter.check_rate_limit(ToolCategory.AGENT):
                return {
                    "error": "Rate limit exceeded for agent tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.AGENT]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await compare_multi_agent_analysis(
                        query=query,
                        agent_types=agent_types,
                        persona=persona,
                        session_id=session_id,
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="multi_agent_comparison",
                        routing_decision=agent_types or ["market", "technical"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id=session_id or "mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="multi_agent_comparison",
                        routing_decision=agent_types or ["market", "technical"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

    except ImportError:
        # Agents module not available
        pass


def register_research_tools(mcp: FastMCP) -> None:
    """Register deep research tools directly on main server"""
    try:
        # Import all research tools from the consolidated research module
        from maverick_mcp.api.routers.research import (
            analyze_market_sentiment,
            company_comprehensive_research,
            comprehensive_research,
            get_research_agent,
        )

        # Register comprehensive research tool with all enhanced features
        @mcp.tool(name="research_comprehensive_research")
        async def research_comprehensive(
            query: str,
            persona: str | None = "moderate",
            research_scope: str | None = "standard",
            max_sources: int | None = 10,
            timeframe: str | None = "1m",
        ) -> dict:
            """
            Perform comprehensive research on any financial topic using web search and AI analysis.

            Enhanced version with:
            - Adaptive timeout based on research scope (basic: 15s, standard: 30s, comprehensive: 60s, exhaustive: 90s)
            - Step-by-step logging for debugging
            - Guaranteed responses to Claude Desktop
            - Optimized parallel execution for faster results

            Perfect for researching stocks, sectors, market trends, company analysis.
            """
            if not _rate_limiter.check_rate_limit(ToolCategory.RESEARCH):
                return {
                    "error": "Rate limit exceeded for research tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.RESEARCH]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await comprehensive_research(
                        query=query,
                        persona=persona or "moderate",
                        research_scope=research_scope or "standard",
                        max_sources=min(
                            max_sources or 25, 25
                        ),  # Increased cap due to adaptive timeout
                        timeframe=timeframe or "1m",
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="comprehensive_research",
                        routing_decision=["deep_research_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="comprehensive_research",
                        routing_decision=["deep_research_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        # Enhanced sentiment analysis (imported above)
        @mcp.tool(name="research_analyze_market_sentiment")
        async def analyze_market_sentiment_tool(
            topic: str,
            timeframe: str | None = "1w",
            persona: str | None = "moderate",
        ) -> dict:
            """
            Analyze market sentiment for stocks, sectors, or market trends.

            Enhanced version with:
            - 20-second timeout protection
            - Streamlined execution for speed
            - Step-by-step logging for debugging
            - Guaranteed responses
            """
            if not _rate_limiter.check_rate_limit(ToolCategory.RESEARCH):
                return {
                    "error": "Rate limit exceeded for research tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.RESEARCH]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await analyze_market_sentiment(
                        topic=topic,
                        timeframe=timeframe or "1w",
                        persona=persona or "moderate",
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=topic[:500] if topic else None,
                        query_classification="sentiment_analysis",
                        routing_decision=["research_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=topic[:500] if topic else None,
                        query_classification="sentiment_analysis",
                        routing_decision=["research_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        # Enhanced company research (imported above)

        @mcp.tool(name="research_company_comprehensive")
        async def research_company_comprehensive(
            symbol: str,
            include_competitive_analysis: bool = False,
            persona: str | None = "moderate",
        ) -> dict:
            """
            Perform comprehensive company research and fundamental analysis.

            Enhanced version with:
            - 20-second timeout protection to prevent hanging
            - Streamlined analysis for faster execution
            - Step-by-step logging for debugging
            - Focus on core financial metrics
            - Guaranteed responses to Claude Desktop
            """
            if not _rate_limiter.check_rate_limit(ToolCategory.RESEARCH):
                return {
                    "error": "Rate limit exceeded for research tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.RESEARCH]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    result = await company_comprehensive_research(
                        symbol=symbol,
                        include_competitive_analysis=include_competitive_analysis
                        or False,
                        persona=persona or "moderate",
                    )
                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=symbol,
                        query_classification="company_research",
                        routing_decision=["research_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=symbol,
                        query_classification="company_research",
                        routing_decision=["research_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        @mcp.tool(name="research_search_financial_news")
        async def search_financial_news(
            query: str,
            timeframe: str = "1w",
            max_results: int = 20,
            persona: str = "moderate",
        ) -> dict:
            """Search for recent financial news and analysis on any topic."""
            if not _rate_limiter.check_rate_limit(ToolCategory.RESEARCH):
                return {
                    "error": "Rate limit exceeded for research tools. Please try again shortly."
                }
            config = TOOL_CATEGORY_CONFIGS[ToolCategory.RESEARCH]
            start_time = time.time()
            request_id = str(uuid.uuid4())[:8]
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    agent = get_research_agent()

                    # Use basic research for news search
                    result = await agent.research_topic(
                        query=f"{query} news",
                        session_id=f"news_{datetime.now().timestamp()}",
                        research_scope="basic",
                        max_sources=max_results,
                        timeframe=timeframe,
                    )

                duration_ms = int((time.time() - start_time) * 1000)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="news_search",
                        routing_decision=["research_agent"],
                        duration_ms=duration_ms,
                        status="success",
                    )
                except Exception:
                    pass

                return {
                    "success": True,
                    "query": query,
                    "news_results": result.get("processed_sources", [])[:max_results],
                    "total_found": len(result.get("processed_sources", [])),
                    "timeframe": timeframe,
                    "persona": persona,
                }
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                error_cat = classify_error(e)
                try:
                    _decision_logger.log_decision(
                        session_id="mcp",
                        request_id=request_id,
                        query_text=query[:500] if query else None,
                        query_classification="news_search",
                        routing_decision=["research_agent"],
                        duration_ms=duration_ms,
                        status="error",
                        error_category=error_cat.value,
                        response_summary=str(e)[:500],
                    )
                except Exception:
                    pass
                if error_cat == ErrorCategory.TRANSIENT:
                    return {"error": f"Temporary error, please retry: {e}"}
                raise

        logger.info("Successfully registered 4 research tools directly")

    except ImportError as e:
        logger.warning(f"Research module not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register research tools: {e}")
        # Don't raise - allow server to continue without research tools


def register_backtesting_tools(mcp: FastMCP) -> None:
    """Register VectorBT backtesting tools directly on main server"""
    try:
        from maverick_mcp.api.routers.backtesting import setup_backtesting_tools

        setup_backtesting_tools(mcp)
        logger.info("Backtesting tools registered successfully")
    except ImportError:
        logger.warning(
            "Backtesting module not available - VectorBT may not be installed"
        )
    except Exception as e:
        logger.error(f"Failed to register backtesting tools: {e}")


def register_mcp_prompts_and_resources(mcp: FastMCP) -> None:
    """Register MCP prompts and resources for better client introspection"""
    try:
        from maverick_mcp.api.routers.mcp_prompts import register_mcp_prompts

        register_mcp_prompts(mcp)
        logger.info("MCP prompts registered successfully")
    except ImportError:
        logger.warning("MCP prompts module not available")
    except Exception as e:
        logger.error(f"Failed to register MCP prompts: {e}")

    # Register introspection tools
    try:
        from maverick_mcp.api.routers.introspection import register_introspection_tools

        register_introspection_tools(mcp)
        logger.info("Introspection tools registered successfully")
    except ImportError:
        logger.warning("Introspection module not available")
    except Exception as e:
        logger.error(f"Failed to register introspection tools: {e}")


def register_decision_log_tools(mcp: FastMCP) -> None:
    """Register decision audit trail tools directly on main server."""
    from maverick_mcp.utils.decision_logger import decision_logger

    @mcp.tool(name="get_decision_log")
    def get_decision_log(
        session_id: str | None = None,
        limit: int = 20,
    ) -> dict:
        """
        Query the agent decision audit trail.

        Returns recent decision records showing query classifications,
        agent routing, token usage, cost estimates, and outcomes.

        Args:
            session_id: Optional session ID to filter by. If omitted, returns
                        the most recent decisions across all sessions.
            limit: Maximum number of records to return (default 20, max 100).

        Returns:
            Dictionary with decision records and summary statistics.
        """
        limit = min(max(1, limit), 100)
        decisions = decision_logger.get_decisions(session_id=session_id, limit=limit)
        cost_summary = decision_logger.get_cost_summary(days=7)
        return {
            "decisions": decisions,
            "count": len(decisions),
            "cost_summary_7d": cost_summary,
        }


def register_tool_registry_status(mcp: FastMCP) -> None:
    """Register the tool registry status endpoint."""

    @mcp.tool(name="get_tool_registry_status")
    async def get_tool_registry_status() -> dict:
        """Get tool registry status including rate limits and available tool categories.

        Returns current rate limit usage and configuration for every tool category.
        Useful for monitoring tool availability and diagnosing rate-limit issues.

        Returns:
            Dictionary with rate_limits (current usage) and tool_categories (config).
        """
        return {
            "rate_limits": _rate_limiter.get_status(),
            "tool_categories": {
                cat.value: {
                    "description": cfg.description,
                    "timeout_seconds": cfg.timeout_seconds,
                    "rate_limit_per_minute": cfg.max_calls_per_minute,
                    "rate_limit_per_hour": cfg.max_calls_per_hour,
                    "requires_api_key": cfg.requires_api_key,
                }
                for cat, cfg in TOOL_CATEGORY_CONFIGS.items()
            },
        }


def register_all_router_tools(mcp: FastMCP) -> None:
    """Register all router tools directly on the main server"""
    logger.info("Starting tool registration process...")

    try:
        register_technical_tools(mcp)
        logger.info("Technical tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register technical tools: {e}")

    try:
        register_screening_tools(mcp)
        logger.info("Screening tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register screening tools: {e}")

    try:
        register_portfolio_tools(mcp)
        logger.info("Portfolio tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register portfolio tools: {e}")

    try:
        register_data_tools(mcp)
        logger.info("Data tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register data tools: {e}")

    try:
        register_performance_tools(mcp)
        logger.info("Performance tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register performance tools: {e}")

    try:
        register_agent_tools(mcp)
        logger.info("Agent tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register agent tools: {e}")

    try:
        # Import and register research tools on the main MCP instance
        from maverick_mcp.api.routers.research import create_research_router

        # Pass the main MCP instance to register tools directly on it
        create_research_router(mcp)
        logger.info("Research tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register research tools: {e}")

    try:
        # Import and register health monitoring tools
        from maverick_mcp.api.routers.health_tools import register_health_tools

        register_health_tools(mcp)
        logger.info("Health monitoring tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register health monitoring tools: {e}")

    # Register backtesting tools
    register_backtesting_tools(mcp)

    # Register MCP prompts and resources for introspection
    register_mcp_prompts_and_resources(mcp)

    try:
        register_decision_log_tools(mcp)
        logger.info("Decision log tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register decision log tools: {e}")

    try:
        register_tool_registry_status(mcp)
        logger.info("Tool registry status tool registered successfully")
    except Exception as e:
        logger.error(f"Failed to register tool registry status: {e}")

    logger.info("Tool registration process completed")
    logger.info("All tools registered:")
    logger.info("   - Technical analysis tools")
    logger.info("   - Stock screening tools")
    logger.info("   - Portfolio analysis tools")
    logger.info("   - Data retrieval tools")
    logger.info("   - Performance monitoring tools")
    logger.info("   - Agent orchestration tools")
    logger.info("   - Research and analysis tools")
    logger.info("   - Health monitoring tools")
    logger.info("   - Backtesting system tools")
    logger.info("   - MCP prompts for introspection")
    logger.info("   - Introspection and discovery tools")
    logger.info("   - Decision audit trail tools")
    logger.info("   - Tool registry status")
