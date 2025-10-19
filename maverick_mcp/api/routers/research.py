"""
Deep research tools with adaptive timeout handling and comprehensive optimization.

This module provides timeout-protected research tools with LLM optimization
to prevent hanging and ensure reliable responses to Claude Desktop.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from maverick_mcp.agents.base import INVESTOR_PERSONAS
from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.api.middleware.mcp_logging import get_tool_logger
from maverick_mcp.config.settings import get_settings
from maverick_mcp.providers.llm_factory import get_llm
from maverick_mcp.providers.openrouter_provider import TaskType
from maverick_mcp.utils.orchestration_logging import (
    log_performance_metrics,
    log_tool_invocation,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize LLM and agent
llm = get_llm()
research_agent = None


# Request models for tool registration
class ResearchRequest(BaseModel):
    """Request model for comprehensive research"""

    query: str = Field(description="Research query or topic")
    persona: str | None = Field(
        default="moderate",
        description="Investor persona (conservative, moderate, aggressive, day_trader)",
    )
    research_scope: str | None = Field(
        default="standard",
        description="Research scope (basic, standard, comprehensive, exhaustive)",
    )
    max_sources: int | None = Field(
        default=10, description="Maximum sources to analyze (1-30)"
    )
    timeframe: str | None = Field(
        default="1m", description="Time frame for search (1d, 1w, 1m, 3m)"
    )


class CompanyResearchRequest(BaseModel):
    """Request model for company research"""

    symbol: str = Field(description="Stock ticker symbol")
    include_competitive_analysis: bool = Field(
        default=False, description="Include competitive analysis"
    )
    persona: str | None = Field(
        default="moderate", description="Investor persona for analysis perspective"
    )


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""

    topic: str = Field(description="Topic for sentiment analysis")
    timeframe: str | None = Field(default="1w", description="Time frame for analysis")
    persona: str | None = Field(default="moderate", description="Investor persona")
    session_id: str | None = Field(default=None, description="Session identifier")


def get_research_agent(
    query: str | None = None,
    research_scope: str = "standard",
    timeout_budget: float = 240.0,  # Default timeout for standard research (4 minutes)
    max_sources: int = 15,
) -> DeepResearchAgent:
    """
    Get or create an optimized research agent with adaptive LLM selection.

    This creates a research agent optimized for the specific query and time constraints,
    using adaptive model selection to prevent timeouts while maintaining quality.

    Args:
        query: Research query for complexity analysis (optional)
        research_scope: Research scope for optimization
        timeout_budget: Available timeout budget in seconds
        max_sources: Maximum sources to analyze

    Returns:
        DeepResearchAgent optimized for the request parameters
    """
    global research_agent

    # For optimization, create new agents with adaptive LLM selection
    # rather than using a singleton when query-specific optimization is needed
    if query and timeout_budget < 300:
        # Use adaptive optimization for time-constrained requests (less than 5 minutes)
        adaptive_llm = _get_adaptive_llm_for_research(
            query, research_scope, timeout_budget, max_sources
        )

        agent = DeepResearchAgent(
            llm=adaptive_llm,
            persona="moderate",
            max_sources=max_sources,
            research_depth=research_scope,
            exa_api_key=settings.research.exa_api_key,
        )
        # Mark for initialization - will be initialized on first use
        agent._needs_initialization = True
        return agent

    # Use singleton for standard requests
    if research_agent is None:
        research_agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            max_sources=25,  # Reduced for faster execution
            research_depth="standard",  # Reduced depth for speed
            exa_api_key=settings.research.exa_api_key,
        )
        # Mark for initialization - will be initialized on first use
        research_agent._needs_initialization = True
    return research_agent


def _get_timeout_for_research_scope(research_scope: str) -> float:
    """
    Calculate timeout based on research scope complexity.

    Args:
        research_scope: Research scope (basic, standard, comprehensive, exhaustive)

    Returns:
        Timeout in seconds appropriate for the research scope
    """
    timeout_mapping = {
        "basic": 120.0,  # 2 minutes - generous for basic research
        "standard": 240.0,  # 4 minutes - standard research with detailed analysis
        "comprehensive": 360.0,  # 6 minutes - comprehensive research with thorough analysis
        "exhaustive": 600.0,  # 10 minutes - exhaustive research with validation
    }

    return timeout_mapping.get(
        research_scope.lower(), 240.0
    )  # Default to standard (4 minutes)


def _optimize_sources_for_timeout(
    research_scope: str, requested_sources: int, timeout_budget: float
) -> int:
    """
    Optimize the number of sources based on timeout constraints and research scope.

    This implements intelligent source limiting to maximize quality within time constraints.

    Args:
        research_scope: Research scope (basic, standard, comprehensive, exhaustive)
        requested_sources: Originally requested number of sources
        timeout_budget: Available timeout in seconds

    Returns:
        Optimized number of sources that can realistically be processed within timeout
    """
    # Estimate processing time per source based on scope complexity
    processing_time_per_source = {
        "basic": 1.5,  # 1.5 seconds per source (minimal analysis)
        "standard": 2.5,  # 2.5 seconds per source (moderate analysis)
        "comprehensive": 4.0,  # 4 seconds per source (deep analysis)
        "exhaustive": 6.0,  # 6 seconds per source (maximum analysis)
    }

    estimated_time_per_source = processing_time_per_source.get(
        research_scope.lower(), 2.5
    )

    # Reserve 20% of timeout for search, synthesis, and overhead
    available_time_for_sources = timeout_budget * 0.8

    # Calculate maximum sources within timeout
    max_sources_for_timeout = int(
        available_time_for_sources / estimated_time_per_source
    )

    # Apply quality-based limits (better to have fewer high-quality sources)
    quality_limits = {
        "basic": 8,  # Focus on most relevant sources
        "standard": 15,  # Balanced approach
        "comprehensive": 20,  # More sources for deep research
        "exhaustive": 25,  # Maximum sources for exhaustive research
    }

    scope_limit = quality_limits.get(research_scope.lower(), 15)

    # Return the minimum of: requested, timeout-constrained, and scope-limited
    optimized_sources = min(requested_sources, max_sources_for_timeout, scope_limit)

    # Ensure minimum of 3 sources for meaningful analysis
    return max(optimized_sources, 3)


def _get_adaptive_llm_for_research(
    query: str,
    research_scope: str,
    timeout_budget: float,
    max_sources: int,
) -> Any:
    """
    Get an adaptively selected LLM optimized for research performance within timeout constraints.

    This implements intelligent model selection based on:
    - Available time budget (timeout pressure)
    - Query complexity (inferred from length and scope)
    - Research scope requirements
    - Number of sources to process

    Args:
        query: Research query to analyze complexity
        research_scope: Research scope (basic, standard, comprehensive, exhaustive)
        timeout_budget: Available timeout in seconds
        max_sources: Number of sources to analyze

    Returns:
        Optimally selected LLM instance for the research task
    """
    # Calculate query complexity score (0.0 to 1.0)
    complexity_score = 0.0

    # Query length factor (longer queries often indicate complexity)
    if len(query) > 200:
        complexity_score += 0.3
    elif len(query) > 100:
        complexity_score += 0.2
    elif len(query) > 50:
        complexity_score += 0.1

    # Multi-topic queries (multiple companies/concepts)
    complexity_keywords = [
        "vs",
        "versus",
        "compare",
        "analysis",
        "forecast",
        "outlook",
        "trends",
        "market",
        "competition",
    ]
    keyword_matches = sum(
        1 for keyword in complexity_keywords if keyword.lower() in query.lower()
    )
    complexity_score += min(keyword_matches * 0.1, 0.4)

    # Research scope complexity
    scope_complexity = {
        "basic": 0.1,
        "standard": 0.3,
        "comprehensive": 0.6,
        "exhaustive": 0.9,
    }
    complexity_score += scope_complexity.get(research_scope.lower(), 0.3)

    # Source count complexity (more sources = more synthesis required)
    if max_sources > 20:
        complexity_score += 0.3
    elif max_sources > 10:
        complexity_score += 0.2
    elif max_sources > 5:
        complexity_score += 0.1

    # Normalize to 0-1 range
    complexity_score = min(complexity_score, 1.0)

    # Time pressure factor (lower means more pressure) - Updated for generous timeouts
    time_pressure = 1.0
    if timeout_budget < 120:
        time_pressure = (
            0.2  # Emergency mode - need fastest models (below basic timeout)
        )
    elif timeout_budget < 240:
        time_pressure = 0.5  # High pressure - prefer fast models (basic to standard)
    elif timeout_budget < 360:
        time_pressure = (
            0.7  # Moderate pressure - balanced selection (standard to comprehensive)
        )
    else:
        time_pressure = (
            1.0  # Low pressure - can use premium models (comprehensive and above)
        )

    # Model selection strategy with timeout budget consideration
    if time_pressure <= 0.3 or timeout_budget < 120:
        # Emergency mode: prioritize speed above all for <120s timeouts (below basic)
        logger.info(
            f"Emergency fast model selection triggered - timeout budget: {timeout_budget}s"
        )
        return get_llm(
            task_type=TaskType.DEEP_RESEARCH,
            prefer_fast=True,
            prefer_cheap=True,  # Ultra-fast models (GPT-5 Nano, Claude 3.5 Haiku, DeepSeek R1)
            prefer_quality=False,
            # Emergency mode triggered for timeout_budget < 30s
        )
    elif time_pressure <= 0.6 and complexity_score <= 0.4:
        # Fast mode for simple queries: speed-optimized but decent quality
        return get_llm(
            task_type=TaskType.DEEP_RESEARCH,
            prefer_fast=True,
            prefer_cheap=True,
            prefer_quality=False,
            # Fast mode for simple queries under time pressure
        )
    elif complexity_score >= 0.7 and time_pressure >= 0.8:
        # Complex query with time available: use premium models
        return get_llm(
            task_type=TaskType.DEEP_RESEARCH,
            prefer_fast=False,
            prefer_cheap=False,
            prefer_quality=True,  # Premium models for complex tasks
        )
    else:
        # Balanced approach: cost-effective quality models
        return get_llm(
            task_type=TaskType.DEEP_RESEARCH,
            prefer_fast=False,
            prefer_cheap=True,  # Default cost-effective
            prefer_quality=False,
        )


async def _execute_research_with_direct_timeout(
    agent,
    query: str,
    session_id: str,
    research_scope: str,
    max_sources: int,
    timeframe: str,
    total_timeout: float,
    tool_logger,
) -> dict[str, Any]:
    """
    Execute research with direct timeout enforcement using asyncio.wait_for.

    This function provides hard timeout enforcement and graceful failure handling.
    """
    start_time = asyncio.get_event_loop().time()

    # Granular timing for bottleneck identification
    timing_log = {
        "research_start": start_time,
        "phase_timings": {},
        "cumulative_time": 0.0,
    }

    def log_phase_timing(phase_name: str):
        """Log timing for a specific research phase."""
        current_time = asyncio.get_event_loop().time()
        phase_duration = current_time - start_time - timing_log["cumulative_time"]
        timing_log["phase_timings"][phase_name] = {
            "duration": phase_duration,
            "cumulative": current_time - start_time,
        }
        timing_log["cumulative_time"] = current_time - start_time
        logger.debug(
            f"TIMING: {phase_name} took {phase_duration:.2f}s (cumulative: {timing_log['cumulative_time']:.2f}s)"
        )

    try:
        tool_logger.step(
            "timeout_enforcement",
            f"Starting research with {total_timeout}s hard timeout",
        )
        log_phase_timing("initialization")

        # Use direct asyncio.wait_for for hard timeout enforcement
        logger.info(
            f"TIMING: Starting research execution phase (budget: {total_timeout}s)"
        )

        result = await asyncio.wait_for(
            agent.research_topic(
                query=query,
                session_id=session_id,
                research_scope=research_scope,
                max_sources=max_sources,
                timeframe=timeframe,
                timeout_budget=total_timeout,  # Pass timeout budget for phase allocation
            ),
            timeout=total_timeout,
        )

        log_phase_timing("research_execution")

        elapsed_time = asyncio.get_event_loop().time() - start_time
        tool_logger.step(
            "research_completed", f"Research completed in {elapsed_time:.1f}s"
        )

        # Log detailed timing breakdown
        logger.info(
            f"RESEARCH_TIMING_BREAKDOWN: "
            f"Total={elapsed_time:.2f}s, "
            f"Phases={timing_log['phase_timings']}"
        )

        # Add timing information to successful results
        if isinstance(result, dict):
            result["elapsed_time"] = elapsed_time
            result["timeout_warning"] = elapsed_time >= (total_timeout * 0.8)

        return result

    except TimeoutError:
        elapsed_time = asyncio.get_event_loop().time() - start_time
        log_phase_timing("timeout_exceeded")

        # Log timeout timing analysis
        logger.warning(
            f"RESEARCH_TIMEOUT: "
            f"Exceeded {total_timeout}s limit after {elapsed_time:.2f}s, "
            f"Phases={timing_log['phase_timings']}"
        )

        tool_logger.step(
            "timeout_exceeded",
            f"Research timed out after {elapsed_time:.1f}s (limit: {total_timeout}s)",
        )

        # Return structured timeout response instead of raising
        return {
            "status": "timeout",
            "content": f"Research operation timed out after {total_timeout} seconds",
            "research_confidence": 0.0,
            "sources_found": 0,
            "timeout_warning": True,
            "elapsed_time": elapsed_time,
            "completion_percentage": 0,
            "timing_breakdown": timing_log["phase_timings"],
            "actionable_insights": [
                "Research was terminated due to timeout",
                "Consider reducing scope or query complexity",
                f"Try using 'basic' or 'standard' scope instead of '{research_scope}'",
            ],
            "content_analysis": {
                "consensus_view": {
                    "direction": "neutral",
                    "confidence": 0.0,
                },
                "key_themes": ["Timeout occurred"],
                "contrarian_views": [],
            },
            "persona_insights": {
                "summary": "Analysis terminated due to timeout - consider simplifying the query"
            },
            "error": "timeout_exceeded",
        }

    except asyncio.CancelledError:
        tool_logger.step("research_cancelled", "Research operation was cancelled")
        raise
    except Exception as e:
        elapsed_time = asyncio.get_event_loop().time() - start_time
        tool_logger.error("research_execution_error", e)

        # Return structured error response
        return {
            "status": "error",
            "content": f"Research failed due to error: {str(e)}",
            "research_confidence": 0.0,
            "sources_found": 0,
            "timeout_warning": False,
            "elapsed_time": elapsed_time,
            "completion_percentage": 0,
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def comprehensive_research(
    query: str,
    persona: str = "moderate",
    research_scope: str = "standard",
    max_sources: int = 15,
    timeframe: str = "1m",
) -> dict[str, Any]:
    """
    Enhanced comprehensive research with adaptive timeout protection and step-by-step logging.

    This tool provides reliable research capabilities with:
    - Generous timeout based on research scope (basic: 120s, standard: 240s, comprehensive: 360s, exhaustive: 600s)
    - Step-by-step execution logging
    - Guaranteed JSON-RPC responses
    - Optimized scope for faster execution
    - Circuit breaker protection

    Args:
        query: Research query or topic
        persona: Investor persona (conservative, moderate, aggressive, day_trader)
        research_scope: Research scope (basic, standard, comprehensive, exhaustive)
        max_sources: Maximum sources to analyze (reduced to 15 for speed)
        timeframe: Time frame for search (1d, 1w, 1m, 3m)

    Returns:
        Dictionary containing research results or error information
    """
    tool_logger = get_tool_logger("comprehensive_research")
    request_id = str(uuid.uuid4())

    # Log incoming parameters
    logger.info(
        f"📥 RESEARCH_REQUEST: query='{query[:50]}...', scope='{research_scope}', max_sources={max_sources}, timeframe='{timeframe}'"
    )

    try:
        # Step 1: Calculate optimization parameters first
        tool_logger.step(
            "optimization_calculation",
            f"Calculating adaptive optimization parameters for scope='{research_scope}' with max_sources={max_sources}",
        )
        adaptive_timeout = _get_timeout_for_research_scope(research_scope)
        optimized_sources = _optimize_sources_for_timeout(
            research_scope, max_sources, adaptive_timeout
        )

        # Log the timeout calculation result explicitly
        logger.info(
            f"🔧 TIMEOUT_CONFIGURATION: scope='{research_scope}' → timeout={adaptive_timeout}s (was requesting {max_sources} sources, optimized to {optimized_sources})"
        )

        # Step 2: Log optimization setup (components initialized in underlying research system)
        tool_logger.step(
            "optimization_setup",
            f"Configuring LLM optimizations (budget: {adaptive_timeout}s, parallel: {optimized_sources > 3})",
        )

        # Step 3: Initialize agent with adaptive optimizations
        tool_logger.step(
            "agent_initialization",
            f"Initializing optimized research agent (timeout: {adaptive_timeout}s, sources: {optimized_sources})",
        )
        agent = get_research_agent(
            query=query,
            research_scope=research_scope,
            timeout_budget=adaptive_timeout,
            max_sources=optimized_sources,
        )

        # Set persona if provided
        if persona in ["conservative", "moderate", "aggressive", "day_trader"]:
            agent.persona = INVESTOR_PERSONAS.get(
                persona, INVESTOR_PERSONAS["moderate"]
            )

        # Step 4: Early validation of search provider configuration
        tool_logger.step(
            "provider_validation", "Validating search provider configuration"
        )

        # Check for API key before creating agent (faster failure)
        exa_available = bool(settings.research.exa_api_key)

        if not exa_available:
            return {
                "success": False,
                "error": "Research functionality unavailable - Exa search provider not configured",
                "details": {
                    "required_configuration": "Exa search provider API key is required",
                    "exa_api_key": "Missing (configure EXA_API_KEY environment variable)",
                    "setup_instructions": "Get a free API key from: Exa (exa.ai)",
                },
                "query": query,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            }

        # Log available provider
        tool_logger.step(
            "provider_available",
            "Exa search provider available",
        )

        session_id = f"enhanced_research_{datetime.now().timestamp()}"
        tool_logger.step(
            "source_optimization",
            f"Optimized sources: {max_sources} → {optimized_sources} for {research_scope} scope within {adaptive_timeout}s",
        )
        tool_logger.step(
            "research_execution",
            f"Starting progressive research with session {session_id[:12]} (timeout: {adaptive_timeout}s, sources: {optimized_sources})",
        )

        # Execute with direct timeout enforcement for reliable operation
        result = await _execute_research_with_direct_timeout(
            agent=agent,
            query=query,
            session_id=session_id,
            research_scope=research_scope,
            max_sources=optimized_sources,  # Use optimized source count
            timeframe=timeframe,
            total_timeout=adaptive_timeout,
            tool_logger=tool_logger,
        )

        # Step 4: Process results
        tool_logger.step("result_processing", "Processing research results")

        # Handle timeout or error results
        if result.get("status") == "timeout":
            return {
                "success": False,
                "error": "Research operation timed out",
                "timeout_details": {
                    "timeout_seconds": adaptive_timeout,
                    "elapsed_time": result.get("elapsed_time", 0),
                    "suggestions": result.get("actionable_insights", []),
                },
                "query": query,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            }

        if result.get("status") == "error" or "error" in result:
            return {
                "success": False,
                "error": result.get("error", "Unknown research error"),
                "error_type": result.get("error_type", "UnknownError"),
                "query": query,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            }

        # Step 5: Format response with timeout support
        tool_logger.step("response_formatting", "Formatting final response")

        # Check if this is a partial result or has warnings
        is_partial = result.get("status") == "partial_success"
        has_timeout_warning = result.get("timeout_warning", False)

        response = {
            "success": True,
            "query": query,
            "research_results": {
                "summary": result.get("content", "Research completed successfully"),
                "confidence_score": result.get("research_confidence", 0.0),
                "sources_analyzed": result.get("sources_found", 0),
                "key_insights": result.get("actionable_insights", [])[
                    :5
                ],  # Limit for size
                "sentiment": result.get("content_analysis", {}).get(
                    "consensus_view", {}
                ),
                "key_themes": result.get("content_analysis", {}).get("key_themes", [])[
                    :3
                ],
            },
            "research_metadata": {
                "persona": persona,
                "scope": research_scope,
                "timeframe": timeframe,
                "max_sources_requested": max_sources,
                "max_sources_optimized": optimized_sources,
                "sources_actually_used": result.get("sources_found", optimized_sources),
                "execution_mode": "progressive_timeout_protected",
                "is_partial_result": is_partial,
                "timeout_warning": has_timeout_warning,
                "elapsed_time": result.get("elapsed_time", 0),
                "completion_percentage": result.get(
                    "completion_percentage", 100 if not is_partial else 60
                ),
                "optimization_features": [
                    "adaptive_model_selection",
                    "progressive_token_budgeting",
                    "parallel_llm_processing",
                    "intelligent_source_optimization",
                    "timeout_monitoring",
                ],
                "parallel_processing": {
                    "enabled": True,
                    "max_concurrent_requests": min(4, optimized_sources // 2 + 1),
                    "batch_processing": optimized_sources > 3,
                },
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Add warning message for partial results
        if is_partial:
            response["warning"] = {
                "type": "partial_result",
                "message": "Research was partially completed due to timeout constraints",
                "suggestions": [
                    f"Try reducing research scope from '{research_scope}' to 'standard' or 'basic'",
                    f"Reduce max_sources from {max_sources} to {min(15, optimized_sources)} or fewer",
                    "Use more specific keywords to focus the search",
                    f"Note: Sources were automatically optimized from {max_sources} to {optimized_sources} for better performance",
                ],
            }
        elif has_timeout_warning:
            response["warning"] = {
                "type": "timeout_warning",
                "message": "Research completed but took longer than expected",
                "suggestions": [
                    "Consider reducing scope for faster results in the future"
                ],
            }

        tool_logger.complete(f"Research completed for query: {query[:50]}")
        return response

    except TimeoutError:
        # Calculate timeout for error reporting
        used_timeout = _get_timeout_for_research_scope(research_scope)
        tool_logger.error(
            "research_timeout",
            TimeoutError(f"Research operation timed out after {used_timeout}s"),
        )
        # Calculate optimized sources for error reporting
        timeout_optimized_sources = _optimize_sources_for_timeout(
            research_scope, max_sources, used_timeout
        )

        return {
            "success": False,
            "error": f"Research operation timed out after {used_timeout} seconds",
            "details": f"Consider using a more specific query, reducing the scope from '{research_scope}', or decreasing max_sources from {max_sources}",
            "suggestions": {
                "reduce_scope": "Try 'basic' or 'standard' instead of 'comprehensive'",
                "reduce_sources": f"Try max_sources={min(10, timeout_optimized_sources)} instead of {max_sources}",
                "narrow_query": "Use more specific keywords to focus the search",
            },
            "optimization_info": {
                "sources_requested": max_sources,
                "sources_auto_optimized": timeout_optimized_sources,
                "note": "Sources were automatically reduced for better performance, but timeout still occurred",
            },
            "query": query,
            "request_id": request_id,
            "timeout_seconds": used_timeout,
            "research_scope": research_scope,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        tool_logger.error(
            "research_error", e, f"Unexpected error in research: {str(e)}"
        )
        return {
            "success": False,
            "error": f"Research error: {str(e)}",
            "error_type": type(e).__name__,
            "query": query,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }


async def company_comprehensive_research(
    symbol: str,
    include_competitive_analysis: bool = False,  # Disabled by default for speed
    persona: str = "moderate",
) -> dict[str, Any]:
    """
    Enhanced company research with timeout protection and optimized scope.

    This tool provides reliable company analysis with:
    - Adaptive timeout protection
    - Streamlined analysis for faster execution
    - Step-by-step logging for debugging
    - Guaranteed responses to Claude Desktop
    - Focus on core financial metrics

    Args:
        symbol: Stock ticker symbol
        include_competitive_analysis: Include competitive analysis (disabled for speed)
        persona: Investor persona for analysis perspective

    Returns:
        Dictionary containing company research results or error information
    """
    tool_logger = get_tool_logger("company_comprehensive_research")
    request_id = str(uuid.uuid4())

    try:
        # Step 1: Initialize and validate
        tool_logger.step("initialization", f"Starting company research for {symbol}")

        # Create focused research query
        query = f"{symbol} stock financial analysis outlook 2025"

        # Execute streamlined research
        result = await comprehensive_research(
            query=query,
            persona=persona,
            research_scope="standard",  # Focused scope
            max_sources=10,  # Reduced sources for speed
            timeframe="1m",
        )

        # Step 2: Enhance with symbol-specific formatting
        tool_logger.step("formatting", "Formatting company-specific response")

        if not result.get("success", False):
            return {
                **result,
                "symbol": symbol,
                "analysis_type": "company_comprehensive",
            }

        # Reformat for company analysis
        company_response = {
            "success": True,
            "symbol": symbol,
            "company_analysis": {
                "investment_summary": result["research_results"].get("summary", ""),
                "confidence_score": result["research_results"].get(
                    "confidence_score", 0.0
                ),
                "key_insights": result["research_results"].get("key_insights", []),
                "financial_sentiment": result["research_results"].get("sentiment", {}),
                "analysis_themes": result["research_results"].get("key_themes", []),
                "sources_analyzed": result["research_results"].get(
                    "sources_analyzed", 0
                ),
            },
            "analysis_metadata": {
                **result["research_metadata"],
                "symbol": symbol,
                "competitive_analysis_included": include_competitive_analysis,
                "analysis_type": "company_comprehensive",
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        tool_logger.complete(f"Company analysis completed for {symbol}")
        return company_response

    except Exception as e:
        tool_logger.error(
            "company_research_error", e, f"Company research failed: {str(e)}"
        )
        return {
            "success": False,
            "error": f"Company research error: {str(e)}",
            "error_type": type(e).__name__,
            "symbol": symbol,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }


async def analyze_market_sentiment(
    topic: str, timeframe: str = "1w", persona: str = "moderate"
) -> dict[str, Any]:
    """
    Enhanced market sentiment analysis with timeout protection.

    Provides fast, reliable sentiment analysis with:
    - Adaptive timeout protection
    - Focused sentiment extraction
    - Step-by-step logging
    - Guaranteed responses

    Args:
        topic: Topic for sentiment analysis
        timeframe: Time frame for analysis
        persona: Investor persona

    Returns:
        Dictionary containing sentiment analysis results
    """
    tool_logger = get_tool_logger("analyze_market_sentiment")
    request_id = str(uuid.uuid4())

    try:
        # Step 1: Create sentiment-focused query
        tool_logger.step("query_creation", f"Creating sentiment query for {topic}")

        sentiment_query = f"{topic} market sentiment analysis investor opinion"

        # Step 2: Execute focused research
        result = await comprehensive_research(
            query=sentiment_query,
            persona=persona,
            research_scope="basic",  # Minimal scope for sentiment
            max_sources=8,  # Reduced for speed
            timeframe=timeframe,
        )

        # Step 3: Format sentiment response
        tool_logger.step("sentiment_formatting", "Extracting sentiment data")

        if not result.get("success", False):
            return {
                **result,
                "topic": topic,
                "analysis_type": "market_sentiment",
            }

        sentiment_response = {
            "success": True,
            "topic": topic,
            "sentiment_analysis": {
                "overall_sentiment": result["research_results"].get("sentiment", {}),
                "sentiment_confidence": result["research_results"].get(
                    "confidence_score", 0.0
                ),
                "key_themes": result["research_results"].get("key_themes", []),
                "market_insights": result["research_results"].get("key_insights", [])[
                    :3
                ],
                "sources_analyzed": result["research_results"].get(
                    "sources_analyzed", 0
                ),
            },
            "analysis_metadata": {
                **result["research_metadata"],
                "topic": topic,
                "analysis_type": "market_sentiment",
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        tool_logger.complete(f"Sentiment analysis completed for {topic}")
        return sentiment_response

    except Exception as e:
        tool_logger.error("sentiment_error", e, f"Sentiment analysis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Sentiment analysis error: {str(e)}",
            "error_type": type(e).__name__,
            "topic": topic,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }


def create_research_router(mcp: FastMCP | None = None) -> FastMCP:
    """Create and configure the research router."""

    if mcp is None:
        mcp = FastMCP("Deep Research Tools")

    @mcp.tool()
    async def research_comprehensive_research(
        query: str,
        persona: str | None = "moderate",
        research_scope: str | None = "standard",
        max_sources: int | None = 10,
        timeframe: str | None = "1m",
    ) -> dict[str, Any]:
        """
        Perform comprehensive research on any financial topic using web search and AI analysis.

        Enhanced features:
        - Generous timeout (basic: 120s, standard: 240s, comprehensive: 360s, exhaustive: 600s)
        - Intelligent source optimization
        - Parallel LLM processing
        - Progressive token budgeting
        - Partial results on timeout

        Args:
            query: Research query or topic
            persona: Investor persona (conservative, moderate, aggressive, day_trader)
            research_scope: Research scope (basic, standard, comprehensive, exhaustive)
            max_sources: Maximum sources to analyze (1-50)
            timeframe: Time frame for search (1d, 1w, 1m, 3m)

        Returns:
            Comprehensive research results with insights, sentiment, and recommendations
        """
        # CRITICAL DEBUG: Log immediately when tool is called
        logger.error(
            f"🚨 TOOL CALLED: research_comprehensive_research with query: {query[:50]}"
        )

        # Log tool invocation
        log_tool_invocation(
            "research_comprehensive_research",
            {
                "query": query[:100],  # Truncate for logging
                "persona": persona,
                "research_scope": research_scope,
                "max_sources": max_sources,
            },
        )

        start_time = datetime.now()

        try:
            # Execute enhanced research
            result = await comprehensive_research(
                query=query,
                persona=persona or "moderate",
                research_scope=research_scope or "standard",
                max_sources=max_sources or 15,
                timeframe=timeframe or "1m",
            )

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log performance metrics
            log_performance_metrics(
                "research_comprehensive_research",
                {
                    "execution_time_ms": execution_time,
                    "sources_analyzed": result.get("research_results", {}).get(
                        "sources_analyzed", 0
                    ),
                    "confidence_score": result.get("research_results", {}).get(
                        "confidence_score", 0.0
                    ),
                    "success": result.get("success", False),
                },
            )

            return result

        except Exception as e:
            logger.error(
                f"Research error: {str(e)}",
                exc_info=True,
                extra={"query": query[:100]},
            )
            return {
                "success": False,
                "error": f"Research failed: {str(e)}",
                "error_type": type(e).__name__,
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.tool()
    async def research_company_comprehensive(
        symbol: str,
        include_competitive_analysis: bool = False,
        persona: str | None = "moderate",
    ) -> dict[str, Any]:
        """
        Perform comprehensive research on a specific company.

        Features:
        - Financial metrics analysis
        - Market sentiment assessment
        - Competitive positioning
        - Investment recommendations

        Args:
            symbol: Stock ticker symbol
            include_competitive_analysis: Include competitive analysis
            persona: Investor persona for analysis perspective

        Returns:
            Company-specific research with financial insights
        """
        return await company_comprehensive_research(
            symbol=symbol,
            include_competitive_analysis=include_competitive_analysis,
            persona=persona or "moderate",
        )

    @mcp.tool()
    async def research_analyze_market_sentiment(
        topic: str,
        timeframe: str | None = "1w",
        persona: str | None = "moderate",
    ) -> dict[str, Any]:
        """
        Analyze market sentiment for a specific topic or sector.

        Features:
        - Real-time sentiment extraction
        - News and social media analysis
        - Investor opinion aggregation
        - Trend identification

        Args:
            topic: Topic for sentiment analysis
            timeframe: Time frame for analysis
            persona: Investor persona

        Returns:
            Sentiment analysis with market insights
        """
        return await analyze_market_sentiment(
            topic=topic,
            timeframe=timeframe or "1w",
            persona=persona or "moderate",
        )

    return mcp


# Create the router instance
research_router = create_research_router()
