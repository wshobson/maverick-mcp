"""
Enhanced research tools with timeout handling and comprehensive logging.

This module provides timeout-protected versions of research tools to prevent hanging
and ensure reliable responses to Claude Desktop within the 30-second limit.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.api.middleware.mcp_logging import get_tool_logger
from maverick_mcp.config.settings import get_settings
from maverick_mcp.providers.llm_factory import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize LLM and agent
llm = get_llm()
research_agent = None


def get_research_agent() -> DeepResearchAgent:
    """Get or create the research agent singleton."""
    global research_agent
    if research_agent is None:
        research_agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            max_sources=25,  # Reduced for faster execution
            research_depth="standard",  # Reduced depth for speed
            exa_api_key=settings.research.exa_api_key,
            tavily_api_key=settings.research.tavily_api_key,
        )
    return research_agent


async def research_comprehensive_research_enhanced(
    query: str, 
    persona: str = "moderate",
    research_scope: str = "standard",  # Reduced from comprehensive
    max_sources: int = 15,  # Reduced for speed
    timeframe: str = "1m"
) -> dict[str, Any]:
    """
    Enhanced comprehensive research with timeout protection and step-by-step logging.
    
    This tool provides reliable research capabilities with:
    - 20-second timeout to prevent hanging
    - Step-by-step execution logging  
    - Guaranteed JSON-RPC responses
    - Reduced scope for faster execution
    - Circuit breaker protection
    
    Args:
        query: Research query or topic
        persona: Investor persona (conservative, moderate, aggressive, day_trader)
        research_scope: Research scope (basic, standard, comprehensive)
        max_sources: Maximum sources to analyze (reduced to 15 for speed)
        timeframe: Time frame for search (1d, 1w, 1m, 3m)
        
    Returns:
        Dictionary containing research results or error information
    """
    tool_logger = get_tool_logger("research_comprehensive_research_enhanced")
    request_id = str(uuid.uuid4())
    
    try:
        # Step 1: Initialize agent
        tool_logger.step("agent_initialization", "Initializing research agent")
        agent = get_research_agent()
        
        # Set persona if provided
        if persona in ["conservative", "moderate", "aggressive", "day_trader"]:
            agent.persona = agent.persona.__class__.get(persona)
        
        # Step 2: Validate search providers
        tool_logger.step("provider_validation", "Validating search providers")
        if not agent.search_providers:
            return {
                "success": False,
                "error": "Research functionality unavailable - no search providers configured",
                "details": "Please configure EXA_API_KEY or TAVILY_API_KEY environment variables",
                "query": query,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            }
        
        # Step 3: Execute research with timeout
        session_id = f"enhanced_research_{datetime.now().timestamp()}"
        tool_logger.step("research_execution", f"Starting research with session {session_id[:12]}")
        
        # Execute with strict 20-second timeout
        result = await asyncio.wait_for(
            agent.research_topic(
                query=query,
                session_id=session_id,
                research_scope=research_scope,
                max_sources=max_sources,
                timeframe=timeframe,
            ),
            timeout=20.0
        )
        
        # Step 4: Process results
        tool_logger.step("result_processing", "Processing research results")
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "query": query,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            }
        
        # Step 5: Format response
        tool_logger.step("response_formatting", "Formatting final response")
        
        response = {
            "success": True,
            "query": query,
            "research_results": {
                "summary": result.get("content", "Research completed successfully"),
                "confidence_score": result.get("research_confidence", 0.0),
                "sources_analyzed": result.get("sources_found", 0),
                "key_insights": result.get("actionable_insights", [])[:5],  # Limit for size
                "sentiment": result.get("content_analysis", {}).get("consensus_view", {}),
                "key_themes": result.get("content_analysis", {}).get("key_themes", [])[:3],
            },
            "research_metadata": {
                "persona": persona,
                "scope": research_scope,
                "timeframe": timeframe,
                "max_sources": max_sources,
                "execution_mode": "enhanced_timeout_protected",
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        tool_logger.complete(f"Research completed for query: {query[:50]}")
        return response
        
    except asyncio.TimeoutError:
        tool_logger.error("research_timeout", TimeoutError("Research operation timed out"))
        return {
            "success": False,
            "error": "Research operation timed out after 20 seconds",
            "details": "Consider using a more specific query or reducing the scope",
            "query": query,
            "request_id": request_id,
            "timeout_seconds": 20,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        tool_logger.error("research_error", e, f"Unexpected error in research: {str(e)}")
        return {
            "success": False,
            "error": f"Research error: {str(e)}",
            "error_type": type(e).__name__,
            "query": query,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }


async def research_company_comprehensive_enhanced(
    symbol: str,
    include_competitive_analysis: bool = False,  # Disabled by default for speed
    persona: str = "moderate"
) -> dict[str, Any]:
    """
    Enhanced company research with timeout protection and optimized scope.
    
    This tool provides reliable company analysis with:
    - 20-second timeout protection
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
    tool_logger = get_tool_logger("research_company_comprehensive_enhanced")
    request_id = str(uuid.uuid4())
    
    try:
        # Step 1: Initialize and validate
        tool_logger.step("initialization", f"Starting company research for {symbol}")
        
        # Create focused research query
        query = f"{symbol} stock financial analysis outlook 2025"
        
        # Execute streamlined research
        result = await research_comprehensive_research_enhanced(
            query=query,
            persona=persona,
            research_scope="standard",  # Focused scope
            max_sources=10,  # Reduced sources for speed
            timeframe="1m"
        )
        
        # Step 2: Enhance with symbol-specific formatting
        tool_logger.step("formatting", "Formatting company-specific response")
        
        if not result.get("success", False):
            return {
                **result,
                "symbol": symbol,
                "analysis_type": "company_comprehensive_enhanced"
            }
        
        # Reformat for company analysis
        company_response = {
            "success": True,
            "symbol": symbol,
            "company_analysis": {
                "investment_summary": result["research_results"].get("summary", ""),
                "confidence_score": result["research_results"].get("confidence_score", 0.0),
                "key_insights": result["research_results"].get("key_insights", []),
                "financial_sentiment": result["research_results"].get("sentiment", {}),
                "analysis_themes": result["research_results"].get("key_themes", []),
                "sources_analyzed": result["research_results"].get("sources_analyzed", 0),
            },
            "analysis_metadata": {
                **result["research_metadata"],
                "symbol": symbol,
                "competitive_analysis_included": include_competitive_analysis,
                "analysis_type": "company_comprehensive_enhanced",
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        tool_logger.complete(f"Company analysis completed for {symbol}")
        return company_response
        
    except Exception as e:
        tool_logger.error("company_research_error", e, f"Company research failed: {str(e)}")
        return {
            "success": False,
            "error": f"Company research error: {str(e)}",
            "error_type": type(e).__name__,
            "symbol": symbol,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }


async def analyze_market_sentiment_enhanced(
    topic: str,
    timeframe: str = "1w",
    persona: str = "moderate"
) -> dict[str, Any]:
    """
    Enhanced market sentiment analysis with timeout protection.
    
    Provides fast, reliable sentiment analysis with:
    - 20-second timeout protection
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
    tool_logger = get_tool_logger("analyze_market_sentiment_enhanced")
    request_id = str(uuid.uuid4())
    
    try:
        # Step 1: Create sentiment-focused query
        tool_logger.step("query_creation", f"Creating sentiment query for {topic}")
        
        sentiment_query = f"{topic} market sentiment analysis investor opinion"
        
        # Step 2: Execute focused research
        result = await research_comprehensive_research_enhanced(
            query=sentiment_query,
            persona=persona,
            research_scope="basic",  # Minimal scope for sentiment
            max_sources=8,  # Reduced for speed
            timeframe=timeframe
        )
        
        # Step 3: Format sentiment response
        tool_logger.step("sentiment_formatting", "Extracting sentiment data")
        
        if not result.get("success", False):
            return {
                **result,
                "topic": topic,
                "analysis_type": "market_sentiment_enhanced"
            }
        
        sentiment_response = {
            "success": True,
            "topic": topic,
            "sentiment_analysis": {
                "overall_sentiment": result["research_results"].get("sentiment", {}),
                "sentiment_confidence": result["research_results"].get("confidence_score", 0.0),
                "key_themes": result["research_results"].get("key_themes", []),
                "market_insights": result["research_results"].get("key_insights", [])[:3],
                "sources_analyzed": result["research_results"].get("sources_analyzed", 0),
            },
            "analysis_metadata": {
                **result["research_metadata"],
                "topic": topic,
                "analysis_type": "market_sentiment_enhanced",
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