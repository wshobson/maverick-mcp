"""
Tool registry to register router tools directly on main server.
This avoids Claude Desktop's issue with mounted router tool names.
"""

import logging
from datetime import datetime

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_technical_tools(mcp: FastMCP) -> None:
    """Register technical analysis tools directly on main server"""
    from maverick_mcp.api.routers.technical import (
        get_full_technical_analysis,
        get_macd_analysis,
        get_rsi_analysis,
        get_stock_chart_analysis,
        get_support_resistance,
    )

    # Register with prefixed names to maintain organization
    mcp.tool(name="technical_get_rsi_analysis")(get_rsi_analysis)
    mcp.tool(name="technical_get_macd_analysis")(get_macd_analysis)
    mcp.tool(name="technical_get_support_resistance")(get_support_resistance)
    mcp.tool(name="technical_get_full_technical_analysis")(get_full_technical_analysis)
    mcp.tool(name="technical_get_stock_chart_analysis")(get_stock_chart_analysis)


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
        compare_tickers,
        portfolio_correlation_analysis,
        risk_adjusted_analysis,
    )

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
        get_news_sentiment,
        get_stock_info,
    )

    mcp.tool(name="data_fetch_stock_data")(fetch_stock_data)
    mcp.tool(name="data_fetch_stock_data_batch")(fetch_stock_data_batch)
    mcp.tool(name="data_get_stock_info")(get_stock_info)
    mcp.tool(name="data_get_news_sentiment")(get_news_sentiment)
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

        # Original agent tools
        mcp.tool(name="agents_analyze_market_with_agent")(analyze_market_with_agent)
        mcp.tool(name="agents_get_agent_streaming_analysis")(
            get_agent_streaming_analysis
        )
        mcp.tool(name="agents_list_available_agents")(list_available_agents)
        mcp.tool(name="agents_compare_personas_analysis")(compare_personas_analysis)

        # New orchestration tools
        mcp.tool(name="agents_orchestrated_analysis")(orchestrated_analysis)
        mcp.tool(name="agents_deep_research_financial")(deep_research_financial)
        mcp.tool(name="agents_compare_multi_agent_analysis")(
            compare_multi_agent_analysis
        )
    except ImportError:
        # Agents module not available
        pass


def register_research_tools(mcp: FastMCP) -> None:
    """Register deep research tools directly on main server"""
    try:
        # Import the tool functions directly from the research router
        # This is cleaner than extracting from router and avoids async complications
        from maverick_mcp.api.routers.research import (
            get_research_agent,
            ResearchRequest, 
            SentimentAnalysisRequest,
            CompanyResearchRequest
        )
        
        # Register research tool functions directly with prefixed names
        @mcp.tool(name="research_comprehensive_research")
        async def comprehensive_research(request: ResearchRequest) -> dict:
            """
            Perform comprehensive research on any financial topic using web search and AI analysis.
            Perfect for researching stocks, sectors, market trends, company analysis.
            """
            agent = get_research_agent()
            if request.persona and request.persona in ["conservative", "moderate", "aggressive", "day_trader"]:
                agent.persona = agent.persona.__class__.get(request.persona)
            
            session_id = request.session_id or f"research_{datetime.now().timestamp()}"
            
            result = await agent.research_topic(
                query=request.query,
                session_id=session_id,
                research_scope=request.research_scope,
                max_sources=request.max_sources,
                timeframe=request.timeframe,
            )
            
            if "error" in result:
                return {"success": False, "error": result["error"], "query": request.query}
                
            return {
                "success": True,
                "query": request.query,
                "research_results": {
                    "summary": result.get("content", "Research completed"),
                    "confidence_score": result.get("research_confidence", 0.0),
                    "sources_analyzed": result.get("sources_found", 0),
                    "persona_insights": result.get("persona_insights", {}),
                    "key_themes": result.get("content_analysis", {}).get("key_themes", []),
                    "sentiment": result.get("content_analysis", {}).get("consensus_view", {}),
                    "actionable_insights": result.get("actionable_insights", []),
                }
            }

        @mcp.tool(name="research_analyze_market_sentiment")
        async def analyze_market_sentiment(request: SentimentAnalysisRequest) -> dict:
            """Analyze market sentiment for stocks, sectors, or market trends."""
            agent = get_research_agent()
            if request.persona and request.persona in ["conservative", "moderate", "aggressive", "day_trader"]:
                agent.persona = agent.persona.__class__.get(request.persona)
            
            session_id = request.session_id or f"sentiment_{datetime.now().timestamp()}"
            
            result = await agent.analyze_market_sentiment(
                topic=request.topic, 
                session_id=session_id, 
                timeframe=request.timeframe
            )
            
            if "error" in result:
                return {"success": False, "error": result["error"], "topic": request.topic}
                
            return {
                "success": True,
                "topic": request.topic,
                "sentiment_analysis": {
                    "overall_sentiment": result.get("content_analysis", {}).get("consensus_view", {}),
                    "sentiment_confidence": result.get("research_confidence", 0.0),
                    "sentiment_themes": result.get("content_analysis", {}).get("key_themes", []),
                    "contrarian_indicators": result.get("content_analysis", {}).get("contrarian_views", []),
                }
            }

        @mcp.tool(name="research_company_comprehensive") 
        async def research_company_comprehensive(request: CompanyResearchRequest) -> dict:
            """Perform comprehensive company research and fundamental analysis."""
            agent = get_research_agent()
            if request.persona and request.persona in ["conservative", "moderate", "aggressive", "day_trader"]:
                agent.persona = agent.persona.__class__.get(request.persona)
            
            session_id = request.session_id or f"company_{request.symbol}_{datetime.now().timestamp()}"
            
            result = await agent.research_company_comprehensive(
                symbol=request.symbol,
                session_id=session_id,
                include_competitive_analysis=request.include_competitive_analysis,
            )
            
            if "error" in result:
                return {"success": False, "error": result["error"], "symbol": request.symbol}
                
            return {
                "success": True,
                "symbol": request.symbol,
                "company_research": {
                    "executive_summary": result.get("content", "Company research completed"),
                    "research_confidence": result.get("research_confidence", 0.0),
                    "fundamental_insights": result.get("persona_insights", {}),
                    "business_analysis": result.get("content_analysis", {}).get("insights", []),
                }
            }

        @mcp.tool(name="research_search_financial_news")
        async def search_financial_news(
            query: str,
            timeframe: str = "1w",
            max_results: int = 20,
            persona: str = "moderate",
        ) -> dict:
            """Search for recent financial news and analysis on any topic."""
            agent = get_research_agent()
            
            # Use basic research for news search
            result = await agent.research_topic(
                query=f"{query} news",
                session_id=f"news_{datetime.now().timestamp()}",
                research_scope="basic",
                max_sources=max_results,
                timeframe=timeframe,
            )
            
            return {
                "success": True,
                "query": query,
                "news_results": result.get("processed_sources", [])[:max_results],
                "total_found": len(result.get("processed_sources", [])),
                "timeframe": timeframe,
                "persona": persona,
            }
            
        logger.info("Successfully registered 4 research tools directly")
        
    except ImportError as e:
        logger.warning(f"Research module not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register research tools: {e}")
        # Don't raise - allow server to continue without research tools


def register_all_router_tools(mcp: FastMCP) -> None:
    """Register all router tools directly on the main server"""
    register_technical_tools(mcp)
    register_screening_tools(mcp)
    register_portfolio_tools(mcp)
    register_data_tools(mcp)
    register_performance_tools(mcp)
    register_agent_tools(mcp)
    register_research_tools(mcp)  # Add deep research tools
