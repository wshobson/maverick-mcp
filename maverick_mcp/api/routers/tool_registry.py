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
        request = TechnicalAnalysisRequest(ticker=ticker, days=days)
        return await get_full_technical_analysis_enhanced(request)

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
        return await get_stock_chart_analysis_enhanced(ticker)


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
        return await get_news_sentiment_enhanced(ticker, timeframe, limit)

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
            return await comprehensive_research(
                query=query,
                persona=persona or "moderate",
                research_scope=research_scope or "standard",
                max_sources=min(
                    max_sources or 25, 25
                ),  # Increased cap due to adaptive timeout
                timeframe=timeframe or "1m",
            )

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
            return await analyze_market_sentiment(
                topic=topic,
                timeframe=timeframe or "1w",
                persona=persona or "moderate",
            )

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
            return await company_comprehensive_research(
                symbol=symbol,
                include_competitive_analysis=include_competitive_analysis or False,
                persona=persona or "moderate",
            )

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


def register_backtesting_tools(mcp: FastMCP) -> None:
    """Register VectorBT backtesting tools directly on main server"""
    try:
        from maverick_mcp.api.routers.backtesting import setup_backtesting_tools

        setup_backtesting_tools(mcp)
        logger.info("✓ Backtesting tools registered successfully")
    except ImportError:
        logger.warning(
            "Backtesting module not available - VectorBT may not be installed"
        )
    except Exception as e:
        logger.error(f"✗ Failed to register backtesting tools: {e}")


def register_finnhub_tools(mcp: FastMCP) -> None:
    """Register Finnhub data tools on main server."""
    from typing import Any

    from maverick_mcp.providers.finnhub_provider import get_finnhub_provider

    provider = get_finnhub_provider()

    if not provider.is_configured:
        logger.warning(
            "Finnhub API key not configured (set FINNHUB_API_KEY). "
            "Finnhub tools will not be registered."
        )
        return

    @mcp.tool(name="data_finnhub_quote")
    def finnhub_get_quote(ticker: str) -> dict[str, Any]:
        """
        Get real-time stock quote from Finnhub.

        Provides current price, change, open, high, low, and previous close
        from Finnhub's real-time feed. Useful as an alternative to yfinance quotes.

        Args:
            ticker: Stock ticker symbol (e.g., AAPL, MSFT)

        Returns:
            Dictionary with real-time quote data
        """
        return provider.get_quote(ticker)

    @mcp.tool(name="data_finnhub_company_profile")
    def finnhub_get_company_profile(ticker: str) -> dict[str, Any]:
        """
        Get company profile from Finnhub.

        Returns company name, industry, market cap, IPO date, logo, and more.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company profile data
        """
        return provider.get_company_profile(ticker)

    @mcp.tool(name="data_finnhub_earnings_calendar")
    def finnhub_get_earnings_calendar(
        days_ahead: int = 7,
    ) -> dict[str, Any]:
        """
        Get upcoming earnings calendar from Finnhub.

        Args:
            days_ahead: Number of days to look ahead (default: 7)

        Returns:
            Dictionary with upcoming earnings events
        """
        from datetime import timedelta

        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        earnings = provider.get_earnings_calendar(from_date, to_date)
        return {
            "status": "success",
            "count": len(earnings),
            "earnings": earnings,
            "from_date": from_date,
            "to_date": to_date,
            "source": "finnhub",
        }

    @mcp.tool(name="data_finnhub_financials")
    def finnhub_get_basic_financials(ticker: str) -> dict[str, Any]:
        """
        Get basic financial metrics from Finnhub.

        Returns PE ratio, PB ratio, beta, market cap, 52-week range,
        ROE, ROA, revenue growth, and EPS growth.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with financial metrics
        """
        return provider.get_basic_financials(ticker)


def register_mcp_prompts_and_resources(mcp: FastMCP) -> None:
    """Register MCP prompts and resources for better client introspection"""
    try:
        from maverick_mcp.api.routers.mcp_prompts import register_mcp_prompts

        register_mcp_prompts(mcp)
        logger.info("✓ MCP prompts registered successfully")
    except ImportError:
        logger.warning("MCP prompts module not available")
    except Exception as e:
        logger.error(f"✗ Failed to register MCP prompts: {e}")

    # Register introspection tools
    try:
        from maverick_mcp.api.routers.introspection import register_introspection_tools

        register_introspection_tools(mcp)
        logger.info("✓ Introspection tools registered successfully")
    except ImportError:
        logger.warning("Introspection module not available")
    except Exception as e:
        logger.error(f"✗ Failed to register introspection tools: {e}")


def register_all_router_tools(mcp: FastMCP) -> None:
    """Register all router tools directly on the main server"""
    logger.info("Starting tool registration process...")

    try:
        register_technical_tools(mcp)
        logger.info("✓ Technical tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register technical tools: {e}")

    try:
        register_screening_tools(mcp)
        logger.info("✓ Screening tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register screening tools: {e}")

    try:
        register_portfolio_tools(mcp)
        logger.info("✓ Portfolio tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register portfolio tools: {e}")

    try:
        register_data_tools(mcp)
        logger.info("✓ Data tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register data tools: {e}")

    try:
        register_performance_tools(mcp)
        logger.info("✓ Performance tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register performance tools: {e}")

    try:
        register_agent_tools(mcp)
        logger.info("✓ Agent tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register agent tools: {e}")

    try:
        # Import and register research tools on the main MCP instance
        from maverick_mcp.api.routers.research import create_research_router

        # Pass the main MCP instance to register tools directly on it
        create_research_router(mcp)
        logger.info("✓ Research tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register research tools: {e}")

    try:
        # Import and register health monitoring tools
        from maverick_mcp.api.routers.health_tools import register_health_tools

        register_health_tools(mcp)
        logger.info("✓ Health monitoring tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register health monitoring tools: {e}")

    try:
        register_finnhub_tools(mcp)
        logger.info("✓ Finnhub tools registered successfully")
    except Exception as e:
        logger.error(f"✗ Failed to register Finnhub tools: {e}")

    # Register backtesting tools
    register_backtesting_tools(mcp)

    # Register MCP prompts and resources for introspection
    register_mcp_prompts_and_resources(mcp)

    logger.info("Tool registration process completed")
    logger.info("📋 All tools registered:")
    logger.info("   • Technical analysis tools")
    logger.info("   • Stock screening tools")
    logger.info("   • Portfolio analysis tools")
    logger.info("   • Data retrieval tools")
    logger.info("   • Performance monitoring tools")
    logger.info("   • Agent orchestration tools")
    logger.info("   • Research and analysis tools")
    logger.info("   • Health monitoring tools")
    logger.info("   • Backtesting system tools")
    logger.info("   • MCP prompts for introspection")
    logger.info("   • Introspection and discovery tools")
