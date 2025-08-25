"""
Tool registry to register router tools directly on main server.
This avoids Claude Desktop's issue with mounted router tool names.
"""

import logging
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
        from maverick_mcp.api.routers.research import research_router

        # TODO: Fix tool extraction from research router
        # The _tools attribute doesn't exist on FastMCP objects
        # For now, skip research tool registration to allow server startup
        logger.warning("Research tools registration temporarily disabled")
        pass
    except ImportError:
        # Research module not available - skip
        pass


def register_all_router_tools(mcp: FastMCP) -> None:
    """Register all router tools directly on the main server"""
    register_technical_tools(mcp)
    register_screening_tools(mcp)
    register_portfolio_tools(mcp)
    register_data_tools(mcp)
    register_performance_tools(mcp)
    register_agent_tools(mcp)
    register_research_tools(mcp)  # Add deep research tools
