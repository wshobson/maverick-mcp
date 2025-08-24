"""
Stock screening router for Maverick-MCP.

This module contains all stock screening related tools including
Maverick, supply/demand breakouts, and other screening strategies.
"""

import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Create the screening router
screening_router: FastMCP = FastMCP("Stock_Screening")


def get_maverick_stocks(limit: int = 20) -> dict[str, Any]:
    """
    Get top Maverick stocks from the screening results.

    DISCLAIMER: Stock screening results are for educational and research purposes only.
    This is not investment advice. Past performance does not guarantee future results.
    Always conduct thorough research and consult financial professionals before investing.

    The Maverick screening strategy identifies stocks with:
    - High momentum strength
    - Technical patterns (Cup & Handle, consolidation, etc.)
    - Momentum characteristics
    - Strong combined scores

    Args:
        limit: Maximum number of stocks to return (default: 20)

    Returns:
        Dictionary containing Maverick stock screening results
    """
    try:
        from maverick_mcp.data.models import MaverickStocks, SessionLocal

        with SessionLocal() as session:
            stocks = MaverickStocks.get_top_stocks(session, limit=limit)

            return {
                "status": "success",
                "count": len(stocks),
                "stocks": [stock.to_dict() for stock in stocks],
                "screening_type": "maverick_bullish",
                "description": "High momentum stocks with bullish technical setups",
            }
    except Exception as e:
        logger.error(f"Error fetching Maverick stocks: {str(e)}")
        return {"error": str(e), "status": "error"}


def get_maverick_bear_stocks(limit: int = 20) -> dict[str, Any]:
    """
    Get top Maverick Bear stocks from the screening results.

    DISCLAIMER: Bearish screening results are for educational purposes only.
    This is not advice to sell short or make bearish trades. Short selling involves
    unlimited risk potential. Always consult financial professionals before trading.

    The Maverick Bear screening identifies stocks with:
    - Weak momentum strength
    - Bearish technical patterns
    - Distribution characteristics
    - High bear scores

    Args:
        limit: Maximum number of stocks to return (default: 20)

    Returns:
        Dictionary containing Maverick Bear stock screening results
    """
    try:
        from maverick_mcp.data.models import MaverickBearStocks, SessionLocal

        with SessionLocal() as session:
            stocks = MaverickBearStocks.get_top_stocks(session, limit=limit)

            return {
                "status": "success",
                "count": len(stocks),
                "stocks": [stock.to_dict() for stock in stocks],
                "screening_type": "maverick_bearish",
                "description": "Weak stocks with bearish technical setups",
            }
    except Exception as e:
        logger.error(f"Error fetching Maverick Bear stocks: {str(e)}")
        return {"error": str(e), "status": "error"}


def get_supply_demand_breakouts(
    limit: int = 20, filter_moving_averages: bool = False
) -> dict[str, Any]:
    """
    Get stocks showing supply/demand breakout patterns from accumulation.

    This screening identifies stocks in the demand expansion phase with:
    - Price above all major moving averages (demand zone)
    - Moving averages in proper alignment indicating accumulation (50 > 150 > 200)
    - Strong momentum strength showing institutional interest
    - Market structure indicating supply absorption and demand dominance

    Args:
        limit: Maximum number of stocks to return (default: 20)
        filter_moving_averages: If True, only return stocks above all moving averages

    Returns:
        Dictionary containing supply/demand breakout screening results
    """
    try:
        from maverick_mcp.data.models import SessionLocal, SupplyDemandBreakoutStocks

        with SessionLocal() as session:
            if filter_moving_averages:
                stocks = SupplyDemandBreakoutStocks.get_stocks_above_moving_averages(
                    session
                )[:limit]
            else:
                stocks = SupplyDemandBreakoutStocks.get_top_stocks(session, limit=limit)

            return {
                "status": "success",
                "count": len(stocks),
                "stocks": [stock.to_dict() for stock in stocks],
                "screening_type": "supply_demand_breakout",
                "description": "Stocks breaking out from accumulation with strong demand dynamics",
            }
    except Exception as e:
        logger.error(f"Error fetching supply/demand breakout stocks: {str(e)}")
        return {"error": str(e), "status": "error"}


def get_all_screening_recommendations() -> dict[str, Any]:
    """
    Get comprehensive screening results from all strategies.

    This tool returns the top stocks from each screening strategy:
    - Maverick Bullish: High momentum growth stocks
    - Maverick Bearish: Weak stocks for short opportunities
    - Supply/Demand Breakouts: Stocks breaking out from accumulation phases

    Returns:
        Dictionary containing all screening results organized by strategy
    """
    try:
        from maverick_mcp.providers.stock_data import StockDataProvider

        provider = StockDataProvider()
        return provider.get_all_screening_recommendations()
    except Exception as e:
        logger.error(f"Error getting all screening recommendations: {e}")
        return {
            "error": str(e),
            "status": "error",
            "maverick_stocks": [],
            "maverick_bear_stocks": [],
            "supply_demand_breakouts": [],
        }


def get_screening_by_criteria(
    min_momentum_score: float | str | None = None,
    min_volume: int | str | None = None,
    max_price: float | str | None = None,
    sector: str | None = None,
    limit: int | str = 20,
) -> dict[str, Any]:
    """
    Get stocks filtered by specific screening criteria.

    This tool allows custom filtering across all screening results based on:
    - Momentum score rating
    - Volume requirements
    - Price constraints
    - Sector preferences

    Args:
        min_momentum_score: Minimum momentum score rating (0-100)
        min_volume: Minimum average daily volume
        max_price: Maximum stock price
        sector: Specific sector to filter (e.g., "Technology")
        limit: Maximum number of results

    Returns:
        Dictionary containing filtered screening results
    """
    try:
        from maverick_mcp.data.models import MaverickStocks, SessionLocal

        # Convert string inputs to appropriate numeric types
        if min_momentum_score is not None:
            min_momentum_score = float(min_momentum_score)
        if min_volume is not None:
            min_volume = int(min_volume)
        if max_price is not None:
            max_price = float(max_price)
        if isinstance(limit, str):
            limit = int(limit)

        with SessionLocal() as session:
            query = session.query(MaverickStocks)

            if min_momentum_score:
                query = query.filter(
                    MaverickStocks.momentum_score >= min_momentum_score
                )

            if min_volume:
                query = query.filter(MaverickStocks.avg_vol_30d >= min_volume)

            if max_price:
                query = query.filter(MaverickStocks.close_price <= max_price)

            # Note: Sector filtering would require joining with Stock table
            # This is a simplified version

            stocks = (
                query.order_by(MaverickStocks.combined_score.desc()).limit(limit).all()
            )

            return {
                "status": "success",
                "count": len(stocks),
                "stocks": [stock.to_dict() for stock in stocks],
                "criteria": {
                    "min_momentum_score": min_momentum_score,
                    "min_volume": min_volume,
                    "max_price": max_price,
                    "sector": sector,
                },
            }
    except Exception as e:
        logger.error(f"Error in custom screening: {str(e)}")
        return {"error": str(e), "status": "error"}
