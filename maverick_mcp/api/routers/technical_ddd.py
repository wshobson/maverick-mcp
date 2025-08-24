"""
Technical analysis router with Domain-Driven Design.

This is the refactored version that delegates all business logic
to the domain and application layers.
"""

from typing import Any

from fastmcp import FastMCP

from maverick_mcp.api.dependencies.technical_analysis import (
    get_technical_analysis_query,
)
from maverick_mcp.utils.logging import get_logger

logger = get_logger("maverick_mcp.routers.technical_ddd")

# Create the technical analysis router
technical_ddd_router: FastMCP = FastMCP("Technical_Analysis_DDD")


async def get_technical_analysis_ddd(
    ticker: str,
    days: int = 365,
) -> dict[str, Any]:
    """
    Get comprehensive technical analysis for a stock using Domain-Driven Design.

    This is a thin controller that delegates all business logic to the
    application and domain layers, following DDD principles.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data (default: 365)

    Returns:
        Complete technical analysis with all indicators
    """
    try:
        # Get the query handler through dependency injection
        query = get_technical_analysis_query()

        # Execute the query - all business logic is in the domain/application layers
        analysis_dto = await query.execute(symbol=ticker, days=days)

        # Convert DTO to dict for MCP response
        return {
            "ticker": ticker,
            "analysis": analysis_dto.model_dump(),
            "status": "success",
        }

    except ValueError as e:
        logger.warning(f"Invalid input for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "status": "invalid_input",
        }
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
        return {
            "ticker": ticker,
            "error": "Technical analysis failed",
            "status": "error",
        }


async def get_rsi_analysis_ddd(
    ticker: str,
    period: int = 14,
    days: int = 365,
) -> dict[str, Any]:
    """
    Get RSI analysis using Domain-Driven Design approach.

    Args:
        ticker: Stock ticker symbol
        period: RSI period (default: 14)
        days: Number of days of historical data (default: 365)

    Returns:
        RSI analysis results
    """
    try:
        # Get query handler
        query = get_technical_analysis_query()

        # Execute query for RSI only
        analysis_dto = await query.execute(
            symbol=ticker,
            days=days,
            indicators=["rsi"],
            rsi_period=period,
        )

        if not analysis_dto.rsi:
            return {
                "ticker": ticker,
                "error": "RSI calculation failed",
                "status": "error",
            }

        return {
            "ticker": ticker,
            "period": period,
            "analysis": analysis_dto.rsi.model_dump(),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in RSI analysis for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "status": "error",
        }


async def get_support_resistance_ddd(
    ticker: str,
    days: int = 365,
) -> dict[str, Any]:
    """
    Get support and resistance levels using DDD approach.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data (default: 365)

    Returns:
        Support and resistance levels
    """
    try:
        # Get query handler
        query = get_technical_analysis_query()

        # Execute query
        analysis_dto = await query.execute(
            symbol=ticker,
            days=days,
            indicators=[],  # No indicators needed, just levels
        )

        return {
            "ticker": ticker,
            "current_price": analysis_dto.current_price,
            "support_levels": [
                {
                    "price": level.price,
                    "strength": level.strength,
                    "distance": level.distance_from_current,
                }
                for level in analysis_dto.support_levels
            ],
            "resistance_levels": [
                {
                    "price": level.price,
                    "strength": level.strength,
                    "distance": level.distance_from_current,
                }
                for level in analysis_dto.resistance_levels
            ],
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error in support/resistance analysis for {ticker}: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e),
            "status": "error",
        }
