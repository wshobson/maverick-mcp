"""
Technical analysis router for MaverickMCP.

This module contains all technical analysis related tools including
indicators, chart patterns, and analysis functions.

DISCLAIMER: All technical analysis tools are for educational purposes only.
Technical indicators are mathematical calculations based on historical data and
do not predict future price movements. Results should not be considered as
investment advice. Always consult qualified financial professionals.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token

from maverick_mcp.core.technical_analysis import (
    analyze_bollinger_bands,
    analyze_macd,
    analyze_rsi,
    analyze_stochastic,
    analyze_trend,
    analyze_volume,
    generate_outlook,
    identify_chart_patterns,
    identify_resistance_levels,
    identify_support_levels,
)
from maverick_mcp.core.visualization import (
    create_plotly_technical_chart,
    plotly_fig_to_base64,
)
from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.utils.logging import PerformanceMonitor, get_logger
from maverick_mcp.utils.mcp_logging import with_logging
from maverick_mcp.utils.stock_helpers import (
    get_stock_dataframe_async,
)

logger = get_logger("maverick_mcp.routers.technical")

# Create the technical analysis router
technical_router: FastMCP = FastMCP("Technical_Analysis")

# Initialize data provider
stock_provider = StockDataProvider()

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=10)


@with_logging("rsi_analysis")
async def get_rsi_analysis(
    ticker: str, period: int = 14, days: int = 365
) -> dict[str, Any]:
    """
    Get RSI analysis for a given ticker.

    Args:
        ticker: Stock ticker symbol
        period: RSI period (default: 14)
        days: Number of days of historical data to analyze (default: 365)

    Returns:
        Dictionary containing RSI analysis
    """
    try:
        # Log analysis parameters
        logger.info(
            "Starting RSI analysis",
            extra={"ticker": ticker, "period": period, "days": days},
        )

        # Fetch stock data with performance monitoring
        with PerformanceMonitor(f"fetch_data_{ticker}"):
            df = await get_stock_dataframe_async(ticker, days)

        # Perform RSI analysis with monitoring
        with PerformanceMonitor(f"rsi_calculation_{ticker}"):
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(executor, analyze_rsi, df)

        # Log successful completion
        logger.info(
            "RSI analysis completed successfully",
            extra={
                "ticker": ticker,
                "rsi_current": analysis.get("current_rsi"),
                "signal": analysis.get("signal"),
            },
        )

        return {"ticker": ticker, "period": period, "analysis": analysis}
    except Exception as e:
        logger.error(
            "Error in RSI analysis",
            exc_info=True,
            extra={"ticker": ticker, "period": period, "error_type": type(e).__name__},
        )
        return {"error": str(e), "status": "error"}


async def get_macd_analysis(
    ticker: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    days: int = 365,
) -> dict[str, Any]:
    """
    Get MACD analysis for a given ticker.

    Args:
        ticker: Stock ticker symbol
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        days: Number of days of historical data to analyze (default: 365)

    Returns:
        Dictionary containing MACD analysis
    """
    try:
        df = await get_stock_dataframe_async(ticker, days)
        analysis = analyze_macd(df)
        return {
            "ticker": ticker,
            "parameters": {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            },
            "analysis": analysis,
        }
    except Exception as e:
        logger.error(f"Error in MACD analysis for {ticker}: {str(e)}")
        return {"error": str(e), "status": "error"}


async def get_support_resistance(ticker: str, days: int = 365) -> dict[str, Any]:
    """
    Get support and resistance levels for a given ticker.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to analyze (default: 365)

    Returns:
        Dictionary containing support and resistance levels
    """
    try:
        df = await get_stock_dataframe_async(ticker, days)
        support = identify_support_levels(df)
        resistance = identify_resistance_levels(df)
        current_price = df["close"].iloc[-1]

        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "support_levels": sorted(support),
            "resistance_levels": sorted(resistance),
        }
    except Exception as e:
        logger.error(f"Error in support/resistance analysis for {ticker}: {str(e)}")
        return {"error": str(e), "status": "error"}


async def get_full_technical_analysis(ticker: str, days: int = 365) -> dict[str, Any]:
    """
    Get comprehensive technical analysis for a given ticker.

    This tool provides a complete technical analysis including:
    - Trend analysis
    - All major indicators (RSI, MACD, Stochastic, Bollinger Bands)
    - Support and resistance levels
    - Volume analysis
    - Chart patterns
    - Overall outlook

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to analyze (default: 365)

    Returns:
        Dictionary containing complete technical analysis
    """
    try:
        # Access authentication context if available (optional for this tool)
        # This demonstrates optional authentication - tool works without auth
        # but provides enhanced features for authenticated users
        has_premium = False
        try:
            access_token = get_access_token()
            if access_token is None:
                raise ValueError("No access token available")

            # Log authenticated user
            logger.info(
                f"Technical analysis requested by authenticated user: {access_token.client_id}",
                extra={"scopes": access_token.scopes},
            )

            # Check for premium features based on scopes
            has_premium = "premium:access" in access_token.scopes
            logger.info(f"Has premium: {has_premium}")
        except Exception:
            # Authentication is optional for this tool
            logger.debug("Technical analysis requested by unauthenticated user")

        df = await get_stock_dataframe_async(ticker, days)

        # Perform all analyses
        trend = analyze_trend(df)
        rsi_analysis = analyze_rsi(df)
        macd_analysis = analyze_macd(df)
        stoch_analysis = analyze_stochastic(df)
        bb_analysis = analyze_bollinger_bands(df)
        volume_analysis = analyze_volume(df)
        patterns = identify_chart_patterns(df)
        support = identify_support_levels(df)
        resistance = identify_resistance_levels(df)
        outlook = generate_outlook(
            df, str(trend), rsi_analysis, macd_analysis, stoch_analysis
        )

        # Get current price and indicators
        current_price = df["close"].iloc[-1]

        # Compile results
        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "trend": trend,
            "outlook": outlook,
            "indicators": {
                "rsi": rsi_analysis,
                "macd": macd_analysis,
                "stochastic": stoch_analysis,
                "bollinger_bands": bb_analysis,
                "volume": volume_analysis,
            },
            "levels": {"support": sorted(support), "resistance": sorted(resistance)},
            "patterns": patterns,
            "last_updated": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error in technical analysis for {ticker}: {str(e)}")
        return {"error": str(e), "status": "error"}


async def get_stock_chart_analysis(ticker: str) -> dict[str, Any]:
    """
    Generate a comprehensive technical analysis chart.

    This tool creates a visual technical analysis including:
    - Price action with candlesticks
    - Moving averages
    - Volume analysis
    - Technical indicators
    - Support and resistance levels

    Args:
        ticker: The ticker symbol of the stock to analyze

    Returns:
        Dictionary containing the chart as properly formatted MCP image content for Claude Desktop display
    """
    try:
        # Use async data fetching
        df = await get_stock_dataframe_async(ticker, 365)

        # Run the chart generation in the executor for performance
        loop = asyncio.get_event_loop()
        chart_content = await loop.run_in_executor(
            executor, _generate_chart_mcp_format, df, ticker
        )
        return chart_content
    except Exception as e:
        logger.error(f"Error generating chart analysis for {ticker}: {e}")
        return {"error": str(e)}


def _generate_chart_mcp_format(df, ticker: str) -> dict[str, Any]:
    """Generate chart in proper MCP content format for Claude Desktop with aggressive size optimization"""
    from maverick_mcp.core.technical_analysis import add_technical_indicators

    df = add_technical_indicators(df)

    # Claude Desktop has a ~100k character limit for responses
    # Base64 images need to be MUCH smaller - aim for ~50k chars max
    chart_configs = [
        {"height": 300, "width": 500, "format": "jpeg"},  # Small primary
        {"height": 250, "width": 400, "format": "jpeg"},  # Smaller fallback
        {"height": 200, "width": 350, "format": "jpeg"},  # Tiny fallback
        {"height": 150, "width": 300, "format": "jpeg"},  # Last resort
    ]

    for config in chart_configs:
        try:
            # Generate chart with current config
            analysis = create_plotly_technical_chart(
                df, ticker, height=config["height"], width=config["width"]
            )

            # Generate base64 data URI
            data_uri = plotly_fig_to_base64(analysis, format=config["format"])

            # Extract base64 data without the data URI prefix
            if data_uri.startswith(f"data:image/{config['format']};base64,"):
                base64_data = data_uri.split(",", 1)[1]
                mime_type = f"image/{config['format']}"
            else:
                # Fallback - assume it's already base64 data
                base64_data = data_uri
                mime_type = f"image/{config['format']}"

            # Very conservative size limit for Claude Desktop
            # Response gets truncated at 100k chars, so aim for 50k max for base64
            max_chars = 50000

            logger.info(
                f"Generated chart for {ticker}: {config['width']}x{config['height']} "
                f"({len(base64_data):,} chars base64)"
            )

            if len(base64_data) <= max_chars:
                # Try multiple formats to work around Claude Desktop bugs
                description = (
                    f"Technical analysis chart for {ticker.upper()} "
                    f"({config['width']}x{config['height']}) showing price action, "
                    f"moving averages, volume, RSI, and MACD indicators."
                )

                return _return_image_with_claude_desktop_workaround(
                    base64_data, mime_type, description, ticker
                )
            else:
                logger.warning(
                    f"Chart for {ticker} too large at {config['width']}x{config['height']} "
                    f"({len(base64_data):,} chars > {max_chars}), trying smaller size..."
                )
                continue

        except Exception as e:
            logger.warning(f"Failed to generate chart with config {config}: {e}")
            continue

    # If all configs failed, return error
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"Unable to generate suitable chart size for {ticker.upper()}. "
                    f"The chart image is too large for Claude Desktop display limits. "
                    f"Please use the text-based technical analysis tool instead: "
                    f"technical_get_full_technical_analysis"
                ),
            }
        ]
    }


def _return_image_with_claude_desktop_workaround(
    base64_data: str, mime_type: str, description: str, ticker: str
) -> dict[str, Any]:
    """
    Return image using multiple formats to work around Claude Desktop bugs.
    Tries alternative MCP format first, fallback to file saving.
    """
    import base64 as b64
    import tempfile
    from pathlib import Path

    # Format 1: Alternative "source" structure (some reports of this working)
    try:
        return {
            "content": [
                {"type": "text", "text": description},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_data,
                    },
                },
            ]
        }
    except Exception as e:
        logger.warning(f"Alternative image format failed: {e}")

    # Format 2: Try original format one more time with different structure
    try:
        return {
            "content": [
                {"type": "text", "text": description},
                {"type": "image", "data": base64_data, "mimeType": mime_type},
            ]
        }
    except Exception as e:
        logger.warning(f"Standard image format failed: {e}")

    # Format 3: File-based fallback (most reliable for Claude Desktop)
    try:
        ext = mime_type.split("/")[-1]  # jpeg, png, etc.

        # Create temp file in a standard location
        temp_dir = Path(tempfile.gettempdir()) / "maverick_mcp_charts"
        temp_dir.mkdir(exist_ok=True)

        chart_file = (
            temp_dir
            / f"{ticker.lower()}_chart_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.{ext}"
        )

        # Decode and save base64 to file
        image_data = b64.b64decode(base64_data)
        chart_file.write_bytes(image_data)

        logger.info(f"Saved chart to file: {chart_file}")

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{description}\n\n"
                        f"📁 **Chart saved to file**: `{chart_file}`\n\n"
                        f"**To view this image:**\n"
                        f"1. Use the filesystem MCP server if configured, or\n"
                        f"2. Ask me to open the file location, or\n"
                        f"3. Navigate to the file manually\n\n"
                        f"*Note: Claude Desktop has a known issue with embedded images. "
                        f"File-based display is the current workaround.*"
                    ),
                }
            ]
        }
    except Exception as e:
        logger.error(f"File fallback also failed: {e}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Unable to display chart for {ticker.upper()} due to "
                        f"Claude Desktop image rendering limitations. "
                        f"Please use the text-based technical analysis instead: "
                        f"`technical_get_full_technical_analysis`"
                    ),
                }
            ]
        }
