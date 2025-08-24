"""
Enhanced data fetching router with dependency injection support.

This module demonstrates how to integrate the new provider interfaces
with FastMCP routers while maintaining backward compatibility.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP

from maverick_mcp.providers.dependencies import (
    get_cache_manager,
    get_configuration,
    get_stock_data_fetcher,
)
from maverick_mcp.providers.interfaces.cache import ICacheManager
from maverick_mcp.providers.interfaces.config import IConfigurationProvider
from maverick_mcp.providers.interfaces.stock_data import IStockDataFetcher
from maverick_mcp.validation.data import (
    FetchStockDataRequest,
    GetNewsRequest,
    GetStockInfoRequest,
    StockDataBatchRequest,
)

logger = logging.getLogger(__name__)

# Create the enhanced data router
data_enhanced_router: FastMCP = FastMCP("Enhanced_Data_Operations")


# Example of new interface-based implementation
@data_enhanced_router.tool()
async def fetch_stock_data_enhanced(
    request: FetchStockDataRequest,
    stock_fetcher: IStockDataFetcher | None = None,
    cache_manager: ICacheManager | None = None,
    config: IConfigurationProvider | None = None,
) -> dict[str, Any]:
    """
    Fetch historical stock data using the new interface-based architecture.

    This function demonstrates how to use dependency injection with the new
    provider interfaces while maintaining the same external API.

    Args:
        request: Stock data request parameters
        stock_fetcher: Optional stock data fetcher (injected if not provided)
        cache_manager: Optional cache manager (injected if not provided)
        config: Optional configuration provider (injected if not provided)

    Returns:
        Dictionary containing the stock data in JSON format
    """
    try:
        # Use dependency injection with fallback to global providers
        fetcher = stock_fetcher or get_stock_data_fetcher()
        cache = cache_manager or get_cache_manager()
        cfg = config or get_configuration()

        logger.debug(
            f"Fetching stock data for {request.ticker} using enhanced interface"
        )

        # Check cache first if enabled
        cache_key = (
            f"stock_data:{request.ticker}:{request.start_date}:{request.end_date}"
        )
        cached_result = None

        if cfg.is_cache_enabled():
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for {request.ticker}")
                return cached_result

        # Fetch data using the interface
        data = await fetcher.get_stock_data(
            symbol=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            use_cache=True,  # The fetcher will handle its own caching
        )

        # Convert to JSON format
        json_data = data.to_json(orient="split", date_format="iso")
        result: dict[str, Any] = json.loads(json_data) if json_data else {}
        result["ticker"] = request.ticker
        result["record_count"] = len(data)
        result["source"] = "enhanced_interface"
        result["timestamp"] = datetime.now(UTC).isoformat()

        # Cache the result if caching is enabled
        if cfg.is_cache_enabled():
            cache_ttl = cfg.get_cache_ttl()
            await cache.set(cache_key, result, ttl=cache_ttl)
            logger.debug(f"Cached result for {request.ticker} (TTL: {cache_ttl}s)")

        return result

    except Exception as e:
        logger.error(f"Error fetching stock data for {request.ticker}: {e}")
        return {
            "error": str(e),
            "ticker": request.ticker,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }


@data_enhanced_router.tool()
async def fetch_stock_data_batch_enhanced(
    request: StockDataBatchRequest,
    stock_fetcher: IStockDataFetcher | None = None,
) -> dict[str, Any]:
    """
    Fetch historical data for multiple tickers using the enhanced interface.

    Args:
        request: Batch stock data request parameters
        stock_fetcher: Optional stock data fetcher (injected if not provided)

    Returns:
        Dictionary with ticker symbols as keys and data/errors as values
    """
    fetcher = stock_fetcher or get_stock_data_fetcher()
    results = {}

    logger.debug(f"Fetching batch stock data for {len(request.tickers)} tickers")

    # Process each ticker
    for ticker in request.tickers:
        try:
            data = await fetcher.get_stock_data(
                symbol=ticker,
                start_date=request.start_date,
                end_date=request.end_date,
                use_cache=True,
            )

            json_data = data.to_json(orient="split", date_format="iso")
            ticker_result: dict[str, Any] = json.loads(json_data) if json_data else {}
            ticker_result["ticker"] = ticker
            ticker_result["record_count"] = len(data)

            results[ticker] = ticker_result

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            results[ticker] = {"error": str(e), "ticker": ticker}

    return {
        "results": results,
        "total_tickers": len(request.tickers),
        "successful": len([r for r in results.values() if "error" not in r]),
        "failed": len([r for r in results.values() if "error" in r]),
        "source": "enhanced_interface",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@data_enhanced_router.tool()
async def get_stock_info_enhanced(
    request: GetStockInfoRequest,
    stock_fetcher: IStockDataFetcher | None = None,
) -> dict[str, Any]:
    """
    Get detailed stock information using the enhanced interface.

    Args:
        request: Stock info request parameters
        stock_fetcher: Optional stock data fetcher (injected if not provided)

    Returns:
        Dictionary with detailed stock information
    """
    try:
        fetcher = stock_fetcher or get_stock_data_fetcher()

        logger.debug(f"Fetching stock info for {request.ticker}")

        info = await fetcher.get_stock_info(request.ticker)

        return {
            "ticker": request.ticker,
            "info": info,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching stock info for {request.ticker}: {e}")
        return {
            "error": str(e),
            "ticker": request.ticker,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }


@data_enhanced_router.tool()
async def get_realtime_data_enhanced(
    ticker: str,
    stock_fetcher: IStockDataFetcher | None = None,
) -> dict[str, Any]:
    """
    Get real-time stock data using the enhanced interface.

    Args:
        ticker: Stock ticker symbol
        stock_fetcher: Optional stock data fetcher (injected if not provided)

    Returns:
        Dictionary with real-time stock data
    """
    try:
        fetcher = stock_fetcher or get_stock_data_fetcher()

        logger.debug(f"Fetching real-time data for {ticker}")

        data = await fetcher.get_realtime_data(ticker)

        if data is None:
            return {
                "error": "Real-time data not available",
                "ticker": ticker,
                "source": "enhanced_interface",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        return {
            "ticker": ticker,
            "data": data,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching real-time data for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }


@data_enhanced_router.tool()
async def get_news_enhanced(
    request: GetNewsRequest,
    stock_fetcher: IStockDataFetcher | None = None,
) -> dict[str, Any]:
    """
    Get news for a stock using the enhanced interface.

    Args:
        request: News request parameters
        stock_fetcher: Optional stock data fetcher (injected if not provided)

    Returns:
        Dictionary with news data
    """
    try:
        fetcher = stock_fetcher or get_stock_data_fetcher()

        logger.debug(f"Fetching news for {request.ticker}")

        news_df = await fetcher.get_news(request.ticker, request.limit)

        # Convert DataFrame to JSON
        if not news_df.empty:
            news_data = news_df.to_dict(orient="records")
        else:
            news_data = []

        return {
            "ticker": request.ticker,
            "news": news_data,
            "count": len(news_data),
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching news for {request.ticker}: {e}")
        return {
            "error": str(e),
            "ticker": request.ticker,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }


@data_enhanced_router.tool()
async def check_market_status_enhanced(
    stock_fetcher: IStockDataFetcher | None = None,
) -> dict[str, Any]:
    """
    Check if the market is currently open using the enhanced interface.

    Args:
        stock_fetcher: Optional stock data fetcher (injected if not provided)

    Returns:
        Dictionary with market status
    """
    try:
        fetcher = stock_fetcher or get_stock_data_fetcher()

        is_open = await fetcher.is_market_open()

        return {
            "market_open": is_open,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return {
            "error": str(e),
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }


@data_enhanced_router.tool()
async def clear_cache_enhanced(
    pattern: str | None = None,
    cache_manager: ICacheManager | None = None,
) -> dict[str, Any]:
    """
    Clear cache entries using the enhanced cache interface.

    Args:
        pattern: Optional pattern to match cache keys (e.g., "stock:*")
        cache_manager: Optional cache manager (injected if not provided)

    Returns:
        Dictionary with cache clearing results
    """
    try:
        cache = cache_manager or get_cache_manager()

        cleared_count = await cache.clear(pattern)

        return {
            "cleared_count": cleared_count,
            "pattern": pattern,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {
            "error": str(e),
            "pattern": pattern,
            "source": "enhanced_interface",
            "timestamp": datetime.now(UTC).isoformat(),
        }
