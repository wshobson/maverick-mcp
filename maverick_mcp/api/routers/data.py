"""
Data fetching router for Maverick-MCP.

This module contains all data retrieval tools including
stock data, news, fundamentals, and caching operations.

Updated to use separated services following Single Responsibility Principle.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import requests
from fastmcp import FastMCP

from maverick_mcp.config.settings import settings
from maverick_mcp.data.models import PriceCache
from maverick_mcp.data.session_management import get_db_session_read_only
from maverick_mcp.domain.stock_analysis import StockAnalysisService
from maverick_mcp.infrastructure.caching import CacheManagementService
from maverick_mcp.infrastructure.data_fetching import StockDataFetchingService
from maverick_mcp.providers.stock_data import (
    StockDataProvider,
)  # Kept for backward compatibility
from maverick_mcp.validation.data import (
    CachedPriceDataRequest,
    ClearCacheRequest,
    FetchStockDataRequest,
    GetChartLinksRequest,
    GetNewsRequest,
    GetStockInfoRequest,
    StockDataBatchRequest,
)

logger = logging.getLogger(__name__)

# Create the data router
data_router: FastMCP = FastMCP("Data_Operations")

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=10)


def fetch_stock_data(request: FetchStockDataRequest) -> dict[str, Any]:
    """
    Fetch historical stock data for a given ticker symbol.

    This is the primary tool for retrieving stock price data. It uses intelligent
    caching to minimize API calls and improve performance.

    Updated to use separated services following Single Responsibility Principle.

    Args:
        ticker: The ticker symbol of the stock (e.g., AAPL, MSFT)
        start_date: Start date for data in YYYY-MM-DD format (default: 1 year ago)
        end_date: End date for data in YYYY-MM-DD format (default: today)

    Returns:
        Dictionary containing the stock data in JSON format with:
        - data: OHLCV price data
        - columns: Column names
        - index: Date index

    Examples:
        >>> fetch_stock_data(FetchStockDataRequest(ticker="AAPL"))
        >>> fetch_stock_data(FetchStockDataRequest(
        ...     ticker="MSFT",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31"
        ... ))
    """
    try:
        # Create services with dependency injection
        data_fetching_service = StockDataFetchingService()

        with get_db_session_read_only() as session:
            cache_service = CacheManagementService(db_session=session)
            stock_analysis_service = StockAnalysisService(
                data_fetching_service=data_fetching_service,
                cache_service=cache_service,
                db_session=session,
            )

            data = stock_analysis_service.get_stock_data(
                request.ticker, request.start_date, request.end_date
            )
            json_data = data.to_json(orient="split", date_format="iso")
            result: dict[str, Any] = json.loads(json_data) if json_data else {}
            result["ticker"] = request.ticker
            result["record_count"] = len(data)
            return result
    except Exception as e:
        logger.error(f"Error fetching stock data for {request.ticker}: {e}")
        return {"error": str(e), "ticker": request.ticker}


def fetch_stock_data_batch(request: StockDataBatchRequest) -> dict[str, Any]:
    """
    Fetch historical data for multiple tickers efficiently.

    This tool fetches data for multiple stocks in a single call,
    which is more efficient than calling fetch_stock_data multiple times.

    Updated to use separated services following Single Responsibility Principle.

    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dictionary with ticker symbols as keys and data/errors as values

    Examples:
        >>> fetch_stock_data_batch(StockDataBatchRequest(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     start_date="2024-01-01"
        ... ))
    """
    results = {}

    # Create services with dependency injection
    data_fetching_service = StockDataFetchingService()

    with get_db_session_read_only() as session:
        cache_service = CacheManagementService(db_session=session)
        stock_analysis_service = StockAnalysisService(
            data_fetching_service=data_fetching_service,
            cache_service=cache_service,
            db_session=session,
        )

        for ticker in request.tickers:
            try:
                data = stock_analysis_service.get_stock_data(
                    ticker, request.start_date, request.end_date
                )
                results[ticker] = {
                    "status": "success",
                    "data": json.loads(
                        data.to_json(orient="split", date_format="iso") or "{}"
                    ),
                    "record_count": len(data),
                }
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                results[ticker] = {"status": "error", "error": str(e)}

    return {
        "results": results,
        "success_count": sum(1 for r in results.values() if r["status"] == "success"),
        "error_count": sum(1 for r in results.values() if r["status"] == "error"),
        "tickers": request.tickers,
    }


def get_stock_info(request: GetStockInfoRequest) -> dict[str, Any]:
    """
    Get detailed fundamental information about a stock.

    This tool retrieves comprehensive stock information including:
    - Company description and sector
    - Market cap and valuation metrics
    - Financial ratios
    - Trading information

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing detailed stock information
    """
    try:
        # Use read-only context manager for automatic session management
        with get_db_session_read_only() as session:
            provider = StockDataProvider(db_session=session)
            info = provider.get_stock_info(request.ticker)

            # Extract key information
            return {
                "ticker": request.ticker,
                "company": {
                    "name": info.get("longName", info.get("shortName")),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "website": info.get("website"),
                    "description": info.get("longBusinessSummary"),
                },
                "market_data": {
                    "current_price": info.get(
                        "currentPrice", info.get("regularMarketPrice")
                    ),
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "float_shares": info.get("floatShares"),
                },
                "valuation": {
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                },
                "financials": {
                    "revenue": info.get("totalRevenue"),
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                },
                "trading": {
                    "avg_volume": info.get("averageVolume"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    "beta": info.get("beta"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                },
            }
    except Exception as e:
        logger.error(f"Error fetching stock info for {request.ticker}: {e}")
        return {"error": str(e), "ticker": request.ticker}


def get_news_sentiment(request: GetNewsRequest) -> dict[str, Any]:
    """
    Retrieve news sentiment analysis for a stock.

    This tool fetches sentiment data from External API,
    providing insights into market sentiment based on recent news.

    Args:
        ticker: The ticker symbol of the stock to analyze

    Returns:
        Dictionary containing news sentiment analysis
    """
    try:
        api_key = settings.external_data.api_key
        base_url = settings.external_data.base_url
        if not api_key:
            logger.info(
                "External sentiment API not configured, providing basic response"
            )
            return {
                "ticker": request.ticker,
                "sentiment": "neutral",
                "message": "External sentiment API not configured - configure EXTERNAL_DATA_API_KEY for enhanced sentiment analysis",
                "status": "fallback_mode",
                "confidence": 0.5,
                "source": "fallback",
            }

        url = f"{base_url}/sentiment/{request.ticker}"
        headers = {"X-API-KEY": api_key}
        logger.info(f"Fetching sentiment for {request.ticker} from {url}")
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 404:
            return {
                "ticker": request.ticker,
                "sentiment": "unavailable",
                "message": f"No sentiment data available for {request.ticker}",
                "status": "not_found",
            }
        elif resp.status_code == 401:
            return {
                "error": "Invalid API key",
                "ticker": request.ticker,
                "sentiment": "unavailable",
                "status": "unauthorized",
            }
        elif resp.status_code == 429:
            return {
                "error": "Rate limit exceeded",
                "ticker": request.ticker,
                "sentiment": "unavailable",
                "status": "rate_limited",
            }

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        return {
            "error": "Request timed out",
            "ticker": request.ticker,
            "sentiment": "unavailable",
            "status": "timeout",
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "Connection error",
            "ticker": request.ticker,
            "sentiment": "unavailable",
            "status": "connection_error",
        }
    except Exception as e:
        logger.error(
            f"Error fetching sentiment from External API for {request.ticker}: {e}"
        )
        return {
            "error": str(e),
            "ticker": request.ticker,
            "sentiment": "unavailable",
            "status": "error",
        }


def get_cached_price_data(request: CachedPriceDataRequest) -> dict[str, Any]:
    """
    Get cached price data directly from the database.

    This tool retrieves data from the local cache without making external API calls.
    Useful for checking what data is available locally.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional, defaults to today)

    Returns:
        Dictionary containing cached price data
    """
    try:
        with get_db_session_read_only() as session:
            df = PriceCache.get_price_data(
                session, request.ticker, request.start_date, request.end_date
            )

            if df.empty:
                return {
                    "status": "success",
                    "ticker": request.ticker,
                    "message": "No cached data found for the specified date range",
                    "data": [],
                }

            # Convert DataFrame to dict format
            data = df.reset_index().to_dict(orient="records")

            return {
                "status": "success",
                "ticker": request.ticker,
                "start_date": request.start_date,
                "end_date": request.end_date or datetime.now(UTC).strftime("%Y-%m-%d"),
                "count": len(data),
                "data": data,
            }
    except Exception as e:
        logger.error(f"Error fetching cached price data for {request.ticker}: {str(e)}")
        return {"error": str(e), "status": "error"}


def get_chart_links(request: GetChartLinksRequest) -> dict[str, Any]:
    """
    Provide links to various financial charting websites.

    This tool generates URLs to popular financial websites where detailed
    stock charts can be viewed, including:
    - TradingView (advanced charting)
    - Finviz (visual screener)
    - Yahoo Finance (comprehensive data)
    - StockCharts (technical analysis)

    Args:
        ticker: The ticker symbol of the stock

    Returns:
        Dictionary containing links to various chart providers
    """
    try:
        ticker = request.ticker
        links = {
            "trading_view": f"https://www.tradingview.com/symbols/{ticker}",
            "finviz": f"https://finviz.com/quote.ashx?t={ticker}",
            "yahoo_finance": f"https://finance.yahoo.com/quote/{ticker}/chart",
            "stock_charts": f"https://stockcharts.com/h-sc/ui?s={ticker}",
            "seeking_alpha": f"https://seekingalpha.com/symbol/{ticker}/charts",
            "marketwatch": f"https://www.marketwatch.com/investing/stock/{ticker}/charts",
        }

        return {
            "ticker": ticker,
            "charts": links,
            "description": "External chart resources for detailed analysis",
        }
    except Exception as e:
        logger.error(f"Error generating chart links for {request.ticker}: {e}")
        return {"error": str(e)}


def clear_cache(request: ClearCacheRequest) -> dict[str, Any]:
    """
    Clear cached data for a specific ticker or all tickers.

    This tool helps manage the local cache by removing stored data,
    forcing fresh data retrieval on the next request.

    Args:
        ticker: Specific ticker to clear (None to clear all)

    Returns:
        Dictionary with cache clearing status
    """
    try:
        from maverick_mcp.data.cache import clear_cache as cache_clear

        if request.ticker:
            pattern = f"stock:{request.ticker}:*"
            count = cache_clear(pattern)
            message = f"Cleared cache for {request.ticker}"
        else:
            count = cache_clear()
            message = "Cleared all cache entries"

        return {"status": "success", "message": message, "entries_cleared": count}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {"error": str(e), "status": "error"}
