"""
Data fetching router for Maverick-MCP.

This module contains all data retrieval tools including
stock data, news, fundamentals, and caching operations.

Updated to use separated services following Single Responsibility Principle.
"""

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests
import requests.exceptions
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
from maverick_mcp.utils.error_handling import safe_error_message
from maverick_mcp.validation.base import TickerValidator

logger = logging.getLogger(__name__)

# Create the data router
data_router: FastMCP = FastMCP("Data_Operations")

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=10)
atexit.register(executor.shutdown, wait=False)


def _dataframe_to_split_dict(df: pd.DataFrame) -> dict[str, Any]:
    """Convert DataFrame to split-orient dict with ISO date strings.

    Avoids the serialize-then-deserialize round-trip of
    ``df.to_json() → json.loads()``.
    """
    if df.empty:
        return {}
    result: dict[str, Any] = df.to_dict(orient="split")
    # Convert DatetimeIndex to ISO strings (matches to_json date_format="iso")
    result["index"] = [
        idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        for idx in result["index"]
    ]
    return result


def fetch_stock_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
) -> dict[str, Any]:
    """
    Fetch historical stock data for a given ticker symbol.

    This is the primary tool for retrieving stock price data. It uses intelligent
    caching to minimize API calls and improve performance.

    Updated to use separated services following Single Responsibility Principle.

    Args:
        ticker: The ticker symbol of the stock (e.g., AAPL, MSFT)
        start_date: Start date for data in YYYY-MM-DD format (default: 1 year ago)
        end_date: End date for data in YYYY-MM-DD format (default: today)
        interval: Data interval - "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"
                  (default: "1d"). Intraday intervals fetch fresh from yfinance.

    Returns:
        Dictionary containing the stock data in JSON format with:
        - data: OHLCV price data
        - columns: Column names
        - index: Date index

    Examples:
        >>> fetch_stock_data(ticker="AAPL")
        >>> fetch_stock_data(
        ...     ticker="MSFT",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31"
        ... )
        >>> fetch_stock_data(ticker="AAPL", interval="5m")
    """
    try:
        ticker = TickerValidator.validate_ticker(ticker)
    except ValueError as e:
        return {"error": str(e), "ticker": ticker}

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
                ticker, start_date, end_date, interval=interval
            )
            # Normalize timestamps to UTC date-only for consistency
            if hasattr(data.index, "tz") and data.index.tz is not None:
                data.index = data.index.tz_convert("UTC").normalize().tz_localize(None)
            result: dict[str, Any] = _dataframe_to_split_dict(data)
            result["ticker"] = ticker
            result["interval"] = interval
            result["record_count"] = len(data)
            return result
    except Exception as e:
        return {
            "error": safe_error_message(e, context=f"fetching stock data for {ticker}"),
            "ticker": ticker,
        }


def fetch_stock_data_batch(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
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
        >>> fetch_stock_data_batch(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     start_date="2024-01-01"
        ... )
    """
    MAX_BATCH_SIZE = 50
    if len(tickers) > MAX_BATCH_SIZE:
        return {
            "error": f"Batch size {len(tickers)} exceeds maximum of {MAX_BATCH_SIZE}",
            "tickers": tickers[:5],
        }

    try:
        tickers = TickerValidator.validate_ticker_list(tickers)
    except ValueError as e:
        return {"error": str(e), "tickers": tickers}

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

        for ticker in tickers:
            try:
                data = stock_analysis_service.get_stock_data(
                    ticker, start_date, end_date
                )
                # Normalize timestamps to UTC date-only for cross-ticker consistency
                if hasattr(data.index, "tz") and data.index.tz is not None:
                    data.index = data.index.tz_convert("UTC").normalize().tz_localize(None)
                results[ticker] = {
                    "status": "success",
                    "data": _dataframe_to_split_dict(data),
                    "record_count": len(data),
                }
            except Exception as e:
                results[ticker] = {
                    "status": "error",
                    "error": safe_error_message(
                        e, context=f"fetching batch data for {ticker}"
                    ),
                }

    return {
        "results": results,
        "success_count": sum(1 for r in results.values() if r["status"] == "success"),
        "error_count": sum(1 for r in results.values() if r["status"] == "error"),
        "tickers": tickers,
    }


def get_stock_info(ticker: str) -> dict[str, Any]:
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
        ticker = TickerValidator.validate_ticker(ticker)
    except ValueError as e:
        return {"error": str(e), "ticker": ticker}

    try:
        # Use read-only context manager for automatic session management
        with get_db_session_read_only() as session:
            provider = StockDataProvider(db_session=session)
            info = provider.get_stock_info(ticker)

            # Extract key information
            return {
                "ticker": ticker,
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
                "earnings": {
                    "trailing_eps": info.get("trailingEps"),
                    "forward_eps": info.get("forwardEps"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue_growth": info.get("revenueGrowth"),
                },
                "balance_sheet": {
                    "total_debt": info.get("totalDebt"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "free_cashflow": info.get("freeCashflow"),
                },
                "dividends": {
                    "dividend_yield": info.get("dividendYield"),
                    "payout_ratio": info.get("payoutRatio"),
                },
            }
    except Exception as e:
        return {
            "error": safe_error_message(e, context=f"fetching stock info for {ticker}"),
            "ticker": ticker,
        }


def get_fundamental_analysis(ticker: str) -> dict[str, Any]:
    """
    Get comprehensive fundamental analysis for a stock.

    Provides earnings analysis, valuation assessment, financial health,
    and a composite fundamental score with letter grade.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing fundamental analysis with scores and grade
    """
    try:
        from maverick_mcp.core.fundamental_analysis import (
            compute_fundamental_score,
            get_earnings_analysis,
            get_financial_health,
            get_valuation_assessment,
        )

        with get_db_session_read_only() as session:
            provider = StockDataProvider(db_session=session)
            info = provider.get_stock_info(ticker)

            scores = compute_fundamental_score(info)
            earnings = get_earnings_analysis(info)
            valuation = get_valuation_assessment(info)
            health = get_financial_health(info)

            return {
                "ticker": ticker,
                "company_name": info.get("longName", info.get("shortName")),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "fundamental_score": scores["fundamental_score"],
                "grade": scores["grade"],
                "scores": scores,
                "earnings": earnings,
                "valuation": valuation,
                "financial_health": health,
            }
    except Exception as e:
        return {
            "error": safe_error_message(
                e, context=f"computing fundamental analysis for {ticker}"
            ),
            "ticker": ticker,
        }


def get_news_sentiment(
    ticker: str,
    timeframe: str = "7d",
    limit: int = 10,
) -> dict[str, Any]:
    """
    Retrieve news sentiment analysis for a stock.

    This tool fetches sentiment data from External API,
    providing insights into market sentiment based on recent news.

    Args:
        ticker: The ticker symbol of the stock to analyze
        timeframe: Time frame for news (1d, 7d, 30d, etc.)
        limit: Maximum number of news articles to analyze

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
                "ticker": ticker,
                "sentiment": "neutral",
                "message": "External sentiment API not configured - configure EXTERNAL_DATA_API_KEY for enhanced sentiment analysis",
                "status": "fallback_mode",
                "confidence": 0.5,
                "source": "fallback",
            }

        url = f"{base_url}/sentiment/{ticker}"
        headers = {"X-API-KEY": api_key}
        logger.info("Fetching sentiment for %s from %s", ticker, url)
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 404:
            return {
                "ticker": ticker,
                "sentiment": "unavailable",
                "message": f"No sentiment data available for {ticker}",
                "status": "not_found",
            }
        elif resp.status_code == 401:
            return {
                "error": "Invalid API key",
                "ticker": ticker,
                "sentiment": "unavailable",
                "status": "unauthorized",
            }
        elif resp.status_code == 429:
            return {
                "error": "Rate limit exceeded",
                "ticker": ticker,
                "sentiment": "unavailable",
                "status": "rate_limited",
            }

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        return {
            "error": "Request timed out",
            "ticker": ticker,
            "sentiment": "unavailable",
            "status": "timeout",
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "Connection error",
            "ticker": ticker,
            "sentiment": "unavailable",
            "status": "connection_error",
        }
    except Exception as e:
        return {
            "error": safe_error_message(e, context=f"fetching sentiment for {ticker}"),
            "ticker": ticker,
            "sentiment": "unavailable",
            "status": "error",
        }


def get_cached_price_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> dict[str, Any]:
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
            df = PriceCache.get_price_data(session, ticker, start_date, end_date)

            if df.empty:
                return {
                    "status": "success",
                    "ticker": ticker,
                    "message": "No cached data found for the specified date range",
                    "data": [],
                }

            # Convert DataFrame to dict format
            data = df.reset_index().to_dict(orient="records")

            return {
                "status": "success",
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date or datetime.now(UTC).strftime("%Y-%m-%d"),
                "count": len(data),
                "data": data,
            }
    except Exception as e:
        return {
            "error": safe_error_message(
                e, context=f"fetching cached price data for {ticker}"
            ),
            "status": "error",
        }


def get_chart_links(ticker: str) -> dict[str, Any]:
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
        return {
            "error": safe_error_message(
                e, context=f"generating chart links for {ticker}"
            ),
        }


def clear_cache(ticker: str | None = None) -> dict[str, Any]:
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

        if ticker:
            pattern = f"stock:{ticker}:*"
            count = cache_clear(pattern)
            message = f"Cleared cache for {ticker}"
        else:
            count = cache_clear()
            message = "Cleared all cache entries"

        return {"status": "success", "message": message, "entries_cleared": count}
    except Exception as e:
        return {
            "error": safe_error_message(e, context="clearing cache"),
            "status": "error",
        }


def warm_cache(
    symbols: list[str] | None = None,
    days: int = 365,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Pre-fetch OHLCV data for watchlist symbols to warm the cache.

    Designed to be called before market open (e.g., 6:30 AM ET via scheduled
    task) so first queries of the day are instant.

    If no symbols provided, warms cache for all portfolio holdings.

    Args:
        symbols: List of ticker symbols to warm (optional)
        days: Days of historical data to fetch (default: 365)
        user_id: User identifier for portfolio lookup
        portfolio_name: Portfolio name for auto-detection

    Returns:
        Dictionary with warm-up statistics
    """
    import time

    start_time = time.time()

    # Auto-detect from portfolio if no symbols provided
    if not symbols:
        try:
            from maverick_mcp.data.models import UserPortfolio, get_db

            db = next(get_db())
            try:
                portfolio = (
                    db.query(UserPortfolio)
                    .filter_by(user_id=user_id, name=portfolio_name)
                    .first()
                )
                if portfolio:
                    symbols = [pos.ticker for pos in portfolio.positions]
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to load portfolio for cache warm-up: %s", e)

    if not symbols:
        return {
            "status": "empty",
            "message": "No symbols to warm — provide symbols or add portfolio positions",
        }

    provider = StockDataProvider()
    from datetime import timedelta

    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
    start_date = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")

    results = {"warmed": [], "failed": [], "already_cached": []}

    for symbol in symbols:
        try:
            provider.get_stock_data(symbol, start_date, end_date)
            results["warmed"].append(symbol)
        except Exception as e:
            logger.warning("Cache warm-up failed for %s: %s", symbol, e)
            results["failed"].append(
                {
                    "symbol": symbol,
                    "error": safe_error_message(
                        e, context=f"warming cache for {symbol}"
                    ),
                }
            )

    elapsed_ms = round((time.time() - start_time) * 1000)

    return {
        "status": "success",
        "symbols_requested": len(symbols),
        "symbols_warmed": len(results["warmed"]),
        "symbols_failed": len(results["failed"]),
        "warmed": results["warmed"],
        "failed": results["failed"],
        "elapsed_ms": elapsed_ms,
    }


def check_watchlist_alerts(
    tickers: list[str] | None = None,
    conditions: list[str] | None = None,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    volume_spike_multiplier: float = 2.0,
    trailing_stop_pct: float = 5.0,
) -> dict[str, Any]:
    """Check technical alert conditions for specified tickers or portfolio.

    Evaluates multiple technical conditions and returns any triggered alerts.
    If no tickers are provided, automatically checks all portfolio positions.

    DISCLAIMER: Alerts are for educational purposes only and do not
    constitute investment advice.

    Supported conditions: rsi_overbought, rsi_oversold, macd_bullish_cross,
    macd_bearish_cross, price_above_resistance, price_below_support,
    volume_spike, bollinger_squeeze, trailing_stop.

    Args:
        tickers: List of ticker symbols to check (optional, auto-uses portfolio)
        conditions: Specific conditions to check (optional, checks all by default)
        rsi_overbought: RSI overbought threshold (default 70)
        rsi_oversold: RSI oversold threshold (default 30)
        volume_spike_multiplier: Volume spike threshold multiplier (default 2.0)
        trailing_stop_pct: Trailing stop drop percentage (default 5.0)

    Returns:
        Dictionary with alert results per ticker and summary counts
    """
    from maverick_mcp.core.watchlist_monitor import (
        check_alerts_for_ticker,
        check_portfolio_alerts,
    )

    try:
        if tickers:
            # Check specific tickers
            results = []
            total_alerts = 0
            for ticker in tickers:
                result = check_alerts_for_ticker(
                    ticker,
                    conditions=conditions,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    volume_spike_multiplier=volume_spike_multiplier,
                    trailing_stop_pct=trailing_stop_pct,
                )
                results.append(result)
                total_alerts += result.get("alert_count", 0)

            results.sort(key=lambda r: r.get("alert_count", 0), reverse=True)
            return {
                "status": "ok",
                "tickers_checked": len(tickers),
                "total_alerts": total_alerts,
                "results": results,
            }
        else:
            # Auto-detect from portfolio
            return check_portfolio_alerts(
                conditions=conditions,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
                volume_spike_multiplier=volume_spike_multiplier,
                trailing_stop_pct=trailing_stop_pct,
            )
    except Exception as e:
        return {
            "error": safe_error_message(e, context="checking watchlist alerts"),
            "status": "error",
        }


def get_intraday_summary(
    ticker: str,
    interval: str = "5m",
) -> dict[str, Any]:
    """Get intraday price summary with VWAP for a single ticker.

    Fetches recent intraday data and calculates VWAP, day range,
    and volume profile. Best used during market hours.

    Args:
        ticker: Stock ticker symbol
        interval: Intraday interval - "1m", "5m", "15m", "30m", "1h"
                  (default: "5m")

    Returns:
        Dictionary with current price, VWAP, day range, and volume stats
    """
    from maverick_mcp.core.technical_analysis import calculate_vwap

    try:
        provider = StockDataProvider()
        df = provider.get_stock_data(ticker, period="1d", interval=interval)

        if df is None or df.empty:
            return {
                "ticker": ticker,
                "status": "error",
                "message": f"No intraday data available for {ticker}",
            }

        current_price = float(df["close"].iloc[-1])
        day_open = float(df["open"].iloc[0])
        day_high = float(df["high"].max())
        day_low = float(df["low"].min())
        total_volume = int(df["volume"].sum())

        # Calculate VWAP
        vwap_series = calculate_vwap(df)
        current_vwap = float(vwap_series.iloc[-1]) if not vwap_series.empty else None

        # Price vs VWAP
        vwap_signal = None
        if current_vwap:
            if current_price > current_vwap:
                vwap_signal = "above_vwap (bullish)"
            else:
                vwap_signal = "below_vwap (bearish)"

        return {
            "ticker": ticker.upper(),
            "interval": interval,
            "status": "ok",
            "current_price": round(current_price, 2),
            "vwap": round(current_vwap, 2) if current_vwap else None,
            "vwap_signal": vwap_signal,
            "day_open": round(day_open, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "day_range": round(day_high - day_low, 2),
            "change_from_open": round(current_price - day_open, 2),
            "change_pct": round(((current_price - day_open) / day_open) * 100, 2)
            if day_open > 0
            else 0.0,
            "total_volume": total_volume,
            "bar_count": len(df),
        }
    except Exception as e:
        return {
            "error": safe_error_message(
                e, context=f"getting intraday summary for {ticker}"
            ),
            "ticker": ticker,
            "status": "error",
        }
