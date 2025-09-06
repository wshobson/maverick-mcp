"""
Market data providers and utilities for Maverick-MCP.
Provides market movers, gainers, losers, and other market-wide data.
"""

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, cast

# Suppress specific pyright warnings for pandas DataFrame column access
# pyright: reportAttributeAccessIssue=false
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from finvizfinance.screener.overview import Overview
from requests.adapters import HTTPAdapter, Retry
from tiingo import TiingoClient

from maverick_mcp.utils.circuit_breaker_decorators import (
    with_market_data_circuit_breaker,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.market_data")

# Initialize Tiingo client
tiingo_config = {"session": True, "api_key": os.getenv("TIINGO_API_KEY")}
tiingo_client = TiingoClient(tiingo_config) if os.getenv("TIINGO_API_KEY") else None

# Market indices - these are standard references
MARKET_INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y Treasury",
}

# Sector ETFs - these are standard references
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Consumer Staples": "XLP",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


class ExternalAPIClient:
    """Client for External API."""

    def __init__(self):
        self.api_key = os.getenv("CAPITAL_COMPANION_API_KEY")
        self.base_url = "https://capitalcompanion.io"
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-KEY": self.api_key}
        ) if self.api_key else None

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @with_market_data_circuit_breaker(use_fallback=False, service="external_api")
    def _make_request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make API request with circuit breaker protection."""
        list_endpoints = [
            "/gainers",
            "/losers",
            "/maverick-full",
            "/maverick-bullish-stocks",
            "/maverick-bearish-stocks",
            "/top-ten-retail",
            "/aggressive-small-caps",
            "/undervalued",
            "/tech-earnings-growth",
            "/unusual-options-activity",
        ]

        if not self.api_key:
            logger.warning("External API key not configured")
            return [] if endpoint in list_endpoints else {}

        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=(5, 30))
        response.raise_for_status()
        result = response.json()
        return result

    def get_gainers(self) -> list[dict[str, Any]]:
        """Get top gainers from External API."""
        result = self._make_request("/gainers")
        return result if isinstance(result, list) else []

    def get_losers(self) -> list[dict[str, Any]]:
        """Get top losers from External API."""
        result = self._make_request("/losers")
        return result if isinstance(result, list) else []

    def get_maverick_full(self) -> list[dict[str, Any]]:
        """Get full maverick stocks list."""
        result = self._make_request("/maverick-full")
        return result if isinstance(result, list) else []

    def get_maverick_bullish(self) -> list[dict[str, Any]]:
        """Get maverick bullish stocks."""
        result = self._make_request("/maverick-bullish-stocks")
        return result if isinstance(result, list) else []

    def get_maverick_bearish(self) -> list[dict[str, Any]]:
        """Get maverick bearish stocks."""
        result = self._make_request("/maverick-bearish-stocks")
        return result if isinstance(result, list) else []

    def get_top_retail(self) -> list[dict[str, Any]]:
        """Get top retail traded stocks."""
        # Note: The endpoint name uses hyphens, not underscores
        result = self._make_request("/top-ten-retail")
        return result if isinstance(result, list) else []

    def get_aggressive_small_caps(self) -> list[dict[str, Any]]:
        """Get aggressive small cap stocks."""
        result = self._make_request("/aggressive-small-caps")
        return result if isinstance(result, list) else []

    def get_undervalued(self) -> list[dict[str, Any]]:
        """Get potentially undervalued large cap stocks."""
        result = self._make_request("/undervalued")
        return result if isinstance(result, list) else []

    def get_tech_earnings_growth(self) -> list[dict[str, Any]]:
        """Get tech stocks with earnings growth over 25%."""
        result = self._make_request("/tech-earnings-growth")
        return result if isinstance(result, list) else []

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get stock quote by symbol."""
        result = self._make_request(f"/quote/{symbol}")
        return result if isinstance(result, dict) else {}


# Initialize External API client
external_api_client = ExternalAPIClient()


@with_market_data_circuit_breaker(use_fallback=False, service="finviz")
def get_finviz_movers(mover_type: str = "gainers", limit: int = 50) -> list[str]:
    """
    Get market movers using finvizfinance screener with circuit breaker protection.

    Args:
        mover_type: Type of movers to get ("gainers", "losers", "active")
        limit: Maximum number of stocks to return

    Returns:
        List of ticker symbols
    """
    foverview = Overview()

    # Set up filters based on mover type
    if mover_type == "gainers":
        filters_dict = {
            "Change": "Up 5%",  # More than 5% gain
            "Average Volume": "Over 1M",  # Liquid stocks
            "Price": "Over $5",  # Avoid penny stocks
        }
    elif mover_type == "losers":
        filters_dict = {
            "Change": "Down 5%",  # More than 5% loss
            "Average Volume": "Over 1M",
            "Price": "Over $5",
        }
    elif mover_type == "active":
        filters_dict = {
            "Average Volume": "Over 20M",  # Very high volume
            "Price": "Over $5",
        }
    else:
        # Default to liquid stocks
        filters_dict = {
            "Average Volume": "Over 10M",
            "Market Cap.": "Large (>10bln)",
            "Price": "Over $10",
        }

    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    if df is not None and not df.empty:
        # Sort by appropriate column
        if mover_type == "gainers" and "Change" in df.columns:
            df = df.sort_values("Change", ascending=False)
        elif mover_type == "losers" and "Change" in df.columns:
            df = df.sort_values("Change", ascending=True)
        elif mover_type == "active" and "Volume" in df.columns:
            df = df.sort_values("Volume", ascending=False)

        # Get ticker symbols
        if "Ticker" in df.columns:
            return list(df["Ticker"].head(limit).tolist())

    logger.debug(f"No finviz data available for {mover_type}")
    return []


def get_finviz_stock_data(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Get stock data for symbols using finvizfinance.

    Note: finvizfinance doesn't support direct symbol filtering,
    so we use yfinance for specific symbol data instead.

    Args:
        symbols: List of ticker symbols

    Returns:
        List of dictionaries with stock data
    """
    # Use yfinance for specific symbol data as finvizfinance
    # doesn't support direct symbol filtering efficiently
    results = []

    for symbol in symbols[:20]:  # Limit to prevent overwhelming
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info and "currentPrice" in info:
                price = info.get("currentPrice", 0)
                prev_close = info.get("previousClose", price)
                change = price - prev_close if prev_close else 0
                change_percent = (change / prev_close * 100) if prev_close else 0
                volume = info.get("volume", 0)

                results.append(
                    {
                        "symbol": symbol,
                        "price": round(price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "volume": volume,
                    }
                )
        except Exception as e:
            logger.debug(f"Error fetching data for {symbol}: {e}")
            continue

    return results


def fetch_tiingo_tickers():
    """
    Fetch active US stock and ETF tickers. First tries External API,
    then falls back to Tiingo if available.

    Returns:
        List of valid ticker symbols
    """
    # Try External API first
    try:
        maverick_full = external_api_client.get_maverick_full()
        if maverick_full:
            # Extract symbols from the maverick full list
            symbols = []
            # Handle different response formats
            if isinstance(maverick_full, dict):
                # API returns {"maverick_stocks": [...]}
                if "maverick_stocks" in maverick_full:
                    for item in maverick_full["maverick_stocks"]:
                        if isinstance(item, str):
                            symbols.append(item)
                        elif isinstance(item, dict) and "symbol" in item:
                            symbols.append(item["symbol"])
            elif isinstance(maverick_full, list):
                # Direct list format
                for item in maverick_full:
                    if isinstance(item, dict) and "symbol" in item:
                        symbols.append(item["symbol"])
                    elif isinstance(item, str):
                        symbols.append(item)

            if symbols:
                return sorted(set(symbols))
    except Exception as e:
        logger.debug(f"Could not fetch from External API: {e}")

    # Fall back to Tiingo if available
    if tiingo_client:
        try:
            asset_types = frozenset(["Stock", "ETF"])
            valid_exchanges = frozenset(["NYSE", "NASDAQ", "BATS", "NYSE ARCA", "AMEX"])
            cutoff_date = datetime(2024, 7, 1)

            tickers = tiingo_client.list_tickers(assetTypes=list(asset_types))

            valid_tickers = set()
            for t in tickers:
                ticker = t["ticker"].strip()
                if (
                    len(ticker) <= 5
                    and ticker.isalpha()
                    and t["exchange"].strip() in valid_exchanges
                    and t["priceCurrency"].strip() == "USD"
                    and t["assetType"].strip() in asset_types
                    and t["endDate"]
                    and datetime.fromisoformat(t["endDate"].rstrip("Z")) > cutoff_date
                ):
                    valid_tickers.add(ticker)

            return sorted(valid_tickers)
        except Exception as e:
            logger.error(f"Error fetching tickers from Tiingo: {str(e)}")

    # Fall back to finvizfinance
    try:
        # Get a mix of liquid stocks from finviz
        finviz_symbols: set[str] = set()

        # Get some active stocks
        active = get_finviz_movers("active", limit=100)
        finviz_symbols.update(active)

        # Get some gainers
        gainers = get_finviz_movers("gainers", limit=50)
        finviz_symbols.update(gainers)

        # Get some losers
        losers = get_finviz_movers("losers", limit=50)
        finviz_symbols.update(losers)

        if finviz_symbols:
            return sorted(finviz_symbols)

    except Exception as e:
        logger.debug(f"Error fetching from finvizfinance: {e}")

    logger.warning("No ticker source available, returning empty list")
    return []


class MarketDataProvider:
    """
    Provider for market-wide data including top gainers, losers, and other market metrics.
    Uses Yahoo Finance and other sources.
    """

    def __init__(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    async def _run_in_executor(self, func, *args) -> Any:
        """Run a blocking function in an executor to make it non-blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    def _fetch_data(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Fetch data from an API with retry logic.

        Args:
            url: API endpoint URL
            params: Optional query parameters

        Returns:
            JSON response as dictionary
        """
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=(5, 30),  # Connect timeout, read timeout
                headers={"User-Agent": "Maverick-MCP/1.0"},
            )
            response.raise_for_status()
            result = response.json()
            return result if isinstance(result, dict) else {}
        except requests.Timeout:
            logger.error(f"Timeout error fetching data from {url}")
            return {}
        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching data from {url}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unknown error fetching data from {url}: {str(e)}")
            return {}

    def get_market_summary(self) -> dict[str, Any]:
        """
        Get a summary of major market indices.

        Returns:
            Dictionary with market summary data
        """
        try:
            import yfinance as yf

            data = {}
            for index, name in MARKET_INDICES.items():
                ticker = yf.Ticker(index)
                history = ticker.history(period="2d")

                if history.empty:
                    continue

                prev_close = (
                    history["Close"].iloc[0]
                    if len(history) > 1
                    else history["Close"].iloc[0]
                )
                current = history["Close"].iloc[-1]
                change = current - prev_close
                change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

                data[index] = {
                    "name": name,
                    "symbol": index,
                    "price": round(current, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                }

            return data
        except Exception as e:
            logger.error(f"Error fetching market summary: {str(e)}")
            return {}

    def get_top_gainers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top gaining stocks in the market.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data
        """
        try:
            # First try External API
            gainers_data = external_api_client.get_gainers()

            if gainers_data:
                results = []
                # Handle different response formats
                gainers_list = []
                if isinstance(gainers_data, dict) and "gainers" in gainers_data:
                    gainers_list = gainers_data["gainers"]
                elif isinstance(gainers_data, list):
                    gainers_list = gainers_data

                for item in gainers_list[:limit]:
                    # Handle different response formats
                    if isinstance(item, dict):
                        # Extract standard fields
                        result = {
                            "symbol": item.get("symbol", item.get("ticker", "")),
                            "price": item.get("price", item.get("current_price", 0)),
                            "change": item.get("change", item.get("price_change", 0)),
                            "change_percent": item.get(
                                "percent_change", item.get("change_percent", 0)
                            ),
                            "volume": item.get("volume", 0),
                        }

                        # Ensure numeric types
                        result["price"] = (
                            float(result["price"]) if result["price"] else 0
                        )
                        result["change"] = (
                            float(result["change"]) if result["change"] else 0
                        )
                        result["change_percent"] = (
                            float(result["change_percent"])
                            if result["change_percent"]
                            else 0
                        )
                        result["volume"] = (
                            int(result["volume"]) if result["volume"] else 0
                        )

                        if result["symbol"]:
                            results.append(result)

                if results:
                    return results[:limit]

            # If External API fails, try finvizfinance
            logger.info("External API gainers unavailable, trying finvizfinance")

            # Try to get gainers from finvizfinance
            symbols = get_finviz_movers("gainers", limit=limit * 2)

            if symbols:
                # First try to get data directly from finviz
                results = get_finviz_stock_data(symbols[:limit])
                if results:
                    # Sort by percent change and return top gainers
                    results.sort(key=lambda x: x["change_percent"], reverse=True)
                    return results[:limit]

            # If finviz doesn't have full data, use yfinance with the symbols
            if not symbols:
                # Last resort: try to get any liquid stocks from finviz
                symbols = get_finviz_movers("active", limit=50)

            if not symbols:
                logger.warning("No symbols available for gainers calculation")
                return []

            # Fetch data for these symbols
            results = []
            batch_str = " ".join(symbols[:50])  # Limit to 50 symbols

            data = yf.download(
                batch_str,
                period="2d",
                group_by="ticker",
                threads=True,
                progress=False,
            )

            if data is None or data.empty:
                logger.warning("No data available from yfinance")
                return []

            for symbol in symbols[:50]:
                try:
                    if len(symbols) == 1:
                        ticker_data = data
                    else:
                        if symbol not in data.columns.get_level_values(0):
                            continue
                        ticker_data = data[symbol]

                    if len(ticker_data) < 2:
                        continue

                    prev_close = ticker_data["Close"].iloc[0]
                    current = ticker_data["Close"].iloc[-1]

                    if pd.isna(prev_close) or pd.isna(current) or prev_close == 0:
                        continue

                    change = current - prev_close
                    change_percent = (change / prev_close) * 100
                    volume = ticker_data["Volume"].iloc[-1]

                    if pd.notna(change_percent) and pd.notna(volume):
                        results.append(
                            {
                                "symbol": symbol,
                                "price": round(current, 2),
                                "change": round(change, 2),
                                "change_percent": round(change_percent, 2),
                                "volume": int(volume),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {str(e)}")
                    continue

            # Sort by percent change and return top gainers
            results.sort(key=lambda x: x["change_percent"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error fetching top gainers: {str(e)}")
            return []

    def get_top_losers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top losing stocks in the market.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data
        """
        try:
            # First try External API
            losers_data = external_api_client.get_losers()

            if losers_data:
                results = []
                # Handle different response formats
                losers_list = []
                if isinstance(losers_data, dict) and "losers" in losers_data:
                    losers_list = losers_data["losers"]
                elif isinstance(losers_data, list):
                    losers_list = losers_data

                for item in losers_list[:limit]:
                    # Handle different response formats
                    if isinstance(item, dict):
                        # Extract standard fields
                        result = {
                            "symbol": item.get("symbol", item.get("ticker", "")),
                            "price": item.get("price", item.get("current_price", 0)),
                            "change": item.get("change", item.get("price_change", 0)),
                            "change_percent": item.get(
                                "percent_change", item.get("change_percent", 0)
                            ),
                            "volume": item.get("volume", 0),
                        }

                        # Ensure numeric types
                        result["price"] = (
                            float(result["price"]) if result["price"] else 0
                        )
                        result["change"] = (
                            float(result["change"]) if result["change"] else 0
                        )
                        result["change_percent"] = (
                            float(result["change_percent"])
                            if result["change_percent"]
                            else 0
                        )
                        result["volume"] = (
                            int(result["volume"]) if result["volume"] else 0
                        )

                        if result["symbol"]:
                            results.append(result)

                if results:
                    return results[:limit]

            # If External API fails, try finvizfinance
            logger.info("External API losers unavailable, trying finvizfinance")

            # Try to get losers from finvizfinance
            symbols = get_finviz_movers("losers", limit=limit * 2)

            if symbols:
                # First try to get data directly from finviz
                results = get_finviz_stock_data(symbols[:limit])
                if results:
                    # Sort by percent change (ascending for losers) and return top losers
                    results.sort(key=lambda x: x["change_percent"])
                    return results[:limit]

            # If finviz doesn't have full data, use yfinance with the symbols
            if not symbols:
                # Last resort: try to get any liquid stocks from finviz
                symbols = get_finviz_movers("active", limit=50)

            if not symbols:
                logger.warning("No symbols available for losers calculation")
                return []

            # Fetch data for these symbols
            results = []
            batch_str = " ".join(symbols[:50])  # Limit to 50 symbols

            data = yf.download(
                batch_str,
                period="2d",
                group_by="ticker",
                threads=True,
                progress=False,
            )

            if data is None or data.empty:
                logger.warning("No data available from yfinance")
                return []

            for symbol in symbols[:50]:
                try:
                    if len(symbols) == 1:
                        ticker_data = data
                    else:
                        if symbol not in data.columns.get_level_values(0):
                            continue
                        ticker_data = data[symbol]

                    if len(ticker_data) < 2:
                        continue

                    prev_close = ticker_data["Close"].iloc[0]
                    current = ticker_data["Close"].iloc[-1]

                    if pd.isna(prev_close) or pd.isna(current) or prev_close == 0:
                        continue

                    change = current - prev_close
                    change_percent = (change / prev_close) * 100
                    volume = ticker_data["Volume"].iloc[-1]

                    if pd.notna(change_percent) and pd.notna(volume):
                        results.append(
                            {
                                "symbol": symbol,
                                "price": round(current, 2),
                                "change": round(change, 2),
                                "change_percent": round(change_percent, 2),
                                "volume": int(volume),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {str(e)}")
                    continue

            # Sort by percent change (ascending for losers) and return top losers
            results.sort(key=lambda x: x["change_percent"])
            return results[:limit]

        except Exception as e:
            logger.error(f"Error fetching top losers: {str(e)}")
            return []

    def get_most_active(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get most active stocks by volume.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data
        """
        try:
            # Use External API's various endpoints for most active stocks
            # First try gainers as they have high volume
            active_data = external_api_client.get_gainers()

            if not active_data:
                # Fall back to maverick stocks
                maverick_data = external_api_client.get_maverick_full()
                if (
                    isinstance(maverick_data, dict)
                    and "maverick_stocks" in maverick_data
                ):
                    active_data = [
                        {"symbol": s}
                        for s in maverick_data["maverick_stocks"][: limit * 2]
                    ]

            if active_data:
                results = []
                symbols = []

                # Extract data depending on format
                data_list = []
                if isinstance(active_data, dict) and "gainers" in active_data:
                    data_list = active_data["gainers"]
                elif isinstance(active_data, list):
                    data_list = active_data

                # Extract symbols from data
                for item in data_list:
                    if isinstance(item, dict):
                        symbol = item.get("symbol", item.get("ticker", ""))
                        if symbol:
                            symbols.append(symbol)
                            # If the API already provides full data, use it
                            if all(
                                k in item
                                for k in ["price", "change", "change_percent", "volume"]
                            ):
                                result = {
                                    "symbol": symbol,
                                    "price": float(item.get("price", 0)),
                                    "change": float(item.get("change", 0)),
                                    "change_percent": float(
                                        item.get("change_percent", 0)
                                    ),
                                    "volume": int(item.get("volume", 0)),
                                }
                                results.append(result)
                    elif isinstance(item, str):
                        symbols.append(item)

                # If we have complete results from API, return them
                if results:
                    return results[:limit]

                # Otherwise fetch additional data for symbols
                if symbols:
                    # Limit symbols to fetch
                    symbols = symbols[
                        : min(limit * 2, 30)
                    ]  # Fetch more than limit to account for potential errors
                    batch_str = " ".join(symbols)

                    data = yf.download(
                        batch_str,
                        period="2d",
                        group_by="ticker",
                        threads=True,
                        progress=False,
                    )

                    if data is None or data.empty:
                        logger.warning("No data available from yfinance")
                        return results[:limit]

                    for symbol in symbols:
                        try:
                            if len(symbols) == 1:
                                ticker_data = data
                            else:
                                if symbol not in data.columns.get_level_values(0):
                                    continue
                                ticker_data = data[symbol]

                            if len(ticker_data) < 2:
                                continue

                            prev_close = ticker_data["Close"].iloc[0]
                            current = ticker_data["Close"].iloc[-1]
                            volume = ticker_data["Volume"].iloc[-1]

                            if (
                                pd.isna(prev_close)
                                or pd.isna(current)
                                or pd.isna(volume)
                                or prev_close == 0
                            ):
                                continue

                            change = current - prev_close
                            change_percent = (change / prev_close) * 100

                            if pd.notna(change_percent) and pd.notna(volume):
                                results.append(
                                    {
                                        "symbol": symbol,
                                        "price": round(current, 2),
                                        "change": round(change, 2),
                                        "change_percent": round(change_percent, 2),
                                        "volume": int(volume),
                                    }
                                )
                        except Exception as e:
                            logger.debug(f"Error processing {symbol}: {str(e)}")
                            continue

                    # Sort by volume and return most active
                    results.sort(key=lambda x: x["volume"], reverse=True)
                    return results[:limit]

            # If no data from External API, try finvizfinance
            logger.info("Trying finvizfinance for most active stocks")

            # Get most active stocks from finviz
            symbols = get_finviz_movers("active", limit=limit * 2)

            if symbols:
                # First try to get data directly from finviz
                results = get_finviz_stock_data(symbols[:limit])
                if results:
                    # Sort by volume and return most active
                    results.sort(key=lambda x: x["volume"], reverse=True)
                    return results[:limit]

                # If finviz doesn't have full data, use yfinance
                batch_str = " ".join(symbols[: limit * 2])

                data = yf.download(
                    batch_str,
                    period="2d",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )

                if data is None or data.empty:
                    logger.warning("No data available from yfinance")
                    return []

                results = []
                for symbol in symbols[: limit * 2]:
                    try:
                        if len(symbols) == 1:
                            ticker_data = data
                        else:
                            if symbol not in data.columns.get_level_values(0):
                                continue
                            ticker_data = data[symbol]

                        if len(ticker_data) < 2:
                            continue

                        prev_close = ticker_data["Close"].iloc[0]
                        current = ticker_data["Close"].iloc[-1]
                        volume = ticker_data["Volume"].iloc[-1]

                        if (
                            pd.isna(prev_close)
                            or pd.isna(current)
                            or pd.isna(volume)
                            or prev_close == 0
                        ):
                            continue

                        change = current - prev_close
                        change_percent = (change / prev_close) * 100

                        if pd.notna(change_percent) and pd.notna(volume):
                            results.append(
                                {
                                    "symbol": symbol,
                                    "price": round(current, 2),
                                    "change": round(change, 2),
                                    "change_percent": round(change_percent, 2),
                                    "volume": int(volume),
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {str(e)}")
                        continue

                # Sort by volume and return most active
                results.sort(key=lambda x: x["volume"], reverse=True)
                return results[:limit]

            logger.warning("No most active stocks data available")
            return []

        except Exception as e:
            logger.error(f"Error fetching most active stocks: {str(e)}")
            return []

    def get_sector_performance(self) -> dict[str, float]:
        """
        Get sector performance data.

        Returns:
            Dictionary mapping sector names to performance percentages
        """
        try:
            import yfinance as yf

            results = {}
            for sector, etf in SECTOR_ETFS.items():
                try:
                    data = yf.Ticker(etf)
                    hist = data.history(period="2d")

                    if len(hist) < 2:
                        continue

                    prev_close = hist["Close"].iloc[0]
                    current = hist["Close"].iloc[-1]
                    change_percent = ((current - prev_close) / prev_close) * 100

                    results[sector] = round(change_percent, 2)
                except Exception as e:
                    logger.debug(f"Error processing sector {sector}: {str(e)}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return {}

    def get_earnings_calendar(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Get upcoming earnings announcements.

        Args:
            days: Number of days to look ahead

        Returns:
            List of dictionaries with earnings data
        """
        try:
            # Get stocks to check for earnings from External API
            stocks_to_check = []

            # Try to get a diverse set of stocks from different External API endpoints
            try:
                # Get gainers for earnings check
                gainers_data = external_api_client.get_gainers()
                if gainers_data:
                    gainers_list = []
                    if isinstance(gainers_data, dict) and "gainers" in gainers_data:
                        gainers_list = gainers_data["gainers"]
                    elif isinstance(gainers_data, list):
                        gainers_list = gainers_data

                    for item in gainers_list[:15]:
                        if isinstance(item, dict) and "symbol" in item:
                            stocks_to_check.append(item["symbol"])

                # Add some tech stocks with earnings growth
                tech_stocks = external_api_client.get_tech_earnings_growth()
                for item in tech_stocks[:10]:
                    if isinstance(item, dict) and "symbol" in item:  # type: ignore[arg-type]
                        symbol = item["symbol"]
                        if symbol not in stocks_to_check:
                            stocks_to_check.append(symbol)
                    elif isinstance(item, str) and item not in stocks_to_check:
                        stocks_to_check.append(item)

                # Add some undervalued stocks
                undervalued = external_api_client.get_undervalued()
                for item in undervalued[:10]:
                    if isinstance(item, dict) and "symbol" in item:  # type: ignore[arg-type]
                        symbol = item["symbol"]
                        if symbol not in stocks_to_check:
                            stocks_to_check.append(symbol)
                    elif isinstance(item, str) and item not in stocks_to_check:
                        stocks_to_check.append(item)

            except Exception as e:
                logger.debug(
                    f"Could not fetch stocks from External API for earnings: {e}"
                )

            # If no stocks from External API, fall back to fetch_tiingo_tickers
            if not stocks_to_check:
                tickers = fetch_tiingo_tickers()
                stocks_to_check = tickers[:50] if tickers else []

            check_stocks = stocks_to_check[:50]  # Limit to 50 stocks for performance

            results = []
            today = datetime.now(UTC).date()
            end_date = today + timedelta(days=days)

            for ticker in check_stocks:
                try:
                    data = yf.Ticker(ticker)

                    # Try to get calendar info
                    if hasattr(data, "calendar") and data.calendar is not None:
                        try:
                            calendar = data.calendar
                            if "Earnings Date" in calendar.index:
                                earnings_date = calendar.loc["Earnings Date"]

                                # Handle different date formats
                                if hasattr(earnings_date, "date"):
                                    earnings_date = earnings_date.date()
                                elif isinstance(earnings_date, str):
                                    earnings_date = datetime.strptime(
                                        earnings_date, "%Y-%m-%d"
                                    ).date()
                                else:
                                    continue

                                # Check if earnings date is within our range
                                if today <= earnings_date <= end_date:
                                    results.append(
                                        {
                                            "ticker": ticker,
                                            "name": data.info.get("shortName", ticker),
                                            "earnings_date": earnings_date.strftime(
                                                "%Y-%m-%d"
                                            ),
                                            "eps_estimate": float(
                                                calendar.loc["EPS Estimate"]
                                            )
                                            if "EPS Estimate" in calendar.index
                                            else None,
                                        }
                                    )
                        except Exception as e:
                            logger.debug(
                                f"Error parsing calendar for {ticker}: {str(e)}"
                            )
                            continue
                except Exception as e:
                    logger.debug(f"Error fetching data for {ticker}: {str(e)}")
                    continue

            # Sort by earnings date
            results.sort(key=lambda x: x["earnings_date"])
            return results

        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {str(e)}")
            return []

    async def get_market_summary_async(self) -> dict[str, Any]:
        """
        Get a summary of major market indices (async version).
        """
        result = await self._run_in_executor(self.get_market_summary)
        return cast(dict[str, Any], result)

    async def get_top_gainers_async(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top gaining stocks in the market (async version).
        """
        result = await self._run_in_executor(self.get_top_gainers, limit)
        return cast(list[dict[str, Any]], result)

    async def get_top_losers_async(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top losing stocks in the market (async version).
        """
        result = await self._run_in_executor(self.get_top_losers, limit)
        return cast(list[dict[str, Any]], result)

    async def get_most_active_async(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get most active stocks by volume (async version).
        """
        result = await self._run_in_executor(self.get_most_active, limit)
        return cast(list[dict[str, Any]], result)

    async def get_sector_performance_async(self) -> dict[str, float]:
        """
        Get sector performance data (async version).
        """
        result = await self._run_in_executor(self.get_sector_performance)
        return cast(dict[str, float], result)

    async def get_market_overview_async(self) -> dict[str, Any]:
        """
        Get comprehensive market overview including summary, gainers, losers, sectors (async version).

        Uses concurrent execution for better performance.
        """
        # Run all tasks concurrently
        tasks = [
            self.get_market_summary_async(),
            self.get_top_gainers_async(5),
            self.get_top_losers_async(5),
            self.get_sector_performance_async(),
        ]

        # Wait for all tasks to complete
        summary, gainers, losers, sectors = await asyncio.gather(*tasks)  # type: ignore[assignment]

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "market_summary": summary,
            "top_gainers": gainers,
            "top_losers": losers,
            "sector_performance": sectors,
        }

    def get_market_overview(self) -> dict[str, Any]:
        """
        Get comprehensive market overview including summary, gainers, losers, sectors.

        Returns:
            Dictionary with market overview data
        """
        summary = self.get_market_summary()
        gainers = self.get_top_gainers(5)
        losers = self.get_top_losers(5)
        sectors = self.get_sector_performance()

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "market_summary": summary,
            "top_gainers": gainers,
            "top_losers": losers,
            "sector_performance": sectors,
        }
