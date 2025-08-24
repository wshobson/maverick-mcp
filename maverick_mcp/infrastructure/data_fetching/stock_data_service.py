"""
Stock Data Fetching Service - Responsible only for fetching data from external sources.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from maverick_mcp.utils.circuit_breaker_decorators import (
    with_stock_data_circuit_breaker,
)

logger = logging.getLogger("maverick_mcp.stock_data_fetching")


class StockDataFetchingService:
    """
    Service responsible ONLY for fetching stock data from external sources.

    This service:
    - Handles data fetching from yfinance, Alpha Vantage, etc.
    - Manages fallback logic between data sources
    - Contains no caching logic
    - Contains no business logic beyond data retrieval
    """

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the stock data fetching service.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.timeout = timeout
        self.max_retries = max_retries

    @with_stock_data_circuit_breaker(use_fallback=False)
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch stock data from yfinance with circuit breaker protection.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Alternative to start/end dates (e.g., '1d', '5d', '1mo', etc.)
            interval: Data interval ('1d', '1wk', '1mo', '1m', '5m', etc.)

        Returns:
            DataFrame with stock data

        Raises:
            Exception: If data fetching fails after retries
        """
        logger.info(
            f"Fetching data from yfinance for {symbol} - "
            f"Start: {start_date}, End: {end_date}, Period: {period}, Interval: {interval}"
        )

        ticker = yf.Ticker(symbol)

        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            if start_date is None:
                start_date = (datetime.now(UTC) - timedelta(days=365)).strftime(
                    "%Y-%m-%d"
                )
            if end_date is None:
                end_date = datetime.now(UTC).strftime("%Y-%m-%d")
            df = ticker.history(start=start_date, end=end_date, interval=interval)

        # Validate and clean the data
        df = self._validate_and_clean_data(df, symbol)
        return df

    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean the fetched data.

        Args:
            df: Raw DataFrame from data source
            symbol: Stock symbol for logging

        Returns:
            Cleaned DataFrame
        """
        # Check if dataframe is empty
        if df.empty:
            logger.warning(f"Empty dataframe returned for {symbol}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Ensure all expected columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(
                    f"Column {col} missing from data for {symbol}, adding default value"
                )
                if col == "Volume":
                    df[col] = 0
                else:
                    df[col] = 0.0

        # Set index name
        df.index.name = "Date"

        # Ensure data types
        df["Volume"] = df["Volume"].astype(int)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)

        return df

    @with_stock_data_circuit_breaker(use_fallback=False)
    def fetch_stock_info(self, symbol: str) -> dict[str, Any]:
        """
        Fetch detailed stock information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        logger.info(f"Fetching stock info for {symbol}")
        ticker = yf.Ticker(symbol)
        return ticker.info

    def fetch_realtime_data(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch real-time data for a single symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with real-time data or None if failed
        """
        try:
            logger.info(f"Fetching real-time data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")

            if data.empty:
                logger.warning(f"No real-time data available for {symbol}")
                return None

            latest = data.iloc[-1]

            # Get previous close for change calculation
            prev_close = ticker.info.get("previousClose", None)
            if prev_close is None:
                # Try to get from 2-day history
                data_2d = ticker.history(period="2d")
                if len(data_2d) > 1:
                    prev_close = data_2d.iloc[0]["Close"]
                else:
                    prev_close = latest["Close"]

            # Calculate change
            price = latest["Close"]
            change = price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

            return {
                "symbol": symbol,
                "price": round(price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(latest["Volume"]),
                "timestamp": data.index[-1],
                "timestamp_display": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "is_real_time": False,  # yfinance data has some delay
            }
        except Exception as e:
            logger.error(f"Error fetching realtime data for {symbol}: {str(e)}")
            return None

    def fetch_multiple_realtime_data(self, symbols: list[str]) -> dict[str, Any]:
        """
        Fetch real-time data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to their real-time data
        """
        logger.info(f"Fetching real-time data for {len(symbols)} symbols")
        results = {}
        for symbol in symbols:
            data = self.fetch_realtime_data(symbol)
            if data:
                results[symbol] = data
        return results

    def fetch_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Fetch news for a stock.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of news items

        Returns:
            DataFrame with news data
        """
        try:
            logger.info(f"Fetching news for {symbol}")
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                return pd.DataFrame(
                    columns=[
                        "title",
                        "publisher",
                        "link",
                        "providerPublishTime",
                        "type",
                    ]
                )

            df = pd.DataFrame(news[:limit])

            # Convert timestamp to datetime
            if "providerPublishTime" in df.columns:
                df["providerPublishTime"] = pd.to_datetime(
                    df["providerPublishTime"], unit="s"
                )

            return df
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return pd.DataFrame(
                columns=["title", "publisher", "link", "providerPublishTime", "type"]
            )

    def fetch_earnings(self, symbol: str) -> dict[str, Any]:
        """
        Fetch earnings information for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with earnings data
        """
        try:
            logger.info(f"Fetching earnings for {symbol}")
            ticker = yf.Ticker(symbol)
            return {
                "earnings": ticker.earnings.to_dict()
                if hasattr(ticker, "earnings") and not ticker.earnings.empty
                else {},
                "earnings_dates": ticker.earnings_dates.to_dict()
                if hasattr(ticker, "earnings_dates") and not ticker.earnings_dates.empty
                else {},
                "earnings_trend": ticker.earnings_trend
                if hasattr(ticker, "earnings_trend")
                else {},
            }
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {str(e)}")
            return {"earnings": {}, "earnings_dates": {}, "earnings_trend": {}}

    def fetch_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Fetch analyst recommendations for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with recommendations
        """
        try:
            logger.info(f"Fetching recommendations for {symbol}")
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations

            if recommendations is None or recommendations.empty:
                return pd.DataFrame(columns=["firm", "toGrade", "fromGrade", "action"])

            return recommendations
        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {str(e)}")
            return pd.DataFrame(columns=["firm", "toGrade", "fromGrade", "action"])

    def check_if_etf(self, symbol: str) -> bool:
        """
        Check if a given symbol is an ETF.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if symbol is an ETF
        """
        try:
            logger.debug(f"Checking if {symbol} is an ETF")
            stock = yf.Ticker(symbol)

            # Check if quoteType exists and is ETF
            if "quoteType" in stock.info:
                return stock.info["quoteType"].upper() == "ETF"

            # Fallback check for common ETF identifiers
            common_etfs = [
                "SPY",
                "QQQ",
                "IWM",
                "DIA",
                "XLB",
                "XLE",
                "XLF",
                "XLI",
                "XLK",
                "XLP",
                "XLU",
                "XLV",
                "XLY",
                "XLC",
                "XLRE",
                "XME",
            ]

            return any(
                [
                    symbol.endswith(("ETF", "FUND")),
                    symbol in common_etfs,
                    "ETF" in stock.info.get("longName", "").upper(),
                ]
            )
        except Exception as e:
            logger.error(f"Error checking if {symbol} is ETF: {e}")
            return False
