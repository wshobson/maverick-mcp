"""
Stock data provider interfaces.

This module defines abstract interfaces for stock data fetching and screening operations.
These interfaces separate concerns between basic data retrieval and advanced screening logic,
following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class IStockDataFetcher(Protocol):
    """
    Interface for fetching basic stock data.

    This interface defines the contract for retrieving historical price data,
    real-time quotes, company information, and related financial data.
    """

    async def get_stock_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Alternative to start/end dates (e.g., '1y', '6mo')
            interval: Data interval ('1d', '1wk', '1mo', etc.)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        ...

    async def get_realtime_data(self, symbol: str) -> dict[str, Any] | None:
        """
        Get real-time stock data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with current price, change, volume, etc. or None if unavailable
        """
        ...

    async def get_stock_info(self, symbol: str) -> dict[str, Any]:
        """
        Get detailed stock information and fundamentals.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company info, financials, and market data
        """
        ...

    async def get_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Get news articles for a stock.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles to return

        Returns:
            DataFrame with news articles
        """
        ...

    async def get_earnings(self, symbol: str) -> dict[str, Any]:
        """
        Get earnings information for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with earnings data and dates
        """
        ...

    async def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendations for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with analyst recommendations
        """
        ...

    async def is_market_open(self) -> bool:
        """
        Check if the stock market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        ...

    async def is_etf(self, symbol: str) -> bool:
        """
        Check if a symbol represents an ETF.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if symbol is an ETF, False otherwise
        """
        ...


@runtime_checkable
class IStockScreener(Protocol):
    """
    Interface for stock screening and recommendation operations.

    This interface defines the contract for generating stock recommendations
    based on various technical and fundamental criteria.
    """

    async def get_maverick_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get bullish Maverick stock recommendations.

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum combined score filter

        Returns:
            List of stock recommendations with technical analysis
        """
        ...

    async def get_maverick_bear_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get bearish Maverick stock recommendations.

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum score filter

        Returns:
            List of bear stock recommendations
        """
        ...

    async def get_trending_recommendations(
        self, limit: int = 20, min_momentum_score: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Get trending stock recommendations.

        Args:
            limit: Maximum number of recommendations
            min_momentum_score: Minimum momentum score filter

        Returns:
            List of trending stock recommendations
        """
        ...

    async def get_all_screening_recommendations(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get all screening recommendations in one call.

        Returns:
            Dictionary with all screening types and their recommendations
        """
        ...


class StockDataProviderBase(ABC):
    """
    Abstract base class for stock data providers.

    This class provides a foundation for implementing both IStockDataFetcher
    and IStockScreener interfaces, with common functionality and error handling.
    """

    @abstractmethod
    def _fetch_stock_data_from_source(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch stock data from the underlying data source.

        This method must be implemented by concrete providers to define
        how data is actually retrieved (e.g., from yfinance, Alpha Vantage, etc.)
        """
        pass

    def _validate_symbol(self, symbol: str) -> str:
        """
        Validate and normalize a stock symbol.

        Args:
            symbol: Raw stock symbol

        Returns:
            Normalized symbol (uppercase, stripped)

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("Symbol cannot be empty after normalization")

        return normalized

    def _validate_date_range(
        self, start_date: str | None, end_date: str | None
    ) -> tuple[str | None, str | None]:
        """
        Validate date range parameters.

        Args:
            start_date: Start date string
            end_date: End date string

        Returns:
            Tuple of validated dates

        Raises:
            ValueError: If date format is invalid
        """
        # Basic validation - can be extended with actual date parsing
        if start_date is not None and not isinstance(start_date, str):
            raise ValueError("start_date must be a string in YYYY-MM-DD format")

        if end_date is not None and not isinstance(end_date, str):
            raise ValueError("end_date must be a string in YYYY-MM-DD format")

        return start_date, end_date

    def _handle_provider_error(self, error: Exception, context: str) -> None:
        """
        Handle provider-specific errors with consistent logging.

        Args:
            error: The exception that occurred
            context: Context information for debugging
        """
        # This would integrate with the logging system
        # For now, we'll re-raise to maintain existing behavior
        raise error
