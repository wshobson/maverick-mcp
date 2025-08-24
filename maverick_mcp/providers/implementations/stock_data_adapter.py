"""
Stock data provider adapter.

This module provides adapters that make the existing StockDataProvider
compatible with the new interface-based architecture while maintaining
all existing functionality.
"""

import asyncio
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from maverick_mcp.providers.interfaces.cache import ICacheManager
from maverick_mcp.providers.interfaces.config import IConfigurationProvider
from maverick_mcp.providers.interfaces.persistence import IDataPersistence
from maverick_mcp.providers.interfaces.stock_data import (
    IStockDataFetcher,
    IStockScreener,
)
from maverick_mcp.providers.stock_data import StockDataProvider

logger = logging.getLogger(__name__)


class StockDataAdapter(IStockDataFetcher, IStockScreener):
    """
    Adapter that makes the existing StockDataProvider compatible with new interfaces.

    This adapter wraps the existing provider and exposes it through the new
    interface contracts, enabling gradual migration to the new architecture.
    """

    def __init__(
        self,
        cache_manager: ICacheManager | None = None,
        persistence: IDataPersistence | None = None,
        config: IConfigurationProvider | None = None,
        db_session: Session | None = None,
    ):
        """
        Initialize the stock data adapter.

        Args:
            cache_manager: Cache manager for data caching
            persistence: Persistence layer for database operations
            config: Configuration provider
            db_session: Optional database session for dependency injection
        """
        self._cache_manager = cache_manager
        self._persistence = persistence
        self._config = config
        self._db_session = db_session

        # Initialize the existing provider
        self._provider = StockDataProvider(db_session=db_session)

        logger.debug("StockDataAdapter initialized")

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
        Fetch historical stock data (async wrapper).

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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._provider.get_stock_data,
            symbol,
            start_date,
            end_date,
            period,
            interval,
            use_cache,
        )

    async def get_realtime_data(self, symbol: str) -> dict[str, Any] | None:
        """
        Get real-time stock data (async wrapper).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with current price, change, volume, etc. or None if unavailable
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_realtime_data, symbol
        )

    async def get_stock_info(self, symbol: str) -> dict[str, Any]:
        """
        Get detailed stock information and fundamentals (async wrapper).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company info, financials, and market data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_stock_info, symbol)

    async def get_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Get news articles for a stock (async wrapper).

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles to return

        Returns:
            DataFrame with news articles
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_news, symbol, limit)

    async def get_earnings(self, symbol: str) -> dict[str, Any]:
        """
        Get earnings information for a stock (async wrapper).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with earnings data and dates
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_earnings, symbol)

    async def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get analyst recommendations for a stock (async wrapper).

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with analyst recommendations
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_recommendations, symbol
        )

    async def is_market_open(self) -> bool:
        """
        Check if the stock market is currently open (async wrapper).

        Returns:
            True if market is open, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.is_market_open)

    async def is_etf(self, symbol: str) -> bool:
        """
        Check if a symbol represents an ETF (async wrapper).

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if symbol is an ETF, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.is_etf, symbol)

    # IStockScreener implementation
    async def get_maverick_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get bullish Maverick stock recommendations (async wrapper).

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum combined score filter

        Returns:
            List of stock recommendations with technical analysis
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_maverick_recommendations, limit, min_score
        )

    async def get_maverick_bear_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get bearish Maverick stock recommendations (async wrapper).

        Args:
            limit: Maximum number of recommendations
            min_score: Minimum score filter

        Returns:
            List of bear stock recommendations
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_maverick_bear_recommendations, limit, min_score
        )

    async def get_trending_recommendations(
        self, limit: int = 20, min_momentum_score: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Get trending stock recommendations (async wrapper).

        Args:
            limit: Maximum number of recommendations
            min_momentum_score: Minimum momentum score filter

        Returns:
            List of trending stock recommendations
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._provider.get_supply_demand_breakout_recommendations,
            limit,
            min_momentum_score,
        )

    async def get_all_screening_recommendations(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get all screening recommendations in one call (async wrapper).

        Returns:
            Dictionary with all screening types and their recommendations
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_all_screening_recommendations
        )

    # Additional methods to expose provider functionality
    def get_sync_provider(self) -> StockDataProvider:
        """
        Get the underlying synchronous provider for backward compatibility.

        Returns:
            The wrapped StockDataProvider instance
        """
        return self._provider

    async def get_all_realtime_data(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get real-time data for multiple symbols (async wrapper).

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to their real-time data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_all_realtime_data, symbols
        )
