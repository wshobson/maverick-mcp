"""
Market data provider adapter.

This module provides adapters that make the existing MarketDataProvider
compatible with the new IMarketDataProvider interface.
"""

import asyncio
import logging
from typing import Any

from maverick_mcp.providers.interfaces.market_data import (
    IMarketDataProvider,
    MarketDataConfig,
)
from maverick_mcp.providers.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class MarketDataAdapter(IMarketDataProvider):
    """
    Adapter that makes the existing MarketDataProvider compatible with IMarketDataProvider interface.

    This adapter wraps the existing provider and exposes it through the new
    interface contracts, enabling gradual migration to the new architecture.
    """

    def __init__(self, config: MarketDataConfig | None = None):
        """
        Initialize the market data adapter.

        Args:
            config: Market data configuration (optional)
        """
        self._config = config
        self._provider = MarketDataProvider()

        logger.debug("MarketDataAdapter initialized")

    async def get_market_summary(self) -> dict[str, Any]:
        """
        Get a summary of major market indices (async wrapper).

        Returns:
            Dictionary with market index data including prices and changes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_market_summary)

    async def get_top_gainers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top gaining stocks in the market (async wrapper).

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data for top gainers
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_top_gainers, limit)

    async def get_top_losers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top losing stocks in the market (async wrapper).

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data for top losers
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_top_losers, limit)

    async def get_most_active(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get most active stocks by volume (async wrapper).

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data for most active stocks
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_most_active, limit)

    async def get_sector_performance(self) -> dict[str, float]:
        """
        Get sector performance data (async wrapper).

        Returns:
            Dictionary mapping sector names to performance percentages
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_sector_performance)

    async def get_earnings_calendar(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Get upcoming earnings announcements (async wrapper).

        Args:
            days: Number of days to look ahead

        Returns:
            List of dictionaries with earnings announcement data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._provider.get_earnings_calendar, days
        )

    async def get_market_overview(self) -> dict[str, Any]:
        """
        Get comprehensive market overview (async wrapper).

        Returns:
            Dictionary with comprehensive market data including:
            - market_summary: Index data
            - top_gainers: Daily gainers
            - top_losers: Daily losers
            - sector_performance: Sector data
            - timestamp: Data timestamp
        """
        # Use the existing async method if available, otherwise wrap the sync version
        if hasattr(self._provider, "get_market_overview_async"):
            return await self._provider.get_market_overview_async()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._provider.get_market_overview)

    def get_sync_provider(self) -> MarketDataProvider:
        """
        Get the underlying synchronous provider for backward compatibility.

        Returns:
            The wrapped MarketDataProvider instance
        """
        return self._provider
