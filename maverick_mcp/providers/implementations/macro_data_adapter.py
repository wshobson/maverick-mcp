"""
Macro data provider adapter.

This module provides adapters that make the existing MacroDataProvider
compatible with the new IMacroDataProvider interface.
"""

import asyncio
import logging
from typing import Any

from maverick_mcp.providers.interfaces.macro_data import (
    IMacroDataProvider,
    MacroDataConfig,
)
from maverick_mcp.providers.macro_data import MacroDataProvider

logger = logging.getLogger(__name__)


class MacroDataAdapter(IMacroDataProvider):
    """
    Adapter that makes the existing MacroDataProvider compatible with IMacroDataProvider interface.

    This adapter wraps the existing provider and exposes it through the new
    interface contracts, enabling gradual migration to the new architecture.
    """

    def __init__(self, config: MacroDataConfig | None = None):
        """
        Initialize the macro data adapter.

        Args:
            config: Macro data configuration (optional)
        """
        self._config = config

        # Initialize the existing provider with configuration
        window_days = config.window_days if config else 365
        self._provider = MacroDataProvider(window_days=window_days)

        logger.debug("MacroDataAdapter initialized")

    async def get_gdp_growth_rate(self) -> dict[str, Any]:
        """
        Get GDP growth rate data (async wrapper).

        Returns:
            Dictionary with current and previous GDP growth rates
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_gdp_growth_rate)

    async def get_unemployment_rate(self) -> dict[str, Any]:
        """
        Get unemployment rate data (async wrapper).

        Returns:
            Dictionary with current and previous unemployment rates
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_unemployment_rate)

    async def get_inflation_rate(self) -> dict[str, Any]:
        """
        Get inflation rate data based on CPI (async wrapper).

        Returns:
            Dictionary with current and previous inflation rates and bounds
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_inflation_rate)

    async def get_vix(self) -> float | None:
        """
        Get VIX (volatility index) data (async wrapper).

        Returns:
            Current VIX value or None if unavailable
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_vix)

    async def get_sp500_performance(self) -> float:
        """
        Get S&P 500 performance over multiple timeframes (async wrapper).

        Returns:
            Weighted performance percentage
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_sp500_performance)

    async def get_nasdaq_performance(self) -> float:
        """
        Get NASDAQ performance over multiple timeframes (async wrapper).

        Returns:
            Weighted performance percentage
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_nasdaq_performance)

    async def get_sp500_momentum(self) -> float:
        """
        Get short-term S&P 500 momentum (async wrapper).

        Returns:
            Momentum percentage over short timeframes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_sp500_momentum)

    async def get_nasdaq_momentum(self) -> float:
        """
        Get short-term NASDAQ momentum (async wrapper).

        Returns:
            Momentum percentage over short timeframes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_nasdaq_momentum)

    async def get_usd_momentum(self) -> float:
        """
        Get USD momentum using broad dollar index (async wrapper).

        Returns:
            USD momentum percentage over short timeframes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_usd_momentum)

    async def get_macro_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive macroeconomic statistics (async wrapper).

        Returns:
            Dictionary with all macro indicators including:
            - gdp_growth_rate: Current and previous GDP growth
            - unemployment_rate: Current and previous unemployment
            - inflation_rate: Current and previous inflation
            - sp500_performance: S&P 500 performance
            - nasdaq_performance: NASDAQ performance
            - vix: Volatility index
            - sentiment_score: Computed sentiment score
            - historical_data: Time series data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_macro_statistics)

    async def get_historical_data(self) -> dict[str, Any]:
        """
        Get historical data for all indicators (async wrapper).

        Returns:
            Dictionary with time series data for various indicators
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._provider.get_historical_data)

    def get_sync_provider(self) -> MacroDataProvider:
        """
        Get the underlying synchronous provider for backward compatibility.

        Returns:
            The wrapped MacroDataProvider instance
        """
        return self._provider
