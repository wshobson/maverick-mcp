"""
Market data provider interface.

This module defines the abstract interface for market-wide data operations,
including market indices, gainers/losers, sector performance, and earnings calendar.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IMarketDataProvider(Protocol):
    """
    Interface for market-wide data operations.

    This interface defines the contract for retrieving market overview data,
    including indices, top movers, sector performance, and earnings calendar.
    """

    async def get_market_summary(self) -> dict[str, Any]:
        """
        Get a summary of major market indices.

        Returns:
            Dictionary with market index data including prices and changes
        """
        ...

    async def get_top_gainers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top gaining stocks in the market.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data for top gainers
        """
        ...

    async def get_top_losers(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get top losing stocks in the market.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data for top losers
        """
        ...

    async def get_most_active(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get most active stocks by volume.

        Args:
            limit: Maximum number of stocks to return

        Returns:
            List of dictionaries with stock data for most active stocks
        """
        ...

    async def get_sector_performance(self) -> dict[str, float]:
        """
        Get sector performance data.

        Returns:
            Dictionary mapping sector names to performance percentages
        """
        ...

    async def get_earnings_calendar(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Get upcoming earnings announcements.

        Args:
            days: Number of days to look ahead

        Returns:
            List of dictionaries with earnings announcement data
        """
        ...

    async def get_market_overview(self) -> dict[str, Any]:
        """
        Get comprehensive market overview including summary, gainers, losers, and sectors.

        Returns:
            Dictionary with comprehensive market data including:
            - market_summary: Index data
            - top_gainers: Daily gainers
            - top_losers: Daily losers
            - sector_performance: Sector data
            - timestamp: Data timestamp
        """
        ...


class MarketDataConfig:
    """
    Configuration class for market data providers.

    This class encapsulates market data-related configuration parameters
    to reduce coupling between providers and configuration sources.
    """

    def __init__(
        self,
        external_api_key: str = "",
        tiingo_api_key: str = "",
        request_timeout: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
        default_limit: int = 10,
        use_fallback_providers: bool = True,
    ):
        """
        Initialize market data configuration.

        Args:
            external_api_key: API key for External API service
            tiingo_api_key: API key for Tiingo service
            request_timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_delay: Delay between requests in seconds
            default_limit: Default number of results to return
            use_fallback_providers: Whether to use fallback data sources
        """
        self.external_api_key = external_api_key
        self.tiingo_api_key = tiingo_api_key
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.default_limit = default_limit
        self.use_fallback_providers = use_fallback_providers

    @property
    def has_external_api_key(self) -> bool:
        """Check if External API key is configured."""
        return bool(self.external_api_key.strip())

    @property
    def has_tiingo_key(self) -> bool:
        """Check if Tiingo API key is configured."""
        return bool(self.tiingo_api_key.strip())


# Market data constants that can be used by implementations
MARKET_INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y Treasury",
}

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
