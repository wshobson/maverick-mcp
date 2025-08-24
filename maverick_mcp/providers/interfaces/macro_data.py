"""
Macroeconomic data provider interface.

This module defines the abstract interface for macroeconomic data operations,
including GDP, inflation, unemployment, and market sentiment indicators.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IMacroDataProvider(Protocol):
    """
    Interface for macroeconomic data operations.

    This interface defines the contract for retrieving economic indicators,
    market sentiment data, and related macroeconomic statistics.
    """

    async def get_gdp_growth_rate(self) -> dict[str, Any]:
        """
        Get GDP growth rate data.

        Returns:
            Dictionary with current and previous GDP growth rates
        """
        ...

    async def get_unemployment_rate(self) -> dict[str, Any]:
        """
        Get unemployment rate data.

        Returns:
            Dictionary with current and previous unemployment rates
        """
        ...

    async def get_inflation_rate(self) -> dict[str, Any]:
        """
        Get inflation rate data based on CPI.

        Returns:
            Dictionary with current and previous inflation rates and bounds
        """
        ...

    async def get_vix(self) -> float | None:
        """
        Get VIX (volatility index) data.

        Returns:
            Current VIX value or None if unavailable
        """
        ...

    async def get_sp500_performance(self) -> float:
        """
        Get S&P 500 performance over multiple timeframes.

        Returns:
            Weighted performance percentage
        """
        ...

    async def get_nasdaq_performance(self) -> float:
        """
        Get NASDAQ performance over multiple timeframes.

        Returns:
            Weighted performance percentage
        """
        ...

    async def get_sp500_momentum(self) -> float:
        """
        Get short-term S&P 500 momentum.

        Returns:
            Momentum percentage over short timeframes
        """
        ...

    async def get_nasdaq_momentum(self) -> float:
        """
        Get short-term NASDAQ momentum.

        Returns:
            Momentum percentage over short timeframes
        """
        ...

    async def get_usd_momentum(self) -> float:
        """
        Get USD momentum using broad dollar index.

        Returns:
            USD momentum percentage over short timeframes
        """
        ...

    async def get_macro_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive macroeconomic statistics.

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
        ...

    async def get_historical_data(self) -> dict[str, Any]:
        """
        Get historical data for all indicators.

        Returns:
            Dictionary with time series data for various indicators
        """
        ...


class MacroDataConfig:
    """
    Configuration class for macroeconomic data providers.

    This class encapsulates macro data-related configuration parameters
    to reduce coupling between providers and configuration sources.
    """

    def __init__(
        self,
        fred_api_key: str = "",
        window_days: int = 365,
        lookback_days: int = 30,
        request_timeout: int = 30,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        sentiment_weights: dict[str, float] | None = None,
        smoothing_factor: float = 0.8,
    ):
        """
        Initialize macro data configuration.

        Args:
            fred_api_key: API key for FRED (Federal Reserve Economic Data)
            window_days: Window for historical data bounds calculation
            lookback_days: Lookback period for momentum calculations
            request_timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl: Cache time-to-live in seconds
            sentiment_weights: Weights for sentiment score calculation
            smoothing_factor: Smoothing factor for sentiment score
        """
        self.fred_api_key = fred_api_key
        self.window_days = window_days
        self.lookback_days = lookback_days
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self.smoothing_factor = smoothing_factor

        # Default sentiment weights
        self.sentiment_weights = sentiment_weights or {
            # Short-term signals (60% total)
            "vix": 0.20,
            "sp500_momentum": 0.20,
            "nasdaq_momentum": 0.15,
            "usd_momentum": 0.05,
            # Macro signals (40% total)
            "inflation_rate": 0.15,
            "gdp_growth_rate": 0.15,
            "unemployment_rate": 0.10,
        }

    @property
    def has_fred_key(self) -> bool:
        """Check if FRED API key is configured."""
        return bool(self.fred_api_key.strip())

    def get_sentiment_weight(self, indicator: str) -> float:
        """Get sentiment weight for a specific indicator."""
        return self.sentiment_weights.get(indicator, 0.0)

    def get_total_sentiment_weight(self) -> float:
        """Get total sentiment weight (should sum to 1.0)."""
        return sum(self.sentiment_weights.values())


# FRED series IDs for common economic indicators
FRED_SERIES_IDS = {
    "gdp_growth_rate": "A191RL1Q225SBEA",
    "unemployment_rate": "UNRATE",
    "core_inflation": "CPILFESL",
    "sp500": "SP500",
    "nasdaq": "NASDAQ100",
    "vix": "VIXCLS",
    "usd_index": "DTWEXBGS",
}

# Default bounds for normalization
DEFAULT_INDICATOR_BOUNDS = {
    "vix": {"min": 10.0, "max": 50.0},
    "sp500_momentum": {"min": -15.0, "max": 15.0},
    "nasdaq_momentum": {"min": -20.0, "max": 20.0},
    "usd_momentum": {"min": -5.0, "max": 5.0},
    "inflation_rate": {"min": 0.0, "max": 10.0},
    "gdp_growth_rate": {"min": -2.0, "max": 6.0},
    "unemployment_rate": {"min": 2.0, "max": 10.0},
}
