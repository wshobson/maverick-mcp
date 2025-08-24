"""
Mock macro data provider implementation for testing.
"""

from datetime import datetime
from typing import Any


class MockMacroDataProvider:
    """
    Mock implementation of IMacroDataProvider for testing.
    """

    def __init__(self, test_data: dict[str, Any] | None = None):
        """
        Initialize the mock macro data provider.

        Args:
            test_data: Optional test data to return
        """
        self._test_data = test_data or {}
        self._call_log: list[dict[str, Any]] = []

    async def get_gdp_growth_rate(self) -> dict[str, Any]:
        """Get mock GDP growth rate."""
        self._log_call("get_gdp_growth_rate", {})

        if "gdp_growth_rate" in self._test_data:
            return self._test_data["gdp_growth_rate"]

        return {
            "current": 2.5,
            "previous": 2.3,
        }

    async def get_unemployment_rate(self) -> dict[str, Any]:
        """Get mock unemployment rate."""
        self._log_call("get_unemployment_rate", {})

        if "unemployment_rate" in self._test_data:
            return self._test_data["unemployment_rate"]

        return {
            "current": 3.8,
            "previous": 3.9,
        }

    async def get_inflation_rate(self) -> dict[str, Any]:
        """Get mock inflation rate."""
        self._log_call("get_inflation_rate", {})

        if "inflation_rate" in self._test_data:
            return self._test_data["inflation_rate"]

        return {
            "current": 3.2,
            "previous": 3.4,
            "bounds": (1.5, 6.8),
        }

    async def get_vix(self) -> float | None:
        """Get mock VIX data."""
        self._log_call("get_vix", {})

        if "vix" in self._test_data:
            return self._test_data["vix"]

        return 18.5

    async def get_sp500_performance(self) -> float:
        """Get mock S&P 500 performance."""
        self._log_call("get_sp500_performance", {})

        if "sp500_performance" in self._test_data:
            return self._test_data["sp500_performance"]

        return 1.25

    async def get_nasdaq_performance(self) -> float:
        """Get mock NASDAQ performance."""
        self._log_call("get_nasdaq_performance", {})

        if "nasdaq_performance" in self._test_data:
            return self._test_data["nasdaq_performance"]

        return 1.85

    async def get_sp500_momentum(self) -> float:
        """Get mock S&P 500 momentum."""
        self._log_call("get_sp500_momentum", {})

        if "sp500_momentum" in self._test_data:
            return self._test_data["sp500_momentum"]

        return 0.75

    async def get_nasdaq_momentum(self) -> float:
        """Get mock NASDAQ momentum."""
        self._log_call("get_nasdaq_momentum", {})

        if "nasdaq_momentum" in self._test_data:
            return self._test_data["nasdaq_momentum"]

        return 1.15

    async def get_usd_momentum(self) -> float:
        """Get mock USD momentum."""
        self._log_call("get_usd_momentum", {})

        if "usd_momentum" in self._test_data:
            return self._test_data["usd_momentum"]

        return -0.35

    async def get_macro_statistics(self) -> dict[str, Any]:
        """Get mock comprehensive macro statistics."""
        self._log_call("get_macro_statistics", {})

        if "macro_statistics" in self._test_data:
            return self._test_data["macro_statistics"]

        return {
            "gdp_growth_rate": 2.5,
            "gdp_growth_rate_previous": 2.3,
            "unemployment_rate": 3.8,
            "unemployment_rate_previous": 3.9,
            "inflation_rate": 3.2,
            "inflation_rate_previous": 3.4,
            "sp500_performance": 1.25,
            "nasdaq_performance": 1.85,
            "vix": 18.5,
            "sentiment_score": 65.5,
            "historical_data": self._generate_mock_historical_data(),
        }

    async def get_historical_data(self) -> dict[str, Any]:
        """Get mock historical data."""
        self._log_call("get_historical_data", {})

        if "historical_data" in self._test_data:
            return self._test_data["historical_data"]

        return self._generate_mock_historical_data()

    def _generate_mock_historical_data(self) -> dict[str, Any]:
        """Generate mock historical data for indicators."""
        return {
            "sp500_performance": [1.0, 1.1, 1.2, 1.25, 1.3],
            "nasdaq_performance": [1.5, 1.6, 1.7, 1.8, 1.85],
            "vix": [20.0, 19.5, 18.8, 18.2, 18.5],
            "gdp_growth_rate": [2.1, 2.2, 2.3, 2.4, 2.5],
            "unemployment_rate": [4.2, 4.1, 4.0, 3.9, 3.8],
            "inflation_rate": [3.8, 3.6, 3.5, 3.4, 3.2],
        }

    # Testing utilities

    def _log_call(self, method: str, args: dict[str, Any]) -> None:
        """Log method calls for testing verification."""
        self._call_log.append(
            {
                "method": method,
                "args": args,
                "timestamp": datetime.now(),
            }
        )

    def get_call_log(self) -> list[dict[str, Any]]:
        """Get the log of method calls."""
        return self._call_log.copy()

    def clear_call_log(self) -> None:
        """Clear the method call log."""
        self._call_log.clear()

    def set_test_data(self, key: str, data: Any) -> None:
        """Set test data for a specific key."""
        self._test_data[key] = data

    def clear_test_data(self) -> None:
        """Clear all test data."""
        self._test_data.clear()
