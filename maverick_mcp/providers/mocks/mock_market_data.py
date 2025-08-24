"""
Mock market data provider implementation for testing.
"""

from datetime import UTC, datetime
from typing import Any


class MockMarketDataProvider:
    """
    Mock implementation of IMarketDataProvider for testing.
    """

    def __init__(self, test_data: dict[str, Any] | None = None):
        """
        Initialize the mock market data provider.

        Args:
            test_data: Optional test data to return
        """
        self._test_data = test_data or {}
        self._call_log: list[dict[str, Any]] = []

    async def get_market_summary(self) -> dict[str, Any]:
        """Get mock market summary."""
        self._log_call("get_market_summary", {})

        if "market_summary" in self._test_data:
            return self._test_data["market_summary"]

        return {
            "^GSPC": {
                "name": "S&P 500",
                "symbol": "^GSPC",
                "price": 4500.25,
                "change": 15.75,
                "change_percent": 0.35,
            },
            "^DJI": {
                "name": "Dow Jones",
                "symbol": "^DJI",
                "price": 35000.50,
                "change": -50.25,
                "change_percent": -0.14,
            },
            "^IXIC": {
                "name": "NASDAQ",
                "symbol": "^IXIC",
                "price": 14000.75,
                "change": 25.30,
                "change_percent": 0.18,
            },
        }

    async def get_top_gainers(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get mock top gainers."""
        self._log_call("get_top_gainers", {"limit": limit})

        if "top_gainers" in self._test_data:
            return self._test_data["top_gainers"][:limit]

        gainers = [
            {
                "symbol": "GAINER1",
                "price": 150.25,
                "change": 15.50,
                "change_percent": 11.50,
                "volume": 2500000,
            },
            {
                "symbol": "GAINER2",
                "price": 85.75,
                "change": 8.25,
                "change_percent": 10.65,
                "volume": 1800000,
            },
            {
                "symbol": "GAINER3",
                "price": 45.30,
                "change": 4.15,
                "change_percent": 10.08,
                "volume": 3200000,
            },
        ]

        return gainers[:limit]

    async def get_top_losers(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get mock top losers."""
        self._log_call("get_top_losers", {"limit": limit})

        if "top_losers" in self._test_data:
            return self._test_data["top_losers"][:limit]

        losers = [
            {
                "symbol": "LOSER1",
                "price": 25.50,
                "change": -5.75,
                "change_percent": -18.38,
                "volume": 4500000,
            },
            {
                "symbol": "LOSER2",
                "price": 67.20,
                "change": -12.80,
                "change_percent": -16.00,
                "volume": 2100000,
            },
            {
                "symbol": "LOSER3",
                "price": 120.45,
                "change": -18.55,
                "change_percent": -13.35,
                "volume": 1600000,
            },
        ]

        return losers[:limit]

    async def get_most_active(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get mock most active stocks."""
        self._log_call("get_most_active", {"limit": limit})

        if "most_active" in self._test_data:
            return self._test_data["most_active"][:limit]

        active = [
            {
                "symbol": "ACTIVE1",
                "price": 200.75,
                "change": 5.25,
                "change_percent": 2.68,
                "volume": 15000000,
            },
            {
                "symbol": "ACTIVE2",
                "price": 95.30,
                "change": -2.15,
                "change_percent": -2.21,
                "volume": 12500000,
            },
            {
                "symbol": "ACTIVE3",
                "price": 155.80,
                "change": 1.85,
                "change_percent": 1.20,
                "volume": 11200000,
            },
        ]

        return active[:limit]

    async def get_sector_performance(self) -> dict[str, float]:
        """Get mock sector performance."""
        self._log_call("get_sector_performance", {})

        if "sector_performance" in self._test_data:
            return self._test_data["sector_performance"]

        return {
            "Technology": 1.25,
            "Healthcare": 0.85,
            "Financials": -0.45,
            "Consumer Discretionary": 0.65,
            "Industrials": 0.35,
            "Energy": -1.15,
            "Utilities": 0.15,
            "Materials": -0.25,
            "Consumer Staples": 0.55,
            "Real Estate": -0.75,
            "Communication Services": 0.95,
        }

    async def get_earnings_calendar(self, days: int = 7) -> list[dict[str, Any]]:
        """Get mock earnings calendar."""
        self._log_call("get_earnings_calendar", {"days": days})

        if "earnings_calendar" in self._test_data:
            return self._test_data["earnings_calendar"]

        base_date = datetime.now(UTC).date()

        return [
            {
                "ticker": "EARN1",
                "name": "Earnings Corp 1",
                "earnings_date": (base_date + datetime.timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                "eps_estimate": 1.25,
            },
            {
                "ticker": "EARN2",
                "name": "Earnings Corp 2",
                "earnings_date": (base_date + datetime.timedelta(days=3)).strftime(
                    "%Y-%m-%d"
                ),
                "eps_estimate": 0.85,
            },
            {
                "ticker": "EARN3",
                "name": "Earnings Corp 3",
                "earnings_date": (base_date + datetime.timedelta(days=5)).strftime(
                    "%Y-%m-%d"
                ),
                "eps_estimate": 2.15,
            },
        ]

    async def get_market_overview(self) -> dict[str, Any]:
        """Get mock comprehensive market overview."""
        self._log_call("get_market_overview", {})

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "market_summary": await self.get_market_summary(),
            "top_gainers": await self.get_top_gainers(5),
            "top_losers": await self.get_top_losers(5),
            "sector_performance": await self.get_sector_performance(),
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
