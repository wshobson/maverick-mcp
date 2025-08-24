"""
Mock stock data provider implementations for testing.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


class MockStockDataFetcher:
    """
    Mock implementation of IStockDataFetcher for testing.

    This implementation provides predictable test data without requiring
    external API calls or database access.
    """

    def __init__(self, test_data: dict[str, pd.DataFrame] | None = None):
        """
        Initialize the mock stock data fetcher.

        Args:
            test_data: Optional dictionary mapping symbols to DataFrames
        """
        self._test_data = test_data or {}
        self._call_log: list[dict[str, Any]] = []

    async def get_stock_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get mock stock data."""
        self._log_call(
            "get_stock_data",
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "period": period,
                "interval": interval,
                "use_cache": use_cache,
            },
        )

        symbol = symbol.upper()

        # Return test data if available
        if symbol in self._test_data:
            df = self._test_data[symbol].copy()

            # Filter by date range if specified
            if start_date or end_date:
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

            return df

        # Generate synthetic data
        return self._generate_synthetic_data(symbol, start_date, end_date, period)

    async def get_realtime_data(self, symbol: str) -> dict[str, Any] | None:
        """Get mock real-time stock data."""
        self._log_call("get_realtime_data", {"symbol": symbol})

        # Return predictable mock data
        return {
            "symbol": symbol.upper(),
            "price": 150.25,
            "change": 2.15,
            "change_percent": 1.45,
            "volume": 1234567,
            "timestamp": datetime.now(),
            "timestamp_display": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_real_time": False,
        }

    async def get_stock_info(self, symbol: str) -> dict[str, Any]:
        """Get mock stock information."""
        self._log_call("get_stock_info", {"symbol": symbol})

        return {
            "symbol": symbol.upper(),
            "longName": f"{symbol.upper()} Inc.",
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 1000000000,
            "previousClose": 148.10,
            "beta": 1.2,
            "dividendYield": 0.02,
            "peRatio": 25.5,
        }

    async def get_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Get mock news data."""
        self._log_call("get_news", {"symbol": symbol, "limit": limit})

        # Generate mock news data
        news_data = []
        for i in range(min(limit, 5)):  # Generate up to 5 mock articles
            news_data.append(
                {
                    "title": f"Mock news article {i + 1} for {symbol}",
                    "publisher": f"Mock Publisher {i + 1}",
                    "link": f"https://example.com/news/{symbol.lower()}/{i + 1}",
                    "providerPublishTime": datetime.now() - timedelta(hours=i),
                    "type": "STORY",
                }
            )

        return pd.DataFrame(news_data)

    async def get_earnings(self, symbol: str) -> dict[str, Any]:
        """Get mock earnings data."""
        self._log_call("get_earnings", {"symbol": symbol})

        return {
            "earnings": {
                "2023": 5.25,
                "2022": 4.80,
                "2021": 4.35,
            },
            "earnings_dates": {
                "next_date": "2024-01-25",
                "eps_estimate": 1.35,
            },
            "earnings_trend": {
                "current_quarter": 1.30,
                "next_quarter": 1.35,
                "current_year": 5.40,
                "next_year": 5.85,
            },
        }

    async def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get mock analyst recommendations."""
        self._log_call("get_recommendations", {"symbol": symbol})

        recommendations_data = [
            {
                "firm": "Mock Investment Bank",
                "toGrade": "Buy",
                "fromGrade": "Hold",
                "action": "up",
            },
            {
                "firm": "Another Mock Firm",
                "toGrade": "Hold",
                "fromGrade": "Hold",
                "action": "main",
            },
        ]

        return pd.DataFrame(recommendations_data)

    async def is_market_open(self) -> bool:
        """Check if market is open (mock)."""
        self._log_call("is_market_open", {})

        # Return True for testing by default
        return True

    async def is_etf(self, symbol: str) -> bool:
        """Check if symbol is an ETF (mock)."""
        self._log_call("is_etf", {"symbol": symbol})

        # Mock ETF detection based on common ETF symbols
        etf_symbols = {"SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "XLK", "XLF"}
        return symbol.upper() in etf_symbols

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic stock data for testing."""

        # Determine date range
        if period:
            days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "1y": 365}.get(period, 30)
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
        else:
            end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
            start_dt = (
                pd.to_datetime(start_date)
                if start_date
                else end_dt - timedelta(days=30)
            )

        # Generate date range (business days only)
        dates = pd.bdate_range(start=start_dt, end=end_dt)

        if len(dates) == 0:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(
                columns=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Dividends",
                    "Stock Splits",
                ]
            )

        # Generate synthetic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol

        base_price = 100.0
        returns = np.random.normal(
            0.001, 0.02, len(dates)
        )  # 0.1% mean return, 2% volatility

        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV data
        data = []
        for _i, (_date, close_price) in enumerate(zip(dates, prices, strict=False)):
            # Generate Open, High, Low based on Close
            volatility = close_price * 0.02  # 2% intraday volatility

            open_price = close_price + np.random.normal(0, volatility * 0.5)
            high_price = max(open_price, close_price) + abs(
                np.random.normal(0, volatility * 0.3)
            )
            low_price = min(open_price, close_price) - abs(
                np.random.normal(0, volatility * 0.3)
            )

            # Ensure High >= Low and prices are positive
            high_price = max(high_price, low_price + 0.01, 0.01)
            low_price = max(low_price, 0.01)

            volume = int(
                np.random.lognormal(15, 0.5)
            )  # Log-normal distribution for volume

            data.append(
                {
                    "Open": round(open_price, 2),
                    "High": round(high_price, 2),
                    "Low": round(low_price, 2),
                    "Close": round(close_price, 2),
                    "Volume": volume,
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                }
            )

        df = pd.DataFrame(data, index=dates)
        df.index.name = "Date"

        return df

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

    def set_test_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Set test data for a specific symbol."""
        self._test_data[symbol.upper()] = data

    def clear_test_data(self) -> None:
        """Clear all test data."""
        self._test_data.clear()


class MockStockScreener:
    """
    Mock implementation of IStockScreener for testing.
    """

    def __init__(
        self, test_recommendations: dict[str, list[dict[str, Any]]] | None = None
    ):
        """
        Initialize the mock stock screener.

        Args:
            test_recommendations: Optional dictionary of test screening results
        """
        self._test_recommendations = test_recommendations or {}
        self._call_log: list[dict[str, Any]] = []

    async def get_maverick_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """Get mock maverick recommendations."""
        self._log_call(
            "get_maverick_recommendations", {"limit": limit, "min_score": min_score}
        )

        if "maverick" in self._test_recommendations:
            results = self._test_recommendations["maverick"]
        else:
            results = self._generate_mock_maverick_recommendations()

        # Apply filters
        if min_score:
            results = [r for r in results if r.get("combined_score", 0) >= min_score]

        return results[:limit]

    async def get_maverick_bear_recommendations(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """Get mock maverick bear recommendations."""
        self._log_call(
            "get_maverick_bear_recommendations",
            {"limit": limit, "min_score": min_score},
        )

        if "bear" in self._test_recommendations:
            results = self._test_recommendations["bear"]
        else:
            results = self._generate_mock_bear_recommendations()

        # Apply filters
        if min_score:
            results = [r for r in results if r.get("score", 0) >= min_score]

        return results[:limit]

    async def get_trending_recommendations(
        self, limit: int = 20, min_momentum_score: float | None = None
    ) -> list[dict[str, Any]]:
        """Get mock trending recommendations."""
        self._log_call(
            "get_trending_recommendations",
            {"limit": limit, "min_momentum_score": min_momentum_score},
        )

        if "trending" in self._test_recommendations:
            results = self._test_recommendations["trending"]
        else:
            results = self._generate_mock_trending_recommendations()

        # Apply filters
        if min_momentum_score:
            results = [
                r for r in results if r.get("momentum_score", 0) >= min_momentum_score
            ]

        return results[:limit]

    async def get_all_screening_recommendations(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get all mock screening recommendations."""
        self._log_call("get_all_screening_recommendations", {})

        return {
            "maverick_stocks": await self.get_maverick_recommendations(),
            "maverick_bear_stocks": await self.get_maverick_bear_recommendations(),
            "supply_demand_breakouts": await self.get_trending_recommendations(),
        }

    def _generate_mock_maverick_recommendations(self) -> list[dict[str, Any]]:
        """Generate mock maverick recommendations."""
        return [
            {
                "symbol": "AAPL",
                "combined_score": 95,
                "momentum_score": 92,
                "pattern": "Cup with Handle",
                "consolidation": "yes",
                "squeeze": "firing",
                "recommendation_type": "maverick_bullish",
                "reason": "Exceptional combined score with outstanding relative strength",
            },
            {
                "symbol": "MSFT",
                "combined_score": 88,
                "momentum_score": 85,
                "pattern": "Flat Base",
                "consolidation": "no",
                "squeeze": "setup",
                "recommendation_type": "maverick_bullish",
                "reason": "Strong combined score with strong relative strength",
            },
        ]

    def _generate_mock_bear_recommendations(self) -> list[dict[str, Any]]:
        """Generate mock bear recommendations."""
        return [
            {
                "symbol": "BEAR1",
                "score": 92,
                "momentum_score": 25,
                "rsi_14": 28,
                "atr_contraction": True,
                "big_down_vol": True,
                "recommendation_type": "maverick_bearish",
                "reason": "Exceptional bear score with weak relative strength, oversold RSI",
            },
            {
                "symbol": "BEAR2",
                "score": 85,
                "momentum_score": 30,
                "rsi_14": 35,
                "atr_contraction": False,
                "big_down_vol": True,
                "recommendation_type": "maverick_bearish",
                "reason": "Strong bear score with weak relative strength",
            },
        ]

    def _generate_mock_trending_recommendations(self) -> list[dict[str, Any]]:
        """Generate mock trending recommendations."""
        return [
            {
                "symbol": "TREND1",
                "momentum_score": 95,
                "close": 150.25,
                "sma_50": 145.50,
                "sma_150": 140.25,
                "sma_200": 135.75,
                "pattern": "Breakout",
                "recommendation_type": "trending_stage2",
                "reason": "Uptrend with exceptional momentum strength",
            },
            {
                "symbol": "TREND2",
                "momentum_score": 88,
                "close": 85.30,
                "sma_50": 82.15,
                "sma_150": 79.80,
                "sma_200": 76.45,
                "pattern": "Higher Lows",
                "recommendation_type": "trending_stage2",
                "reason": "Uptrend with strong momentum strength",
            },
        ]

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

    def set_test_recommendations(
        self, screening_type: str, recommendations: list[dict[str, Any]]
    ) -> None:
        """Set test recommendations for a specific screening type."""
        self._test_recommendations[screening_type] = recommendations

    def clear_test_recommendations(self) -> None:
        """Clear all test recommendations."""
        self._test_recommendations.clear()
