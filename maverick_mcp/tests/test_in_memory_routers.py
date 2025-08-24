"""
In-memory tests for domain-specific routers using FastMCP patterns.

Tests individual router functionality in isolation using FastMCP's
router mounting and in-memory testing capabilities.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client, FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from maverick_mcp.api.routers.data import data_router
from maverick_mcp.api.routers.portfolio import portfolio_router
from maverick_mcp.api.routers.screening import screening_router
from maverick_mcp.api.routers.technical import technical_router
from maverick_mcp.data.models import (
    Base,
    MaverickStocks,
    Stock,
    SupplyDemandBreakoutStocks,
)


@pytest.fixture
def test_server():
    """Create a test server with only specific routers mounted."""
    test_mcp: FastMCP = FastMCP("TestMaverick-MCP")
    return test_mcp


@pytest.fixture
def screening_db():
    """Create test database with screening data."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Add test stocks
        stocks = [
            Stock(
                ticker_symbol="AAPL",
                company_name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
            ),
            Stock(
                ticker_symbol="MSFT",
                company_name="Microsoft Corp.",
                sector="Technology",
                industry="Software",
            ),
            Stock(
                ticker_symbol="GOOGL",
                company_name="Alphabet Inc.",
                sector="Technology",
                industry="Internet",
            ),
            Stock(
                ticker_symbol="AMZN",
                company_name="Amazon.com Inc.",
                sector="Consumer Cyclical",
                industry="Internet Retail",
            ),
            Stock(
                ticker_symbol="TSLA",
                company_name="Tesla Inc.",
                sector="Consumer Cyclical",
                industry="Auto Manufacturers",
            ),
        ]
        session.add_all(stocks)
        session.commit()

        # Add Maverick screening results
        maverick_stocks = [
            MaverickStocks(
                id=1,
                stock="AAPL",
                close=150.0,
                open=148.0,
                high=152.0,
                low=147.0,
                volume=10000000,
                combined_score=92,
                momentum_score=88,
                adr_pct=2.5,
                atr=3.2,
                pat="Cup and Handle",
                sqz="Yes",
                consolidation="trending",
                entry="151.50",
                compression_score=85,
                pattern_detected=1,
                ema_21=149.0,
                sma_50=148.0,
                sma_150=145.0,
                sma_200=140.0,
                avg_vol_30d=9500000,
            ),
            MaverickStocks(
                id=2,
                stock="MSFT",
                close=300.0,
                open=298.0,
                high=302.0,
                low=297.0,
                volume=8000000,
                combined_score=89,
                momentum_score=82,
                adr_pct=2.1,
                atr=4.5,
                pat="Ascending Triangle",
                sqz="No",
                consolidation="trending",
                entry="301.00",
                compression_score=80,
                pattern_detected=1,
                ema_21=299.0,
                sma_50=298.0,
                sma_150=295.0,
                sma_200=290.0,
                avg_vol_30d=7500000,
            ),
        ]
        session.add_all(maverick_stocks)

        # Add trending screening results
        trending_stocks = [
            SupplyDemandBreakoutStocks(
                id=1,
                stock="GOOGL",
                close=140.0,
                open=138.0,
                high=142.0,
                low=137.0,
                volume=5000000,
                momentum_score=91,
                adr_pct=2.8,
                atr=3.5,
                pat="Base Breakout",
                sqz="Yes",
                consolidation="trending",
                entry="141.00",
                ema_21=139.0,
                sma_50=138.0,
                sma_150=135.0,
                sma_200=130.0,
                avg_volume_30d=4800000,
            ),
        ]
        session.add_all(trending_stocks)
        session.commit()

    with patch("maverick_mcp.data.models.engine", engine):
        with patch("maverick_mcp.data.models.SessionLocal", lambda: Session(engine)):
            yield engine


class TestTechnicalRouter:
    """Test technical analysis router functionality."""

    @pytest.mark.asyncio
    async def test_rsi_calculation(self, test_server, screening_db):
        """Test RSI calculation through the router."""
        test_server.mount("/technical", technical_router)

        # Mock price data for RSI calculation
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            # Create 30 days of price data
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=30)
            prices = pd.DataFrame(
                {
                    "Close": [
                        100 + (i % 5) - 2 for i in range(30)
                    ],  # Oscillating prices
                    "High": [101 + (i % 5) - 2 for i in range(30)],
                    "Low": [99 + (i % 5) - 2 for i in range(30)],
                    "Open": [100 + (i % 5) - 2 for i in range(30)],
                    "Volume": [1000000] * 30,
                },
                index=dates,
            )
            mock_data.return_value = prices

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "/technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
                )

                assert len(result) > 0
                assert result[0].text is not None
                # RSI should be calculated
                assert "rsi" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_macd_analysis(self, test_server, screening_db):
        """Test MACD analysis with custom parameters."""
        test_server.mount("/technical", technical_router)

        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            # Create trending price data
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=50)
            prices = pd.DataFrame(
                {
                    "Close": [100 + (i * 0.5) for i in range(50)],  # Upward trend
                    "High": [101 + (i * 0.5) for i in range(50)],
                    "Low": [99 + (i * 0.5) for i in range(50)],
                    "Open": [100 + (i * 0.5) for i in range(50)],
                    "Volume": [1000000] * 50,
                },
                index=dates,
            )
            mock_data.return_value = prices

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "/technical_get_macd_analysis",
                    {
                        "ticker": "MSFT",
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                )

                assert len(result) > 0
                assert result[0].text is not None
                data = eval(result[0].text)
                assert "analysis" in data
                assert "histogram" in data["analysis"]
                assert "indicator" in data["analysis"]

    @pytest.mark.asyncio
    async def test_support_resistance(self, test_server, screening_db):
        """Test support and resistance level detection."""
        test_server.mount("/technical", technical_router)

        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            # Create price data with clear levels
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=100)
            prices = []
            for i in range(100):
                if i % 20 < 10:
                    price = 100  # Support level
                else:
                    price = 110  # Resistance level
                prices.append(
                    {
                        "High": price + 1,
                        "Low": price - 1,
                        "Close": price,
                        "Open": price,
                        "Volume": 1000000,
                    }
                )
            prices_df = pd.DataFrame(prices, index=dates)
            mock_data.return_value = prices_df

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "/technical_get_support_resistance",
                    {"ticker": "GOOGL", "days": 90},
                )

                assert len(result) > 0
                assert result[0].text is not None
                data = eval(result[0].text)
                assert "support_levels" in data
                assert "resistance_levels" in data
                assert len(data["support_levels"]) > 0
                assert len(data["resistance_levels"]) > 0


class TestScreeningRouter:
    """Test stock screening router functionality."""

    @pytest.mark.asyncio
    async def test_maverick_screening(self, test_server, screening_db):
        """Test Maverick bullish screening."""
        test_server.mount("/screening", screening_router)

        async with Client(test_server) as client:
            result = await client.call_tool(
                "/screening_get_maverick_stocks", {"limit": 10}
            )

            assert len(result) > 0
            assert result[0].text is not None
            data = eval(result[0].text)

            assert "stocks" in data
            assert len(data["stocks"]) == 2  # AAPL and MSFT
            assert (
                data["stocks"][0]["combined_score"]
                > data["stocks"][1]["combined_score"]
            )  # Sorted by combined score
            assert all(
                stock["combined_score"] > 0 for stock in data["stocks"]
            )  # Score should be positive

    @pytest.mark.asyncio
    async def test_trending_screening(self, test_server, screening_db):
        """Test trending screening."""
        test_server.mount("/screening", screening_router)

        async with Client(test_server) as client:
            result = await client.call_tool(
                "/screening_get_trending_stocks", {"limit": 5}
            )

            assert len(result) > 0
            assert result[0].text is not None
            data = eval(result[0].text)

            assert "stocks" in data
            assert len(data["stocks"]) == 1  # Only GOOGL
            assert data["stocks"][0]["stock"] == "GOOGL"
            assert (
                data["stocks"][0]["momentum_score"] > 0
            )  # Momentum score should be positive

    @pytest.mark.asyncio
    async def test_all_screenings(self, test_server, screening_db):
        """Test combined screening results."""
        test_server.mount("/screening", screening_router)

        async with Client(test_server) as client:
            result = await client.call_tool(
                "/screening_get_all_screening_recommendations", {}
            )

            assert len(result) > 0
            assert result[0].text is not None
            data = eval(result[0].text)

            assert "maverick_stocks" in data
            assert "maverick_bear_stocks" in data
            assert "trending_stocks" in data
            assert len(data["maverick_stocks"]) == 2
            assert len(data["trending_stocks"]) == 1


class TestPortfolioRouter:
    """Test portfolio analysis router functionality."""

    @pytest.mark.asyncio
    async def test_risk_analysis(self, test_server, screening_db):
        """Test portfolio risk analysis."""
        test_server.mount("/portfolio", portfolio_router)

        # Mock stock data for risk calculations
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            # Create price data with volatility
            import numpy as np
            import pandas as pd

            prices = []
            base_price = 100.0
            for _ in range(252):  # One year of trading days
                # Add some random walk
                change = np.random.normal(0, 2)
                base_price = float(base_price * (1 + change / 100))
                prices.append(
                    {
                        "close": base_price,
                        "high": base_price + 1,
                        "low": base_price - 1,
                        "open": base_price,
                        "volume": 1000000,
                    }
                )
            dates = pd.date_range(end="2024-01-31", periods=252)
            prices_df = pd.DataFrame(prices, index=dates)
            mock_data.return_value = prices_df

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "/portfolio_risk_adjusted_analysis",
                    {"ticker": "AAPL", "risk_level": 50.0},
                )

                assert len(result) > 0
                assert result[0].text is not None
                data = eval(result[0].text)

                assert "risk_level" in data or "analysis" in data
                assert "ticker" in data

    @pytest.mark.asyncio
    async def test_correlation_analysis(self, test_server, screening_db):
        """Test correlation analysis between stocks."""
        test_server.mount("/portfolio", portfolio_router)

        # Mock correlated stock data
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import numpy as np

            def create_correlated_data(base_return, correlation):
                import pandas as pd

                prices = []
                base_price = 100
                for _ in range(100):
                    # Create correlated returns
                    return_pct = base_return + (correlation * np.random.normal(0, 1))
                    base_price = base_price * (1 + return_pct / 100)
                    prices.append(
                        {
                            "close": base_price,
                            "high": base_price + 1,
                            "low": base_price - 1,
                            "open": base_price,
                            "volume": 1000000,
                        }
                    )
                dates = pd.date_range(end="2024-01-31", periods=100)
                return pd.DataFrame(prices, index=dates)

            # Return different data for different tickers
            mock_data.side_effect = [
                create_correlated_data(0.1, 0),  # AAPL
                create_correlated_data(0.1, 0.8),  # MSFT (high correlation)
                create_correlated_data(0.1, -0.3),  # GOOGL (negative correlation)
            ]

            async with Client(test_server) as client:
                result = await client.call_tool(
                    "/portfolio_portfolio_correlation_analysis",
                    {"tickers": ["AAPL", "MSFT", "GOOGL"]},
                )

                assert len(result) > 0
                assert result[0].text is not None

                # Handle NaN values in response
                result_text = result[0].text.replace("NaN", "null")
                import json

                data = json.loads(result_text.replace("'", '"'))

                assert "correlation_matrix" in data
                assert len(data["correlation_matrix"]) == 3
                assert "recommendation" in data


class TestDataRouter:
    """Test data fetching router functionality."""

    @pytest.mark.asyncio
    async def test_batch_fetch_with_validation(self, test_server, screening_db):
        """Test batch data fetching with validation."""
        test_server.mount("/data", data_router)

        async with Client(test_server) as client:
            # Test with valid tickers
            result = await client.call_tool(
                "/data_fetch_stock_data_batch",
                {
                    "request": {
                        "tickers": ["AAPL", "MSFT"],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    }
                },
            )

            assert len(result) > 0
            assert result[0].text is not None
            data = eval(result[0].text)
            assert "results" in data
            assert len(data["results"]) == 2

            # Test with invalid ticker format
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "/data_fetch_stock_data_batch",
                    {
                        "request": {
                            "tickers": [
                                "AAPL",
                                "invalid_ticker",
                            ],  # lowercase not allowed
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-31",
                        }
                    },
                )

            assert "validation error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cache_operations(self, test_server, screening_db):
        """Test cache management operations."""
        test_server.mount("/data", data_router)

        # Patch the _get_redis_client to test cache operations
        with patch("maverick_mcp.data.cache._get_redis_client") as mock_redis_client:
            cache_instance = Mock()
            cache_instance.get.return_value = '{"cached": true, "data": "test"}'
            cache_instance.set.return_value = True
            cache_instance.delete.return_value = 1
            cache_instance.keys.return_value = [b"stock:AAPL:1", b"stock:AAPL:2"]
            mock_redis_client.return_value = cache_instance

            async with Client(test_server) as client:
                # Test cache clear
                result = await client.call_tool(
                    "/data_clear_cache", {"request": {"ticker": "AAPL"}}
                )

                assert len(result) > 0
                assert result[0].text is not None
                assert (
                    "clear" in result[0].text.lower()
                    or "success" in result[0].text.lower()
                )
                # Verify cache operations
                assert cache_instance.keys.called or cache_instance.delete.called


class TestConcurrentOperations:
    """Test concurrent operations and performance."""

    @pytest.mark.asyncio
    async def test_concurrent_router_calls(self, test_server, screening_db):
        """Test multiple routers being called concurrently."""
        # Mount all routers
        test_server.mount("/technical", technical_router)
        test_server.mount("/screening", screening_router)
        test_server.mount("/portfolio", portfolio_router)
        test_server.mount("/data", data_router)

        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=30)
            mock_data.return_value = pd.DataFrame(
                {
                    "Close": [100 + i for i in range(30)],
                    "High": [101 + i for i in range(30)],
                    "Low": [99 + i for i in range(30)],
                    "Open": [100 + i for i in range(30)],
                    "Volume": [1000000] * 30,
                },
                index=dates,
            )

            async with Client(test_server) as client:
                # Create concurrent tasks across different routers
                tasks = [
                    client.call_tool(
                        "/technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
                    ),
                    client.call_tool("/screening_get_maverick_stocks", {"limit": 5}),
                    client.call_tool(
                        "/data_fetch_stock_data_batch",
                        {
                            "request": {
                                "tickers": ["AAPL", "MSFT"],
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-31",
                            }
                        },
                    ),
                ]

                results = await asyncio.gather(*tasks)

                # All should complete successfully
                assert len(results) == 3
                for result in results:
                    assert len(result) > 0
                    assert result[0].text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
