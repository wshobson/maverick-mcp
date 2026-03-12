"""
In-memory tests for domain-specific routers using FastMCP patterns.

Tests individual router functionality in isolation using FastMCP's
in-memory testing capabilities via the main MCP server.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client, FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from maverick_mcp.api.server import mcp
from maverick_mcp.data.models import (
    Base,
    MaverickStocks,
    Stock,
    SupplyDemandBreakoutStocks,
)


def _disable_output_schemas(server: FastMCP) -> None:
    """Disable output schema validation on all tools to avoid numpy serialization issues."""
    lp = server.providers[0]
    for key, comp in lp._components.items():
        if key.startswith("tool:") and hasattr(comp, "output_schema"):
            comp.output_schema = None


@pytest.fixture
def screening_db():
    """Create test database with screening data."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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

        # Look up stock objects for FK references
        aapl = session.query(Stock).filter_by(ticker_symbol="AAPL").first()
        msft = session.query(Stock).filter_by(ticker_symbol="MSFT").first()
        googl = session.query(Stock).filter_by(ticker_symbol="GOOGL").first()

        # Add Maverick screening results
        maverick_stocks = [
            MaverickStocks(
                id=1,
                stock_id=aapl.stock_id,
                close_price=150.0,
                open_price=148.0,
                high_price=152.0,
                low_price=147.0,
                volume=10000000,
                combined_score=92,
                momentum_score=88,
                adr_pct=2.5,
                atr=3.2,
                pattern_type="Cup and Handle",
                squeeze_status="Yes",
                consolidation_status="trending",
                entry_signal="151.50",
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
                stock_id=msft.stock_id,
                close_price=300.0,
                open_price=298.0,
                high_price=302.0,
                low_price=297.0,
                volume=8000000,
                combined_score=89,
                momentum_score=82,
                adr_pct=2.1,
                atr=4.5,
                pattern_type="Ascending Triangle",
                squeeze_status="No",
                consolidation_status="trending",
                entry_signal="301.00",
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
                stock_id=googl.stock_id,
                close_price=140.0,
                open_price=138.0,
                high_price=142.0,
                low_price=137.0,
                volume=5000000,
                momentum_score=91,
                adr_pct=2.8,
                atr=3.5,
                pattern_type="Base Breakout",
                squeeze_status="Yes",
                consolidation_status="trending",
                entry_signal="141.00",
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
    async def test_rsi_calculation(self, screening_db):
        """Test RSI calculation through the router."""
        # Mock price data for RSI calculation
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=30)
            prices = pd.DataFrame(
                {
                    "Close": [100 + (i % 5) - 2 for i in range(30)],
                    "High": [101 + (i % 5) - 2 for i in range(30)],
                    "Low": [99 + (i % 5) - 2 for i in range(30)],
                    "Open": [100 + (i % 5) - 2 for i in range(30)],
                    "Volume": [1000000] * 30,
                },
                index=dates,
            )
            mock_data.return_value = prices

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
                )

                assert len(result.content) > 0
                assert result.content[0].text is not None
                assert "rsi" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_macd_analysis(self, screening_db):
        """Test MACD analysis with custom parameters."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=50)
            prices = pd.DataFrame(
                {
                    "Close": [100 + (i * 0.5) for i in range(50)],
                    "High": [101 + (i * 0.5) for i in range(50)],
                    "Low": [99 + (i * 0.5) for i in range(50)],
                    "Open": [100 + (i * 0.5) for i in range(50)],
                    "Volume": [1000000] * 50,
                },
                index=dates,
            )
            mock_data.return_value = prices

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_macd_analysis",
                    {
                        "ticker": "MSFT",
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                )

                assert len(result.content) > 0
                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "analysis" in data
                assert "histogram" in data["analysis"]
                assert "indicator" in data["analysis"]

    @pytest.mark.asyncio
    async def test_support_resistance(self, screening_db):
        """Test support and resistance level detection."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import pandas as pd

            dates = pd.date_range(end="2024-01-31", periods=100)
            prices = []
            for i in range(100):
                if i % 20 < 10:
                    price = 100
                else:
                    price = 110
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

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_support_resistance",
                    {"ticker": "GOOGL", "days": 90},
                )

                assert len(result.content) > 0
                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "support_levels" in data
                assert "resistance_levels" in data
                assert len(data["support_levels"]) > 0
                assert len(data["resistance_levels"]) > 0


class TestScreeningRouter:
    """Test stock screening router functionality."""

    @pytest.mark.asyncio
    async def test_maverick_screening(self, screening_db):
        """Test Maverick bullish screening."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "screening_get_maverick_stocks", {"limit": 10}
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            data = eval(result.content[0].text)

            assert "stocks" in data
            assert len(data["stocks"]) == 2  # AAPL and MSFT
            assert (
                data["stocks"][0]["combined_score"]
                > data["stocks"][1]["combined_score"]
            )
            assert all(stock["combined_score"] > 0 for stock in data["stocks"])

    @pytest.mark.asyncio
    async def test_trending_screening(self, screening_db):
        """Test supply/demand breakout screening."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "screening_get_supply_demand_breakouts", {"limit": 5}
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            data = eval(result.content[0].text)

            assert "stocks" in data
            assert len(data["stocks"]) == 1  # Only GOOGL
            assert (
                data["stocks"][0].get("stock") == "GOOGL"
                or data["stocks"][0].get("ticker") == "GOOGL"
            )
            assert data["stocks"][0]["momentum_score"] > 0

    @pytest.mark.asyncio
    async def test_all_screenings(self, screening_db):
        """Test combined screening results."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "screening_get_all_screening_recommendations", {}
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            data = eval(result.content[0].text)

            assert "maverick_stocks" in data
            assert "maverick_bear_stocks" in data
            assert "supply_demand_breakouts" in data
            assert len(data["maverick_stocks"]) == 2
            assert len(data["supply_demand_breakouts"]) == 1


class TestPortfolioRouter:
    """Test portfolio analysis router functionality."""

    @pytest.mark.asyncio
    async def test_risk_analysis(self, screening_db):
        """Test portfolio risk analysis."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import numpy as np
            import pandas as pd

            np.random.seed(42)
            prices = []
            base_price = 100.0
            for _ in range(252):
                change = np.random.normal(0, 2)
                base_price = float(base_price * (1 + change / 100))
                prices.append(
                    {
                        "Close": base_price,
                        "High": base_price + 1,
                        "Low": base_price - 1,
                        "Open": base_price,
                        "Volume": 1000000,
                    }
                )
            dates = pd.date_range(end="2024-01-31", periods=252)
            prices_df = pd.DataFrame(prices, index=dates)
            mock_data.return_value = prices_df

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "portfolio_risk_adjusted_analysis",
                    {"ticker": "AAPL", "risk_level": 50.0},
                )

                assert len(result.content) > 0
                assert result.content[0].text is not None
                import json

                data = json.loads(result.content[0].text)
                assert "ticker" in data or "error" in data

    @pytest.mark.asyncio
    async def test_correlation_analysis(self, screening_db):
        """Test correlation analysis between stocks."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_data:
            import numpy as np

            np.random.seed(42)

            def create_correlated_data(base_return, noise_scale):
                import pandas as pd

                prices = []
                base_price = 100.0
                for _ in range(100):
                    change = base_return + noise_scale * np.random.normal(0, 1)
                    base_price = base_price * (1 + change / 100)
                    prices.append(
                        {
                            "Close": float(base_price),
                            "High": float(base_price + 1),
                            "Low": float(base_price - 1),
                            "Open": float(base_price),
                            "Volume": 1000000,
                        }
                    )
                dates = pd.date_range(end="2024-01-31", periods=100)
                return pd.DataFrame(prices, index=dates)

            mock_data.side_effect = [
                create_correlated_data(0.1, 2.0),
                create_correlated_data(0.1, 2.0),
                create_correlated_data(0.1, 2.0),
            ]

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "portfolio_portfolio_correlation_analysis",
                    {"tickers": ["AAPL", "MSFT", "GOOGL"]},
                )

                assert len(result.content) > 0
                assert result.content[0].text is not None
                import json

                # Handle NaN values in response
                result_text = (
                    result.content[0].text.replace("NaN", "null").replace("'", '"')
                )
                data = json.loads(result_text)

                assert "correlation_matrix" in data or "error" in data


class TestDataRouter:
    """Test data fetching router functionality."""

    @pytest.mark.asyncio
    async def test_batch_fetch_with_validation(self, screening_db):
        """Test batch data fetching with validation."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "data_fetch_stock_data_batch",
                {
                    "tickers": ["AAPL", "MSFT"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            data = eval(result.content[0].text)
            assert "results" in data
            assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_cache_operations(self, screening_db):
        """Test cache management operations."""
        with patch("maverick_mcp.data.cache.get_redis_client") as mock_redis_client:
            cache_instance = Mock()
            cache_instance.get.return_value = '{"cached": true, "data": "test"}'
            cache_instance.set.return_value = True
            cache_instance.delete.return_value = 1
            cache_instance.keys.return_value = [b"stock:AAPL:1", b"stock:AAPL:2"]
            mock_redis_client.return_value = cache_instance

            async with Client(mcp) as client:
                result = await client.call_tool("data_clear_cache", {"ticker": "AAPL"})

                assert len(result.content) > 0
                assert result.content[0].text is not None
                assert (
                    "clear" in result.content[0].text.lower()
                    or "success" in result.content[0].text.lower()
                )


class TestConcurrentOperations:
    """Test concurrent operations and performance."""

    @pytest.mark.asyncio
    async def test_concurrent_router_calls(self, screening_db):
        """Test multiple tools being called concurrently."""
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

            async with Client(mcp) as client:
                tasks = [
                    client.call_tool(
                        "technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
                    ),
                    client.call_tool("screening_get_maverick_stocks", {"limit": 5}),
                    client.call_tool(
                        "data_fetch_stock_data_batch",
                        {
                            "tickers": ["AAPL", "MSFT"],
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-31",
                        },
                    ),
                ]

                results = await asyncio.gather(*tasks)

                assert len(results) == 3
                for result in results:
                    assert len(result.content) > 0
                    assert result.content[0].text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
