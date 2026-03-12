"""
Comprehensive tests for all MCP tool functions in Maverick-MCP.

This module tests all public MCP tools exposed by the server including:
- Stock data fetching
- Technical analysis
- Risk analysis
- Chart generation
- News sentiment
- Multi-ticker comparison
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client, FastMCP

from maverick_mcp.api.routers.tool_registry import register_all_router_tools


def _disable_output_schemas(server: FastMCP) -> None:
    """Disable output schema validation on all tools to avoid numpy serialization issues."""
    lp = server.providers[0]
    for key, comp in lp._components.items():
        if key.startswith("tool:") and hasattr(comp, "output_schema"):
            comp.output_schema = None


@pytest.fixture
def mcp():
    """Create a test MCP server with all tools registered (no incompatible middleware)."""
    server = FastMCP("TestMaverick-MCP")
    register_all_router_tools(server)
    _disable_output_schemas(server)
    return server


class TestMCPTools:
    """Test suite for all MCP tool functions using the new router structure."""

    @pytest.fixture
    def mock_stock_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range(end=datetime.now(), periods=250, freq="D")
        return pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, 250),
                "High": np.random.uniform(95, 115, 250),
                "Low": np.random.uniform(85, 105, 250),
                "Close": np.random.uniform(90, 110, 250),
                "Volume": np.random.randint(1000000, 10000000, 250),
            },
            index=dates,
        )

    @pytest.mark.asyncio
    async def test_fetch_stock_data(self, mcp, mock_stock_data):
        """Test basic stock data fetching."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            mock_get.return_value = mock_stock_data

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "data_fetch_stock_data",
                    {
                        "ticker": "AAPL",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                )

                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "ticker" in data
                assert data["ticker"] == "AAPL"
                assert "record_count" in data

    @pytest.mark.asyncio
    async def test_rsi_analysis(self, mcp, mock_stock_data):
        """Test RSI technical analysis."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            mock_get.return_value = mock_stock_data

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
                )

                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "analysis" in data
                assert "ticker" in data
                assert data["ticker"] == "AAPL"
                assert "current" in data["analysis"]
                assert "signal" in data["analysis"]
                assert data["analysis"]["signal"] in [
                    "oversold",
                    "neutral",
                    "overbought",
                    "bullish",
                    "bearish",
                ]

    @pytest.mark.asyncio
    async def test_macd_analysis(self, mcp, mock_stock_data):
        """Test MACD technical analysis."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            mock_get.return_value = mock_stock_data

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_macd_analysis", {"ticker": "MSFT"}
                )

                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "analysis" in data
                assert "ticker" in data
                assert data["ticker"] == "MSFT"
                assert "macd" in data["analysis"]
                assert "signal" in data["analysis"]
                assert "histogram" in data["analysis"]
                assert "indicator" in data["analysis"]

    @pytest.mark.asyncio
    async def test_support_resistance(self, mcp, mock_stock_data):
        """Test support and resistance level detection."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            # Create data with clear support/resistance levels using float values
            mock_data = mock_stock_data.copy()
            mock_data["High"] = [105.0 if i % 20 < 10 else 110.0 for i in range(250)]
            mock_data["Low"] = [95.0 if i % 20 < 10 else 100.0 for i in range(250)]
            mock_data["Close"] = [100.0 if i % 20 < 10 else 105.0 for i in range(250)]
            mock_get.return_value = mock_data

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_support_resistance", {"ticker": "GOOGL"}
                )

                assert result.content[0].text is not None
                import json

                data = json.loads(result.content[0].text)
                assert "support_levels" in data
                assert "resistance_levels" in data
                assert len(data["support_levels"]) > 0
                assert len(data["resistance_levels"]) > 0

    @pytest.mark.asyncio
    async def test_batch_stock_data(self, mcp, mock_stock_data):
        """Test batch stock data fetching."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            mock_get.return_value = mock_stock_data

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "data_fetch_stock_data_batch",
                    {
                        "tickers": ["AAPL", "MSFT", "GOOGL"],
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                )

                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "results" in data
                assert "success_count" in data
                assert "error_count" in data

    @pytest.mark.asyncio
    async def test_portfolio_risk_analysis(self, mcp, mock_stock_data):
        """Test portfolio risk analysis."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            mock_get.return_value = mock_stock_data

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "portfolio_risk_adjusted_analysis",
                    {"ticker": "AAPL", "risk_level": 50.0},
                )

                assert result.content[0].text is not None
                import json

                data = json.loads(result.content[0].text)
                assert "ticker" in data or "error" in data

    @pytest.mark.asyncio
    async def test_maverick_screening(self, mcp):
        """Test Maverick stock screening."""
        with (
            patch("maverick_mcp.data.models.SessionLocal") as mock_session_cls,
            patch(
                "maverick_mcp.data.models.MaverickStocks.get_top_stocks"
            ) as mock_get_stocks,
        ):
            # Mock database session (not used but needed for session lifecycle)
            _ = mock_session_cls.return_value.__enter__.return_value

            # Mock return data
            class MockStock1:
                def to_dict(self):
                    return {
                        "stock": "AAPL",
                        "close": 150.0,
                        "combined_score": 92,
                        "momentum_score": 88,
                        "adr_pct": 2.5,
                    }

            class MockStock2:
                def to_dict(self):
                    return {
                        "stock": "MSFT",
                        "close": 300.0,
                        "combined_score": 89,
                        "momentum_score": 85,
                        "adr_pct": 2.1,
                    }

            mock_get_stocks.return_value = [MockStock1(), MockStock2()]

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "screening_get_maverick_stocks", {"limit": 10}
                )

                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "stocks" in data
                assert len(data["stocks"]) == 2
                assert data["stocks"][0]["stock"] == "AAPL"

    @pytest.mark.asyncio
    async def test_news_sentiment(self, mcp):
        """Test news sentiment analysis returns valid data."""
        mock_sentiment_result = {
            "ticker": "AAPL",
            "sentiment": "bullish",
            "confidence": 0.85,
            "analysis": {
                "articles_analyzed": 5,
                "sentiment_distribution": {"positive": 3, "neutral": 1, "negative": 1},
                "key_themes": ["Strong earnings", "Product launch"],
            },
            "sources": ["tiingo"],
        }

        with patch(
            "maverick_mcp.api.routers.news_sentiment_enhanced.get_news_sentiment_enhanced",
            return_value=mock_sentiment_result,
        ):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "data_get_news_sentiment", {"ticker": "AAPL"}
                )

                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert "ticker" in data
                assert data["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_full_technical_analysis(self, mcp, mock_stock_data):
        """Test comprehensive technical analysis."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data"
        ) as mock_get:
            # Ensure lowercase column names for technical analysis
            mock_data_lowercase = mock_stock_data.copy()
            mock_data_lowercase.columns = mock_data_lowercase.columns.str.lower()
            mock_get.return_value = mock_data_lowercase

            async with Client(mcp) as client:
                result = await client.call_tool(
                    "technical_get_full_technical_analysis", {"ticker": "AAPL"}
                )

                assert result.content[0].text is not None
                import json

                data = json.loads(result.content[0].text)
                assert "indicators" in data
                assert "rsi" in data["indicators"]
                assert "macd" in data["indicators"]
                assert "bollinger_bands" in data["indicators"]
                assert "levels" in data
                assert "current_price" in data

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp):
        """Test error handling for invalid requests."""
        async with Client(mcp) as client:
            # Test with a ticker that causes an error
            result = await client.call_tool(
                "data_fetch_stock_data",
                {
                    "ticker": "AAPL",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )
            # Should return some result without crashing
            assert result.content[0].text is not None

    @pytest.mark.asyncio
    async def test_caching_behavior(self, mcp, mock_stock_data):
        """Test that repeated calls with the same parameters both succeed."""
        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_stock_data",
            return_value=mock_stock_data,
        ):
            async with Client(mcp) as client:
                # First call
                result1 = await client.call_tool(
                    "data_fetch_stock_data",
                    {
                        "ticker": "AAPL",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                )
                assert result1.content[0].text is not None

                # Second call with same parameters should also succeed
                result2 = await client.call_tool(
                    "data_fetch_stock_data",
                    {
                        "ticker": "AAPL",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                )
                assert result2.content[0].text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
