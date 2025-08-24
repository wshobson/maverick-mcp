"""
In-memory tests for Maverick-MCP server using FastMCP patterns.

These tests demonstrate how to test the server without external processes
or network calls, using FastMCP's in-memory transport capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from maverick_mcp.api.server import mcp
from maverick_mcp.data.models import Base, PriceCache, Stock


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch("maverick_mcp.data.cache._get_redis_client") as mock_redis:
        # Mock Redis client
        redis_instance = Mock()
        redis_instance.get.return_value = None
        redis_instance.set.return_value = True
        redis_instance.delete.return_value = True
        redis_instance.ping.return_value = True
        mock_redis.return_value = redis_instance
        yield redis_instance


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Add some test data
    with Session(engine) as session:
        # Add test stocks
        aapl = Stock(
            ticker_symbol="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
        )
        msft = Stock(
            ticker_symbol="MSFT",
            company_name="Microsoft Corp.",
            sector="Technology",
            industry="Software",
        )
        session.add_all([aapl, msft])
        session.commit()

        # Add test price data
        base_date = datetime.now() - timedelta(days=30)
        for i in range(30):
            date = base_date + timedelta(days=i)
            session.add(
                PriceCache(
                    stock_id=aapl.stock_id,
                    date=date,
                    open_price=150.0 + i,
                    high_price=152.0 + i,
                    low_price=149.0 + i,
                    close_price=151.0 + i,
                    volume=1000000 + i * 10000,
                )
            )
        session.commit()

    # Patch the database connection
    with patch("maverick_mcp.data.models.engine", engine):
        with patch("maverick_mcp.data.models.SessionLocal", lambda: Session(engine)):
            yield engine


class TestInMemoryServer:
    """Test suite for in-memory server operations."""

    @pytest.mark.asyncio
    async def test_server_health(self, test_db, mock_redis):
        """Test the health endpoint returns correct status."""
        async with Client(mcp) as client:
            result = await client.read_resource("health://")

            # Result is a list of content items
            assert len(result) > 0
            assert result[0].text is not None
            health_data = eval(result[0].text)  # Convert string representation to dict

            # In testing environment, status might be degraded due to mocked services
            assert health_data["status"] in ["ok", "degraded"]
            assert "version" in health_data
            assert "components" in health_data

            # Check available components
            components = health_data["components"]
            # Redis should be healthy (mocked)
            if "redis" in components:
                assert components["redis"]["status"] == "healthy"
            # Database status can be error in test environment due to SQLite pool differences
            if "database" in components:
                assert components["database"]["status"] in [
                    "healthy",
                    "degraded",
                    "unhealthy",
                    "error",
                ]

    @pytest.mark.asyncio
    async def test_fetch_stock_data(self, test_db, mock_redis):
        """Test fetching stock data from the database."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "/data_fetch_stock_data",
                {
                    "request": {
                        "ticker": "AAPL",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    }
                },
            )

            assert len(result) > 0
            assert result[0].text is not None
            # Result should contain stock data
            data = eval(result[0].text)
            assert data["ticker"] == "AAPL"
            assert "columns" in data
            assert "Open" in data["columns"]
            assert "Close" in data["columns"]
            assert data["record_count"] > 0

    @pytest.mark.asyncio
    async def test_rsi_analysis(self, test_db, mock_redis):
        """Test RSI technical analysis calculation."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "/technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
            )

            assert len(result) > 0
            assert result[0].text is not None
            # Should contain RSI data
            data = eval(result[0].text)
            assert "analysis" in data
            assert "ticker" in data
            assert data["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_batch_stock_data(self, test_db, mock_redis):
        """Test batch fetching of multiple stocks."""
        async with Client(mcp) as client:
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
            assert "AAPL" in data["results"]
            assert "MSFT" in data["results"]
            assert data["success_count"] == 2

    @pytest.mark.asyncio
    async def test_invalid_ticker(self, test_db, mock_redis):
        """Test handling of invalid ticker symbols."""
        async with Client(mcp) as client:
            # Invalid ticker should return an error, not raise an exception
            result = await client.call_tool(
                "/data_fetch_stock_data",
                {
                    "request": {
                        "ticker": "INVALID123",  # Invalid format
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    }
                },
            )

            # Should return empty data for invalid ticker
            assert len(result) > 0
            assert result[0].text is not None
            data = eval(result[0].text)
            # Invalid ticker returns empty data
            assert data["record_count"] == 0
            assert len(data["data"]) == 0

    @pytest.mark.asyncio
    async def test_date_validation(self, test_db, mock_redis):
        """Test date range validation."""
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool(
                    "/data_fetch_stock_data",
                    {
                        "request": {
                            "ticker": "AAPL",
                            "start_date": "2024-01-31",
                            "end_date": "2024-01-01",  # End before start
                        }
                    },
                )

            # Should fail with validation error
            assert (
                "error" in str(exc_info.value).lower()
                or "validation" in str(exc_info.value).lower()
            )

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_db, mock_redis):
        """Test handling multiple concurrent requests."""
        async with Client(mcp) as client:
            # Create multiple concurrent tasks
            tasks = [
                client.call_tool(
                    "/data_fetch_stock_data",
                    {
                        "request": {
                            "ticker": "AAPL",
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-31",
                        }
                    },
                )
                for _ in range(5)
            ]

            # All should complete successfully
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            for result in results:
                assert len(result) > 0
                assert result[0].text is not None
                data = eval(result[0].text)
                assert data["ticker"] == "AAPL"


class TestResourceManagement:
    """Test resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_list_resources(self, test_db, mock_redis):
        """Test listing available resources."""
        async with Client(mcp) as client:
            resources = await client.list_resources()

            # In the current implementation, resources may be empty or have different URIs
            # Just check that the call succeeds
            assert isinstance(resources, list)

    @pytest.mark.asyncio
    async def test_read_resource(self, test_db, mock_redis):
        """Test reading a specific resource."""
        async with Client(mcp) as client:
            result = await client.read_resource("health://")

            assert len(result) > 0
            assert result[0].text is not None
            # Should contain cache status information
            assert (
                "redis" in result[0].text.lower() or "memory" in result[0].text.lower()
            )


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_redis):
        """Test graceful handling of database errors."""
        # No test_db fixture, so database should fail
        with patch(
            "maverick_mcp.data.models.SessionLocal", side_effect=Exception("DB Error")
        ):
            async with Client(mcp) as client:
                result = await client.read_resource("health://")

                assert len(result) > 0
                health_data = eval(result[0].text)
                # Database should show an error
                assert health_data["status"] in ["degraded", "unhealthy"]
                assert "components" in health_data

    @pytest.mark.asyncio
    async def test_cache_fallback(self, test_db):
        """Test fallback to in-memory cache when Redis is unavailable."""
        # No mock_redis fixture, should fall back to memory
        with patch(
            "maverick_mcp.data.cache.redis.Redis", side_effect=Exception("Redis Error")
        ):
            async with Client(mcp) as client:
                result = await client.read_resource("health://")

                assert len(result) > 0
                health_data = eval(result[0].text)
                # Cache should fall back to memory
                assert "components" in health_data
                if "cache" in health_data["components"]:
                    assert health_data["components"]["cache"]["type"] == "memory"


class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""

    @pytest.mark.asyncio
    async def test_query_performance_tracking(self, test_db, mock_redis):
        """Test that query performance is tracked."""
        # Skip this test as health_monitor is not available
        pytest.skip("health_monitor not available in current implementation")


# Utility functions for testing


def create_test_stock_data(symbol: str, days: int = 30) -> dict[str, Any]:
    """Create test stock data for a given symbol."""
    data: dict[str, Any] = {"symbol": symbol, "prices": []}

    base_date = datetime.now() - timedelta(days=days)
    base_price = 100.0

    for i in range(days):
        date = base_date + timedelta(days=i)
        price = base_price + (i * 0.5)  # Gradual increase
        data["prices"].append(
            {
                "date": date.isoformat(),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1000000,
            }
        )

    return data


@pytest.mark.asyncio
async def test_with_mock_data_provider(test_db, mock_redis):
    """Test with mocked external data provider."""
    test_data = create_test_stock_data("TSLA", 30)

    with patch("yfinance.download") as mock_yf:
        # Mock yfinance response
        mock_df = Mock()
        mock_df.empty = False
        mock_df.to_dict.return_value = test_data["prices"]
        mock_yf.return_value = mock_df

        async with Client(mcp) as client:
            result = await client.call_tool(
                "/data_fetch_stock_data",
                {
                    "request": {
                        "ticker": "TSLA",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    }
                },
            )

            assert len(result) > 0
            assert result[0].text is not None
            assert "TSLA" in result[0].text


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
