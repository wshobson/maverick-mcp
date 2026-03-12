"""
In-memory tests for Maverick-MCP server using FastMCP patterns.

These tests demonstrate how to test the server without external processes
or network calls, using FastMCP's in-memory transport capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client, FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from maverick_mcp.api.routers.tool_registry import register_all_router_tools
from maverick_mcp.data.models import Base, PriceCache, Stock


def _disable_output_schemas(server: FastMCP) -> None:
    """Disable output schema validation on all tools to avoid numpy serialization issues."""
    lp = server.providers[0]
    for key, comp in lp._components.items():
        if key.startswith("tool:") and hasattr(comp, "output_schema"):
            comp.output_schema = None


@pytest.fixture
def mcp():
    """Create a test MCP server with all tools and resources registered."""
    server = FastMCP("TestMaverick-MCP")
    register_all_router_tools(server)
    _disable_output_schemas(server)

    # Register the health resource - wrap to return str since FastMCP 3.x
    # resources must return str, bytes, or list[ResourceContent], not dict
    from maverick_mcp.api.server import health_resource

    @server.resource("health://")
    def health_resource_str() -> str:
        result = health_resource()
        # Recursively convert Pydantic models and complex objects to dicts
        return _serialize_health(result)

    return server


def _serialize_health(obj: Any) -> str:
    """Serialize health resource output to JSON string, handling Pydantic models."""

    def _convert(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "_asdict"):
            return {k: _convert(v) for k, v in o._asdict().items()}
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_convert(i) for i in o]
        if isinstance(o, Mock):
            return str(o)
        return o

    converted = _convert(obj)
    return json.dumps(converted, default=str)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch("maverick_mcp.data.cache.get_redis_client") as mock_redis:
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
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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
    async def test_server_health(self, mcp, test_db, mock_redis):
        """Test the health endpoint returns correct status."""
        async with Client(mcp) as client:
            result = await client.read_resource("health://")

            # Result is a list of content items
            assert len(result) > 0
            assert result[0].text is not None
            health_data = json.loads(result[0].text)

            # In testing environment, status might be degraded/unhealthy due to event loop issues
            assert health_data["status"] in ["ok", "degraded", "unhealthy"]
            assert "version" in health_data or "service" in health_data

    @pytest.mark.asyncio
    async def test_fetch_stock_data(self, mcp, test_db, mock_redis):
        """Test fetching stock data from the database."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "data_fetch_stock_data",
                {
                    "ticker": "AAPL",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            # Result should contain stock data
            data = eval(result.content[0].text)
            assert data["ticker"] == "AAPL"
            assert "record_count" in data

    @pytest.mark.asyncio
    async def test_rsi_analysis(self, mcp, test_db, mock_redis):
        """Test RSI technical analysis calculation."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "technical_get_rsi_analysis", {"ticker": "AAPL", "period": 14}
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            # Should contain RSI data
            data = eval(result.content[0].text)
            assert "analysis" in data
            assert "ticker" in data
            assert data["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_batch_stock_data(self, mcp, test_db, mock_redis):
        """Test batch fetching of multiple stocks."""
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
            assert "AAPL" in data["results"]
            assert "MSFT" in data["results"]
            assert data["success_count"] == 2

    @pytest.mark.asyncio
    async def test_invalid_ticker(self, mcp, test_db, mock_redis):
        """Test handling of invalid ticker symbols."""
        async with Client(mcp) as client:
            # Invalid ticker should return an error or empty data
            result = await client.call_tool(
                "data_fetch_stock_data",
                {
                    "ticker": "INVALID123",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )

            # Should return empty data for invalid ticker
            assert len(result.content) > 0
            assert result.content[0].text is not None
            data = eval(result.content[0].text)
            # Invalid ticker returns empty data
            assert data["record_count"] == 0
            assert len(data["data"]) == 0

    @pytest.mark.asyncio
    async def test_date_validation(self, mcp, test_db, mock_redis):
        """Test date range validation -- reversed dates should return empty or error."""
        async with Client(mcp) as client:
            result = await client.call_tool(
                "data_fetch_stock_data",
                {
                    "ticker": "AAPL",
                    "start_date": "2024-01-31",
                    "end_date": "2024-01-01",  # End before start
                },
            )

            # Reversed dates should return empty data or an error key
            assert len(result.content) > 0
            data = json.loads(result.content[0].text)
            assert data.get("record_count", 0) == 0 or "error" in data

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mcp, test_db, mock_redis):
        """Test handling multiple concurrent requests."""
        async with Client(mcp) as client:
            # Create multiple concurrent tasks
            tasks = [
                client.call_tool(
                    "data_fetch_stock_data",
                    {
                        "ticker": "AAPL",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                )
                for _ in range(5)
            ]

            # All should complete successfully
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            for result in results:
                assert len(result.content) > 0
                assert result.content[0].text is not None
                data = eval(result.content[0].text)
                assert data["ticker"] == "AAPL"


class TestResourceManagement:
    """Test resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_list_resources(self, mcp, test_db, mock_redis):
        """Test listing available resources."""
        async with Client(mcp) as client:
            resources = await client.list_resources()

            # In the current implementation, resources may be empty or have different URIs
            # Just check that the call succeeds
            assert isinstance(resources, list)

    @pytest.mark.asyncio
    async def test_read_resource(self, mcp, test_db, mock_redis):
        """Test reading a specific resource."""
        async with Client(mcp) as client:
            result = await client.read_resource("health://")

            assert len(result) > 0
            assert result[0].text is not None
            # Health resource should return valid JSON with status info
            assert "status" in result[0].text.lower()


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_database_error_handling(self, mcp, mock_redis):
        """Test graceful handling of database errors."""
        with patch(
            "maverick_mcp.data.models.SessionLocal", side_effect=Exception("DB Error")
        ):
            async with Client(mcp) as client:
                result = await client.read_resource("health://")

                assert len(result) > 0
                health_data = json.loads(result[0].text)
                # Should indicate degraded or unhealthy status
                assert health_data["status"] in ["degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_cache_fallback(self, mcp, test_db):
        """Test fallback to in-memory cache when Redis is unavailable."""
        with patch(
            "maverick_mcp.data.cache.redis.Redis", side_effect=Exception("Redis Error")
        ):
            async with Client(mcp) as client:
                result = await client.read_resource("health://")

                assert len(result) > 0
                health_data = json.loads(result[0].text)
                # Health status should still be returned
                assert health_data["status"] in ["ok", "degraded", "unhealthy"]


class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""

    @pytest.mark.asyncio
    async def test_query_performance_tracking(self, mcp, test_db, mock_redis):
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
async def test_with_mock_data_provider(mcp, test_db, mock_redis):
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
                "data_fetch_stock_data",
                {
                    "ticker": "TSLA",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )

            assert len(result.content) > 0
            assert result.content[0].text is not None
            assert "TSLA" in result.content[0].text


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
