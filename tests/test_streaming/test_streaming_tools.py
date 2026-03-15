"""Unit tests for the streaming MCP tools."""

from unittest.mock import AsyncMock, patch

import pytest

from maverick_mcp.streaming.price_stream_manager import PriceStreamManager
from maverick_mcp.streaming.tools import (
    get_price_snapshot,
    get_stream_status,
    set_poll_interval,
    start_price_stream,
    stop_price_stream,
    subscribe,
    unsubscribe,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    PriceStreamManager.reset()
    yield
    PriceStreamManager.reset()


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_returns_status(self):
        with patch.object(PriceStreamManager, "start", new_callable=AsyncMock):
            result = await start_price_stream(["AAPL", "MSFT"], interval=20.0)
            assert result["status"] == "streaming_started"
            assert "/ws/prices" in result["websocket_url"]

    @pytest.mark.asyncio
    async def test_stop_returns_confirmation(self):
        mgr = PriceStreamManager.get_instance()
        with patch.object(mgr, "_poll_loop", new_callable=AsyncMock):
            await mgr.start(["AAPL"])
        result = await stop_price_stream()
        assert result["status"] == "stopped"
        assert result["was_running"] is True


class TestSubscribeUnsubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_adds_tickers(self):
        with patch.object(PriceStreamManager, "start", new_callable=AsyncMock):
            result = await subscribe(["AAPL", "GOOGL"])
            assert result["status"] == "subscribed"
            assert "AAPL" in result["subscribed_tickers"]

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_tickers(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["AAPL", "MSFT"])
        result = await unsubscribe(["MSFT"])
        assert result["status"] == "unsubscribed"
        assert "MSFT" not in result["subscribed_tickers"]


class TestStatusAndConfig:
    @pytest.mark.asyncio
    async def test_get_stream_status(self):
        result = await get_stream_status()
        assert "running" in result
        assert "subscribed_tickers" in result

    @pytest.mark.asyncio
    async def test_set_poll_interval(self):
        result = await set_poll_interval(30.0)
        assert result["status"] == "interval_updated"
        assert result["poll_interval"] == 30.0

    @pytest.mark.asyncio
    async def test_set_poll_interval_clamped(self):
        result = await set_poll_interval(1.0)
        assert result["poll_interval"] == 5.0


class TestPriceSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_delegates_to_manager(self):
        mock_result = {
            "results": {"AAPL": {"price": 180.0}},
            "timestamp": "2026-03-13T00:00:00Z",
        }
        with patch.object(
            PriceStreamManager.get_instance(),
            "get_price_snapshot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await get_price_snapshot(["AAPL"])
            assert result["results"]["AAPL"]["price"] == 180.0
