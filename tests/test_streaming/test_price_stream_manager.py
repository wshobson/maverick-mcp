"""Unit tests for PriceStreamManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maverick_mcp.streaming.price_stream_manager import PriceStreamManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the PriceStreamManager singleton before/after each test."""
    PriceStreamManager.reset()
    yield
    PriceStreamManager.reset()


# --- Singleton ---


class TestSingleton:
    def test_get_instance_returns_same_object(self):
        a = PriceStreamManager.get_instance()
        b = PriceStreamManager.get_instance()
        assert a is b

    def test_reset_clears_instance(self):
        _ = PriceStreamManager.get_instance()
        PriceStreamManager.reset()
        assert PriceStreamManager._instance is None


# --- Subscribe / Unsubscribe ---


class TestSubscriptions:
    def test_subscribe_adds_tickers(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["AAPL", "msft"])
        assert mgr.subscribed_tickers == ["AAPL", "MSFT"]

    def test_subscribe_uppercases(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["goog"])
        assert "GOOG" in mgr.subscribed_tickers

    def test_subscribe_respects_max(self):
        mgr = PriceStreamManager.get_instance()
        mgr._config.max_subscriptions = 2
        mgr.subscribe(["A", "B", "C"])
        assert len(mgr.subscribed_tickers) == 2

    def test_unsubscribe_removes_tickers(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["AAPL", "MSFT", "GOOGL"])
        mgr.unsubscribe(["MSFT"])
        assert mgr.subscribed_tickers == ["AAPL", "GOOGL"]

    def test_unsubscribe_clears_state(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["AAPL"])
        mgr._last_prices["AAPL"] = 150.0
        mgr._last_alert_states["AAPL"] = {"rsi_overbought"}
        mgr.unsubscribe(["AAPL"])
        assert "AAPL" not in mgr._last_prices
        assert "AAPL" not in mgr._last_alert_states

    def test_unsubscribe_nonexistent_ticker_is_noop(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["AAPL"])
        mgr.unsubscribe(["ZZZZZ"])
        assert mgr.subscribed_tickers == ["AAPL"]


# --- Poll Interval ---


class TestPollInterval:
    def test_set_poll_interval_normal(self):
        mgr = PriceStreamManager.get_instance()
        mgr.set_poll_interval(20.0)
        assert mgr.poll_interval == 20.0

    def test_set_poll_interval_clamps_low(self):
        mgr = PriceStreamManager.get_instance()
        mgr.set_poll_interval(1.0)
        assert mgr.poll_interval == 5.0

    def test_set_poll_interval_clamps_high(self):
        mgr = PriceStreamManager.get_instance()
        mgr.set_poll_interval(120.0)
        assert mgr.poll_interval == 60.0


# --- Status ---


class TestStatus:
    def test_get_status_has_required_keys(self):
        mgr = PriceStreamManager.get_instance()
        status = mgr.get_status()
        assert "running" in status
        assert "poll_interval" in status
        assert "subscribed_tickers" in status
        assert "connection_count" in status
        assert "last_prices" in status
        assert "timestamp" in status

    def test_get_status_reflects_state(self):
        mgr = PriceStreamManager.get_instance()
        mgr.subscribe(["AAPL"])
        status = mgr.get_status()
        assert status["subscribed_tickers"] == ["AAPL"]
        assert status["running"] is False

    def test_get_alert_state_structure(self):
        mgr = PriceStreamManager.get_instance()
        state = mgr.get_alert_state()
        assert "running" in state
        assert "alerts_by_ticker" in state
        assert "timestamp" in state


# --- Start / Stop ---


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        mgr = PriceStreamManager.get_instance()
        # Patch the poll loop to avoid actually running
        with patch.object(mgr, "_poll_loop", new_callable=AsyncMock):
            await mgr.start(["AAPL"])
            assert mgr.is_running is True
            assert mgr._poll_task is not None
            await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        mgr = PriceStreamManager.get_instance()
        with patch.object(mgr, "_poll_loop", new_callable=AsyncMock):
            await mgr.start(["AAPL"])
            await mgr.stop()
            assert mgr.is_running is False
            assert mgr._poll_task is None

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        mgr = PriceStreamManager.get_instance()
        with patch.object(mgr, "_poll_loop", new_callable=AsyncMock):
            await mgr.start(["AAPL"])
            task1 = mgr._poll_task
            await mgr.start(["MSFT"])
            assert mgr._poll_task is task1  # Same task, not restarted
            await mgr.stop()


# --- Alert Delta Detection ---


class TestAlertDelta:
    @pytest.mark.asyncio
    async def test_emits_on_new_alerts(self):
        mgr = PriceStreamManager.get_instance()
        mgr._last_alert_states["AAPL"] = set()

        mock_alerts = [
            {
                "type": "rsi_overbought",
                "severity": "warning",
                "message": "RSI high",
                "value": 75,
                "threshold": 70,
            }
        ]

        with patch.object(
            mgr,
            "_evaluate_alerts_for_ticker",
            new_callable=AsyncMock,
            return_value=mock_alerts,
        ):
            broadcast_mock = AsyncMock()
            with patch.object(mgr, "_broadcast", broadcast_mock):
                await mgr._evaluate_and_broadcast_alerts("AAPL")
                broadcast_mock.assert_called_once()
                call_msg = broadcast_mock.call_args[0][0]
                assert call_msg["type"] == "alert_change"
                assert len(call_msg["new_alerts"]) == 1
                assert call_msg["resolved_alerts"] == []

    @pytest.mark.asyncio
    async def test_emits_on_resolved_alerts(self):
        mgr = PriceStreamManager.get_instance()
        mgr._last_alert_states["AAPL"] = {"rsi_overbought", "volume_spike"}

        with patch.object(
            mgr, "_evaluate_alerts_for_ticker", new_callable=AsyncMock, return_value=[]
        ):
            broadcast_mock = AsyncMock()
            with patch.object(mgr, "_broadcast", broadcast_mock):
                await mgr._evaluate_and_broadcast_alerts("AAPL")
                broadcast_mock.assert_called_once()
                call_msg = broadcast_mock.call_args[0][0]
                assert call_msg["type"] == "alert_change"
                assert set(call_msg["resolved_alerts"]) == {
                    "rsi_overbought",
                    "volume_spike",
                }

    @pytest.mark.asyncio
    async def test_no_broadcast_when_unchanged(self):
        mgr = PriceStreamManager.get_instance()
        mgr._last_alert_states["AAPL"] = {"rsi_overbought"}

        mock_alerts = [
            {
                "type": "rsi_overbought",
                "severity": "warning",
                "message": "...",
                "value": 75,
                "threshold": 70,
            }
        ]

        with patch.object(
            mgr,
            "_evaluate_alerts_for_ticker",
            new_callable=AsyncMock,
            return_value=mock_alerts,
        ):
            broadcast_mock = AsyncMock()
            with patch.object(mgr, "_broadcast", broadcast_mock):
                await mgr._evaluate_and_broadcast_alerts("AAPL")
                broadcast_mock.assert_not_called()


# --- Price Snapshot ---


class TestPriceSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_returns_prices(self):
        mgr = PriceStreamManager.get_instance()

        mock_prices = {
            "AAPL": {
                "price": 178.50,
                "change": 2.30,
                "change_pct": 1.31,
                "volume": 50000000,
                "previous_close": 176.20,
            }
        }

        with patch.object(
            mgr, "_fetch_prices", new_callable=AsyncMock, return_value=mock_prices
        ):
            with patch.object(
                mgr,
                "_evaluate_alerts_for_ticker",
                new_callable=AsyncMock,
                return_value=[],
            ):
                result = await mgr.get_price_snapshot(["AAPL"])
                assert "results" in result
                assert "AAPL" in result["results"]
                assert result["results"]["AAPL"]["price"] == 178.50

    @pytest.mark.asyncio
    async def test_snapshot_empty_tickers(self):
        mgr = PriceStreamManager.get_instance()
        result = await mgr.get_price_snapshot([])
        assert "error" in result

    @pytest.mark.asyncio
    async def test_snapshot_includes_alerts(self):
        mgr = PriceStreamManager.get_instance()
        mock_prices = {
            "AAPL": {
                "price": 180.0,
                "change": 0,
                "change_pct": 0,
                "volume": 0,
                "previous_close": 180.0,
            }
        }
        mock_alerts = [
            {
                "type": "rsi_overbought",
                "severity": "warning",
                "message": "...",
                "value": 75,
                "threshold": 70,
            }
        ]

        with patch.object(
            mgr, "_fetch_prices", new_callable=AsyncMock, return_value=mock_prices
        ):
            with patch.object(
                mgr,
                "_evaluate_alerts_for_ticker",
                new_callable=AsyncMock,
                return_value=mock_alerts,
            ):
                result = await mgr.get_price_snapshot(["AAPL"])
                assert len(result["results"]["AAPL"]["alerts"]) == 1


# --- Broadcast ---


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_to_connected_clients(self):
        mgr = PriceStreamManager.get_instance()
        ws_mock = MagicMock()
        ws_mock.client_state = MagicMock()
        # Simulate CONNECTED state
        from starlette.websockets import WebSocketState

        ws_mock.client_state = WebSocketState.CONNECTED
        ws_mock.send_json = AsyncMock()

        mgr._connections["test-1"] = ws_mock
        mgr._last_send["test-1"] = 0.0

        await mgr._broadcast({"type": "test"})
        ws_mock.send_json.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        mgr = PriceStreamManager.get_instance()
        ws_mock = MagicMock()
        from starlette.websockets import WebSocketState

        ws_mock.client_state = WebSocketState.DISCONNECTED
        ws_mock.send_json = AsyncMock()

        mgr._connections["dead-1"] = ws_mock
        mgr._last_send["dead-1"] = 0.0

        await mgr._broadcast({"type": "test"})
        assert "dead-1" not in mgr._connections
