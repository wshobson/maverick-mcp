"""Singleton manager for real-time price polling and WebSocket broadcast.

Fetches prices via YFinancePool on a configurable interval, evaluates
watchlist alerts for state-change detection, and broadcasts updates to
all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from starlette.websockets import WebSocket, WebSocketState

from maverick_mcp.streaming.config import StreamingConfig

logger = logging.getLogger("maverick_mcp.streaming")


class PriceStreamManager:
    """Singleton that drives the price-polling loop and WebSocket fan-out.

    Usage::

        manager = PriceStreamManager.get_instance()
        await manager.start(["AAPL", "MSFT"])
        # ... later ...
        await manager.stop()
    """

    _instance: PriceStreamManager | None = None

    # ------------------------------------------------------------------ #
    # Singleton
    # ------------------------------------------------------------------ #

    @classmethod
    def get_instance(cls) -> PriceStreamManager:
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self._config = StreamingConfig()

        # Subscriptions
        self._subscribed_tickers: set[str] = set()

        # WebSocket connections: id -> WebSocket
        self._connections: dict[str, WebSocket] = {}

        # Background task
        self._poll_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._poll_count: int = 0

        # Last-known state (for delta detection)
        self._last_prices: dict[str, float] = {}
        self._last_alert_states: dict[str, set[str]] = {}

        # Rate limiting (connection_id -> last send timestamp)
        self._last_send: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self, tickers: list[str] | None = None) -> None:
        """Start the polling loop.  Optionally subscribe to *tickers*."""
        if tickers:
            self.subscribe(tickers)
        if self._running:
            return
        self._running = True
        self._poll_count = 0
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Price stream started (interval=%.1fs, tickers=%s)",
            self._config.poll_interval,
            sorted(self._subscribed_tickers),
        )

    async def stop(self) -> None:
        """Cancel the polling task and close all WebSocket connections."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Close WebSocket connections gracefully
        for _conn_id, ws in list(self._connections.items()):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass
        self._connections.clear()
        self._last_send.clear()
        self._subscribed_tickers.clear()
        self._last_prices.clear()
        self._last_alert_states.clear()
        logger.info("Price stream stopped")

    def subscribe(self, tickers: list[str]) -> None:
        """Add *tickers* to the subscription set (uppercased)."""
        for t in tickers:
            sym = t.strip().upper()
            if sym and len(self._subscribed_tickers) < self._config.max_subscriptions:
                self._subscribed_tickers.add(sym)

    def unsubscribe(self, tickers: list[str]) -> None:
        """Remove *tickers* from the subscription set."""
        for t in tickers:
            sym = t.strip().upper()
            self._subscribed_tickers.discard(sym)
            self._last_prices.pop(sym, None)
            self._last_alert_states.pop(sym, None)

    def set_poll_interval(self, seconds: float) -> None:
        """Set the poll interval, clamped to [5, 60]."""
        self._config.poll_interval = max(5.0, min(60.0, seconds))

    @property
    def poll_interval(self) -> float:
        return self._config.poll_interval

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def subscribed_tickers(self) -> list[str]:
        return sorted(self._subscribed_tickers)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    async def register_connection(self, connection_id: str, ws: WebSocket) -> None:
        """Register a new WebSocket connection."""
        if len(self._connections) >= self._config.max_connections:
            await ws.close(code=1013, reason="Max connections reached")
            return
        self._connections[connection_id] = ws
        self._last_send[connection_id] = 0.0
        logger.info(
            "WebSocket connected: %s (total=%d)",
            connection_id[:8],
            len(self._connections),
        )

    async def unregister_connection(self, connection_id: str) -> None:
        """Remove a disconnected WebSocket."""
        self._connections.pop(connection_id, None)
        self._last_send.pop(connection_id, None)
        logger.info("WebSocket disconnected: %s", connection_id[:8])

    # ------------------------------------------------------------------ #
    # Status / snapshot helpers
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict[str, Any]:
        """Return current streaming state for MCP tools / WebSocket status."""
        return {
            "running": self._running,
            "poll_interval": self._config.poll_interval,
            "subscribed_tickers": self.subscribed_tickers,
            "connection_count": self.connection_count,
            "last_prices": dict(self._last_prices),
            "poll_count": self._poll_count,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

    def get_alert_state(self) -> dict[str, Any]:
        """Return current alert state for the MCP resource."""
        return {
            "running": self._running,
            "subscribed_tickers": self.subscribed_tickers,
            "alerts_by_ticker": {
                t: sorted(alerts) for t, alerts in self._last_alert_states.items()
            },
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

    async def get_price_snapshot(self, tickers: list[str]) -> dict[str, Any]:
        """One-shot price fetch + alert evaluation (works on all transports).

        This is the STDIO-compatible alternative to WebSocket streaming.
        """
        symbols = [t.strip().upper() for t in tickers if t.strip()]
        if not symbols:
            return {"error": "No tickers provided", "results": {}}

        prices = await self._fetch_prices(symbols)
        results: dict[str, Any] = {}

        for sym in symbols:
            price_data = prices.get(sym)
            if price_data is None:
                results[sym] = {"error": f"No data for {sym}"}
                continue

            # Evaluate alerts
            alerts = await self._evaluate_alerts_for_ticker(sym)
            results[sym] = {
                **price_data,
                "alerts": alerts,
            }

        return {
            "results": results,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Polling loop (private)
    # ------------------------------------------------------------------ #

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until ``stop()`` is called."""
        last_heartbeat = time.monotonic()

        while self._running:
            try:
                if self._subscribed_tickers:
                    self._poll_count += 1
                    tickers = list(self._subscribed_tickers)

                    # Fetch prices
                    prices = await self._fetch_prices(tickers)

                    # Broadcast price updates
                    now_iso = datetime.now(tz=UTC).isoformat()
                    for sym, data in prices.items():
                        prev_price = self._last_prices.get(sym)
                        self._last_prices[sym] = data["price"]

                        # Skip if below change threshold
                        if (
                            self._config.price_change_threshold > 0
                            and prev_price is not None
                            and prev_price != 0
                        ):
                            change_pct = (
                                abs(data["price"] - prev_price) / prev_price * 100
                            )
                            if change_pct < self._config.price_change_threshold:
                                continue

                        await self._broadcast(
                            {
                                "type": "price_update",
                                "ticker": sym,
                                "price": data["price"],
                                "change": data.get("change", 0),
                                "change_pct": data.get("change_pct", 0),
                                "volume": data.get("volume", 0),
                                "timestamp": now_iso,
                            }
                        )

                    # Alert evaluation (every Nth cycle)
                    if self._poll_count % self._config.alert_eval_every_n_polls == 0:
                        for sym in tickers:
                            await self._evaluate_and_broadcast_alerts(sym)

                # Heartbeat
                elapsed = time.monotonic() - last_heartbeat
                if elapsed >= self._config.heartbeat_interval:
                    last_heartbeat = time.monotonic()
                    await self._broadcast(
                        {
                            "type": "heartbeat",
                            "timestamp": datetime.now(tz=UTC).isoformat(),
                            "subscriptions": len(self._subscribed_tickers),
                            "connections": len(self._connections),
                        }
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in price stream poll loop")

            await asyncio.sleep(self._config.poll_interval)

    # ------------------------------------------------------------------ #
    # Price fetching (private)
    # ------------------------------------------------------------------ #

    async def _fetch_prices(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest prices for *tickers* using YFinancePool in a thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_fetch_prices, tickers)

    def _sync_fetch_prices(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        """Blocking price fetch via yfinance (runs in thread executor)."""
        from maverick_mcp.utils.yfinance_pool import get_yfinance_pool

        pool = get_yfinance_pool()
        results: dict[str, dict[str, Any]] = {}

        try:
            if len(tickers) == 1:
                # Single ticker — use get_ticker for speed
                sym = tickers[0]
                ticker_obj = pool.get_ticker(sym)
                info = ticker_obj.fast_info
                price = float(getattr(info, "last_price", 0) or 0)
                prev_close = float(getattr(info, "previous_close", 0) or 0)
                volume = int(getattr(info, "last_volume", 0) or 0)
                change = price - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0
                results[sym] = {
                    "price": round(price, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": volume,
                    "previous_close": round(prev_close, 2),
                }
            else:
                # Multiple tickers — batch download for efficiency
                df = pool.batch_download(tickers, period="1d", interval="1d")
                if df is not None and not df.empty:
                    for sym in tickers:
                        try:
                            if sym in df.columns.get_level_values(0):
                                close = float(df[sym]["Close"].iloc[-1])
                                volume = int(df[sym]["Volume"].iloc[-1])
                                # Get previous close from ticker for change calc
                                t_obj = pool.get_ticker(sym)
                                prev_close = float(
                                    getattr(t_obj.fast_info, "previous_close", 0) or 0
                                )
                                change = close - prev_close if prev_close else 0
                                change_pct = (
                                    (change / prev_close * 100) if prev_close else 0
                                )
                                results[sym] = {
                                    "price": round(close, 2),
                                    "change": round(change, 2),
                                    "change_pct": round(change_pct, 2),
                                    "volume": volume,
                                    "previous_close": round(prev_close, 2),
                                }
                        except Exception as e:
                            logger.debug("Failed to extract data for %s: %s", sym, e)
        except Exception:
            logger.exception("Error fetching prices for %s", tickers)

        return results

    # ------------------------------------------------------------------ #
    # Alert evaluation (private)
    # ------------------------------------------------------------------ #

    async def _evaluate_alerts_for_ticker(self, ticker: str) -> list[dict[str, Any]]:
        """Evaluate watchlist alerts for a single ticker (runs blocking code in thread)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_evaluate_alerts, ticker)

    def _sync_evaluate_alerts(self, ticker: str) -> list[dict[str, Any]]:
        """Blocking alert evaluation using watchlist_monitor functions."""
        try:
            from maverick_mcp.core.watchlist_monitor import (
                _fetch_analysis_data,
                evaluate_alerts,
            )

            df = _fetch_analysis_data(ticker, days=90)
            if df is None:
                return []
            return evaluate_alerts(df)
        except Exception as e:
            logger.debug("Alert evaluation failed for %s: %s", ticker, e)
            return []

    async def _evaluate_and_broadcast_alerts(self, ticker: str) -> None:
        """Evaluate alerts and broadcast only on state changes."""
        current_alerts = await self._evaluate_alerts_for_ticker(ticker)
        current_types = {a["type"] for a in current_alerts}
        previous_types = self._last_alert_states.get(ticker, set())

        new_alerts = current_types - previous_types
        resolved_alerts = previous_types - current_types

        if new_alerts or resolved_alerts:
            self._last_alert_states[ticker] = current_types
            await self._broadcast(
                {
                    "type": "alert_change",
                    "ticker": ticker,
                    "new_alerts": [
                        a for a in current_alerts if a["type"] in new_alerts
                    ],
                    "resolved_alerts": sorted(resolved_alerts),
                    "active_alerts": current_alerts,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
            )
        else:
            # Update state even if no broadcast needed
            self._last_alert_states[ticker] = current_types

    # ------------------------------------------------------------------ #
    # Broadcast (private)
    # ------------------------------------------------------------------ #

    async def _broadcast(self, message: dict[str, Any]) -> None:
        """Send *message* to all connected WebSocket clients, respecting rate limits."""
        now = time.monotonic()
        min_interval = 1.0 / self._config.max_messages_per_second
        dead_connections: list[str] = []

        for conn_id, ws in list(self._connections.items()):
            # Rate limiting
            last = self._last_send.get(conn_id, 0.0)
            if (now - last) < min_interval:
                continue

            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(message)
                    self._last_send[conn_id] = now
                else:
                    dead_connections.append(conn_id)
            except Exception:
                dead_connections.append(conn_id)

        # Clean up dead connections
        for conn_id in dead_connections:
            self._connections.pop(conn_id, None)
            self._last_send.pop(conn_id, None)
