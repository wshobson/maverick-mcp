"""MCP tools for controlling the real-time price streaming service.

All functions are registered on the main FastMCP instance via
``tool_registry.register_streaming_tools()``.

The ``get_price_snapshot`` tool works on every transport (including STDIO)
by performing a one-shot price + alert fetch without requiring WebSocket.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("maverick_mcp.streaming.tools")


def _get_manager():
    """Lazy import to avoid circular dependencies at module load."""
    from maverick_mcp.streaming.price_stream_manager import PriceStreamManager

    return PriceStreamManager.get_instance()


# ------------------------------------------------------------------ #
# Stream lifecycle
# ------------------------------------------------------------------ #


async def start_price_stream(
    tickers: list[str],
    interval: float = 15.0,
) -> dict[str, Any]:
    """Start real-time price streaming for the given tickers.

    Begins a background polling loop that fetches prices from yfinance
    and evaluates watchlist alerts. When a WebSocket client connects to
    ``/ws/prices``, it receives live ``price_update`` and ``alert_change``
    messages automatically.

    If the server is running in STDIO transport (no WebSocket support),
    this tool returns a one-shot price snapshot instead.

    Args:
        tickers: List of stock ticker symbols to stream (e.g. ["AAPL", "MSFT"]).
        interval: Poll interval in seconds (5-60, default 15).

    Returns:
        Dictionary with streaming status or price snapshot.
    """
    manager = _get_manager()
    manager.set_poll_interval(interval)
    await manager.start(tickers)

    return {
        "status": "streaming_started",
        "websocket_url": "/ws/prices",
        "message": (
            f"Streaming {len(tickers)} ticker(s) every {manager.poll_interval}s. "
            "Connect via WebSocket at /ws/prices to receive live updates, "
            "or use streaming_get_price_snapshot for on-demand checks."
        ),
        **manager.get_status(),
    }


async def stop_price_stream() -> dict[str, Any]:
    """Stop the real-time price streaming service.

    Cancels the background polling loop and disconnects all WebSocket clients.

    Returns:
        Dictionary confirming the stream has stopped.
    """
    manager = _get_manager()
    was_running = manager.is_running
    await manager.stop()
    return {
        "status": "stopped",
        "was_running": was_running,
        "message": "Price stream stopped. All WebSocket connections closed.",
    }


# ------------------------------------------------------------------ #
# Subscription management
# ------------------------------------------------------------------ #


async def subscribe(tickers: list[str]) -> dict[str, Any]:
    """Add tickers to the active price stream.

    If the stream is not running, it will be started automatically.

    Args:
        tickers: Ticker symbols to add (e.g. ["GOOGL", "TSLA"]).

    Returns:
        Dictionary with updated subscription list.
    """
    manager = _get_manager()
    manager.subscribe(tickers)
    if not manager.is_running and manager.subscribed_tickers:
        await manager.start()
    return {
        "status": "subscribed",
        "subscribed_tickers": manager.subscribed_tickers,
        "message": f"Now streaming {len(manager.subscribed_tickers)} ticker(s).",
    }


async def unsubscribe(tickers: list[str]) -> dict[str, Any]:
    """Remove tickers from the active price stream.

    If no tickers remain, the stream continues running (with heartbeats only).

    Args:
        tickers: Ticker symbols to remove.

    Returns:
        Dictionary with updated subscription list.
    """
    manager = _get_manager()
    manager.unsubscribe(tickers)
    return {
        "status": "unsubscribed",
        "subscribed_tickers": manager.subscribed_tickers,
        "message": f"Now streaming {len(manager.subscribed_tickers)} ticker(s).",
    }


# ------------------------------------------------------------------ #
# Status / configuration
# ------------------------------------------------------------------ #


async def get_stream_status() -> dict[str, Any]:
    """Get current state of the price streaming service.

    Returns:
        Dictionary with running state, poll interval, subscribed tickers,
        connected clients, last known prices, and poll count.
    """
    manager = _get_manager()
    return manager.get_status()


async def set_poll_interval(seconds: float) -> dict[str, Any]:
    """Change the price polling interval.

    The value is clamped to the range [5, 60] seconds.

    Args:
        seconds: New polling interval in seconds.

    Returns:
        Dictionary confirming the new interval.
    """
    manager = _get_manager()
    manager.set_poll_interval(seconds)
    return {
        "status": "interval_updated",
        "poll_interval": manager.poll_interval,
        "message": f"Poll interval set to {manager.poll_interval}s.",
    }


# ------------------------------------------------------------------ #
# Snapshot (works on ALL transports including STDIO)
# ------------------------------------------------------------------ #


async def get_price_snapshot(tickers: list[str]) -> dict[str, Any]:
    """Get a one-shot price and alert snapshot for the given tickers.

    This tool works on all transports including STDIO. It fetches the
    current price and evaluates watchlist alerts for each ticker,
    returning the same data format as WebSocket ``price_update`` messages.

    Use this when WebSocket streaming is not available or when you need
    a single point-in-time check.

    Args:
        tickers: List of stock ticker symbols (e.g. ["AAPL", "MSFT"]).

    Returns:
        Dictionary with per-ticker price data, change metrics, and active alerts.
    """
    manager = _get_manager()
    return await manager.get_price_snapshot(tickers)
