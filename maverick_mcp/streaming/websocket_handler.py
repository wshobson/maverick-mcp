"""FastAPI WebSocket endpoint for real-time price streaming.

Mounted on the FastAPI app at ``/ws/prices`` by ``server.py``.

Client protocol (JSON text frames):

    → {"action": "subscribe",   "tickers": ["AAPL", "MSFT"]}
    → {"action": "unsubscribe", "tickers": ["MSFT"]}
    → {"action": "set_interval","interval": 10}
    → {"action": "status"}

Server pushes ``price_update``, ``alert_change``, and ``heartbeat`` messages.
"""

from __future__ import annotations

import json
import logging
import uuid

from starlette.websockets import WebSocket, WebSocketDisconnect

from maverick_mcp.streaming.price_stream_manager import PriceStreamManager

logger = logging.getLogger("maverick_mcp.streaming.ws")


async def websocket_prices(ws: WebSocket) -> None:
    """Handle a single WebSocket connection for price streaming."""
    await ws.accept()
    connection_id = str(uuid.uuid4())
    manager = PriceStreamManager.get_instance()

    await manager.register_connection(connection_id, ws)

    try:
        # Send initial status on connect
        await ws.send_json(
            {
                "type": "connected",
                "connection_id": connection_id[:8],
                **manager.get_status(),
            }
        )

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action", "")

            if action == "subscribe":
                tickers = msg.get("tickers", [])
                if not isinstance(tickers, list):
                    await ws.send_json(
                        {"type": "error", "message": "'tickers' must be a list"}
                    )
                    continue
                manager.subscribe(tickers)
                # Auto-start the polling loop if not running
                if not manager.is_running:
                    await manager.start()
                await ws.send_json(
                    {
                        "type": "subscribed",
                        "tickers": manager.subscribed_tickers,
                    }
                )

            elif action == "unsubscribe":
                tickers = msg.get("tickers", [])
                if not isinstance(tickers, list):
                    await ws.send_json(
                        {"type": "error", "message": "'tickers' must be a list"}
                    )
                    continue
                manager.unsubscribe(tickers)
                await ws.send_json(
                    {
                        "type": "unsubscribed",
                        "tickers": manager.subscribed_tickers,
                    }
                )

            elif action == "set_interval":
                interval = msg.get("interval")
                if not isinstance(interval, int | float):
                    await ws.send_json(
                        {"type": "error", "message": "'interval' must be a number"}
                    )
                    continue
                manager.set_poll_interval(float(interval))
                await ws.send_json(
                    {
                        "type": "interval_set",
                        "interval": manager.poll_interval,
                    }
                )

            elif action == "status":
                await ws.send_json({"type": "status", **manager.get_status()})

            else:
                await ws.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown action: {action!r}. "
                        "Valid: subscribe, unsubscribe, set_interval, status",
                    }
                )

    except WebSocketDisconnect:
        logger.debug("Client disconnected: %s", connection_id[:8])
    except Exception:
        logger.exception("WebSocket error for %s", connection_id[:8])
    finally:
        await manager.unregister_connection(connection_id)
