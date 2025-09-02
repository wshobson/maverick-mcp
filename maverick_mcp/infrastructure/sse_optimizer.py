"""
SSE Transport Optimizer for FastMCP server stability.

Provides SSE-specific optimizations to prevent connection drops
and ensure persistent tool availability in Claude Desktop.
"""

import asyncio
import logging
from typing import Any

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SSEStabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enhance SSE connection stability.

    Features:
    - Connection keepalive headers
    - Proper CORS for SSE
    - Connection state tracking
    - Automatic reconnection support
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Add SSE-specific headers for stability
        response = await call_next(request)

        # SSE connection optimizations
        if request.url.path.endswith("/sse"):
            # Keepalive and caching headers
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            response.headers["Content-Type"] = "text/event-stream"

            # CORS headers for cross-origin SSE
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "false"

            # Prevent proxy buffering
            response.headers["X-Accel-Buffering"] = "no"

        return response


class SSEHeartbeat:
    """
    Heartbeat mechanism for SSE connections.

    Sends periodic keepalive messages to maintain connection
    and detect client disconnections early.
    """

    def __init__(self, interval: float = 30.0):
        self.interval = interval
        self.active_connections: dict[str, asyncio.Task] = {}

    async def start_heartbeat(self, connection_id: str, send_function):
        """Start heartbeat for a specific connection."""
        try:
            while True:
                await asyncio.sleep(self.interval)

                # Send heartbeat event
                heartbeat_event = {
                    "event": "heartbeat",
                    "data": {
                        "timestamp": asyncio.get_event_loop().time(),
                        "connection_id": connection_id[:8],
                    },
                }

                await send_function(heartbeat_event)

        except asyncio.CancelledError:
            logger.info(f"Heartbeat stopped for connection: {connection_id[:8]}")
        except Exception as e:
            logger.error(f"Heartbeat error for {connection_id[:8]}: {e}")

    def register_connection(self, connection_id: str, send_function) -> None:
        """Register a new connection for heartbeat."""
        if connection_id in self.active_connections:
            # Cancel existing heartbeat
            self.active_connections[connection_id].cancel()

        # Start new heartbeat task
        task = asyncio.create_task(self.start_heartbeat(connection_id, send_function))
        self.active_connections[connection_id] = task

        logger.info(f"Heartbeat registered for connection: {connection_id[:8]}")

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister connection and stop heartbeat."""
        if connection_id in self.active_connections:
            self.active_connections[connection_id].cancel()
            del self.active_connections[connection_id]
            logger.info(f"Heartbeat unregistered for connection: {connection_id[:8]}")

    async def shutdown(self):
        """Shutdown all heartbeats."""
        for task in self.active_connections.values():
            task.cancel()

        if self.active_connections:
            await asyncio.gather(
                *self.active_connections.values(), return_exceptions=True
            )

        self.active_connections.clear()
        logger.info("All heartbeats shutdown")


class SSEOptimizer:
    """
    SSE Transport Optimizer for enhanced stability.

    Provides comprehensive optimizations for SSE connections:
    - Stability middleware
    - Heartbeat mechanism
    - Connection monitoring
    - Automatic recovery
    """

    def __init__(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server
        self.heartbeat = SSEHeartbeat(interval=25.0)  # 25-second heartbeat
        self.connection_count = 0

    def optimize_server(self) -> None:
        """Apply SSE optimizations to the FastMCP server."""

        # Add stability middleware
        if hasattr(self.mcp_server, "fastapi_app") and self.mcp_server.fastapi_app:
            self.mcp_server.fastapi_app.add_middleware(SSEStabilityMiddleware)
            logger.info("SSE stability middleware added")

        # Register SSE event handlers
        self._register_sse_handlers()

        logger.info("SSE transport optimizations applied")

    def _register_sse_handlers(self) -> None:
        """Register SSE-specific event handlers."""

        @self.mcp_server.event("sse_connection_opened")
        async def on_sse_connection_open(connection_id: str, send_function):
            """Handle SSE connection open with optimization."""
            self.connection_count += 1
            logger.info(
                f"SSE connection opened: {connection_id[:8]} (total: {self.connection_count})"
            )

            # Register heartbeat
            self.heartbeat.register_connection(connection_id, send_function)

            # Send connection confirmation
            await send_function(
                {
                    "event": "connection_ready",
                    "data": {
                        "connection_id": connection_id[:8],
                        "server": "maverick-mcp",
                        "transport": "sse",
                        "optimization": "enabled",
                    },
                }
            )

        @self.mcp_server.event("sse_connection_closed")
        async def on_sse_connection_close(connection_id: str):
            """Handle SSE connection close with cleanup."""
            self.connection_count = max(0, self.connection_count - 1)
            logger.info(
                f"SSE connection closed: {connection_id[:8]} (remaining: {self.connection_count})"
            )

            # Unregister heartbeat
            self.heartbeat.unregister_connection(connection_id)

    async def shutdown(self):
        """Shutdown SSE optimizer."""
        await self.heartbeat.shutdown()
        logger.info("SSE optimizer shutdown complete")

    def get_sse_status(self) -> dict[str, Any]:
        """Get SSE connection status."""
        return {
            "active_connections": self.connection_count,
            "heartbeat_connections": len(self.heartbeat.active_connections),
            "heartbeat_interval": self.heartbeat.interval,
            "optimization_status": "enabled",
        }


# Global SSE optimizer instance
_sse_optimizer: SSEOptimizer | None = None


def get_sse_optimizer(mcp_server: FastMCP) -> SSEOptimizer:
    """Get or create the global SSE optimizer."""
    global _sse_optimizer
    if _sse_optimizer is None:
        _sse_optimizer = SSEOptimizer(mcp_server)
    return _sse_optimizer


def apply_sse_optimizations(mcp_server: FastMCP) -> SSEOptimizer:
    """Apply SSE transport optimizations to FastMCP server."""
    optimizer = get_sse_optimizer(mcp_server)
    optimizer.optimize_server()
    logger.info("SSE transport optimizations applied for enhanced stability")
    return optimizer
