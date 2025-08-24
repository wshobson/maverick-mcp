"""
Simple SSE implementation for MCP Inspector compatibility.

This implements a direct SSE handler that works with MCP Inspector's expectations.
"""

import asyncio
import logging
from uuid import uuid4

from mcp import types
from mcp.server.session import ServerSession
from starlette.requests import Request
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class SimpleSSEHandler:
    """Simple SSE handler for MCP Inspector."""

    def __init__(self):
        self.sessions: dict[str, ServerSession] = {}

    async def handle_sse(self, request: Request):
        """Handle SSE connection with bidirectional JSON-RPC over SSE."""
        session_id = str(uuid4())
        logger.info(f"New Simple SSE connection: {session_id}")

        # Create MCP session
        session = ServerSession(
            create_initialization_options=lambda: types.InitializationOptions(
                server_name="MaverickMCP", server_version="1.0.0"
            )
        )
        self.sessions[session_id] = session

        async def event_generator():
            """Generate SSE events."""
            try:
                # Just keep the connection alive - Inspector will send messages via POST
                while True:
                    # Send keepalive every 30 seconds
                    await asyncio.sleep(30)
                    yield ": keepalive\n\n"

            finally:
                # Cleanup on disconnect
                if session_id in self.sessions:
                    del self.sessions[session_id]
                logger.info(f"Simple SSE connection closed: {session_id}")

        # Return SSE response with proper headers
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            },
        )


# Create global handler instance
simple_sse = SimpleSSEHandler()
