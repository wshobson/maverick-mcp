"""
MCP Inspector-compatible SSE handler.

This module implements an SSE handler that's compatible with MCP Inspector's
expectations, where JSON-RPC messages are exchanged directly over the SSE
connection rather than via a separate POST endpoint.
"""

import json
import logging
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import StreamingResponse

from maverick_mcp.api.server import mcp

logger = logging.getLogger(__name__)


class InspectorSSEHandler:
    """SSE handler compatible with MCP Inspector."""

    def __init__(self, mcp_instance):
        self.mcp = mcp_instance
        self.sessions = {}

    async def handle_sse(self, request: Request):
        """Handle SSE connection from MCP Inspector."""
        session_id = str(uuid4())
        logger.info(f"New SSE connection: {session_id}")

        async def event_generator():
            """Generate SSE events."""
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'sessionId': session_id})}\n\n"

            # Keep connection alive
            while True:
                # In a real implementation, we'd process incoming messages here
                # For now, just keep the connection alive
                import asyncio

                await asyncio.sleep(30)
                yield ": keepalive\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def handle_message(self, request: Request):
        """Handle JSON-RPC message from client."""
        # Get session ID from query params or headers
        session_id = request.query_params.get("session_id")
        if not session_id:
            return {"error": "Missing session_id"}

        # Get JSON-RPC message
        try:
            message = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"error": "Invalid JSON"}

        logger.info(f"Received message for session {session_id}: {message}")

        # Process the message through MCP
        # This is where we'd integrate with the actual MCP server
        # For now, return a mock response
        if message.get("method") == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {"name": "MaverickMCP", "version": "1.0.0"},
                },
            }

        return {"jsonrpc": "2.0", "id": message.get("id"), "result": {}}


# Create global handler instance
inspector_handler = InspectorSSEHandler(mcp)
