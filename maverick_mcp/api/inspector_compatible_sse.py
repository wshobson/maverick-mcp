"""
MCP Inspector-compatible SSE implementation.

This implements a proper bidirectional SSE handler that works with MCP Inspector,
handling JSON-RPC messages directly over the SSE connection.
"""

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import StreamingResponse

from maverick_mcp.api.server import mcp

logger = logging.getLogger(__name__)


class InspectorCompatibleSSE:
    """SSE handler that properly implements MCP protocol for Inspector."""

    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        self.message_queues: dict[str, asyncio.Queue] = {}

    async def handle_sse(self, request: Request):
        """Handle SSE connection from MCP Inspector."""
        session_id = str(uuid4())
        logger.info(f"New Inspector SSE connection: {session_id}")

        # Create a message queue for this session
        message_queue: asyncio.Queue = asyncio.Queue()
        self.message_queues[session_id] = message_queue

        # Create a simple session state tracker
        session_state = {
            "initialized": False,
            "server_name": "MaverickMCP",
            "server_version": "1.0.0",
            "protocol_version": "2024-11-05",
        }
        self.sessions[session_id] = session_state

        async def event_generator():
            """Generate SSE events and handle bidirectional communication."""
            try:
                # Send initial connection event with session info
                connection_msg = {
                    "type": "connection",
                    "sessionId": session_id,
                    "endpoint": f"/inspector/message?session_id={session_id}",
                }
                yield f"data: {json.dumps(connection_msg)}\n\n"

                # Process incoming messages from the queue
                while True:
                    try:
                        # Wait for messages with timeout for keepalive
                        message = await asyncio.wait_for(
                            message_queue.get(), timeout=30.0
                        )

                        # Process the message through MCP session
                        if isinstance(message, dict) and "jsonrpc" in message:
                            # Handle the JSON-RPC request
                            response = await self._process_message(
                                session_state, message
                            )
                            if response:
                                yield f"data: {json.dumps(response)}\n\n"

                    except TimeoutError:
                        # Send keepalive
                        yield ": keepalive\n\n"
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {"code": -32603, "message": str(e)},
                            "id": None,
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"

            finally:
                # Cleanup on disconnect
                if session_id in self.sessions:
                    del self.sessions[session_id]
                if session_id in self.message_queues:
                    del self.message_queues[session_id]
                logger.info(f"Inspector SSE connection closed: {session_id}")

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
            },
        )

    async def handle_message(self, request: Request):
        """Handle incoming JSON-RPC messages from Inspector."""
        session_id = request.query_params.get("session_id")
        if not session_id or session_id not in self.message_queues:
            return {"error": "Invalid or missing session_id"}

        try:
            message = await request.json()
            logger.info(f"Inspector message for session {session_id}: {message}")

            # Put message in queue for processing
            await self.message_queues[session_id].put(message)

            # Return acknowledgment
            return {"status": "queued"}

        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return {"error": str(e)}

    async def _process_message(
        self, session_state: dict[str, Any], message: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process a JSON-RPC message through the MCP session."""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        try:
            # Handle different MCP methods
            if method == "initialize":
                # Mark session as initialized
                session_state["initialized"] = True
                protocol_version = params.get("protocolVersion", "2024-11-05")

                # Get server capabilities
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": protocol_version,
                        "capabilities": {
                            "tools": {"listChanged": True}
                            if hasattr(mcp, "_tool_manager")
                            and hasattr(mcp._tool_manager, "tools")
                            and mcp._tool_manager.tools
                            else {},
                            "resources": {"listChanged": True}
                            if hasattr(mcp, "_resource_manager")
                            and hasattr(mcp._resource_manager, "resources")
                            and mcp._resource_manager.resources
                            else {},
                            "prompts": {"listChanged": True}
                            if hasattr(mcp, "_prompt_manager")
                            and hasattr(mcp._prompt_manager, "prompts")
                            and mcp._prompt_manager.prompts
                            else {},
                        },
                        "serverInfo": {
                            "name": session_state["server_name"],
                            "version": session_state["server_version"],
                        },
                    },
                }

            elif method == "tools/list":
                # List available tools
                tools = []
                if (
                    hasattr(mcp, "_tool_manager")
                    and hasattr(mcp._tool_manager, "tools")
                    and hasattr(mcp._tool_manager.tools, "items")
                ):
                    for tool_name, tool_func in mcp._tool_manager.tools.items():
                        tools.append(
                            {
                                "name": tool_name,
                                "description": tool_func.__doc__ or "No description",
                                "inputSchema": getattr(tool_func, "input_schema", {}),
                            }
                        )

                return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}

            elif method == "resources/list":
                # List available resources
                resources = []
                if (
                    hasattr(mcp, "_resource_manager")
                    and hasattr(mcp._resource_manager, "resources")
                    and hasattr(mcp._resource_manager.resources, "items")
                ):
                    for (
                        resource_uri,
                        resource_func,
                    ) in mcp._resource_manager.resources.items():
                        resources.append(
                            {
                                "uri": resource_uri,
                                "name": getattr(
                                    resource_func, "__name__", str(resource_func)
                                ),
                                "description": getattr(resource_func, "__doc__", None)
                                or "No description",
                            }
                        )

                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"resources": resources},
                }

            elif method == "tools/call":
                # Call a tool
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})

                if (
                    hasattr(mcp, "_tool_manager")
                    and hasattr(mcp._tool_manager, "tools")
                    and hasattr(mcp._tool_manager.tools, "__contains__")
                    and tool_name in mcp._tool_manager.tools
                ):
                    tool_func = mcp._tool_manager.tools[tool_name]
                    try:
                        # Execute the tool
                        result = await tool_func(**tool_args)

                        return {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(result, default=str),
                                    }
                                ]
                            },
                        }
                    except Exception as tool_error:
                        return {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {
                                "code": -32603,
                                "message": f"Tool execution error: {str(tool_error)}",
                            },
                        }
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool not found: {tool_name}",
                        },
                    }

            else:
                # Method not found
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.error(f"Error processing {method}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)},
            }


# Create global handler instance
inspector_sse = InspectorCompatibleSSE()
