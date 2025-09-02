"""
Enhanced connection management for FastMCP server stability.

Provides session persistence, connection monitoring, and tool registration consistency
to prevent tools from disappearing in Claude Desktop.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class ConnectionSession:
    """Represents a single MCP connection session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.tools_registered = False
        self.is_active = True

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self, timeout: float = 300.0) -> bool:
        """Check if session is expired (default 5 minutes)."""
        return time.time() - self.last_activity > timeout


class MCPConnectionManager:
    """
    Enhanced connection manager for FastMCP server stability.

    Features:
    - Single connection initialization pattern
    - Session persistence across reconnections
    - Tool registration consistency
    - Connection monitoring and debugging
    - Automatic cleanup of stale sessions
    """

    def __init__(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server
        self.active_sessions: dict[str, ConnectionSession] = {}
        self.tools_registered = False
        self.initialization_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

        # Connection monitoring
        self.connection_count = 0
        self.total_connections = 0
        self.failed_connections = 0

        logger.info("MCP Connection Manager initialized")

    async def handle_new_connection(self, session_id: str | None = None) -> str:
        """
        Handle new MCP connection with single initialization pattern.

        Args:
            session_id: Optional session ID, generates new one if not provided

        Returns:
            Session ID for the connection
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        async with self.initialization_lock:
            # Create new session
            session = ConnectionSession(session_id)
            self.active_sessions[session_id] = session
            self.connection_count += 1
            self.total_connections += 1

            logger.info(
                f"New MCP connection: {session_id[:8]} "
                f"(active: {self.connection_count}, total: {self.total_connections})"
            )

            # Ensure tools are registered only once
            if not self.tools_registered:
                await self._register_tools_once()
                self.tools_registered = True
                session.tools_registered = True
                logger.info("Tools registered for first connection")
            else:
                logger.info("Tools already registered, skipping registration")

            # Start cleanup task if not already running
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        return session_id

    async def handle_connection_close(self, session_id: str):
        """Handle connection close event."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            self.connection_count = max(0, self.connection_count - 1)

            logger.info(
                f"Connection closed: {session_id[:8]} (active: {self.connection_count})"
            )

            # Remove session after delay to handle quick reconnections
            await asyncio.sleep(5.0)
            self.active_sessions.pop(session_id, None)

    async def update_session_activity(self, session_id: str):
        """Update session activity timestamp."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].update_activity()

    async def _register_tools_once(self):
        """Register tools only once to prevent conflicts."""
        try:
            from maverick_mcp.api.routers.tool_registry import register_all_router_tools

            register_all_router_tools(self.mcp_server)
            logger.info("Successfully registered all MCP tools")
        except Exception as e:
            logger.error(f"Failed to register tools: {e}")
            self.failed_connections += 1
            raise

    async def _cleanup_loop(self):
        """Background cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                current_time = time.time()
                expired_sessions = [
                    sid
                    for sid, session in self.active_sessions.items()
                    if session.is_expired()
                ]

                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id[:8]}")
                    self.active_sessions.pop(session_id, None)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def get_connection_status(self) -> dict[str, Any]:
        """Get current connection status for monitoring."""
        active_count = sum(1 for s in self.active_sessions.values() if s.is_active)

        return {
            "active_connections": active_count,
            "total_sessions": len(self.active_sessions),
            "total_connections": self.total_connections,
            "failed_connections": self.failed_connections,
            "tools_registered": self.tools_registered,
            "sessions": [
                {
                    "id": sid[:8],
                    "active": session.is_active,
                    "age_seconds": time.time() - session.created_at,
                    "last_activity": time.time() - session.last_activity,
                }
                for sid, session in self.active_sessions.items()
            ],
        }

    async def shutdown(self):
        """Cleanup on server shutdown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("MCP Connection Manager shutdown complete")


# Global connection manager instance
_connection_manager: MCPConnectionManager | None = None


def get_connection_manager(mcp_server: FastMCP) -> MCPConnectionManager:
    """Get or create the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = MCPConnectionManager(mcp_server)
    return _connection_manager


async def initialize_connection_management(mcp_server: FastMCP) -> MCPConnectionManager:
    """Initialize enhanced connection management."""
    manager = get_connection_manager(mcp_server)
    logger.info("Enhanced MCP connection management initialized")
    return manager
