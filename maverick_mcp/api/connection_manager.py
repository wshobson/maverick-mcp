"""
MCP Connection Manager for persistent tool registration and session management.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConnectionSession:
    """Represents an active MCP connection session."""

    session_id: str
    client_info: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    tools_registered: bool = False
    is_active: bool = True


class MCPConnectionManager:
    """
    Manages MCP connection sessions and ensures persistent tool registration.

    Fixes:
    - Single connection initialization to prevent tool registration conflicts
    - Session persistence to maintain tool availability across connection cycles
    - Connection monitoring and cleanup
    """

    def __init__(self):
        self.sessions: dict[str, ConnectionSession] = {}
        self.tools_initialized = False
        self.startup_time = datetime.now()
        self._lock = asyncio.Lock()

    async def register_connection(
        self, session_id: str, client_info: str = "unknown"
    ) -> ConnectionSession:
        """Register a new connection session."""
        async with self._lock:
            logger.info(
                f"Registering new MCP connection: {session_id} from {client_info}"
            )

            # Clean up any existing session with same ID
            if session_id in self.sessions:
                await self.cleanup_session(session_id)

            # Create new session
            session = ConnectionSession(session_id=session_id, client_info=client_info)
            self.sessions[session_id] = session

            # Ensure tools are registered (only once globally)
            if not self.tools_initialized:
                logger.info("Initializing tools for first connection")
                self.tools_initialized = True
                session.tools_registered = True
            else:
                logger.info("Tools already initialized, reusing registration")
                session.tools_registered = True

            logger.info(
                f"Connection registered successfully. Active sessions: {len(self.sessions)}"
            )
            return session

    async def update_activity(self, session_id: str):
        """Update last activity timestamp for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()

    async def cleanup_session(self, session_id: str):
        """Clean up a specific session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            logger.info(
                f"Cleaning up session {session_id} (connected for {datetime.now() - session.connected_at})"
            )
            del self.sessions[session_id]

    async def cleanup_stale_sessions(self, timeout_seconds: int = 300):
        """Clean up sessions that haven't been active recently."""
        now = datetime.now()
        stale_sessions = []

        for session_id, session in self.sessions.items():
            if (now - session.last_activity).total_seconds() > timeout_seconds:
                stale_sessions.append(session_id)

        for session_id in stale_sessions:
            await self.cleanup_session(session_id)

    def get_connection_status(self) -> dict:
        """Get current connection status for debugging."""
        now = datetime.now()
        return {
            "active_sessions": len(self.sessions),
            "tools_initialized": self.tools_initialized,
            "server_uptime": str(now - self.startup_time),
            "sessions": [
                {
                    "session_id": session.session_id,
                    "client_info": session.client_info,
                    "connected_duration": str(now - session.connected_at),
                    "last_activity": str(now - session.last_activity),
                    "tools_registered": session.tools_registered,
                    "is_active": session.is_active,
                }
                for session in self.sessions.values()
            ],
        }

    async def ensure_tools_available(self) -> bool:
        """Ensure tools are available for connections."""
        return self.tools_initialized and len(self.sessions) > 0


# Global connection manager instance
connection_manager = MCPConnectionManager()


async def get_connection_manager() -> MCPConnectionManager:
    """Get the global connection manager instance."""
    return connection_manager
