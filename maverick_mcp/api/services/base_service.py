"""
Base service class for MaverickMCP API services.

Provides common functionality and dependency injection patterns
for all service classes.
"""

from abc import ABC, abstractmethod
from typing import Any

from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

# Auth imports removed in personal use version
# from maverick_mcp.auth.jwt_enhanced import EnhancedJWTManager
# from maverick_mcp.auth.key_manager_jwt import KeyManager
from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger


class BaseService(ABC):
    """
    Base service class providing common functionality for all services.

    This class implements dependency injection patterns and provides
    shared utilities that all services need.
    """

    def __init__(
        self,
        mcp: FastMCP,
        db_session_factory: Any = None,
    ):
        """
        Initialize base service with dependencies.

        Args:
            mcp: FastMCP instance for tool/resource registration
            db_session_factory: Optional async database session factory
        """
        self.mcp = mcp
        self.db_session_factory = db_session_factory
        self.logger = get_logger(
            f"maverick_mcp.services.{self.__class__.__name__.lower()}"
        )

    @property
    def settings(self):
        """Get application settings."""
        return settings

    async def get_db_session(self) -> AsyncSession:
        """
        Get async database session.

        Returns:
            AsyncSession instance

        Raises:
            RuntimeError: If database session factory not available
        """
        if not self.db_session_factory:
            raise RuntimeError("Database session factory not configured")
        return self.db_session_factory()

    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return False  # Auth disabled in personal use version

    def is_credit_enabled(self) -> bool:
        """Check if credit system is enabled."""
        return False  # Credit system disabled in personal use version

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return settings.api.debug

    def log_tool_usage(self, tool_name: str, user_id: int | None = None, **kwargs):
        """
        Log tool usage for monitoring purposes.

        Args:
            tool_name: Name of the tool being used
            user_id: Optional user ID if authenticated
            **kwargs: Additional context for logging
        """
        context = {
            "tool_name": tool_name,
            "user_id": user_id,
            "auth_enabled": self.is_auth_enabled(),
            "credit_enabled": self.is_credit_enabled(),
            **kwargs,
        }
        self.logger.info(f"Tool usage: {tool_name}", extra=context)

    @abstractmethod
    def register_tools(self):
        """
        Register service tools with the MCP instance.

        This method should be implemented by subclasses to register
        their specific tools and resources.
        """
        pass
