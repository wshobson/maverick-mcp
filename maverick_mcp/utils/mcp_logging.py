"""
Integration of structured logging with FastMCP.

This module provides:
- Automatic request context capture from MCP
- Tool execution logging
- Performance monitoring
- Error tracking
"""

import functools
import time
from collections.abc import Callable
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .logging import (
    PerformanceMonitor,
    get_logger,
    log_cache_operation,
    log_database_query,
    log_external_api_call,
    request_id_var,
    request_start_var,
    tool_name_var,
    user_id_var,
)


def with_logging(tool_name: str | None = None):
    """
    Decorator for FastMCP tools that adds structured logging.

    Automatically logs:
    - Tool invocation with parameters
    - Execution time
    - Success/failure status
    - Context information (request ID, user)

    Example:
        @mcp.tool()
        @with_logging()
        async def fetch_stock_data(context: Context, ticker: str) -> dict:
            # Tool implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context
            context = None
            for arg in args:
                if isinstance(arg, Context):
                    context = arg
                    break

            # Get tool name
            actual_tool_name = tool_name or func.__name__

            # Set context variables
            if context:
                # Extract request ID from context metadata if available
                request_id = getattr(context, "request_id", None) or str(time.time())
                request_id_var.set(request_id)

                # Extract user info if available
                user_id = getattr(context, "user_id", None)
                if user_id:
                    user_id_var.set(user_id)

            tool_name_var.set(actual_tool_name)
            request_start_var.set(time.time())

            # Get logger
            logger = get_logger(f"maverick_mcp.tools.{actual_tool_name}")

            # Log tool invocation
            logger.info(
                f"Tool invoked: {actual_tool_name}",
                extra={
                    "tool_name": actual_tool_name,
                    "has_context": context is not None,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            try:
                # Use context's progress callback if available
                if context and hasattr(context, "report_progress"):
                    await context.report_progress(
                        progress=0, total=100, message=f"Starting {actual_tool_name}"
                    )

                # Execute the tool
                with PerformanceMonitor(f"tool_{actual_tool_name}"):
                    result = await func(*args, **kwargs)

                # Log success
                logger.info(
                    f"Tool completed: {actual_tool_name}",
                    extra={"tool_name": actual_tool_name, "status": "success"},
                )

                # Report completion
                if context and hasattr(context, "report_progress"):
                    await context.report_progress(
                        progress=100, total=100, message=f"Completed {actual_tool_name}"
                    )

                return result

            except ToolError as e:
                # Log tool-specific error
                logger.warning(
                    f"Tool error in {actual_tool_name}: {str(e)}",
                    extra={
                        "tool_name": actual_tool_name,
                        "status": "tool_error",
                        "error_message": str(e),
                    },
                )
                raise

            except Exception as e:
                # Log unexpected error
                logger.error(
                    f"Unexpected error in {actual_tool_name}: {str(e)}",
                    exc_info=True,
                    extra={
                        "tool_name": actual_tool_name,
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                raise

            finally:
                # Clear context vars
                request_id_var.set(None)
                tool_name_var.set(None)
                user_id_var.set(None)
                request_start_var.set(None)

        return wrapper

    return decorator


def log_mcp_context(context: Context, operation: str, **extra):
    """
    Log information from MCP context.

    Args:
        context: FastMCP context object
        operation: Description of the operation
        **extra: Additional fields to log
    """
    logger = get_logger("maverick_mcp.context")

    log_data = {
        "operation": operation,
        "has_request_id": hasattr(context, "request_id"),
        "can_report_progress": hasattr(context, "report_progress"),
        "can_log": hasattr(context, "info"),
    }

    # Add any extra fields
    log_data.update(extra)

    logger.info(f"MCP Context: {operation}", extra=log_data)


class LoggingStockDataProvider:
    """
    Wrapper for StockDataProvider that adds logging.

    This demonstrates how to add logging to existing classes.
    """

    def __init__(self, provider):
        self.provider = provider
        self.logger = get_logger("maverick_mcp.providers.stock_data")

    async def get_stock_data(
        self, ticker: str, start_date: str, end_date: str, **kwargs
    ):
        """Get stock data with logging."""
        with PerformanceMonitor(f"fetch_stock_data_{ticker}"):
            # Check cache first
            cache_key = f"stock:{ticker}:{start_date}:{end_date}"

            # Log cache check
            start = time.time()
            cached_data = await self._check_cache(cache_key)
            cache_duration = int((time.time() - start) * 1000)

            if cached_data:
                log_cache_operation(
                    "get", cache_key, hit=True, duration_ms=cache_duration
                )
                return cached_data
            else:
                log_cache_operation(
                    "get", cache_key, hit=False, duration_ms=cache_duration
                )

            # Fetch from provider
            try:
                start = time.time()
                data = await self.provider.get_stock_data(
                    ticker, start_date, end_date, **kwargs
                )
                api_duration = int((time.time() - start) * 1000)

                log_external_api_call(
                    service="yfinance",
                    endpoint=f"/quote/{ticker}",
                    method="GET",
                    status_code=200,
                    duration_ms=api_duration,
                )

                # Cache the result
                await self._set_cache(cache_key, data)

                return data

            except Exception as e:
                log_external_api_call(
                    service="yfinance",
                    endpoint=f"/quote/{ticker}",
                    method="GET",
                    error=str(e),
                )
                raise

    async def _check_cache(self, key: str):
        """Check cache (placeholder)."""
        # This would integrate with actual cache
        return None

    async def _set_cache(self, key: str, data: Any):
        """Set cache (placeholder)."""
        # This would integrate with actual cache
        pass


# SQL query logging wrapper
class LoggingSession:
    """Wrapper for SQLAlchemy session that logs queries."""

    def __init__(self, session):
        self.session = session
        self.logger = get_logger("maverick_mcp.database")

    def execute(self, query, params=None):
        """Execute query with logging."""
        start = time.time()
        try:
            result = self.session.execute(query, params)
            duration = int((time.time() - start) * 1000)
            log_database_query(str(query), params, duration)
            return result
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            log_database_query(str(query), params, duration)
            self.logger.error(
                f"Database query failed: {str(e)}",
                extra={"query": str(query)[:200], "error_type": type(e).__name__},
            )
            raise

    def __getattr__(self, name):
        """Proxy other methods to the wrapped session."""
        return getattr(self.session, name)


# Example usage in routers
def setup_router_logging(router):
    """
    Add logging middleware to a FastMCP router.

    This should be called when setting up routers.
    """
    logger = get_logger(f"maverick_mcp.routers.{router.__class__.__name__}")

    # Log router initialization
    logger.info(
        "Router initialized",
        extra={
            "router_class": router.__class__.__name__,
            "tool_count": len(getattr(router, "tools", [])),
        },
    )

    # Add middleware to log all requests (if supported by FastMCP)
    # This is a placeholder for when FastMCP supports middleware
    pass
