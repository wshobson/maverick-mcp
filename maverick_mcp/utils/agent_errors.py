"""
Agent-friendly error handler with helpful fix suggestions.

This module provides decorators and utilities to catch common errors
and provide actionable solutions for agents.
"""

import asyncio
import functools
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Common error patterns and their fixes
ERROR_FIXES = {
    # DataFrame column errors
    "KeyError.*close": {
        "error": "DataFrame column 'close' not found",
        "fix": "Use 'Close' with capital C - DataFrame columns are case-sensitive",
        "example": "df['Close'] not df['close']",
    },
    "KeyError.*open": {
        "error": "DataFrame column 'open' not found",
        "fix": "Use 'Open' with capital O",
        "example": "df['Open'] not df['open']",
    },
    "KeyError.*high": {
        "error": "DataFrame column 'high' not found",
        "fix": "Use 'High' with capital H",
        "example": "df['High'] not df['high']",
    },
    "KeyError.*low": {
        "error": "DataFrame column 'low' not found",
        "fix": "Use 'Low' with capital L",
        "example": "df['Low'] not df['low']",
    },
    "KeyError.*volume": {
        "error": "DataFrame column 'volume' not found",
        "fix": "Use 'Volume' with capital V",
        "example": "df['Volume'] not df['volume']",
    },
    # Authentication errors
    "401.*Unauthorized": {
        "error": "Authentication required",
        "fix": "Set AUTH_ENABLED=false for development or use generate_dev_token tool",
        "example": "AUTH_ENABLED=false python -m maverick_mcp.api.server",
    },
    "402.*Payment Required": {
        "error": "Insufficient credits",
        "fix": "Set CREDIT_SYSTEM_ENABLED=false for development",
        "example": "CREDIT_SYSTEM_ENABLED=false python -m maverick_mcp.api.server",
    },
    # Connection errors
    "Redis.*Connection.*refused": {
        "error": "Redis connection failed",
        "fix": "Start Redis: brew services start redis",
        "example": "Or set REDIS_HOST=none to skip caching",
    },
    "psycopg2.*could not connect": {
        "error": "PostgreSQL connection failed",
        "fix": "Use SQLite for development: DATABASE_URL=sqlite:///dev.db",
        "example": "Or start PostgreSQL: brew services start postgresql",
    },
    # Import errors
    "ModuleNotFoundError.*maverick": {
        "error": "Maverick MCP modules not found",
        "fix": "Install dependencies: uv sync",
        "example": "Make sure you're in the project root directory",
    },
    "ImportError.*ta_lib": {
        "error": "TA-Lib not installed",
        "fix": "Install TA-Lib: brew install ta-lib && uv pip install ta-lib",
        "example": "TA-Lib requires system libraries",
    },
    # Type errors
    "TypeError.*NoneType.*has no attribute": {
        "error": "Trying to access attribute on None",
        "fix": "Check if the object exists before accessing attributes",
        "example": "if obj is not None: obj.attribute",
    },
    # Value errors
    "ValueError.*not enough values to unpack": {
        "error": "Unpacking mismatch",
        "fix": "Check the return value - it might be None or have fewer values",
        "example": "result = func(); if result: a, b = result",
    },
    # Async errors
    "RuntimeError.*no running event loop": {
        "error": "Async function called without event loop",
        "fix": "Use asyncio.run() or await in async context",
        "example": "asyncio.run(async_function())",
    },
    # File errors
    "FileNotFoundError": {
        "error": "File not found",
        "fix": "Check the file path - use absolute paths for reliability",
        "example": "Path(__file__).parent / 'data.csv'",
    },
    # Port errors
    "Address already in use.*8000": {
        "error": "Port 8000 already in use",
        "fix": "Stop the existing server: make stop",
        "example": "Or use a different port: --port 8001",
    },
}


def find_error_fix(error_str: str) -> dict[str, str] | None:
    """Find a fix suggestion for the given error string."""
    import re

    error_str_lower = str(error_str).lower()

    for pattern, fix_info in ERROR_FIXES.items():
        if re.search(pattern.lower(), error_str_lower):
            return fix_info

    return None


def agent_friendly_errors(
    func: Callable[..., T] | None = None,
    *,
    provide_fix: bool = True,
    log_errors: bool = True,
    reraise: bool = True,
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that catches errors and provides helpful fix suggestions.

    Args:
        provide_fix: Whether to include fix suggestions
        log_errors: Whether to log errors
        reraise: Whether to re-raise the error after logging

    Usage:
        @agent_friendly_errors
        def my_function():
            ...

        @agent_friendly_errors(reraise=False)
        def my_function():
            ...
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__

                # Build error info
                error_info = {
                    "function": f.__name__,
                    "error_type": error_type,
                    "error_message": error_msg,
                }

                # Find fix suggestion
                if provide_fix:
                    fix_info = find_error_fix(error_msg)
                    if fix_info:
                        error_info["fix_suggestion"] = fix_info

                # Log the error
                if log_errors:
                    logger.error(
                        f"Error in {f.__name__}: {error_type}: {error_msg}",
                        extra=error_info,
                        exc_info=True,
                    )

                    if fix_info:
                        logger.info(
                            f"💡 Fix suggestion: {fix_info['fix']}",
                            extra={"example": fix_info.get("example", "")},
                        )

                # Create enhanced error message
                if fix_info and provide_fix:
                    enhanced_msg = (
                        f"{error_msg}\n\n"
                        f"💡 Fix: {fix_info['fix']}\n"
                        f"Example: {fix_info.get('example', '')}"
                    )
                    # Replace the error message
                    e.args = (enhanced_msg,) + e.args[1:]

                if reraise:
                    raise

                # Return error info if not re-raising
                return error_info  # type: ignore[return-value]

        # Add async support
        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return await f(*args, **kwargs)
                except Exception as e:
                    # Same error handling logic
                    error_msg = str(e)
                    error_type = type(e).__name__

                    error_info = {
                        "function": f.__name__,
                        "error_type": error_type,
                        "error_message": error_msg,
                    }

                    if provide_fix:
                        fix_info = find_error_fix(error_msg)
                        if fix_info:
                            error_info["fix_suggestion"] = fix_info

                    if log_errors:
                        logger.error(
                            f"Error in {f.__name__}: {error_type}: {error_msg}",
                            extra=error_info,
                            exc_info=True,
                        )

                        if fix_info:
                            logger.info(
                                f"💡 Fix suggestion: {fix_info['fix']}",
                                extra={"example": fix_info.get("example", "")},
                            )

                    if fix_info and provide_fix:
                        enhanced_msg = (
                            f"{error_msg}\n\n"
                            f"💡 Fix: {fix_info['fix']}\n"
                            f"Example: {fix_info.get('example', '')}"
                        )
                        e.args = (enhanced_msg,) + e.args[1:]

                    if reraise:
                        raise

                    return error_info  # type: ignore[return-value]

            return async_wrapper  # type: ignore[return-value]

        return wrapper

    # Handle being called with or without parentheses
    if func is None:
        return decorator
    else:
        return decorator(func)


# Context manager for agent-friendly error handling
class AgentErrorContext:
    """Context manager that provides helpful error messages."""

    def __init__(self, operation: str = "operation"):
        self.operation = operation

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = str(exc_val)
            fix_info = find_error_fix(error_msg)

            if fix_info:
                logger.error(
                    f"Error during {self.operation}: {exc_type.__name__}: {error_msg}"
                )
                logger.info(
                    f"💡 Fix: {fix_info['fix']}",
                    extra={"example": fix_info.get("example", "")},
                )
                # Don't suppress the exception
                return False

        return False


# Utility function to get common error context
def get_error_context(error: Exception) -> dict[str, Any]:
    """Extract useful context from an error."""
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc().split("\n"),
    }

    # Add specific context based on error type
    if isinstance(error, KeyError):
        context["key"] = error.args[0] if error.args else "unknown"
    elif isinstance(error, ValueError):
        context["value_error_details"] = error.args
    elif isinstance(error, ConnectionError):
        context["connection_type"] = "network"
    elif hasattr(error, "response"):  # HTTP errors
        context["status_code"] = getattr(error.response, "status_code", None)
        context["response_text"] = getattr(error.response, "text", None)

    return context
