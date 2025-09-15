"""
Comprehensive MCP Logging Middleware for debugging tool calls and protocol communication.

This middleware provides:
- Tool call lifecycle logging
- MCP protocol message logging
- Request/response payload logging
- Error tracking with full context
- Performance metrics collection
- Timeout detection and logging
"""

import asyncio
import functools
import json
import logging
import time
import traceback
import uuid
from typing import Any

from fastmcp import FastMCP

try:
    from fastmcp.server.middleware import Middleware, MiddlewareContext

    MIDDLEWARE_AVAILABLE = True
except ImportError:
    # Fallback for older FastMCP versions
    MIDDLEWARE_AVAILABLE = False

    class Middleware:  # type: ignore
        """Fallback Middleware class for older FastMCP versions."""

        pass

    MiddlewareContext = Any

from maverick_mcp.utils.logging import (
    get_logger,
    request_id_var,
    request_start_var,
    tool_name_var,
)

logger = get_logger("maverick_mcp.middleware.mcp_logging")


class MCPLoggingMiddleware(Middleware if MIDDLEWARE_AVAILABLE else object):
    """
    Comprehensive MCP protocol and tool call logging middleware for FastMCP 2.0+.

    Logs:
    - Tool call lifecycle with execution details
    - Resource access and prompt retrievals
    - Error conditions with full context
    - Performance metrics (execution time, memory usage)
    - Timeout detection and warnings
    """

    def __init__(
        self,
        include_payloads: bool = True,
        max_payload_length: int = 2000,
        log_level: int = logging.INFO,
    ):
        if MIDDLEWARE_AVAILABLE:
            super().__init__()
        self.include_payloads = include_payloads
        self.max_payload_length = max_payload_length
        self.log_level = log_level
        self.logger = get_logger("maverick_mcp.mcp_protocol")

    async def on_call_tool(self, context: MiddlewareContext, call_next) -> Any:
        """Log tool call lifecycle with comprehensive details."""
        if not MIDDLEWARE_AVAILABLE:
            return await call_next(context)

        request_id = str(uuid.uuid4())
        request_start_var.set(time.time())
        request_id_var.set(request_id)

        start_time = time.time()
        tool_name = getattr(context.message, "name", "unknown_tool")
        tool_name_var.set(tool_name)

        # Extract arguments if available
        arguments = getattr(context.message, "arguments", {})

        # Log tool call start
        self._log_tool_call_start(request_id, tool_name, arguments)

        try:
            # Execute with timeout detection
            result = await asyncio.wait_for(call_next(context), timeout=25.0)

            # Log successful completion
            execution_time = time.time() - start_time
            self._log_tool_call_success(request_id, tool_name, result, execution_time)

            return result

        except TimeoutError:
            execution_time = time.time() - start_time
            self._log_tool_call_timeout(request_id, tool_name, execution_time)
            raise

        except Exception as e:
            # Log error with full context
            execution_time = time.time() - start_time
            self._log_tool_call_error(
                request_id, tool_name, e, execution_time, arguments
            )
            raise

    async def on_read_resource(self, context: MiddlewareContext, call_next) -> Any:
        """Log resource access."""
        if not MIDDLEWARE_AVAILABLE:
            return await call_next(context)

        resource_uri = getattr(context.message, "uri", "unknown_resource")
        start_time = time.time()

        print(f"🔗 RESOURCE ACCESS: {resource_uri}")

        try:
            result = await call_next(context)
            execution_time = time.time() - start_time
            print(f"✅ RESOURCE SUCCESS: {resource_uri} ({execution_time:.2f}s)")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(
                f"❌ RESOURCE ERROR: {resource_uri} ({execution_time:.2f}s) - {type(e).__name__}: {str(e)}"
            )
            raise

    def _log_tool_call_start(self, request_id: str, tool_name: str, arguments: dict):
        """Log tool call initiation."""
        log_data = {
            "request_id": request_id,
            "direction": "incoming",
            "tool_name": tool_name,
            "timestamp": time.time(),
        }

        # Add arguments if requested (debug mode)
        if self.include_payloads and arguments:
            try:
                args_str = json.dumps(arguments)[: self.max_payload_length]
                log_data["arguments"] = args_str
            except Exception as e:
                log_data["args_error"] = str(e)

        self.logger.info("TOOL_CALL_START", extra=log_data)

        # Console output for immediate visibility
        args_preview = ""
        if arguments:
            args_str = str(arguments)
            args_preview = f" with {args_str[:50]}{'...' if len(args_str) > 50 else ''}"
        print(f"🔧 TOOL CALL: {tool_name}{args_preview} [{request_id[:8]}]")

    def _log_tool_call_success(
        self, request_id: str, tool_name: str, result: Any, execution_time: float
    ):
        """Log successful tool completion."""
        log_data = {
            "request_id": request_id,
            "direction": "outgoing",
            "tool_name": tool_name,
            "execution_time": execution_time,
            "status": "success",
            "timestamp": time.time(),
        }

        # Add result preview if requested (debug mode)
        if self.include_payloads and result is not None:
            try:
                result_str = (
                    json.dumps(result)[: self.max_payload_length]
                    if not isinstance(result, str)
                    else result[: self.max_payload_length]
                )
                log_data["result_preview"] = result_str
                log_data["result_type"] = type(result).__name__
            except Exception as e:
                log_data["result_error"] = str(e)

        self.logger.info("TOOL_CALL_SUCCESS", extra=log_data)

        # Console output with color coding based on execution time
        status_icon = (
            "🟢" if execution_time < 5.0 else "🟡" if execution_time < 15.0 else "🟠"
        )
        print(
            f"{status_icon} TOOL SUCCESS: {tool_name} [{request_id[:8]}] {execution_time:.2f}s"
        )

    def _log_tool_call_timeout(
        self, request_id: str, tool_name: str, execution_time: float
    ):
        """Log tool timeout."""
        log_data = {
            "request_id": request_id,
            "direction": "outgoing",
            "tool_name": tool_name,
            "execution_time": execution_time,
            "status": "timeout",
            "timeout_seconds": 25.0,
            "error_type": "timeout",
            "timestamp": time.time(),
        }

        self.logger.error("TOOL_CALL_TIMEOUT", extra=log_data)
        print(
            f"⏰ TOOL TIMEOUT: {tool_name} [{request_id[:8]}] {execution_time:.2f}s (exceeded 25s limit)"
        )

    def _log_tool_call_error(
        self,
        request_id: str,
        tool_name: str,
        error: Exception,
        execution_time: float,
        arguments: dict,
    ):
        """Log tool error with full context."""
        log_data = {
            "request_id": request_id,
            "direction": "outgoing",
            "tool_name": tool_name,
            "execution_time": execution_time,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
        }

        # Add arguments for debugging
        if self.include_payloads and arguments:
            try:
                log_data["arguments"] = json.dumps(arguments)[: self.max_payload_length]
            except Exception as e:
                log_data["args_error"] = str(e)

        self.logger.error("TOOL_CALL_ERROR", extra=log_data)

        # Console output with error details
        print(
            f"❌ TOOL ERROR: {tool_name} [{request_id[:8]}] {execution_time:.2f}s - {type(error).__name__}: {str(error)}"
        )


class ToolExecutionLogger:
    """
    Specific logger for individual tool execution steps.

    Use this within tools to log execution progress and debug issues.
    """

    def __init__(self, tool_name: str, request_id: str | None = None):
        self.tool_name = tool_name
        self.request_id = request_id or request_id_var.get() or str(uuid.uuid4())
        self.logger = get_logger(f"maverick_mcp.tools.{tool_name}")
        self.start_time = time.time()
        self.step_times = {}

    def step(self, step_name: str, message: str | None = None):
        """Log a step in tool execution."""
        current_time = time.time()
        step_duration = current_time - self.start_time
        self.step_times[step_name] = step_duration

        log_message = message or f"Executing step: {step_name}"

        self.logger.info(
            log_message,
            extra={
                "request_id": self.request_id,
                "tool_name": self.tool_name,
                "step": step_name,
                "step_duration": step_duration,
                "total_duration": current_time - self.start_time,
            },
        )

        # Console progress indicator
        print(f"  📊 {self.tool_name} -> {step_name} ({step_duration:.2f}s)")

    def error(self, step_name: str, error: Exception, message: str | None = None):
        """Log an error in tool execution."""
        current_time = time.time()
        step_duration = current_time - self.start_time

        log_message = message or f"Error in step: {step_name}"

        self.logger.error(
            log_message,
            extra={
                "request_id": self.request_id,
                "tool_name": self.tool_name,
                "step": step_name,
                "step_duration": step_duration,
                "total_duration": current_time - self.start_time,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )

        # Console error indicator
        print(
            f"  ❌ {self.tool_name} -> {step_name} ERROR: {type(error).__name__}: {str(error)}"
        )

    def complete(self, result_summary: str | None = None):
        """Log completion of tool execution."""
        total_duration = time.time() - self.start_time

        log_message = result_summary or "Tool execution completed"

        self.logger.info(
            log_message,
            extra={
                "request_id": self.request_id,
                "tool_name": self.tool_name,
                "total_duration": total_duration,
                "step_times": self.step_times,
                "status": "completed",
            },
        )

        # Console completion
        print(f"  ✅ {self.tool_name} completed ({total_duration:.2f}s)")


def add_mcp_logging_middleware(
    server: FastMCP,
    include_payloads: bool = True,
    max_payload_length: int = 2000,
    log_level: int = logging.INFO,
):
    """
    Add comprehensive MCP logging middleware to a FastMCP server.

    Args:
        server: FastMCP server instance
        include_payloads: Whether to log request/response payloads (debug mode)
        max_payload_length: Maximum length of logged payloads
        log_level: Minimum logging level
    """
    if not MIDDLEWARE_AVAILABLE:
        logger.warning("FastMCP middleware not available - requires FastMCP 2.9+")
        print("⚠️  FastMCP middleware not available - tool logging will be limited")
        return

    middleware = MCPLoggingMiddleware(
        include_payloads=include_payloads,
        max_payload_length=max_payload_length,
        log_level=log_level,
    )

    # Use the correct FastMCP 2.0 middleware registration method
    try:
        if hasattr(server, "add_middleware"):
            server.add_middleware(middleware)
            logger.info("✅ FastMCP 2.0 middleware registered successfully")
        elif hasattr(server, "middleware"):
            # Fallback for different API structure
            if isinstance(server.middleware, list):
                server.middleware.append(middleware)
            else:
                server.middleware = [middleware]
            logger.info("✅ FastMCP middleware registered via fallback method")
        else:
            # Manual middleware application as decorator
            logger.warning("Using decorator-style middleware registration")
            _apply_middleware_as_decorators(server, middleware)

    except Exception as e:
        logger.error(f"Failed to register FastMCP middleware: {e}")
        print(f"⚠️  Middleware registration failed: {e}")

    logger.info(
        "MCP logging middleware setup completed",
        extra={
            "include_payloads": include_payloads,
            "max_payload_length": max_payload_length,
            "log_level": logging.getLevelName(log_level),
        },
    )


def _apply_middleware_as_decorators(server: FastMCP, middleware: MCPLoggingMiddleware):
    """Apply middleware functionality via decorators if direct middleware isn't available."""
    # This is a fallback approach - wrap tool execution with logging
    original_tool_method = server.tool

    def logging_tool_decorator(*args, **kwargs):
        def decorator(func):
            # Wrap the original tool function with logging
            @functools.wraps(func)
            async def wrapper(*func_args, **func_kwargs):
                # Simple console logging as fallback
                func_name = getattr(func, "__name__", "unknown_tool")
                print(f"🔧 TOOL CALL: {func_name}")
                start_time = time.time()
                try:
                    result = await func(*func_args, **func_kwargs)
                    execution_time = time.time() - start_time
                    print(f"🟢 TOOL SUCCESS: {func_name} ({execution_time:.2f}s)")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    print(
                        f"❌ TOOL ERROR: {func_name} ({execution_time:.2f}s) - {type(e).__name__}: {str(e)}"
                    )
                    raise

            # Register the wrapped function
            return original_tool_method(*args, **kwargs)(wrapper)

        return decorator

    # Replace the server's tool decorator
    server.tool = logging_tool_decorator
    logger.info("Applied middleware as tool decorators (fallback mode)")


# Convenience function for tool developers
def get_tool_logger(tool_name: str) -> ToolExecutionLogger:
    """Get a tool execution logger for the current request."""
    return ToolExecutionLogger(tool_name)
