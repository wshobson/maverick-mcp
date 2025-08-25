"""
Structured logging with request context for MaverickMCP.

This module provides structured logging capabilities that:
- Capture request context (request ID, user, tool name)
- Track performance metrics (duration, memory usage)
- Support JSON output for log aggregation
- Integrate with FastMCP's context system
"""

import functools
import json
import logging
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import psutil
from fastmcp import Context as MCPContext

# Context variables for request tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)  # type: ignore[assignment]
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)  # type: ignore[assignment]
tool_name_var: ContextVar[str | None] = ContextVar("tool_name", default=None)  # type: ignore[assignment]
request_start_var: ContextVar[float | None] = ContextVar("request_start", default=None)  # type: ignore[assignment]


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id

        tool_name = tool_name_var.get()
        if tool_name:
            log_data["tool_name"] = tool_name

        # Add request duration if available
        request_start = request_start_var.get()
        if request_start:
            log_data["duration_ms"] = int((time.time() - request_start) * 1000)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__
                if record.exc_info[0]
                else "Unknown",
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class RequestContextLogger:
    """Logger that automatically includes request context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with additional context fields."""
        extra = kwargs.get("extra", {})

        # Add performance metrics
        process = psutil.Process()
        extra["memory_mb"] = process.memory_info().rss / 1024 / 1024
        extra["cpu_percent"] = process.cpu_percent(interval=0.1)

        kwargs["extra"] = extra
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


def setup_structured_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
    use_stderr: bool = False,
) -> None:
    """
    Set up structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "text")
        log_file: Optional log file path
        use_stderr: If True, send console logs to stderr instead of stdout
    """
    # Configure warnings filter to suppress known deprecation warnings
    import warnings

    # Suppress pandas_ta pkg_resources deprecation warning
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module="pandas_ta.*",
    )

    # Suppress passlib crypt deprecation warning
    warnings.filterwarnings(
        "ignore",
        message="'crypt' is deprecated and slated for removal.*",
        category=DeprecationWarning,
        module="passlib.*",
    )

    # Suppress LangChain Pydantic v1 deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message=".*pydantic.* is deprecated.*",
        category=DeprecationWarning,
        module="langchain.*",
    )

    # Suppress Starlette cookie deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message=".*cookie.*deprecated.*",
        category=DeprecationWarning,
        module="starlette.*",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler - use stderr for stdio transport to avoid interfering with JSON-RPC
    console_handler = logging.StreamHandler(sys.stderr if use_stderr else sys.stdout)

    if log_format == "json":
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> RequestContextLogger:
    """Get a logger with request context support."""
    return RequestContextLogger(logging.getLogger(name))


def log_tool_execution(func: Callable) -> Callable:
    """
    Decorator to log tool execution with context.

    Automatically captures:
    - Tool name
    - Request ID
    - Execution time
    - Success/failure status
    - Input parameters (sanitized)
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

        # Set tool name
        tool_name = getattr(func, "__name__", "unknown_function")
        tool_name_var.set(tool_name)

        # Set start time
        start_time = time.time()
        request_start_var.set(start_time)

        # Get logger
        logger = get_logger(f"maverick_mcp.tools.{tool_name}")

        # Check if context is available (but not used in this decorator)
        for arg in args:
            if isinstance(arg, MCPContext):
                break

        # Sanitize parameters for logging (hide sensitive data)
        safe_kwargs = _sanitize_params(kwargs)

        logger.info(
            "Tool execution started",
            extra={
                "tool_name": tool_name,
                "request_id": request_id,
                "parameters": safe_kwargs,
            },
        )

        try:
            # Execute the tool
            result = await func(*args, **kwargs)

            # Log success
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Tool execution completed successfully",
                extra={
                    "tool_name": tool_name,
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "status": "success",
                },
            )

            return result

        except Exception as e:
            # Log error
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Tool execution failed: {str(e)}",
                exc_info=True,
                extra={
                    "tool_name": tool_name,
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            raise

        finally:
            # Clear context vars
            request_id_var.set(None)
            tool_name_var.set(None)
            request_start_var.set(None)

    return wrapper


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize parameters for logging by hiding sensitive data.

    Args:
        params: Original parameters

    Returns:
        Sanitized parameters safe for logging
    """
    sensitive_keys = {"password", "api_key", "secret", "token", "auth"}
    sanitized = {}

    for key, value in params.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_params(value)
        elif isinstance(value, list) and len(value) > 10:
            # Truncate long lists
            sanitized[key] = f"[{len(value)} items]"
        elif isinstance(value, str) and len(value) > 1000:
            # Truncate long strings
            sanitized[key] = value[:100] + f"... ({len(value)} chars total)"
        else:
            sanitized[key] = value

    return sanitized


def log_database_query(
    query: str, params: dict | None = None, duration_ms: int | None = None
):
    """Log database query execution."""
    logger = get_logger("maverick_mcp.database")

    extra = {"query_type": _get_query_type(query), "query_length": len(query)}

    if duration_ms is not None:
        extra["duration_ms"] = duration_ms
        extra["slow_query"] = duration_ms > 1000  # Mark queries over 1 second as slow

    if params:
        extra["param_count"] = len(params)

    logger.info("Database query executed", extra=extra)

    # Log the actual query at debug level
    logger.debug(
        f"Query details: {query[:200]}..."
        if len(query) > 200
        else f"Query details: {query}",
        extra={"params": _sanitize_params(params) if params else None},
    )


def _get_query_type(query: str) -> str:
    """Extract query type from SQL query."""
    query_upper = query.strip().upper()
    if query_upper.startswith("SELECT"):
        return "SELECT"
    elif query_upper.startswith("INSERT"):
        return "INSERT"
    elif query_upper.startswith("UPDATE"):
        return "UPDATE"
    elif query_upper.startswith("DELETE"):
        return "DELETE"
    elif query_upper.startswith("CREATE"):
        return "CREATE"
    elif query_upper.startswith("DROP"):
        return "DROP"
    else:
        return "OTHER"


def log_cache_operation(
    operation: str, key: str, hit: bool = False, duration_ms: int | None = None
):
    """Log cache operation."""
    logger = get_logger("maverick_mcp.cache")

    extra = {"operation": operation, "cache_key": key, "cache_hit": hit}

    if duration_ms is not None:
        extra["duration_ms"] = duration_ms

    logger.info(f"Cache {operation}: {'hit' if hit else 'miss'} for {key}", extra=extra)


def log_external_api_call(
    service: str,
    endpoint: str,
    method: str = "GET",
    status_code: int | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
):
    """Log external API call."""
    logger = get_logger("maverick_mcp.external_api")

    extra: dict[str, Any] = {"service": service, "endpoint": endpoint, "method": method}

    if status_code is not None:
        extra["status_code"] = status_code
        extra["success"] = 200 <= status_code < 300

    if duration_ms is not None:
        extra["duration_ms"] = duration_ms

    if error:
        extra["error"] = error
        logger.error(
            f"External API call failed: {service} {method} {endpoint}", extra=extra
        )
    else:
        logger.info(f"External API call: {service} {method} {endpoint}", extra=extra)


# Performance monitoring context manager
class PerformanceMonitor:
    """Context manager for monitoring performance of code blocks."""

    def __init__(self, operation_name: str, logger: RequestContextLogger | None = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger("maverick_mcp.performance")
        self.start_time: float | None = None
        self.start_memory: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.time() - (self.start_time or 0)) * 1000)
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = end_memory - (self.start_memory or 0)

        extra = {
            "operation": self.operation_name,
            "duration_ms": duration_ms,
            "memory_delta_mb": round(memory_delta, 2),
            "success": exc_type is None,
        }

        if exc_type:
            extra["error_type"] = exc_type.__name__
            self.logger.error(
                f"Operation '{self.operation_name}' failed after {duration_ms}ms",
                extra=extra,
            )
        else:
            self.logger.info(
                f"Operation '{self.operation_name}' completed in {duration_ms}ms",
                extra=extra,
            )
