"""
Structured logging configuration with correlation IDs and error tracking.
"""

import json
import logging
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any

# Context variable for correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with additional metadata."""
        # Get correlation ID from context
        correlation_id = correlation_id_var.get()

        # Build structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
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
                "message",
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
                log_entry[key] = value

        return json.dumps(log_entry)


class CorrelationIDMiddleware:
    """Middleware to inject correlation IDs into requests."""

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a unique correlation ID."""
        return f"mcp-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def set_correlation_id(correlation_id: str | None = None) -> str:
        """Set correlation ID in context."""
        if not correlation_id:
            correlation_id = CorrelationIDMiddleware.generate_correlation_id()
        correlation_id_var.set(correlation_id)
        return correlation_id

    @staticmethod
    def get_correlation_id() -> str | None:
        """Get current correlation ID from context."""
        return correlation_id_var.get()


def with_correlation_id(func):
    """Decorator to ensure correlation ID exists for function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not correlation_id_var.get():
            CorrelationIDMiddleware.set_correlation_id()
        return func(*args, **kwargs)

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        if not correlation_id_var.get():
            CorrelationIDMiddleware.set_correlation_id()
        return await func(*args, **kwargs)

    return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper


class ErrorLogger:
    """Enhanced error logging with context and metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._error_counts: dict[str, int] = {}

    def log_error(
        self,
        error: Exception,
        context: dict[str, Any],
        level: int = logging.ERROR,
        mask_sensitive: bool = True,
    ):
        """Log error with full context and tracking."""
        error_type = type(error).__name__
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

        # Mask sensitive data if requested
        if mask_sensitive:
            context = self._mask_sensitive_data(context)

        # Create structured error log
        self.logger.log(
            level,
            f"{error_type}: {str(error)}",
            extra={
                "error_type": error_type,
                "error_message": str(error),
                "error_count": self._error_counts[error_type],
                "context": context,
                "stack_trace": traceback.format_exc() if sys.exc_info()[0] else None,
            },
        )

    def _mask_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive fields in logging data."""
        sensitive_fields = {
            "password",
            "token",
            "api_key",
            "secret",
            "credit_card",
            "ssn",
            "email",
            "phone",
            "address",
            "bearer",
            "authorization",
            "x-api-key",
        }

        masked_data = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                masked_data[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_data(value)
            else:
                masked_data[key] = value

        return masked_data

    def get_error_stats(self) -> dict[str, int]:
        """Get error count statistics."""
        return self._error_counts.copy()


def setup_logging(
    level: int = logging.INFO, use_json: bool = True, log_file: str | None = None
):
    """Configure application logging with structured output."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if use_json:
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

    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


# Import guard for asyncio
try:
    import asyncio
except ImportError:
    asyncio = None
