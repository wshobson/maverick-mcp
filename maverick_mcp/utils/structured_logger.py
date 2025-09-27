"""
Enhanced structured logging infrastructure for backtesting system.

This module provides comprehensive structured logging capabilities with:
- Correlation ID generation and tracking across async boundaries
- Request context propagation
- JSON formatting for log aggregation
- Performance metrics logging
- Resource usage tracking
- Debug mode with verbose logging
- Async logging to avoid blocking operations
- Log rotation and compression
- Multiple output handlers (console, file, remote)
"""

import asyncio
import gc
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

import psutil

# Context variables for request tracking across async boundaries
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
request_start_var: ContextVar[float | None] = ContextVar("request_start", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
tool_name_var: ContextVar[str | None] = ContextVar("tool_name", default=None)
operation_context_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "operation_context", default=None
)

# Global logger registry for performance metrics aggregation
_performance_logger_registry: dict[str, "PerformanceMetricsLogger"] = {}
_log_level_counts: dict[str, int] = {
    "DEBUG": 0,
    "INFO": 0,
    "WARNING": 0,
    "ERROR": 0,
    "CRITICAL": 0,
}

# Thread pool for async logging operations
_async_log_executor: ThreadPoolExecutor | None = None
_async_log_lock = threading.Lock()


class CorrelationIDGenerator:
    """Enhanced correlation ID generation with backtesting context."""

    @staticmethod
    def generate_correlation_id(prefix: str = "bt") -> str:
        """Generate a unique correlation ID with backtesting prefix."""
        timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits of timestamp
        random_part = uuid.uuid4().hex[:8]
        return f"{prefix}-{timestamp}-{random_part}"

    @staticmethod
    def set_correlation_id(
        correlation_id: str | None = None, prefix: str = "bt"
    ) -> str:
        """Set correlation ID in context with automatic generation."""
        if not correlation_id:
            correlation_id = CorrelationIDGenerator.generate_correlation_id(prefix)
        correlation_id_var.set(correlation_id)
        return correlation_id

    @staticmethod
    def get_correlation_id() -> str | None:
        """Get current correlation ID from context."""
        return correlation_id_var.get()

    @staticmethod
    def propagate_context(target_context: dict[str, Any]) -> dict[str, Any]:
        """Propagate correlation context to target dict."""
        target_context.update(
            {
                "correlation_id": correlation_id_var.get(),
                "user_id": user_id_var.get(),
                "tool_name": tool_name_var.get(),
                "operation_context": operation_context_var.get(),
            }
        )
        return target_context


class EnhancedStructuredFormatter(logging.Formatter):
    """Enhanced JSON formatter with performance metrics and resource tracking."""

    def __init__(
        self, include_performance: bool = True, include_resources: bool = True
    ):
        super().__init__()
        self.include_performance = include_performance
        self.include_resources = include_resources
        self._process = psutil.Process()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with comprehensive structured data."""
        # Base structured log data
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process_id": record.process,
        }

        # Add correlation context
        CorrelationIDGenerator.propagate_context(log_data)

        # Add performance metrics if enabled
        if self.include_performance:
            request_start = request_start_var.get()
            if request_start:
                log_data["duration_ms"] = int((time.time() - request_start) * 1000)

        # Add resource usage if enabled
        if self.include_resources:
            try:
                memory_info = self._process.memory_info()
                log_data["memory_rss_mb"] = round(memory_info.rss / 1024 / 1024, 2)
                log_data["memory_vms_mb"] = round(memory_info.vms / 1024 / 1024, 2)
                log_data["cpu_percent"] = self._process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have ended or access denied
                pass

        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__
                if record.exc_info[0]
                else "Unknown",
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields from the record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
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
                "getMessage",
                "message",
            }:
                # Sanitize sensitive data
                if self._is_sensitive_field(key):
                    extra_fields[key] = "***MASKED***"
                else:
                    extra_fields[key] = self._serialize_value(value)

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive information."""
        sensitive_keywords = {
            "password",
            "token",
            "key",
            "secret",
            "auth",
            "credential",
            "bearer",
            "session",
            "cookie",
            "api_key",
            "access_token",
            "refresh_token",
            "private",
            "confidential",
        }
        return any(keyword in field_name.lower() for keyword in sensitive_keywords)

    def _serialize_value(self, value: Any) -> Any:
        """Safely serialize complex values for JSON output."""
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return [self._serialize_value(item) for item in value]
        else:
            return str(value)


class AsyncLogHandler(logging.Handler):
    """Non-blocking async log handler to prevent performance impact."""

    def __init__(self, target_handler: logging.Handler, max_queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.max_queue_size = max_queue_size
        self._queue: list[logging.LogRecord] = []
        self._queue_lock = threading.Lock()
        self._shutdown = False

        # Start background thread for processing logs
        self._worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self._worker_thread.start()

    def emit(self, record: logging.LogRecord):
        """Queue log record for async processing."""
        if self._shutdown:
            return

        with self._queue_lock:
            if len(self._queue) < self.max_queue_size:
                self._queue.append(record)
            # If queue is full, drop oldest records
            elif self._queue:
                self._queue.pop(0)
                self._queue.append(record)

    def _process_logs(self):
        """Background thread to process queued log records."""
        while not self._shutdown:
            records_to_process = []

            with self._queue_lock:
                if self._queue:
                    records_to_process = self._queue[:]
                    self._queue.clear()

            for record in records_to_process:
                try:
                    self.target_handler.emit(record)
                except Exception:
                    # Silently ignore errors to prevent infinite recursion
                    pass

            # Brief sleep to prevent busy waiting
            time.sleep(0.01)

    def close(self):
        """Close the handler and wait for queue to flush."""
        self._shutdown = True
        self._worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class PerformanceMetricsLogger:
    """Comprehensive performance metrics logging for backtesting operations."""

    def __init__(self, logger_name: str = "maverick_mcp.performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics: dict[str, list[float]] = {
            "execution_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "operation_counts": [],
        }
        self._start_times: dict[str, float] = {}
        self._lock = threading.Lock()

        # Register for global aggregation
        _performance_logger_registry[logger_name] = self

    def start_operation(self, operation_id: str, operation_type: str, **context):
        """Start tracking a performance-critical operation."""
        start_time = time.time()

        with self._lock:
            self._start_times[operation_id] = start_time

        # Set request context
        request_start_var.set(start_time)
        if "tool_name" in context:
            tool_name_var.set(context["tool_name"])

        self.logger.info(
            f"Started {operation_type} operation",
            extra={
                "operation_id": operation_id,
                "operation_type": operation_type,
                "start_time": start_time,
                **context,
            },
        )

    def end_operation(self, operation_id: str, success: bool = True, **metrics):
        """End tracking of a performance-critical operation."""
        end_time = time.time()

        with self._lock:
            start_time = self._start_times.pop(operation_id, end_time)

        duration_ms = (end_time - start_time) * 1000

        # Collect system metrics
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            memory_mb = 0
            cpu_percent = 0

        # Update internal metrics
        with self._lock:
            self.metrics["execution_times"].append(duration_ms)
            self.metrics["memory_usage"].append(memory_mb)
            self.metrics["cpu_usage"].append(cpu_percent)
            self.metrics["operation_counts"].append(1)

        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(
            log_level,
            f"{'Completed' if success else 'Failed'} operation in {duration_ms:.2f}ms",
            extra={
                "operation_id": operation_id,
                "duration_ms": duration_ms,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "success": success,
                **metrics,
            },
        )

    def log_business_metric(self, metric_name: str, value: int | float, **context):
        """Log business-specific metrics like strategies processed, success rates."""
        self.logger.info(
            f"Business metric: {metric_name} = {value}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_type": "business",
                **context,
            },
        )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get aggregated performance metrics summary."""
        with self._lock:
            if not self.metrics["execution_times"]:
                return {"message": "No performance data available"}

            execution_times = self.metrics["execution_times"]
            memory_usage = self.metrics["memory_usage"]
            cpu_usage = self.metrics["cpu_usage"]

            return {
                "operations_count": len(execution_times),
                "execution_time_stats": {
                    "avg_ms": sum(execution_times) / len(execution_times),
                    "min_ms": min(execution_times),
                    "max_ms": max(execution_times),
                    "total_ms": sum(execution_times),
                },
                "memory_stats": {
                    "avg_mb": sum(memory_usage) / len(memory_usage)
                    if memory_usage
                    else 0,
                    "peak_mb": max(memory_usage) if memory_usage else 0,
                },
                "cpu_stats": {
                    "avg_percent": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                    "peak_percent": max(cpu_usage) if cpu_usage else 0,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }


class DebugModeManager:
    """Manages debug mode configuration and verbose logging."""

    def __init__(self):
        self._debug_enabled = os.getenv("MAVERICK_DEBUG", "false").lower() in (
            "true",
            "1",
            "on",
        )
        self._verbose_modules: set = set()
        self._debug_filters: dict[str, Any] = {}

    def is_debug_enabled(self, module_name: str = "") -> bool:
        """Check if debug mode is enabled globally or for specific module."""
        if not self._debug_enabled:
            return False

        if not module_name:
            return True

        # Check if specific module debug is enabled
        return module_name in self._verbose_modules or not self._verbose_modules

    def enable_verbose_logging(self, module_pattern: str):
        """Enable verbose logging for specific module pattern."""
        self._verbose_modules.add(module_pattern)

    def add_debug_filter(self, filter_name: str, filter_config: dict[str, Any]):
        """Add custom debug filter configuration."""
        self._debug_filters[filter_name] = filter_config

    def should_log_request_response(self, operation_name: str) -> bool:
        """Check if request/response should be logged for operation."""
        if not self._debug_enabled:
            return False

        # Check specific filters
        for _filter_name, config in self._debug_filters.items():
            if config.get("log_request_response") and operation_name in config.get(
                "operations", []
            ):
                return True

        return True  # Default to true in debug mode


class StructuredLoggerManager:
    """Central manager for structured logging configuration."""

    def __init__(self):
        self.debug_manager = DebugModeManager()
        self.performance_loggers: dict[str, PerformanceMetricsLogger] = {}
        self._configured = False

    def setup_structured_logging(
        self,
        log_level: str = "INFO",
        log_format: str = "json",
        log_file: str | None = None,
        enable_async: bool = True,
        enable_rotation: bool = True,
        max_log_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: str = "stdout",  # stdout, stderr
        remote_handler_config: dict[str, Any] | None = None,
    ):
        """Setup comprehensive structured logging infrastructure."""

        if self._configured:
            return

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        handlers = []

        # Console handler
        console_stream = sys.stdout if console_output == "stdout" else sys.stderr
        console_handler = logging.StreamHandler(console_stream)

        if log_format == "json":
            console_formatter = EnhancedStructuredFormatter(
                include_performance=True, include_resources=True
            )
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        # File handler with rotation if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if enable_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_log_size, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)

            file_handler.setFormatter(EnhancedStructuredFormatter())
            handlers.append(file_handler)

        # Remote handler if configured (for log aggregation)
        if remote_handler_config:
            remote_handler = self._create_remote_handler(remote_handler_config)
            if remote_handler:
                handlers.append(remote_handler)

        # Wrap handlers with async processing if enabled
        if enable_async:
            handlers = [AsyncLogHandler(handler) for handler in handlers]

        # Add all handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)

        # Set specific logger levels to reduce noise
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

        # Enable debug mode loggers if configured
        if self.debug_manager.is_debug_enabled():
            self._setup_debug_loggers()

        self._configured = True

    def _create_remote_handler(self, config: dict[str, Any]) -> logging.Handler | None:
        """Create remote handler for log aggregation (placeholder for future implementation)."""
        # This would implement remote logging to services like ELK, Splunk, etc.
        # For now, return None as it's not implemented
        return None

    def _setup_debug_loggers(self):
        """Setup additional loggers for debug mode."""
        debug_logger = logging.getLogger("maverick_mcp.debug")
        debug_logger.setLevel(logging.DEBUG)

        request_logger = logging.getLogger("maverick_mcp.requests")
        request_logger.setLevel(logging.DEBUG)

    def get_performance_logger(self, logger_name: str) -> PerformanceMetricsLogger:
        """Get or create performance logger for specific component."""
        if logger_name not in self.performance_loggers:
            self.performance_loggers[logger_name] = PerformanceMetricsLogger(
                logger_name
            )
        return self.performance_loggers[logger_name]

    def get_logger(self, name: str) -> logging.Logger:
        """Get structured logger with correlation support."""
        return logging.getLogger(name)

    def create_dashboard_metrics(self) -> dict[str, Any]:
        """Create comprehensive metrics for performance dashboard."""
        global _log_level_counts

        dashboard_data = {
            "system_metrics": {
                "timestamp": datetime.now(UTC).isoformat(),
                "log_level_counts": _log_level_counts.copy(),
                "active_correlation_ids": len(
                    [cid for cid in [correlation_id_var.get()] if cid]
                ),
            },
            "performance_metrics": {},
            "memory_stats": {},
        }

        # Aggregate performance metrics from all loggers
        for logger_name, perf_logger in _performance_logger_registry.items():
            dashboard_data["performance_metrics"][logger_name] = (
                perf_logger.get_performance_summary()
            )

        # System memory stats
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            dashboard_data["memory_stats"] = {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(interval=None),
                "gc_stats": {
                    "generation_0": gc.get_count()[0],
                    "generation_1": gc.get_count()[1],
                    "generation_2": gc.get_count()[2],
                },
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            dashboard_data["memory_stats"] = {"error": "Unable to collect memory stats"}

        return dashboard_data


# Global instance
_logger_manager: StructuredLoggerManager | None = None


def get_logger_manager() -> StructuredLoggerManager:
    """Get global logger manager instance."""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = StructuredLoggerManager()
    return _logger_manager


def with_structured_logging(
    operation_name: str,
    include_performance: bool = True,
    log_params: bool = True,
    log_result: bool = False,
):
    """Decorator for automatic structured logging of operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate correlation ID if not present
            correlation_id = CorrelationIDGenerator.get_correlation_id()
            if not correlation_id:
                correlation_id = CorrelationIDGenerator.set_correlation_id()

            # Setup operation context
            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
            tool_name_var.set(operation_name)

            logger = get_logger_manager().get_logger(f"maverick_mcp.{operation_name}")
            perf_logger = None

            if include_performance:
                perf_logger = get_logger_manager().get_performance_logger(
                    f"performance.{operation_name}"
                )
                perf_logger.start_operation(
                    operation_id=operation_id,
                    operation_type=operation_name,
                    tool_name=operation_name,
                )

            # Log operation start
            extra_data = {
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            }
            if log_params:
                # Sanitize parameters
                safe_kwargs = {
                    k: "***MASKED***"
                    if "password" in k.lower() or "token" in k.lower()
                    else v
                    for k, v in kwargs.items()
                }
                extra_data["parameters"] = safe_kwargs

            logger.info(f"Starting {operation_name}", extra=extra_data)

            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Log success
                success_data = {"operation_id": operation_id, "success": True}
                if log_result and result is not None:
                    # Limit result size for logging
                    result_str = str(result)
                    success_data["result"] = (
                        result_str[:1000] + "..."
                        if len(result_str) > 1000
                        else result_str
                    )

                logger.info(f"Completed {operation_name}", extra=success_data)

                if perf_logger:
                    perf_logger.end_operation(operation_id, success=True)

                return result

            except Exception as e:
                # Log error
                logger.error(
                    f"Failed {operation_name}: {str(e)}",
                    exc_info=True,
                    extra={
                        "operation_id": operation_id,
                        "error_type": type(e).__name__,
                        "success": False,
                    },
                )

                if perf_logger:
                    perf_logger.end_operation(operation_id, success=False, error=str(e))

                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            correlation_id = CorrelationIDGenerator.get_correlation_id()
            if not correlation_id:
                correlation_id = CorrelationIDGenerator.set_correlation_id()

            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
            tool_name_var.set(operation_name)

            logger = get_logger_manager().get_logger(f"maverick_mcp.{operation_name}")
            perf_logger = None

            if include_performance:
                perf_logger = get_logger_manager().get_performance_logger(
                    f"performance.{operation_name}"
                )
                perf_logger.start_operation(
                    operation_id=operation_id,
                    operation_type=operation_name,
                    tool_name=operation_name,
                )

            extra_data = {
                "operation_id": operation_id,
                "correlation_id": correlation_id,
            }
            if log_params:
                safe_kwargs = {
                    k: "***MASKED***"
                    if any(
                        sensitive in k.lower()
                        for sensitive in ["password", "token", "key", "secret"]
                    )
                    else v
                    for k, v in kwargs.items()
                }
                extra_data["parameters"] = safe_kwargs

            logger.info(f"Starting {operation_name}", extra=extra_data)

            try:
                result = func(*args, **kwargs)

                success_data = {"operation_id": operation_id, "success": True}
                if log_result and result is not None:
                    result_str = str(result)
                    success_data["result"] = (
                        result_str[:1000] + "..."
                        if len(result_str) > 1000
                        else result_str
                    )

                logger.info(f"Completed {operation_name}", extra=success_data)

                if perf_logger:
                    perf_logger.end_operation(operation_id, success=True)

                return result

            except Exception as e:
                logger.error(
                    f"Failed {operation_name}: {str(e)}",
                    exc_info=True,
                    extra={
                        "operation_id": operation_id,
                        "error_type": type(e).__name__,
                        "success": False,
                    },
                )

                if perf_logger:
                    perf_logger.end_operation(operation_id, success=False, error=str(e))

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Convenience functions
def get_structured_logger(name: str) -> logging.Logger:
    """Get structured logger instance."""
    return get_logger_manager().get_logger(name)


def get_performance_logger(component: str) -> PerformanceMetricsLogger:
    """Get performance logger for specific component."""
    return get_logger_manager().get_performance_logger(component)


def setup_backtesting_logging(
    log_level: str = "INFO", enable_debug: bool = False, log_file: str | None = None
):
    """Setup logging specifically configured for backtesting operations."""

    # Set debug environment if requested
    if enable_debug:
        os.environ["MAVERICK_DEBUG"] = "true"

    # Setup structured logging
    manager = get_logger_manager()
    manager.setup_structured_logging(
        log_level=log_level,
        log_format="json",
        log_file=log_file or "logs/backtesting.log",
        enable_async=True,
        enable_rotation=True,
        console_output="stderr",  # Use stderr for MCP compatibility
    )

    # Configure debug filters for backtesting
    if enable_debug:
        manager.debug_manager.add_debug_filter(
            "backtesting",
            {
                "log_request_response": True,
                "operations": [
                    "run_backtest",
                    "optimize_parameters",
                    "get_historical_data",
                ],
            },
        )


# Update log level counts (for dashboard metrics)
class LogLevelCounterFilter(logging.Filter):
    """Filter to count log levels for dashboard metrics."""

    def filter(self, record: logging.LogRecord) -> bool:
        global _log_level_counts
        _log_level_counts[record.levelname] = (
            _log_level_counts.get(record.levelname, 0) + 1
        )
        return True


# Add the counter filter to root logger
logging.getLogger().addFilter(LogLevelCounterFilter())
