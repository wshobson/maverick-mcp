"""
Debug utilities for backtesting system troubleshooting.

This module provides comprehensive debugging tools including:
- Request/response logging
- Performance profiling
- Memory analysis
- Error tracking
- Debug mode utilities
"""

import inspect
import json
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from typing import Any

import psutil

from maverick_mcp.utils.structured_logger import (
    CorrelationIDGenerator,
    get_logger_manager,
    get_performance_logger,
    get_structured_logger,
)


class DebugProfiler:
    """Comprehensive debug profiler for performance analysis."""

    def __init__(self):
        self.logger = get_structured_logger("maverick_mcp.debug")
        self.performance_logger = get_performance_logger("debug_profiler")
        self._profiles: dict[str, dict[str, Any]] = {}

    def start_profile(self, profile_name: str, **context):
        """Start a debug profiling session."""
        profile_id = f"{profile_name}_{int(time.time() * 1000)}"

        profile_data = {
            "profile_name": profile_name,
            "profile_id": profile_id,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": self._get_cpu_usage(),
            "context": context,
            "checkpoints": [],
        }

        self._profiles[profile_id] = profile_data

        self.logger.debug(
            f"Started debug profile: {profile_name}",
            extra={
                "profile_id": profile_id,
                "start_memory_mb": profile_data["start_memory"],
                "start_cpu_percent": profile_data["start_cpu"],
                **context,
            },
        )

        return profile_id

    def checkpoint(self, profile_id: str, checkpoint_name: str, **data):
        """Add a checkpoint to an active profile."""
        if profile_id not in self._profiles:
            self.logger.warning(f"Profile {profile_id} not found for checkpoint")
            return

        profile = self._profiles[profile_id]
        current_time = time.time()
        elapsed_ms = (current_time - profile["start_time"]) * 1000

        checkpoint_data = {
            "name": checkpoint_name,
            "timestamp": current_time,
            "elapsed_ms": elapsed_ms,
            "memory_mb": self._get_memory_usage(),
            "cpu_percent": self._get_cpu_usage(),
            "data": data,
        }

        profile["checkpoints"].append(checkpoint_data)

        self.logger.debug(
            f"Profile checkpoint: {checkpoint_name} at {elapsed_ms:.2f}ms",
            extra={
                "profile_id": profile_id,
                "checkpoint": checkpoint_name,
                "elapsed_ms": elapsed_ms,
                "memory_mb": checkpoint_data["memory_mb"],
                **data,
            },
        )

    def end_profile(
        self, profile_id: str, success: bool = True, **final_data
    ) -> dict[str, Any]:
        """End a debug profiling session and return comprehensive results."""
        if profile_id not in self._profiles:
            self.logger.warning(f"Profile {profile_id} not found for ending")
            return {}

        profile = self._profiles.pop(profile_id)
        end_time = time.time()
        total_duration_ms = (end_time - profile["start_time"]) * 1000

        # Calculate memory and CPU deltas
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()
        memory_delta = end_memory - profile["start_memory"]

        results = {
            "profile_name": profile["profile_name"],
            "profile_id": profile_id,
            "success": success,
            "total_duration_ms": total_duration_ms,
            "start_time": profile["start_time"],
            "end_time": end_time,
            "memory_stats": {
                "start_mb": profile["start_memory"],
                "end_mb": end_memory,
                "delta_mb": memory_delta,
                "peak_usage": max(cp["memory_mb"] for cp in profile["checkpoints"])
                if profile["checkpoints"]
                else end_memory,
            },
            "cpu_stats": {
                "start_percent": profile["start_cpu"],
                "end_percent": end_cpu,
                "avg_percent": sum(cp["cpu_percent"] for cp in profile["checkpoints"])
                / len(profile["checkpoints"])
                if profile["checkpoints"]
                else end_cpu,
            },
            "checkpoints": profile["checkpoints"],
            "checkpoint_count": len(profile["checkpoints"]),
            "context": profile["context"],
            "final_data": final_data,
        }

        # Log profile completion
        log_level = "info" if success else "error"
        getattr(self.logger, log_level)(
            f"Completed debug profile: {profile['profile_name']} in {total_duration_ms:.2f}ms",
            extra={
                "profile_results": results,
                "performance_summary": {
                    "duration_ms": total_duration_ms,
                    "memory_delta_mb": memory_delta,
                    "checkpoint_count": len(profile["checkpoints"]),
                    "success": success,
                },
            },
        )

        return results

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    @staticmethod
    def _get_cpu_usage() -> float:
        """Get current CPU usage percentage."""
        try:
            process = psutil.Process()
            return process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0


class RequestResponseLogger:
    """Detailed request/response logging for debugging."""

    def __init__(self, max_payload_size: int = 5000):
        self.logger = get_structured_logger("maverick_mcp.requests")
        self.max_payload_size = max_payload_size

    def log_request(self, operation: str, **request_data):
        """Log detailed request information."""
        correlation_id = CorrelationIDGenerator.get_correlation_id()

        # Sanitize and truncate request data
        sanitized_data = self._sanitize_data(request_data)
        truncated_data = self._truncate_data(sanitized_data)

        self.logger.info(
            f"Request: {operation}",
            extra={
                "operation": operation,
                "correlation_id": correlation_id,
                "request_data": truncated_data,
                "request_size": len(json.dumps(request_data, default=str)),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def log_response(
        self, operation: str, success: bool, duration_ms: float, **response_data
    ):
        """Log detailed response information."""
        correlation_id = CorrelationIDGenerator.get_correlation_id()

        # Sanitize and truncate response data
        sanitized_data = self._sanitize_data(response_data)
        truncated_data = self._truncate_data(sanitized_data)

        log_method = self.logger.info if success else self.logger.error

        log_method(
            f"Response: {operation} ({'success' if success else 'failure'}) in {duration_ms:.2f}ms",
            extra={
                "operation": operation,
                "correlation_id": correlation_id,
                "success": success,
                "duration_ms": duration_ms,
                "response_data": truncated_data,
                "response_size": len(json.dumps(response_data, default=str)),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def _sanitize_data(self, data: Any) -> Any:
        """Remove sensitive information from data."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(
                    sensitive in key.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list | tuple):
            return [self._sanitize_data(item) for item in data]
        else:
            return data

    def _truncate_data(self, data: Any) -> Any:
        """Truncate data to prevent log overflow."""
        data_str = json.dumps(data, default=str)
        if len(data_str) > self.max_payload_size:
            truncated = data_str[: self.max_payload_size]
            return f"{truncated}... (truncated, original size: {len(data_str)})"
        return data


class ErrorTracker:
    """Comprehensive error tracking and analysis."""

    def __init__(self):
        self.logger = get_structured_logger("maverick_mcp.errors")
        self._error_stats: dict[str, dict[str, Any]] = {}

    def track_error(
        self,
        error: Exception,
        operation: str,
        context: dict[str, Any],
        severity: str = "error",
    ):
        """Track error with detailed context and statistics."""
        error_type = type(error).__name__
        error_key = f"{operation}_{error_type}"

        # Update error statistics
        if error_key not in self._error_stats:
            self._error_stats[error_key] = {
                "first_seen": datetime.now(UTC),
                "last_seen": datetime.now(UTC),
                "count": 0,
                "operation": operation,
                "error_type": error_type,
                "contexts": [],
            }

        stats = self._error_stats[error_key]
        stats["last_seen"] = datetime.now(UTC)
        stats["count"] += 1

        # Keep only recent contexts (last 10)
        stats["contexts"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "context": context,
                "error_message": str(error),
            }
        )
        stats["contexts"] = stats["contexts"][-10:]  # Keep only last 10

        # Get stack trace
        stack_trace = traceback.format_exception(
            type(error), error, error.__traceback__
        )

        # Log the error
        correlation_id = CorrelationIDGenerator.get_correlation_id()

        log_data = {
            "operation": operation,
            "correlation_id": correlation_id,
            "error_type": error_type,
            "error_message": str(error),
            "error_count": stats["count"],
            "first_seen": stats["first_seen"].isoformat(),
            "last_seen": stats["last_seen"].isoformat(),
            "context": context,
            "stack_trace": stack_trace,
            "severity": severity,
        }

        if severity == "critical":
            self.logger.critical(
                f"Critical error in {operation}: {error}", extra=log_data
            )
        elif severity == "error":
            self.logger.error(f"Error in {operation}: {error}", extra=log_data)
        elif severity == "warning":
            self.logger.warning(f"Warning in {operation}: {error}", extra=log_data)

    def get_error_summary(self) -> dict[str, Any]:
        """Get comprehensive error statistics summary."""
        if not self._error_stats:
            return {"message": "No errors tracked"}

        summary = {
            "total_error_types": len(self._error_stats),
            "total_errors": sum(stats["count"] for stats in self._error_stats.values()),
            "error_breakdown": {},
            "most_common_errors": [],
            "recent_errors": [],
        }

        # Error breakdown by type
        for _error_key, stats in self._error_stats.items():
            summary["error_breakdown"][stats["error_type"]] = (
                summary["error_breakdown"].get(stats["error_type"], 0) + stats["count"]
            )

        # Most common errors
        sorted_errors = sorted(
            self._error_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )
        summary["most_common_errors"] = [
            {
                "operation": stats["operation"],
                "error_type": stats["error_type"],
                "count": stats["count"],
                "first_seen": stats["first_seen"].isoformat(),
                "last_seen": stats["last_seen"].isoformat(),
            }
            for _, stats in sorted_errors[:10]
        ]

        # Recent errors
        all_contexts = []
        for stats in self._error_stats.values():
            for context in stats["contexts"]:
                all_contexts.append(
                    {
                        "operation": stats["operation"],
                        "error_type": stats["error_type"],
                        **context,
                    }
                )

        summary["recent_errors"] = sorted(
            all_contexts, key=lambda x: x["timestamp"], reverse=True
        )[:20]

        return summary


class DebugContextManager:
    """Context manager for debug sessions with automatic cleanup."""

    def __init__(
        self,
        operation_name: str,
        enable_profiling: bool = True,
        enable_request_logging: bool = True,
        enable_error_tracking: bool = True,
        **context,
    ):
        self.operation_name = operation_name
        self.enable_profiling = enable_profiling
        self.enable_request_logging = enable_request_logging
        self.enable_error_tracking = enable_error_tracking
        self.context = context

        # Initialize components
        self.profiler = DebugProfiler() if enable_profiling else None
        self.request_logger = (
            RequestResponseLogger() if enable_request_logging else None
        )
        self.error_tracker = ErrorTracker() if enable_error_tracking else None

        self.profile_id = None
        self.start_time = None

    def __enter__(self):
        """Enter debug context."""
        self.start_time = time.time()

        # Set correlation ID if not present
        if not CorrelationIDGenerator.get_correlation_id():
            CorrelationIDGenerator.set_correlation_id()

        # Start profiling
        if self.profiler:
            self.profile_id = self.profiler.start_profile(
                self.operation_name, **self.context
            )

        # Log request
        if self.request_logger:
            self.request_logger.log_request(self.operation_name, **self.context)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit debug context with cleanup."""
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0
        success = exc_type is None

        # Track error if occurred
        if not success and self.error_tracker and exc_val:
            self.error_tracker.track_error(
                exc_val, self.operation_name, self.context, severity="error"
            )

        # End profiling
        if self.profiler and self.profile_id:
            self.profiler.end_profile(
                self.profile_id,
                success=success,
                exception_type=exc_type.__name__ if exc_type else None,
            )

        # Log response
        if self.request_logger:
            response_data = {"exception": str(exc_val)} if exc_val else {}
            self.request_logger.log_response(
                self.operation_name, success, duration_ms, **response_data
            )

    def checkpoint(self, name: str, **data):
        """Add a checkpoint during debug session."""
        if self.profiler and self.profile_id:
            self.profiler.checkpoint(self.profile_id, name, **data)


# Decorator for automatic debug wrapping
def debug_operation(
    operation_name: str | None = None,
    enable_profiling: bool = True,
    enable_request_logging: bool = True,
    enable_error_tracking: bool = True,
    **default_context,
):
    """Decorator to automatically wrap operations with debug context."""

    def decorator(func: Callable) -> Callable:
        actual_operation_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract additional context from function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            context = {**default_context}
            # Add non-sensitive parameters to context
            for param_name, param_value in bound_args.arguments.items():
                if not any(
                    sensitive in param_name.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    if (
                        isinstance(param_value, str | int | float | bool)
                        or param_value is None
                    ):
                        context[param_name] = param_value

            with DebugContextManager(
                actual_operation_name,
                enable_profiling,
                enable_request_logging,
                enable_error_tracking,
                **context,
            ) as debug_ctx:
                result = await func(*args, **kwargs)
                debug_ctx.checkpoint(
                    "function_completed", result_type=type(result).__name__
                )
                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            context = {**default_context}
            for param_name, param_value in bound_args.arguments.items():
                if not any(
                    sensitive in param_name.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    if (
                        isinstance(param_value, str | int | float | bool)
                        or param_value is None
                    ):
                        context[param_name] = param_value

            with DebugContextManager(
                actual_operation_name,
                enable_profiling,
                enable_request_logging,
                enable_error_tracking,
                **context,
            ) as debug_ctx:
                result = func(*args, **kwargs)
                debug_ctx.checkpoint(
                    "function_completed", result_type=type(result).__name__
                )
                return result

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextmanager
def debug_session(
    session_name: str, **context
) -> Generator[DebugContextManager, None, None]:
    """Context manager for manual debug sessions."""
    with DebugContextManager(session_name, **context) as debug_ctx:
        yield debug_ctx


# Global debug utilities
_debug_profiler = DebugProfiler()
_error_tracker = ErrorTracker()


def get_debug_profiler() -> DebugProfiler:
    """Get global debug profiler instance."""
    return _debug_profiler


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance."""
    return _error_tracker


def print_debug_summary():
    """Print comprehensive debug summary to console."""
    print("\n" + "=" * 80)
    print("MAVERICK MCP DEBUG SUMMARY")
    print("=" * 80)

    # Performance metrics
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 40)
    try:
        manager = get_logger_manager()
        dashboard_data = manager.create_dashboard_metrics()

        print(
            f"Log Level Counts: {dashboard_data.get('system_metrics', {}).get('log_level_counts', {})}"
        )
        print(
            f"Active Correlation IDs: {dashboard_data.get('system_metrics', {}).get('active_correlation_ids', 0)}"
        )

        if "memory_stats" in dashboard_data:
            memory_stats = dashboard_data["memory_stats"]
            print(
                f"Memory Usage: {memory_stats.get('rss_mb', 0):.1f}MB RSS, {memory_stats.get('cpu_percent', 0):.1f}% CPU"
            )

    except Exception as e:
        print(f"Error getting performance metrics: {e}")

    # Error summary
    print("\n🚨 ERROR SUMMARY")
    print("-" * 40)
    try:
        error_summary = _error_tracker.get_error_summary()
        if "message" in error_summary:
            print(error_summary["message"])
        else:
            print(f"Total Error Types: {error_summary['total_error_types']}")
            print(f"Total Errors: {error_summary['total_errors']}")

            if error_summary["most_common_errors"]:
                print("\nMost Common Errors:")
                for error in error_summary["most_common_errors"][:5]:
                    print(
                        f"  {error['error_type']} in {error['operation']}: {error['count']} times"
                    )

    except Exception as e:
        print(f"Error getting error summary: {e}")

    print("\n" + "=" * 80)


def enable_debug_mode():
    """Enable comprehensive debug mode."""
    import os

    os.environ["MAVERICK_DEBUG"] = "true"
    print("🐛 Debug mode enabled")
    print("   - Verbose logging activated")
    print("   - Request/response logging enabled")
    print("   - Performance profiling enabled")
    print("   - Error tracking enhanced")


def disable_debug_mode():
    """Disable debug mode."""
    import os

    if "MAVERICK_DEBUG" in os.environ:
        del os.environ["MAVERICK_DEBUG"]
    print("🐛 Debug mode disabled")
