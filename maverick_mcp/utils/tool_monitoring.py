"""
Tool execution monitoring utilities.

This module provides functions for monitoring tool execution,
including timing, error handling, and performance analysis.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from maverick_mcp.utils.logging import get_logger

logger = get_logger("maverick_mcp.utils.tool_monitoring")


@dataclass
class ExecutionResult:
    """Result of tool execution monitoring."""

    result: Any
    execution_time: float
    success: bool
    error: Exception | None = None


class ToolMonitor:
    """Monitors tool execution for performance and errors."""

    def __init__(self, tool_name: str, user_id: int | None = None):
        """
        Initialize tool monitor.

        Args:
            tool_name: Name of the tool being monitored
            user_id: ID of the user executing the tool
        """
        self.tool_name = tool_name
        self.user_id = user_id

    async def execute_with_monitoring(
        self,
        func: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        estimation: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute a tool function with comprehensive monitoring.

        Args:
            func: The async function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            estimation: Optional estimation data for comparison

        Returns:
            ExecutionResult: Contains result, timing, and error information
        """
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log successful execution
            self._log_successful_execution(execution_time, result, estimation)

            # Check for potential underestimation
            self._check_for_underestimation(execution_time, estimation)

            return ExecutionResult(
                result=result,
                execution_time=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Log failed execution
            self._log_failed_execution(execution_time, e)

            return ExecutionResult(
                result=None,
                execution_time=execution_time,
                success=False,
                error=e,
            )

    def _log_successful_execution(
        self,
        execution_time: float,
        result: Any,
        estimation: dict[str, Any] | None,
    ) -> None:
        """Log successful tool execution."""
        log_data = {
            "tool_name": self.tool_name,
            "user_id": self.user_id,
            "execution_time_seconds": round(execution_time, 3),
            "has_result": result is not None,
        }

        if estimation:
            log_data.update(
                {
                    "estimated_llm_calls": estimation.get("llm_calls", 0),
                    "estimated_tokens": estimation.get("total_tokens", 0),
                }
            )

        logger.info(f"Tool executed successfully: {self.tool_name}", extra=log_data)

    def _log_failed_execution(self, execution_time: float, error: Exception) -> None:
        """Log failed tool execution."""
        logger.error(
            f"Tool execution failed: {self.tool_name}",
            extra={
                "tool_name": self.tool_name,
                "user_id": self.user_id,
                "execution_time_seconds": round(execution_time, 3),
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

    def _check_for_underestimation(
        self,
        execution_time: float,
        estimation: dict[str, Any] | None,
    ) -> None:
        """Check if execution time indicates potential underestimation."""
        # Long execution time may indicate underestimation
        if execution_time > 30:
            log_data = {
                "tool_name": self.tool_name,
                "execution_time_seconds": round(execution_time, 3),
                "note": "Consider reviewing estimate if this persists",
            }

            if estimation:
                log_data.update(
                    {
                        "estimated_llm_calls": estimation.get("llm_calls", 0),
                        "estimated_tokens": estimation.get("total_tokens", 0),
                        "complexity": estimation.get("complexity", "unknown"),
                        "confidence": estimation.get("confidence", 0.5),
                    }
                )

            logger.warning(
                f"Long execution time for tool: {self.tool_name}", extra=log_data
            )

    def add_usage_info_to_result(
        self, result: Any, usage_info: dict[str, Any] | None
    ) -> Any:
        """
        Add usage information to the tool result.

        Args:
            result: The tool execution result
            usage_info: Usage information to add

        Returns:
            The result with usage info added (if applicable)
        """
        if usage_info and isinstance(result, dict):
            result["usage"] = usage_info

        return result


def create_tool_monitor(tool_name: str, user_id: int | None = None) -> ToolMonitor:
    """
    Create a tool monitor instance.

    Args:
        tool_name: Name of the tool being monitored
        user_id: ID of the user executing the tool

    Returns:
        ToolMonitor: Configured tool monitor
    """
    return ToolMonitor(tool_name, user_id)


def should_alert_for_performance(
    execution_time: float, threshold: float = 30.0
) -> tuple[bool, str]:
    """
    Check if an alert should be raised for performance issues.

    Args:
        execution_time: Execution time in seconds
        threshold: Performance threshold in seconds

    Returns:
        tuple: (should_alert, alert_message)
    """
    if execution_time > threshold:
        return (
            True,
            f"Tool execution took {execution_time:.2f}s (threshold: {threshold}s)",
        )

    return False, ""
