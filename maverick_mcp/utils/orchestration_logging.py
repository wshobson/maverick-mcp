"""
Comprehensive Orchestration Logging System

Provides structured logging for research agent orchestration with:
- Request ID tracking across all components
- Performance timing and metrics
- Parallel execution visibility
- Agent communication tracking
- Resource usage monitoring
"""

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any


# Color codes for better readability in terminal
class LogColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class OrchestrationLogger:
    """Enhanced logger for orchestration components with structured output."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(f"maverick_mcp.orchestration.{component_name}")
        self.request_id: str | None = None
        self.session_context: dict[str, Any] = {}

    def set_request_context(
        self, request_id: str | None = None, session_id: str | None = None, **kwargs
    ):
        """Set context for this request that will be included in all logs."""
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.session_context = {
            "session_id": session_id,
            "request_id": self.request_id,
            **kwargs,
        }

    def _format_message(self, level: str, message: str, **kwargs) -> str:
        """Format log message with consistent structure and colors."""
        color = {
            "DEBUG": LogColors.OKCYAN,
            "INFO": LogColors.OKGREEN,
            "WARNING": LogColors.WARNING,
            "ERROR": LogColors.FAIL,
        }.get(level, "")

        # Build context string
        context_parts = []
        if self.request_id:
            context_parts.append(f"req:{self.request_id}")
        if self.session_context.get("session_id"):
            context_parts.append(f"session:{self.session_context['session_id']}")

        context_str = f"[{' | '.join(context_parts)}]" if context_parts else ""

        # Add component and extra info
        extra_info = " | ".join(f"{k}:{v}" for k, v in kwargs.items() if v is not None)
        extra_str = f" | {extra_info}" if extra_info else ""

        return f"{color}🔧 {self.component_name}{LogColors.ENDC} {context_str}: {message}{extra_str}"

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message("DEBUG", message, **kwargs))

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message("INFO", message, **kwargs))

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message("WARNING", message, **kwargs))

    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message("ERROR", message, **kwargs))


# Global registry of component loggers
_component_loggers: dict[str, OrchestrationLogger] = {}


def get_orchestration_logger(component_name: str) -> OrchestrationLogger:
    """Get or create an orchestration logger for a component."""
    if component_name not in _component_loggers:
        _component_loggers[component_name] = OrchestrationLogger(component_name)
    return _component_loggers[component_name]


def log_method_call(
    component: str | None = None,
    include_params: bool = True,
    include_timing: bool = True,
):
    """
    Decorator to log method entry/exit with timing and parameters.

    Args:
        component: Component name override
        include_params: Whether to log method parameters
        include_timing: Whether to log execution timing
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Determine component name
            comp_name = component
            if not comp_name and args and hasattr(args[0], "__class__"):
                comp_name = args[0].__class__.__name__
            if not comp_name:
                comp_name = func.__module__.split(".")[-1]

            logger = get_orchestration_logger(comp_name)

            # Log method entry
            params_str = ""
            if include_params:
                # Sanitize parameters for logging
                safe_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                if safe_kwargs:
                    params_str = f" | params: {safe_kwargs}"

            logger.info(f"🚀 START {func.__name__}{params_str}")

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)

                # Log successful completion
                duration = time.time() - start_time
                timing_str = f" | duration: {duration:.3f}s" if include_timing else ""

                # Include result summary if available
                result_summary = ""
                if isinstance(result, dict):
                    if "execution_mode" in result:
                        result_summary += f" | mode: {result['execution_mode']}"
                    if "research_confidence" in result:
                        result_summary += (
                            f" | confidence: {result['research_confidence']:.2f}"
                        )
                    if "parallel_execution_stats" in result:
                        stats = result["parallel_execution_stats"]
                        result_summary += f" | tasks: {stats.get('successful_tasks', 0)}/{stats.get('total_tasks', 0)}"

                logger.info(f"✅ SUCCESS {func.__name__}{timing_str}{result_summary}")
                return result

            except Exception as e:
                # Log error
                duration = time.time() - start_time
                timing_str = f" | duration: {duration:.3f}s" if include_timing else ""
                logger.error(f"❌ ERROR {func.__name__}{timing_str} | error: {str(e)}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Handle synchronous functions
            comp_name = component
            if not comp_name and args and hasattr(args[0], "__class__"):
                comp_name = args[0].__class__.__name__
            if not comp_name:
                comp_name = func.__module__.split(".")[-1]

            logger = get_orchestration_logger(comp_name)

            # Log method entry
            params_str = ""
            if include_params:
                safe_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                if safe_kwargs:
                    params_str = f" | params: {safe_kwargs}"

            logger.info(f"🚀 START {func.__name__}{params_str}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                duration = time.time() - start_time
                timing_str = f" | duration: {duration:.3f}s" if include_timing else ""
                logger.info(f"✅ SUCCESS {func.__name__}{timing_str}")
                return result

            except Exception as e:
                duration = time.time() - start_time
                timing_str = f" | duration: {duration:.3f}s" if include_timing else ""
                logger.error(f"❌ ERROR {func.__name__}{timing_str} | error: {str(e)}")
                raise

        # Return appropriate wrapper based on function type
        if hasattr(func, "_is_coroutine") or "async" in str(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def log_parallel_execution(component: str, task_description: str, task_count: int):
    """Context manager for logging parallel execution blocks."""
    logger = get_orchestration_logger(component)

    logger.info(f"🔄 PARALLEL_START {task_description} | tasks: {task_count}")
    start_time = time.time()

    try:
        yield logger

        duration = time.time() - start_time
        logger.info(
            f"🎯 PARALLEL_SUCCESS {task_description} | duration: {duration:.3f}s | tasks: {task_count}"
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"💥 PARALLEL_ERROR {task_description} | duration: {duration:.3f}s | error: {str(e)}"
        )
        raise


@contextmanager
def log_agent_execution(
    agent_type: str, task_id: str, focus_areas: list[str] | None = None
):
    """Context manager for logging individual agent execution."""
    logger = get_orchestration_logger(f"{agent_type}Agent")

    focus_str = f" | focus: {focus_areas}" if focus_areas else ""
    logger.info(f"🤖 AGENT_START {task_id}{focus_str}")

    start_time = time.time()

    try:
        yield logger

        duration = time.time() - start_time
        logger.info(f"🎉 AGENT_SUCCESS {task_id} | duration: {duration:.3f}s")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"🔥 AGENT_ERROR {task_id} | duration: {duration:.3f}s | error: {str(e)}"
        )
        raise


def log_tool_invocation(tool_name: str, request_data: dict[str, Any] | None = None):
    """Log MCP tool invocation with request details."""
    logger = get_orchestration_logger("MCPToolRegistry")

    request_summary = ""
    if request_data:
        if "query" in request_data:
            request_summary += f" | query: '{request_data['query'][:50]}...'"
        if "research_scope" in request_data:
            request_summary += f" | scope: {request_data['research_scope']}"
        if "persona" in request_data:
            request_summary += f" | persona: {request_data['persona']}"

    logger.info(f"🔧 TOOL_INVOKE {tool_name}{request_summary}")


def log_synthesis_operation(
    operation: str, input_count: int, output_summary: str | None = None
):
    """Log result synthesis operations."""
    logger = get_orchestration_logger("ResultSynthesis")

    summary_str = f" | output: {output_summary}" if output_summary else ""
    logger.info(f"🧠 SYNTHESIS {operation} | inputs: {input_count}{summary_str}")


def log_fallback_trigger(component: str, reason: str, fallback_action: str):
    """Log when fallback mechanisms are triggered."""
    logger = get_orchestration_logger(component)
    logger.warning(f"⚠️ FALLBACK_TRIGGER {reason} | action: {fallback_action}")


def log_performance_metrics(component: str, metrics: dict[str, Any]):
    """Log performance metrics for monitoring."""
    logger = get_orchestration_logger(component)

    metrics_str = " | ".join(f"{k}: {v}" for k, v in metrics.items())
    logger.info(f"📊 PERFORMANCE_METRICS | {metrics_str}")


def log_resource_usage(
    component: str,
    api_calls: int | None = None,
    cache_hits: int | None = None,
    memory_mb: float | None = None,
):
    """Log resource usage statistics."""
    logger = get_orchestration_logger(component)

    usage_parts = []
    if api_calls is not None:
        usage_parts.append(f"api_calls: {api_calls}")
    if cache_hits is not None:
        usage_parts.append(f"cache_hits: {cache_hits}")
    if memory_mb is not None:
        usage_parts.append(f"memory_mb: {memory_mb:.1f}")

    if usage_parts:
        usage_str = " | ".join(usage_parts)
        logger.info(f"📈 RESOURCE_USAGE | {usage_str}")


# Export key functions
__all__ = [
    "OrchestrationLogger",
    "get_orchestration_logger",
    "log_method_call",
    "log_parallel_execution",
    "log_agent_execution",
    "log_tool_invocation",
    "log_synthesis_operation",
    "log_fallback_trigger",
    "log_performance_metrics",
    "log_resource_usage",
]
