"""
Monitoring middleware for FastMCP and FastAPI applications.

This module provides comprehensive monitoring middleware that automatically:
- Tracks request metrics (count, duration, size)
- Creates distributed traces for all requests
- Monitors database and cache operations
- Tracks business metrics and user behavior
- Integrates with Prometheus and OpenTelemetry
"""

import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from maverick_mcp.utils.logging import get_logger, user_id_var
from maverick_mcp.utils.monitoring import (
    active_connections,
    concurrent_requests,
    request_counter,
    request_duration,
    request_size_bytes,
    response_size_bytes,
    track_authentication,
    track_rate_limit_hit,
    track_security_violation,
    update_performance_metrics,
)
from maverick_mcp.utils.tracing import get_tracing_service, trace_operation

logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive monitoring middleware for FastAPI applications.

    Automatically tracks:
    - Request/response metrics
    - Distributed tracing
    - Performance monitoring
    - Security events
    - Business metrics
    """

    def __init__(self, app, enable_detailed_logging: bool = True):
        super().__init__(app)
        self.enable_detailed_logging = enable_detailed_logging
        self.tracing = get_tracing_service()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive monitoring."""
        # Start timing
        start_time = time.time()

        # Track active connections
        active_connections.inc()
        concurrent_requests.inc()

        # Extract request information
        method = request.method
        path = request.url.path
        endpoint = self._normalize_endpoint(path)
        user_agent = request.headers.get("user-agent", "unknown")

        # Calculate request size
        content_length = request.headers.get("content-length")
        req_size = int(content_length) if content_length else 0

        # Extract user information for monitoring
        user_id = self._extract_user_id(request)
        user_type = self._determine_user_type(request, user_id)

        # Set context variables for logging
        if user_id:
            user_id_var.set(str(user_id))

        response = None
        status_code = 500
        error_type = None

        # Create tracing span for the entire request
        with trace_operation(
            f"{method} {endpoint}",
            attributes={
                "http.method": method,
                "http.route": endpoint,
                "http.user_agent": user_agent[:100],  # Truncate long user agents
                "user.id": str(user_id) if user_id else "anonymous",
                "user.type": user_type,
                "http.request_size": req_size,
            },
        ) as span:
            try:
                # Process the request
                response = await call_next(request)
                status_code = response.status_code

                # Track successful request
                if span:
                    span.set_attribute("http.status_code", status_code)
                    span.set_attribute(
                        "http.response_size", self._get_response_size(response)
                    )

                # Track authentication events
                if self._is_auth_endpoint(endpoint):
                    auth_status = "success" if 200 <= status_code < 300 else "failure"
                    track_authentication(
                        method="bearer_token",
                        status=auth_status,
                        user_agent=user_agent[:50],
                    )

                # Track rate limiting
                if status_code == 429:
                    track_rate_limit_hit(
                        user_id=str(user_id) if user_id else "anonymous",
                        endpoint=endpoint,
                        limit_type="request_rate",
                    )

            except Exception as e:
                error_type = type(e).__name__
                status_code = 500

                # Record exception in trace
                if span:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", error_type)

                # Track security violations for certain errors
                if self._is_security_error(e):
                    track_security_violation(
                        violation_type=error_type,
                        severity="high" if status_code >= 400 else "medium",
                    )

                # Re-raise the exception
                raise

            finally:
                # Calculate duration
                duration = time.time() - start_time

                # Determine final status for metrics
                final_status = "success" if 200 <= status_code < 400 else "error"

                # Track request metrics
                request_counter.labels(
                    method=method,
                    endpoint=endpoint,
                    status=final_status,
                    user_type=user_type,
                ).inc()

                request_duration.labels(
                    method=method, endpoint=endpoint, user_type=user_type
                ).observe(duration)

                # Track request/response sizes
                if req_size > 0:
                    request_size_bytes.labels(method=method, endpoint=endpoint).observe(
                        req_size
                    )

                if response:
                    resp_size = self._get_response_size(response)
                    if resp_size > 0:
                        response_size_bytes.labels(
                            method=method, endpoint=endpoint, status=str(status_code)
                        ).observe(resp_size)

                # Update performance metrics periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    try:
                        update_performance_metrics()
                    except Exception as e:
                        logger.warning(f"Failed to update performance metrics: {e}")

                # Log detailed request information
                if self.enable_detailed_logging:
                    self._log_request_details(
                        method, endpoint, status_code, duration, user_id, error_type
                    )

                # Update connection counters
                active_connections.dec()
                concurrent_requests.dec()

        return response

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics (replace IDs with placeholders)."""
        # Replace UUIDs and IDs in paths
        import re

        # Replace UUID patterns
        path = re.sub(r"/[a-f0-9-]{36}", "/{uuid}", path)

        # Replace numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)

        # Replace API keys or tokens
        path = re.sub(r"/[a-zA-Z0-9]{20,}", "/{token}", path)

        return path

    def _extract_user_id(self, request: Request) -> str | None:
        """Extract user ID from request (from JWT, session, etc.)."""
        # Check Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                # In a real implementation, you'd decode the JWT
                # For now, we'll check if there's a user context
                if hasattr(request.state, "user_id"):
                    return request.state.user_id
            except Exception:
                pass

        # Check for user ID in path parameters
        if hasattr(request, "path_params") and "user_id" in request.path_params:
            return request.path_params["user_id"]

        return None

    def _determine_user_type(self, request: Request, user_id: str | None) -> str:
        """Determine user type for metrics."""
        if not user_id:
            return "anonymous"

        # Check if it's an admin user (you'd implement your own logic)
        if hasattr(request.state, "user_role"):
            return request.state.user_role

        # Check for API key usage
        if request.headers.get("x-api-key"):
            return "api_user"

        return "authenticated"

    def _is_auth_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is authentication-related."""
        auth_endpoints = ["/login", "/auth", "/token", "/signup", "/register"]
        return any(auth_ep in endpoint for auth_ep in auth_endpoints)

    def _is_security_error(self, exception: Exception) -> bool:
        """Check if exception indicates a security issue."""
        security_errors = [
            "PermissionError",
            "Unauthorized",
            "Forbidden",
            "ValidationError",
            "SecurityError",
        ]
        return any(error in str(type(exception)) for error in security_errors)

    def _get_response_size(self, response: Response) -> int:
        """Calculate response size in bytes."""
        content_length = response.headers.get("content-length")
        if content_length:
            return int(content_length)

        # Estimate size if content-length is not set
        if hasattr(response, "body") and response.body:
            return len(response.body)

        return 0

    def _log_request_details(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_id: str | None,
        error_type: str | None,
    ):
        """Log detailed request information."""
        log_data = {
            "http_method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": int(duration * 1000),
            "user_id": str(user_id) if user_id else None,
        }

        if error_type:
            log_data["error_type"] = error_type

        if status_code >= 400:
            logger.warning(f"HTTP {status_code}: {method} {endpoint}", extra=log_data)
        else:
            logger.info(f"HTTP {status_code}: {method} {endpoint}", extra=log_data)


class MCPToolMonitoringWrapper:
    """
    Wrapper for MCP tools to add monitoring capabilities.

    This class wraps MCP tool execution to automatically:
    - Track tool usage metrics
    - Create distributed traces
    - Monitor performance
    - Track credit usage
    """

    def __init__(self, enable_tracing: bool = True):
        self.enable_tracing = enable_tracing
        self.tracing = get_tracing_service()

    def monitor_tool(self, tool_func: Callable) -> Callable:
        """
        Decorator to add monitoring to MCP tools.

        Args:
            tool_func: The MCP tool function to monitor

        Returns:
            Wrapped function with monitoring
        """
        from functools import wraps

        @wraps(tool_func)
        async def wrapper(*args, **kwargs):
            tool_name = tool_func.__name__
            start_time = time.time()

            # Extract user context from args
            user_id = None
            for arg in args:
                if hasattr(arg, "user_id"):
                    user_id = arg.user_id
                    break
                # Check if it's an MCP context
                if hasattr(arg, "__class__") and "Context" in arg.__class__.__name__:
                    # Extract user from context if available
                    if hasattr(arg, "user_id"):
                        user_id = arg.user_id

            # Set context for logging
            if user_id:
                user_id_var.set(str(user_id))

            # Create tracing span
            with trace_operation(
                f"tool.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "user.id": str(user_id) if user_id else "anonymous",
                    "tool.args_count": len(args),
                    "tool.kwargs_count": len(kwargs),
                },
            ) as span:
                try:
                    # Execute the tool
                    result = await tool_func(*args, **kwargs)

                    # Calculate execution time
                    duration = time.time() - start_time

                    # Track successful execution
                    from maverick_mcp.utils.monitoring import track_tool_usage

                    track_tool_usage(
                        tool_name=tool_name,
                        user_id=str(user_id) if user_id else "anonymous",
                        credits_spent=self._estimate_credits(tool_name, duration),
                        duration=duration,
                        status="success",
                        complexity=self._determine_complexity(tool_name, kwargs),
                    )

                    # Add attributes to span
                    if span:
                        span.set_attribute("tool.duration_seconds", duration)
                        span.set_attribute("tool.success", True)
                        span.set_attribute("tool.result_size", len(str(result)))

                    # Add usage information to result if it's a dict
                    if isinstance(result, dict):
                        result["_monitoring"] = {
                            "execution_time_ms": int(duration * 1000),
                            "tool_name": tool_name,
                            "timestamp": time.time(),
                        }

                    return result

                except Exception as e:
                    # Calculate execution time
                    duration = time.time() - start_time
                    error_type = type(e).__name__

                    # Track failed execution
                    from maverick_mcp.utils.monitoring import track_tool_error

                    track_tool_error(
                        tool_name=tool_name,
                        error_type=error_type,
                        complexity=self._determine_complexity(tool_name, kwargs),
                    )

                    # Add error attributes to span
                    if span:
                        span.set_attribute("tool.duration_seconds", duration)
                        span.set_attribute("tool.success", False)
                        span.set_attribute("error.type", error_type)
                        span.record_exception(e)

                    logger.error(
                        f"Tool execution failed: {tool_name}",
                        extra={
                            "tool_name": tool_name,
                            "user_id": str(user_id) if user_id else None,
                            "duration_ms": int(duration * 1000),
                            "error_type": error_type,
                        },
                        exc_info=True,
                    )

                    # Re-raise the exception
                    raise

        return wrapper

    def _estimate_credits(self, tool_name: str, duration: float) -> int:
        """Estimate credits used based on tool complexity and duration."""
        # Simple credit estimation without external mapping
        # Base credits for most tools
        base_credits = 1

        # Complex tools get more base credits
        complex_tools = [
            "get_portfolio_optimization",
            "get_market_analysis",
            "screen_stocks",
            "get_full_technical_analysis",
        ]
        if any(complex_tool in tool_name for complex_tool in complex_tools):
            base_credits = 3

        # Adjust for long-running operations
        if duration > 30:
            return base_credits * 2
        elif duration > 10:
            return base_credits + 1

        return base_credits

    def _determine_complexity(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        """Determine tool complexity based on parameters."""
        # Simple heuristics for complexity
        if "limit" in kwargs:
            limit = kwargs.get("limit", 0)
            if limit > 100:
                return "high"
            elif limit > 50:
                return "medium"

        if "symbols" in kwargs:
            symbols = kwargs.get("symbols", [])
            if isinstance(symbols, list) and len(symbols) > 10:
                return "high"
            elif isinstance(symbols, list) and len(symbols) > 5:
                return "medium"

        # Check for complex analysis tools
        complex_tools = [
            "get_portfolio_optimization",
            "get_market_analysis",
            "screen_stocks",
        ]
        if any(complex_tool in tool_name for complex_tool in complex_tools):
            return "high"

        return "standard"


def create_monitoring_middleware(
    enable_detailed_logging: bool = True,
) -> MonitoringMiddleware:
    """Create a monitoring middleware instance."""
    return MonitoringMiddleware(enable_detailed_logging=enable_detailed_logging)


def create_tool_monitor(enable_tracing: bool = True) -> MCPToolMonitoringWrapper:
    """Create a tool monitoring wrapper instance."""
    return MCPToolMonitoringWrapper(enable_tracing=enable_tracing)


# Global instances
_monitoring_middleware: MonitoringMiddleware | None = None
_tool_monitor: MCPToolMonitoringWrapper | None = None


def get_monitoring_middleware() -> MonitoringMiddleware:
    """Get or create the global monitoring middleware."""
    global _monitoring_middleware
    if _monitoring_middleware is None:
        _monitoring_middleware = create_monitoring_middleware()
    return _monitoring_middleware


def get_tool_monitor() -> MCPToolMonitoringWrapper:
    """Get or create the global tool monitor."""
    global _tool_monitor
    if _tool_monitor is None:
        _tool_monitor = create_tool_monitor()
    return _tool_monitor
