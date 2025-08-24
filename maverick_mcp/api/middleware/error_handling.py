"""
Error handling middleware for FastAPI applications.

This middleware provides centralized error handling, logging, and monitoring
integration for all unhandled exceptions in the API.
"""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from maverick_mcp.api.error_handling import handle_api_error
from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.monitoring import get_monitoring_service

logger = get_logger(__name__)
monitoring = get_monitoring_service()


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and handle all unhandled exceptions.

    This middleware:
    1. Catches any unhandled exceptions from route handlers
    2. Logs errors with full context
    3. Sends errors to monitoring (Sentry)
    4. Returns structured error responses to clients
    5. Adds request IDs for tracing
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and handle any exceptions."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Add request ID to response headers
        start_time = time.time()

        try:
            # Add breadcrumb for monitoring
            monitoring.add_breadcrumb(
                message=f"{request.method} {request.url.path}",
                category="request",
                level="info",
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.url.query),
                },
            )

            # Process the request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log successful request
            duration = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration": duration,
                },
            )

            return response

        except Exception as exc:
            # Calculate request duration
            duration = time.time() - start_time

            # Log the error
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "duration": duration,
                    "error_type": type(exc).__name__,
                },
            )

            # Handle the error and get structured response
            error_response = handle_api_error(
                request,
                exc,
                context={
                    "request_id": request_id,
                    "duration": duration,
                },
            )

            # Add request ID to error response
            error_response.headers["X-Request-ID"] = request_id

            return error_response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request tracing information.

    This middleware adds:
    1. Request IDs to all requests
    2. User context for authenticated requests
    3. Performance tracking
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add tracing context to requests."""
        # Check if request ID already exists (from error handling middleware)
        if not hasattr(request.state, "request_id"):
            request.state.request_id = str(uuid.uuid4())

        # Extract user context if available
        user_id = None
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "id", None)
            monitoring.set_user_context(user_id)

        # Add monitoring context
        monitoring.add_breadcrumb(
            message="Request context",
            category="request",
            data={
                "request_id": request.state.request_id,
                "user_id": user_id,
                "path": request.url.path,
            },
        )

        # Process request with monitoring transaction
        with monitoring.transaction(
            name=f"{request.method} {request.url.path}", op="http.server"
        ):
            response = await call_next(request)

        # Clear user context after request
        if user_id:
            monitoring.set_user_context(None)

        return response
