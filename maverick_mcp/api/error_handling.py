"""
Enhanced error handling framework for MaverickMCP API.

This module provides centralized error handling with structured responses,
proper logging, monitoring integration, and client-friendly error messages.
"""

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, OperationalError

from maverick_mcp.exceptions import (
    APIRateLimitError,
    AuthenticationError,
    AuthorizationError,
    CacheConnectionError,
    CircuitBreakerError,
    ConflictError,
    DatabaseConnectionError,
    DataIntegrityError,
    DataNotFoundError,
    ExternalServiceError,
    MaverickException,
    NotFoundError,
    RateLimitError,
    ValidationError,
    WebhookError,
)
from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.monitoring import get_monitoring_service
from maverick_mcp.validation.responses import error_response, validation_error_response

logger = get_logger(__name__)
monitoring = get_monitoring_service()


class ErrorHandler:
    """Centralized error handler with monitoring integration."""

    def __init__(self):
        self.error_mappings = self._build_error_mappings()

    def _build_error_mappings(self) -> dict[type[Exception], dict[str, Any]]:
        """Build mapping of exception types to response details."""
        return {
            # MaverickMCP exceptions
            ValidationError: {
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "code": "VALIDATION_ERROR",
                "log_level": "warning",
            },
            AuthenticationError: {
                "status_code": status.HTTP_401_UNAUTHORIZED,
                "code": "AUTHENTICATION_ERROR",
                "log_level": "warning",
            },
            AuthorizationError: {
                "status_code": status.HTTP_403_FORBIDDEN,
                "code": "AUTHORIZATION_ERROR",
                "log_level": "warning",
            },
            DataNotFoundError: {
                "status_code": status.HTTP_404_NOT_FOUND,
                "code": "DATA_NOT_FOUND",
                "log_level": "info",
            },
            APIRateLimitError: {
                "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
                "code": "RATE_LIMIT_EXCEEDED",
                "log_level": "warning",
            },
            CircuitBreakerError: {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "code": "SERVICE_UNAVAILABLE",
                "log_level": "error",
            },
            DatabaseConnectionError: {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "code": "DATABASE_CONNECTION_ERROR",
                "log_level": "error",
            },
            CacheConnectionError: {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "code": "CACHE_CONNECTION_ERROR",
                "log_level": "error",
            },
            DataIntegrityError: {
                "status_code": status.HTTP_409_CONFLICT,
                "code": "DATA_INTEGRITY_ERROR",
                "log_level": "error",
            },
            # API errors from validation module
            NotFoundError: {
                "status_code": status.HTTP_404_NOT_FOUND,
                "code": "NOT_FOUND",
                "log_level": "info",
            },
            ConflictError: {
                "status_code": status.HTTP_409_CONFLICT,
                "code": "CONFLICT",
                "log_level": "warning",
            },
            RateLimitError: {
                "status_code": status.HTTP_429_TOO_MANY_REQUESTS,
                "code": "RATE_LIMIT_EXCEEDED",
                "log_level": "warning",
            },
            ExternalServiceError: {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "code": "EXTERNAL_SERVICE_ERROR",
                "log_level": "error",
            },
            WebhookError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "code": "WEBHOOK_ERROR",
                "log_level": "warning",
            },
            # SQLAlchemy exceptions
            IntegrityError: {
                "status_code": status.HTTP_409_CONFLICT,
                "code": "DATABASE_INTEGRITY_ERROR",
                "log_level": "error",
            },
            OperationalError: {
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "code": "DATABASE_OPERATIONAL_ERROR",
                "log_level": "error",
            },
            # Third-party API exceptions
            ValueError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "code": "INVALID_REQUEST",
                "log_level": "warning",
            },
            KeyError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "code": "MISSING_REQUIRED_FIELD",
                "log_level": "warning",
            },
            TypeError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "code": "TYPE_ERROR",
                "log_level": "warning",
            },
        }

    def handle_exception(
        self,
        request: Request,
        exception: Exception,
        context: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """
        Handle exception and return structured error response.

        Args:
            request: FastAPI request object
            exception: The exception to handle
            context: Additional context for logging

        Returns:
            JSONResponse with structured error
        """
        # Generate trace ID for this error
        trace_id = str(uuid.uuid4())

        # Get error details from mapping
        error_info = self._get_error_info(exception)

        # Log the error with full context
        self._log_error(
            exception=exception,
            trace_id=trace_id,
            request=request,
            error_info=error_info,
            context=context,
        )

        # Send to monitoring service
        self._send_to_monitoring(
            exception=exception,
            trace_id=trace_id,
            request=request,
            context=context,
        )

        # Build client-friendly response
        response_data = self._build_error_response(
            exception=exception,
            error_info=error_info,
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=error_info["status_code"],
            content=response_data,
        )

    def _get_error_info(self, exception: Exception) -> dict[str, Any]:
        """Get error information for the exception type."""
        # Check for exact type match first
        exc_type = type(exception)
        if exc_type in self.error_mappings:
            return self.error_mappings[exc_type]

        # Check for inheritance
        for error_type, info in self.error_mappings.items():
            if isinstance(exception, error_type):
                return info

        # Default for unknown exceptions
        return {
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "code": "INTERNAL_ERROR",
            "log_level": "error",
        }

    def _log_error(
        self,
        exception: Exception,
        trace_id: str,
        request: Request,
        error_info: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log error with full context."""
        log_data = {
            "trace_id": trace_id,
            "error_type": type(exception).__name__,
            "error_code": error_info["code"],
            "status_code": error_info["status_code"],
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        # Add exception details if available
        if isinstance(exception, MaverickException):
            log_data["error_details"] = exception.to_dict()

        # Add custom context
        if context:
            log_data["context"] = context

        # Log at appropriate level
        log_level = error_info["log_level"]
        if log_level == "error":
            logger.error(
                f"Error handling request: {str(exception)}",
                exc_info=True,
                extra=log_data,
            )
        elif log_level == "warning":
            logger.warning(
                f"Request failed: {str(exception)}",
                extra=log_data,
            )
        else:
            logger.info(
                f"Request rejected: {str(exception)}",
                extra=log_data,
            )

    def _send_to_monitoring(
        self,
        exception: Exception,
        trace_id: str,
        request: Request,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send error to monitoring service (Sentry)."""
        monitoring_context = {
            "trace_id": trace_id,
            "request": {
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
            },
        }

        if context:
            monitoring_context["custom_context"] = context

        # Only send certain errors to Sentry
        error_info = self._get_error_info(exception)
        if error_info["log_level"] in ["error", "warning"]:
            monitoring.capture_exception(exception, **monitoring_context)

    def _build_error_response(
        self,
        exception: Exception,
        error_info: dict[str, Any],
        trace_id: str,
    ) -> dict[str, Any]:
        """Build client-friendly error response."""
        # Extract error details
        if isinstance(exception, MaverickException):
            message = exception.message
            context = exception.context
        elif isinstance(exception, HTTPException):
            message = exception.detail
            context = None
        else:
            # Generic message for unknown errors
            message = self._get_safe_error_message(exception, error_info["code"])
            context = None

        return error_response(
            code=error_info["code"],
            message=message,
            status_code=error_info["status_code"],
            context=context,
            trace_id=trace_id,
        )

    def _get_safe_error_message(self, exception: Exception, code: str) -> str:
        """Get safe error message for client."""
        safe_messages = {
            "INTERNAL_ERROR": "An unexpected error occurred. Please try again later.",
            "DATABASE_INTEGRITY_ERROR": "Data conflict detected. Please check your input.",
            "DATABASE_OPERATIONAL_ERROR": "Database temporarily unavailable.",
            "INVALID_REQUEST": "Invalid request format.",
            "MISSING_REQUIRED_FIELD": "Required field missing from request.",
            "TYPE_ERROR": "Invalid data type in request.",
        }

        return safe_messages.get(code, str(exception))


# Global error handler instance
error_handler = ErrorHandler()


def handle_api_error(
    request: Request,
    exception: Exception,
    context: dict[str, Any] | None = None,
) -> JSONResponse:
    """
    Main entry point for API error handling.

    Args:
        request: FastAPI request
        exception: Exception to handle
        context: Additional context

    Returns:
        Structured error response
    """
    return error_handler.handle_exception(request, exception, context)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation errors."""
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "code": "VALIDATION_ERROR",
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "context": {"input": error.get("input")},
            }
        )

    trace_id = str(uuid.uuid4())

    # Log validation errors
    logger.warning(
        "Request validation failed",
        extra={
            "trace_id": trace_id,
            "path": request.url.path,
            "errors": errors,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=validation_error_response(errors, trace_id),
    )


def create_error_handlers() -> dict[Any, Callable]:
    """Create error handlers for FastAPI app."""
    return {
        RequestValidationError: validation_exception_handler,
        Exception: lambda request, exc: handle_api_error(request, exc),
    }


# Decorator for wrapping functions with error handling
def with_error_handling(context_fn: Callable[[Any], dict[str, Any]] | None = None):
    """
    Decorator to wrap functions with proper error handling.

    Args:
        context_fn: Optional function to extract context from arguments
    """

    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Extract context if function provided
                context = context_fn(*args, **kwargs) if context_fn else {}

                # Get request from args/kwargs
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                if not request and "request" in kwargs:
                    request = kwargs["request"]

                if request:
                    return handle_api_error(request, e, context)
                else:
                    # Re-raise if no request object
                    raise

        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract context if function provided
                context = context_fn(*args, **kwargs) if context_fn else {}

                # Get request from args/kwargs
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                if not request and "request" in kwargs:
                    request = kwargs["request"]

                if request:
                    return handle_api_error(request, e, context)
                else:
                    # Re-raise if no request object
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
