"""
Validation middleware for FastAPI to standardize error handling.

This module provides middleware to catch validation errors and
return standardized error responses.
"""

import logging
import time
import traceback
import uuid

from fastapi import Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from maverick_mcp.exceptions import MaverickException

from .responses import error_response, validation_error_response

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle validation errors and API exceptions."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and handle exceptions."""
        # Generate trace ID for request tracking
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            return response

        except MaverickException as e:
            logger.warning(
                f"API error: {e.error_code} - {e.message}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                    "error_code": e.error_code,
                },
            )
            return JSONResponse(
                status_code=e.status_code,
                content=error_response(
                    code=e.error_code,
                    message=e.message,
                    status_code=e.status_code,
                    field=e.field,
                    context=e.context,
                    trace_id=trace_id,
                ),
            )

        except RequestValidationError as e:
            logger.warning(
                f"Request validation error: {str(e)}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                },
            )

            # Convert Pydantic validation errors to our format
            errors = []
            for error in e.errors():
                errors.append(
                    {
                        "code": "VALIDATION_ERROR",
                        "field": ".".join(str(x) for x in error["loc"]),
                        "message": error["msg"],
                        "context": {"input": error.get("input"), "type": error["type"]},
                    }
                )

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=validation_error_response(errors=errors, trace_id=trace_id),
            )

        except ValidationError as e:
            logger.warning(
                f"Pydantic validation error: {str(e)}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                },
            )

            # Convert Pydantic validation errors
            errors = []
            for error in e.errors():
                errors.append(
                    {
                        "code": "VALIDATION_ERROR",
                        "field": ".".join(str(x) for x in error["loc"]),
                        "message": error["msg"],
                        "context": {"input": error.get("input"), "type": error["type"]},
                    }
                )

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=validation_error_response(errors=errors, trace_id=trace_id),
            )

        except Exception as e:
            logger.error(
                f"Unexpected error: {str(e)}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                    "traceback": traceback.format_exc(),
                },
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response(
                    code="INTERNAL_ERROR",
                    message="An unexpected error occurred",
                    status_code=500,
                    trace_id=trace_id,
                ),
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting based on API keys."""

    def __init__(self, app, rate_limit_store=None):
        super().__init__(app)
        self.rate_limit_store = rate_limit_store or {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Check rate limits before processing request."""
        # Skip rate limiting for health checks and internal endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Extract API key from headers
        api_key = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        elif "x-api-key" in request.headers:
            api_key = request.headers["x-api-key"]

        if api_key:
            # Check rate limit (simplified implementation)
            # In production, use Redis or similar for distributed rate limiting
            current_time = int(time.time())
            window_start = current_time - 60  # 1-minute window

            # Clean old entries
            key_requests = self.rate_limit_store.get(api_key, [])
            key_requests = [ts for ts in key_requests if ts > window_start]

            # Check limit (default 60 requests per minute)
            if len(key_requests) >= 60:
                trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=error_response(
                        code="RATE_LIMIT_EXCEEDED",
                        message="Rate limit exceeded",
                        status_code=429,
                        context={
                            "limit": 60,
                            "window": "1 minute",
                            "retry_after": 60 - (current_time % 60),
                        },
                        trace_id=trace_id,
                    ),
                    headers={"Retry-After": "60"},
                )

            # Add current request
            key_requests.append(current_time)
            self.rate_limit_store[api_key] = key_requests

        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and request validation."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers and validate requests."""
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content=error_response(
                        code="UNSUPPORTED_MEDIA_TYPE",
                        message="Content-Type must be application/json",
                        status_code=415,
                        trace_id=trace_id,
                    ),
                )

        # Validate request size (10MB limit)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:
            trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=error_response(
                    code="REQUEST_TOO_LARGE",
                    message="Request entity too large (max 10MB)",
                    status_code=413,
                    trace_id=trace_id,
                ),
            )

        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        return response
