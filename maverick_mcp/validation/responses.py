"""
Base response models for API standardization.

This module provides standard response formats for all API endpoints
to ensure consistency across the application.
"""

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model with standard fields."""

    success: bool = Field(..., description="Whether the request was successful")
    message: str | None = Field(None, description="Human-readable message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp",
    )
    request_id: str | None = Field(None, description="Request tracking ID")


class DataResponse(BaseResponse, Generic[T]):
    """Response model with data payload."""

    data: T = Field(..., description="Response data")


class ListResponse(BaseResponse, Generic[T]):
    """Response model for paginated lists."""

    data: list[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether more items are available")


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(..., description="Error code")
    field: str | None = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    context: dict[str, Any] | None = Field(None, description="Additional context")


class ErrorResponse(BaseResponse):
    """Standard error response model."""

    success: bool = Field(default=False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")
    status_code: int = Field(..., description="HTTP status code")
    trace_id: str | None = Field(None, description="Error trace ID for debugging")


class ValidationErrorResponse(ErrorResponse):
    """Response for validation errors."""

    errors: list[ErrorDetail] = Field(..., description="List of validation errors")


class BatchOperationResult(BaseModel):
    """Result of a batch operation on a single item."""

    id: str = Field(..., description="Item identifier")
    success: bool = Field(..., description="Whether the operation succeeded")
    error: ErrorDetail | None = Field(None, description="Error if operation failed")
    data: dict[str, Any] | None = Field(None, description="Operation result data")


class BatchResponse(BaseResponse):
    """Response for batch operations."""

    results: list[BatchOperationResult] = Field(
        ..., description="Results for each item"
    )
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    partial: bool = Field(..., description="Whether some operations failed")


class HealthStatus(BaseModel):
    """Health check status for a component."""

    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Status (healthy, unhealthy, degraded)")
    latency_ms: float | None = Field(None, description="Response time in milliseconds")
    details: dict[str, Any] | None = Field(None, description="Additional details")


class HealthResponse(BaseResponse):
    """Health check response."""

    status: str = Field(..., description="Overall status")
    components: list[HealthStatus] = Field(
        ..., description="Status of individual components"
    )
    version: str | None = Field(None, description="Application version")
    uptime_seconds: int | None = Field(None, description="Uptime in seconds")


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int = Field(..., description="Request limit")
    remaining: int = Field(..., description="Remaining requests")
    reset: datetime = Field(..., description="When the limit resets")
    retry_after: int | None = Field(None, description="Seconds to wait before retrying")


class RateLimitResponse(ErrorResponse):
    """Response when rate limit is exceeded."""

    rate_limit: RateLimitInfo = Field(..., description="Rate limit details")


class WebhookEvent(BaseModel):
    """Webhook event payload."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="When the event occurred")
    data: dict[str, Any] = Field(..., description="Event data")
    signature: str | None = Field(None, description="Event signature for verification")


class WebhookResponse(BaseResponse):
    """Response for webhook endpoints."""

    event_id: str = Field(..., description="Processed event ID")
    status: str = Field(..., description="Processing status")


# Helper functions for creating responses
def success_response(
    data: Any = None,
    message: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Create a successful response."""
    response = {"success": True, "timestamp": datetime.now(UTC).isoformat()}

    if message:
        response["message"] = message
    if request_id:
        response["request_id"] = request_id
    if data is not None:
        response["data"] = data

    return response


def error_response(
    code: str,
    message: str,
    status_code: int,
    field: str | None = None,
    context: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Create an error response."""
    return {
        "success": False,
        "timestamp": datetime.now(UTC).isoformat(),
        "error": {
            "code": code,
            "message": message,
            "field": field,
            "context": context,
        },
        "status_code": status_code,
        "trace_id": trace_id,
    }


def validation_error_response(
    errors: list[dict[str, Any]], trace_id: str | None = None
) -> dict[str, Any]:
    """Create a validation error response."""
    return {
        "success": False,
        "timestamp": datetime.now(UTC).isoformat(),
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Validation failed",
        },
        "errors": errors,
        "status_code": 422,
        "trace_id": trace_id,
    }
