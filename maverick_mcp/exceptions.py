"""
Custom exception classes for MaverickMCP with comprehensive error handling.

This module provides a unified exception hierarchy with proper error codes,
HTTP status codes, and standardized error responses.
"""

from typing import Any


class MaverickException(Exception):
    """Base exception for all Maverick errors."""

    # Default values can be overridden by subclasses
    error_code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        status_code: int | None = None,
        field: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.error_code
        self.status_code = status_code or self.__class__.status_code
        self.field = field
        self.context = context or {}
        self.recoverable = recoverable

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result: dict[str, Any] = {
            "code": self.error_code,
            "message": self.message,
        }
        if self.field:
            result["field"] = self.field
        if self.context:
            result["context"] = self.context
        return result

    def __repr__(self) -> str:
        """String representation of the exception."""
        return f"{self.__class__.__name__}('{self.message}', code='{self.error_code}')"


# Validation exceptions
class ValidationError(MaverickException):
    """Raised when input validation fails."""


# Research and agent exceptions
class ResearchError(MaverickException):
    """Raised when research operations fail."""

    error_code = "RESEARCH_ERROR"
    status_code = 500

    def __init__(
        self,
        message: str,
        research_type: str | None = None,
        provider: str | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
        field: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            field=field,
            context=context,
            recoverable=recoverable,
        )
        self.research_type = research_type
        self.provider = provider

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = super().to_dict()
        if self.research_type:
            result["research_type"] = self.research_type
        if self.provider:
            result["provider"] = self.provider
        return result


class WebSearchError(ResearchError):
    """Raised when web search operations fail."""

    error_code = "WEB_SEARCH_ERROR"


class ContentAnalysisError(ResearchError):
    """Raised when content analysis fails."""

    error_code = "CONTENT_ANALYSIS_ERROR"


class AgentExecutionError(MaverickException):
    """Raised when agent execution fails."""

    error_code = "AGENT_EXECUTION_ERROR"
    status_code = 500


# Authentication/Authorization exceptions
class AuthenticationError(MaverickException):
    """Raised when authentication fails."""

    error_code = "AUTHENTICATION_ERROR"
    status_code = 401

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, **kwargs)


class AuthorizationError(MaverickException):
    """Raised when authorization fails."""

    error_code = "AUTHORIZATION_ERROR"
    status_code = 403

    def __init__(
        self,
        message: str = "Insufficient permissions",
        resource: str | None = None,
        action: str | None = None,
        **kwargs,
    ):
        if resource and action:
            message = f"Unauthorized access to {resource} for action '{action}'"
        super().__init__(message, **kwargs)
        if resource:
            self.context["resource"] = resource
        if action:
            self.context["action"] = action


class InsufficientCreditsError(MaverickException):
    """Raised when user has insufficient credits."""

    error_code = "INSUFFICIENT_CREDITS"
    status_code = 402

    def __init__(
        self, required: int, available: int, message: str | None = None, **kwargs
    ):
        if not message:
            message = (
                f"Insufficient credits: required {required}, available {available}"
            )
        super().__init__(message, **kwargs)
        self.context["required_credits"] = required
        self.context["available_credits"] = available


# Resource exceptions
class NotFoundError(MaverickException):
    """Raised when a requested resource is not found."""

    error_code = "NOT_FOUND"
    status_code = 404

    def __init__(self, resource: str, identifier: str | None = None, **kwargs):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(message, **kwargs)
        self.context["resource"] = resource
        if identifier:
            self.context["identifier"] = identifier


class ConflictError(MaverickException):
    """Raised when there's a conflict with existing data."""

    error_code = "CONFLICT"
    status_code = 409

    def __init__(self, message: str, field: str | None = None, **kwargs):
        super().__init__(message, field=field, **kwargs)


# Rate limiting exceptions
class RateLimitError(MaverickException):
    """Raised when rate limit is exceeded."""

    error_code = "RATE_LIMIT_EXCEEDED"
    status_code = 429

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if retry_after:
            self.context["retry_after"] = retry_after


# Payment exceptions
class PaymentRequiredError(MaverickException):
    """Raised when payment is required for an action."""

    error_code = "PAYMENT_REQUIRED"
    status_code = 402

    def __init__(
        self,
        message: str = "Payment required",
        required_credits: int | None = None,
        current_balance: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if required_credits is not None:
            self.context["required_credits"] = required_credits
        if current_balance is not None:
            self.context["current_balance"] = current_balance


# External service exceptions
class ExternalServiceError(MaverickException):
    """Raised when an external service fails."""

    error_code = "EXTERNAL_SERVICE_ERROR"
    status_code = 503

    def __init__(
        self, service: str, message: str, original_error: str | None = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.context["service"] = service
        if original_error:
            self.context["original_error"] = original_error


# Data provider exceptions
class DataProviderError(MaverickException):
    """Base exception for data provider errors."""

    error_code = "DATA_PROVIDER_ERROR"
    status_code = 503

    def __init__(self, provider: str, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context["provider"] = provider


class DataNotFoundError(DataProviderError):
    """Raised when requested data is not found."""

    error_code = "DATA_NOT_FOUND"
    status_code = 404

    def __init__(self, symbol: str, date_range: tuple | None = None, **kwargs):
        message = f"Data not found for symbol '{symbol}'"
        if date_range:
            message += f" in range {date_range[0]} to {date_range[1]}"
        super().__init__("cache", message, **kwargs)
        self.context["symbol"] = symbol
        if date_range:
            self.context["date_range"] = date_range


class APIRateLimitError(DataProviderError):
    """Raised when API rate limit is exceeded."""

    error_code = "RATE_LIMIT_EXCEEDED"
    status_code = 429

    def __init__(self, provider: str, retry_after: int | None = None, **kwargs):
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(provider, message, recoverable=True, **kwargs)
        if retry_after:
            self.context["retry_after"] = retry_after


class APIConnectionError(DataProviderError):
    """Raised when API connection fails."""

    error_code = "API_CONNECTION_ERROR"
    status_code = 503

    def __init__(self, provider: str, endpoint: str, reason: str, **kwargs):
        message = f"Failed to connect to {provider} at {endpoint}: {reason}"
        super().__init__(provider, message, recoverable=True, **kwargs)
        self.context["endpoint"] = endpoint
        self.context["connection_reason"] = reason


# Database exceptions
class DatabaseError(MaverickException):
    """Base exception for database errors."""

    error_code = "DATABASE_ERROR"
    status_code = 500

    def __init__(self, operation: str, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context["operation"] = operation


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    error_code = "DATABASE_CONNECTION_ERROR"
    status_code = 503

    def __init__(self, reason: str, **kwargs):
        message = f"Database connection failed: {reason}"
        super().__init__("connect", message, recoverable=True, **kwargs)


class DataIntegrityError(DatabaseError):
    """Raised when data integrity check fails."""

    error_code = "DATA_INTEGRITY_ERROR"
    status_code = 422

    def __init__(
        self,
        message: str,
        table: str | None = None,
        constraint: str | None = None,
        **kwargs,
    ):
        super().__init__("integrity_check", message, recoverable=False, **kwargs)
        if table:
            self.context["table"] = table
        if constraint:
            self.context["constraint"] = constraint


# Cache exceptions
class CacheError(MaverickException):
    """Base exception for cache errors."""

    error_code = "CACHE_ERROR"
    status_code = 503

    def __init__(self, operation: str, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context["operation"] = operation


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""

    error_code = "CACHE_CONNECTION_ERROR"
    status_code = 503

    def __init__(self, cache_type: str, reason: str, **kwargs):
        message = f"{cache_type} cache connection failed: {reason}"
        super().__init__("connect", message, recoverable=True, **kwargs)
        self.context["cache_type"] = cache_type


# Configuration exceptions
class ConfigurationError(MaverickException):
    """Raised when there's a configuration problem."""

    error_code = "CONFIGURATION_ERROR"
    status_code = 500

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        if config_key:
            self.context["config_key"] = config_key


# Subscription exceptions
class SubscriptionError(MaverickException):
    """Raised when there's a subscription-related issue."""

    error_code = "SUBSCRIPTION_ERROR"
    status_code = 403

    def __init__(
        self,
        message: str,
        required_plan: str | None = None,
        current_plan: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if required_plan:
            self.context["required_plan"] = required_plan
        if current_plan:
            self.context["current_plan"] = current_plan


# Webhook exceptions
class WebhookError(MaverickException):
    """Raised when webhook processing fails."""

    error_code = "WEBHOOK_ERROR"
    status_code = 400

    def __init__(
        self,
        message: str,
        event_type: str | None = None,
        event_id: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if event_type:
            self.context["event_type"] = event_type
        if event_id:
            self.context["event_id"] = event_id


# Credit exceptions
class CreditError(MaverickException):
    """Raised when there's a credit-related issue."""

    error_code = "CREDIT_ERROR"
    status_code = 402

    def __init__(
        self,
        message: str,
        balance: int | None = None,
        required: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if balance is not None:
            self.context["balance"] = balance
        if required is not None:
            self.context["required"] = required


# Agent-specific exceptions
class AgentInitializationError(MaverickException):
    """Raised when agent initialization fails."""

    error_code = "AGENT_INIT_ERROR"
    status_code = 500

    def __init__(self, agent_type: str, reason: str, **kwargs):
        message = f"Failed to initialize {agent_type}: {reason}"
        super().__init__(message, **kwargs)
        self.context["agent_type"] = agent_type
        self.context["reason"] = reason


class PersonaConfigurationError(MaverickException):
    """Raised when persona configuration is invalid."""

    error_code = "PERSONA_CONFIG_ERROR"
    status_code = 400

    def __init__(self, persona: str, valid_personas: list, **kwargs):
        message = (
            f"Invalid persona '{persona}'. Valid options: {', '.join(valid_personas)}"
        )
        super().__init__(message, **kwargs)
        self.context["invalid_persona"] = persona
        self.context["valid_personas"] = valid_personas


class ToolRegistrationError(MaverickException):
    """Raised when tool registration fails."""

    error_code = "TOOL_REGISTRATION_ERROR"
    status_code = 500

    def __init__(self, tool_name: str, reason: str, **kwargs):
        message = f"Failed to register tool '{tool_name}': {reason}"
        super().__init__(message, **kwargs)
        self.context["tool_name"] = tool_name
        self.context["reason"] = reason


# Circuit breaker exceptions
class CircuitBreakerError(MaverickException):
    """Raised when circuit breaker is open."""

    error_code = "CIRCUIT_BREAKER_OPEN"
    status_code = 503

    def __init__(self, service: str, failure_count: int, threshold: int, **kwargs):
        message = (
            f"Circuit breaker open for {service}: {failure_count}/{threshold} failures"
        )
        super().__init__(message, recoverable=True, **kwargs)
        self.context["service"] = service
        self.context["failure_count"] = failure_count
        self.context["threshold"] = threshold


# Parameter validation exceptions
class ParameterValidationError(ValidationError):
    """Raised when function parameters are invalid."""

    error_code = "PARAMETER_VALIDATION_ERROR"
    status_code = 400

    def __init__(self, param_name: str, expected_type: str, actual_type: str, **kwargs):
        reason = f"Expected {expected_type}, got {actual_type}"
        message = f"Validation failed for '{param_name}': {reason}"
        super().__init__(message, field=param_name, **kwargs)
        self.context["expected_type"] = expected_type
        self.context["actual_type"] = actual_type


# Error code constants
ERROR_CODES = {
    "VALIDATION_ERROR": "Request validation failed",
    "AUTHENTICATION_ERROR": "Authentication failed",
    "AUTHORIZATION_ERROR": "Insufficient permissions",
    "NOT_FOUND": "Resource not found",
    "CONFLICT": "Resource conflict",
    "RATE_LIMIT_EXCEEDED": "Too many requests",
    "PAYMENT_REQUIRED": "Payment required",
    "INSUFFICIENT_CREDITS": "Insufficient credits",
    "EXTERNAL_SERVICE_ERROR": "External service unavailable",
    "DATA_PROVIDER_ERROR": "Data provider error",
    "DATA_NOT_FOUND": "Data not found",
    "API_CONNECTION_ERROR": "API connection failed",
    "DATABASE_ERROR": "Database error",
    "DATABASE_CONNECTION_ERROR": "Database connection failed",
    "DATA_INTEGRITY_ERROR": "Data integrity violation",
    "CACHE_ERROR": "Cache error",
    "CACHE_CONNECTION_ERROR": "Cache connection failed",
    "CONFIGURATION_ERROR": "Configuration error",
    "SUBSCRIPTION_ERROR": "Subscription issue",
    "WEBHOOK_ERROR": "Webhook processing failed",
    "CREDIT_ERROR": "Credit balance issue",
    "AGENT_INIT_ERROR": "Agent initialization failed",
    "PERSONA_CONFIG_ERROR": "Invalid persona configuration",
    "TOOL_REGISTRATION_ERROR": "Tool registration failed",
    "CIRCUIT_BREAKER_OPEN": "Service unavailable - circuit breaker open",
    "PARAMETER_VALIDATION_ERROR": "Invalid parameter",
    "INTERNAL_ERROR": "Internal server error",
}


def get_error_message(code: str) -> str:
    """Get human-readable message for error code."""
    return ERROR_CODES.get(code, "Unknown error")


# Backward compatibility alias
MaverickMCPError = MaverickException
