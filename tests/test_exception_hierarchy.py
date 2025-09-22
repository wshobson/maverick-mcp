"""
Test the consolidated exception hierarchy.
"""

from maverick_mcp.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CacheConnectionError,
    CircuitBreakerError,
    ConfigurationError,
    ConflictError,
    CreditError,
    DataIntegrityError,
    DataNotFoundError,
    DataProviderError,
    ExternalServiceError,
    InsufficientCreditsError,
    MaverickException,
    MaverickMCPError,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    SubscriptionError,
    ValidationError,
    WebhookError,
    get_error_message,
)


class TestExceptionHierarchy:
    """Test the consolidated exception hierarchy."""

    def test_base_exception(self):
        """Test base MaverickException."""
        exc = MaverickException("Test error")
        assert exc.message == "Test error"
        assert exc.error_code == "INTERNAL_ERROR"
        assert exc.status_code == 500
        assert exc.field is None
        assert exc.context == {}
        assert exc.recoverable is True

    def test_base_exception_with_params(self):
        """Test base exception with custom parameters."""
        exc = MaverickException(
            "Custom error",
            error_code="CUSTOM_ERROR",
            status_code=400,
            field="test_field",
            context={"key": "value"},
            recoverable=False,
        )
        assert exc.message == "Custom error"
        assert exc.error_code == "CUSTOM_ERROR"
        assert exc.status_code == 400
        assert exc.field == "test_field"
        assert exc.context == {"key": "value"}
        assert exc.recoverable is False

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Invalid email format", field="email")
        assert exc.message == "Invalid email format"
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.status_code == 422
        assert exc.field == "email"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError()
        assert exc.message == "Authentication failed"
        assert exc.error_code == "AUTHENTICATION_ERROR"
        assert exc.status_code == 401

    def test_authorization_error(self):
        """Test AuthorizationError."""
        exc = AuthorizationError(resource="credits", action="deduct")
        assert "Unauthorized access to credits for action 'deduct'" in exc.message
        assert exc.error_code == "AUTHORIZATION_ERROR"
        assert exc.status_code == 403
        assert exc.context["resource"] == "credits"
        assert exc.context["action"] == "deduct"

    def test_insufficient_credits_error(self):
        """Test InsufficientCreditsError."""
        exc = InsufficientCreditsError(required=100, available=50)
        assert "Insufficient credits: required 100, available 50" in exc.message
        assert exc.error_code == "INSUFFICIENT_CREDITS"
        assert exc.status_code == 402
        assert exc.context["required_credits"] == 100
        assert exc.context["available_credits"] == 50

    def test_not_found_error(self):
        """Test NotFoundError."""
        exc = NotFoundError("Stock", identifier="AAPL")
        assert exc.message == "Stock not found: AAPL"
        assert exc.error_code == "NOT_FOUND"
        assert exc.status_code == 404
        assert exc.context["resource"] == "Stock"
        assert exc.context["identifier"] == "AAPL"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(retry_after=60)
        assert exc.message == "Rate limit exceeded"
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert exc.status_code == 429
        assert exc.context["retry_after"] == 60

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        exc = ExternalServiceError(
            "MarketDataAPI", "Service request failed", original_error="Connection timeout"
        )
        assert exc.message == "Service request failed"
        assert exc.error_code == "EXTERNAL_SERVICE_ERROR"
        assert exc.status_code == 503
        assert exc.context["service"] == "MarketDataAPI"
        assert exc.context["original_error"] == "Connection timeout"

    def test_data_provider_error(self):
        """Test DataProviderError."""
        exc = DataProviderError("yfinance", "API request failed")
        assert exc.message == "API request failed"
        assert exc.error_code == "DATA_PROVIDER_ERROR"
        assert exc.status_code == 503
        assert exc.context["provider"] == "yfinance"

    def test_data_not_found_error(self):
        """Test DataNotFoundError."""
        exc = DataNotFoundError("AAPL", date_range=("2024-01-01", "2024-01-31"))
        assert "Data not found for symbol 'AAPL'" in exc.message
        assert "in range 2024-01-01 to 2024-01-31" in exc.message
        assert exc.error_code == "DATA_NOT_FOUND"
        assert exc.status_code == 404
        assert exc.context["symbol"] == "AAPL"
        assert exc.context["date_range"] == ("2024-01-01", "2024-01-31")

    def test_exception_to_dict(self):
        """Test exception to_dict method."""
        exc = ValidationError("Invalid value", field="age")
        exc.context["min_value"] = 0
        exc.context["max_value"] = 120

        result = exc.to_dict()
        assert result == {
            "code": "VALIDATION_ERROR",
            "message": "Invalid value",
            "field": "age",
            "context": {"min_value": 0, "max_value": 120},
        }

    def test_backward_compatibility(self):
        """Test backward compatibility alias."""
        assert MaverickMCPError is MaverickException

        # Old code should still work
        exc = MaverickMCPError("Legacy error")
        assert isinstance(exc, MaverickException)
        assert exc.message == "Legacy error"

    def test_get_error_message(self):
        """Test error message lookup."""
        assert get_error_message("VALIDATION_ERROR") == "Request validation failed"
        assert get_error_message("NOT_FOUND") == "Resource not found"
        assert get_error_message("UNKNOWN_CODE") == "Unknown error"

    def test_inheritance_chain(self):
        """Test that all custom exceptions inherit from MaverickException."""
        exceptions = [
            ValidationError("test"),
            AuthenticationError(),
            AuthorizationError(),
            InsufficientCreditsError(10, 5),
            NotFoundError("test"),
            ConflictError("test"),
            RateLimitError(),
            PaymentRequiredError(),
            ExternalServiceError("test", "test"),
            DataProviderError("test", "test"),
            DataNotFoundError("test"),
            DataIntegrityError("test"),
            CacheConnectionError("test", "test"),
            ConfigurationError("test"),
            SubscriptionError("test"),
            WebhookError("test"),
            CreditError("test"),
            CircuitBreakerError("test", 5, 10),
        ]

        for exc in exceptions:
            assert isinstance(exc, MaverickException)
            assert hasattr(exc, "error_code")
            assert hasattr(exc, "status_code")
            assert hasattr(exc, "message")
            assert hasattr(exc, "context")
            assert hasattr(exc, "to_dict")
