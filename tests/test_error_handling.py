"""
Comprehensive test suite for error handling and recovery mechanisms.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from maverick_mcp.agents.market_analysis import MarketAnalysisAgent
from maverick_mcp.exceptions import (
    AgentInitializationError,
    APIRateLimitError,
    CircuitBreakerError,
    InsufficientCreditsError,
    PersonaConfigurationError,
    ValidationError,
)
from maverick_mcp.logging_config import CorrelationIDMiddleware, ErrorLogger


# Mock tool input model
class MockToolInput(BaseModel):
    """Input for mock tool."""

    query: str = Field(default="test", description="Test query")


# Create a proper mock tool that LangChain can work with
class MockTool(BaseTool):
    """Mock tool for testing."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    args_schema: type[BaseModel] = MockToolInput

    def _run(self, query: str = "test") -> str:
        """Run the tool."""
        return f"Mock result for: {query}"

    async def _arun(self, query: str = "test") -> str:
        """Run the tool asynchronously."""
        return f"Mock result for: {query}"


# Create a mock tool with configurable set_persona method
class MockPersonaAwareTool(BaseTool):
    """Mock tool that can have a set_persona method."""

    name: str = "mock_persona_tool"
    description: str = "A mock persona-aware tool for testing"
    args_schema: type[BaseModel] = MockToolInput
    _fail_on_set_persona: bool = False  # Private attribute using underscore

    def __init__(self, fail_on_set_persona: bool = False, **kwargs):
        """Initialize with option to fail on set_persona."""
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "_fail_on_set_persona", fail_on_set_persona)

    def set_persona(self, persona: str) -> None:
        """Set the persona for the tool."""
        if self._fail_on_set_persona:
            raise Exception("Tool configuration failed")

    def _run(self, query: str = "test") -> str:
        """Run the tool."""
        return f"Mock result for: {query}"

    async def _arun(self, query: str = "test") -> str:
        """Run the tool asynchronously."""
        return f"Mock result for: {query}"


class TestAgentErrorHandling:
    """Test error handling in agent initialization and operation."""

    @pytest.mark.asyncio
    async def test_invalid_persona_error(self):
        """Test that invalid persona raises PersonaConfigurationError."""
        mock_llm = Mock()

        with pytest.raises(PersonaConfigurationError) as exc_info:
            MarketAnalysisAgent(llm=mock_llm, persona="invalid_persona")

        assert "Invalid persona 'invalid_persona'" in str(exc_info.value)
        assert exc_info.value.context["invalid_persona"] == "invalid_persona"
        assert "conservative" in exc_info.value.context["valid_personas"]

    @pytest.mark.asyncio
    async def test_no_tools_initialization_error(self):
        """Test that agent initialization fails gracefully with no tools."""
        mock_llm = Mock()

        with patch(
            "maverick_mcp.agents.market_analysis.get_tool_registry"
        ) as mock_registry:
            # Mock registry to return no tools
            mock_registry.return_value.get_tool.return_value = None

            # Also need to mock the directly instantiated tools
            with (
                patch(
                    "maverick_mcp.agents.market_analysis.PositionSizeTool",
                    return_value=None,
                ),
                patch(
                    "maverick_mcp.agents.market_analysis.RiskMetricsTool",
                    return_value=None,
                ),
                patch(
                    "maverick_mcp.agents.market_analysis.TechnicalStopsTool",
                    return_value=None,
                ),
                patch(
                    "maverick_mcp.agents.market_analysis.NewsSentimentTool",
                    return_value=None,
                ),
                patch(
                    "maverick_mcp.agents.market_analysis.MarketBreadthTool",
                    return_value=None,
                ),
                patch(
                    "maverick_mcp.agents.market_analysis.SectorSentimentTool",
                    return_value=None,
                ),
            ):
                with pytest.raises(AgentInitializationError) as exc_info:
                    MarketAnalysisAgent(llm=mock_llm, persona="moderate")

                assert "No tools available" in str(exc_info.value)
                assert exc_info.value.context["agent_type"] == "MarketAnalysisAgent"

    @pytest.mark.asyncio
    async def test_tool_registry_failure(self):
        """Test handling of tool registry failures."""
        mock_llm = Mock()

        with patch(
            "maverick_mcp.agents.market_analysis.get_tool_registry"
        ) as mock_registry:
            # Simulate registry failure
            mock_registry.side_effect = Exception("Registry connection failed")

            with pytest.raises(AgentInitializationError) as exc_info:
                MarketAnalysisAgent(llm=mock_llm, persona="moderate")

            assert "Registry connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_initialization_with_retry(self):
        """Test successful initialization after transient failure."""
        mock_llm = Mock()
        attempts = 0

        def mock_get_tool(name):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                return None  # First attempt fails
            return MockTool()  # Second attempt succeeds with proper tool

        with patch(
            "maverick_mcp.agents.market_analysis.get_tool_registry"
        ) as mock_registry:
            mock_registry.return_value.get_tool = mock_get_tool

            # Should succeed on retry
            agent = MarketAnalysisAgent(llm=mock_llm, persona="moderate")
            assert agent is not None


class TestDataProviderErrorHandling:
    """Test error handling in data providers."""

    def test_api_rate_limit_error(self):
        """Test API rate limit error handling."""
        error = APIRateLimitError(provider="yahoo_finance", retry_after=60)

        assert error.recoverable is True
        assert error.context["retry_after"] == 60
        assert "Rate limit exceeded" in str(error)

        # Test error dictionary conversion
        error_dict = error.to_dict()
        assert error_dict["code"] == "RATE_LIMIT_EXCEEDED"
        assert (
            error_dict["message"]
            == "Rate limit exceeded for yahoo_finance. Retry after 60 seconds"
        )
        assert error_dict["context"]["retry_after"] == 60

    def test_data_not_found_error(self):
        """Test data not found error with date range."""
        from maverick_mcp.exceptions import DataNotFoundError

        error = DataNotFoundError(
            symbol="INVALID", date_range=("2024-01-01", "2024-01-31")
        )

        assert "INVALID" in str(error)
        assert "2024-01-01" in str(error)
        assert error.context["symbol"] == "INVALID"


class TestCircuitBreakerIntegration:
    """Test circuit breaker error handling."""

    def test_circuit_breaker_open_error(self):
        """Test circuit breaker open error."""
        error = CircuitBreakerError(
            service="stock_data_api", failure_count=5, threshold=3
        )

        assert error.recoverable is True
        assert error.context["failure_count"] == 5
        assert error.context["threshold"] == 3
        assert "Circuit breaker open" in str(error)


class TestValidationErrors:
    """Test validation error handling."""

    def test_parameter_validation_error(self):
        """Test parameter validation error."""
        from maverick_mcp.exceptions import ParameterValidationError

        error = ParameterValidationError(
            param_name="start_date", expected_type="datetime", actual_type="str"
        )

        assert error.recoverable is True  # Default is True in new implementation
        assert "Expected datetime, got str" in str(error)
        assert (
            error.field == "start_date"
        )  # ParameterValidationError inherits from ValidationError which uses "field"
        assert error.context["expected_type"] == "datetime"
        assert error.context["actual_type"] == "str"

    def test_validation_error_with_details(self):
        """Test validation error with detailed context."""
        error = ValidationError(message="Invalid ticker format", field="ticker")
        error.context["value"] = "ABC123"

        assert error.recoverable is True  # Default is True now
        assert "Invalid ticker format" in str(error)
        assert error.field == "ticker"
        assert error.context["value"] == "ABC123"


class TestErrorLogging:
    """Test structured error logging functionality."""

    def test_error_logger_masking(self):
        """Test that sensitive data is masked in logs."""
        logger = Mock()
        error_logger = ErrorLogger(logger)

        sensitive_context = {
            "api_key": "secret123",
            "user_data": {"email": "user@example.com", "password": "password123"},
            "safe_field": "visible_data",
        }

        error = ValueError("Test error")
        error_logger.log_error(error, sensitive_context)

        # Check that log was called
        assert logger.log.called

        # Get the extra data passed to logger
        call_args = logger.log.call_args
        extra_data = call_args[1]["extra"]

        # Verify sensitive data was masked
        assert extra_data["context"]["api_key"] == "***MASKED***"
        assert extra_data["context"]["user_data"]["password"] == "***MASKED***"
        assert extra_data["context"]["safe_field"] == "visible_data"

    def test_error_counting(self):
        """Test error count tracking."""
        logger = Mock()
        error_logger = ErrorLogger(logger)

        # Log same error type multiple times
        for _i in range(3):
            error_logger.log_error(ValueError("Test"), {})

        # Log different error type
        error_logger.log_error(TypeError("Test"), {})

        stats = error_logger.get_error_stats()
        assert stats["ValueError"] == 3
        assert stats["TypeError"] == 1


class TestCorrelationIDMiddleware:
    """Test correlation ID tracking."""

    def test_correlation_id_generation(self):
        """Test correlation ID generation and retrieval."""
        # Generate new ID
        correlation_id = CorrelationIDMiddleware.set_correlation_id()
        assert correlation_id.startswith("mcp-")
        assert len(correlation_id) == 12  # "mcp-" + 8 hex chars

        # Retrieve same ID
        retrieved_id = CorrelationIDMiddleware.get_correlation_id()
        assert retrieved_id == correlation_id

    def test_correlation_id_persistence(self):
        """Test that correlation ID persists across function calls."""
        correlation_id = CorrelationIDMiddleware.set_correlation_id()

        def inner_function():
            return CorrelationIDMiddleware.get_correlation_id()

        assert inner_function() == correlation_id


class TestAuthenticationErrors:
    """Test authentication and authorization error handling."""

    def test_insufficient_credits_error(self):
        """Test insufficient credits error."""
        error = InsufficientCreditsError(required=50, available=10)

        assert error.recoverable is True  # Default is True in new implementation
        assert error.context["required_credits"] == 50
        assert error.context["available_credits"] == 10
        # InsufficientCreditsError now has its own message format
        assert "Insufficient credits: required 50, available 10" in str(error)


# Integration test for complete error flow
class TestErrorFlowIntegration:
    """Test complete error handling flow from agent to logging."""

    @pytest.mark.asyncio
    async def test_complete_error_flow(self):
        """Test error propagation from tool through agent to logging."""
        mock_llm = Mock()

        with patch(
            "maverick_mcp.agents.market_analysis.get_tool_registry"
        ) as mock_registry:
            # Create a proper mock tool that will fail on set_persona
            mock_tool = MockPersonaAwareTool(fail_on_set_persona=True)
            mock_registry.return_value.get_tool.return_value = mock_tool

            # Agent should still initialize but log warning
            with patch("maverick_mcp.agents.market_analysis.logger") as mock_logger:
                MarketAnalysisAgent(llm=mock_llm, persona="moderate")

                # Verify warning was logged
                assert mock_logger.warning.called
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "Failed to set persona" in warning_msg
