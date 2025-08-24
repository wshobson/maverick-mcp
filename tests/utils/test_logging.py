"""
Unit tests for maverick_mcp.utils.logging module.

This module contains comprehensive tests for the structured logging system
to ensure proper logging functionality and context management.
"""

import asyncio
import json
import logging
import time
from unittest.mock import Mock, patch

import pytest

from maverick_mcp.utils.logging import (
    PerformanceMonitor,
    RequestContextLogger,
    StructuredFormatter,
    _get_query_type,
    _sanitize_params,
    get_logger,
    log_cache_operation,
    log_database_query,
    log_external_api_call,
    log_tool_execution,
    request_id_var,
    request_start_var,
    setup_structured_logging,
    tool_name_var,
    user_id_var,
)


class TestStructuredFormatter:
    """Test the StructuredFormatter class."""

    def test_basic_format(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Parse the JSON output
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

    def test_format_with_context(self):
        """Test formatting with request context."""
        formatter = StructuredFormatter()

        # Set context variables
        request_id_var.set("test-request-123")
        user_id_var.set("user-456")
        tool_name_var.set("test_tool")
        request_start_var.set(time.time() - 0.5)  # 500ms ago

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["request_id"] == "test-request-123"
        assert log_data["user_id"] == "user-456"
        assert log_data["tool_name"] == "test_tool"
        assert "duration_ms" in log_data
        assert log_data["duration_ms"] >= 400  # Should be around 500ms

        # Clean up
        request_id_var.set(None)
        user_id_var.set(None)
        tool_name_var.set(None)
        request_start_var.set(None)

    def test_format_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"
        assert isinstance(log_data["exception"]["traceback"], list)

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.custom_field = "custom_value"
        record.user_action = "button_click"

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["custom_field"] == "custom_value"
        assert log_data["user_action"] == "button_click"


class TestRequestContextLogger:
    """Test the RequestContextLogger class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def context_logger(self, mock_logger):
        """Create a RequestContextLogger with mocked dependencies."""
        with patch("maverick_mcp.utils.logging.psutil.Process") as mock_process:
            mock_process.return_value.memory_info.return_value.rss = (
                100 * 1024 * 1024
            )  # 100MB
            mock_process.return_value.cpu_percent.return_value = 15.5
            return RequestContextLogger(mock_logger)

    def test_info_logging(self, context_logger, mock_logger):
        """Test info level logging."""
        context_logger.info("Test message", extra={"custom": "value"})

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args

        assert call_args[0][0] == logging.INFO
        assert call_args[0][1] == "Test message"
        assert "extra" in call_args[1]
        assert call_args[1]["extra"]["custom"] == "value"
        assert "memory_mb" in call_args[1]["extra"]
        assert "cpu_percent" in call_args[1]["extra"]

    def test_error_logging(self, context_logger, mock_logger):
        """Test error level logging."""
        context_logger.error("Error message")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args

        assert call_args[0][0] == logging.ERROR
        assert call_args[0][1] == "Error message"

    def test_debug_logging(self, context_logger, mock_logger):
        """Test debug level logging."""
        context_logger.debug("Debug message")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args

        assert call_args[0][0] == logging.DEBUG
        assert call_args[0][1] == "Debug message"

    def test_warning_logging(self, context_logger, mock_logger):
        """Test warning level logging."""
        context_logger.warning("Warning message")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args

        assert call_args[0][0] == logging.WARNING
        assert call_args[0][1] == "Warning message"

    def test_critical_logging(self, context_logger, mock_logger):
        """Test critical level logging."""
        context_logger.critical("Critical message")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args

        assert call_args[0][0] == logging.CRITICAL
        assert call_args[0][1] == "Critical message"


class TestLoggingSetup:
    """Test logging setup functions."""

    def test_setup_structured_logging_json_format(self):
        """Test setting up structured logging with JSON format."""
        with patch("maverick_mcp.utils.logging.logging.getLogger") as mock_get_logger:
            mock_root_logger = Mock()
            mock_root_logger.handlers = []  # Empty list of handlers
            mock_get_logger.return_value = mock_root_logger

            setup_structured_logging(log_level="DEBUG", log_format="json")

            mock_root_logger.setLevel.assert_called_with(logging.DEBUG)
            mock_root_logger.addHandler.assert_called()

    def test_setup_structured_logging_text_format(self):
        """Test setting up structured logging with text format."""
        with patch("maverick_mcp.utils.logging.logging.getLogger") as mock_get_logger:
            mock_root_logger = Mock()
            mock_root_logger.handlers = []  # Empty list of handlers
            mock_get_logger.return_value = mock_root_logger

            setup_structured_logging(log_level="INFO", log_format="text")

            mock_root_logger.setLevel.assert_called_with(logging.INFO)

    def test_setup_structured_logging_with_file(self):
        """Test setting up structured logging with file output."""
        with patch("maverick_mcp.utils.logging.logging.getLogger") as mock_get_logger:
            with patch(
                "maverick_mcp.utils.logging.logging.FileHandler"
            ) as mock_file_handler:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []  # Empty list of handlers
                mock_get_logger.return_value = mock_root_logger

                setup_structured_logging(log_file="/tmp/test.log")

                mock_file_handler.assert_called_with("/tmp/test.log")
                assert mock_root_logger.addHandler.call_count == 2  # Console + File

    def test_get_logger(self):
        """Test getting a logger with context support."""
        logger = get_logger("test_module")

        assert isinstance(logger, RequestContextLogger)


class TestToolExecutionLogging:
    """Test the log_tool_execution decorator."""

    @pytest.mark.asyncio
    async def test_successful_tool_execution(self):
        """Test logging for successful tool execution."""

        @log_tool_execution
        async def test_tool(param1, param2="default"):
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "success"}

        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = await test_tool("test_value", param2="custom")

            assert result == {"result": "success"}
            assert mock_logger.info.call_count >= 2  # Start + Success

            # Check that request context was set and cleared
            assert request_id_var.get() is None
            assert tool_name_var.get() is None
            assert request_start_var.get() is None

    @pytest.mark.asyncio
    async def test_failed_tool_execution(self):
        """Test logging for failed tool execution."""

        @log_tool_execution
        async def failing_tool():
            raise ValueError("Test error")

        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with pytest.raises(ValueError, match="Test error"):
                await failing_tool()

            mock_logger.error.assert_called_once()

            # Check that context was cleared even after exception
            assert request_id_var.get() is None
            assert tool_name_var.get() is None
            assert request_start_var.get() is None


class TestParameterSanitization:
    """Test parameter sanitization for logging."""

    def test_sanitize_sensitive_params(self):
        """Test sanitization of sensitive parameters."""
        params = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "key_secret",
            "auth_token": "token_value",
            "normal_param": "normal_value",
        }

        sanitized = _sanitize_params(params)

        assert sanitized["username"] == "testuser"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["auth_token"] == "***REDACTED***"
        assert sanitized["normal_param"] == "normal_value"

    def test_sanitize_nested_params(self):
        """Test sanitization of nested parameters."""
        params = {
            "config": {
                "database_url": "postgresql://user:pass@host/db",
                "secret_key": "secret",
                "debug": True,
            },
            "normal": "value",
        }

        sanitized = _sanitize_params(params)

        assert sanitized["config"]["database_url"] == "postgresql://user:pass@host/db"
        assert sanitized["config"]["secret_key"] == "***REDACTED***"
        assert sanitized["config"]["debug"] is True
        assert sanitized["normal"] == "value"

    def test_sanitize_long_lists(self):
        """Test sanitization of long lists."""
        params = {
            "short_list": [1, 2, 3],
            "long_list": list(range(100)),
        }

        sanitized = _sanitize_params(params)

        assert sanitized["short_list"] == [1, 2, 3]
        assert sanitized["long_list"] == "[100 items]"

    def test_sanitize_long_strings(self):
        """Test sanitization of long strings."""
        long_string = "x" * 2000
        params = {
            "short_string": "hello",
            "long_string": long_string,
        }

        sanitized = _sanitize_params(params)

        assert sanitized["short_string"] == "hello"
        assert "... (2000 chars total)" in sanitized["long_string"]
        assert len(sanitized["long_string"]) < 200


class TestDatabaseQueryLogging:
    """Test database query logging."""

    def test_log_database_query_basic(self):
        """Test basic database query logging."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_database_query("SELECT * FROM users", {"user_id": 123}, 250)

            mock_logger.info.assert_called_once()
            mock_logger.debug.assert_called_once()

    def test_get_query_type(self):
        """Test query type detection."""
        assert _get_query_type("SELECT * FROM users") == "SELECT"
        assert _get_query_type("INSERT INTO users VALUES (1, 'test')") == "INSERT"
        assert _get_query_type("UPDATE users SET name = 'test'") == "UPDATE"
        assert _get_query_type("DELETE FROM users WHERE id = 1") == "DELETE"
        assert _get_query_type("CREATE TABLE test (id INT)") == "CREATE"
        assert _get_query_type("DROP TABLE test") == "DROP"
        assert _get_query_type("EXPLAIN SELECT * FROM users") == "OTHER"

    def test_slow_query_detection(self):
        """Test slow query detection."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_database_query("SELECT * FROM large_table", duration_ms=1500)

            # Check that slow_query flag is set in extra
            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["slow_query"] is True


class TestCacheOperationLogging:
    """Test cache operation logging."""

    def test_log_cache_hit(self):
        """Test logging cache hit."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_cache_operation("get", "stock_data:AAPL", hit=True, duration_ms=5)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "hit" in call_args[0][0]
            assert call_args[1]["extra"]["cache_hit"] is True

    def test_log_cache_miss(self):
        """Test logging cache miss."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_cache_operation("get", "stock_data:MSFT", hit=False)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "miss" in call_args[0][0]
            assert call_args[1]["extra"]["cache_hit"] is False


class TestExternalAPILogging:
    """Test external API call logging."""

    def test_log_successful_api_call(self):
        """Test logging successful API call."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_external_api_call(
                service="yahoo_finance",
                endpoint="/v8/finance/chart/AAPL",
                method="GET",
                status_code=200,
                duration_ms=150,
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["success"] is True

    def test_log_failed_api_call(self):
        """Test logging failed API call."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_external_api_call(
                service="yahoo_finance",
                endpoint="/v8/finance/chart/INVALID",
                method="GET",
                status_code=404,
                duration_ms=1000,
                error="Symbol not found",
            )

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert call_args[1]["extra"]["success"] is False
            assert call_args[1]["extra"]["error"] == "Symbol not found"


class TestPerformanceMonitor:
    """Test the PerformanceMonitor context manager."""

    def test_successful_operation(self):
        """Test monitoring successful operation."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with PerformanceMonitor("test_operation"):
                time.sleep(0.1)  # Simulate work

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "completed" in call_args[0][0]
            assert call_args[1]["extra"]["success"] is True
            assert call_args[1]["extra"]["duration_ms"] >= 100

    def test_failed_operation(self):
        """Test monitoring failed operation."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with pytest.raises(ValueError):
                with PerformanceMonitor("failing_operation"):
                    raise ValueError("Test error")

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "failed" in call_args[0][0]
            assert call_args[1]["extra"]["success"] is False
            assert call_args[1]["extra"]["error_type"] == "ValueError"

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        with patch("maverick_mcp.utils.logging.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with PerformanceMonitor("memory_test"):
                # Simulate memory allocation
                data = list(range(1000))
                del data

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "memory_delta_mb" in call_args[1]["extra"]


if __name__ == "__main__":
    pytest.main([__file__])
