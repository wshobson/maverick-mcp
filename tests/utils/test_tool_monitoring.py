"""Tests for tool_monitoring utilities."""

from unittest.mock import Mock

import pytest

from maverick_mcp.utils.credit_estimation import EstimationResult
from maverick_mcp.utils.tool_monitoring import (
    ExecutionResult,
    ToolMonitor,
    create_tool_monitor,
    should_alert_for_performance,
)


class TestExecutionResult:
    """Test ExecutionResult class."""

    def test_execution_result_success(self):
        """Test creating a successful ExecutionResult."""
        result = ExecutionResult(
            result={"data": "test"},
            execution_time=1.5,
            success=True,
        )

        assert result.result == {"data": "test"}
        assert result.execution_time == 1.5
        assert result.success is True
        assert result.error is None

    def test_execution_result_failure(self):
        """Test creating a failed ExecutionResult."""
        error = ValueError("Test error")
        result = ExecutionResult(
            result=None,
            execution_time=0.5,
            success=False,
            error=error,
        )

        assert result.result is None
        assert result.execution_time == 0.5
        assert result.success is False
        assert result.error == error


class TestToolMonitor:
    """Test ToolMonitor class."""

    def test_tool_monitor_creation(self):
        """Test creating a ToolMonitor."""
        monitor = ToolMonitor("test_tool", 123)

        assert monitor.tool_name == "test_tool"
        assert monitor.user_id == 123

    def test_tool_monitor_no_user(self):
        """Test creating a ToolMonitor without user."""
        monitor = ToolMonitor("test_tool")

        assert monitor.tool_name == "test_tool"
        assert monitor.user_id is None

    async def test_execute_with_monitoring_success(self):
        """Test successful tool execution with monitoring."""
        monitor = ToolMonitor("test_tool", 123)

        # Mock function
        async def mock_func(arg1, kwarg1=None):
            return {"result": f"{arg1}-{kwarg1}"}

        # Create estimation for testing
        estimation = EstimationResult(
            llm_calls=2,
            total_tokens=500,
            estimate=Mock(),
            confidence=0.8,
        )

        result = await monitor.execute_with_monitoring(
            mock_func,
            ("test_arg",),
            {"kwarg1": "test_kwarg"},
            estimation,
        )

        assert result.success is True
        assert result.result == {"result": "test_arg-test_kwarg"}
        assert result.execution_time > 0
        assert result.error is None

    async def test_execute_with_monitoring_failure(self):
        """Test failed tool execution with monitoring."""
        monitor = ToolMonitor("test_tool", 123)

        # Mock function that raises error
        async def mock_func():
            raise ValueError("Test error")

        result = await monitor.execute_with_monitoring(mock_func, (), {})

        assert result.success is False
        assert result.result is None
        assert result.execution_time > 0
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Test error"

    async def test_execute_with_monitoring_long_execution(self):
        """Test monitoring of long execution time."""
        monitor = ToolMonitor("test_tool", 123)

        # Mock function with long execution
        async def mock_func():
            import asyncio

            await asyncio.sleep(0.1)  # Simulate work
            return {"result": "slow"}

        # Create estimation to test underestimation detection
        mock_estimate = Mock()
        mock_estimate.complexity.value = "simple"
        estimation = EstimationResult(
            llm_calls=1,
            total_tokens=100,
            estimate=mock_estimate,
            confidence=0.9,
        )

        # Mock time.time to simulate long execution
        with pytest.MonkeyPatch().context() as m:
            call_count = 0

            def mock_time():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return 1000.0  # Start time
                else:
                    return 1035.0  # End time (35 seconds later)

            m.setattr("time.time", mock_time)
            result = await monitor.execute_with_monitoring(
                mock_func, (), {}, estimation
            )

        assert result.success is True
        assert result.execution_time == 35.0

    def test_add_usage_info_to_result_dict(self):
        """Test adding usage info to dict result."""
        monitor = ToolMonitor("test_tool")

        result = {"data": "test"}
        usage_info = {"credits_charged": 5, "remaining_balance": 95}

        updated_result = monitor.add_usage_info_to_result(result, usage_info)

        assert updated_result == {
            "data": "test",
            "usage": {"credits_charged": 5, "remaining_balance": 95},
        }

    def test_add_usage_info_to_result_non_dict(self):
        """Test adding usage info to non-dict result."""
        monitor = ToolMonitor("test_tool")

        result = "string result"
        usage_info = {"credits_charged": 5}

        updated_result = monitor.add_usage_info_to_result(result, usage_info)

        # Should return original result unchanged
        assert updated_result == "string result"

    def test_add_usage_info_to_result_none_usage(self):
        """Test adding None usage info."""
        monitor = ToolMonitor("test_tool")

        result = {"data": "test"}

        updated_result = monitor.add_usage_info_to_result(result, None)

        # Should return original result unchanged
        assert updated_result == {"data": "test"}


class TestCreateToolMonitor:
    """Test create_tool_monitor function."""

    def test_create_tool_monitor_with_user(self):
        """Test creating monitor with user ID."""
        monitor = create_tool_monitor("test_tool", 123)

        assert isinstance(monitor, ToolMonitor)
        assert monitor.tool_name == "test_tool"
        assert monitor.user_id == 123

    def test_create_tool_monitor_without_user(self):
        """Test creating monitor without user ID."""
        monitor = create_tool_monitor("test_tool")

        assert isinstance(monitor, ToolMonitor)
        assert monitor.tool_name == "test_tool"
        assert monitor.user_id is None


class TestShouldAlertForPerformance:
    """Test should_alert_for_performance function."""

    def test_should_alert_slow_execution(self):
        """Test alert for slow execution."""
        should_alert, message = should_alert_for_performance(45.0, 30.0)

        assert should_alert is True
        assert "45.00s" in message
        assert "threshold: 30.0s" in message

    def test_should_not_alert_fast_execution(self):
        """Test no alert for fast execution."""
        should_alert, message = should_alert_for_performance(15.0, 30.0)

        assert should_alert is False
        assert message == ""

    def test_should_alert_default_threshold(self):
        """Test alert with default threshold."""
        should_alert, message = should_alert_for_performance(35.0)

        assert should_alert is True
        assert "35.00s" in message
        assert "threshold: 30.0s" in message
