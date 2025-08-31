"""
Comprehensive test suite for OrchestrationLogger functionality.

This test suite covers:
- OrchestrationLogger initialization and configuration
- Context tracking and performance metrics
- Visual indicators and structured output formatting
- Parallel execution logging
- Method call decorators and context managers
- Resource usage monitoring
- Error handling and fallback logging
"""

import asyncio
import logging
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from maverick_mcp.utils.orchestration_logging import (
    LogColors,
    OrchestrationLogger,
    get_orchestration_logger,
    log_agent_execution,
    log_fallback_trigger,
    log_method_call,
    log_parallel_execution,
    log_performance_metrics,
    log_resource_usage,
    log_synthesis_operation,
    log_tool_invocation,
)


class TestLogColors:
    """Test LogColors utility class."""

    def test_color_constants(self):
        """Test that color constants are defined."""
        assert hasattr(LogColors, 'HEADER')
        assert hasattr(LogColors, 'OKBLUE')
        assert hasattr(LogColors, 'OKCYAN')
        assert hasattr(LogColors, 'OKGREEN')
        assert hasattr(LogColors, 'WARNING')
        assert hasattr(LogColors, 'FAIL')
        assert hasattr(LogColors, 'ENDC')
        assert hasattr(LogColors, 'BOLD')
        assert hasattr(LogColors, 'UNDERLINE')
        
        # Verify they contain ANSI escape sequences
        assert LogColors.HEADER.startswith('\033[')
        assert LogColors.ENDC == '\033[0m'


class TestOrchestrationLogger:
    """Test OrchestrationLogger main functionality."""

    def test_logger_initialization(self):
        """Test OrchestrationLogger initialization."""
        logger = OrchestrationLogger("TestComponent")
        
        assert logger.component_name == "TestComponent"
        assert logger.request_id is None
        assert logger.session_context == {}
        assert isinstance(logger.logger, logging.Logger)
        assert logger.logger.name == "maverick_mcp.orchestration.TestComponent"

    def test_set_request_context(self):
        """Test setting request context."""
        logger = OrchestrationLogger("TestComponent")
        
        # Test with explicit request_id
        logger.set_request_context(
            request_id="req_123",
            session_id="session_456",
            custom_param="value"
        )
        
        assert logger.request_id == "req_123"
        assert logger.session_context["session_id"] == "session_456"
        assert logger.session_context["request_id"] == "req_123"
        assert logger.session_context["custom_param"] == "value"

    def test_set_request_context_auto_id(self):
        """Test auto-generation of request ID."""
        logger = OrchestrationLogger("TestComponent")
        
        logger.set_request_context(session_id="session_789")
        
        assert logger.request_id is not None
        assert len(logger.request_id) == 8  # UUID truncated to 8 chars
        assert logger.session_context["session_id"] == "session_789"
        assert logger.session_context["request_id"] == logger.request_id

    def test_format_message_with_context(self):
        """Test message formatting with context."""
        logger = OrchestrationLogger("TestComponent")
        logger.set_request_context(request_id="req_123", session_id="session_456")
        
        formatted = logger._format_message("INFO", "Test message", param1="value1", param2=42)
        
        assert "TestComponent" in formatted
        assert "req:req_123" in formatted
        assert "session:session_456" in formatted
        assert "Test message" in formatted
        assert "param1:value1" in formatted
        assert "param2:42" in formatted

    def test_format_message_without_context(self):
        """Test message formatting without context."""
        logger = OrchestrationLogger("TestComponent")
        
        formatted = logger._format_message("WARNING", "Warning message", error="test")
        
        assert "TestComponent" in formatted
        assert "Warning message" in formatted
        assert "error:test" in formatted
        # Should not contain context brackets when no context
        assert "req:" not in formatted
        assert "session:" not in formatted

    def test_format_message_color_coding(self):
        """Test color coding in message formatting."""
        logger = OrchestrationLogger("TestComponent")
        
        debug_msg = logger._format_message("DEBUG", "Debug message")
        info_msg = logger._format_message("INFO", "Info message")
        warning_msg = logger._format_message("WARNING", "Warning message")
        error_msg = logger._format_message("ERROR", "Error message")
        
        assert LogColors.OKCYAN in debug_msg
        assert LogColors.OKGREEN in info_msg
        assert LogColors.WARNING in warning_msg
        assert LogColors.FAIL in error_msg
        
        # All should end with reset color
        assert LogColors.ENDC in debug_msg
        assert LogColors.ENDC in info_msg
        assert LogColors.ENDC in warning_msg
        assert LogColors.ENDC in error_msg

    def test_logging_methods(self):
        """Test all logging level methods."""
        logger = OrchestrationLogger("TestComponent")
        
        with patch.object(logger.logger, 'debug') as mock_debug, \
             patch.object(logger.logger, 'info') as mock_info, \
             patch.object(logger.logger, 'warning') as mock_warning, \
             patch.object(logger.logger, 'error') as mock_error:
            
            logger.debug("Debug message", param="debug")
            logger.info("Info message", param="info") 
            logger.warning("Warning message", param="warning")
            logger.error("Error message", param="error")
            
            mock_debug.assert_called_once()
            mock_info.assert_called_once()
            mock_warning.assert_called_once()
            mock_error.assert_called_once()

    def test_none_value_filtering(self):
        """Test filtering of None values in message formatting."""
        logger = OrchestrationLogger("TestComponent")
        
        formatted = logger._format_message(
            "INFO", 
            "Test message", 
            param1="value1",
            param2=None,  # Should be filtered out
            param3="value3"
        )
        
        assert "param1:value1" in formatted
        assert "param2:None" not in formatted
        assert "param3:value3" in formatted


class TestGlobalLoggerRegistry:
    """Test global logger registry functionality."""

    def test_get_orchestration_logger_creation(self):
        """Test creation of new orchestration logger."""
        logger = get_orchestration_logger("NewComponent")
        
        assert isinstance(logger, OrchestrationLogger)
        assert logger.component_name == "NewComponent"

    def test_get_orchestration_logger_reuse(self):
        """Test reuse of existing orchestration logger."""
        logger1 = get_orchestration_logger("ReuseComponent")
        logger2 = get_orchestration_logger("ReuseComponent")
        
        assert logger1 is logger2  # Should be the same instance

    def test_multiple_component_loggers(self):
        """Test multiple independent component loggers."""
        logger_a = get_orchestration_logger("ComponentA")
        logger_b = get_orchestration_logger("ComponentB")
        
        assert logger_a is not logger_b
        assert logger_a.component_name == "ComponentA"
        assert logger_b.component_name == "ComponentB"


class TestLogMethodCallDecorator:
    """Test log_method_call decorator functionality."""

    @pytest.fixture
    def sample_class(self):
        """Create sample class for decorator testing."""
        class SampleClass:
            def __init__(self):
                self.name = "SampleClass"
            
            @log_method_call(component="TestComponent")
            async def async_method(self, param1: str, param2: int = 10):
                await asyncio.sleep(0.01)
                return f"result_{param1}_{param2}"
            
            @log_method_call(component="TestComponent", include_params=False)
            async def async_method_no_params(self):
                return "no_params_result"
            
            @log_method_call(component="TestComponent", include_timing=False)
            async def async_method_no_timing(self):
                return "no_timing_result"
            
            @log_method_call()
            def sync_method(self, value: str):
                return f"sync_{value}"
        
        return SampleClass

    @pytest.mark.asyncio
    async def test_async_method_decoration_success(self, sample_class):
        """Test successful async method decoration."""
        instance = sample_class()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = await instance.async_method("test", param2=20)
            
            assert result == "result_test_20"
            
            # Verify logging calls
            assert mock_logger.info.call_count == 2  # Start and success
            start_call = mock_logger.info.call_args_list[0][0][0]
            success_call = mock_logger.info.call_args_list[1][0][0]
            
            assert "🚀 START async_method" in start_call
            assert "params:" in start_call
            assert "✅ SUCCESS async_method" in success_call
            assert "duration:" in success_call

    @pytest.mark.asyncio
    async def test_async_method_decoration_no_params(self, sample_class):
        """Test async method decoration without parameter logging."""
        instance = sample_class()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            await instance.async_method_no_params()
            
            start_call = mock_logger.info.call_args_list[0][0][0]
            assert "params:" not in start_call

    @pytest.mark.asyncio
    async def test_async_method_decoration_no_timing(self, sample_class):
        """Test async method decoration without timing."""
        instance = sample_class()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            await instance.async_method_no_timing()
            
            success_call = mock_logger.info.call_args_list[1][0][0]
            assert "duration:" not in success_call

    @pytest.mark.asyncio
    async def test_async_method_decoration_error(self, sample_class):
        """Test async method decoration with error handling."""
        class ErrorClass:
            @log_method_call(component="ErrorComponent")
            async def failing_method(self):
                raise ValueError("Test error")
        
        instance = ErrorClass()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError, match="Test error"):
                await instance.failing_method()
            
            # Should log error
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args[0][0]
            assert "❌ ERROR failing_method" in error_call
            assert "error: Test error" in error_call

    def test_sync_method_decoration(self, sample_class):
        """Test synchronous method decoration."""
        instance = sample_class()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = instance.sync_method("test_value")
            
            assert result == "sync_test_value"
            
            # Should log start and success
            assert mock_logger.info.call_count == 2
            assert "🚀 START sync_method" in mock_logger.info.call_args_list[0][0][0]
            assert "✅ SUCCESS sync_method" in mock_logger.info.call_args_list[1][0][0]

    def test_component_name_inference(self):
        """Test automatic component name inference."""
        class InferenceTest:
            @log_method_call()
            def test_method(self):
                return "test"
        
        instance = InferenceTest()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            instance.test_method()
            
            # Should infer component name from class
            mock_get_logger.assert_called_with("InferenceTest")

    def test_result_summary_extraction(self, sample_class):
        """Test extraction of result summaries for logging."""
        class ResultClass:
            @log_method_call(component="ResultComponent")
            async def method_with_result_info(self):
                return {
                    "execution_mode": "parallel",
                    "research_confidence": 0.85,
                    "parallel_execution_stats": {
                        "successful_tasks": 3,
                        "total_tasks": 4
                    }
                }
        
        instance = ResultClass()
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = asyncio.run(instance.method_with_result_info())
            
            success_call = mock_logger.info.call_args_list[1][0][0]
            assert "mode: parallel" in success_call
            assert "confidence: 0.85" in success_call
            assert "tasks: 3/4" in success_call


class TestContextManagers:
    """Test context manager utilities."""

    def test_log_parallel_execution_success(self):
        """Test log_parallel_execution context manager success case."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with log_parallel_execution("TestComponent", "test operation", 3) as logger:
                assert logger == mock_logger
                time.sleep(0.01)  # Simulate work
            
            # Should log start and success
            assert mock_logger.info.call_count == 2
            start_call = mock_logger.info.call_args_list[0][0][0]
            success_call = mock_logger.info.call_args_list[1][0][0]
            
            assert "🔄 PARALLEL_START test operation" in start_call
            assert "tasks: 3" in start_call
            assert "🎯 PARALLEL_SUCCESS test operation" in success_call
            assert "duration:" in success_call

    def test_log_parallel_execution_error(self):
        """Test log_parallel_execution context manager error case."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError, match="Test error"):
                with log_parallel_execution("TestComponent", "failing operation", 2):
                    raise ValueError("Test error")
            
            # Should log start and error
            assert mock_logger.info.call_count == 1  # Only start
            assert mock_logger.error.call_count == 1
            
            start_call = mock_logger.info.call_args[0][0]
            error_call = mock_logger.error.call_args[0][0]
            
            assert "🔄 PARALLEL_START failing operation" in start_call
            assert "💥 PARALLEL_ERROR failing operation" in error_call
            assert "error: Test error" in error_call

    def test_log_agent_execution_success(self):
        """Test log_agent_execution context manager success case."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with log_agent_execution("fundamental", "task_123", ["earnings", "valuation"]) as logger:
                assert logger == mock_logger
                time.sleep(0.01)
            
            # Should log start and success
            assert mock_logger.info.call_count == 2
            start_call = mock_logger.info.call_args_list[0][0][0]
            success_call = mock_logger.info.call_args_list[1][0][0]
            
            assert "🤖 AGENT_START task_123" in start_call
            assert "focus: ['earnings', 'valuation']" in start_call
            assert "🎉 AGENT_SUCCESS task_123" in success_call

    def test_log_agent_execution_without_focus(self):
        """Test log_agent_execution without focus areas."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with log_agent_execution("sentiment", "task_456"):
                pass
            
            start_call = mock_logger.info.call_args_list[0][0][0]
            assert "focus:" not in start_call

    def test_log_agent_execution_error(self):
        """Test log_agent_execution context manager error case."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(RuntimeError, match="Agent failed"):
                with log_agent_execution("technical", "task_789"):
                    raise RuntimeError("Agent failed")
            
            # Should log start and error
            error_call = mock_logger.error.call_args[0][0]
            assert "🔥 AGENT_ERROR task_789" in error_call
            assert "error: Agent failed" in error_call


class TestUtilityLoggingFunctions:
    """Test utility logging functions."""

    def test_log_tool_invocation_basic(self):
        """Test basic tool invocation logging."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_tool_invocation("test_tool")
            
            mock_logger.info.assert_called_once()
            call_arg = mock_logger.info.call_args[0][0]
            assert "🔧 TOOL_INVOKE test_tool" in call_arg

    def test_log_tool_invocation_with_request_data(self):
        """Test tool invocation logging with request data."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            request_data = {
                "query": "This is a test query that is longer than 50 characters to test truncation",
                "research_scope": "comprehensive",
                "persona": "moderate"
            }
            
            log_tool_invocation("research_tool", request_data)
            
            call_arg = mock_logger.info.call_args[0][0]
            assert "🔧 TOOL_INVOKE research_tool" in call_arg
            assert "query: 'This is a test query that is longer than 50 charac...'" in call_arg
            assert "scope: comprehensive" in call_arg
            assert "persona: moderate" in call_arg

    def test_log_synthesis_operation(self):
        """Test synthesis operation logging."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_synthesis_operation("parallel_research", 5, "Combined insights from multiple agents")
            
            call_arg = mock_logger.info.call_args[0][0]
            assert "🧠 SYNTHESIS parallel_research" in call_arg
            assert "inputs: 5" in call_arg
            assert "output: Combined insights from multiple agents" in call_arg

    def test_log_synthesis_operation_without_output(self):
        """Test synthesis operation logging without output summary."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_synthesis_operation("basic_synthesis", 3)
            
            call_arg = mock_logger.info.call_args[0][0]
            assert "🧠 SYNTHESIS basic_synthesis" in call_arg
            assert "inputs: 3" in call_arg
            assert "output:" not in call_arg

    def test_log_fallback_trigger(self):
        """Test fallback trigger logging."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_fallback_trigger("ParallelOrchestrator", "API timeout", "switch to sequential")
            
            mock_logger.warning.assert_called_once()
            call_arg = mock_logger.warning.call_args[0][0]
            assert "⚠️ FALLBACK_TRIGGER API timeout" in call_arg
            assert "action: switch to sequential" in call_arg

    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            metrics = {
                "total_tasks": 5,
                "successful_tasks": 4,
                "failed_tasks": 1,
                "parallel_efficiency": 2.3,
                "total_duration": 1.5
            }
            
            log_performance_metrics("TestComponent", metrics)
            
            call_arg = mock_logger.info.call_args[0][0]
            assert "📊 PERFORMANCE_METRICS" in call_arg
            assert "total_tasks: 5" in call_arg
            assert "successful_tasks: 4" in call_arg
            assert "parallel_efficiency: 2.3" in call_arg

    def test_log_resource_usage_complete(self):
        """Test resource usage logging with all parameters."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_resource_usage("ResourceComponent", api_calls=15, cache_hits=8, memory_mb=45.7)
            
            call_arg = mock_logger.info.call_args[0][0]
            assert "📈 RESOURCE_USAGE" in call_arg
            assert "api_calls: 15" in call_arg
            assert "cache_hits: 8" in call_arg
            assert "memory_mb: 45.7" in call_arg

    def test_log_resource_usage_partial(self):
        """Test resource usage logging with partial parameters."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_resource_usage("ResourceComponent", api_calls=10, cache_hits=None)
            
            call_arg = mock_logger.info.call_args[0][0]
            assert "api_calls: 10" in call_arg
            assert "cache_hits" not in call_arg
            assert "memory_mb" not in call_arg

    def test_log_resource_usage_no_params(self):
        """Test resource usage logging with no valid parameters."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_resource_usage("ResourceComponent", api_calls=None, cache_hits=None, memory_mb=None)
            
            # Should not call logger if no valid parameters
            mock_logger.info.assert_not_called()


class TestIntegratedLoggingScenarios:
    """Test integrated logging scenarios."""

    @pytest.mark.asyncio
    async def test_complete_parallel_research_logging(self):
        """Test complete parallel research logging scenario."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Simulate complete parallel research workflow
            class MockResearchAgent:
                @log_method_call(component="ParallelOrchestrator")
                async def execute_parallel_research(self, topic: str, session_id: str):
                    # Set context for this research session
                    orchestration_logger = get_orchestration_logger("ParallelOrchestrator")
                    orchestration_logger.set_request_context(
                        session_id=session_id,
                        research_topic=topic[:50],
                        task_count=3
                    )
                    
                    # Log tool invocation
                    log_tool_invocation("deep_research", {
                        "query": topic,
                        "research_scope": "comprehensive"
                    })
                    
                    # Execute parallel tasks
                    with log_parallel_execution("ParallelOrchestrator", "research execution", 3):
                        # Simulate parallel agent executions
                        for i, agent_type in enumerate(["fundamental", "sentiment", "technical"]):
                            with log_agent_execution(agent_type, f"task_{i}", ["focus1", "focus2"]):
                                await asyncio.sleep(0.01)  # Simulate work
                    
                    # Log synthesis
                    log_synthesis_operation("parallel_research_synthesis", 3, "Comprehensive analysis")
                    
                    # Log performance metrics
                    log_performance_metrics("ParallelOrchestrator", {
                        "successful_tasks": 3,
                        "failed_tasks": 0,
                        "parallel_efficiency": 2.5
                    })
                    
                    # Log resource usage
                    log_resource_usage("ParallelOrchestrator", api_calls=15, cache_hits=5)
                    
                    return {
                        "status": "success",
                        "execution_mode": "parallel",
                        "research_confidence": 0.85
                    }
            
            agent = MockResearchAgent()
            result = await agent.execute_parallel_research(
                topic="Apple Inc comprehensive analysis",
                session_id="integrated_test_123"
            )
            
            # Verify comprehensive logging occurred
            assert mock_logger.info.call_count >= 8  # Multiple info logs expected
            assert result["status"] == "success"

    def test_logging_component_isolation(self):
        """Test that different components maintain separate logging contexts."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger_a = Mock()
            mock_logger_b = Mock()
            
            # Mock different loggers for different components
            def get_logger_side_effect(component_name):
                if component_name == "ComponentA":
                    return mock_logger_a
                elif component_name == "ComponentB":
                    return mock_logger_b
                else:
                    return Mock()
            
            mock_get_logger.side_effect = get_logger_side_effect
            
            # Component A operations
            log_performance_metrics("ComponentA", {"metric_a": 1})
            log_resource_usage("ComponentA", api_calls=5)
            
            # Component B operations  
            log_performance_metrics("ComponentB", {"metric_b": 2})
            log_fallback_trigger("ComponentB", "test reason", "test action")
            
            # Verify isolation
            assert mock_logger_a.info.call_count == 2  # Performance + resource
            assert mock_logger_b.info.call_count == 1  # Performance only
            assert mock_logger_b.warning.call_count == 1  # Fallback trigger

    @pytest.mark.asyncio
    async def test_error_propagation_with_logging(self):
        """Test that errors are properly logged and propagated."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            class ErrorComponent:
                @log_method_call(component="ErrorComponent")
                async def failing_operation(self):
                    with log_parallel_execution("ErrorComponent", "failing task", 1):
                        with log_agent_execution("test_agent", "failing_task"):
                            raise RuntimeError("Simulated failure")
            
            component = ErrorComponent()
            
            # Should properly propagate the error while logging it
            with pytest.raises(RuntimeError, match="Simulated failure"):
                await component.failing_operation()
            
            # Verify error was logged at multiple levels
            assert mock_logger.error.call_count >= 2  # Method and context manager errors

    def test_performance_timing_accuracy(self):
        """Test timing accuracy in logging decorators and context managers."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_method_call(component="TimingTest")
            def timed_function():
                time.sleep(0.1)  # Sleep for ~100ms
                return "completed"
            
            result = timed_function()
            
            assert result == "completed"
            
            # Check that timing was logged
            success_call = mock_logger.info.call_args_list[1][0][0]
            assert "duration:" in success_call
            
            # Extract duration (rough check - timing can be imprecise in tests)
            duration_part = [part for part in success_call.split() if "duration:" in part][0]
            duration_value = float(duration_part.split(":")[-1].replace("s", ""))
            assert 0.05 <= duration_value <= 0.5  # Should be around 0.1s with some tolerance


class TestLoggingUnderLoad:
    """Test logging behavior under various load conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_logging_safety(self):
        """Test that concurrent logging operations are safe."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_method_call(component="ConcurrentTest")
            async def concurrent_task(task_id: int):
                with log_agent_execution("test_agent", f"task_{task_id}"):
                    await asyncio.sleep(0.01)
                return f"result_{task_id}"
            
            # Run multiple tasks concurrently
            tasks = [concurrent_task(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            # Verify all tasks completed
            assert len(results) == 5
            assert all("result_" in result for result in results)
            
            # Logging should have occurred for all tasks
            assert mock_logger.info.call_count >= 10  # At least 2 per task

    def test_high_frequency_logging(self):
        """Test logging performance under high frequency operations."""
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Perform many logging operations quickly
            start_time = time.time()
            
            for i in range(100):
                log_performance_metrics(f"Component_{i % 5}", {
                    "operation_id": i,
                    "timestamp": time.time()
                })
            
            end_time = time.time()
            
            # Should complete quickly
            assert (end_time - start_time) < 1.0  # Should take less than 1 second
            
            # All operations should have been logged
            assert mock_logger.info.call_count == 100

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self):
        """Test that logging doesn't consume excessive memory."""
        import gc
        import sys
        
        with patch('maverick_mcp.utils.orchestration_logging.get_orchestration_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Get baseline memory
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Perform many logging operations
            for i in range(50):
                logger = OrchestrationLogger(f"TestComponent_{i}")
                logger.set_request_context(
                    session_id=f"session_{i}",
                    request_id=f"req_{i}",
                    large_data=f"data_{'x' * 100}_{i}"  # Some larger context data
                )
                logger.info("Test message", param1=f"value_{i}", param2=i)
            
            # Check memory growth
            gc.collect()
            final_objects = len(gc.get_objects())
            
            # Memory growth should be reasonable (not growing indefinitely)
            object_growth = final_objects - initial_objects
            assert object_growth < 1000  # Reasonable threshold for test