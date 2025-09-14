"""
Comprehensive tests for the circuit breaker system.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from maverick_mcp.exceptions import CircuitBreakerError, ExternalServiceError
from maverick_mcp.utils.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    EnhancedCircuitBreaker,
    FailureDetectionStrategy,
    circuit_breaker,
    get_all_circuit_breakers,
    get_circuit_breaker,
    get_circuit_breaker_status,
    reset_all_circuit_breakers,
)


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics collection."""

    def test_metrics_initialization(self):
        """Test metrics are initialized correctly."""
        metrics = CircuitBreakerMetrics(window_size=10)
        stats = metrics.get_stats()

        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["failure_rate"] == 0.0
        assert stats["avg_duration"] == 0.0
        assert stats["timeout_rate"] == 0.0

    def test_record_successful_call(self):
        """Test recording successful calls."""
        metrics = CircuitBreakerMetrics()

        metrics.record_call(True, 0.5)
        metrics.record_call(True, 1.0)

        stats = metrics.get_stats()
        assert stats["total_calls"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["failure_rate"] == 0.0
        assert stats["avg_duration"] == 0.75

    def test_record_failed_call(self):
        """Test recording failed calls."""
        metrics = CircuitBreakerMetrics()

        metrics.record_call(False, 2.0)
        metrics.record_call(True, 1.0)

        stats = metrics.get_stats()
        assert stats["total_calls"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["failure_rate"] == 0.5
        assert stats["avg_duration"] == 1.5

    def test_window_cleanup(self):
        """Test old data is cleaned up."""
        metrics = CircuitBreakerMetrics(window_size=1)  # 1 second window

        metrics.record_call(True, 0.5)
        time.sleep(1.1)  # Wait for window to expire
        metrics.record_call(True, 1.0)

        stats = metrics.get_stats()
        assert stats["total_calls"] == 1  # Old call should be removed


class TestEnhancedCircuitBreaker:
    """Test enhanced circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker is initialized correctly."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=3,
            recovery_timeout=5,
        )
        breaker = EnhancedCircuitBreaker(config)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_consecutive_failures_opens_circuit(self):
        """Test circuit opens after consecutive failures."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=3,
            detection_strategy=FailureDetectionStrategy.CONSECUTIVE_FAILURES,
        )
        breaker = EnhancedCircuitBreaker(config)

        # Fail 3 times
        for _ in range(3):
            try:
                breaker.call_sync(lambda: 1 / 0)
            except ZeroDivisionError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_failure_rate_opens_circuit(self):
        """Test circuit opens based on failure rate."""
        config = CircuitBreakerConfig(
            name="test",
            failure_rate_threshold=0.5,
            detection_strategy=FailureDetectionStrategy.FAILURE_RATE,
        )
        breaker = EnhancedCircuitBreaker(config)

        # Need minimum calls for rate calculation
        for i in range(10):
            try:
                if i % 2 == 0:  # 50% failure rate
                    breaker.call_sync(lambda: 1 / 0)
                else:
                    breaker.call_sync(lambda: "success")
            except (ZeroDivisionError, CircuitBreakerError):
                pass

        stats = breaker._metrics.get_stats()
        assert stats["failure_rate"] >= 0.5
        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_blocks_calls_when_open(self):
        """Test circuit breaker blocks calls when open."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            recovery_timeout=60,
        )
        breaker = EnhancedCircuitBreaker(config)

        # Open the circuit
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        # Next call should be blocked
        with pytest.raises(CircuitBreakerError) as exc_info:
            breaker.call_sync(lambda: "success")

        assert "Circuit breaker open for test:" in str(exc_info.value)
        assert exc_info.value.context["state"] == "open"

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery to half-open then closed."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            recovery_timeout=1,  # 1 second
            success_threshold=2,
        )
        breaker = EnhancedCircuitBreaker(config)

        # Open the circuit
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # First successful call should move to half-open
        result = breaker.call_sync(lambda: "success1")
        assert result == "success1"
        assert breaker.state == CircuitState.HALF_OPEN

        # Second successful call should close the circuit
        result = breaker.call_sync(lambda: "success2")
        assert result == "success2"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Test failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            recovery_timeout=1,
        )
        breaker = EnhancedCircuitBreaker(config)

        # Open the circuit
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        # Wait for recovery
        time.sleep(1.1)

        # Fail in half-open state
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        assert breaker.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
        )
        breaker = EnhancedCircuitBreaker(config)

        # Open the circuit
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test circuit breaker with async functions."""
        config = CircuitBreakerConfig(
            name="test_async",
            failure_threshold=2,
        )
        breaker = EnhancedCircuitBreaker(config)

        async def failing_func():
            raise ValueError("Async failure")

        async def success_func():
            return "async success"

        # Test failures
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call_async(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Test blocking
        with pytest.raises(CircuitBreakerError):
            await breaker.call_async(success_func)

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Test async timeout handling."""
        config = CircuitBreakerConfig(
            name="test_timeout",
            timeout_threshold=0.1,  # 100ms
            failure_threshold=1,
        )
        breaker = EnhancedCircuitBreaker(config)

        async def slow_func():
            await asyncio.sleep(0.5)  # 500ms
            return "done"

        with pytest.raises(ExternalServiceError) as exc_info:
            await breaker.call_async(slow_func)

        assert "timed out" in str(exc_info.value)
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""

    def test_sync_decorator(self):
        """Test decorator with sync function."""
        call_count = 0

        @circuit_breaker(name="test_decorator", failure_threshold=2)
        def test_func(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Test failure")
            return "success"

        # Successful calls
        assert test_func() == "success"
        assert test_func() == "success"

        # Failures
        for _ in range(2):
            with pytest.raises(ValueError):
                test_func(should_fail=True)

        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            test_func()

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator with async function."""

        @circuit_breaker(name="test_async_decorator", failure_threshold=1)
        async def async_test_func(should_fail=False):
            if should_fail:
                raise ValueError("Async test failure")
            return "async success"

        # Success
        result = await async_test_func()
        assert result == "async success"

        # Failure
        with pytest.raises(ValueError):
            await async_test_func(should_fail=True)

        # Circuit open
        with pytest.raises(CircuitBreakerError):
            await async_test_func()


class TestCircuitBreakerRegistry:
    """Test global circuit breaker registry."""

    def test_get_circuit_breaker(self):
        """Test getting circuit breaker by name."""

        # Create a breaker via decorator
        @circuit_breaker(name="registry_test")
        def test_func():
            return "test"

        # Call to initialize
        test_func()

        # Get from registry
        breaker = get_circuit_breaker("registry_test")
        assert breaker is not None
        assert breaker.config.name == "registry_test"

    def test_get_all_circuit_breakers(self):
        """Test getting all circuit breakers."""
        # Clear existing (from other tests)
        from maverick_mcp.utils.circuit_breaker import _breakers

        _breakers.clear()

        # Create multiple breakers
        @circuit_breaker(name="breaker1")
        def func1():
            pass

        @circuit_breaker(name="breaker2")
        def func2():
            pass

        # Initialize
        func1()
        func2()

        all_breakers = get_all_circuit_breakers()
        assert len(all_breakers) == 2
        assert "breaker1" in all_breakers
        assert "breaker2" in all_breakers

    def test_reset_all_circuit_breakers(self):
        """Test resetting all circuit breakers."""

        # Create and open a breaker
        @circuit_breaker(name="reset_test", failure_threshold=1)
        def failing_func():
            raise ValueError("Fail")

        with pytest.raises(ValueError):
            failing_func()

        breaker = get_circuit_breaker("reset_test")
        assert breaker.state == CircuitState.OPEN

        # Reset all
        reset_all_circuit_breakers()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_status(self):
        """Test getting status of all circuit breakers."""

        # Create a breaker
        @circuit_breaker(name="status_test")
        def test_func():
            return "test"

        test_func()

        status = get_circuit_breaker_status()
        assert "status_test" in status
        assert status["status_test"]["state"] == "closed"
        assert status["status_test"]["name"] == "status_test"


class TestServiceSpecificCircuitBreakers:
    """Test service-specific circuit breaker implementations."""

    def test_stock_data_circuit_breaker(self):
        """Test stock data circuit breaker with fallback."""
        from maverick_mcp.utils.circuit_breaker_services import StockDataCircuitBreaker

        breaker = StockDataCircuitBreaker()

        # Mock a failing function
        def failing_fetch(symbol, start, end):
            raise Exception("API Error")

        # Mock fallback data
        with patch.object(breaker.fallback_chain, "execute_sync") as mock_fallback:
            import pandas as pd

            mock_fallback.return_value = pd.DataFrame({"Close": [100, 101, 102]})

            # Should use fallback
            result = breaker.fetch_with_fallback(
                failing_fetch, "AAPL", "2024-01-01", "2024-01-31"
            )

            assert not result.empty
            assert len(result) == 3
            mock_fallback.assert_called_once()

    def test_market_data_circuit_breaker(self):
        """Test market data circuit breaker with fallback."""
        from maverick_mcp.utils.circuit_breaker_services import MarketDataCircuitBreaker

        breaker = MarketDataCircuitBreaker("finviz")

        # Mock failing function
        def failing_fetch(mover_type):
            raise Exception("Finviz Error")

        # Should return fallback
        result = breaker.fetch_with_fallback(failing_fetch, "gainers")

        assert isinstance(result, dict)
        assert "movers" in result
        assert result["movers"] == []
        assert result["metadata"]["is_fallback"] is True

    def test_economic_data_circuit_breaker(self):
        """Test economic data circuit breaker with fallback."""
        from maverick_mcp.utils.circuit_breaker_services import (
            EconomicDataCircuitBreaker,
        )

        breaker = EconomicDataCircuitBreaker()

        # Mock failing function
        def failing_fetch(series_id, start, end):
            raise Exception("FRED API Error")

        # Should return default values
        result = breaker.fetch_with_fallback(
            failing_fetch, "GDP", "2024-01-01", "2024-01-31"
        )

        import pandas as pd

        assert isinstance(result, pd.Series)
        assert result.attrs["is_fallback"] is True
        assert all(result == 2.5)  # Default GDP value
