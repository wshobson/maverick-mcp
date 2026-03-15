"""
Comprehensive tests for the circuit breaker system.

Tests cover the adapter-based implementation backed by the ``circuitbreaker``
library, the backward-compatible facade in ``circuit_breaker.py``, and the
service-specific wrappers in ``circuit_breaker_services.py``.
"""

import time
from unittest.mock import patch

import pytest

from maverick_mcp.exceptions import CircuitBreakerError
from maverick_mcp.utils.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
    EnhancedCircuitBreaker,
    circuit_breaker,
    get_all_circuit_breakers,
    get_circuit_breaker,
    get_circuit_breaker_status,
    reset_all_circuit_breakers,
)
from maverick_mcp.utils.circuit_breaker_adapter import (
    MaverickCircuitBreaker,
    _breakers,
    _breakers_lock,
    get_or_create_breaker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_breaker(name: str, **kwargs) -> MaverickCircuitBreaker:
    """Create a breaker that is NOT in the global registry."""
    return MaverickCircuitBreaker(name=name, **kwargs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MaverickCircuitBreaker (adapter)
# ---------------------------------------------------------------------------


class TestMaverickCircuitBreaker:
    """Test the adapter-based MaverickCircuitBreaker."""

    def test_initialization(self):
        """Test breaker is initialized correctly."""
        breaker = _fresh_breaker("test_init", failure_threshold=3, recovery_timeout=5)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_consecutive_failures_opens_circuit(self):
        """Test circuit opens after consecutive failures."""
        breaker = _fresh_breaker("test_open", failure_threshold=3)

        for _ in range(3):
            try:
                breaker.call_sync(lambda: 1 / 0)
            except ZeroDivisionError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_blocks_calls_when_open(self):
        """Test circuit breaker blocks calls when open."""
        breaker = _fresh_breaker("test_block", failure_threshold=1, recovery_timeout=60)

        # Open the circuit
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        # Next call should be blocked
        with pytest.raises(CircuitBreakerError) as exc_info:
            breaker.call_sync(lambda: "success")

        assert exc_info.value.context["state"] == "open"

    def test_recovery(self):
        """Test circuit breaker recovery to half-open then closed."""
        breaker = _fresh_breaker(
            "test_recovery", failure_threshold=1, recovery_timeout=1
        )

        # Open the circuit
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # Successful call should move to half-open then closed
        result = breaker.call_sync(lambda: "success")
        assert result == "success"
        # Library moves back to CLOSED after 1 success in half-open
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Test failure in half-open state reopens circuit."""
        breaker = _fresh_breaker("test_reopen", failure_threshold=1, recovery_timeout=1)

        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        time.sleep(1.1)

        # Fail in half-open state
        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        assert breaker.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        breaker = _fresh_breaker("test_reset", failure_threshold=1)

        try:
            breaker.call_sync(lambda: 1 / 0)
        except ZeroDivisionError:
            pass
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test circuit breaker with async functions."""
        breaker = _fresh_breaker("test_async_cb", failure_threshold=2)

        async def failing_func():
            raise ValueError("Async failure")

        async def success_func():
            return "async success"

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call_async(failing_func)

        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            await breaker.call_async(success_func)

    def test_get_status(self):
        """Test get_status returns expected format."""
        breaker = _fresh_breaker("test_status")
        breaker.call_sync(lambda: "ok")

        status = breaker.get_status()
        assert status["name"] == "test_status"
        assert status["state"] == "closed"
        assert "metrics" in status
        assert "config" in status
        assert status["metrics"]["total_calls"] == 1


# ---------------------------------------------------------------------------
# EnhancedCircuitBreaker (backward-compatible alias)
# ---------------------------------------------------------------------------


class TestEnhancedCircuitBreaker:
    """Test backward-compatible EnhancedCircuitBreaker alias."""

    def test_config_constructor_works(self):
        """EnhancedCircuitBreaker(config) should still work."""
        config = CircuitBreakerConfig(
            name="compat_test",
            failure_threshold=3,
            recovery_timeout=5,
        )
        breaker = EnhancedCircuitBreaker(config)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.config.name == "compat_test"

    def test_is_same_as_maverick_breaker(self):
        """EnhancedCircuitBreaker is MaverickCircuitBreaker."""
        assert EnhancedCircuitBreaker is MaverickCircuitBreaker


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""

    def setup_method(self):
        """Clear registry between tests."""
        with _breakers_lock:
            _breakers.clear()

    def test_sync_decorator(self):
        """Test decorator with sync function."""

        @circuit_breaker(name="test_decorator", failure_threshold=2)
        def test_func(should_fail=False):
            if should_fail:
                raise ValueError("Test failure")
            return "success"

        assert test_func() == "success"
        assert test_func() == "success"

        for _ in range(2):
            with pytest.raises(ValueError):
                test_func(should_fail=True)

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

        result = await async_test_func()
        assert result == "async success"

        with pytest.raises(ValueError):
            await async_test_func(should_fail=True)

        with pytest.raises(CircuitBreakerError):
            await async_test_func()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCircuitBreakerRegistry:
    """Test global circuit breaker registry."""

    def setup_method(self):
        """Clear registry between tests."""
        with _breakers_lock:
            _breakers.clear()

    def test_get_circuit_breaker(self):
        """Test getting circuit breaker by name."""

        @circuit_breaker(name="registry_test")
        def test_func():
            return "test"

        test_func()

        breaker = get_circuit_breaker("registry_test")
        assert breaker is not None
        assert breaker.config.name == "registry_test"

    def test_get_all_circuit_breakers(self):
        """Test getting all circuit breakers."""

        @circuit_breaker(name="breaker1")
        def func1():
            pass

        @circuit_breaker(name="breaker2")
        def func2():
            pass

        func1()
        func2()

        all_breakers = get_all_circuit_breakers()
        assert len(all_breakers) == 2
        assert "breaker1" in all_breakers
        assert "breaker2" in all_breakers

    def test_reset_all_circuit_breakers(self):
        """Test resetting all circuit breakers."""

        @circuit_breaker(name="reset_test", failure_threshold=1)
        def failing_func():
            raise ValueError("Fail")

        with pytest.raises(ValueError):
            failing_func()

        breaker = get_circuit_breaker("reset_test")
        assert breaker.state == CircuitState.OPEN

        reset_all_circuit_breakers()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_status(self):
        """Test getting status of all circuit breakers."""

        @circuit_breaker(name="status_test")
        def test_func():
            return "test"

        test_func()

        status = get_circuit_breaker_status()
        assert "status_test" in status
        assert status["status_test"]["state"] == "closed"
        assert status["status_test"]["name"] == "status_test"


# ---------------------------------------------------------------------------
# Service-specific circuit breakers
# ---------------------------------------------------------------------------


class TestServiceSpecificCircuitBreakers:
    """Test service-specific circuit breaker implementations."""

    def setup_method(self):
        """Clear registry so each test gets fresh breakers."""
        with _breakers_lock:
            _breakers.clear()

    def test_stock_data_circuit_breaker(self):
        """Test stock data circuit breaker with fallback."""
        from maverick_mcp.utils.circuit_breaker_services import StockDataCircuitBreaker

        breaker = StockDataCircuitBreaker()

        def failing_fetch(symbol, start, end):
            raise Exception("API Error")

        with patch.object(breaker.fallback_chain, "execute_sync") as mock_fallback:
            import pandas as pd

            mock_fallback.return_value = pd.DataFrame({"Close": [100, 101, 102]})

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

        def failing_fetch(mover_type):
            raise Exception("Finviz Error")

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

        def failing_fetch(series_id, start, end):
            raise Exception("FRED API Error")

        result = breaker.fetch_with_fallback(
            failing_fetch, "GDP", "2024-01-01", "2024-01-31"
        )

        import pandas as pd

        assert isinstance(result, pd.Series)
        assert result.attrs["is_fallback"] is True
        assert all(result == 2.5)  # Default GDP value

    def test_get_or_create_breaker_reuses(self):
        """Same name returns the same breaker instance."""
        b1 = get_or_create_breaker("reuse_test")
        b2 = get_or_create_breaker("reuse_test")
        assert b1 is b2

    def test_get_or_create_breaker_uses_service_config(self):
        """Known services get their configured thresholds."""
        breaker = get_or_create_breaker("yfinance")
        # The library stores thresholds on instance attrs (_failure_threshold)
        assert breaker._cb._failure_threshold == 3
        assert breaker._cb._recovery_timeout == 120
