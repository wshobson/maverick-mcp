"""
Comprehensive circuit breaker implementation for all external API calls.

This module provides circuit breakers for:
- yfinance (Yahoo Finance)
- Tiingo API
- FRED API
- OpenRouter AI API
- Exa Search API
- Any other external services

Circuit breakers help prevent cascade failures and provide graceful degradation.
"""

import asyncio
import functools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import CircuitBreakerError, ExternalServiceError

logger = logging.getLogger(__name__)
settings = get_settings()

P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureDetectionStrategy(Enum):
    """Strategies for detecting failures."""

    CONSECUTIVE_FAILURES = "consecutive"  # N failures in a row
    FAILURE_RATE = "failure_rate"  # % of failures in time window
    TIMEOUT_RATE = "timeout_rate"  # % of timeouts in time window
    COMBINED = "combined"  # Any of the above


class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        failure_rate_threshold: float = 0.5,
        timeout_threshold: float = 10.0,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        window_size: int = 60,
        detection_strategy: FailureDetectionStrategy = FailureDetectionStrategy.COMBINED,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        """
        Initialize circuit breaker configuration.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of consecutive failures before opening
            failure_rate_threshold: Failure rate (0-1) before opening
            timeout_threshold: Timeout in seconds for calls
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Successes needed in half-open to close
            window_size: Time window in seconds for rate calculations
            detection_strategy: Strategy for detecting failures
            expected_exceptions: Exceptions to catch and count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.timeout_threshold = timeout_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.detection_strategy = detection_strategy
        self.expected_exceptions = expected_exceptions


class CircuitBreakerMetrics:
    """Metrics collection for circuit breakers."""

    def __init__(self, window_size: int = 300):
        """Initialize metrics with a time window."""
        self.window_size = window_size
        self.calls: deque[tuple[float, bool, float]] = (
            deque()
        )  # (timestamp, success, duration)
        self.state_changes: deque[tuple[float, CircuitState]] = deque()
        self._lock = threading.RLock()

    def record_call(self, success: bool, duration: float):
        """Record a call result."""
        with self._lock:
            now = time.time()
            self.calls.append((now, success, duration))
            self._cleanup_old_data(now)

    def record_state_change(self, new_state: CircuitState):
        """Record a state change."""
        with self._lock:
            now = time.time()
            self.state_changes.append((now, new_state))
            self._cleanup_old_data(now)

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            now = time.time()
            self._cleanup_old_data(now)

            if not self.calls:
                return {
                    "total_calls": 0,
                    "success_rate": 1.0,
                    "failure_rate": 0.0,
                    "avg_duration": 0.0,
                    "timeout_rate": 0.0,
                }

            total = len(self.calls)
            successes = sum(1 for _, success, _ in self.calls if success)
            failures = total - successes
            durations = [duration for _, _, duration in self.calls]
            timeouts = sum(
                1
                for _, success, duration in self.calls
                if not success and duration >= 10.0
            )

            return {
                "total_calls": total,
                "success_rate": successes / total if total > 0 else 1.0,
                "failure_rate": failures / total if total > 0 else 0.0,
                "avg_duration": sum(durations) / len(durations) if durations else 0.0,
                "timeout_rate": timeouts / total if total > 0 else 0.0,
                "min_duration": min(durations) if durations else 0.0,
                "max_duration": max(durations) if durations else 0.0,
            }

    def get_total_calls(self) -> int:
        """Get total number of calls in the window."""
        with self._lock:
            now = time.time()
            self._cleanup_old_data(now)
            return len(self.calls)

    def get_success_rate(self) -> float:
        """Get success rate in the window."""
        stats = self.get_stats()
        return stats["success_rate"]

    def get_failure_rate(self) -> float:
        """Get failure rate in the window."""
        stats = self.get_stats()
        return stats["failure_rate"]

    def get_average_response_time(self) -> float:
        """Get average response time in the window."""
        stats = self.get_stats()
        return stats["avg_duration"]

    def get_last_failure_time(self) -> float | None:
        """Get timestamp of last failure."""
        with self._lock:
            for timestamp, success, _ in reversed(self.calls):
                if not success:
                    return timestamp
            return None

    def get_uptime_percentage(self) -> float:
        """Get uptime percentage based on state changes."""
        with self._lock:
            if not self.state_changes:
                return 100.0

            now = time.time()
            window_start = now - self.window_size
            uptime = 0.0
            last_time = window_start
            last_state = CircuitState.CLOSED

            for timestamp, state in self.state_changes:
                if timestamp < window_start:
                    last_state = state
                    continue

                if last_state == CircuitState.CLOSED:
                    uptime += timestamp - last_time

                last_time = timestamp
                last_state = state

            if last_state == CircuitState.CLOSED:
                uptime += now - last_time

            total_time = now - window_start
            return (uptime / total_time * 100) if total_time > 0 else 100.0

    def _cleanup_old_data(self, now: float):
        """Remove data outside the window."""
        cutoff = now - self.window_size

        # Clean up calls
        while self.calls and self.calls[0][0] < cutoff:
            self.calls.popleft()

        # Clean up state changes (keep longer history)
        state_cutoff = now - (self.window_size * 10)
        while self.state_changes and self.state_changes[0][0] < state_cutoff:
            self.state_changes.popleft()


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with failure rate tracking, timeouts, and metrics.
    Thread-safe and supports both sync and async operations.
    """

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize enhanced circuit breaker."""
        self.config = config
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_successes = 0
        self._last_failure_time: float | None = None
        self._metrics = CircuitBreakerMetrics(config.window_size)

        # Thread-safe locks
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Get consecutive failures count."""
        with self._lock:
            return self._consecutive_failures

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    def time_until_retry(self) -> float | None:
        """Get time until next retry attempt."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._last_failure_time:
                return max(
                    0,
                    self.config.recovery_timeout
                    - (time.time() - self._last_failure_time),
                )
            return None

    def _should_open(self) -> bool:
        """Determine if circuit should open based on detection strategy."""
        stats = self._metrics.get_stats()

        if (
            self.config.detection_strategy
            == FailureDetectionStrategy.CONSECUTIVE_FAILURES
        ):
            return self._consecutive_failures >= self.config.failure_threshold

        elif self.config.detection_strategy == FailureDetectionStrategy.FAILURE_RATE:
            return (
                stats["total_calls"] >= 5  # Minimum calls for rate calculation
                and stats["failure_rate"] >= self.config.failure_rate_threshold
            )

        elif self.config.detection_strategy == FailureDetectionStrategy.TIMEOUT_RATE:
            return (
                stats["total_calls"] >= 5
                and stats["timeout_rate"] >= self.config.failure_rate_threshold
            )

        else:  # COMBINED
            return (
                self._consecutive_failures >= self.config.failure_threshold
                or (
                    stats["total_calls"] >= 5
                    and stats["failure_rate"] >= self.config.failure_rate_threshold
                )
                or (
                    stats["total_calls"] >= 5
                    and stats["timeout_rate"] >= self.config.failure_rate_threshold
                )
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.config.recovery_timeout

    def _transition_state(self, new_state: CircuitState):
        """Transition to a new state."""
        if self._state != new_state:
            logger.info(
                f"Circuit breaker '{self.config.name}' transitioning from {self._state.value} to {new_state.value}"
            )
            self._state = new_state
            self._metrics.record_state_change(new_state)

    def _on_success(self, duration: float):
        """Handle successful call."""
        with self._lock:
            self._metrics.record_call(True, duration)
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_state(CircuitState.CLOSED)
                    self._half_open_successes = 0

    def _on_failure(self, duration: float):
        """Handle failed call."""
        with self._lock:
            self._metrics.record_call(False, duration)
            self._consecutive_failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_state(CircuitState.OPEN)
                self._half_open_successes = 0
            elif self._state == CircuitState.CLOSED and self._should_open():
                self._transition_state(CircuitState.OPEN)

    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Call function through circuit breaker (sync version)."""
        return self.call_sync(func, *args, **kwargs)

    async def call_async(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Call async function through circuit breaker with timeout support.

        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        # Check if we should attempt reset
        async with self._async_lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_state(CircuitState.HALF_OPEN)
                    self._half_open_successes = 0
                else:
                    time_until_retry = self.config.recovery_timeout
                    if self._last_failure_time:
                        time_until_retry = max(
                            0,
                            self.config.recovery_timeout
                            - (time.time() - self._last_failure_time),
                        )
                    raise CircuitBreakerError(
                        service=self.config.name,
                        failure_count=self._consecutive_failures,
                        threshold=self.config.failure_threshold,
                        context={
                            "state": self._state.value,
                            "time_until_retry": round(time_until_retry, 1),
                        },
                    )

        start_time = time.time()
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout_threshold
            )
            duration = time.time() - start_time
            self._on_success(duration)
            return result

        except TimeoutError as e:
            duration = time.time() - start_time
            self._on_failure(duration)
            logger.warning(
                f"Circuit breaker '{self.config.name}' timeout after {duration:.2f}s"
            )
            raise ExternalServiceError(
                service=self.config.name,
                message=f"Service timed out after {self.config.timeout_threshold}s",
                context={
                    "timeout": self.config.timeout_threshold,
                },
            ) from e

        except self.config.expected_exceptions:
            duration = time.time() - start_time
            self._on_failure(duration)
            raise

    def call_sync(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Call sync function through circuit breaker.

        For sync functions, timeout is enforced differently depending on the function type.
        HTTP requests should use their own timeout parameters.
        """
        # Check if we should attempt reset
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_state(CircuitState.HALF_OPEN)
                    self._half_open_successes = 0
                else:
                    time_until_retry = self.config.recovery_timeout
                    if self._last_failure_time:
                        time_until_retry = max(
                            0,
                            self.config.recovery_timeout
                            - (time.time() - self._last_failure_time),
                        )
                    raise CircuitBreakerError(
                        service=self.config.name,
                        failure_count=self._consecutive_failures,
                        threshold=self.config.failure_threshold,
                        context={
                            "state": self._state.value,
                            "time_until_retry": round(time_until_retry, 1),
                        },
                    )

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            self._on_success(duration)
            return result

        except self.config.expected_exceptions:
            duration = time.time() - start_time
            self._on_failure(duration)
            raise

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_state(CircuitState.CLOSED)
            self._consecutive_failures = 0
            self._half_open_successes = 0
            self._last_failure_time = None
            logger.info(f"Circuit breaker '{self.config.name}' manually reset")

    def get_status(self) -> dict[str, Any]:
        """Get detailed circuit breaker status."""
        with self._lock:
            stats = self._metrics.get_stats()
            time_until_retry = None

            if self._state == CircuitState.OPEN and self._last_failure_time:
                time_until_retry = max(
                    0,
                    self.config.recovery_timeout
                    - (time.time() - self._last_failure_time),
                )

            return {
                "name": self.config.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "time_until_retry": round(time_until_retry, 1)
                if time_until_retry
                else None,
                "metrics": stats,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "failure_rate_threshold": self.config.failure_rate_threshold,
                    "timeout_threshold": self.config.timeout_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "detection_strategy": self.config.detection_strategy.value,
                },
            }


# Global registry of circuit breakers
_breakers: dict[str, EnhancedCircuitBreaker] = {}
_breakers_lock = threading.Lock()


def _get_or_create_breaker(config: CircuitBreakerConfig) -> EnhancedCircuitBreaker:
    """Get or create a circuit breaker."""
    with _breakers_lock:
        if config.name not in _breakers:
            _breakers[config.name] = EnhancedCircuitBreaker(config)
        return _breakers[config.name]


def register_circuit_breaker(name: str, breaker: EnhancedCircuitBreaker):
    """Register a circuit breaker in the global registry."""
    with _breakers_lock:
        _breakers[name] = breaker
        logger.debug(f"Registered circuit breaker: {name}")


def get_circuit_breaker(name: str) -> EnhancedCircuitBreaker | None:
    """Get a circuit breaker by name."""
    return _breakers.get(name)


def get_all_circuit_breakers() -> dict[str, EnhancedCircuitBreaker]:
    """Get all circuit breakers."""
    return _breakers.copy()


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    for breaker in _breakers.values():
        breaker.reset()


def get_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    """Get status of all circuit breakers."""
    return {name: breaker.get_status() for name, breaker in _breakers.items()}


def circuit_breaker(
    name: str | None = None,
    failure_threshold: int | None = None,
    failure_rate_threshold: float | None = None,
    timeout_threshold: float | None = None,
    recovery_timeout: int | None = None,
    expected_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable:
    """
    Decorator to apply circuit breaker to a function.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Override default failure threshold
        failure_rate_threshold: Override default failure rate threshold
        timeout_threshold: Override default timeout threshold
        recovery_timeout: Override default recovery timeout
        expected_exceptions: Exceptions to catch (defaults to Exception)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Create config with overrides
        cb_name = name or f"{func.__module__}.{getattr(func, '__name__', 'unknown')}"
        config = CircuitBreakerConfig(
            name=cb_name,
            failure_threshold=failure_threshold
            or settings.agent.circuit_breaker_failure_threshold,
            failure_rate_threshold=failure_rate_threshold or 0.5,
            timeout_threshold=timeout_threshold or 30.0,
            recovery_timeout=recovery_timeout
            or settings.agent.circuit_breaker_recovery_timeout,
            expected_exceptions=expected_exceptions or (Exception,),
        )

        # Get or create circuit breaker for this function
        breaker = _get_or_create_breaker(config)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)

            return cast(Callable[..., T], async_wrapper)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call_sync(func, *args, **kwargs)

            return cast(Callable[..., T], sync_wrapper)

    return decorator


# Circuit breaker configurations for different services
CIRCUIT_BREAKER_CONFIGS = {
    "yfinance": CircuitBreakerConfig(
        name="yfinance",
        failure_threshold=3,
        failure_rate_threshold=0.6,
        timeout_threshold=30.0,
        recovery_timeout=120,
        success_threshold=2,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "tiingo": CircuitBreakerConfig(
        name="tiingo",
        failure_threshold=5,
        failure_rate_threshold=0.7,
        timeout_threshold=15.0,
        recovery_timeout=60,
        success_threshold=3,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "fred_api": CircuitBreakerConfig(
        name="fred_api",
        failure_threshold=3,
        failure_rate_threshold=0.5,
        timeout_threshold=20.0,
        recovery_timeout=180,
        success_threshold=2,
        window_size=600,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "openrouter": CircuitBreakerConfig(
        name="openrouter",
        failure_threshold=5,
        failure_rate_threshold=0.6,
        timeout_threshold=60.0,  # AI APIs can be slower
        recovery_timeout=120,
        success_threshold=2,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "exa": CircuitBreakerConfig(
        name="exa",
        failure_threshold=4,
        failure_rate_threshold=0.6,
        timeout_threshold=30.0,
        recovery_timeout=90,
        success_threshold=2,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "news_api": CircuitBreakerConfig(
        name="news_api",
        failure_threshold=3,
        failure_rate_threshold=0.5,
        timeout_threshold=25.0,
        recovery_timeout=120,
        success_threshold=2,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "finviz": CircuitBreakerConfig(
        name="finviz",
        failure_threshold=3,
        failure_rate_threshold=0.6,
        timeout_threshold=20.0,
        recovery_timeout=150,
        success_threshold=2,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
    "external_api": CircuitBreakerConfig(
        name="external_api",
        failure_threshold=4,
        failure_rate_threshold=0.6,
        timeout_threshold=25.0,
        recovery_timeout=120,
        success_threshold=2,
        window_size=300,
        detection_strategy=FailureDetectionStrategy.COMBINED,
        expected_exceptions=(Exception,),
    ),
}


def initialize_circuit_breakers() -> dict[str, EnhancedCircuitBreaker]:
    """Initialize all circuit breakers for external services."""
    circuit_breakers = {}

    for service_name, config in CIRCUIT_BREAKER_CONFIGS.items():
        try:
            breaker = EnhancedCircuitBreaker(config)
            register_circuit_breaker(service_name, breaker)
            circuit_breakers[service_name] = breaker
            logger.info(f"Initialized circuit breaker for {service_name}")
        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker for {service_name}: {e}")

    logger.info(f"Initialized {len(circuit_breakers)} circuit breakers")
    return circuit_breakers


def with_circuit_breaker(service_name: str):
    """Decorator to wrap functions with a circuit breaker.

    Args:
        service_name: Name of the service/circuit breaker to use

    Usage:
        @with_circuit_breaker("yfinance")
        def fetch_stock_data(symbol: str):
            # API call code here
            pass
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            breaker = get_circuit_breaker(service_name)
            if not breaker:
                logger.warning(f"Circuit breaker '{service_name}' not found, executing without protection")
                return func(*args, **kwargs)

            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def with_async_circuit_breaker(service_name: str):
    """Decorator to wrap async functions with a circuit breaker.

    Args:
        service_name: Name of the service/circuit breaker to use

    Usage:
        @with_async_circuit_breaker("tiingo")
        async def fetch_real_time_data(symbol: str):
            # Async API call code here
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            breaker = get_circuit_breaker(service_name)
            if not breaker:
                logger.warning(f"Circuit breaker '{service_name}' not found, executing without protection")
                return await func(*args, **kwargs)

            return await breaker.call_async(func, *args, **kwargs)

        return wrapper

    return decorator


class CircuitBreakerManager:
    """Manager for all circuit breakers in the application."""

    def __init__(self):
        self._breakers = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all circuit breakers."""
        if self._initialized:
            return True

        try:
            self._breakers = initialize_circuit_breakers()
            self._initialized = True
            logger.info("Circuit breaker manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker manager: {e}")
            return False

    def get_breaker(self, service_name: str) -> EnhancedCircuitBreaker | None:
        """Get a circuit breaker by service name."""
        if not self._initialized:
            self.initialize()

        return self._breakers.get(service_name)

    def get_all_breakers(self) -> dict[str, EnhancedCircuitBreaker]:
        """Get all circuit breakers."""
        if not self._initialized:
            self.initialize()

        return self._breakers.copy()

    def reset_breaker(self, service_name: str) -> bool:
        """Reset a specific circuit breaker."""
        breaker = self.get_breaker(service_name)
        if breaker:
            breaker.reset()
            logger.info(f"Reset circuit breaker for {service_name}")
            return True
        return False

    def reset_all_breakers(self) -> int:
        """Reset all circuit breakers."""
        reset_count = 0
        for service_name, breaker in self._breakers.items():
            try:
                breaker.reset()
                reset_count += 1
                logger.info(f"Reset circuit breaker for {service_name}")
            except Exception as e:
                logger.error(f"Failed to reset circuit breaker for {service_name}: {e}")

        logger.info(f"Reset {reset_count} circuit breakers")
        return reset_count

    def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all circuit breakers."""
        if not self._initialized:
            self.initialize()

        status = {}
        for service_name, breaker in self._breakers.items():
            try:
                metrics = breaker.get_metrics()
                status[service_name] = {
                    "name": service_name,
                    "state": breaker.state.value,
                    "consecutive_failures": breaker.consecutive_failures,
                    "time_until_retry": breaker.time_until_retry(),
                    "metrics": {
                        "total_calls": metrics.get_total_calls(),
                        "success_rate": metrics.get_success_rate(),
                        "failure_rate": metrics.get_failure_rate(),
                        "avg_response_time": metrics.get_average_response_time(),
                        "last_failure_time": metrics.get_last_failure_time(),
                        "uptime_percentage": metrics.get_uptime_percentage(),
                    },
                }
            except Exception as e:
                status[service_name] = {
                    "name": service_name,
                    "state": "error",
                    "error": str(e),
                }

        return status


# Global circuit breaker manager instance
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    return _circuit_breaker_manager


def initialize_all_circuit_breakers() -> bool:
    """Initialize all circuit breakers (convenience function)."""
    return _circuit_breaker_manager.initialize()


def get_all_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    """Get status of all circuit breakers (convenience function)."""
    return _circuit_breaker_manager.get_health_status()


# Specific circuit breaker decorators for common services

def with_yfinance_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Decorator for yfinance API calls."""
    return cast(F, with_circuit_breaker("yfinance")(func))


def with_tiingo_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Decorator for Tiingo API calls."""
    return cast(F, with_circuit_breaker("tiingo")(func))


def with_fred_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Decorator for FRED API calls."""
    return cast(F, with_circuit_breaker("fred_api")(func))


def with_openrouter_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Decorator for OpenRouter API calls."""
    return cast(F, with_circuit_breaker("openrouter")(func))


def with_exa_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Decorator for Exa API calls."""
    return cast(F, with_circuit_breaker("exa")(func))


# Async versions

def with_async_yfinance_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Async decorator for yfinance API calls."""
    return cast(F, with_async_circuit_breaker("yfinance")(func))


def with_async_tiingo_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Async decorator for Tiingo API calls."""
    return cast(F, with_async_circuit_breaker("tiingo")(func))


def with_async_fred_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Async decorator for FRED API calls."""
    return cast(F, with_async_circuit_breaker("fred_api")(func))


def with_async_openrouter_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Async decorator for OpenRouter API calls."""
    return cast(F, with_async_circuit_breaker("openrouter")(func))


def with_async_exa_circuit_breaker(func: F) -> F:  # noqa: UP047
    """Async decorator for Exa API calls."""
    return cast(F, with_async_circuit_breaker("exa")(func))
