"""
Circuit Breaker pattern for resilient external API calls.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern to prevent repeated calls
    to failing services and allow them time to recover.
    """

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout: int | None = None,
        expected_exception: type[Exception] = Exception,
        name: str = "CircuitBreaker",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit (uses config default if None)
            recovery_timeout: Seconds to wait before testing recovery (uses config default if None)
            expected_exception: Exception type to catch
            name: Name for logging
        """
        self.failure_threshold = (
            failure_threshold or settings.agent.circuit_breaker_failure_threshold
        )
        self.recovery_timeout = (
            recovery_timeout or settings.agent.circuit_breaker_recovery_timeout
        )
        self.expected_exception = expected_exception
        self.name = name

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"{self.name}: Attempting reset (half-open)")
                else:
                    raise Exception(f"{self.name}: Circuit breaker is OPEN")

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset on half-open or reduce failure count
            await self._on_success()
            return result

        except self.expected_exception as e:
            # Failure - increment counter and possibly open circuit
            await self._on_failure()
            raise e

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"{self.name}: Circuit breaker CLOSED after recovery")
            elif self._failure_count > 0:
                self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"{self.name}: Circuit breaker OPEN after {self._failure_count} failures"
                )
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"{self.name}: Circuit breaker OPEN after half-open test failed"
                )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False

        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    async def reset(self):
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(f"{self.name}: Circuit breaker manually RESET")

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_until_retry": self._get_time_until_retry(),
        }

    def _get_time_until_retry(self) -> float | None:
        """Get seconds until retry is allowed."""
        if self._state != CircuitState.OPEN or self._last_failure_time is None:
            return None

        elapsed = time.time() - self._last_failure_time
        remaining = self.recovery_timeout - elapsed
        return max(0, remaining)


class CircuitBreakerManager:
    """Manage multiple circuit breakers."""

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = Exception,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception,
                    name=name,
                )
            return self._breakers[name]

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()
