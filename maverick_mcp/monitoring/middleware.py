"""
Monitoring middleware for automatic metrics collection.

This module provides middleware components that automatically track:
- API calls and response times
- Strategy execution performance
- Resource usage during operations
- Anomaly detection triggers
"""

import asyncio
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

from maverick_mcp.monitoring.metrics import get_backtesting_metrics
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsMiddleware:
    """
    Middleware for automatic metrics collection during backtesting operations.

    Provides decorators and context managers for seamless metrics integration.
    """

    def __init__(self):
        self.collector = get_backtesting_metrics()
        self.logger = get_logger(f"{__name__}.MetricsMiddleware")

    def track_api_call(
        self,
        provider: str,
        endpoint: str,
        method: str = "GET"
    ):
        """
        Decorator to automatically track API call metrics.

        Usage:
            @middleware.track_api_call("tiingo", "/daily/{symbol}")
            async def get_stock_data(symbol: str):
                # API call logic here
                pass
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200
                error_type = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = getattr(e, 'status_code', 500)
                    error_type = type(e).__name__
                    raise
                finally:
                    duration = time.time() - start_time
                    self.collector.track_api_call(
                        provider=provider,
                        endpoint=endpoint,
                        method=method,
                        status_code=status_code,
                        duration=duration,
                        error_type=error_type
                    )

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200
                error_type = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = getattr(e, 'status_code', 500)
                    error_type = type(e).__name__
                    raise
                finally:
                    duration = time.time() - start_time
                    self.collector.track_api_call(
                        provider=provider,
                        endpoint=endpoint,
                        method=method,
                        status_code=status_code,
                        duration=duration,
                        error_type=error_type
                    )

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def track_strategy_execution(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str = "1D"
    ):
        """
        Decorator to automatically track strategy execution metrics.

        Usage:
            @middleware.track_strategy_execution("RSI_Strategy", "AAPL")
            def run_backtest(data):
                # Strategy execution logic here
                return results
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.collector.track_backtest_execution(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_points=kwargs.get('data_points', 0)
                ):
                    result = await func(*args, **kwargs)

                    # Extract performance metrics from result if available
                    if isinstance(result, dict):
                        self._extract_and_track_performance(
                            result, strategy_name, symbol, timeframe
                        )

                    return result

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.collector.track_backtest_execution(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_points=kwargs.get('data_points', 0)
                ):
                    result = func(*args, **kwargs)

                    # Extract performance metrics from result if available
                    if isinstance(result, dict):
                        self._extract_and_track_performance(
                            result, strategy_name, symbol, timeframe
                        )

                    return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def track_resource_usage(self, operation_type: str):
        """
        Decorator to automatically track resource usage for operations.

        Usage:
            @middleware.track_resource_usage("vectorbt_backtest")
            def run_vectorbt_analysis(data):
                # VectorBT analysis logic here
                pass
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_memory = process.memory_info().rss / 1024 / 1024
                    duration = time.time() - start_time
                    memory_used = max(0, end_memory - start_memory)

                    # Determine data size category
                    data_size = "unknown"
                    if 'data' in kwargs:
                        data_length = len(kwargs['data']) if hasattr(kwargs['data'], '__len__') else 0
                        data_size = self.collector._categorize_data_size(data_length)

                    self.collector.track_resource_usage(
                        operation_type=operation_type,
                        memory_mb=memory_used,
                        computation_time=duration,
                        data_size=data_size
                    )

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_memory = process.memory_info().rss / 1024 / 1024
                    duration = time.time() - start_time
                    memory_used = max(0, end_memory - start_memory)

                    # Determine data size category
                    data_size = "unknown"
                    if 'data' in kwargs:
                        data_length = len(kwargs['data']) if hasattr(kwargs['data'], '__len__') else 0
                        data_size = self.collector._categorize_data_size(data_length)

                    self.collector.track_resource_usage(
                        operation_type=operation_type,
                        memory_mb=memory_used,
                        computation_time=duration,
                        data_size=data_size
                    )

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @asynccontextmanager
    async def track_database_operation(
        self,
        query_type: str,
        table_name: str,
        operation: str
    ):
        """
        Context manager to track database operation performance.

        Usage:
            async with middleware.track_database_operation("SELECT", "stocks", "fetch"):
                result = await db.execute(query)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.collector.track_database_operation(
                query_type=query_type,
                table_name=table_name,
                operation=operation,
                duration=duration
            )

    def _extract_and_track_performance(
        self,
        result: dict[str, Any],
        strategy_name: str,
        symbol: str,
        timeframe: str
    ):
        """Extract and track strategy performance metrics from results."""
        try:
            # Extract common performance metrics from result dictionary
            returns = result.get('total_return', result.get('returns', 0.0))
            sharpe_ratio = result.get('sharpe_ratio', 0.0)
            max_drawdown = result.get('max_drawdown', result.get('max_dd', 0.0))
            win_rate = result.get('win_rate', result.get('win_ratio', 0.0))
            total_trades = result.get('total_trades', result.get('num_trades', 0))
            winning_trades = result.get('winning_trades', 0)

            # Convert win rate to percentage if it's in decimal form
            if win_rate <= 1.0:
                win_rate *= 100

            # Convert max drawdown to positive percentage if negative
            if max_drawdown < 0:
                max_drawdown = abs(max_drawdown) * 100

            # Extract winning trades from win rate if not provided directly
            if winning_trades == 0 and total_trades > 0:
                winning_trades = int(total_trades * (win_rate / 100))

            # Determine period from timeframe or use default
            period_mapping = {
                '1D': '1Y', '1H': '3M', '5m': '1M', '1m': '1W'
            }
            period = period_mapping.get(timeframe, '1Y')

            # Track the performance metrics
            self.collector.track_strategy_performance(
                strategy_name=strategy_name,
                symbol=symbol,
                period=period,
                returns=returns,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades
            )

            self.logger.debug(
                f"Tracked strategy performance for {strategy_name}",
                extra={
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'returns': returns,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': total_trades
                }
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to extract performance metrics from result: {e}",
                extra={'result_keys': list(result.keys()) if isinstance(result, dict) else 'not_dict'}
            )


# Global middleware instance
_middleware_instance: MetricsMiddleware | None = None


def get_metrics_middleware() -> MetricsMiddleware:
    """Get or create the global metrics middleware instance."""
    global _middleware_instance
    if _middleware_instance is None:
        _middleware_instance = MetricsMiddleware()
    return _middleware_instance


# Convenience decorators using global middleware instance
def track_api_call(provider: str, endpoint: str, method: str = "GET"):
    """Convenience decorator for API call tracking."""
    return get_metrics_middleware().track_api_call(provider, endpoint, method)


def track_strategy_execution(strategy_name: str, symbol: str, timeframe: str = "1D"):
    """Convenience decorator for strategy execution tracking."""
    return get_metrics_middleware().track_strategy_execution(strategy_name, symbol, timeframe)


def track_resource_usage(operation_type: str):
    """Convenience decorator for resource usage tracking."""
    return get_metrics_middleware().track_resource_usage(operation_type)


def track_database_operation(query_type: str, table_name: str, operation: str):
    """Convenience context manager for database operation tracking."""
    return get_metrics_middleware().track_database_operation(query_type, table_name, operation)


# Example circuit breaker with metrics
class MetricsCircuitBreaker:
    """
    Circuit breaker with integrated metrics tracking.

    Automatically tracks circuit breaker state changes and failures.
    """

    def __init__(
        self,
        provider: str,
        endpoint: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.provider = provider
        self.endpoint = endpoint
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open

        self.collector = get_backtesting_metrics()
        self.logger = get_logger(f"{__name__}.MetricsCircuitBreaker")

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection and metrics tracking."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                self.collector.track_circuit_breaker(
                    self.provider, self.endpoint, self.state, 0
                )
            else:
                raise Exception(f"Circuit breaker is open for {self.provider}/{self.endpoint}")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset failure count and close circuit if half-open
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                self.collector.track_circuit_breaker(
                    self.provider, self.endpoint, self.state, 0
                )
                self.logger.info(f"Circuit breaker closed for {self.provider}/{self.endpoint}")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            # Track failure
            self.collector.track_circuit_breaker(
                self.provider, self.endpoint, self.state, 1
            )

            # Open circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                self.collector.track_circuit_breaker(
                    self.provider, self.endpoint, self.state, 0
                )
                self.logger.warning(
                    f"Circuit breaker opened for {self.provider}/{self.endpoint} "
                    f"after {self.failure_count} failures"
                )

            raise e
