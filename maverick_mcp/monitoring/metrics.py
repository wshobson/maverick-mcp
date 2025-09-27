"""
Comprehensive Prometheus metrics for MaverickMCP backtesting system.

This module provides specialized metrics for monitoring:
- Backtesting execution performance and reliability
- Strategy performance over time
- API rate limiting and failure tracking
- Resource usage and optimization
- Anomaly detection and alerting
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Custom registry for backtesting metrics to avoid conflicts
BACKTESTING_REGISTRY = CollectorRegistry()

# =============================================================================
# BACKTESTING EXECUTION METRICS
# =============================================================================

# Backtest execution counters
backtest_executions_total = Counter(
    "maverick_backtest_executions_total",
    "Total number of backtesting executions",
    ["strategy_name", "status", "symbol", "timeframe"],
    registry=BACKTESTING_REGISTRY,
)

backtest_execution_duration = Histogram(
    "maverick_backtest_execution_duration_seconds",
    "Duration of backtesting executions in seconds",
    ["strategy_name", "symbol", "timeframe", "data_size"],
    buckets=(
        0.1,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
        float("inf"),
    ),
    registry=BACKTESTING_REGISTRY,
)

backtest_data_points_processed = Counter(
    "maverick_backtest_data_points_processed_total",
    "Total number of data points processed during backtesting",
    ["strategy_name", "symbol", "timeframe"],
    registry=BACKTESTING_REGISTRY,
)

backtest_memory_usage = Histogram(
    "maverick_backtest_memory_usage_mb",
    "Memory usage during backtesting in MB",
    ["strategy_name", "symbol", "complexity"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

# =============================================================================
# STRATEGY PERFORMANCE METRICS
# =============================================================================

# Strategy returns and performance
strategy_returns = Histogram(
    "maverick_strategy_returns_percent",
    "Strategy returns in percentage",
    ["strategy_name", "symbol", "period"],
    buckets=(-50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

strategy_sharpe_ratio = Histogram(
    "maverick_strategy_sharpe_ratio",
    "Strategy Sharpe ratio",
    ["strategy_name", "symbol", "period"],
    buckets=(-2, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

strategy_max_drawdown = Histogram(
    "maverick_strategy_max_drawdown_percent",
    "Maximum drawdown percentage for strategy",
    ["strategy_name", "symbol", "period"],
    buckets=(0, 5, 10, 15, 20, 30, 40, 50, 75, 100, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

strategy_win_rate = Histogram(
    "maverick_strategy_win_rate_percent",
    "Win rate percentage for strategy",
    ["strategy_name", "symbol", "period"],
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    registry=BACKTESTING_REGISTRY,
)

strategy_trades_total = Counter(
    "maverick_strategy_trades_total",
    "Total number of trades executed by strategy",
    ["strategy_name", "symbol", "trade_type", "outcome"],
    registry=BACKTESTING_REGISTRY,
)

# Strategy execution latency
strategy_execution_latency = Summary(
    "maverick_strategy_execution_latency_seconds",
    "Strategy execution latency for signal generation",
    ["strategy_name", "complexity"],
    registry=BACKTESTING_REGISTRY,
)

# =============================================================================
# API RATE LIMITING AND FAILURE METRICS
# =============================================================================

# API call tracking
api_calls_total = Counter(
    "maverick_api_calls_total",
    "Total API calls made to external providers",
    ["provider", "endpoint", "method", "status_code"],
    registry=BACKTESTING_REGISTRY,
)

api_call_duration = Histogram(
    "maverick_api_call_duration_seconds",
    "API call duration in seconds",
    ["provider", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

# Rate limiting metrics
rate_limit_hits = Counter(
    "maverick_rate_limit_hits_total",
    "Total rate limit hits by provider",
    ["provider", "endpoint", "limit_type"],
    registry=BACKTESTING_REGISTRY,
)

rate_limit_remaining = Gauge(
    "maverick_rate_limit_remaining",
    "Remaining API calls before hitting rate limit",
    ["provider", "endpoint", "window"],
    registry=BACKTESTING_REGISTRY,
)

rate_limit_reset_time = Gauge(
    "maverick_rate_limit_reset_timestamp",
    "Timestamp when rate limit resets",
    ["provider", "endpoint"],
    registry=BACKTESTING_REGISTRY,
)

# API failures and errors
api_failures_total = Counter(
    "maverick_api_failures_total",
    "Total API failures by error type",
    ["provider", "endpoint", "error_type", "error_code"],
    registry=BACKTESTING_REGISTRY,
)

api_retry_attempts = Counter(
    "maverick_api_retry_attempts_total",
    "Total API retry attempts",
    ["provider", "endpoint", "retry_number"],
    registry=BACKTESTING_REGISTRY,
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "maverick_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["provider", "endpoint"],
    registry=BACKTESTING_REGISTRY,
)

circuit_breaker_failures = Counter(
    "maverick_circuit_breaker_failures_total",
    "Circuit breaker failure count",
    ["provider", "endpoint"],
    registry=BACKTESTING_REGISTRY,
)

# =============================================================================
# RESOURCE USAGE AND PERFORMANCE METRICS
# =============================================================================

# VectorBT specific metrics
vectorbt_memory_usage = Gauge(
    "maverick_vectorbt_memory_usage_mb",
    "VectorBT memory usage in MB",
    ["operation_type"],
    registry=BACKTESTING_REGISTRY,
)

vectorbt_computation_time = Histogram(
    "maverick_vectorbt_computation_time_seconds",
    "VectorBT computation time in seconds",
    ["operation_type", "data_size"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

# Database query performance
database_query_duration = Histogram(
    "maverick_database_query_duration_seconds",
    "Database query execution time",
    ["query_type", "table_name", "operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")),
    registry=BACKTESTING_REGISTRY,
)

database_connection_pool_usage = Gauge(
    "maverick_database_connection_pool_usage",
    "Database connection pool usage",
    ["pool_type", "status"],
    registry=BACKTESTING_REGISTRY,
)

# Cache performance metrics
cache_operations_total = Counter(
    "maverick_cache_operations_total",
    "Total cache operations",
    ["cache_type", "operation", "status"],
    registry=BACKTESTING_REGISTRY,
)

cache_hit_ratio = Histogram(
    "maverick_cache_hit_ratio",
    "Cache hit ratio percentage",
    ["cache_type", "key_pattern"],
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100),
    registry=BACKTESTING_REGISTRY,
)

# =============================================================================
# ANOMALY DETECTION METRICS
# =============================================================================

# Performance anomaly detection
strategy_performance_anomalies = Counter(
    "maverick_strategy_performance_anomalies_total",
    "Detected strategy performance anomalies",
    ["strategy_name", "anomaly_type", "severity"],
    registry=BACKTESTING_REGISTRY,
)

data_quality_issues = Counter(
    "maverick_data_quality_issues_total",
    "Data quality issues detected",
    ["data_source", "issue_type", "symbol"],
    registry=BACKTESTING_REGISTRY,
)

resource_usage_alerts = Counter(
    "maverick_resource_usage_alerts_total",
    "Resource usage threshold alerts",
    ["resource_type", "threshold_type"],
    registry=BACKTESTING_REGISTRY,
)

# Threshold monitoring gauges
performance_thresholds = Gauge(
    "maverick_performance_thresholds",
    "Performance monitoring thresholds",
    ["metric_name", "threshold_type"],  # threshold_type: warning, critical
    registry=BACKTESTING_REGISTRY,
)

# =============================================================================
# BUSINESS METRICS FOR TRADING
# =============================================================================

# Portfolio performance metrics
portfolio_value = Gauge(
    "maverick_portfolio_value_usd",
    "Current portfolio value in USD",
    ["portfolio_id", "currency"],
    registry=BACKTESTING_REGISTRY,
)

portfolio_daily_pnl = Histogram(
    "maverick_portfolio_daily_pnl_usd",
    "Daily PnL in USD",
    ["portfolio_id", "strategy"],
    buckets=(
        -10000,
        -5000,
        -1000,
        -500,
        -100,
        0,
        100,
        500,
        1000,
        5000,
        10000,
        float("inf"),
    ),
    registry=BACKTESTING_REGISTRY,
)

active_positions = Gauge(
    "maverick_active_positions_count",
    "Number of active positions",
    ["portfolio_id", "symbol", "position_type"],
    registry=BACKTESTING_REGISTRY,
)

# =============================================================================
# METRICS COLLECTOR CLASS
# =============================================================================


@dataclass
class PerformanceThreshold:
    """Configuration for performance thresholds."""

    metric_name: str
    warning_value: float
    critical_value: float
    comparison_type: str = "greater_than"  # greater_than, less_than, equal_to


class BacktestingMetricsCollector:
    """
    Comprehensive metrics collector for backtesting operations.

    Provides high-level interfaces for tracking backtesting performance,
    strategy metrics, API usage, and anomaly detection.
    """

    def __init__(self):
        self.logger = get_logger(f"{__name__}.BacktestingMetricsCollector")
        self._anomaly_thresholds = self._initialize_default_thresholds()
        self._lock = threading.Lock()

        # Initialize performance thresholds in Prometheus
        self._setup_performance_thresholds()

        self.logger.info("Backtesting metrics collector initialized")

    def _initialize_default_thresholds(self) -> dict[str, PerformanceThreshold]:
        """Initialize default performance thresholds for anomaly detection."""
        return {
            "sharpe_ratio_low": PerformanceThreshold(
                "sharpe_ratio", 0.5, 0.0, "less_than"
            ),
            "max_drawdown_high": PerformanceThreshold(
                "max_drawdown", 20.0, 30.0, "greater_than"
            ),
            "win_rate_low": PerformanceThreshold("win_rate", 40.0, 30.0, "less_than"),
            "execution_time_high": PerformanceThreshold(
                "execution_time", 60.0, 120.0, "greater_than"
            ),
            "api_failure_rate_high": PerformanceThreshold(
                "api_failure_rate", 5.0, 10.0, "greater_than"
            ),
            "memory_usage_high": PerformanceThreshold(
                "memory_usage", 1000, 2000, "greater_than"
            ),
        }

    def _setup_performance_thresholds(self):
        """Setup performance threshold gauges."""
        for _threshold_name, threshold in self._anomaly_thresholds.items():
            performance_thresholds.labels(
                metric_name=threshold.metric_name, threshold_type="warning"
            ).set(threshold.warning_value)

            performance_thresholds.labels(
                metric_name=threshold.metric_name, threshold_type="critical"
            ).set(threshold.critical_value)

    @contextmanager
    def track_backtest_execution(
        self, strategy_name: str, symbol: str, timeframe: str, data_points: int = 0
    ):
        """
        Context manager for tracking backtest execution metrics.

        Args:
            strategy_name: Name of the trading strategy
            symbol: Trading symbol (e.g., 'AAPL')
            timeframe: Data timeframe (e.g., '1D', '1H')
            data_points: Number of data points being processed
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Determine data size category
        data_size = self._categorize_data_size(data_points)

        try:
            yield

            # Success metrics
            duration = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory

            backtest_executions_total.labels(
                strategy_name=strategy_name,
                status="success",
                symbol=symbol,
                timeframe=timeframe,
            ).inc()

            backtest_execution_duration.labels(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                data_size=data_size,
            ).observe(duration)

            if data_points > 0:
                backtest_data_points_processed.labels(
                    strategy_name=strategy_name, symbol=symbol, timeframe=timeframe
                ).inc(data_points)

            if memory_used > 0:
                complexity = self._categorize_complexity(data_points, duration)
                backtest_memory_usage.labels(
                    strategy_name=strategy_name, symbol=symbol, complexity=complexity
                ).observe(memory_used)

            # Check for performance anomalies
            self._check_execution_anomalies(strategy_name, duration, memory_used)

        except Exception as e:
            # Error metrics
            backtest_executions_total.labels(
                strategy_name=strategy_name,
                status="failure",
                symbol=symbol,
                timeframe=timeframe,
            ).inc()

            self.logger.error(f"Backtest execution failed for {strategy_name}: {e}")
            raise

    def track_strategy_performance(
        self,
        strategy_name: str,
        symbol: str,
        period: str,
        returns: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
        winning_trades: int,
    ):
        """
        Track comprehensive strategy performance metrics.

        Args:
            strategy_name: Name of the trading strategy
            symbol: Trading symbol
            period: Performance period (e.g., '1Y', '6M', '3M')
            returns: Total returns in percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            win_rate: Win rate percentage
            total_trades: Total number of trades
            winning_trades: Number of winning trades
        """
        # Record performance metrics
        strategy_returns.labels(
            strategy_name=strategy_name, symbol=symbol, period=period
        ).observe(returns)

        strategy_sharpe_ratio.labels(
            strategy_name=strategy_name, symbol=symbol, period=period
        ).observe(sharpe_ratio)

        strategy_max_drawdown.labels(
            strategy_name=strategy_name, symbol=symbol, period=period
        ).observe(max_drawdown)

        strategy_win_rate.labels(
            strategy_name=strategy_name, symbol=symbol, period=period
        ).observe(win_rate)

        # Record trade counts
        strategy_trades_total.labels(
            strategy_name=strategy_name,
            symbol=symbol,
            trade_type="total",
            outcome="all",
        ).inc(total_trades)

        strategy_trades_total.labels(
            strategy_name=strategy_name,
            symbol=symbol,
            trade_type="winning",
            outcome="profit",
        ).inc(winning_trades)

        losing_trades = total_trades - winning_trades
        strategy_trades_total.labels(
            strategy_name=strategy_name,
            symbol=symbol,
            trade_type="losing",
            outcome="loss",
        ).inc(losing_trades)

        # Check for performance anomalies
        self._check_strategy_anomalies(
            strategy_name, sharpe_ratio, max_drawdown, win_rate
        )

    def track_api_call(
        self,
        provider: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        error_type: str | None = None,
        remaining_calls: int | None = None,
        reset_time: datetime | None = None,
    ):
        """
        Track API call metrics including rate limiting and failures.

        Args:
            provider: API provider name (e.g., 'tiingo', 'yahoo')
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
            error_type: Type of error if request failed
            remaining_calls: Remaining API calls before rate limit
            reset_time: When rate limit resets
        """
        # Basic API call tracking
        api_calls_total.labels(
            provider=provider,
            endpoint=endpoint,
            method=method,
            status_code=str(status_code),
        ).inc()

        api_call_duration.labels(provider=provider, endpoint=endpoint).observe(duration)

        # Rate limiting metrics
        if remaining_calls is not None:
            rate_limit_remaining.labels(
                provider=provider, endpoint=endpoint, window="current"
            ).set(remaining_calls)

        if reset_time is not None:
            rate_limit_reset_time.labels(provider=provider, endpoint=endpoint).set(
                reset_time.timestamp()
            )

        # Failure tracking
        if status_code >= 400:
            error_code = self._categorize_error_code(status_code)
            api_failures_total.labels(
                provider=provider,
                endpoint=endpoint,
                error_type=error_type or "unknown",
                error_code=error_code,
            ).inc()

            # Check for rate limiting
            if status_code == 429:
                rate_limit_hits.labels(
                    provider=provider, endpoint=endpoint, limit_type="requests_per_hour"
                ).inc()

        # Check for API anomalies
        self._check_api_anomalies(provider, endpoint, status_code, duration)

    def track_circuit_breaker(
        self, provider: str, endpoint: str, state: str, failure_count: int
    ):
        """Track circuit breaker state and failures."""
        state_mapping = {"closed": 0, "open": 1, "half-open": 2}
        circuit_breaker_state.labels(provider=provider, endpoint=endpoint).set(
            state_mapping.get(state, 0)
        )

        if failure_count > 0:
            circuit_breaker_failures.labels(provider=provider, endpoint=endpoint).inc(
                failure_count
            )

    def track_resource_usage(
        self,
        operation_type: str,
        memory_mb: float,
        computation_time: float,
        data_size: str = "unknown",
    ):
        """Track resource usage for VectorBT operations."""
        vectorbt_memory_usage.labels(operation_type=operation_type).set(memory_mb)

        vectorbt_computation_time.labels(
            operation_type=operation_type, data_size=data_size
        ).observe(computation_time)

        # Check for resource usage anomalies
        if memory_mb > self._anomaly_thresholds["memory_usage_high"].warning_value:
            resource_usage_alerts.labels(
                resource_type="memory",
                threshold_type="warning"
                if memory_mb
                < self._anomaly_thresholds["memory_usage_high"].critical_value
                else "critical",
            ).inc()

    def track_database_operation(
        self, query_type: str, table_name: str, operation: str, duration: float
    ):
        """Track database operation performance."""
        database_query_duration.labels(
            query_type=query_type, table_name=table_name, operation=operation
        ).observe(duration)

    def track_cache_operation(
        self, cache_type: str, operation: str, hit: bool, key_pattern: str = "general"
    ):
        """Track cache operation performance."""
        status = "hit" if hit else "miss"
        cache_operations_total.labels(
            cache_type=cache_type, operation=operation, status=status
        ).inc()

    def detect_anomaly(self, anomaly_type: str, severity: str, context: dict[str, Any]):
        """Record detected anomaly."""
        strategy_name = context.get("strategy_name", "unknown")

        strategy_performance_anomalies.labels(
            strategy_name=strategy_name, anomaly_type=anomaly_type, severity=severity
        ).inc()

        self.logger.warning(
            f"Anomaly detected: {anomaly_type} (severity: {severity})",
            extra={"context": context},
        )

    def update_portfolio_metrics(
        self,
        portfolio_id: str,
        portfolio_value_usd: float,
        daily_pnl_usd: float,
        strategy: str,
        positions: list[dict[str, Any]],
    ):
        """Update portfolio-related metrics."""
        portfolio_value.labels(portfolio_id=portfolio_id, currency="USD").set(
            portfolio_value_usd
        )

        portfolio_daily_pnl.labels(
            portfolio_id=portfolio_id, strategy=strategy
        ).observe(daily_pnl_usd)

        # Update position counts
        for position in positions:
            active_positions.labels(
                portfolio_id=portfolio_id,
                symbol=position.get("symbol", "unknown"),
                position_type=position.get("type", "long"),
            ).set(position.get("quantity", 0))

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _categorize_data_size(self, data_points: int) -> str:
        """Categorize data size for metrics labeling."""
        if data_points < 100:
            return "small"
        elif data_points < 1000:
            return "medium"
        elif data_points < 10000:
            return "large"
        else:
            return "xlarge"

    def _categorize_complexity(self, data_points: int, duration: float) -> str:
        """Categorize operation complexity."""
        if data_points < 1000 and duration < 10:
            return "simple"
        elif data_points < 10000 and duration < 60:
            return "moderate"
        else:
            return "complex"

    def _categorize_error_code(self, status_code: int) -> str:
        """Categorize HTTP error codes."""
        if 400 <= status_code < 500:
            return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "other"

    def _check_execution_anomalies(
        self, strategy_name: str, duration: float, memory_mb: float
    ):
        """Check for execution performance anomalies."""
        threshold = self._anomaly_thresholds["execution_time_high"]
        if duration > threshold.critical_value:
            self.detect_anomaly(
                "execution_time_high",
                "critical",
                {
                    "strategy_name": strategy_name,
                    "duration": duration,
                    "threshold": threshold.critical_value,
                },
            )
        elif duration > threshold.warning_value:
            self.detect_anomaly(
                "execution_time_high",
                "warning",
                {
                    "strategy_name": strategy_name,
                    "duration": duration,
                    "threshold": threshold.warning_value,
                },
            )

    def _check_strategy_anomalies(
        self,
        strategy_name: str,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
    ):
        """Check for strategy performance anomalies."""
        # Check Sharpe ratio
        threshold = self._anomaly_thresholds["sharpe_ratio_low"]
        if sharpe_ratio < threshold.critical_value:
            self.detect_anomaly(
                "sharpe_ratio_low",
                "critical",
                {"strategy_name": strategy_name, "sharpe_ratio": sharpe_ratio},
            )
        elif sharpe_ratio < threshold.warning_value:
            self.detect_anomaly(
                "sharpe_ratio_low",
                "warning",
                {"strategy_name": strategy_name, "sharpe_ratio": sharpe_ratio},
            )

        # Check max drawdown
        threshold = self._anomaly_thresholds["max_drawdown_high"]
        if max_drawdown > threshold.critical_value:
            self.detect_anomaly(
                "max_drawdown_high",
                "critical",
                {"strategy_name": strategy_name, "max_drawdown": max_drawdown},
            )
        elif max_drawdown > threshold.warning_value:
            self.detect_anomaly(
                "max_drawdown_high",
                "warning",
                {"strategy_name": strategy_name, "max_drawdown": max_drawdown},
            )

        # Check win rate
        threshold = self._anomaly_thresholds["win_rate_low"]
        if win_rate < threshold.critical_value:
            self.detect_anomaly(
                "win_rate_low",
                "critical",
                {"strategy_name": strategy_name, "win_rate": win_rate},
            )
        elif win_rate < threshold.warning_value:
            self.detect_anomaly(
                "win_rate_low",
                "warning",
                {"strategy_name": strategy_name, "win_rate": win_rate},
            )

    def _check_api_anomalies(
        self, provider: str, endpoint: str, status_code: int, duration: float
    ):
        """Check for API call anomalies."""
        # Check API response time
        if duration > 30.0:  # 30 second threshold
            self.detect_anomaly(
                "api_response_slow",
                "warning" if duration < 60.0 else "critical",
                {"provider": provider, "endpoint": endpoint, "duration": duration},
            )

        # Check for repeated failures
        if status_code >= 500:
            self.detect_anomaly(
                "api_server_error",
                "critical",
                {
                    "provider": provider,
                    "endpoint": endpoint,
                    "status_code": status_code,
                },
            )

    def get_metrics_text(self) -> str:
        """Get all backtesting metrics in Prometheus text format."""
        return generate_latest(BACKTESTING_REGISTRY).decode("utf-8")


# =============================================================================
# GLOBAL INSTANCES AND CONVENIENCE FUNCTIONS
# =============================================================================

# Global metrics collector instance
_metrics_collector: BacktestingMetricsCollector | None = None
_collector_lock = threading.Lock()


def get_backtesting_metrics() -> BacktestingMetricsCollector:
    """Get or create the global backtesting metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = BacktestingMetricsCollector()
    return _metrics_collector


# Convenience functions for common operations
def track_backtest_execution(
    strategy_name: str, symbol: str, timeframe: str, data_points: int = 0
):
    """Convenience function to track backtest execution."""
    return get_backtesting_metrics().track_backtest_execution(
        strategy_name, symbol, timeframe, data_points
    )


def track_strategy_performance(
    strategy_name: str,
    symbol: str,
    period: str,
    returns: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    total_trades: int,
    winning_trades: int,
):
    """Convenience function to track strategy performance."""
    get_backtesting_metrics().track_strategy_performance(
        strategy_name,
        symbol,
        period,
        returns,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        total_trades,
        winning_trades,
    )


def track_api_call_metrics(
    provider: str,
    endpoint: str,
    method: str,
    status_code: int,
    duration: float,
    error_type: str | None = None,
    remaining_calls: int | None = None,
    reset_time: datetime | None = None,
):
    """Convenience function to track API call metrics."""
    get_backtesting_metrics().track_api_call(
        provider,
        endpoint,
        method,
        status_code,
        duration,
        error_type,
        remaining_calls,
        reset_time,
    )


def track_anomaly_detection(anomaly_type: str, severity: str, context: dict[str, Any]):
    """Convenience function to track detected anomalies."""
    get_backtesting_metrics().detect_anomaly(anomaly_type, severity, context)


def get_metrics_for_prometheus() -> str:
    """Get backtesting metrics in Prometheus format."""
    return get_backtesting_metrics().get_metrics_text()
