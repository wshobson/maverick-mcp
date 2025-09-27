"""
Example integration of monitoring metrics into backtesting code.

This module demonstrates how to integrate the monitoring system
into existing backtesting strategies and data providers.
"""

import asyncio
from typing import Any

import numpy as np
import pandas as pd

from maverick_mcp.monitoring.metrics import get_backtesting_metrics
from maverick_mcp.monitoring.middleware import (
    MetricsCircuitBreaker,
    get_metrics_middleware,
    track_api_call,
    track_resource_usage,
    track_strategy_execution,
)
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class MonitoredStockDataProvider:
    """
    Example stock data provider with integrated monitoring.

    Shows how to add metrics tracking to data fetching operations.
    """

    def __init__(self):
        self.circuit_breaker = MetricsCircuitBreaker(
            provider="tiingo",
            endpoint="/daily",
            failure_threshold=5,
            recovery_timeout=60,
        )
        self.logger = get_logger(f"{__name__}.MonitoredStockDataProvider")

    @track_api_call("tiingo", "/daily/{symbol}")
    async def get_stock_data(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch stock data with automatic API call tracking.

        The @track_api_call decorator automatically tracks:
        - Request duration
        - Success/failure status
        - Rate limiting metrics
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)

        # Simulate occasional API errors for demonstration
        if np.random.random() < 0.05:  # 5% error rate
            raise Exception("API rate limit exceeded")

        # Generate sample data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.random.uniform(100, 200, len(dates)),
                "High": np.random.uniform(150, 250, len(dates)),
                "Low": np.random.uniform(50, 150, len(dates)),
                "Close": np.random.uniform(100, 200, len(dates)),
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            }
        )

        # Track additional metrics
        collector = get_backtesting_metrics()
        collector.track_cache_operation(
            cache_type="api_response",
            operation="fetch",
            hit=False,  # Assume cache miss for this example
            key_pattern=f"stock_data_{symbol}",
        )

        return data

    async def get_stock_data_with_circuit_breaker(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock data with circuit breaker protection.

        Automatically tracks circuit breaker state changes and failures.
        """
        return await self.circuit_breaker.call(self.get_stock_data, symbol=symbol)


class MonitoredTradingStrategy:
    """
    Example trading strategy with comprehensive monitoring.

    Shows how to add metrics tracking to strategy execution.
    """

    def __init__(self, name: str):
        self.name = name
        self.data_provider = MonitoredStockDataProvider()
        self.middleware = get_metrics_middleware()
        self.logger = get_logger(f"{__name__}.MonitoredTradingStrategy")

    @track_strategy_execution("RSI_Strategy", "AAPL", "1D")
    async def run_backtest(
        self, symbol: str, data: pd.DataFrame = None, data_points: int = None
    ) -> dict[str, Any]:
        """
        Run backtest with automatic strategy execution tracking.

        The @track_strategy_execution decorator automatically tracks:
        - Execution duration
        - Memory usage
        - Success/failure status
        - Data points processed
        """
        if data is None:
            data = await self.data_provider.get_stock_data(symbol)

        # Track the actual data points being processed
        actual_data_points = len(data)

        # Simulate strategy calculations
        await self._calculate_rsi_signals(data)

        # Simulate backtest execution
        performance_results = await self._simulate_trading(data, symbol)

        # Add data points info to results for metrics tracking
        performance_results["data_points_processed"] = actual_data_points

        return performance_results

    @track_resource_usage("rsi_calculation")
    async def _calculate_rsi_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI signals with resource usage tracking.

        The @track_resource_usage decorator tracks:
        - Memory usage during calculation
        - Computation time
        - Data size category
        """
        # Simulate RSI calculation (simplified)
        await asyncio.sleep(0.05)  # Simulate computation time

        # Calculate RSI (simplified version)
        data["rsi"] = np.random.uniform(20, 80, len(data))
        data["signal"] = np.where(
            data["rsi"] < 30, 1, np.where(data["rsi"] > 70, -1, 0)
        )

        return data

    async def _simulate_trading(
        self, data: pd.DataFrame, symbol: str
    ) -> dict[str, Any]:
        """
        Simulate trading and calculate performance metrics.

        Returns comprehensive performance metrics that will be
        automatically tracked by the strategy execution decorator.
        """
        # Simulate trading logic
        signals = data["signal"]

        # Calculate returns (simplified)
        total_return = np.random.uniform(-10, 30)  # Random return between -10% and 30%
        sharpe_ratio = np.random.uniform(-0.5, 2.5)  # Random Sharpe ratio
        max_drawdown = np.random.uniform(5, 25)  # Random drawdown 5-25%
        win_rate = np.random.uniform(0.35, 0.75)  # Random win rate 35-75%

        # Count trades
        position_changes = np.diff(signals)
        total_trades = np.sum(np.abs(position_changes))
        winning_trades = int(total_trades * win_rate)

        # Track portfolio updates
        collector = get_backtesting_metrics()
        collector.update_portfolio_metrics(
            portfolio_id="demo_portfolio",
            portfolio_value_usd=100000 * (1 + total_return / 100),
            daily_pnl_usd=total_return * 1000,  # Simulated daily PnL
            strategy=self.name,
            positions=[{"symbol": symbol, "quantity": 100, "type": "long"}],
        )

        # Return performance metrics in expected format
        return {
            "strategy_name": self.name,
            "symbol": symbol,
            "total_return": total_return,
            "returns": total_return,  # Alternative key name
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_dd": max_drawdown,  # Alternative key name
            "win_rate": win_rate * 100,  # Convert to percentage
            "win_ratio": win_rate,  # Alternative key name
            "total_trades": int(total_trades),
            "num_trades": int(total_trades),  # Alternative key name
            "winning_trades": winning_trades,
            "performance_summary": {
                "profitable": total_return > 0,
                "risk_adjusted_return": sharpe_ratio,
                "maximum_loss": max_drawdown,
            },
        }


class MonitoredDatabaseRepository:
    """
    Example database repository with monitoring integration.

    Shows how to add database operation tracking.
    """

    def __init__(self):
        self.middleware = get_metrics_middleware()
        self.logger = get_logger(f"{__name__}.MonitoredDatabaseRepository")

    async def save_backtest_results(
        self, strategy_name: str, symbol: str, results: dict[str, Any]
    ) -> bool:
        """
        Save backtest results with database operation tracking.
        """
        async with self.middleware.track_database_operation(
            query_type="INSERT", table_name="backtest_results", operation="save_results"
        ):
            # Simulate database save operation
            await asyncio.sleep(0.02)

            # Simulate occasional database errors
            if np.random.random() < 0.01:  # 1% error rate
                raise Exception("Database connection timeout")

            self.logger.info(
                f"Saved backtest results for {strategy_name} on {symbol}",
                extra={
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "total_return": results.get("total_return", 0),
                },
            )

            return True

    async def get_historical_performance(
        self, strategy_name: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """
        Retrieve historical performance with tracking.
        """
        async with self.middleware.track_database_operation(
            query_type="SELECT",
            table_name="backtest_results",
            operation="get_performance",
        ):
            # Simulate database query
            await asyncio.sleep(0.01)

            # Generate sample historical data
            historical_data = []
            for i in range(days):
                historical_data.append(
                    {
                        "date": f"2024-01-{i + 1:02d}",
                        "strategy": strategy_name,
                        "return": np.random.uniform(-2, 3),
                        "sharpe_ratio": np.random.uniform(0.5, 2.0),
                    }
                )

            return historical_data


# Demonstration function
async def demonstrate_monitoring_integration():
    """
    Demonstrate comprehensive monitoring integration.

    This function shows how all monitoring components work together
    in a typical backtesting workflow.
    """
    logger.info("Starting monitoring integration demonstration")

    # Initialize components
    strategy = MonitoredTradingStrategy("RSI_Momentum")
    repository = MonitoredDatabaseRepository()

    # List of symbols to test
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    for symbol in symbols:
        try:
            logger.info(f"Running backtest for {symbol}")

            # Run backtest (automatically tracked)
            results = await strategy.run_backtest(
                symbol=symbol,
                data_points=252,  # One year of trading days
            )

            # Save results (automatically tracked)
            await repository.save_backtest_results(
                strategy_name=strategy.name, symbol=symbol, results=results
            )

            # Get historical performance (automatically tracked)
            historical = await repository.get_historical_performance(
                strategy_name=strategy.name, days=30
            )

            logger.info(
                f"Completed backtest for {symbol}",
                extra={
                    "symbol": symbol,
                    "total_return": results.get("total_return", 0),
                    "sharpe_ratio": results.get("sharpe_ratio", 0),
                    "historical_records": len(historical),
                },
            )

        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")

            # Manually track the anomaly
            collector = get_backtesting_metrics()
            collector.detect_anomaly(
                anomaly_type="backtest_execution_failure",
                severity="critical",
                context={
                    "strategy_name": strategy.name,
                    "symbol": symbol,
                    "error": str(e),
                },
            )

    logger.info("Monitoring integration demonstration completed")


# Alert checking function
async def check_and_report_anomalies():
    """
    Example function to check for anomalies and generate alerts.

    This would typically be run periodically by a scheduler.
    """
    logger.info("Checking for performance anomalies")

    collector = get_backtesting_metrics()

    # Simulate checking various metrics and detecting anomalies
    anomalies_detected = 0

    # Example: Check if any strategy has poor recent performance
    strategies = ["RSI_Momentum", "MACD_Trend", "Bollinger_Bands"]

    for strategy in strategies:
        # Simulate performance check
        recent_sharpe = np.random.uniform(-1.0, 2.5)
        recent_drawdown = np.random.uniform(5, 35)

        if recent_sharpe < 0.5:
            collector.detect_anomaly(
                anomaly_type="low_sharpe_ratio",
                severity="warning" if recent_sharpe > 0 else "critical",
                context={
                    "strategy_name": strategy,
                    "sharpe_ratio": recent_sharpe,
                    "threshold": 0.5,
                },
            )
            anomalies_detected += 1

        if recent_drawdown > 25:
            collector.detect_anomaly(
                anomaly_type="high_drawdown",
                severity="critical",
                context={
                    "strategy_name": strategy,
                    "max_drawdown": recent_drawdown,
                    "threshold": 25,
                },
            )
            anomalies_detected += 1

    logger.info(f"Anomaly check completed. Detected {anomalies_detected} anomalies")

    return anomalies_detected


if __name__ == "__main__":
    # Run the demonstration
    async def main():
        await demonstrate_monitoring_integration()
        await check_and_report_anomalies()

        # Print metrics summary
        collector = get_backtesting_metrics()
        metrics_text = collector.get_metrics_text()
        print("\n" + "=" * 50)
        print("PROMETHEUS METRICS SAMPLE:")
        print("=" * 50)
        print(metrics_text[:1000] + "..." if len(metrics_text) > 1000 else metrics_text)

    asyncio.run(main())
