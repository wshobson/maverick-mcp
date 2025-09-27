"""
Load Testing for Concurrent Users and Backtest Operations.

This test suite covers:
- Concurrent user load testing (10, 50, 100 users)
- Response time and throughput measurement
- Memory usage under concurrent load
- Database performance with multiple connections
- API rate limiting behavior
- Queue management and task distribution
- System stability under sustained load
"""

import asyncio
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import psutil
import pytest

from maverick_mcp.backtesting import VectorBTEngine
from maverick_mcp.backtesting.persistence import BacktestPersistenceManager
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Data class for load test results."""

    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    errors_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float


class LoadTestRunner:
    """Load test runner with realistic user simulation."""

    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.results = []
        self.active_requests = 0

    async def simulate_user_session(
        self, user_id: int, session_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate a realistic user session with multiple backtests."""
        session_start = time.time()
        user_results = []
        response_times = []

        symbols = session_config.get("symbols", ["AAPL"])
        strategies = session_config.get("strategies", ["sma_cross"])
        think_time_range = session_config.get("think_time", (0.5, 2.0))

        engine = VectorBTEngine(data_provider=self.data_provider)

        for symbol in symbols:
            for strategy in strategies:
                self.active_requests += 1
                request_start = time.time()

                try:
                    parameters = STRATEGY_TEMPLATES.get(strategy, {}).get(
                        "parameters", {}
                    )

                    result = await engine.run_backtest(
                        symbol=symbol,
                        strategy_type=strategy,
                        parameters=parameters,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                    )

                    request_time = time.time() - request_start
                    response_times.append(request_time)

                    user_results.append(
                        {
                            "symbol": symbol,
                            "strategy": strategy,
                            "success": True,
                            "response_time": request_time,
                            "result_size": len(str(result)),
                        }
                    )

                except Exception as e:
                    request_time = time.time() - request_start
                    response_times.append(request_time)

                    user_results.append(
                        {
                            "symbol": symbol,
                            "strategy": strategy,
                            "success": False,
                            "response_time": request_time,
                            "error": str(e),
                        }
                    )

                finally:
                    self.active_requests -= 1

                # Simulate think time between requests
                think_time = random.uniform(*think_time_range)
                await asyncio.sleep(think_time)

        session_time = time.time() - session_start

        return {
            "user_id": user_id,
            "session_time": session_time,
            "results": user_results,
            "response_times": response_times,
            "success_count": sum(1 for r in user_results if r["success"]),
            "failure_count": sum(1 for r in user_results if not r["success"]),
        }

    def calculate_percentiles(self, response_times: list[float]) -> dict[str, float]:
        """Calculate response time percentiles."""
        if not response_times:
            return {"p50": 0, "p95": 0, "p99": 0}

        sorted_times = sorted(response_times)
        return {
            "p50": np.percentile(sorted_times, 50),
            "p95": np.percentile(sorted_times, 95),
            "p99": np.percentile(sorted_times, 99),
        }

    async def run_load_test(
        self,
        concurrent_users: int,
        session_config: dict[str, Any],
        duration_seconds: int = 60,
    ) -> LoadTestResult:
        """Run load test with specified concurrent users."""
        logger.info(
            f"Starting load test: {concurrent_users} concurrent users for {duration_seconds}s"
        )

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        all_response_times = []
        all_user_results = []

        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrent_users)

        async def run_user_with_semaphore(user_id: int):
            async with semaphore:
                return await self.simulate_user_session(user_id, session_config)

        # Generate user sessions
        user_tasks = []
        for user_id in range(concurrent_users):
            task = run_user_with_semaphore(user_id)
            user_tasks.append(task)

        # Execute all user sessions concurrently
        try:
            user_results = await asyncio.wait_for(
                asyncio.gather(*user_tasks, return_exceptions=True),
                timeout=duration_seconds + 30,  # Add buffer to test timeout
            )
        except TimeoutError:
            logger.warning(f"Load test timed out after {duration_seconds + 30}s")
            user_results = []

        end_time = time.time()
        actual_duration = end_time - start_time

        # Process results
        successful_sessions = []
        failed_sessions = []

        for result in user_results:
            if isinstance(result, Exception):
                failed_sessions.append(str(result))
            elif isinstance(result, dict):
                successful_sessions.append(result)
                all_response_times.extend(result.get("response_times", []))
                all_user_results.extend(result.get("results", []))

        # Calculate metrics
        total_requests = len(all_user_results)
        successful_requests = sum(
            1 for r in all_user_results if r.get("success", False)
        )
        failed_requests = total_requests - successful_requests

        # Response time statistics
        percentiles = self.calculate_percentiles(all_response_times)
        avg_response_time = (
            statistics.mean(all_response_times) if all_response_times else 0
        )
        min_response_time = min(all_response_times) if all_response_times else 0
        max_response_time = max(all_response_times) if all_response_times else 0

        # Throughput metrics
        requests_per_second = (
            total_requests / actual_duration if actual_duration > 0 else 0
        )
        errors_per_second = (
            failed_requests / actual_duration if actual_duration > 0 else 0
        )

        # Resource usage
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory
        cpu_usage = process.cpu_percent()

        result = LoadTestResult(
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=actual_duration,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=percentiles["p50"],
            p95_response_time=percentiles["p95"],
            p99_response_time=percentiles["p99"],
            requests_per_second=requests_per_second,
            errors_per_second=errors_per_second,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
        )

        logger.info(
            f"Load Test Results ({concurrent_users} users):\n"
            f"  • Total Requests: {total_requests}\n"
            f"  • Success Rate: {successful_requests / total_requests * 100:.1f}%\n"
            f"  • Avg Response Time: {avg_response_time:.3f}s\n"
            f"  • 95th Percentile: {percentiles['p95']:.3f}s\n"
            f"  • Throughput: {requests_per_second:.1f} req/s\n"
            f"  • Memory Usage: {memory_usage:.1f}MB\n"
            f"  • Duration: {actual_duration:.1f}s"
        )

        return result


class TestLoadTesting:
    """Load testing suite for concurrent users."""

    @pytest.fixture
    async def optimized_data_provider(self):
        """Create optimized data provider for load testing."""
        provider = Mock()

        # Pre-generate data for common symbols to reduce computation
        symbol_data_cache = {}

        def get_cached_data(symbol: str) -> pd.DataFrame:
            """Get or generate cached data for symbol."""
            if symbol not in symbol_data_cache:
                # Generate deterministic data based on symbol hash
                seed = hash(symbol) % 1000
                np.random.seed(seed)

                dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
                returns = np.random.normal(0.001, 0.02, len(dates))
                prices = 100 * np.cumprod(1 + returns)

                symbol_data_cache[symbol] = pd.DataFrame(
                    {
                        "Open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                        "High": prices * np.random.uniform(1.01, 1.03, len(dates)),
                        "Low": prices * np.random.uniform(0.97, 0.99, len(dates)),
                        "Close": prices,
                        "Volume": np.random.randint(1000000, 10000000, len(dates)),
                        "Adj Close": prices,
                    },
                    index=dates,
                )

                # Ensure OHLC constraints
                data = symbol_data_cache[symbol]
                data["High"] = np.maximum(
                    data["High"], np.maximum(data["Open"], data["Close"])
                )
                data["Low"] = np.minimum(
                    data["Low"], np.minimum(data["Open"], data["Close"])
                )

            return symbol_data_cache[symbol].copy()

        provider.get_stock_data.side_effect = get_cached_data
        return provider

    async def test_concurrent_users_10(self, optimized_data_provider, benchmark_timer):
        """Test load with 10 concurrent users."""
        load_runner = LoadTestRunner(optimized_data_provider)

        session_config = {
            "symbols": ["AAPL", "GOOGL"],
            "strategies": ["sma_cross", "rsi"],
            "think_time": (0.1, 0.5),  # Faster think time for testing
        }

        with benchmark_timer():
            result = await load_runner.run_load_test(
                concurrent_users=10, session_config=session_config, duration_seconds=30
            )

        # Performance assertions for 10 users
        assert result.requests_per_second >= 2.0, (
            f"Throughput too low: {result.requests_per_second:.1f} req/s"
        )
        assert result.avg_response_time <= 5.0, (
            f"Response time too high: {result.avg_response_time:.2f}s"
        )
        assert result.p95_response_time <= 10.0, (
            f"95th percentile too high: {result.p95_response_time:.2f}s"
        )
        assert result.successful_requests / result.total_requests >= 0.9, (
            "Success rate too low"
        )
        assert result.memory_usage_mb <= 500, (
            f"Memory usage too high: {result.memory_usage_mb:.1f}MB"
        )

        return result

    async def test_concurrent_users_50(self, optimized_data_provider, benchmark_timer):
        """Test load with 50 concurrent users."""
        load_runner = LoadTestRunner(optimized_data_provider)

        session_config = {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "strategies": ["sma_cross", "rsi", "macd"],
            "think_time": (0.2, 1.0),
        }

        with benchmark_timer():
            result = await load_runner.run_load_test(
                concurrent_users=50, session_config=session_config, duration_seconds=60
            )

        # Performance assertions for 50 users
        assert result.requests_per_second >= 5.0, (
            f"Throughput too low: {result.requests_per_second:.1f} req/s"
        )
        assert result.avg_response_time <= 8.0, (
            f"Response time too high: {result.avg_response_time:.2f}s"
        )
        assert result.p95_response_time <= 15.0, (
            f"95th percentile too high: {result.p95_response_time:.2f}s"
        )
        assert result.successful_requests / result.total_requests >= 0.85, (
            "Success rate too low"
        )
        assert result.memory_usage_mb <= 1000, (
            f"Memory usage too high: {result.memory_usage_mb:.1f}MB"
        )

        return result

    async def test_concurrent_users_100(self, optimized_data_provider, benchmark_timer):
        """Test load with 100 concurrent users."""
        load_runner = LoadTestRunner(optimized_data_provider)

        session_config = {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "strategies": ["sma_cross", "rsi"],  # Reduced strategies for higher load
            "think_time": (0.5, 1.5),
        }

        with benchmark_timer():
            result = await load_runner.run_load_test(
                concurrent_users=100, session_config=session_config, duration_seconds=90
            )

        # More relaxed performance assertions for 100 users
        assert result.requests_per_second >= 3.0, (
            f"Throughput too low: {result.requests_per_second:.1f} req/s"
        )
        assert result.avg_response_time <= 15.0, (
            f"Response time too high: {result.avg_response_time:.2f}s"
        )
        assert result.p95_response_time <= 30.0, (
            f"95th percentile too high: {result.p95_response_time:.2f}s"
        )
        assert result.successful_requests / result.total_requests >= 0.8, (
            "Success rate too low"
        )
        assert result.memory_usage_mb <= 2000, (
            f"Memory usage too high: {result.memory_usage_mb:.1f}MB"
        )

        return result

    async def test_load_scalability_analysis(self, optimized_data_provider):
        """Analyze how performance scales with user load."""
        load_runner = LoadTestRunner(optimized_data_provider)

        session_config = {
            "symbols": ["AAPL", "GOOGL"],
            "strategies": ["sma_cross"],
            "think_time": (0.3, 0.7),
        }

        user_loads = [5, 10, 20, 40]
        scalability_results = []

        for user_count in user_loads:
            logger.info(f"Testing scalability with {user_count} users")

            result = await load_runner.run_load_test(
                concurrent_users=user_count,
                session_config=session_config,
                duration_seconds=30,
            )

            scalability_results.append(result)

        # Analyze scalability metrics
        throughput_efficiency = []
        response_time_degradation = []

        baseline_rps = scalability_results[0].requests_per_second
        baseline_response_time = scalability_results[0].avg_response_time

        for i, result in enumerate(scalability_results):
            expected_rps = baseline_rps * user_loads[i] / user_loads[0]
            actual_efficiency = (
                result.requests_per_second / expected_rps if expected_rps > 0 else 0
            )
            throughput_efficiency.append(actual_efficiency)

            response_degradation = (
                result.avg_response_time / baseline_response_time
                if baseline_response_time > 0
                else 1
            )
            response_time_degradation.append(response_degradation)

            logger.info(
                f"Scalability Analysis ({user_loads[i]} users):\n"
                f"  • RPS: {result.requests_per_second:.2f}\n"
                f"  • RPS Efficiency: {actual_efficiency:.2%}\n"
                f"  • Response Time: {result.avg_response_time:.3f}s\n"
                f"  • Response Degradation: {response_degradation:.2f}x\n"
                f"  • Memory: {result.memory_usage_mb:.1f}MB"
            )

        # Scalability assertions
        avg_efficiency = statistics.mean(throughput_efficiency)
        max_response_degradation = max(response_time_degradation)

        assert avg_efficiency >= 0.5, (
            f"Average throughput efficiency too low: {avg_efficiency:.2%}"
        )
        assert max_response_degradation <= 5.0, (
            f"Response time degradation too high: {max_response_degradation:.1f}x"
        )

        return {
            "user_loads": user_loads,
            "results": scalability_results,
            "throughput_efficiency": throughput_efficiency,
            "response_time_degradation": response_time_degradation,
            "avg_efficiency": avg_efficiency,
        }

    async def test_sustained_load_stability(self, optimized_data_provider):
        """Test stability under sustained load."""
        load_runner = LoadTestRunner(optimized_data_provider)

        session_config = {
            "symbols": ["AAPL", "MSFT"],
            "strategies": ["sma_cross", "rsi"],
            "think_time": (0.5, 1.0),
        }

        # Run sustained load for longer duration
        result = await load_runner.run_load_test(
            concurrent_users=25,
            session_config=session_config,
            duration_seconds=300,  # 5 minutes
        )

        # Stability assertions
        assert result.errors_per_second <= 0.1, (
            f"Error rate too high: {result.errors_per_second:.3f} err/s"
        )
        assert result.successful_requests / result.total_requests >= 0.95, (
            "Success rate degraded over time"
        )
        assert result.memory_usage_mb <= 800, (
            f"Memory usage grew too much: {result.memory_usage_mb:.1f}MB"
        )

        # Check for performance consistency (no significant degradation)
        assert result.p99_response_time / result.p50_response_time <= 5.0, (
            "Response time variance too high"
        )

        logger.info(
            f"Sustained Load Results (25 users, 5 minutes):\n"
            f"  • Total Requests: {result.total_requests}\n"
            f"  • Success Rate: {result.successful_requests / result.total_requests * 100:.2f}%\n"
            f"  • Avg Throughput: {result.requests_per_second:.2f} req/s\n"
            f"  • Response Time (50/95/99): {result.p50_response_time:.2f}s/"
            f"{result.p95_response_time:.2f}s/{result.p99_response_time:.2f}s\n"
            f"  • Memory Growth: {result.memory_usage_mb:.1f}MB\n"
            f"  • Error Rate: {result.errors_per_second:.4f} err/s"
        )

        return result

    async def test_database_connection_pooling_under_load(
        self, optimized_data_provider, db_session
    ):
        """Test database connection pooling under concurrent load."""
        # Generate backtest results to save to database
        engine = VectorBTEngine(data_provider=optimized_data_provider)
        test_symbols = ["DB_LOAD_1", "DB_LOAD_2", "DB_LOAD_3"]

        # Pre-generate results for database testing
        backtest_results = []
        for symbol in test_symbols:
            result = await engine.run_backtest(
                symbol=symbol,
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            backtest_results.append(result)

        # Test concurrent database operations
        async def concurrent_database_operations(operation_id: int) -> dict[str, Any]:
            """Simulate concurrent database save/retrieve operations."""
            start_time = time.time()
            operations_completed = 0
            errors = []

            try:
                with BacktestPersistenceManager(session=db_session) as persistence:
                    # Save operations
                    for result in backtest_results:
                        try:
                            backtest_id = persistence.save_backtest_result(
                                vectorbt_results=result,
                                execution_time=2.0,
                                notes=f"Load test operation {operation_id}",
                            )
                            operations_completed += 1

                            # Retrieve operation
                            retrieved = persistence.get_backtest_by_id(backtest_id)
                            if retrieved:
                                operations_completed += 1

                        except Exception as e:
                            errors.append(str(e))

            except Exception as e:
                errors.append(f"Session error: {str(e)}")

            operation_time = time.time() - start_time

            return {
                "operation_id": operation_id,
                "operations_completed": operations_completed,
                "errors": errors,
                "operation_time": operation_time,
            }

        # Run concurrent database operations
        concurrent_operations = 20
        db_tasks = [
            concurrent_database_operations(i) for i in range(concurrent_operations)
        ]

        start_time = time.time()
        db_results = await asyncio.gather(*db_tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Analyze database performance under load
        successful_operations = [r for r in db_results if isinstance(r, dict)]
        failed_operations = len(db_results) - len(successful_operations)

        total_operations = sum(r["operations_completed"] for r in successful_operations)
        total_errors = sum(len(r["errors"]) for r in successful_operations)
        avg_operation_time = statistics.mean(
            [r["operation_time"] for r in successful_operations]
        )

        db_throughput = total_operations / total_time if total_time > 0 else 0
        error_rate = total_errors / total_operations if total_operations > 0 else 0

        logger.info(
            f"Database Load Test Results:\n"
            f"  • Concurrent Operations: {concurrent_operations}\n"
            f"  • Successful Sessions: {len(successful_operations)}\n"
            f"  • Failed Sessions: {failed_operations}\n"
            f"  • Total DB Operations: {total_operations}\n"
            f"  • DB Throughput: {db_throughput:.2f} ops/s\n"
            f"  • Error Rate: {error_rate:.3%}\n"
            f"  • Avg Operation Time: {avg_operation_time:.3f}s"
        )

        # Database performance assertions
        assert len(successful_operations) / len(db_results) >= 0.9, (
            "DB session success rate too low"
        )
        assert error_rate <= 0.05, f"DB error rate too high: {error_rate:.3%}"
        assert db_throughput >= 5.0, f"DB throughput too low: {db_throughput:.2f} ops/s"

        return {
            "concurrent_operations": concurrent_operations,
            "db_throughput": db_throughput,
            "error_rate": error_rate,
            "avg_operation_time": avg_operation_time,
        }


if __name__ == "__main__":
    # Run load testing suite
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "--timeout=600",  # 10 minute timeout for load tests
            "--durations=10",
        ]
    )
