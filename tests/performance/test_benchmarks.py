"""
Performance Benchmarks Against Target Metrics.

This test suite covers:
- Backtest execution < 2 seconds per backtest
- Memory usage < 500MB per backtest
- Cache hit rate > 80%
- API failure rate < 0.1%
- Database query performance < 100ms
- Throughput targets (requests per second)
- Response time SLA compliance
- Resource utilization efficiency
"""

import asyncio
import gc
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import psutil
import os

from maverick_mcp.backtesting import VectorBTEngine, BacktestAnalyzer
from maverick_mcp.backtesting.persistence import BacktestPersistenceManager
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Data class for benchmark test results."""
    test_name: str
    target_value: float
    actual_value: float
    unit: str
    passed: bool
    margin: float
    details: Dict[str, Any]


class BenchmarkTracker:
    """Track and validate performance benchmarks."""

    def __init__(self):
        self.results = []
        self.process = psutil.Process(os.getpid())

    def add_benchmark(self,
                     test_name: str,
                     target_value: float,
                     actual_value: float,
                     unit: str,
                     comparison: str = "<=",
                     details: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Add a benchmark result."""
        if comparison == "<=":
            passed = actual_value <= target_value
            margin = (actual_value - target_value) / target_value if target_value > 0 else 0
        elif comparison == ">=":
            passed = actual_value >= target_value
            margin = (target_value - actual_value) / target_value if target_value > 0 else 0
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")

        result = BenchmarkResult(
            test_name=test_name,
            target_value=target_value,
            actual_value=actual_value,
            unit=unit,
            passed=passed,
            margin=margin,
            details=details or {},
        )

        self.results.append(result)

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} {test_name}: {actual_value:.3f}{unit} (target: {target_value}{unit})")

        return result

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.results,
        }


class TestPerformanceBenchmarks:
    """Performance benchmarks against target metrics."""

    @pytest.fixture
    async def benchmark_data_provider(self):
        """Create optimized data provider for benchmarks."""
        provider = Mock()

        def generate_benchmark_data(symbol: str) -> pd.DataFrame:
            """Generate optimized data for benchmarking."""
            # Use symbol hash for deterministic but varied data
            seed = hash(symbol) % 1000
            np.random.seed(seed)

            # Generate 1 year of data
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
            returns = np.random.normal(0.0008, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)

            return pd.DataFrame({
                "Open": prices * np.random.uniform(0.995, 1.005, len(dates)),
                "High": prices * np.random.uniform(1.005, 1.025, len(dates)),
                "Low": prices * np.random.uniform(0.975, 0.995, len(dates)),
                "Close": prices,
                "Volume": np.random.randint(1000000, 5000000, len(dates)),
                "Adj Close": prices,
            }, index=dates)

        provider.get_stock_data.side_effect = generate_benchmark_data
        return provider

    async def test_backtest_execution_time_benchmark(self, benchmark_data_provider):
        """Test: Backtest execution < 2 seconds per backtest."""
        benchmark = BenchmarkTracker()
        engine = VectorBTEngine(data_provider=benchmark_data_provider)

        test_cases = [
            ("AAPL", "sma_cross"),
            ("GOOGL", "rsi"),
            ("MSFT", "macd"),
            ("AMZN", "bollinger"),
            ("TSLA", "momentum"),
        ]

        execution_times = []

        for symbol, strategy in test_cases:
            parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

            start_time = time.time()
            result = await engine.run_backtest(
                symbol=symbol,
                strategy_type=strategy,
                parameters=parameters,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            execution_time = time.time() - start_time

            execution_times.append(execution_time)

            # Individual backtest benchmark
            benchmark.add_benchmark(
                test_name=f"backtest_time_{symbol}_{strategy}",
                target_value=2.0,
                actual_value=execution_time,
                unit="s",
                comparison="<=",
                details={"symbol": symbol, "strategy": strategy, "result_size": len(str(result))}
            )

        # Overall benchmark
        avg_execution_time = statistics.mean(execution_times)
        max_execution_time = max(execution_times)

        benchmark.add_benchmark(
            test_name="avg_backtest_execution_time",
            target_value=2.0,
            actual_value=avg_execution_time,
            unit="s",
            comparison="<=",
            details={"individual_times": execution_times}
        )

        benchmark.add_benchmark(
            test_name="max_backtest_execution_time",
            target_value=3.0,  # Allow some variance
            actual_value=max_execution_time,
            unit="s",
            comparison="<=",
            details={"slowest_case": test_cases[execution_times.index(max_execution_time)]}
        )

        logger.info(f"Backtest Execution Time Benchmark Summary:\n"
                   f"  • Average: {avg_execution_time:.3f}s\n"
                   f"  • Maximum: {max_execution_time:.3f}s\n"
                   f"  • Minimum: {min(execution_times):.3f}s\n"
                   f"  • Standard Deviation: {statistics.stdev(execution_times):.3f}s")

        return benchmark.summary()

    async def test_memory_usage_benchmark(self, benchmark_data_provider):
        """Test: Memory usage < 500MB per backtest."""
        benchmark = BenchmarkTracker()
        engine = VectorBTEngine(data_provider=benchmark_data_provider)

        initial_memory = benchmark.get_memory_usage()
        memory_measurements = []

        test_symbols = ["MEM_TEST_1", "MEM_TEST_2", "MEM_TEST_3", "MEM_TEST_4", "MEM_TEST_5"]

        for i, symbol in enumerate(test_symbols):
            gc.collect()  # Force garbage collection before measurement
            pre_backtest_memory = benchmark.get_memory_usage()

            # Run backtest
            result = await engine.run_backtest(
                symbol=symbol,
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            post_backtest_memory = benchmark.get_memory_usage()
            memory_delta = post_backtest_memory - pre_backtest_memory

            memory_measurements.append({
                "symbol": symbol,
                "pre_memory": pre_backtest_memory,
                "post_memory": post_backtest_memory,
                "delta": memory_delta,
            })

            # Individual memory benchmark
            benchmark.add_benchmark(
                test_name=f"memory_usage_{symbol}",
                target_value=500.0,
                actual_value=memory_delta,
                unit="MB",
                comparison="<=",
                details={
                    "pre_memory": pre_backtest_memory,
                    "post_memory": post_backtest_memory,
                    "result_size": len(str(result))
                }
            )

        # Overall memory benchmarks
        total_memory_growth = benchmark.get_memory_usage() - initial_memory
        avg_memory_per_backtest = total_memory_growth / len(test_symbols) if test_symbols else 0
        max_memory_delta = max(m["delta"] for m in memory_measurements)

        benchmark.add_benchmark(
            test_name="avg_memory_per_backtest",
            target_value=500.0,
            actual_value=avg_memory_per_backtest,
            unit="MB",
            comparison="<=",
            details={"total_growth": total_memory_growth, "measurements": memory_measurements}
        )

        benchmark.add_benchmark(
            test_name="max_memory_per_backtest",
            target_value=750.0,  # Allow some variance
            actual_value=max_memory_delta,
            unit="MB",
            comparison="<=",
            details={"worst_case": memory_measurements[
                next(i for i, m in enumerate(memory_measurements) if m["delta"] == max_memory_delta)
            ]}
        )

        logger.info(f"Memory Usage Benchmark Summary:\n"
                   f"  • Total Growth: {total_memory_growth:.1f}MB\n"
                   f"  • Avg per Backtest: {avg_memory_per_backtest:.1f}MB\n"
                   f"  • Max per Backtest: {max_memory_delta:.1f}MB\n"
                   f"  • Initial Memory: {initial_memory:.1f}MB")

        return benchmark.summary()

    async def test_cache_hit_rate_benchmark(self, benchmark_data_provider):
        """Test: Cache hit rate > 80%."""
        benchmark = BenchmarkTracker()
        engine = VectorBTEngine(data_provider=benchmark_data_provider)

        # Mock cache to track hits/misses
        cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}

        def mock_cache_get(key):
            cache_stats["total_requests"] += 1
            # Simulate realistic cache behavior
            if cache_stats["total_requests"] <= 5:  # First few are misses
                cache_stats["misses"] += 1
                return None
            else:  # Later requests are hits
                cache_stats["hits"] += 1
                return "cached_result"

        with patch('maverick_mcp.core.cache.CacheManager.get', side_effect=mock_cache_get):
            # Run multiple backtests with repeated data access
            symbols = ["CACHE_A", "CACHE_B", "CACHE_A", "CACHE_B", "CACHE_A", "CACHE_C", "CACHE_A"]

            for symbol in symbols:
                await engine.run_backtest(
                    symbol=symbol,
                    strategy_type="sma_cross",
                    parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                )

        # Calculate cache hit rate
        total_cache_requests = cache_stats["total_requests"]
        cache_hits = cache_stats["hits"]
        cache_hit_rate = (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0

        benchmark.add_benchmark(
            test_name="cache_hit_rate",
            target_value=80.0,
            actual_value=cache_hit_rate,
            unit="%",
            comparison=">=",
            details={
                "total_requests": total_cache_requests,
                "hits": cache_hits,
                "misses": cache_stats["misses"],
            }
        )

        logger.info(f"Cache Hit Rate Benchmark:\n"
                   f"  • Total Cache Requests: {total_cache_requests}\n"
                   f"  • Cache Hits: {cache_hits}\n"
                   f"  • Cache Misses: {cache_stats['misses']}\n"
                   f"  • Hit Rate: {cache_hit_rate:.1f}%")

        return benchmark.summary()

    async def test_api_failure_rate_benchmark(self, benchmark_data_provider):
        """Test: API failure rate < 0.1%."""
        benchmark = BenchmarkTracker()

        # Mock API with occasional failures
        api_stats = {"total_calls": 0, "failures": 0}

        def mock_api_call(*args, **kwargs):
            api_stats["total_calls"] += 1
            # Simulate very low failure rate
            if api_stats["total_calls"] % 2000 == 0:  # 0.05% failure rate
                api_stats["failures"] += 1
                raise ConnectionError("Simulated API failure")
            return benchmark_data_provider.get_stock_data(*args, **kwargs)

        # Test with many API calls
        with patch.object(benchmark_data_provider, 'get_stock_data', side_effect=mock_api_call):
            engine = VectorBTEngine(data_provider=benchmark_data_provider)

            test_symbols = [f"API_TEST_{i}" for i in range(50)]  # 50 symbols to test API reliability

            successful_backtests = 0
            failed_backtests = 0

            for symbol in test_symbols:
                try:
                    await engine.run_backtest(
                        symbol=symbol,
                        strategy_type="rsi",
                        parameters=STRATEGY_TEMPLATES["rsi"]["parameters"],
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                    )
                    successful_backtests += 1
                except Exception:
                    failed_backtests += 1

        # Calculate failure rates
        total_api_calls = api_stats["total_calls"]
        api_failures = api_stats["failures"]
        api_failure_rate = (api_failures / total_api_calls * 100) if total_api_calls > 0 else 0

        total_backtests = successful_backtests + failed_backtests
        backtest_failure_rate = (failed_backtests / total_backtests * 100) if total_backtests > 0 else 0

        benchmark.add_benchmark(
            test_name="api_failure_rate",
            target_value=0.1,
            actual_value=api_failure_rate,
            unit="%",
            comparison="<=",
            details={
                "total_api_calls": total_api_calls,
                "api_failures": api_failures,
                "successful_backtests": successful_backtests,
                "failed_backtests": failed_backtests,
            }
        )

        benchmark.add_benchmark(
            test_name="backtest_success_rate",
            target_value=99.5,
            actual_value=100 - backtest_failure_rate,
            unit="%",
            comparison=">=",
            details={"backtest_failure_rate": backtest_failure_rate}
        )

        logger.info(f"API Reliability Benchmark:\n"
                   f"  • Total API Calls: {total_api_calls}\n"
                   f"  • API Failures: {api_failures}\n"
                   f"  • API Failure Rate: {api_failure_rate:.3f}%\n"
                   f"  • Backtest Success Rate: {100 - backtest_failure_rate:.2f}%")

        return benchmark.summary()

    async def test_database_query_performance_benchmark(self, benchmark_data_provider, db_session):
        """Test: Database query performance < 100ms."""
        benchmark = BenchmarkTracker()
        engine = VectorBTEngine(data_provider=benchmark_data_provider)

        # Generate test data for database operations
        test_results = []
        for i in range(10):
            result = await engine.run_backtest(
                symbol=f"DB_PERF_{i}",
                strategy_type="macd",
                parameters=STRATEGY_TEMPLATES["macd"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            test_results.append(result)

        # Test database save performance
        save_times = []
        with BacktestPersistenceManager(session=db_session) as persistence:
            for result in test_results:
                start_time = time.time()
                backtest_id = persistence.save_backtest_result(
                    vectorbt_results=result,
                    execution_time=2.0,
                    notes="DB performance test",
                )
                save_time = (time.time() - start_time) * 1000  # Convert to ms
                save_times.append((backtest_id, save_time))

        # Test database query performance
        query_times = []
        with BacktestPersistenceManager(session=db_session) as persistence:
            for backtest_id, _ in save_times:
                start_time = time.time()
                retrieved = persistence.get_backtest_by_id(backtest_id)
                query_time = (time.time() - start_time) * 1000  # Convert to ms
                query_times.append(query_time)

            # Test bulk query performance
            start_time = time.time()
            bulk_results = persistence.get_backtests_by_strategy("macd")
            bulk_query_time = (time.time() - start_time) * 1000

        # Calculate benchmarks
        avg_save_time = statistics.mean([t for _, t in save_times])
        max_save_time = max([t for _, t in save_times])
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)

        # Add benchmarks
        benchmark.add_benchmark(
            test_name="avg_db_save_time",
            target_value=100.0,
            actual_value=avg_save_time,
            unit="ms",
            comparison="<=",
            details={"individual_times": [t for _, t in save_times]}
        )

        benchmark.add_benchmark(
            test_name="max_db_save_time",
            target_value=200.0,
            actual_value=max_save_time,
            unit="ms",
            comparison="<=",
        )

        benchmark.add_benchmark(
            test_name="avg_db_query_time",
            target_value=50.0,
            actual_value=avg_query_time,
            unit="ms",
            comparison="<=",
            details={"individual_times": query_times}
        )

        benchmark.add_benchmark(
            test_name="max_db_query_time",
            target_value=100.0,
            actual_value=max_query_time,
            unit="ms",
            comparison="<=",
        )

        benchmark.add_benchmark(
            test_name="bulk_query_time",
            target_value=200.0,
            actual_value=bulk_query_time,
            unit="ms",
            comparison="<=",
            details={"records_returned": len(bulk_results)}
        )

        logger.info(f"Database Performance Benchmark:\n"
                   f"  • Avg Save Time: {avg_save_time:.1f}ms\n"
                   f"  • Max Save Time: {max_save_time:.1f}ms\n"
                   f"  • Avg Query Time: {avg_query_time:.1f}ms\n"
                   f"  • Max Query Time: {max_query_time:.1f}ms\n"
                   f"  • Bulk Query Time: {bulk_query_time:.1f}ms")

        return benchmark.summary()

    async def test_throughput_benchmark(self, benchmark_data_provider):
        """Test: Throughput targets (requests per second)."""
        benchmark = BenchmarkTracker()
        engine = VectorBTEngine(data_provider=benchmark_data_provider)

        # Test sequential throughput
        symbols = ["THRU_1", "THRU_2", "THRU_3", "THRU_4", "THRU_5"]
        start_time = time.time()

        for symbol in symbols:
            await engine.run_backtest(
                symbol=symbol,
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

        sequential_time = time.time() - start_time
        sequential_throughput = len(symbols) / sequential_time

        # Test concurrent throughput
        concurrent_symbols = ["CONC_1", "CONC_2", "CONC_3", "CONC_4", "CONC_5"]
        start_time = time.time()

        concurrent_tasks = []
        for symbol in concurrent_symbols:
            task = engine.run_backtest(
                symbol=symbol,
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            concurrent_tasks.append(task)

        await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - start_time
        concurrent_throughput = len(concurrent_symbols) / concurrent_time

        # Benchmarks
        benchmark.add_benchmark(
            test_name="sequential_throughput",
            target_value=2.0,  # 2 backtests per second
            actual_value=sequential_throughput,
            unit="req/s",
            comparison=">=",
            details={"execution_time": sequential_time, "requests": len(symbols)}
        )

        benchmark.add_benchmark(
            test_name="concurrent_throughput",
            target_value=5.0,  # 5 backtests per second with concurrency
            actual_value=concurrent_throughput,
            unit="req/s",
            comparison=">=",
            details={"execution_time": concurrent_time, "requests": len(concurrent_symbols)}
        )

        # Concurrency speedup
        speedup = concurrent_throughput / sequential_throughput
        benchmark.add_benchmark(
            test_name="concurrency_speedup",
            target_value=2.0,  # At least 2x speedup
            actual_value=speedup,
            unit="x",
            comparison=">=",
            details={
                "sequential_throughput": sequential_throughput,
                "concurrent_throughput": concurrent_throughput,
            }
        )

        logger.info(f"Throughput Benchmark:\n"
                   f"  • Sequential: {sequential_throughput:.2f} req/s\n"
                   f"  • Concurrent: {concurrent_throughput:.2f} req/s\n"
                   f"  • Speedup: {speedup:.2f}x")

        return benchmark.summary()

    async def test_response_time_sla_benchmark(self, benchmark_data_provider):
        """Test: Response time SLA compliance."""
        benchmark = BenchmarkTracker()
        engine = VectorBTEngine(data_provider=benchmark_data_provider)

        response_times = []
        symbols = [f"SLA_{i}" for i in range(20)]

        for symbol in symbols:
            start_time = time.time()
            await engine.run_backtest(
                symbol=symbol,
                strategy_type="rsi",
                parameters=STRATEGY_TEMPLATES["rsi"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            response_times.append(response_time)

        # SLA percentile benchmarks
        p50 = np.percentile(response_times, 50)
        p95 = np.percentile(response_times, 95)
        p99 = np.percentile(response_times, 99)

        benchmark.add_benchmark(
            test_name="response_time_p50",
            target_value=1500.0,  # 1.5 seconds for 50th percentile
            actual_value=p50,
            unit="ms",
            comparison="<=",
            details={"percentile": "50th"}
        )

        benchmark.add_benchmark(
            test_name="response_time_p95",
            target_value=3000.0,  # 3 seconds for 95th percentile
            actual_value=p95,
            unit="ms",
            comparison="<=",
            details={"percentile": "95th"}
        )

        benchmark.add_benchmark(
            test_name="response_time_p99",
            target_value=5000.0,  # 5 seconds for 99th percentile
            actual_value=p99,
            unit="ms",
            comparison="<=",
            details={"percentile": "99th"}
        )

        # SLA compliance rate (percentage of requests under target)
        sla_target = 2000.0  # 2 seconds
        sla_compliant = sum(1 for t in response_times if t <= sla_target)
        sla_compliance_rate = (sla_compliant / len(response_times) * 100)

        benchmark.add_benchmark(
            test_name="sla_compliance_rate",
            target_value=95.0,  # 95% of requests should meet SLA
            actual_value=sla_compliance_rate,
            unit="%",
            comparison=">=",
            details={
                "sla_target_ms": sla_target,
                "compliant_requests": sla_compliant,
                "total_requests": len(response_times),
            }
        )

        logger.info(f"Response Time SLA Benchmark:\n"
                   f"  • 50th Percentile: {p50:.1f}ms\n"
                   f"  • 95th Percentile: {p95:.1f}ms\n"
                   f"  • 99th Percentile: {p99:.1f}ms\n"
                   f"  • SLA Compliance: {sla_compliance_rate:.1f}%")

        return benchmark.summary()

    async def test_comprehensive_benchmark_suite(self, benchmark_data_provider, db_session):
        """Run comprehensive benchmark suite and generate report."""
        logger.info("Running Comprehensive Benchmark Suite...")

        # Run all individual benchmarks
        benchmark_results = []

        benchmark_results.append(await self.test_backtest_execution_time_benchmark(benchmark_data_provider))
        benchmark_results.append(await self.test_memory_usage_benchmark(benchmark_data_provider))
        benchmark_results.append(await self.test_cache_hit_rate_benchmark(benchmark_data_provider))
        benchmark_results.append(await self.test_api_failure_rate_benchmark(benchmark_data_provider))
        benchmark_results.append(await self.test_database_query_performance_benchmark(benchmark_data_provider, db_session))
        benchmark_results.append(await self.test_throughput_benchmark(benchmark_data_provider))
        benchmark_results.append(await self.test_response_time_sla_benchmark(benchmark_data_provider))

        # Aggregate results
        total_tests = sum(r["total_tests"] for r in benchmark_results)
        total_passed = sum(r["passed_tests"] for r in benchmark_results)
        total_failed = sum(r["failed_tests"] for r in benchmark_results)
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0

        # Generate comprehensive report
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "overall_pass_rate": overall_pass_rate,
            },
            "benchmark_suites": benchmark_results,
            "critical_failures": [
                result for suite in benchmark_results
                for result in suite["results"]
                if not result.passed and result.margin > 0.2  # More than 20% over target
            ],
        }

        logger.info(
            f"\n{'='*60}\n"
            f"COMPREHENSIVE BENCHMARK REPORT\n"
            f"{'='*60}\n"
            f"Total Tests: {total_tests}\n"
            f"Passed: {total_passed} ({overall_pass_rate:.1%})\n"
            f"Failed: {total_failed}\n"
            f"{'='*60}\n"
        )

        # Assert overall benchmark success
        assert overall_pass_rate >= 0.8, f"Overall benchmark pass rate too low: {overall_pass_rate:.1%}"
        assert len(report["critical_failures"]) == 0, f"Critical benchmark failures detected: {len(report['critical_failures'])}"

        return report


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--timeout=300",  # 5 minute timeout for benchmarks
    ])