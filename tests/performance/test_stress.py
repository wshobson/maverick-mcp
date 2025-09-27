"""
Stress Testing for Resource Usage Under Load.

This test suite covers:
- Sustained load testing (1+ hour)
- Memory leak detection over time
- CPU utilization monitoring under stress
- Database connection pool exhaustion
- File descriptor limits testing
- Network connection limits
- Queue overflow scenarios
- System stability under extreme conditions
"""

import asyncio
import gc
import logging
import resource
import threading
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
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""

    timestamp: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    cpu_percent: float
    threads: int
    file_descriptors: int
    connections: int
    swap_usage_mb: float


class ResourceMonitor:
    """Monitor system resources over time."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: list[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start continuous resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info(
            f"Resource monitoring stopped. Collected {len(self.snapshots)} snapshots"
        )

    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a resource snapshot."""
        memory_info = self.process.memory_info()

        # Get file descriptor count
        try:
            fd_count = self.process.num_fds()
        except AttributeError:
            # Windows doesn't have num_fds()
            fd_count = len(self.process.open_files())

        # Get connection count
        try:
            connections = len(self.process.connections())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            connections = 0

        # Get swap usage
        try:
            swap = psutil.swap_memory()
            swap_used_mb = swap.used / 1024 / 1024
        except Exception:
            swap_used_mb = 0

        return ResourceSnapshot(
            timestamp=time.time(),
            memory_rss_mb=memory_info.rss / 1024 / 1024,
            memory_vms_mb=memory_info.vms / 1024 / 1024,
            memory_percent=self.process.memory_percent(),
            cpu_percent=self.process.cpu_percent(),
            threads=self.process.num_threads(),
            file_descriptors=fd_count,
            connections=connections,
            swap_usage_mb=swap_used_mb,
        )

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return self._take_snapshot()

    def analyze_trends(self) -> dict[str, Any]:
        """Analyze resource usage trends."""
        if len(self.snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Calculate trends
        timestamps = [s.timestamp for s in self.snapshots]
        memories = [s.memory_rss_mb for s in self.snapshots]
        cpus = [s.cpu_percent for s in self.snapshots]
        fds = [s.file_descriptors for s in self.snapshots]

        # Linear regression for memory trend
        n = len(timestamps)
        sum_t = sum(timestamps)
        sum_m = sum(memories)
        sum_tm = sum(t * m for t, m in zip(timestamps, memories, strict=False))
        sum_tt = sum(t * t for t in timestamps)

        memory_slope = (
            (n * sum_tm - sum_t * sum_m) / (n * sum_tt - sum_t * sum_t)
            if n * sum_tt != sum_t * sum_t
            else 0
        )

        return {
            "duration_seconds": timestamps[-1] - timestamps[0],
            "initial_memory_mb": memories[0],
            "final_memory_mb": memories[-1],
            "memory_growth_mb": memories[-1] - memories[0],
            "memory_growth_rate_mb_per_hour": memory_slope * 3600,
            "peak_memory_mb": max(memories),
            "avg_cpu_percent": sum(cpus) / len(cpus),
            "peak_cpu_percent": max(cpus),
            "initial_file_descriptors": fds[0],
            "final_file_descriptors": fds[-1],
            "fd_growth": fds[-1] - fds[0],
            "peak_file_descriptors": max(fds),
            "snapshots_count": len(self.snapshots),
        }


class StressTestRunner:
    """Run various stress tests."""

    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.resource_monitor = ResourceMonitor(interval=2.0)

    async def sustained_load_test(
        self, duration_minutes: int = 60, concurrent_load: int = 10
    ) -> dict[str, Any]:
        """Run sustained load test for extended duration."""
        logger.info(
            f"Starting sustained load test: {duration_minutes} minutes with {concurrent_load} concurrent operations"
        )

        self.resource_monitor.start_monitoring()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        total_operations = 0
        total_errors = 0
        operation_times = []

        try:
            # Create semaphore for concurrent control
            semaphore = asyncio.Semaphore(concurrent_load)

            async def sustained_operation(operation_id: int):
                """Single sustained operation."""
                nonlocal total_operations, total_errors

                engine = VectorBTEngine(data_provider=self.data_provider)
                symbol = f"STRESS_{operation_id % 20}"  # Cycle through 20 symbols
                strategy = ["sma_cross", "rsi", "macd"][
                    operation_id % 3
                ]  # Cycle through strategies

                try:
                    async with semaphore:
                        op_start = time.time()

                        await engine.run_backtest(
                            symbol=symbol,
                            strategy_type=strategy,
                            parameters=STRATEGY_TEMPLATES[strategy]["parameters"],
                            start_date="2023-01-01",
                            end_date="2023-12-31",
                        )

                        op_time = time.time() - op_start
                        operation_times.append(op_time)
                        total_operations += 1

                        if total_operations % 100 == 0:
                            logger.info(f"Completed {total_operations} operations")

                except Exception as e:
                    total_errors += 1
                    logger.error(f"Operation {operation_id} failed: {e}")

            # Run operations continuously until duration expires
            operation_id = 0
            active_tasks = []

            while time.time() < end_time:
                # Start new operation
                task = asyncio.create_task(sustained_operation(operation_id))
                active_tasks.append(task)
                operation_id += 1

                # Clean up completed tasks
                active_tasks = [t for t in active_tasks if not t.done()]

                # Control task creation rate
                await asyncio.sleep(0.1)

                # Prevent task accumulation
                if len(active_tasks) > concurrent_load * 2:
                    await asyncio.sleep(1.0)

            # Wait for remaining tasks to complete
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        finally:
            self.resource_monitor.stop_monitoring()

        actual_duration = time.time() - start_time
        trend_analysis = self.resource_monitor.analyze_trends()

        return {
            "duration_minutes": actual_duration / 60,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "error_rate": total_errors / total_operations
            if total_operations > 0
            else 0,
            "operations_per_minute": total_operations / (actual_duration / 60),
            "avg_operation_time": sum(operation_times) / len(operation_times)
            if operation_times
            else 0,
            "resource_trends": trend_analysis,
            "concurrent_load": concurrent_load,
        }

    async def memory_leak_detection_test(
        self, iterations: int = 1000
    ) -> dict[str, Any]:
        """Test for memory leaks over many iterations."""
        logger.info(f"Starting memory leak detection test with {iterations} iterations")

        self.resource_monitor.start_monitoring()
        engine = VectorBTEngine(data_provider=self.data_provider)

        initial_memory = self.resource_monitor.get_current_snapshot().memory_rss_mb
        memory_measurements = []

        try:
            for i in range(iterations):
                # Run backtest operation
                symbol = f"LEAK_TEST_{i % 10}"

                await engine.run_backtest(
                    symbol=symbol,
                    strategy_type="sma_cross",
                    parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                )

                # Force garbage collection every 50 iterations
                if i % 50 == 0:
                    gc.collect()
                    snapshot = self.resource_monitor.get_current_snapshot()
                    memory_measurements.append(
                        {
                            "iteration": i,
                            "memory_mb": snapshot.memory_rss_mb,
                            "memory_growth": snapshot.memory_rss_mb - initial_memory,
                        }
                    )

                    if i % 200 == 0:
                        logger.info(
                            f"Iteration {i}: Memory = {snapshot.memory_rss_mb:.1f}MB "
                            f"(+{snapshot.memory_rss_mb - initial_memory:.1f}MB)"
                        )

        finally:
            self.resource_monitor.stop_monitoring()

        final_memory = self.resource_monitor.get_current_snapshot().memory_rss_mb
        total_growth = final_memory - initial_memory

        # Analyze memory leak pattern
        if len(memory_measurements) > 2:
            iterations_list = [m["iteration"] for m in memory_measurements]
            growth_list = [m["memory_growth"] for m in memory_measurements]

            # Linear regression to detect memory leak
            n = len(iterations_list)
            sum_x = sum(iterations_list)
            sum_y = sum(growth_list)
            sum_xy = sum(
                x * y for x, y in zip(iterations_list, growth_list, strict=False)
            )
            sum_xx = sum(x * x for x in iterations_list)

            if n * sum_xx != sum_x * sum_x:
                leak_rate = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            else:
                leak_rate = 0

            # Memory leak per 1000 iterations
            leak_per_1000_iterations = leak_rate * 1000
        else:
            leak_rate = 0
            leak_per_1000_iterations = 0

        return {
            "iterations": iterations,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "total_memory_growth_mb": total_growth,
            "leak_rate_mb_per_iteration": leak_rate,
            "leak_per_1000_iterations_mb": leak_per_1000_iterations,
            "memory_measurements": memory_measurements,
            "leak_detected": abs(leak_per_1000_iterations)
            > 10.0,  # More than 10MB per 1000 iterations
        }

    async def cpu_stress_test(
        self, duration_minutes: int = 10, cpu_target: float = 0.9
    ) -> dict[str, Any]:
        """Test CPU utilization under stress."""
        logger.info(
            f"Starting CPU stress test: {duration_minutes} minutes at {cpu_target * 100}% target"
        )

        self.resource_monitor.start_monitoring()

        # Create CPU-intensive background load
        stop_event = threading.Event()
        cpu_threads = []

        def cpu_intensive_task():
            """CPU-intensive computation."""
            while not stop_event.is_set():
                # Perform CPU-intensive work
                for _ in range(10000):
                    _ = sum(i**2 for i in range(100))
                time.sleep(0.001)  # Brief pause

        try:
            # Start CPU load threads
            num_cpu_threads = max(1, int(psutil.cpu_count() * cpu_target))
            for _ in range(num_cpu_threads):
                thread = threading.Thread(target=cpu_intensive_task, daemon=True)
                thread.start()
                cpu_threads.append(thread)

            # Run backtests under CPU stress
            engine = VectorBTEngine(data_provider=self.data_provider)
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)

            operations_completed = 0
            cpu_stress_errors = 0
            response_times = []

            while time.time() < end_time:
                try:
                    op_start = time.time()

                    symbol = f"CPU_STRESS_{operations_completed % 5}"
                    await asyncio.wait_for(
                        engine.run_backtest(
                            symbol=symbol,
                            strategy_type="rsi",
                            parameters=STRATEGY_TEMPLATES["rsi"]["parameters"],
                            start_date="2023-01-01",
                            end_date="2023-12-31",
                        ),
                        timeout=30.0,  # Prevent hanging under CPU stress
                    )

                    response_time = time.time() - op_start
                    response_times.append(response_time)
                    operations_completed += 1

                except TimeoutError:
                    cpu_stress_errors += 1
                    logger.warning("Operation timed out under CPU stress")

                except Exception as e:
                    cpu_stress_errors += 1
                    logger.error(f"Operation failed under CPU stress: {e}")

                # Brief pause between operations
                await asyncio.sleep(1.0)

        finally:
            # Stop CPU stress
            stop_event.set()
            for thread in cpu_threads:
                thread.join(timeout=1.0)

            self.resource_monitor.stop_monitoring()

        trend_analysis = self.resource_monitor.analyze_trends()

        return {
            "duration_minutes": duration_minutes,
            "cpu_target_percent": cpu_target * 100,
            "operations_completed": operations_completed,
            "cpu_stress_errors": cpu_stress_errors,
            "error_rate": cpu_stress_errors / (operations_completed + cpu_stress_errors)
            if (operations_completed + cpu_stress_errors) > 0
            else 0,
            "avg_response_time": sum(response_times) / len(response_times)
            if response_times
            else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "avg_cpu_utilization": trend_analysis["avg_cpu_percent"],
            "peak_cpu_utilization": trend_analysis["peak_cpu_percent"],
        }

    async def database_connection_exhaustion_test(
        self, db_session, max_connections: int = 50
    ) -> dict[str, Any]:
        """Test database behavior under connection exhaustion."""
        logger.info(
            f"Starting database connection exhaustion test with {max_connections} connections"
        )

        # Generate test data
        engine = VectorBTEngine(data_provider=self.data_provider)
        test_results = []

        for i in range(5):
            result = await engine.run_backtest(
                symbol=f"DB_EXHAUST_{i}",
                strategy_type="macd",
                parameters=STRATEGY_TEMPLATES["macd"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            test_results.append(result)

        # Test connection exhaustion
        async def database_operation(conn_id: int) -> dict[str, Any]:
            """Single database operation holding connection."""
            try:
                with BacktestPersistenceManager(session=db_session) as persistence:
                    # Hold connection and perform operations
                    saved_ids = []

                    for result in test_results:
                        backtest_id = persistence.save_backtest_result(
                            vectorbt_results=result,
                            execution_time=2.0,
                            notes=f"Connection exhaustion test {conn_id}",
                        )
                        saved_ids.append(backtest_id)

                    # Perform queries
                    for backtest_id in saved_ids:
                        persistence.get_backtest_by_id(backtest_id)

                    # Hold connection for some time
                    await asyncio.sleep(2.0)

                    return {
                        "connection_id": conn_id,
                        "operations_completed": len(saved_ids) * 2,  # Save + retrieve
                        "success": True,
                    }

            except Exception as e:
                return {
                    "connection_id": conn_id,
                    "error": str(e),
                    "success": False,
                }

        # Create many concurrent database operations
        start_time = time.time()

        connection_tasks = [database_operation(i) for i in range(max_connections)]

        # Execute with timeout to prevent hanging
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*connection_tasks, return_exceptions=True), timeout=60.0
            )
        except TimeoutError:
            logger.warning("Database connection test timed out")
            results = []

        execution_time = time.time() - start_time

        # Analyze results
        successful_connections = sum(
            1 for r in results if isinstance(r, dict) and r.get("success", False)
        )
        failed_connections = len(results) - successful_connections

        total_operations = sum(
            r.get("operations_completed", 0)
            for r in results
            if isinstance(r, dict) and r.get("success", False)
        )

        return {
            "max_connections_attempted": max_connections,
            "successful_connections": successful_connections,
            "failed_connections": failed_connections,
            "connection_success_rate": successful_connections / max_connections
            if max_connections > 0
            else 0,
            "total_operations": total_operations,
            "execution_time": execution_time,
            "operations_per_second": total_operations / execution_time
            if execution_time > 0
            else 0,
        }

    async def file_descriptor_exhaustion_test(self) -> dict[str, Any]:
        """Test file descriptor usage patterns."""
        logger.info("Starting file descriptor exhaustion test")

        initial_snapshot = self.resource_monitor.get_current_snapshot()
        initial_fds = initial_snapshot.file_descriptors

        self.resource_monitor.start_monitoring()

        # Get system file descriptor limit
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        except Exception:
            soft_limit, hard_limit = 1024, 4096  # Default assumptions

        logger.info(
            f"FD limits - Soft: {soft_limit}, Hard: {hard_limit}, Initial: {initial_fds}"
        )

        try:
            engine = VectorBTEngine(data_provider=self.data_provider)

            # Run many operations to stress file descriptor usage
            fd_measurements = []
            max_operations = min(100, soft_limit // 10)  # Conservative approach

            for i in range(max_operations):
                await engine.run_backtest(
                    symbol=f"FD_TEST_{i}",
                    strategy_type="sma_cross",
                    parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                )

                if i % 10 == 0:
                    snapshot = self.resource_monitor.get_current_snapshot()
                    fd_measurements.append(
                        {
                            "iteration": i,
                            "file_descriptors": snapshot.file_descriptors,
                            "fd_growth": snapshot.file_descriptors - initial_fds,
                        }
                    )

                    if snapshot.file_descriptors > soft_limit * 0.8:
                        logger.warning(
                            f"High FD usage detected: {snapshot.file_descriptors}/{soft_limit}"
                        )

        finally:
            self.resource_monitor.stop_monitoring()

        final_snapshot = self.resource_monitor.get_current_snapshot()
        final_fds = final_snapshot.file_descriptors
        fd_growth = final_fds - initial_fds

        # Analyze FD usage pattern
        peak_fds = max([m["file_descriptors"] for m in fd_measurements] + [final_fds])
        fd_utilization = peak_fds / soft_limit if soft_limit > 0 else 0

        return {
            "initial_file_descriptors": initial_fds,
            "final_file_descriptors": final_fds,
            "peak_file_descriptors": peak_fds,
            "fd_growth": fd_growth,
            "soft_limit": soft_limit,
            "hard_limit": hard_limit,
            "fd_utilization_percent": fd_utilization * 100,
            "fd_measurements": fd_measurements,
            "operations_completed": max_operations,
        }


class TestStressTesting:
    """Stress testing suite."""

    @pytest.fixture
    async def stress_data_provider(self):
        """Create data provider optimized for stress testing."""
        provider = Mock()

        # Cache to reduce computation overhead during stress tests
        data_cache = {}

        def get_stress_test_data(symbol: str) -> pd.DataFrame:
            """Get cached data for stress testing."""
            if symbol not in data_cache:
                # Generate smaller dataset for faster stress testing
                dates = pd.date_range(start="2023-06-01", end="2023-12-31", freq="D")
                seed = hash(symbol) % 1000
                np.random.seed(seed)

                returns = np.random.normal(0.001, 0.02, len(dates))
                prices = 100 * np.cumprod(1 + returns)

                data_cache[symbol] = pd.DataFrame(
                    {
                        "Open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                        "High": prices * np.random.uniform(1.01, 1.03, len(dates)),
                        "Low": prices * np.random.uniform(0.97, 0.99, len(dates)),
                        "Close": prices,
                        "Volume": np.random.randint(1000000, 5000000, len(dates)),
                        "Adj Close": prices,
                    },
                    index=dates,
                )

            return data_cache[symbol].copy()

        provider.get_stock_data.side_effect = get_stress_test_data
        return provider

    @pytest.mark.slow
    async def test_sustained_load_15_minutes(self, stress_data_provider):
        """Test sustained load for 15 minutes (abbreviated from 1 hour for CI)."""
        stress_runner = StressTestRunner(stress_data_provider)

        result = await stress_runner.sustained_load_test(
            duration_minutes=15,  # Reduced for CI/testing
            concurrent_load=8,
        )

        # Assertions for sustained load
        assert result["error_rate"] <= 0.05, (
            f"Error rate too high: {result['error_rate']:.3f}"
        )
        assert result["operations_per_minute"] >= 10, (
            f"Throughput too low: {result['operations_per_minute']:.1f} ops/min"
        )

        # Resource growth should be reasonable
        trends = result["resource_trends"]
        assert trends["memory_growth_rate_mb_per_hour"] <= 100, (
            f"Memory growth rate too high: {trends['memory_growth_rate_mb_per_hour']:.1f} MB/hour"
        )

        logger.info(
            f"✓ Sustained load test completed: {result['total_operations']} operations in {result['duration_minutes']:.1f} minutes"
        )
        return result

    async def test_memory_leak_detection(self, stress_data_provider):
        """Test for memory leaks over many iterations."""
        stress_runner = StressTestRunner(stress_data_provider)

        result = await stress_runner.memory_leak_detection_test(iterations=200)

        # Memory leak assertions
        assert not result["leak_detected"], (
            f"Memory leak detected: {result['leak_per_1000_iterations_mb']:.2f} MB per 1000 iterations"
        )
        assert result["total_memory_growth_mb"] <= 300, (
            f"Total memory growth too high: {result['total_memory_growth_mb']:.1f} MB"
        )

        logger.info(
            f"✓ Memory leak test completed: {result['total_memory_growth_mb']:.1f}MB growth over {result['iterations']} iterations"
        )
        return result

    async def test_cpu_stress_resilience(self, stress_data_provider):
        """Test system resilience under CPU stress."""
        stress_runner = StressTestRunner(stress_data_provider)

        result = await stress_runner.cpu_stress_test(
            duration_minutes=5,  # Reduced for testing
            cpu_target=0.7,  # 70% CPU utilization
        )

        # CPU stress assertions
        assert result["error_rate"] <= 0.2, (
            f"Error rate too high under CPU stress: {result['error_rate']:.3f}"
        )
        assert result["avg_response_time"] <= 10.0, (
            f"Response time too slow under CPU stress: {result['avg_response_time']:.2f}s"
        )
        assert result["operations_completed"] >= 10, (
            f"Too few operations completed: {result['operations_completed']}"
        )

        logger.info(
            f"✓ CPU stress test completed: {result['operations_completed']} operations with {result['avg_cpu_utilization']:.1f}% avg CPU"
        )
        return result

    async def test_database_connection_stress(self, stress_data_provider, db_session):
        """Test database performance under connection stress."""
        stress_runner = StressTestRunner(stress_data_provider)

        result = await stress_runner.database_connection_exhaustion_test(
            db_session=db_session,
            max_connections=20,  # Reduced for testing
        )

        # Database stress assertions
        assert result["connection_success_rate"] >= 0.8, (
            f"Connection success rate too low: {result['connection_success_rate']:.3f}"
        )
        assert result["operations_per_second"] >= 5.0, (
            f"Database throughput too low: {result['operations_per_second']:.2f} ops/s"
        )

        logger.info(
            f"✓ Database stress test completed: {result['successful_connections']}/{result['max_connections_attempted']} connections succeeded"
        )
        return result

    async def test_file_descriptor_management(self, stress_data_provider):
        """Test file descriptor usage under stress."""
        stress_runner = StressTestRunner(stress_data_provider)

        result = await stress_runner.file_descriptor_exhaustion_test()

        # File descriptor assertions
        assert result["fd_utilization_percent"] <= 50.0, (
            f"FD utilization too high: {result['fd_utilization_percent']:.1f}%"
        )
        assert result["fd_growth"] <= 100, f"FD growth too high: {result['fd_growth']}"

        logger.info(
            f"✓ File descriptor test completed: {result['peak_file_descriptors']} peak FDs ({result['fd_utilization_percent']:.1f}% utilization)"
        )
        return result

    async def test_queue_overflow_scenarios(self, stress_data_provider):
        """Test queue management under overflow conditions."""
        # Simulate queue overflow by creating more tasks than can be processed
        max_queue_size = 50
        overflow_tasks = 100

        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
        processed_tasks = 0
        overflow_errors = 0

        async def queue_task(task_id: int):
            nonlocal processed_tasks, overflow_errors

            try:
                async with semaphore:
                    engine = VectorBTEngine(data_provider=stress_data_provider)

                    await engine.run_backtest(
                        symbol=f"QUEUE_{task_id % 10}",
                        strategy_type="sma_cross",
                        parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                        start_date="2023-06-01",
                        end_date="2023-12-31",
                    )

                    processed_tasks += 1

            except Exception as e:
                overflow_errors += 1
                logger.error(f"Queue task {task_id} failed: {e}")

        # Create tasks faster than they can be processed
        start_time = time.time()

        tasks = []
        for i in range(overflow_tasks):
            task = asyncio.create_task(queue_task(i))
            tasks.append(task)

            # Create tasks rapidly to test queue management
            if i < max_queue_size:
                await asyncio.sleep(0.01)  # Rapid creation
            else:
                await asyncio.sleep(0.1)  # Slower creation after queue fills

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time

        # Queue overflow assertions
        processing_success_rate = processed_tasks / overflow_tasks
        assert processing_success_rate >= 0.8, (
            f"Queue processing success rate too low: {processing_success_rate:.3f}"
        )
        assert execution_time < 120.0, (
            f"Queue processing took too long: {execution_time:.1f}s"
        )

        logger.info(
            f"✓ Queue overflow test completed: {processed_tasks}/{overflow_tasks} tasks processed in {execution_time:.1f}s"
        )

        return {
            "overflow_tasks": overflow_tasks,
            "processed_tasks": processed_tasks,
            "overflow_errors": overflow_errors,
            "processing_success_rate": processing_success_rate,
            "execution_time": execution_time,
        }

    async def test_comprehensive_stress_suite(self, stress_data_provider, db_session):
        """Run comprehensive stress testing suite."""
        logger.info("Starting Comprehensive Stress Testing Suite...")

        stress_results = {}

        # Run individual stress tests
        stress_results["sustained_load"] = await self.test_sustained_load_15_minutes(
            stress_data_provider
        )
        stress_results["memory_leak"] = await self.test_memory_leak_detection(
            stress_data_provider
        )
        stress_results["cpu_stress"] = await self.test_cpu_stress_resilience(
            stress_data_provider
        )
        stress_results["database_stress"] = await self.test_database_connection_stress(
            stress_data_provider, db_session
        )
        stress_results["file_descriptors"] = await self.test_file_descriptor_management(
            stress_data_provider
        )
        stress_results["queue_overflow"] = await self.test_queue_overflow_scenarios(
            stress_data_provider
        )

        # Aggregate stress test analysis
        total_tests = len(stress_results)
        passed_tests = 0
        critical_failures = []

        for test_name, result in stress_results.items():
            # Simple pass/fail based on whether test completed without major issues
            test_passed = True

            if test_name == "sustained_load" and result["error_rate"] > 0.1:
                test_passed = False
                critical_failures.append(
                    f"Sustained load error rate: {result['error_rate']:.3f}"
                )
            elif test_name == "memory_leak" and result["leak_detected"]:
                test_passed = False
                critical_failures.append(
                    f"Memory leak detected: {result['leak_per_1000_iterations_mb']:.2f} MB/1k iterations"
                )
            elif test_name == "cpu_stress" and result["error_rate"] > 0.3:
                test_passed = False
                critical_failures.append(
                    f"CPU stress error rate: {result['error_rate']:.3f}"
                )

            if test_passed:
                passed_tests += 1

        overall_pass_rate = passed_tests / total_tests

        logger.info(
            f"\n{'=' * 60}\n"
            f"COMPREHENSIVE STRESS TEST REPORT\n"
            f"{'=' * 60}\n"
            f"Total Tests: {total_tests}\n"
            f"Passed: {passed_tests}\n"
            f"Overall Pass Rate: {overall_pass_rate:.1%}\n"
            f"Critical Failures: {len(critical_failures)}\n"
            f"{'=' * 60}\n"
        )

        # Assert overall stress test success
        assert overall_pass_rate >= 0.8, (
            f"Overall stress test pass rate too low: {overall_pass_rate:.1%}"
        )
        assert len(critical_failures) <= 1, (
            f"Too many critical failures: {critical_failures}"
        )

        return {
            "overall_pass_rate": overall_pass_rate,
            "critical_failures": critical_failures,
            "stress_results": stress_results,
        }


if __name__ == "__main__":
    # Run stress testing suite
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "--timeout=1800",  # 30 minute timeout for stress tests
            "-m",
            "not slow",  # Skip slow tests by default
        ]
    )
