"""
Profiling Tests for Bottleneck Identification.

This test suite covers:
- Profile critical code paths with cProfile
- Identify slow database queries with timing
- Find memory allocation hotspots
- Document optimization opportunities
- Line-by-line profiling of key functions
- Call graph analysis for performance
- I/O bottleneck identification
- CPU-bound vs I/O-bound analysis
"""

import asyncio
import cProfile
import io
import logging
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, patch
import sys

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.backtesting import VectorBTEngine, BacktestAnalyzer
from maverick_mcp.backtesting.persistence import BacktestPersistenceManager
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiler for backtesting operations."""

    def __init__(self):
        self.profiling_data = {}
        self.memory_snapshots = []

    @contextmanager
    def profile_cpu(self, operation_name: str):
        """Profile CPU usage of an operation."""
        profiler = cProfile.Profile()
        start_time = time.time()

        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            execution_time = time.time() - start_time

            # Capture profiling stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions

            self.profiling_data[operation_name] = {
                "execution_time": execution_time,
                "cpu_profile": stats_stream.getvalue(),
                "stats_object": stats,
            }

    @contextmanager
    def profile_memory(self, operation_name: str):
        """Profile memory usage of an operation."""
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()

        try:
            yield
        finally:
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_data = {
                "start_memory_mb": start_memory[0] / 1024 / 1024,
                "current_memory_mb": current_memory / 1024 / 1024,
                "peak_memory_mb": peak_memory / 1024 / 1024,
                "memory_growth_mb": (current_memory - start_memory[0]) / 1024 / 1024,
            }

            if operation_name in self.profiling_data:
                self.profiling_data[operation_name]["memory_profile"] = memory_data
            else:
                self.profiling_data[operation_name] = {"memory_profile": memory_data}

    def profile_database_query(self, query_name: str, query_func: Callable) -> Dict[str, Any]:
        """Profile database query performance."""
        start_time = time.time()

        try:
            result = query_func()
            execution_time = time.time() - start_time

            return {
                "query_name": query_name,
                "execution_time_ms": execution_time * 1000,
                "success": True,
                "result_size": len(str(result)) if result else 0,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "query_name": query_name,
                "execution_time_ms": execution_time * 1000,
                "success": False,
                "error": str(e),
            }

    def analyze_hotspots(self, operation_name: str) -> Dict[str, Any]:
        """Analyze performance hotspots from profiling data."""
        if operation_name not in self.profiling_data:
            return {"error": f"No profiling data for {operation_name}"}

        data = self.profiling_data[operation_name]
        stats = data.get("stats_object")

        if not stats:
            return {"error": "No CPU profiling stats available"}

        # Extract top functions by cumulative time
        stats.sort_stats('cumulative')
        top_functions = []

        for func_data in list(stats.stats.items())[:10]:
            func_name, (cc, nc, tt, ct, callers) = func_data
            top_functions.append({
                "function": f"{func_name[0]}:{func_name[1]}({func_name[2]})",
                "cumulative_time": ct,
                "total_time": tt,
                "call_count": nc,
                "time_per_call": ct / nc if nc > 0 else 0,
            })

        # Extract top functions by self time
        stats.sort_stats('tottime')
        self_time_functions = []

        for func_data in list(stats.stats.items())[:10]:
            func_name, (cc, nc, tt, ct, callers) = func_data
            self_time_functions.append({
                "function": f"{func_name[0]}:{func_name[1]}({func_name[2]})",
                "self_time": tt,
                "cumulative_time": ct,
                "call_count": nc,
            })

        return {
            "operation_name": operation_name,
            "total_execution_time": data.get("execution_time", 0),
            "top_functions_by_cumulative": top_functions,
            "top_functions_by_self_time": self_time_functions,
            "memory_profile": data.get("memory_profile", {}),
        }

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        optimization_opportunities = []
        performance_summary = {}

        for operation_name, data in self.profiling_data.items():
            analysis = self.analyze_hotspots(operation_name)

            performance_summary[operation_name] = {
                "execution_time": data.get("execution_time", 0),
                "peak_memory_mb": data.get("memory_profile", {}).get("peak_memory_mb", 0),
            }

            # Identify optimization opportunities
            if "top_functions_by_cumulative" in analysis:
                for func in analysis["top_functions_by_cumulative"][:3]:  # Top 3 functions
                    if func["cumulative_time"] > 0.1:  # More than 100ms
                        optimization_opportunities.append({
                            "operation": operation_name,
                            "function": func["function"],
                            "issue": "High cumulative time",
                            "time": func["cumulative_time"],
                            "priority": "High" if func["cumulative_time"] > 1.0 else "Medium",
                        })

            # Memory optimization opportunities
            memory_profile = data.get("memory_profile", {})
            if memory_profile.get("peak_memory_mb", 0) > 100:  # More than 100MB
                optimization_opportunities.append({
                    "operation": operation_name,
                    "issue": "High memory usage",
                    "memory_mb": memory_profile["peak_memory_mb"],
                    "priority": "High" if memory_profile["peak_memory_mb"] > 500 else "Medium",
                })

        return {
            "performance_summary": performance_summary,
            "optimization_opportunities": optimization_opportunities,
            "total_operations_profiled": len(self.profiling_data),
        }


class TestPerformanceProfiling:
    """Performance profiling test suite."""

    @pytest.fixture
    async def profiling_data_provider(self):
        """Create data provider for profiling tests."""
        provider = Mock()

        def generate_profiling_data(symbol: str) -> pd.DataFrame:
            """Generate data with known performance characteristics."""
            # Generate larger dataset to create measurable performance impact
            dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")  # 4 years
            np.random.seed(hash(symbol) % 1000)

            returns = np.random.normal(0.0008, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)

            # Add some computationally expensive operations
            high_prices = prices * np.random.uniform(1.01, 1.05, len(dates))
            low_prices = prices * np.random.uniform(0.95, 0.99, len(dates))

            # Simulate expensive volume calculations
            base_volume = np.random.randint(1000000, 10000000, len(dates))
            volume_multiplier = np.exp(np.random.normal(0, 0.1, len(dates)))  # Log-normal distribution
            volumes = (base_volume * volume_multiplier).astype(int)

            return pd.DataFrame({
                "Open": prices * np.random.uniform(0.995, 1.005, len(dates)),
                "High": high_prices,
                "Low": low_prices,
                "Close": prices,
                "Volume": volumes,
                "Adj Close": prices,
            }, index=dates)

        provider.get_stock_data.side_effect = generate_profiling_data
        return provider

    async def test_profile_backtest_execution(self, profiling_data_provider):
        """Profile complete backtest execution to identify bottlenecks."""
        profiler = PerformanceProfiler()
        engine = VectorBTEngine(data_provider=profiling_data_provider)

        strategies_to_profile = ["sma_cross", "rsi", "macd", "bollinger"]

        for strategy in strategies_to_profile:
            with profiler.profile_cpu(f"backtest_{strategy}"):
                with profiler.profile_memory(f"backtest_{strategy}"):
                    result = await engine.run_backtest(
                        symbol="PROFILE_TEST",
                        strategy_type=strategy,
                        parameters=STRATEGY_TEMPLATES[strategy]["parameters"],
                        start_date="2022-01-01",
                        end_date="2023-12-31",
                    )

        # Analyze profiling results
        report = profiler.generate_optimization_report()

        # Log performance analysis
        logger.info("Backtest Execution Profiling Results:")
        for operation, summary in report["performance_summary"].items():
            logger.info(f"  {operation}: {summary['execution_time']:.3f}s, "
                       f"{summary['peak_memory_mb']:.1f}MB peak")

        # Log optimization opportunities
        if report["optimization_opportunities"]:
            logger.info("Optimization Opportunities:")
            for opportunity in report["optimization_opportunities"]:
                priority_symbol = "🔴" if opportunity["priority"] == "High" else "🟡"
                logger.info(f"  {priority_symbol} {opportunity['operation']}: {opportunity['issue']}")

        # Performance assertions
        max_execution_time = max(
            summary["execution_time"] for summary in report["performance_summary"].values()
        )
        assert max_execution_time <= 5.0, f"Slowest backtest took too long: {max_execution_time:.2f}s"

        high_priority_issues = [
            opp for opp in report["optimization_opportunities"] if opp["priority"] == "High"
        ]
        assert len(high_priority_issues) <= 2, f"Too many high-priority performance issues: {len(high_priority_issues)}"

        return report

    async def test_profile_data_loading_bottlenecks(self, profiling_data_provider):
        """Profile data loading operations to identify I/O bottlenecks."""
        profiler = PerformanceProfiler()
        engine = VectorBTEngine(data_provider=profiling_data_provider)

        symbols = ["DATA_1", "DATA_2", "DATA_3", "DATA_4", "DATA_5"]

        # Profile data loading operations
        for symbol in symbols:
            with profiler.profile_cpu(f"data_loading_{symbol}"):
                with profiler.profile_memory(f"data_loading_{symbol}"):
                    # Profile the data fetching specifically
                    data = await engine.get_historical_data(
                        symbol=symbol,
                        start_date="2020-01-01",
                        end_date="2023-12-31"
                    )

        # Analyze data loading performance
        data_loading_times = []
        data_loading_memory = []

        for symbol in symbols:
            operation_name = f"data_loading_{symbol}"
            if operation_name in profiler.profiling_data:
                data_loading_times.append(profiler.profiling_data[operation_name]["execution_time"])
                memory_profile = profiler.profiling_data[operation_name].get("memory_profile", {})
                data_loading_memory.append(memory_profile.get("peak_memory_mb", 0))

        avg_loading_time = np.mean(data_loading_times) if data_loading_times else 0
        max_loading_time = max(data_loading_times) if data_loading_times else 0
        avg_loading_memory = np.mean(data_loading_memory) if data_loading_memory else 0

        logger.info(f"Data Loading Performance Analysis:")
        logger.info(f"  Average Loading Time: {avg_loading_time:.3f}s")
        logger.info(f"  Maximum Loading Time: {max_loading_time:.3f}s")
        logger.info(f"  Average Memory Usage: {avg_loading_memory:.1f}MB")

        # Performance assertions for data loading
        assert avg_loading_time <= 0.5, f"Average data loading too slow: {avg_loading_time:.3f}s"
        assert max_loading_time <= 1.0, f"Slowest data loading too slow: {max_loading_time:.3f}s"
        assert avg_loading_memory <= 50.0, f"Data loading memory usage too high: {avg_loading_memory:.1f}MB"

        return {
            "avg_loading_time": avg_loading_time,
            "max_loading_time": max_loading_time,
            "avg_loading_memory": avg_loading_memory,
            "individual_times": data_loading_times,
        }

    async def test_profile_database_query_performance(self, profiling_data_provider, db_session):
        """Profile database queries to identify slow operations."""
        profiler = PerformanceProfiler()
        engine = VectorBTEngine(data_provider=profiling_data_provider)

        # Generate test data for database profiling
        test_results = []
        for i in range(10):
            result = await engine.run_backtest(
                symbol=f"DB_PROFILE_{i}",
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            test_results.append(result)

        # Profile database operations
        query_profiles = []

        with BacktestPersistenceManager(session=db_session) as persistence:
            # Profile save operations
            for i, result in enumerate(test_results):
                query_profile = profiler.profile_database_query(
                    f"save_backtest_{i}",
                    lambda r=result: persistence.save_backtest_result(
                        vectorbt_results=r,
                        execution_time=2.0,
                        notes="Database profiling test",
                    )
                )
                query_profiles.append(query_profile)

            # Get saved IDs for retrieval profiling
            saved_ids = [qp.get("result") for qp in query_profiles if qp.get("success")]

            # Profile retrieval operations
            for i, backtest_id in enumerate(saved_ids[:5]):  # Profile first 5 retrievals
                query_profile = profiler.profile_database_query(
                    f"retrieve_backtest_{i}",
                    lambda bid=backtest_id: persistence.get_backtest_by_id(bid)
                )
                query_profiles.append(query_profile)

            # Profile bulk query operations
            bulk_query_profile = profiler.profile_database_query(
                "bulk_query_by_strategy",
                lambda: persistence.get_backtests_by_strategy("sma_cross")
            )
            query_profiles.append(bulk_query_profile)

        # Analyze database query performance
        save_times = [qp["execution_time_ms"] for qp in query_profiles if "save_backtest" in qp["query_name"] and qp["success"]]
        retrieve_times = [qp["execution_time_ms"] for qp in query_profiles if "retrieve_backtest" in qp["query_name"] and qp["success"]]

        avg_save_time = np.mean(save_times) if save_times else 0
        avg_retrieve_time = np.mean(retrieve_times) if retrieve_times else 0
        bulk_query_time = bulk_query_profile["execution_time_ms"] if bulk_query_profile["success"] else 0

        logger.info(f"Database Query Performance Analysis:")
        logger.info(f"  Average Save Time: {avg_save_time:.1f}ms")
        logger.info(f"  Average Retrieve Time: {avg_retrieve_time:.1f}ms")
        logger.info(f"  Bulk Query Time: {bulk_query_time:.1f}ms")

        # Identify slow queries
        slow_queries = [qp for qp in query_profiles if qp["execution_time_ms"] > 100 and qp["success"]]
        logger.info(f"  Slow Queries (>100ms): {len(slow_queries)}")

        # Performance assertions for database queries
        assert avg_save_time <= 50.0, f"Average save time too slow: {avg_save_time:.1f}ms"
        assert avg_retrieve_time <= 20.0, f"Average retrieve time too slow: {avg_retrieve_time:.1f}ms"
        assert bulk_query_time <= 100.0, f"Bulk query too slow: {bulk_query_time:.1f}ms"
        assert len(slow_queries) <= 2, f"Too many slow queries: {len(slow_queries)}"

        return {
            "avg_save_time": avg_save_time,
            "avg_retrieve_time": avg_retrieve_time,
            "bulk_query_time": bulk_query_time,
            "slow_queries": len(slow_queries),
            "query_profiles": query_profiles,
        }

    async def test_profile_memory_allocation_patterns(self, profiling_data_provider):
        """Profile memory allocation patterns to identify hotspots."""
        profiler = PerformanceProfiler()
        engine = VectorBTEngine(data_provider=profiling_data_provider)

        # Test different memory usage patterns
        memory_test_cases = [
            ("small_dataset", "2023-06-01", "2023-12-31"),
            ("medium_dataset", "2022-01-01", "2023-12-31"),
            ("large_dataset", "2020-01-01", "2023-12-31"),
        ]

        memory_profiles = []

        for case_name, start_date, end_date in memory_test_cases:
            with profiler.profile_memory(f"memory_{case_name}"):
                result = await engine.run_backtest(
                    symbol="MEMORY_TEST",
                    strategy_type="macd",
                    parameters=STRATEGY_TEMPLATES["macd"]["parameters"],
                    start_date=start_date,
                    end_date=end_date,
                )

            memory_data = profiler.profiling_data[f"memory_{case_name}"]["memory_profile"]
            memory_profiles.append({
                "case": case_name,
                "peak_memory_mb": memory_data["peak_memory_mb"],
                "memory_growth_mb": memory_data["memory_growth_mb"],
                "data_points": len(pd.date_range(start=start_date, end=end_date, freq="D")),
            })

        # Analyze memory scaling
        data_points = [mp["data_points"] for mp in memory_profiles]
        peak_memories = [mp["peak_memory_mb"] for mp in memory_profiles]

        # Calculate memory efficiency (MB per 1000 data points)
        memory_efficiency = [
            (peak_mem / data_pts * 1000) for peak_mem, data_pts in zip(peak_memories, data_points)
        ]

        avg_memory_efficiency = np.mean(memory_efficiency)

        logger.info(f"Memory Allocation Pattern Analysis:")
        for profile in memory_profiles:
            efficiency = profile["peak_memory_mb"] / profile["data_points"] * 1000
            logger.info(f"  {profile['case']}: {profile['peak_memory_mb']:.1f}MB peak "
                       f"({efficiency:.2f} MB/1k points)")

        logger.info(f"  Average Memory Efficiency: {avg_memory_efficiency:.2f} MB per 1000 data points")

        # Memory efficiency assertions
        assert avg_memory_efficiency <= 5.0, f"Memory efficiency too poor: {avg_memory_efficiency:.2f} MB/1k points"
        assert max(peak_memories) <= 200.0, f"Peak memory usage too high: {max(peak_memories):.1f}MB"

        return {
            "memory_profiles": memory_profiles,
            "avg_memory_efficiency": avg_memory_efficiency,
            "peak_memory_usage": max(peak_memories),
        }

    async def test_profile_cpu_vs_io_bound_operations(self, profiling_data_provider):
        """Profile CPU-bound vs I/O-bound operations to optimize resource usage."""
        profiler = PerformanceProfiler()
        engine = VectorBTEngine(data_provider=profiling_data_provider)

        # Profile CPU-intensive strategy
        with profiler.profile_cpu("cpu_intensive_strategy"):
            cpu_result = await engine.run_backtest(
                symbol="CPU_TEST",
                strategy_type="bollinger",  # More calculations
                parameters=STRATEGY_TEMPLATES["bollinger"]["parameters"],
                start_date="2022-01-01",
                end_date="2023-12-31",
            )

        # Profile I/O-intensive operations (multiple data fetches)
        with profiler.profile_cpu("io_intensive_operations"):
            io_symbols = ["IO_1", "IO_2", "IO_3", "IO_4", "IO_5"]
            io_results = []

            for symbol in io_symbols:
                result = await engine.run_backtest(
                    symbol=symbol,
                    strategy_type="sma_cross",  # Simpler calculations
                    parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                    start_date="2023-06-01",
                    end_date="2023-12-31",
                )
                io_results.append(result)

        # Analyze CPU vs I/O characteristics
        cpu_analysis = profiler.analyze_hotspots("cpu_intensive_strategy")
        io_analysis = profiler.analyze_hotspots("io_intensive_operations")

        cpu_time = cpu_analysis.get("total_execution_time", 0)
        io_time = io_analysis.get("total_execution_time", 0)

        # Analyze function call patterns
        cpu_top_functions = cpu_analysis.get("top_functions_by_cumulative", [])
        io_top_functions = io_analysis.get("top_functions_by_cumulative", [])

        # Calculate I/O vs CPU characteristics
        cpu_bound_ratio = cpu_time / (cpu_time + io_time) if (cpu_time + io_time) > 0 else 0

        logger.info(f"CPU vs I/O Bound Analysis:")
        logger.info(f"  CPU-Intensive Operation: {cpu_time:.3f}s")
        logger.info(f"  I/O-Intensive Operations: {io_time:.3f}s")
        logger.info(f"  CPU-Bound Ratio: {cpu_bound_ratio:.2%}")

        logger.info(f"  Top CPU-Intensive Functions:")
        for func in cpu_top_functions[:3]:
            logger.info(f"    {func['function']}: {func['cumulative_time']:.3f}s")

        logger.info(f"  Top I/O-Intensive Functions:")
        for func in io_top_functions[:3]:
            logger.info(f"    {func['function']}: {func['cumulative_time']:.3f}s")

        # Performance balance assertions
        assert cpu_time <= 3.0, f"CPU-intensive operation too slow: {cpu_time:.3f}s"
        assert io_time <= 5.0, f"I/O-intensive operations too slow: {io_time:.3f}s"

        return {
            "cpu_time": cpu_time,
            "io_time": io_time,
            "cpu_bound_ratio": cpu_bound_ratio,
            "cpu_top_functions": cpu_top_functions[:5],
            "io_top_functions": io_top_functions[:5],
        }

    async def test_comprehensive_profiling_suite(self, profiling_data_provider, db_session):
        """Run comprehensive profiling suite and generate optimization report."""
        logger.info("Starting Comprehensive Performance Profiling Suite...")

        profiling_results = {}

        # Run all profiling tests
        profiling_results["backtest_execution"] = await self.test_profile_backtest_execution(profiling_data_provider)
        profiling_results["data_loading"] = await self.test_profile_data_loading_bottlenecks(profiling_data_provider)
        profiling_results["database_queries"] = await self.test_profile_database_query_performance(profiling_data_provider, db_session)
        profiling_results["memory_allocation"] = await self.test_profile_memory_allocation_patterns(profiling_data_provider)
        profiling_results["cpu_vs_io"] = await self.test_profile_cpu_vs_io_bound_operations(profiling_data_provider)

        # Generate comprehensive optimization report
        optimization_report = {
            "executive_summary": {
                "profiling_areas": len(profiling_results),
                "performance_bottlenecks": [],
                "optimization_priorities": [],
            },
            "detailed_analysis": profiling_results,
        }

        # Identify key bottlenecks and priorities
        bottlenecks = []
        priorities = []

        # Analyze backtest execution performance
        backtest_report = profiling_results["backtest_execution"]
        high_priority_issues = [
            opp for opp in backtest_report.get("optimization_opportunities", [])
            if opp["priority"] == "High"
        ]
        if high_priority_issues:
            bottlenecks.append("High-priority performance issues in backtest execution")
            priorities.append("Optimize hot functions in strategy calculations")

        # Analyze data loading performance
        data_loading = profiling_results["data_loading"]
        if data_loading["max_loading_time"] > 0.8:
            bottlenecks.append("Slow data loading operations")
            priorities.append("Implement data caching or optimize data provider")

        # Analyze database performance
        db_performance = profiling_results["database_queries"]
        if db_performance["slow_queries"] > 1:
            bottlenecks.append("Multiple slow database queries detected")
            priorities.append("Add database indexes or optimize query patterns")

        # Analyze memory efficiency
        memory_analysis = profiling_results["memory_allocation"]
        if memory_analysis["avg_memory_efficiency"] > 3.0:
            bottlenecks.append("High memory usage per data point")
            priorities.append("Optimize memory allocation patterns")

        optimization_report["executive_summary"]["performance_bottlenecks"] = bottlenecks
        optimization_report["executive_summary"]["optimization_priorities"] = priorities

        # Log comprehensive report
        logger.info(
            f"\n{'='*60}\n"
            f"COMPREHENSIVE PROFILING REPORT\n"
            f"{'='*60}\n"
            f"Profiling Areas Analyzed: {len(profiling_results)}\n"
            f"Performance Bottlenecks: {len(bottlenecks)}\n"
            f"{'='*60}\n"
        )

        if bottlenecks:
            logger.info("🔍 PERFORMANCE BOTTLENECKS IDENTIFIED:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                logger.info(f"  {i}. {bottleneck}")

        if priorities:
            logger.info("\n🎯 OPTIMIZATION PRIORITIES:")
            for i, priority in enumerate(priorities, 1):
                logger.info(f"  {i}. {priority}")

        logger.info(f"\n{'='*60}")

        # Assert profiling success
        assert len(bottlenecks) <= 3, f"Too many performance bottlenecks identified: {len(bottlenecks)}"

        return optimization_report


if __name__ == "__main__":
    # Run profiling tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--timeout=300",  # 5 minute timeout for profiling tests
    ])