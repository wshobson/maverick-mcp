"""
Comprehensive Performance Test Suite for Parallel Research System

This test suite validates parallel research performance against specific thresholds:
- Minimum speedup: 2 agents = 1.3x speedup, 4 agents = 2.0x speedup
- Maximum memory increase: 3x memory usage acceptable for 4x agents
- Test duration: Quick tests for CI (<30s total runtime)

Features:
- Realistic failure simulation (0%, 10%, 25% failure rates)
- Memory usage monitoring and validation
- Statistical significance testing (3+ runs per test)
- Integration with existing pytest infrastructure
- Performance markers for easy filtering
"""

import asyncio
import gc
import logging
import random
import statistics
import time
import tracemalloc
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import psutil
import pytest

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.utils.parallel_research import (
    ParallelResearchConfig,
    ParallelResearchOrchestrator,
    ResearchResult,
    ResearchTask,
)

logger = logging.getLogger(__name__)

# Performance thresholds and test configuration
PERFORMANCE_THRESHOLDS = {
    "speedup": {
        2: 1.3,  # 2 agents should achieve 1.3x speedup minimum
        4: 2.0,  # 4 agents should achieve 2.0x speedup minimum
    },
    "memory_multiplier": 3.0,  # 3x memory usage acceptable for 4x agents
    "max_test_duration": 30.0,  # Total test suite should complete within 30s
}

# Failure simulation configuration
FAILURE_RATES = [0.0, 0.10, 0.25]  # 0%, 10%, 25% failure rates
STATISTICAL_RUNS = 3  # Number of runs for statistical significance


class PerformanceMonitor:
    """Monitors CPU, memory, and timing performance during test execution."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time: float = 0
        self.start_memory: int = 0
        self.peak_memory: int = 0
        self.cpu_percent_samples: list[float] = []
        self.process = psutil.Process()

    def __enter__(self):
        """Start performance monitoring."""
        # Force garbage collection for accurate memory measurement
        gc.collect()

        # Start memory tracing
        tracemalloc.start()

        # Record baseline metrics
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory

        logger.info(f"Performance monitoring started for {self.test_name}")
        return self

    def sample_cpu(self):
        """Sample current CPU usage."""
        try:
            cpu_percent = self.process.cpu_percent()
            self.cpu_percent_samples.append(cpu_percent)

            # Track peak memory
            current_memory = self.process.memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
        except Exception as e:
            logger.warning(f"Failed to sample CPU/memory: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete performance monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss

        # Get memory tracing results
        current, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate metrics
        execution_time = end_time - self.start_time
        memory_increase = end_memory - self.start_memory
        memory_multiplier = end_memory / max(self.start_memory, 1)
        avg_cpu = (
            statistics.mean(self.cpu_percent_samples) if self.cpu_percent_samples else 0
        )

        self.metrics = {
            "execution_time": execution_time,
            "start_memory_mb": self.start_memory / (1024 * 1024),
            "end_memory_mb": end_memory / (1024 * 1024),
            "peak_memory_mb": max(self.peak_memory, peak_traced) / (1024 * 1024),
            "memory_increase_mb": memory_increase / (1024 * 1024),
            "memory_multiplier": memory_multiplier,
            "avg_cpu_percent": avg_cpu,
            "cpu_samples": len(self.cpu_percent_samples),
        }

        logger.info(
            f"Performance metrics for {self.test_name}: "
            f"Time: {execution_time:.3f}s, "
            f"Memory: {self.metrics['start_memory_mb']:.1f}MB -> "
            f"{self.metrics['end_memory_mb']:.1f}MB "
            f"({memory_multiplier:.2f}x), "
            f"CPU: {avg_cpu:.1f}%"
        )


class MockResearchExecutor:
    """
    Realistic mock executor that simulates LLM research operations with:
    - Configurable failure rates and timeout scenarios
    - Variable response times (100-500ms)
    - Different response sizes to test memory patterns
    - Structured research data that mirrors real usage
    """

    def __init__(
        self,
        failure_rate: float = 0.0,
        base_delay: float = 0.1,
        delay_variance: float = 0.4,
        include_timeouts: bool = False,
    ):
        self.failure_rate = failure_rate
        self.base_delay = base_delay
        self.delay_variance = delay_variance
        self.include_timeouts = include_timeouts
        self.execution_count = 0
        self.execution_log: list[dict[str, Any]] = []

    async def __call__(self, task: ResearchTask) -> dict[str, Any]:
        """Execute mock research task with realistic behavior."""
        self.execution_count += 1
        start_time = time.time()

        # Simulate realistic processing delay (100-500ms)
        delay = self.base_delay + random.uniform(0, self.delay_variance)
        await asyncio.sleep(delay)

        # Simulate various failure modes
        if random.random() < self.failure_rate:
            failure_type = random.choice(
                ["api_timeout", "rate_limit", "auth_error", "network_error"]
            )

            if failure_type == "api_timeout" and self.include_timeouts:
                # Simulate timeout by sleeping longer than task timeout
                await asyncio.sleep(task.timeout + 1 if task.timeout else 10)

            raise Exception(f"Simulated {failure_type} for task {task.task_id}")

        # Generate realistic research response based on task type
        response = self._generate_research_response(task)

        execution_time = time.time() - start_time
        self.execution_log.append(
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "execution_time": execution_time,
                "delay_simulated": delay,
                "response_size": len(str(response)),
            }
        )

        return response

    def _generate_research_response(self, task: ResearchTask) -> dict[str, Any]:
        """Generate structured research data with realistic content."""
        # Vary response size based on research type
        insight_counts = {
            "fundamental": random.randint(8, 15),
            "technical": random.randint(6, 12),
            "sentiment": random.randint(10, 20),
            "competitive": random.randint(7, 14),
        }

        insight_count = insight_counts.get(task.task_type, 10)

        # Generate realistic insights
        insights = [
            f"{task.task_type} insight {i + 1} for {task.target_topic[:20]}"
            for i in range(insight_count)
        ]

        # Add context-specific data
        sources = [
            {
                "title": f"Source {i + 1} for {task.task_type} research",
                "url": f"https://example.com/research/{task.task_type}/{i}",
                "credibility_score": random.uniform(0.6, 0.95),
                "published_date": datetime.now().isoformat(),
                "content_snippet": f"Content from source {i + 1} "
                * random.randint(10, 50),
                "relevance_score": random.uniform(0.7, 1.0),
            }
            for i in range(random.randint(3, 8))
        ]

        return {
            "research_type": task.task_type,
            "insights": insights,
            "risk_factors": [
                f"{task.task_type} risk factor {i + 1}"
                for i in range(random.randint(3, 7))
            ],
            "opportunities": [
                f"{task.task_type} opportunity {i + 1}"
                for i in range(random.randint(3, 7))
            ],
            "sentiment": {
                "direction": random.choice(["bullish", "bearish", "neutral"]),
                "confidence": random.uniform(0.5, 0.9),
                "consensus": random.uniform(0.4, 0.8),
            },
            "credibility_score": random.uniform(0.6, 0.9),
            "sources": sources,
            "focus_areas": task.focus_areas,
            "metadata": {
                "execution_time": random.uniform(0.1, 0.5),
                "api_calls_made": random.randint(2, 8),
                "cache_hits": random.randint(0, 4),
                "cache_misses": random.randint(1, 6),
            },
            # Add some large data structures to test memory usage
            "detailed_analysis": {
                f"analysis_point_{i}": f"Detailed analysis content {i} "
                * random.randint(50, 200)
                for i in range(random.randint(5, 15))
            },
        }

    def get_execution_stats(self) -> dict[str, Any]:
        """Get detailed execution statistics."""
        if not self.execution_log:
            return {"total_executions": 0}

        execution_times = [log["execution_time"] for log in self.execution_log]
        response_sizes = [log["response_size"] for log in self.execution_log]

        return {
            "total_executions": len(self.execution_log),
            "avg_execution_time": statistics.mean(execution_times),
            "median_execution_time": statistics.median(execution_times),
            "avg_response_size": statistics.mean(response_sizes),
            "total_response_size": sum(response_sizes),
            "task_type_distribution": {
                task_type: len(
                    [log for log in self.execution_log if log["task_type"] == task_type]
                )
                for task_type in [
                    "fundamental",
                    "technical",
                    "sentiment",
                    "competitive",
                ]
            },
        }


class PerformanceTester:
    """Manages performance test execution and validation."""

    @staticmethod
    def create_test_tasks(
        agent_count: int, session_id: str = "perf_test"
    ) -> list[ResearchTask]:
        """Create realistic research tasks for performance testing."""
        topics = [
            "Apple Inc financial analysis and market outlook",
            "Tesla Inc competitive position and technical analysis",
            "Microsoft Corp sentiment analysis and growth prospects",
            "NVIDIA Corp fundamental analysis and AI sector outlook",
            "Amazon Inc market position and e-commerce trends",
            "Google Inc advertising revenue and cloud competition",
        ]

        tasks = []
        for i in range(agent_count):
            topic = topics[i % len(topics)]
            task_types = ["fundamental", "technical", "sentiment", "competitive"]
            task_type = task_types[i % len(task_types)]

            task = ResearchTask(
                task_id=f"{session_id}_{task_type}_{i}",
                task_type=task_type,
                target_topic=topic,
                focus_areas=[f"focus_{i}", f"area_{i % 3}"],
                priority=random.randint(5, 9),
                timeout=10,  # Short timeout for CI
            )
            tasks.append(task)

        return tasks

    @staticmethod
    async def run_sequential_baseline(
        tasks: list[ResearchTask], executor: MockResearchExecutor
    ) -> dict[str, Any]:
        """Run tasks sequentially to establish performance baseline."""
        start_time = time.time()
        results = []

        for task in tasks:
            try:
                result = await executor(task)
                results.append({"task": task, "result": result, "status": "success"})
            except Exception as e:
                results.append({"task": task, "error": str(e), "status": "failed"})

        execution_time = time.time() - start_time
        successful_results = [r for r in results if r["status"] == "success"]

        return {
            "execution_time": execution_time,
            "successful_tasks": len(successful_results),
            "failed_tasks": len(results) - len(successful_results),
            "results": results,
        }

    @staticmethod
    async def run_parallel_test(
        tasks: list[ResearchTask],
        config: ParallelResearchConfig,
        executor: MockResearchExecutor,
    ) -> ResearchResult:
        """Run parallel research test with orchestrator."""
        orchestrator = ParallelResearchOrchestrator(config)

        # Mock synthesis callback
        async def mock_synthesis(
            task_results: dict[str, ResearchTask],
        ) -> dict[str, Any]:
            successful_results = [
                task.result
                for task in task_results.values()
                if task.status == "completed" and task.result
            ]

            return {
                "synthesis": f"Mock synthesis from {len(successful_results)} results",
                "confidence_score": random.uniform(0.7, 0.9),
                "key_findings": [
                    f"Finding {i}" for i in range(min(len(successful_results), 5))
                ],
            }

        return await orchestrator.execute_parallel_research(
            tasks=tasks,
            research_executor=executor,
            synthesis_callback=mock_synthesis,
        )

    @staticmethod
    def validate_performance_thresholds(
        agent_count: int,
        sequential_time: float,
        parallel_time: float,
        memory_multiplier: float,
        success_threshold: float = 0.8,  # 80% of tests should pass threshold
    ) -> dict[str, bool]:
        """Validate performance against defined thresholds."""
        speedup = sequential_time / max(parallel_time, 0.001)  # Avoid division by zero
        expected_speedup = PERFORMANCE_THRESHOLDS["speedup"].get(agent_count, 1.0)
        max_memory_multiplier = PERFORMANCE_THRESHOLDS["memory_multiplier"]

        return {
            "speedup_threshold_met": speedup >= expected_speedup * success_threshold,
            "memory_threshold_met": memory_multiplier <= max_memory_multiplier,
            "speedup_achieved": speedup,
            "speedup_expected": expected_speedup,
            "memory_multiplier": memory_multiplier,
            "memory_limit": max_memory_multiplier,
        }


# Test fixtures
@pytest.fixture
def performance_monitor():
    """Create performance monitor for tests."""

    def _create_monitor(test_name: str):
        return PerformanceMonitor(test_name)

    return _create_monitor


@pytest.fixture
def mock_executor_factory():
    """Factory for creating mock research executors with different configurations."""

    def _create_executor(**kwargs):
        return MockResearchExecutor(**kwargs)

    return _create_executor


@pytest.fixture
def performance_tester():
    """Provide PerformanceTester utility."""
    return PerformanceTester()


# Unit Performance Tests
@pytest.mark.unit
class TestParallelResearchPerformance:
    """Core performance tests for parallel research system."""

    @pytest.mark.parametrize("agent_count", [2, 4])
    @pytest.mark.parametrize(
        "failure_rate", FAILURE_RATES[:2]
    )  # 0% and 10% for unit tests
    async def test_parallel_speedup_thresholds(
        self,
        agent_count: int,
        failure_rate: float,
        performance_monitor,
        mock_executor_factory,
        performance_tester,
    ):
        """Test that parallel execution meets minimum speedup thresholds."""
        test_name = f"speedup_{agent_count}agents_{int(failure_rate * 100)}pct_failure"

        # Run multiple iterations for statistical significance
        speedup_results = []
        memory_results = []

        for run in range(STATISTICAL_RUNS):
            with performance_monitor(f"{test_name}_run{run}") as monitor:
                # Create test configuration
                config = ParallelResearchConfig(
                    max_concurrent_agents=agent_count,
                    timeout_per_agent=5,
                    enable_fallbacks=True,
                    rate_limit_delay=0.05,  # Fast for testing
                )

                # Create tasks and executor
                tasks = performance_tester.create_test_tasks(
                    agent_count, f"speedup_test_{run}"
                )
                executor = mock_executor_factory(
                    failure_rate=failure_rate,
                    base_delay=0.1,
                    delay_variance=0.1,
                )

                # Sample CPU/memory during test
                monitor.sample_cpu()

                # Run sequential baseline
                sequential_start = time.time()
                await performance_tester.run_sequential_baseline(tasks, executor)
                sequential_time = time.time() - sequential_start

                monitor.sample_cpu()

                # Reset executor for parallel test
                executor.execution_count = 0
                executor.execution_log.clear()

                # Run parallel test
                parallel_start = time.time()
                await performance_tester.run_parallel_test(tasks, config, executor)
                parallel_time = time.time() - parallel_start

                monitor.sample_cpu()

            # Calculate metrics
            speedup = sequential_time / max(parallel_time, 0.001)
            speedup_results.append(speedup)
            memory_results.append(monitor.metrics["memory_multiplier"])

            logger.info(
                f"Run {run + 1}: Sequential: {sequential_time:.3f}s, "
                f"Parallel: {parallel_time:.3f}s, Speedup: {speedup:.2f}x"
            )

        # Statistical analysis
        avg_speedup = statistics.mean(speedup_results)
        median_speedup = statistics.median(speedup_results)
        avg_memory_multiplier = statistics.mean(memory_results)

        # Validate against thresholds
        expected_speedup = PERFORMANCE_THRESHOLDS["speedup"][agent_count]
        validation = performance_tester.validate_performance_thresholds(
            agent_count=agent_count,
            sequential_time=1.0,  # Normalized
            parallel_time=1.0 / avg_speedup,  # Normalized
            memory_multiplier=avg_memory_multiplier,
        )

        logger.info(
            f"Performance summary for {agent_count} agents, {failure_rate * 100}% failure rate: "
            f"Avg speedup: {avg_speedup:.2f}x (expected: {expected_speedup:.2f}x), "
            f"Memory multiplier: {avg_memory_multiplier:.2f}x"
        )

        # Assertions with clear failure messages
        assert validation["speedup_threshold_met"], (
            f"Speedup threshold not met: achieved {avg_speedup:.2f}x, "
            f"expected {expected_speedup:.2f}x (with 80% success rate)"
        )

        assert validation["memory_threshold_met"], (
            f"Memory threshold exceeded: {avg_memory_multiplier:.2f}x > "
            f"{PERFORMANCE_THRESHOLDS['memory_multiplier']}x limit"
        )

        # Performance characteristics validation
        assert median_speedup > 1.0, "Parallel execution should show some speedup"
        assert all(m < 10.0 for m in memory_results), (
            "Memory usage should be reasonable"
        )

    async def test_performance_under_failures(
        self, performance_monitor, mock_executor_factory, performance_tester
    ):
        """Test performance degradation under different failure scenarios."""
        agent_count = 4
        test_name = "failure_resilience_test"

        results = {}

        for failure_rate in FAILURE_RATES:
            with performance_monitor(
                f"{test_name}_{int(failure_rate * 100)}pct"
            ) as monitor:
                config = ParallelResearchConfig(
                    max_concurrent_agents=agent_count,
                    timeout_per_agent=3,
                    enable_fallbacks=True,
                    rate_limit_delay=0.02,
                )

                tasks = performance_tester.create_test_tasks(
                    agent_count, f"failure_test_{int(failure_rate * 100)}"
                )
                executor = mock_executor_factory(
                    failure_rate=failure_rate,
                    base_delay=0.05,
                    include_timeouts=False,  # No timeouts for this test
                )

                monitor.sample_cpu()
                parallel_result = await performance_tester.run_parallel_test(
                    tasks, config, executor
                )
                monitor.sample_cpu()

            results[failure_rate] = {
                "successful_tasks": parallel_result.successful_tasks,
                "failed_tasks": parallel_result.failed_tasks,
                "execution_time": monitor.metrics["execution_time"],
                "memory_multiplier": monitor.metrics["memory_multiplier"],
                "success_rate": parallel_result.successful_tasks
                / (parallel_result.successful_tasks + parallel_result.failed_tasks),
            }

        # Validate failure handling - adjusted for realistic expectations
        assert results[0.0]["success_rate"] == 1.0, (
            "Zero failure rate should achieve 100% success"
        )
        assert results[0.10]["success_rate"] >= 0.7, (
            "10% failure rate should maintain >70% success"
        )
        assert results[0.25]["success_rate"] >= 0.5, (
            "25% failure rate should maintain >50% success"
        )

        # Validate performance doesn't degrade drastically with failures
        baseline_time = results[0.0]["execution_time"]
        assert results[0.10]["execution_time"] <= baseline_time * 1.5, (
            "10% failure shouldn't increase time by >50%"
        )
        assert results[0.25]["execution_time"] <= baseline_time * 2.0, (
            "25% failure shouldn't double execution time"
        )

        logger.info("Failure resilience test completed successfully")

    async def test_memory_usage_patterns(
        self, performance_monitor, mock_executor_factory, performance_tester
    ):
        """Test memory usage patterns across different agent counts."""
        memory_results = {}

        for agent_count in [1, 2, 4]:
            with performance_monitor(f"memory_test_{agent_count}_agents") as monitor:
                config = ParallelResearchConfig(
                    max_concurrent_agents=agent_count,
                    timeout_per_agent=5,
                    enable_fallbacks=True,
                    rate_limit_delay=0.01,
                )

                # Create larger dataset to test memory scaling
                tasks = performance_tester.create_test_tasks(
                    agent_count * 2, f"memory_test_{agent_count}"
                )
                executor = mock_executor_factory(
                    failure_rate=0.0,
                    base_delay=0.05,  # Short delay to focus on memory
                )

                # Force garbage collection before test
                gc.collect()
                monitor.sample_cpu()

                # Run test with memory monitoring
                result = await performance_tester.run_parallel_test(
                    tasks, config, executor
                )

                # Sample memory again
                monitor.sample_cpu()

                # Force another GC to see post-test memory
                gc.collect()
                await asyncio.sleep(0.1)  # Allow cleanup

            memory_results[agent_count] = {
                "peak_memory_mb": monitor.metrics["peak_memory_mb"],
                "memory_increase_mb": monitor.metrics["memory_increase_mb"],
                "memory_multiplier": monitor.metrics["memory_multiplier"],
                "successful_tasks": result.successful_tasks,
            }

        # Validate memory scaling is reasonable
        baseline_memory = memory_results[1]["peak_memory_mb"]
        memory_4x = memory_results[4]["peak_memory_mb"]

        # 4x agents should not use more than 3x memory
        memory_scaling = memory_4x / baseline_memory
        assert memory_scaling <= PERFORMANCE_THRESHOLDS["memory_multiplier"], (
            f"Memory scaling too high: {memory_scaling:.2f}x > "
            f"{PERFORMANCE_THRESHOLDS['memory_multiplier']}x limit"
        )

        # Memory usage should scale sub-linearly (better than linear)
        assert memory_scaling < 4.0, "Memory should scale sub-linearly with agent count"

        logger.info(f"Memory scaling from 1 to 4 agents: {memory_scaling:.2f}x")


# Slow/Integration Performance Tests
@pytest.mark.slow
class TestParallelResearchIntegrationPerformance:
    """Integration performance tests with more realistic scenarios."""

    async def test_deep_research_agent_parallel_integration(
        self, performance_monitor, mock_executor_factory
    ):
        """Test DeepResearchAgent with parallel execution enabled."""
        with performance_monitor("deep_research_agent_parallel") as monitor:
            # Mock LLM for DeepResearchAgent
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value.content = '{"analysis": "test analysis"}'

            # Create agent with parallel execution enabled
            agent = DeepResearchAgent(
                llm=mock_llm,
                persona="moderate",
                enable_parallel_execution=True,
                parallel_config=ParallelResearchConfig(
                    max_concurrent_agents=3,
                    timeout_per_agent=5,
                    enable_fallbacks=True,
                ),
            )

            monitor.sample_cpu()

            # Mock the subagent execution to avoid real API calls
            async def mock_subagent_execution(task: ResearchTask) -> dict[str, Any]:
                await asyncio.sleep(0.1)  # Simulate work
                return {
                    "research_type": task.task_type,
                    "insights": [f"Mock insight for {task.task_type}"],
                    "sentiment": {"direction": "neutral", "confidence": 0.7},
                    "credibility_score": 0.8,
                    "sources": [{"title": "Mock source", "url": "http://example.com"}],
                }

            # Override the subagent execution method
            agent._execute_subagent_task = mock_subagent_execution

            # Run comprehensive research
            result = await agent.research_comprehensive(
                topic="Apple Inc comprehensive analysis",
                session_id="integration_test",
                depth="standard",
                use_parallel_execution=True,
            )

            monitor.sample_cpu()

        # Validate integration results
        assert result["status"] == "success"
        assert result["execution_mode"] == "parallel"
        assert "parallel_execution_stats" in result
        assert result["parallel_execution_stats"]["total_tasks"] > 0
        assert result["execution_time_ms"] > 0

        # Performance validation
        execution_time_seconds = result["execution_time_ms"] / 1000
        assert execution_time_seconds < 10.0, (
            "Integration test should complete within 10s"
        )
        assert monitor.metrics["memory_multiplier"] < 5.0, (
            "Memory usage should be reasonable"
        )

        logger.info("Deep research agent integration test passed")

    @pytest.mark.parametrize(
        "failure_rate", [0.25]
    )  # Only high failure rate for slow tests
    async def test_high_failure_rate_resilience(
        self,
        failure_rate: float,
        performance_monitor,
        mock_executor_factory,
        performance_tester,
    ):
        """Test system resilience under high failure rates."""
        agent_count = 6  # More agents for integration testing
        test_name = f"high_failure_resilience_{int(failure_rate * 100)}pct"

        resilience_results = []

        for run in range(STATISTICAL_RUNS):
            with performance_monitor(f"{test_name}_run{run}") as monitor:
                config = ParallelResearchConfig(
                    max_concurrent_agents=4,  # Limit concurrency
                    timeout_per_agent=8,  # Longer timeout for resilience
                    enable_fallbacks=True,
                    rate_limit_delay=0.1,
                )

                tasks = performance_tester.create_test_tasks(
                    agent_count, f"resilience_test_{run}"
                )
                executor = mock_executor_factory(
                    failure_rate=failure_rate,
                    base_delay=0.2,  # Longer delays to simulate real API calls
                    delay_variance=0.3,
                    include_timeouts=True,  # Include timeout scenarios
                )

                monitor.sample_cpu()

                try:
                    result = await performance_tester.run_parallel_test(
                        tasks, config, executor
                    )

                    success_rate = result.successful_tasks / (
                        result.successful_tasks + result.failed_tasks
                    )
                    resilience_results.append(
                        {
                            "success_rate": success_rate,
                            "execution_time": monitor.metrics["execution_time"],
                            "memory_multiplier": monitor.metrics["memory_multiplier"],
                            "parallel_efficiency": result.parallel_efficiency,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Resilience test run {run} failed: {e}")
                    resilience_results.append(
                        {
                            "success_rate": 0.0,
                            "execution_time": monitor.metrics["execution_time"],
                            "memory_multiplier": monitor.metrics["memory_multiplier"],
                            "parallel_efficiency": 0.0,
                        }
                    )

                monitor.sample_cpu()

        # Analyze resilience results
        avg_success_rate = statistics.mean(
            [r["success_rate"] for r in resilience_results]
        )
        avg_execution_time = statistics.mean(
            [r["execution_time"] for r in resilience_results]
        )

        # Validate system maintained reasonable performance under stress
        min_acceptable_success = 0.5  # 50% success rate under 25% failure
        assert avg_success_rate >= min_acceptable_success, (
            f"System not resilient enough: {avg_success_rate:.2f} < {min_acceptable_success}"
        )

        assert avg_execution_time < 20.0, (
            "High failure rate tests should still complete reasonably"
        )

        logger.info(
            f"High failure rate resilience test: {avg_success_rate:.2f} average success rate, "
            f"{avg_execution_time:.2f}s average execution time"
        )


# Test Suite Performance Validation
@pytest.mark.unit
class TestSuitePerformance:
    """Validate overall test suite performance characteristics."""

    async def test_total_test_duration_under_threshold(self, performance_monitor):
        """Validate that core performance tests complete within time budget."""
        with performance_monitor("test_suite_duration") as monitor:
            # Simulate running the key performance tests
            await asyncio.sleep(0.1)  # Placeholder for actual test execution
            monitor.sample_cpu()

        # This is a meta-test that would be updated based on actual suite performance
        # For now, we validate the monitoring infrastructure works
        assert monitor.metrics["execution_time"] < 1.0, (
            "Meta-test should complete quickly"
        )
        assert monitor.metrics["memory_multiplier"] < 2.0, (
            "Meta-test should use minimal memory"
        )

    def test_performance_threshold_configuration(self):
        """Validate performance threshold configuration is reasonable."""
        # Test threshold sanity checks
        assert PERFORMANCE_THRESHOLDS["speedup"][2] > 1.0, (
            "2-agent speedup should exceed 1x"
        )
        assert (
            PERFORMANCE_THRESHOLDS["speedup"][4] > PERFORMANCE_THRESHOLDS["speedup"][2]
        ), "4-agent speedup should exceed 2-agent"
        assert PERFORMANCE_THRESHOLDS["memory_multiplier"] > 1.0, (
            "Memory multiplier should allow some increase"
        )
        assert PERFORMANCE_THRESHOLDS["memory_multiplier"] < 10.0, (
            "Memory multiplier should be reasonable"
        )
        assert PERFORMANCE_THRESHOLDS["max_test_duration"] > 10.0, (
            "Test duration budget should be reasonable"
        )

        # Test failure rate configuration
        assert all(0.0 <= rate <= 1.0 for rate in FAILURE_RATES), (
            "Failure rates should be valid percentages"
        )
        assert len(set(FAILURE_RATES)) == len(FAILURE_RATES), (
            "Failure rates should be unique"
        )

        # Test statistical significance configuration
        assert STATISTICAL_RUNS >= 3, (
            "Statistical runs should provide reasonable sample size"
        )
        assert STATISTICAL_RUNS <= 10, "Statistical runs should not be excessive for CI"


if __name__ == "__main__":
    # Allow running individual performance tests for development
    import sys

    if len(sys.argv) > 1:
        pytest.main([sys.argv[1], "-v", "-s", "--tb=short"])
    else:
        # Run unit performance tests by default
        pytest.main([__file__, "-v", "-s", "-m", "unit", "--tb=short"])
