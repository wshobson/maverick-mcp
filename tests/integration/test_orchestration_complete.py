#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for MaverickMCP Orchestration Capabilities

This test suite thoroughly validates all orchestration features including:
- agents_orchestrated_analysis with different personas and routing strategies
- agents_deep_research_financial with various research depths and focus areas
- agents_compare_multi_agent_analysis with different agent combinations

The tests simulate real Claude Desktop usage patterns and validate end-to-end
functionality with comprehensive error handling and performance monitoring.
"""

import asyncio
import json
import logging
import os
import sys
import time
import tracemalloc
from datetime import datetime
from typing import Any

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Test configuration
TEST_CONFIG = {
    "timeout_seconds": 300,  # 5 minutes max per test
    "concurrent_limit": 3,  # Maximum concurrent tests
    "performance_monitoring": True,
    "detailed_validation": True,
    "save_results": True,
}

# Test scenarios for each orchestration tool
ORCHESTRATED_ANALYSIS_SCENARIOS = [
    {
        "name": "Conservative_LLM_Powered_Routing",
        "query": "Analyze AAPL for long-term investment",
        "persona": "conservative",
        "routing_strategy": "llm_powered",
        "max_agents": 2,
        "parallel_execution": False,
    },
    {
        "name": "Aggressive_Rule_Based_Routing",
        "query": "Find high momentum growth stocks in tech",
        "persona": "aggressive",
        "routing_strategy": "rule_based",
        "max_agents": 3,
        "parallel_execution": True,
    },
    {
        "name": "Moderate_Hybrid_Routing",
        "query": "Portfolio analysis for MSFT GOOGL NVDA",
        "persona": "moderate",
        "routing_strategy": "hybrid",
        "max_agents": 2,
        "parallel_execution": True,
    },
    {
        "name": "Day_Trader_Fast_Execution",
        "query": "Quick technical analysis on SPY options",
        "persona": "day_trader",
        "routing_strategy": "rule_based",
        "max_agents": 1,
        "parallel_execution": False,
    },
]

DEEP_RESEARCH_SCENARIOS = [
    {
        "name": "Basic_Company_Research",
        "research_topic": "Tesla Inc stock analysis",
        "persona": "moderate",
        "research_depth": "basic",
        "focus_areas": ["fundamentals"],
        "timeframe": "7d",
    },
    {
        "name": "Standard_Sector_Research",
        "research_topic": "renewable energy sector trends",
        "persona": "conservative",
        "research_depth": "standard",
        "focus_areas": ["market_sentiment", "competitive_landscape"],
        "timeframe": "30d",
    },
    {
        "name": "Comprehensive_Market_Research",
        "research_topic": "AI and machine learning investment opportunities",
        "persona": "aggressive",
        "research_depth": "comprehensive",
        "focus_areas": ["fundamentals", "technicals", "market_sentiment"],
        "timeframe": "90d",
    },
    {
        "name": "Exhaustive_Crypto_Research",
        "research_topic": "Bitcoin and cryptocurrency market analysis",
        "persona": "day_trader",
        "research_depth": "exhaustive",
        "focus_areas": ["technicals", "market_sentiment", "competitive_landscape"],
        "timeframe": "1y",
    },
]

MULTI_AGENT_COMPARISON_SCENARIOS = [
    {
        "name": "Market_vs_Supervisor_Stock_Analysis",
        "query": "Should I invest in Apple stock now?",
        "agent_types": ["market", "supervisor"],
        "persona": "moderate",
    },
    {
        "name": "Conservative_Multi_Agent_Portfolio",
        "query": "Build a balanced portfolio for retirement",
        "agent_types": ["market", "supervisor"],
        "persona": "conservative",
    },
    {
        "name": "Aggressive_Growth_Strategy",
        "query": "Find the best growth stocks for 2025",
        "agent_types": ["market", "supervisor"],
        "persona": "aggressive",
    },
]

ERROR_HANDLING_SCENARIOS = [
    {
        "tool": "orchestrated_analysis",
        "params": {
            "query": "",  # Empty query
            "persona": "invalid_persona",
            "routing_strategy": "unknown_strategy",
        },
    },
    {
        "tool": "deep_research_financial",
        "params": {
            "research_topic": "XYZ",
            "research_depth": "invalid_depth",
            "focus_areas": ["invalid_area"],
        },
    },
    {
        "tool": "compare_multi_agent_analysis",
        "params": {
            "query": "test",
            "agent_types": ["nonexistent_agent"],
            "persona": "unknown",
        },
    },
]


class TestResult:
    """Container for individual test results."""

    def __init__(self, test_name: str, tool_name: str):
        self.test_name = test_name
        self.tool_name = tool_name
        self.start_time = time.time()
        self.end_time: float | None = None
        self.success = False
        self.error: str | None = None
        self.response: dict[str, Any] | None = None
        self.execution_time_ms: float | None = None
        self.memory_usage_mb: float | None = None
        self.validation_results: dict[str, bool] = {}

    def mark_completed(
        self, success: bool, response: dict | None = None, error: str | None = None
    ):
        """Mark test as completed with results."""
        self.end_time = time.time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.response = response
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "tool_name": self.tool_name,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "error": self.error,
            "validation_results": self.validation_results,
            "response_summary": self._summarize_response() if self.response else None,
        }

    def _summarize_response(self) -> dict[str, Any]:
        """Create summary of response for reporting."""
        if not self.response:
            return {}

        summary = {
            "status": self.response.get("status"),
            "agent_type": self.response.get("agent_type"),
            "persona": self.response.get("persona"),
        }

        # Add tool-specific summary fields
        if "agents_used" in self.response:
            summary["agents_used"] = self.response["agents_used"]
        if "sources_analyzed" in self.response:
            summary["sources_analyzed"] = self.response["sources_analyzed"]
        if "agents_compared" in self.response:
            summary["agents_compared"] = self.response["agents_compared"]

        return summary


class IntegrationTestSuite:
    """Comprehensive integration test suite for MCP orchestration tools."""

    def __init__(self):
        self.setup_logging()
        self.results: list[TestResult] = []
        self.start_time = time.time()
        self.session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Import MCP tools
        self._import_tools()

    def setup_logging(self):
        """Configure logging for test execution."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _import_tools(self):
        """Import and initialize MCP tools."""
        try:
            from maverick_mcp.api.routers.agents import (
                compare_multi_agent_analysis,
                deep_research_financial,
                orchestrated_analysis,
            )

            self.orchestrated_analysis = orchestrated_analysis
            self.deep_research_financial = deep_research_financial
            self.compare_multi_agent_analysis = compare_multi_agent_analysis

            self.logger.info("Successfully imported MCP orchestration tools")

        except ImportError as e:
            self.logger.error(f"Failed to import MCP tools: {e}")
            raise RuntimeError(f"Cannot run tests without MCP tools: {e}")

    def print_header(self, title: str):
        """Print formatted test section header."""
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")

    def print_progress(self, current: int, total: int, test_name: str):
        """Print progress indicator."""
        percentage = (current / total) * 100
        print(f"[{current:2d}/{total}] ({percentage:5.1f}%) {test_name}")

    async def test_orchestrated_analysis(self) -> list[TestResult]:
        """Test agents_orchestrated_analysis with various scenarios."""
        self.print_header("Testing Orchestrated Analysis Tool")
        results = []

        for i, scenario in enumerate(ORCHESTRATED_ANALYSIS_SCENARIOS):
            test_name = f"orchestrated_analysis_{scenario['name']}"
            self.print_progress(i + 1, len(ORCHESTRATED_ANALYSIS_SCENARIOS), test_name)

            result = TestResult(test_name, "agents_orchestrated_analysis")

            try:
                if TEST_CONFIG["performance_monitoring"]:
                    tracemalloc.start()

                # Add unique session ID for each test
                scenario_params = scenario.copy()
                scenario_params["session_id"] = f"{self.session_id}_{test_name}"

                # Execute with timeout
                response = await asyncio.wait_for(
                    self.orchestrated_analysis(**scenario_params),
                    timeout=TEST_CONFIG["timeout_seconds"],
                )

                # Validate response
                validation_results = self._validate_orchestrated_response(response)
                result.validation_results = validation_results

                success = (
                    all(validation_results.values())
                    and response.get("status") == "success"
                )
                result.mark_completed(success, response)

                if TEST_CONFIG["performance_monitoring"]:
                    current, peak = tracemalloc.get_traced_memory()
                    result.memory_usage_mb = peak / 1024 / 1024
                    tracemalloc.stop()

                self.logger.info(
                    f"✓ {test_name}: {'PASS' if success else 'FAIL'} "
                    f"({result.execution_time_ms:.0f}ms)"
                )

            except TimeoutError:
                result.mark_completed(False, error="Test timeout")
                self.logger.warning(f"✗ {test_name}: TIMEOUT")

            except Exception as e:
                result.mark_completed(False, error=str(e))
                self.logger.error(f"✗ {test_name}: ERROR - {e}")

            results.append(result)

        return results

    async def test_deep_research_financial(self) -> list[TestResult]:
        """Test agents_deep_research_financial with various scenarios."""
        self.print_header("Testing Deep Research Financial Tool")
        results = []

        for i, scenario in enumerate(DEEP_RESEARCH_SCENARIOS):
            test_name = f"deep_research_{scenario['name']}"
            self.print_progress(i + 1, len(DEEP_RESEARCH_SCENARIOS), test_name)

            result = TestResult(test_name, "agents_deep_research_financial")

            try:
                if TEST_CONFIG["performance_monitoring"]:
                    tracemalloc.start()

                # Add unique session ID
                scenario_params = scenario.copy()
                scenario_params["session_id"] = f"{self.session_id}_{test_name}"

                response = await asyncio.wait_for(
                    self.deep_research_financial(**scenario_params),
                    timeout=TEST_CONFIG["timeout_seconds"],
                )

                validation_results = self._validate_research_response(response)
                result.validation_results = validation_results

                success = (
                    all(validation_results.values())
                    and response.get("status") == "success"
                )
                result.mark_completed(success, response)

                if TEST_CONFIG["performance_monitoring"]:
                    current, peak = tracemalloc.get_traced_memory()
                    result.memory_usage_mb = peak / 1024 / 1024
                    tracemalloc.stop()

                self.logger.info(
                    f"✓ {test_name}: {'PASS' if success else 'FAIL'} "
                    f"({result.execution_time_ms:.0f}ms)"
                )

            except TimeoutError:
                result.mark_completed(False, error="Test timeout")
                self.logger.warning(f"✗ {test_name}: TIMEOUT")

            except Exception as e:
                result.mark_completed(False, error=str(e))
                self.logger.error(f"✗ {test_name}: ERROR - {e}")

            results.append(result)

        return results

    async def test_compare_multi_agent_analysis(self) -> list[TestResult]:
        """Test agents_compare_multi_agent_analysis with various scenarios."""
        self.print_header("Testing Multi-Agent Comparison Tool")
        results = []

        for i, scenario in enumerate(MULTI_AGENT_COMPARISON_SCENARIOS):
            test_name = f"multi_agent_{scenario['name']}"
            self.print_progress(i + 1, len(MULTI_AGENT_COMPARISON_SCENARIOS), test_name)

            result = TestResult(test_name, "agents_compare_multi_agent_analysis")

            try:
                if TEST_CONFIG["performance_monitoring"]:
                    tracemalloc.start()

                scenario_params = scenario.copy()
                scenario_params["session_id"] = f"{self.session_id}_{test_name}"

                response = await asyncio.wait_for(
                    self.compare_multi_agent_analysis(**scenario_params),
                    timeout=TEST_CONFIG["timeout_seconds"],
                )

                validation_results = self._validate_comparison_response(response)
                result.validation_results = validation_results

                success = (
                    all(validation_results.values())
                    and response.get("status") == "success"
                )
                result.mark_completed(success, response)

                if TEST_CONFIG["performance_monitoring"]:
                    current, peak = tracemalloc.get_traced_memory()
                    result.memory_usage_mb = peak / 1024 / 1024
                    tracemalloc.stop()

                self.logger.info(
                    f"✓ {test_name}: {'PASS' if success else 'FAIL'} "
                    f"({result.execution_time_ms:.0f}ms)"
                )

            except TimeoutError:
                result.mark_completed(False, error="Test timeout")
                self.logger.warning(f"✗ {test_name}: TIMEOUT")

            except Exception as e:
                result.mark_completed(False, error=str(e))
                self.logger.error(f"✗ {test_name}: ERROR - {e}")

            results.append(result)

        return results

    async def test_error_handling(self) -> list[TestResult]:
        """Test error handling with invalid inputs."""
        self.print_header("Testing Error Handling")
        results = []

        for i, scenario in enumerate(ERROR_HANDLING_SCENARIOS):
            test_name = f"error_handling_{scenario['tool']}"
            self.print_progress(i + 1, len(ERROR_HANDLING_SCENARIOS), test_name)

            result = TestResult(test_name, scenario["tool"])

            try:
                # Get the tool function
                tool_func = getattr(self, scenario["tool"])

                # Add session ID
                params = scenario["params"].copy()
                params["session_id"] = f"{self.session_id}_{test_name}"

                response = await asyncio.wait_for(
                    tool_func(**params),
                    timeout=60,  # Shorter timeout for error cases
                )

                # For error handling tests, we expect graceful error handling
                # Success means the tool returned an error response without crashing
                has_error_field = (
                    "error" in response or response.get("status") == "error"
                )
                success = has_error_field and isinstance(response, dict)

                result.validation_results = {"graceful_error_handling": success}
                result.mark_completed(success, response)

                self.logger.info(
                    f"✓ {test_name}: {'PASS' if success else 'FAIL'} - "
                    f"Graceful error handling: {has_error_field}"
                )

            except TimeoutError:
                result.mark_completed(False, error="Test timeout")
                self.logger.warning(f"✗ {test_name}: TIMEOUT")

            except Exception as e:
                # For error handling tests, exceptions are actually failures
                result.mark_completed(False, error=f"Unhandled exception: {str(e)}")
                self.logger.error(f"✗ {test_name}: UNHANDLED EXCEPTION - {e}")

            results.append(result)

        return results

    async def test_concurrent_execution(self) -> list[TestResult]:
        """Test concurrent execution of multiple tools."""
        self.print_header("Testing Concurrent Execution")
        results = []

        # Create concurrent test scenarios
        concurrent_tasks = [
            (
                "concurrent_orchestrated",
                self.orchestrated_analysis,
                {
                    "query": "Analyze MSFT for investment",
                    "persona": "moderate",
                    "routing_strategy": "llm_powered",
                    "session_id": f"{self.session_id}_concurrent_1",
                },
            ),
            (
                "concurrent_research",
                self.deep_research_financial,
                {
                    "research_topic": "Amazon business model",
                    "persona": "conservative",
                    "research_depth": "standard",
                    "session_id": f"{self.session_id}_concurrent_2",
                },
            ),
            (
                "concurrent_comparison",
                self.compare_multi_agent_analysis,
                {
                    "query": "Best tech stocks for portfolio",
                    "persona": "aggressive",
                    "session_id": f"{self.session_id}_concurrent_3",
                },
            ),
        ]

        start_time = time.time()

        try:
            # Execute all tasks concurrently
            concurrent_results = await asyncio.gather(
                *[
                    task_func(**task_params)
                    for _, task_func, task_params in concurrent_tasks
                ],
                return_exceptions=True,
            )

            execution_time = (time.time() - start_time) * 1000

            # Process results
            for i, (task_name, _, _) in enumerate(concurrent_tasks):
                result = TestResult(task_name, "concurrent_execution")

                if i < len(concurrent_results):
                    response = concurrent_results[i]

                    if isinstance(response, Exception):
                        result.mark_completed(False, error=str(response))
                        success = False
                    else:
                        success = (
                            isinstance(response, dict)
                            and response.get("status") != "error"
                        )
                        result.mark_completed(success, response)

                    result.validation_results = {"concurrent_execution": success}
                    self.logger.info(f"✓ {task_name}: {'PASS' if success else 'FAIL'}")
                else:
                    result.mark_completed(False, error="No result returned")
                    self.logger.error(f"✗ {task_name}: No result returned")

                results.append(result)

            # Add overall concurrent test result
            concurrent_summary = TestResult(
                "concurrent_execution_summary", "performance"
            )
            concurrent_summary.execution_time_ms = execution_time
            concurrent_summary.validation_results = {
                "all_completed": len(concurrent_results) == len(concurrent_tasks),
                "no_crashes": all(
                    not isinstance(r, Exception) for r in concurrent_results
                ),
                "reasonable_time": execution_time < 180000,  # 3 minutes
            }
            concurrent_summary.mark_completed(
                all(concurrent_summary.validation_results.values()),
                {
                    "concurrent_tasks": len(concurrent_tasks),
                    "total_time_ms": execution_time,
                },
            )
            results.append(concurrent_summary)

            self.logger.info(
                f"Concurrent execution completed in {execution_time:.0f}ms"
            )

        except Exception as e:
            error_result = TestResult("concurrent_execution_error", "performance")
            error_result.mark_completed(False, error=str(e))
            results.append(error_result)
            self.logger.error(f"✗ Concurrent execution failed: {e}")

        return results

    def _validate_orchestrated_response(
        self, response: dict[str, Any]
    ) -> dict[str, bool]:
        """Validate orchestrated analysis response format."""
        validations = {
            "has_status": "status" in response,
            "has_agent_type": "agent_type" in response,
            "has_persona": "persona" in response,
            "has_session_id": "session_id" in response,
            "status_is_success": response.get("status") == "success",
            "has_routing_strategy": "routing_strategy" in response,
            "has_execution_time": "execution_time_ms" in response
            and isinstance(response.get("execution_time_ms"), int | float),
        }

        # Additional validations for successful responses
        if response.get("status") == "success":
            validations.update(
                {
                    "has_agents_used": "agents_used" in response,
                    "agents_used_is_list": isinstance(
                        response.get("agents_used"), list
                    ),
                    "has_synthesis_confidence": "synthesis_confidence" in response,
                }
            )

        return validations

    def _validate_research_response(self, response: dict[str, Any]) -> dict[str, bool]:
        """Validate deep research response format."""
        validations = {
            "has_status": "status" in response,
            "has_agent_type": "agent_type" in response,
            "has_persona": "persona" in response,
            "has_research_topic": "research_topic" in response,
            "status_is_success": response.get("status") == "success",
            "has_research_depth": "research_depth" in response,
            "has_focus_areas": "focus_areas" in response,
        }

        if response.get("status") == "success":
            validations.update(
                {
                    "has_sources_analyzed": "sources_analyzed" in response,
                    "sources_analyzed_is_numeric": isinstance(
                        response.get("sources_analyzed"), int | float
                    ),
                    "has_research_confidence": "research_confidence" in response,
                    "has_validation_checks": "validation_checks_passed" in response,
                }
            )

        return validations

    def _validate_comparison_response(
        self, response: dict[str, Any]
    ) -> dict[str, bool]:
        """Validate multi-agent comparison response format."""
        validations = {
            "has_status": "status" in response,
            "has_query": "query" in response,
            "has_persona": "persona" in response,
            "status_is_success": response.get("status") == "success",
            "has_agents_compared": "agents_compared" in response,
        }

        if response.get("status") == "success":
            validations.update(
                {
                    "agents_compared_is_list": isinstance(
                        response.get("agents_compared"), list
                    ),
                    "has_comparison": "comparison" in response,
                    "comparison_is_dict": isinstance(response.get("comparison"), dict),
                    "has_execution_times": "execution_times_ms" in response,
                    "has_insights": "insights" in response,
                }
            )

        return validations

    async def run_performance_benchmark(self):
        """Run performance benchmarks for all tools."""
        self.print_header("Performance Benchmarking")

        benchmark_scenarios = [
            (
                "orchestrated_fast",
                self.orchestrated_analysis,
                {
                    "query": "Quick AAPL analysis",
                    "persona": "moderate",
                    "routing_strategy": "rule_based",
                    "max_agents": 1,
                    "parallel_execution": False,
                },
            ),
            (
                "research_basic",
                self.deep_research_financial,
                {
                    "research_topic": "Microsoft",
                    "research_depth": "basic",
                    "persona": "moderate",
                },
            ),
            (
                "comparison_minimal",
                self.compare_multi_agent_analysis,
                {
                    "query": "Compare AAPL vs MSFT",
                    "agent_types": ["market"],
                    "persona": "moderate",
                },
            ),
        ]

        performance_results = []

        for test_name, tool_func, params in benchmark_scenarios:
            print(f"Benchmarking {test_name}...")

            # Add session ID
            params["session_id"] = f"{self.session_id}_benchmark_{test_name}"

            # Run multiple iterations for average performance
            times = []

            for i in range(3):  # 3 iterations for average
                start_time = time.time()

                try:
                    await tool_func(**params)
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000
                    times.append(execution_time)

                except Exception as e:
                    self.logger.error(
                        f"Benchmark {test_name} iteration {i + 1} failed: {e}"
                    )
                    times.append(float("inf"))

            # Calculate performance metrics
            valid_times = [t for t in times if t != float("inf")]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)

                performance_results.append(
                    {
                        "test": test_name,
                        "avg_time_ms": avg_time,
                        "min_time_ms": min_time,
                        "max_time_ms": max_time,
                        "successful_runs": len(valid_times),
                    }
                )

                print(
                    f"  {test_name}: Avg={avg_time:.0f}ms, Min={min_time:.0f}ms, Max={max_time:.0f}ms"
                )
            else:
                print(f"  {test_name}: All iterations failed")

        return performance_results

    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.print_header("Test Results Summary")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        total_time = time.time() - self.start_time

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests / total_tests * 100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests / total_tests * 100:.1f}%)")
        print(f"Total Execution Time: {total_time:.2f}s")

        # Group results by tool
        by_tool = {}
        for result in self.results:
            if result.tool_name not in by_tool:
                by_tool[result.tool_name] = []
            by_tool[result.tool_name].append(result)

        print("\nResults by Tool:")
        for tool_name, tool_results in by_tool.items():
            tool_passed = sum(1 for r in tool_results if r.success)
            tool_total = len(tool_results)
            print(
                f"  {tool_name}: {tool_passed}/{tool_total} passed "
                f"({tool_passed / tool_total * 100:.1f}%)"
            )

        # Performance summary
        execution_times = [
            r.execution_time_ms for r in self.results if r.execution_time_ms
        ]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            print("\nPerformance Summary:")
            print(f"  Average execution time: {avg_time:.0f}ms")
            print(f"  Fastest test: {min(execution_times):.0f}ms")
            print(f"  Slowest test: {max(execution_times):.0f}ms")

        # Failed tests details
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print("\nFailed Tests:")
            for result in failed_results:
                print(f"  ✗ {result.test_name}: {result.error}")

        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time_s": total_time,
            },
            "by_tool": {
                tool: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.success),
                    "pass_rate": sum(1 for r in results if r.success) / len(results),
                }
                for tool, results in by_tool.items()
            },
            "performance": {
                "avg_execution_time_ms": avg_time if execution_times else None,
                "min_execution_time_ms": min(execution_times)
                if execution_times
                else None,
                "max_execution_time_ms": max(execution_times)
                if execution_times
                else None,
            },
            "failed_tests": [r.to_dict() for r in failed_results],
        }

    def save_results(self, report: dict[str, Any], performance_data: list[dict]):
        """Save detailed results to files."""
        if not TEST_CONFIG["save_results"]:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_results = {
            "test_session": self.session_id,
            "timestamp": timestamp,
            "config": TEST_CONFIG,
            "summary": report,
            "detailed_results": [r.to_dict() for r in self.results],
            "performance_benchmarks": performance_data,
        }

        with open(f"integration_test_results_{timestamp}.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: integration_test_results_{timestamp}.json")

    async def run_all_tests(self):
        """Run the complete test suite."""
        self.print_header(
            f"MaverickMCP Orchestration Integration Test Suite - {self.session_id}"
        )

        print("Test Configuration:")
        for key, value in TEST_CONFIG.items():
            print(f"  {key}: {value}")

        try:
            # Run all test categories
            orchestrated_results = await self.test_orchestrated_analysis()
            self.results.extend(orchestrated_results)

            research_results = await self.test_deep_research_financial()
            self.results.extend(research_results)

            comparison_results = await self.test_compare_multi_agent_analysis()
            self.results.extend(comparison_results)

            error_handling_results = await self.test_error_handling()
            self.results.extend(error_handling_results)

            concurrent_results = await self.test_concurrent_execution()
            self.results.extend(concurrent_results)

            # Performance benchmarks
            performance_data = await self.run_performance_benchmark()

            # Generate and save report
            report = self.generate_test_report()
            self.save_results(report, performance_data)

            # Final status
            total_passed = sum(1 for r in self.results if r.success)
            total_tests = len(self.results)

            if total_passed == total_tests:
                print(f"\n🎉 ALL TESTS PASSED! ({total_passed}/{total_tests})")
                return 0
            else:
                print(f"\n⚠️  SOME TESTS FAILED ({total_passed}/{total_tests} passed)")
                return 1

        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            print(f"\n💥 TEST SUITE EXECUTION FAILED: {e}")
            return 2


async def main():
    """Main test execution function."""
    # Set environment variables for testing if needed
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set - tests will use mock responses")

    test_suite = IntegrationTestSuite()
    exit_code = await test_suite.run_all_tests()

    return exit_code


if __name__ == "__main__":
    # Make the script executable
    import sys

    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)  # SIGINT exit code
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
