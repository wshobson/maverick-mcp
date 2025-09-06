#!/usr/bin/env python3
"""
Performance benchmarking script for ExaSearch integration.

This script runs performance tests across different research depths, focus areas,
and configurations to validate the ExaSearch integration performance and reliability.

Usage:
    python scripts/benchmark_exa_research.py --depth all --focus all --parallel
    python scripts/benchmark_exa_research.py --depth basic --focus fundamentals
    python scripts/benchmark_exa_research.py --quick --no-parallel
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.config.settings import get_settings
from maverick_mcp.providers.llm_factory import get_llm


class ExaResearchBenchmark:
    """Comprehensive benchmarking suite for ExaSearch research integration."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.settings = get_settings()
        self.llm = get_llm()

        # Test configurations
        self.test_queries = [
            "AAPL stock financial analysis and investment outlook",
            "Tesla market sentiment and competitive position",
            "Microsoft earnings performance and growth prospects",
            "tech sector analysis and market trends",
            "artificial intelligence investment opportunities",
        ]

        self.research_depths = ["basic", "standard", "comprehensive", "exhaustive"]

        self.focus_areas = {
            "fundamentals": ["earnings", "valuation", "financial_health"],
            "technicals": ["chart_patterns", "technical_indicators", "price_action"],
            "sentiment": ["market_sentiment", "analyst_ratings", "news_sentiment"],
            "competitive": ["competitive_position", "market_share", "industry_trends"],
        }

        self.benchmark_results = []

    async def run_single_benchmark(
        self,
        query: str,
        depth: str,
        focus_area: str | None = None,
        parallel: bool = True,
        timeout_budget: float = 60.0,
    ) -> dict[str, Any]:
        """Run a single benchmark test."""
        print(
            f"🔍 Testing: {query[:30]}... | Depth: {depth} | Focus: {focus_area} | Parallel: {parallel}"
        )

        start_time = time.time()
        session_id = f"benchmark_{int(start_time)}"

        try:
            # Create research agent
            agent = DeepResearchAgent(
                llm=self.llm,
                persona="moderate",
                exa_api_key=self.settings.research.exa_api_key,
                research_depth=depth,
                enable_parallel_execution=parallel,
            )

            # Initialize agent
            await agent.initialize()

            # Check if search providers are available
            if not agent.search_providers:
                return {
                    "status": "skipped",
                    "reason": "No search providers available (check EXA_API_KEY)",
                    "query": query,
                    "depth": depth,
                    "focus_area": focus_area,
                    "parallel": parallel,
                }

            # Setup focus areas if specified
            focus_areas_list = None
            if focus_area and focus_area in self.focus_areas:
                focus_areas_list = self.focus_areas[focus_area]

            # Execute research
            result = await agent.research_comprehensive(
                topic=query,
                session_id=session_id,
                depth=depth,
                focus_areas=focus_areas_list,
                timeout_budget=timeout_budget,
                use_parallel_execution=parallel,
            )

            execution_time = time.time() - start_time

            # Analyze result
            success = result.get("status") == "success"
            sources_analyzed = result.get("sources_analyzed", 0)
            confidence_score = result.get("confidence_score", 0.0)

            # Extract parallel execution stats if available
            parallel_stats = result.get("parallel_execution_stats", {})

            benchmark_result = {
                "timestamp": datetime.now().isoformat(),
                "status": "success" if success else "failed",
                "query": query,
                "depth": depth,
                "focus_area": focus_area,
                "parallel": parallel,
                "execution_time": execution_time,
                "timeout_budget": timeout_budget,
                "sources_analyzed": sources_analyzed,
                "confidence_score": confidence_score,
                "parallel_stats": parallel_stats,
                "error": result.get("error") if not success else None,
            }

            # Additional metrics for successful runs
            if success:
                findings = result.get("findings", {})
                benchmark_result.update(
                    {
                        "synthesis_available": bool(findings.get("synthesis")),
                        "key_insights_count": len(findings.get("key_insights", [])),
                        "citations_count": len(result.get("citations", [])),
                    }
                )

            return benchmark_result

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "query": query,
                "depth": depth,
                "focus_area": focus_area,
                "parallel": parallel,
                "execution_time": execution_time,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def benchmark_depth_comparison(self, query: str) -> list[dict[str, Any]]:
        """Benchmark performance across all research depths."""
        print(f"\n📊 Depth Comparison Benchmark: {query[:40]}...")

        depth_results = []

        for depth in self.research_depths:
            # Adjust timeout based on depth complexity
            timeout_budget = {
                "basic": 15.0,
                "standard": 30.0,
                "comprehensive": 45.0,
                "exhaustive": 60.0,
            }.get(depth, 30.0)

            result = await self.run_single_benchmark(
                query=query,
                depth=depth,
                timeout_budget=timeout_budget,
            )

            depth_results.append(result)

            # Brief pause between tests
            await asyncio.sleep(2)

        return depth_results

    async def benchmark_parallel_vs_sequential(
        self, query: str, depth: str = "standard"
    ) -> dict[str, Any]:
        """Compare parallel vs sequential execution performance."""
        print(f"\n⚡ Parallel vs Sequential Benchmark: {query[:40]}...")

        # Sequential execution
        sequential_result = await self.run_single_benchmark(
            query=query,
            depth=depth,
            parallel=False,
            timeout_budget=45.0,
        )

        await asyncio.sleep(2)

        # Parallel execution
        parallel_result = await self.run_single_benchmark(
            query=query,
            depth=depth,
            parallel=True,
            timeout_budget=45.0,
        )

        # Calculate performance improvement
        speedup = 1.0
        if (
            sequential_result.get("status") == "success"
            and parallel_result.get("status") == "success"
        ):
            seq_time = sequential_result["execution_time"]
            par_time = parallel_result["execution_time"]
            speedup = seq_time / par_time if par_time > 0 else 1.0

        return {
            "query": query,
            "depth": depth,
            "sequential": sequential_result,
            "parallel": parallel_result,
            "speedup": speedup,
            "parallel_efficiency": parallel_result.get("parallel_stats", {}).get(
                "parallel_efficiency", 1.0
            ),
        }

    async def benchmark_focus_areas(
        self, query: str, depth: str = "standard"
    ) -> list[dict[str, Any]]:
        """Benchmark performance across different focus areas."""
        print(f"\n🎯 Focus Areas Benchmark: {query[:40]}...")

        focus_results = []

        for focus_name in self.focus_areas.keys():
            result = await self.run_single_benchmark(
                query=query,
                depth=depth,
                focus_area=focus_name,
                timeout_budget=30.0,
            )

            focus_results.append(result)
            await asyncio.sleep(1)

        return focus_results

    async def benchmark_timeout_resilience(self) -> list[dict[str, Any]]:
        """Test timeout resilience with various timeout budgets."""
        print("\n⏱️ Timeout Resilience Benchmark...")

        query = "comprehensive market analysis with multiple sectors"
        timeout_budgets = [10.0, 15.0, 20.0, 30.0, 45.0]

        timeout_results = []

        for timeout in timeout_budgets:
            result = await self.run_single_benchmark(
                query=query,
                depth="comprehensive",
                timeout_budget=timeout,
            )

            timeout_results.append(
                {
                    **result,
                    "expected_timeout": timeout,
                    "timed_out": result.get("status") in ["timeout", "error"]
                    and "timeout" in str(result.get("error", "")),
                }
            )

            await asyncio.sleep(1)

        return timeout_results

    async def run_comprehensive_benchmark(
        self,
        depths: list[str] | None = None,
        focus_areas: list[str] | None = None,
        include_parallel: bool = True,
        include_timeout: bool = True,
        quick_mode: bool = False,
    ):
        """Run comprehensive benchmark suite."""
        print("🚀 Starting Comprehensive ExaSearch Integration Benchmark")
        print(
            f"⚙️  Configuration: Depths={depths or 'all'}, Focus={focus_areas or 'all'}, Parallel={include_parallel}"
        )

        # Check prerequisites
        if not self.settings.research.exa_api_key:
            print("❌ EXA_API_KEY not configured. Please set the environment variable.")
            return

        benchmark_start = time.time()

        # Select test queries based on mode
        test_queries = self.test_queries[:2] if quick_mode else self.test_queries
        selected_depths = depths or (
            ["basic", "standard"] if quick_mode else self.research_depths
        )

        # 1. Depth Comparison Benchmarks
        print("\n" + "=" * 60)
        print("📈 DEPTH COMPARISON BENCHMARKS")
        print("=" * 60)

        for query in test_queries[:2]:  # Use first 2 queries for depth comparison
            depth_results = await self.benchmark_depth_comparison(query)
            self.benchmark_results.extend(depth_results)

        # 2. Parallel vs Sequential Benchmarks
        if include_parallel:
            print("\n" + "=" * 60)
            print("⚡ PARALLEL vs SEQUENTIAL BENCHMARKS")
            print("=" * 60)

            for query in test_queries[:2]:
                for depth in selected_depths[:2]:  # Test first 2 depths
                    comparison = await self.benchmark_parallel_vs_sequential(
                        query, depth
                    )
                    self.benchmark_results.append(
                        {
                            "benchmark_type": "parallel_comparison",
                            **comparison,
                        }
                    )

        # 3. Focus Area Benchmarks
        if focus_areas:
            print("\n" + "=" * 60)
            print("🎯 FOCUS AREA BENCHMARKS")
            print("=" * 60)

            query = test_queries[0]  # Use first query for focus testing
            focus_results = await self.benchmark_focus_areas(query)
            self.benchmark_results.extend(focus_results)

        # 4. Timeout Resilience Tests
        if include_timeout and not quick_mode:
            print("\n" + "=" * 60)
            print("⏱️ TIMEOUT RESILIENCE BENCHMARKS")
            print("=" * 60)

            timeout_results = await self.benchmark_timeout_resilience()
            self.benchmark_results.extend(timeout_results)

        total_time = time.time() - benchmark_start

        # Generate summary report
        self.generate_summary_report(total_time)
        self.save_detailed_results()

    def generate_summary_report(self, total_time: float):
        """Generate and print summary report."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 60)

        # Filter successful results
        successful_results = [
            r for r in self.benchmark_results if r.get("status") == "success"
        ]
        failed_results = [
            r for r in self.benchmark_results if r.get("status") in ["failed", "error"]
        ]

        print(f"📋 Total Tests: {len(self.benchmark_results)}")
        print(f"✅ Successful: {len(successful_results)}")
        print(f"❌ Failed: {len(failed_results)}")
        print(f"⏱️ Total Time: {total_time:.1f}s")

        if successful_results:
            execution_times = [r["execution_time"] for r in successful_results]
            confidence_scores = [
                r.get("confidence_score", 0) for r in successful_results
            ]
            sources_counts = [r.get("sources_analyzed", 0) for r in successful_results]

            print("\n📈 Performance Metrics:")
            print(f"   Avg Execution Time: {statistics.mean(execution_times):.2f}s")
            print(
                f"   Min/Max Time: {min(execution_times):.2f}s / {max(execution_times):.2f}s"
            )
            print(f"   Avg Confidence Score: {statistics.mean(confidence_scores):.2f}")
            print(f"   Avg Sources Analyzed: {statistics.mean(sources_counts):.1f}")

        # Depth performance analysis
        depth_results = {}
        for result in successful_results:
            depth = result.get("depth")
            if depth:
                if depth not in depth_results:
                    depth_results[depth] = []
                depth_results[depth].append(result["execution_time"])

        if depth_results:
            print("\n📊 Performance by Depth:")
            for depth, times in sorted(depth_results.items()):
                avg_time = statistics.mean(times)
                print(f"   {depth:12}: {avg_time:6.2f}s avg ({len(times)} tests)")

        # Parallel efficiency analysis
        parallel_results = [
            r
            for r in self.benchmark_results
            if r.get("benchmark_type") == "parallel_comparison"
        ]
        if parallel_results:
            speedups = [r["speedup"] for r in parallel_results if r["speedup"] > 0]
            if speedups:
                print("\n⚡ Parallel Performance:")
                print(f"   Avg Speedup: {statistics.mean(speedups):.2f}x")
                print(f"   Max Speedup: {max(speedups):.2f}x")

        # Error analysis
        if failed_results:
            print("\n❌ Error Analysis:")
            error_types = {}
            for result in failed_results:
                error_type = result.get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            for error_type, count in error_types.items():
                print(f"   {error_type}: {count} occurrences")

    def save_detailed_results(self):
        """Save detailed benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"exa_benchmark_results_{timestamp}.json"

        # Prepare summary data
        summary_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.benchmark_results),
                "successful_tests": len(
                    [r for r in self.benchmark_results if r.get("status") == "success"]
                ),
                "failed_tests": len(
                    [
                        r
                        for r in self.benchmark_results
                        if r.get("status") in ["failed", "error"]
                    ]
                ),
            },
            "configuration": {
                "exa_api_available": bool(self.settings.research.exa_api_key),
                "test_queries": self.test_queries,
                "research_depths": self.research_depths,
                "focus_areas": list(self.focus_areas.keys()),
            },
            "results": self.benchmark_results,
        }

        with open(results_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {results_file}")

        # Also save a CSV for easy analysis
        csv_file = self.output_dir / f"exa_benchmark_summary_{timestamp}.csv"
        self.save_csv_summary(csv_file)
        print(f"📊 CSV summary saved to: {csv_file}")

    def save_csv_summary(self, csv_file: Path):
        """Save a CSV summary for easy analysis."""
        import csv

        # Filter for successful results with key metrics
        successful_results = [
            r for r in self.benchmark_results if r.get("status") == "success"
        ]

        if not successful_results:
            return

        fieldnames = [
            "timestamp",
            "query",
            "depth",
            "focus_area",
            "parallel",
            "execution_time",
            "timeout_budget",
            "sources_analyzed",
            "confidence_score",
            "key_insights_count",
            "citations_count",
        ]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in successful_results:
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark ExaSearch research integration"
    )

    parser.add_argument(
        "--depth",
        choices=["basic", "standard", "comprehensive", "exhaustive", "all"],
        default="all",
        help="Research depths to test",
    )

    parser.add_argument(
        "--focus",
        choices=["fundamentals", "technicals", "sentiment", "competitive", "all"],
        default="all",
        help="Focus areas to test",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Include parallel execution tests",
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Skip parallel execution tests",
    )

    parser.add_argument(
        "--timeout",
        action="store_true",
        default=True,
        help="Include timeout resilience tests",
    )

    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Skip timeout resilience tests",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer tests, faster execution)",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Parse arguments
    depths = None if args.depth == "all" else [args.depth]
    focus_areas = None if args.focus == "all" else [args.focus]
    include_parallel = args.parallel and not args.no_parallel
    include_timeout = args.timeout and not args.no_timeout

    # Run benchmark
    benchmark = ExaResearchBenchmark(output_dir=args.output_dir)

    await benchmark.run_comprehensive_benchmark(
        depths=depths,
        focus_areas=focus_areas,
        include_parallel=include_parallel,
        include_timeout=include_timeout,
        quick_mode=args.quick,
    )


if __name__ == "__main__":
    asyncio.run(main())
