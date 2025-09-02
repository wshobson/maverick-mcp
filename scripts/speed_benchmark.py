#!/usr/bin/env python3
"""
Speed Benchmarking Script for MaverickMCP Research Agents

This script provides comprehensive speed benchmarking capabilities for:
1. Validating speed optimization claims (2-3x improvement)
2. Testing emergency mode performance (<30s completion)
3. Verifying model selection optimization
4. Resolving timeout issues
5. Continuous integration speed validation

Usage:
    python scripts/speed_benchmark.py --mode full                    # Full benchmark suite
    python scripts/speed_benchmark.py --mode quick                   # Quick CI validation
    python scripts/speed_benchmark.py --mode emergency              # Emergency mode tests
    python scripts/speed_benchmark.py --mode comparison             # Before/after comparison
    python scripts/speed_benchmark.py --query "Apple Inc analysis"  # Custom query test
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_speed_optimization_validation import (
    QueryComplexity,
    SPEED_TEST_QUERIES,
    SPEED_THRESHOLDS,
    MODEL_SPEED_BENCHMARKS,
    SpeedOptimizationValidator,
    SpeedTestMonitor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpeedBenchmarkRunner:
    """Runs comprehensive speed benchmarks and generates reports."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_version": "1.0.0",
            "system_info": self._get_system_info(),
            "test_results": {},
            "summary": {},
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation suitable for CI pipeline."""
        logger.info("🚀 Running quick speed validation for CI...")
        
        validator = SpeedOptimizationValidator()
        results = {
            "emergency_mode": [],
            "model_selection": [],
            "speedup_validation": [],
        }
        
        # Test 1: Emergency mode performance (1 query per complexity)
        for complexity in [QueryComplexity.EMERGENCY, QueryComplexity.SIMPLE]:
            query = SPEED_TEST_QUERIES[complexity][0]
            
            with SpeedTestMonitor(f"quick_{complexity.value}", complexity) as monitor:
                if complexity == QueryComplexity.EMERGENCY:
                    result = await validator.test_emergency_mode_performance(query)
                else:
                    result = await validator.test_baseline_vs_optimized_performance(
                        query, complexity
                    )
                
                results["emergency_mode" if complexity == QueryComplexity.EMERGENCY else "speedup_validation"].append({
                    "complexity": complexity.value,
                    "query": query[:50] + "...",
                    "execution_time": monitor.total_execution_time,
                    "result": result,
                })
        
        # Test 2: Model selection validation
        from tests.test_speed_optimization_validation import MockOpenRouterProvider
        from maverick_mcp.utils.llm_optimization import AdaptiveModelSelector
        
        provider = MockOpenRouterProvider()
        selector = AdaptiveModelSelector(provider)
        
        emergency_config = selector.select_model_for_time_budget(
            task_type="quick_answer",  # Use string instead of enum
            time_remaining_seconds=10.0,
            complexity_score=0.3,
            content_size_tokens=200,
        )
        
        results["model_selection"].append({
            "scenario": "emergency_10s",
            "selected_model": emergency_config.model_id,
            "timeout_seconds": emergency_config.timeout_seconds,
            "is_fast_model": emergency_config.model_id in [
                "google/gemini-2.5-flash", 
                "openai/gpt-4o-mini"
            ],
        })
        
        return results
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive speed benchmark suite."""
        logger.info("🏁 Running full speed benchmark suite...")
        
        validator = SpeedOptimizationValidator()
        results = {
            "complexity_tests": {},
            "speedup_analysis": {},
            "timeout_resolution": {},
            "model_optimization": {},
        }
        
        # Test all complexity levels
        for complexity in QueryComplexity:
            logger.info(f"Testing {complexity.value} queries...")
            
            complexity_results = []
            queries = SPEED_TEST_QUERIES[complexity]
            
            for query in queries[:2]:  # Test 2 queries per complexity
                with SpeedTestMonitor(f"full_{complexity.value}", complexity) as monitor:
                    if complexity == QueryComplexity.EMERGENCY:
                        result = await validator.test_emergency_mode_performance(query)
                    else:
                        result = await validator.test_baseline_vs_optimized_performance(
                            query, complexity
                        )
                    
                    complexity_results.append({
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "execution_time": monitor.total_execution_time,
                        "result": result,
                    })
            
            results["complexity_tests"][complexity.value] = complexity_results
        
        # Speedup analysis
        speedup_results = []
        for complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            query = SPEED_TEST_QUERIES[complexity][0]
            result = await validator.test_baseline_vs_optimized_performance(query, complexity)
            if result["baseline_success"] and result["optimized_success"]:
                speedup_results.append({
                    "complexity": complexity.value,
                    "speedup_factor": result["speedup_factor"],
                    "baseline_time": result["baseline_time"],
                    "optimized_time": result["optimized_time"],
                })
        
        results["speedup_analysis"] = {
            "individual_results": speedup_results,
            "average_speedup": statistics.mean([r["speedup_factor"] for r in speedup_results]) if speedup_results else 0,
            "meets_2x_threshold": all(r["speedup_factor"] >= 2.0 for r in speedup_results),
            "meets_3x_threshold": all(r["speedup_factor"] >= 3.0 for r in speedup_results),
        }
        
        # Model optimization validation
        from tests.test_speed_optimization_validation import MockOpenRouterProvider
        from maverick_mcp.utils.llm_optimization import AdaptiveModelSelector
        
        provider = MockOpenRouterProvider()
        selector = AdaptiveModelSelector(provider)
        
        model_tests = []
        for time_budget, expected_speed_tier in [(8, "ultra_fast"), (20, "fast"), (60, "balanced")]:
            config = selector.select_model_for_time_budget(
                task_type="market_analysis",  # Use string
                time_remaining_seconds=time_budget,
                complexity_score=0.5,
                content_size_tokens=1000,
            )
            
            model_tests.append({
                "time_budget": time_budget,
                "selected_model": config.model_id,
                "expected_tier": expected_speed_tier,
                "timeout_seconds": config.timeout_seconds,
                "max_tokens": config.max_tokens,
            })
        
        results["model_optimization"] = model_tests
        
        return results
    
    async def run_emergency_mode_benchmark(self) -> Dict[str, Any]:
        """Run focused emergency mode benchmark."""
        logger.info("⚡ Running emergency mode benchmark...")
        
        validator = SpeedOptimizationValidator()
        emergency_queries = SPEED_TEST_QUERIES[QueryComplexity.EMERGENCY]
        
        results = {
            "emergency_tests": [],
            "timeout_compliance": {},
            "model_selection": {},
        }
        
        timeout_failures = 0
        total_tests = 0
        
        for query in emergency_queries:
            total_tests += 1
            
            with SpeedTestMonitor(f"emergency_{total_tests}", QueryComplexity.EMERGENCY) as monitor:
                result = await validator.test_emergency_mode_performance(query)
                
                if result["execution_time"] >= SPEED_THRESHOLDS["emergency_mode_max_time"]:
                    timeout_failures += 1
                
                results["emergency_tests"].append({
                    "query": query[:80] + "..." if len(query) > 80 else query,
                    "execution_time": result["execution_time"],
                    "within_budget": result["within_budget"],
                    "success": result["success"],
                    "emergency_mode_used": result.get("emergency_mode_used", False),
                })
        
        results["timeout_compliance"] = {
            "total_tests": total_tests,
            "timeout_failures": timeout_failures,
            "failure_rate": timeout_failures / max(total_tests, 1),
            "passes_threshold": (timeout_failures / max(total_tests, 1)) <= SPEED_THRESHOLDS["timeout_failure_threshold"],
        }
        
        return results
    
    async def run_custom_query_benchmark(self, query: str) -> Dict[str, Any]:
        """Run benchmark on a custom query."""
        logger.info(f"🎯 Running custom query benchmark: {query[:50]}...")
        
        validator = SpeedOptimizationValidator()
        
        # Determine complexity heuristically
        complexity = QueryComplexity.SIMPLE
        if len(query.split()) > 20:
            complexity = QueryComplexity.MODERATE
        if len(query.split()) > 40:
            complexity = QueryComplexity.COMPLEX
        
        with SpeedTestMonitor("custom_query", complexity) as monitor:
            if "quick" in query.lower() or "urgent" in query.lower():
                result = await validator.test_emergency_mode_performance(query)
            else:
                result = await validator.test_baseline_vs_optimized_performance(query, complexity)
        
        return {
            "query": query,
            "detected_complexity": complexity.value,
            "execution_time": monitor.total_execution_time,
            "result": result,
        }
    
    async def run_comparison_benchmark(self) -> Dict[str, Any]:
        """Run before/after comparison benchmark."""
        logger.info("📊 Running baseline vs optimized comparison benchmark...")
        
        validator = SpeedOptimizationValidator()
        results = {
            "comparisons": [],
            "summary": {},
        }
        
        # Test across different complexities
        for complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            query = SPEED_TEST_QUERIES[complexity][0]
            
            result = await validator.test_baseline_vs_optimized_performance(query, complexity)
            
            if result["baseline_success"] and result["optimized_success"]:
                results["comparisons"].append({
                    "complexity": complexity.value,
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "baseline_time": result["baseline_time"],
                    "optimized_time": result["optimized_time"],
                    "speedup_factor": result["speedup_factor"],
                    "improvement_percent": ((result["baseline_time"] - result["optimized_time"]) / result["baseline_time"]) * 100,
                })
        
        if results["comparisons"]:
            speedups = [c["speedup_factor"] for c in results["comparisons"]]
            results["summary"] = {
                "average_speedup": statistics.mean(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "meets_2x_target": all(s >= 2.0 for s in speedups),
                "meets_3x_target": all(s >= 3.0 for s in speedups),
            }
        
        return results
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# MaverickMCP Speed Benchmark Report")
        report.append(f"Generated: {self.results['timestamp']}")
        report.append(f"System: {self.results['system_info']['platform']}")
        report.append("")
        
        # Quick validation results
        if "emergency_mode" in test_results:
            report.append("## Quick Validation Results")
            emergency_results = test_results["emergency_mode"]
            if emergency_results:
                avg_time = statistics.mean([r["execution_time"] for r in emergency_results])
                report.append(f"- Emergency mode average time: {avg_time:.2f}s")
                
            speedup_results = test_results.get("speedup_validation", [])
            if speedup_results:
                for result in speedup_results:
                    if "result" in result and "speedup_factor" in result["result"]:
                        speedup = result["result"]["speedup_factor"]
                        report.append(f"- {result['complexity']} speedup: {speedup:.2f}x")
            report.append("")
        
        # Full benchmark results
        if "complexity_tests" in test_results:
            report.append("## Complexity Test Results")
            for complexity, results in test_results["complexity_tests"].items():
                if results:
                    avg_time = statistics.mean([r["execution_time"] for r in results])
                    threshold_map = {
                        "simple": SPEED_THRESHOLDS["simple_query_max_time"],
                        "moderate": SPEED_THRESHOLDS["moderate_query_max_time"],
                        "complex": SPEED_THRESHOLDS["complex_query_max_time"],
                        "emergency": SPEED_THRESHOLDS["emergency_mode_max_time"],
                    }
                    threshold = threshold_map.get(complexity, 30.0)
                    passes = "✅" if avg_time < threshold else "❌"
                    report.append(f"- {complexity.title()}: {avg_time:.2f}s (threshold: {threshold}s) {passes}")
            report.append("")
        
        # Speedup analysis
        if "speedup_analysis" in test_results:
            analysis = test_results["speedup_analysis"]
            report.append("## Speedup Analysis")
            if "average_speedup" in analysis:
                avg_speedup = analysis["average_speedup"]
                report.append(f"- Average speedup: {avg_speedup:.2f}x")
                
                meets_2x = "✅" if analysis.get("meets_2x_threshold", False) else "❌"
                meets_3x = "✅" if analysis.get("meets_3x_threshold", False) else "❌"
                
                report.append(f"- Meets 2x target: {meets_2x}")
                report.append(f"- Meets 3x target: {meets_3x}")
            report.append("")
        
        # Emergency mode results
        if "emergency_tests" in test_results:
            report.append("## Emergency Mode Performance")
            emergency_tests = test_results["emergency_tests"]
            if emergency_tests:
                avg_time = statistics.mean([t["execution_time"] for t in emergency_tests])
                success_rate = sum(1 for t in emergency_tests if t["success"]) / len(emergency_tests)
                
                report.append(f"- Average emergency time: {avg_time:.2f}s")
                report.append(f"- Success rate: {success_rate:.1%}")
                
                if "timeout_compliance" in test_results:
                    compliance = test_results["timeout_compliance"]
                    passes_compliance = "✅" if compliance.get("passes_threshold", False) else "❌"
                    report.append(f"- Timeout compliance: {passes_compliance} ({compliance.get('failure_rate', 0):.1%} failure rate)")
            report.append("")
        
        # Custom query results
        if "query" in test_results:
            report.append("## Custom Query Results")
            report.append(f"- Query: {test_results['query'][:100]}...")
            report.append(f"- Complexity: {test_results['detected_complexity']}")
            report.append(f"- Execution time: {test_results['execution_time']:.2f}s")
            report.append("")
        
        # Comparison results
        if "comparisons" in test_results:
            report.append("## Baseline vs Optimized Comparison")
            for comp in test_results["comparisons"]:
                report.append(f"- {comp['complexity'].title()}: {comp['baseline_time']:.2f}s → {comp['optimized_time']:.2f}s ({comp['speedup_factor']:.2f}x speedup)")
            
            if "summary" in test_results:
                summary = test_results["summary"]
                report.append(f"- Overall average speedup: {summary['average_speedup']:.2f}x")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        # Generate recommendations based on results
        recommendations = []
        
        if "speedup_analysis" in test_results:
            analysis = test_results["speedup_analysis"]
            if not analysis.get("meets_2x_threshold", True):
                recommendations.append("🔧 Investigate speed optimizations - not meeting 2x speedup target")
            elif analysis.get("meets_3x_threshold", False):
                recommendations.append("🎉 Excellent performance - exceeding 3x speedup target!")
            else:
                recommendations.append("📈 Good performance - meeting 2x target, opportunity for 3x")
        
        if "timeout_compliance" in test_results:
            compliance = test_results["timeout_compliance"]
            if not compliance.get("passes_threshold", True):
                recommendations.append("⚠️  Address timeout issues in emergency mode")
        
        if not recommendations:
            recommendations.append("✨ All performance targets met - system is well optimized!")
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        return "\n".join(report)
    
    def save_results(self, test_results: Dict[str, Any], mode: str):
        """Save benchmark results to files."""
        self.results["test_results"][mode] = test_results
        
        # Save JSON results
        json_file = self.output_dir / f"speed_benchmark_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save markdown report
        report = self.generate_report(test_results)
        md_file = self.output_dir / f"speed_benchmark_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Results saved to: {json_file} and {md_file}")
        
        return json_file, md_file


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="MaverickMCP Speed Benchmark Tool")
    parser.add_argument(
        "--mode", 
        choices=["full", "quick", "emergency", "comparison"],
        default="quick",
        help="Benchmark mode to run"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query to benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create benchmark runner
    runner = SpeedBenchmarkRunner(args.output_dir)
    
    try:
        # Run appropriate benchmark
        if args.query:
            logger.info(f"Running custom query benchmark: {args.query[:50]}...")
            results = await runner.run_custom_query_benchmark(args.query)
            mode = "custom"
        elif args.mode == "full":
            results = await runner.run_full_benchmark()
            mode = "full"
        elif args.mode == "quick":
            results = await runner.run_quick_validation()
            mode = "quick"
        elif args.mode == "emergency":
            results = await runner.run_emergency_mode_benchmark()
            mode = "emergency"
        elif args.mode == "comparison":
            results = await runner.run_comparison_benchmark()
            mode = "comparison"
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
        
        # Save results and generate report
        json_file, md_file = runner.save_results(results, mode)
        
        # Print summary to console
        report = runner.generate_report(results)
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        # Determine exit code based on results
        exit_code = 0
        
        # Check for critical failures
        if "timeout_compliance" in results:
            if not results["timeout_compliance"].get("passes_threshold", True):
                exit_code = 1
        
        if "speedup_analysis" in results:
            if not results["speedup_analysis"].get("meets_2x_threshold", True):
                exit_code = 1
        
        if exit_code == 0:
            logger.info("🎉 All benchmarks passed!")
        else:
            logger.warning("⚠️  Some benchmarks failed - check results for details")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)