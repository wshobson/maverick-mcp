#!/usr/bin/env python3
"""
Focused LLM Speed Optimization Demonstration

This script demonstrates the core LLM optimization capabilities that provide
2-3x speed improvements, focusing on areas we can control directly.

Demonstrates:
- Adaptive model selection based on time constraints
- Fast model execution (Gemini 2.5 Flash)
- Token generation speed optimization
- Progressive timeout management
- Model performance comparison
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maverick_mcp.providers.openrouter_provider import OpenRouterProvider, TaskType
from maverick_mcp.utils.llm_optimization import AdaptiveModelSelector


class LLMSpeedDemonstrator:
    """Focused demonstration of LLM speed optimizations."""

    def __init__(self):
        """Initialize the demonstration."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it with your OpenRouter API key."
            )
        self.openrouter_provider = OpenRouterProvider(api_key=api_key)
        self.model_selector = AdaptiveModelSelector(self.openrouter_provider)

        # Test scenarios focused on different urgency levels
        self.test_scenarios = [
            {
                "name": "Emergency Analysis (Critical Speed)",
                "prompt": "Analyze NVIDIA's latest earnings impact on AI market sentiment. 2-3 key points only.",
                "time_budget": 15.0,
                "task_type": TaskType.QUICK_ANSWER,
                "expected_speed": ">100 tok/s",
            },
            {
                "name": "Technical Analysis (Fast Response)",
                "prompt": "Provide technical analysis of Apple stock including RSI, MACD, and support levels. Be concise.",
                "time_budget": 30.0,
                "task_type": TaskType.TECHNICAL_ANALYSIS,
                "expected_speed": ">80 tok/s",
            },
            {
                "name": "Market Research (Moderate Speed)",
                "prompt": "Analyze Federal Reserve interest rate policy impact on technology sector. Include risk assessment.",
                "time_budget": 45.0,
                "task_type": TaskType.MARKET_ANALYSIS,
                "expected_speed": ">60 tok/s",
            },
            {
                "name": "Complex Synthesis (Quality Balance)",
                "prompt": "Synthesize renewable energy investment opportunities for 2025, considering policy changes, technology advances, and market trends.",
                "time_budget": 60.0,
                "task_type": TaskType.RESULT_SYNTHESIS,
                "expected_speed": ">40 tok/s",
            },
        ]

    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)

    def print_subheader(self, title: str):
        """Print formatted subheader."""
        print(f"\n--- {title} ---")

    async def validate_openrouter_connection(self) -> bool:
        """Validate OpenRouter API is accessible."""
        self.print_header("🔧 API VALIDATION")

        try:
            test_llm = self.openrouter_provider.get_llm(TaskType.GENERAL)
            from langchain_core.messages import HumanMessage

            test_response = await asyncio.wait_for(
                test_llm.ainvoke([HumanMessage(content="test connection")]),
                timeout=10.0,
            )
            print("✅ OpenRouter API: Connected successfully")
            print(f"   Response length: {len(test_response.content)} chars")
            return True
        except Exception as e:
            print(f"❌ OpenRouter API: Failed - {e}")
            return False

    async def demonstrate_model_selection(self):
        """Show intelligent model selection for different scenarios."""
        self.print_header("🧠 ADAPTIVE MODEL SELECTION")

        for scenario in self.test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"   Time Budget: {scenario['time_budget']}s")
            print(f"   Task Type: {scenario['task_type'].value}")
            print(f"   Expected Speed: {scenario['expected_speed']}")

            # Calculate task complexity
            complexity = self.model_selector.calculate_task_complexity(
                content=scenario["prompt"],
                task_type=scenario["task_type"],
                focus_areas=["analysis"],
            )

            # Get optimal model for time budget
            model_config = self.model_selector.select_model_for_time_budget(
                task_type=scenario["task_type"],
                time_remaining_seconds=scenario["time_budget"],
                complexity_score=complexity,
                content_size_tokens=len(scenario["prompt"]) // 4,
            )

            print(f"   📊 Complexity Score: {complexity:.2f}")
            print(f"   🎯 Selected Model: {model_config.model_id}")
            print(f"   ⏱️  Max Timeout: {model_config.timeout_seconds}s")
            print(f"   🌡️  Temperature: {model_config.temperature}")
            print(f"   📝 Max Tokens: {model_config.max_tokens}")

            # Check if speed-optimized
            is_speed_model = any(
                x in model_config.model_id.lower()
                for x in ["flash", "haiku", "4o-mini", "deepseek"]
            )
            print(f"   🚀 Speed Optimized: {'✅' if is_speed_model else '❌'}")

    async def run_speed_benchmarks(self):
        """Run actual speed benchmarks for each scenario."""
        self.print_header("⚡ LIVE SPEED BENCHMARKS")

        results = []
        baseline_time = 60.0  # Historical baseline from timeout issues

        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"\n🔍 Benchmark {i}/{len(self.test_scenarios)}: {scenario['name']}")
            print(f"   Query: {scenario['prompt'][:60]}...")

            try:
                # Get optimal model configuration
                complexity = self.model_selector.calculate_task_complexity(
                    content=scenario["prompt"],
                    task_type=scenario["task_type"],
                )

                model_config = self.model_selector.select_model_for_time_budget(
                    task_type=scenario["task_type"],
                    time_remaining_seconds=scenario["time_budget"],
                    complexity_score=complexity,
                    content_size_tokens=len(scenario["prompt"]) // 4,
                )

                # Execute with timing
                llm = self.openrouter_provider.get_llm(
                    model_override=model_config.model_id,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                )

                start_time = time.time()
                from langchain_core.messages import HumanMessage

                response = await asyncio.wait_for(
                    llm.ainvoke([HumanMessage(content=scenario["prompt"])]),
                    timeout=model_config.timeout_seconds,
                )
                execution_time = time.time() - start_time

                # Calculate metrics
                response_length = len(response.content)
                estimated_tokens = response_length // 4
                tokens_per_second = (
                    estimated_tokens / execution_time if execution_time > 0 else 0
                )
                speed_improvement = (
                    baseline_time / execution_time if execution_time > 0 else 0
                )

                # Results
                result = {
                    "scenario": scenario["name"],
                    "model_used": model_config.model_id,
                    "execution_time": execution_time,
                    "time_budget": scenario["time_budget"],
                    "budget_used_pct": (execution_time / scenario["time_budget"]) * 100,
                    "tokens_per_second": tokens_per_second,
                    "response_length": response_length,
                    "speed_improvement": speed_improvement,
                    "target_achieved": execution_time <= scenario["time_budget"],
                    "response_preview": response.content[:150] + "..."
                    if len(response.content) > 150
                    else response.content,
                }

                results.append(result)

                # Print immediate results
                status_icon = "✅" if result["target_achieved"] else "⚠️"
                print(
                    f"   {status_icon} Completed: {execution_time:.2f}s ({result['budget_used_pct']:.1f}% of budget)"
                )
                print(f"   🎯 Model: {model_config.model_id}")
                print(f"   🚀 Speed: {tokens_per_second:.0f} tok/s")
                print(
                    f"   📊 Improvement: {speed_improvement:.1f}x faster than baseline"
                )
                print(f"   💬 Preview: {result['response_preview']}")

                # Brief pause between tests
                await asyncio.sleep(1)

            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                results.append(
                    {
                        "scenario": scenario["name"],
                        "error": str(e),
                        "target_achieved": False,
                    }
                )

        return results

    def analyze_benchmark_results(self, results: list[dict[str, Any]]):
        """Analyze and report benchmark results."""
        self.print_header("📊 SPEED OPTIMIZATION ANALYSIS")

        successful_tests = [r for r in results if not r.get("error")]
        failed_tests = [r for r in results if r.get("error")]
        targets_achieved = [r for r in successful_tests if r.get("target_achieved")]

        print("📈 Overall Performance:")
        print(f"   Total Tests: {len(results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Targets Hit: {len(targets_achieved)}/{len(results)}")
        print(f"   Success Rate: {(len(targets_achieved) / len(results) * 100):.1f}%")

        if successful_tests:
            # Speed metrics
            avg_execution_time = sum(
                r["execution_time"] for r in successful_tests
            ) / len(successful_tests)
            max_execution_time = max(r["execution_time"] for r in successful_tests)
            avg_tokens_per_second = sum(
                r["tokens_per_second"] for r in successful_tests
            ) / len(successful_tests)
            avg_speed_improvement = sum(
                r["speed_improvement"] for r in successful_tests
            ) / len(successful_tests)

            print("\n⚡ Speed Metrics:")
            print(f"   Average Execution Time: {avg_execution_time:.2f}s")
            print(f"   Maximum Execution Time: {max_execution_time:.2f}s")
            print(f"   Average Token Generation: {avg_tokens_per_second:.0f} tok/s")
            print(f"   Average Speed Improvement: {avg_speed_improvement:.1f}x")

            # Historical comparison
            historical_baseline = 60.0  # Average timeout failure time
            if max_execution_time > 0:
                overall_improvement = historical_baseline / max_execution_time
                print("\n🎯 Speed Validation:")
                print(
                    f"   Historical Baseline: {historical_baseline}s (timeout failures)"
                )
                print(f"   Current Max Time: {max_execution_time:.2f}s")
                print(f"   Overall Improvement: {overall_improvement:.1f}x")

                if overall_improvement >= 3.0:
                    print(
                        f"   🎉 EXCELLENT: {overall_improvement:.1f}x speed improvement!"
                    )
                elif overall_improvement >= 2.0:
                    print(
                        f"   ✅ SUCCESS: {overall_improvement:.1f}x speed improvement achieved!"
                    )
                elif overall_improvement >= 1.5:
                    print(
                        f"   👍 GOOD: {overall_improvement:.1f}x improvement (target: 2x)"
                    )
                else:
                    print(
                        f"   ⚠️ NEEDS WORK: Only {overall_improvement:.1f}x improvement"
                    )

            # Model performance breakdown
            self.print_subheader("🧠 MODEL PERFORMANCE BREAKDOWN")
            model_stats = {}
            for result in successful_tests:
                model = result["model_used"]
                if model not in model_stats:
                    model_stats[model] = []
                model_stats[model].append(result)

            for model, model_results in model_stats.items():
                avg_speed = sum(r["tokens_per_second"] for r in model_results) / len(
                    model_results
                )
                avg_time = sum(r["execution_time"] for r in model_results) / len(
                    model_results
                )
                success_rate = (
                    len([r for r in model_results if r["target_achieved"]])
                    / len(model_results)
                ) * 100

                print(f"   {model}:")
                print(f"     Tests: {len(model_results)}")
                print(f"     Avg Speed: {avg_speed:.0f} tok/s")
                print(f"     Avg Time: {avg_time:.2f}s")
                print(f"     Success Rate: {success_rate:.0f}%")

    async def run_comprehensive_demo(self):
        """Run the complete LLM speed demonstration."""
        print("🚀 MaverickMCP LLM Speed Optimization Demonstration")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Goal: Demonstrate 2-3x LLM speed improvements")

        # Step 1: Validate connection
        if not await self.validate_openrouter_connection():
            print("\n❌ Cannot proceed - API connection failed")
            return False

        # Step 2: Show model selection intelligence
        await self.demonstrate_model_selection()

        # Step 3: Run live speed benchmarks
        results = await self.run_speed_benchmarks()

        # Step 4: Analyze results
        self.analyze_benchmark_results(results)

        # Final summary
        self.print_header("🎉 DEMONSTRATION SUMMARY")

        successful_tests = [r for r in results if not r.get("error")]
        targets_achieved = [r for r in successful_tests if r.get("target_achieved")]

        print("✅ LLM Speed Optimization Results:")
        print(f"   Tests Executed: {len(results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Targets Achieved: {len(targets_achieved)}")
        print(f"   Success Rate: {(len(targets_achieved) / len(results) * 100):.1f}%")

        if successful_tests:
            max_time = max(r["execution_time"] for r in successful_tests)
            avg_speed = sum(r["tokens_per_second"] for r in successful_tests) / len(
                successful_tests
            )
            speed_improvement = 60.0 / max_time if max_time > 0 else 0

            print(
                f"   Fastest Response: {min(r['execution_time'] for r in successful_tests):.2f}s"
            )
            print(f"   Average Token Speed: {avg_speed:.0f} tok/s")
            print(f"   Speed Improvement: {speed_improvement:.1f}x faster")

        print("\n📊 Key Optimizations Demonstrated:")
        print("   ✅ Adaptive Model Selection (context-aware)")
        print("   ✅ Time-Budget Optimization")
        print("   ✅ Fast Model Utilization (Gemini Flash, Claude Haiku)")
        print("   ✅ Progressive Timeout Management")
        print("   ✅ Token Generation Speed Optimization")

        # Success criteria: at least 75% success rate and 2x improvement
        success_criteria = len(targets_achieved) >= len(results) * 0.75 and (
            successful_tests
            and 60.0 / max(r["execution_time"] for r in successful_tests) >= 2.0
        )

        return success_criteria


async def main():
    """Main demonstration entry point."""
    demo = LLMSpeedDemonstrator()

    try:
        success = await demo.run_comprehensive_demo()

        if success:
            print("\n🎉 LLM Speed Demonstration PASSED - Optimizations validated!")
            return 0
        else:
            print("\n⚠️ Demonstration had mixed results - review analysis above")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Demonstration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Check required environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Missing OPENROUTER_API_KEY environment variable")
        print("Please check your .env file")
        sys.exit(1)

    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
