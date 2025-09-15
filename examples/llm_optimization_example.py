"""
LLM Optimization Example for Research Agents - Speed-Optimized Edition.

This example demonstrates how to use the comprehensive LLM optimization strategies
with new speed-optimized models to prevent research agent timeouts while maintaining
research quality. Features 2-3x speed improvements with Gemini 2.5 Flash and GPT-4o Mini.
"""

import asyncio
import logging
import os
import time
from typing import Any

from maverick_mcp.agents.optimized_research import (
    OptimizedDeepResearchAgent,
    create_optimized_research_agent,
)
from maverick_mcp.config.llm_optimization_config import (
    ModelSelectionStrategy,
    ResearchComplexity,
    create_adaptive_config,
    create_balanced_config,
    create_emergency_config,
    create_fast_config,
)
from maverick_mcp.providers.openrouter_provider import (
    OpenRouterProvider,
    TaskType,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptimizationExamples:
    """Examples demonstrating LLM optimization strategies."""

    def __init__(self, openrouter_api_key: str):
        """Initialize with OpenRouter API key."""
        self.openrouter_api_key = openrouter_api_key

    async def example_1_emergency_research(self) -> dict[str, Any]:
        """
        Example 1: Emergency research with <20 second time budget.

        Use case: Real-time alerts or urgent market events requiring immediate analysis.
        """
        logger.info("🚨 Example 1: Emergency Research (<20s)")

        # Create emergency configuration (for optimization reference)
        _ = create_emergency_config(time_budget=15.0)

        # Create optimized agent
        agent = create_optimized_research_agent(
            openrouter_api_key=self.openrouter_api_key,
            persona="aggressive",  # Aggressive for quick decisions
            time_budget_seconds=15.0,
            target_confidence=0.6,  # Lower bar for emergency
        )

        # Execute emergency research
        start_time = time.time()

        result = await agent.research_comprehensive(
            topic="NVDA earnings surprise impact",
            session_id="emergency_001",
            depth="basic",
            focus_areas=["sentiment", "catalyst"],
            time_budget_seconds=15.0,
            target_confidence=0.6,
        )

        execution_time = time.time() - start_time

        logger.info(f"✅ Emergency research completed in {execution_time:.2f}s")
        logger.info(
            f"Optimization features used: {result.get('optimization_metrics', {}).get('optimization_features_used', [])}"
        )

        return {
            "scenario": "emergency",
            "time_budget": 15.0,
            "actual_time": execution_time,
            "success": execution_time < 20,  # Success if under 20s
            "confidence": result.get("findings", {}).get("confidence_score", 0),
            "sources_processed": result.get("sources_analyzed", 0),
            "optimization_features": result.get("optimization_metrics", {}).get(
                "optimization_features_used", []
            ),
        }

    async def example_2_fast_research(self) -> dict[str, Any]:
        """
        Example 2: Fast research with 45 second time budget.

        Use case: Quick analysis for trading decisions or portfolio updates.
        """
        logger.info("⚡ Example 2: Fast Research (45s)")

        # Create fast configuration
        _ = create_fast_config(time_budget=45.0)

        # Create optimized agent
        agent = create_optimized_research_agent(
            openrouter_api_key=self.openrouter_api_key,
            persona="moderate",
            time_budget_seconds=45.0,
            target_confidence=0.7,
        )

        start_time = time.time()

        result = await agent.research_comprehensive(
            topic="Tesla Q4 2024 delivery numbers analysis",
            session_id="fast_001",
            depth="standard",
            focus_areas=["fundamental", "sentiment"],
            time_budget_seconds=45.0,
            target_confidence=0.7,
        )

        execution_time = time.time() - start_time

        logger.info(f"✅ Fast research completed in {execution_time:.2f}s")

        return {
            "scenario": "fast",
            "time_budget": 45.0,
            "actual_time": execution_time,
            "success": execution_time < 60,
            "confidence": result.get("findings", {}).get("confidence_score", 0),
            "sources_processed": result.get("sources_analyzed", 0),
            "early_terminated": result.get("findings", {}).get(
                "early_terminated", False
            ),
        }

    async def example_3_balanced_research(self) -> dict[str, Any]:
        """
        Example 3: Balanced research with 2 minute time budget.

        Use case: Standard research for investment decisions.
        """
        logger.info("⚖️ Example 3: Balanced Research (120s)")

        # Create balanced configuration
        _ = create_balanced_config(time_budget=120.0)

        agent = create_optimized_research_agent(
            openrouter_api_key=self.openrouter_api_key,
            persona="conservative",
            time_budget_seconds=120.0,
            target_confidence=0.75,
        )

        start_time = time.time()

        result = await agent.research_comprehensive(
            topic="Microsoft cloud services competitive position 2024",
            session_id="balanced_001",
            depth="comprehensive",
            focus_areas=["competitive", "fundamental", "technical"],
            time_budget_seconds=120.0,
            target_confidence=0.75,
        )

        execution_time = time.time() - start_time

        logger.info(f"✅ Balanced research completed in {execution_time:.2f}s")

        return {
            "scenario": "balanced",
            "time_budget": 120.0,
            "actual_time": execution_time,
            "success": execution_time < 150,  # 25% buffer
            "confidence": result.get("findings", {}).get("confidence_score", 0),
            "sources_processed": result.get("sources_analyzed", 0),
            "processing_mode": result.get("findings", {}).get(
                "processing_mode", "unknown"
            ),
        }

    async def example_4_adaptive_research(self) -> dict[str, Any]:
        """
        Example 4: Adaptive research that adjusts based on complexity and available time.

        Use case: Dynamic research where time constraints may vary.
        """
        logger.info("🎯 Example 4: Adaptive Research")

        # Simulate varying time constraints
        scenarios = [
            {
                "time_budget": 30,
                "complexity": ResearchComplexity.SIMPLE,
                "topic": "Apple stock price today",
            },
            {
                "time_budget": 90,
                "complexity": ResearchComplexity.MODERATE,
                "topic": "Federal Reserve interest rate policy impact on tech stocks",
            },
            {
                "time_budget": 180,
                "complexity": ResearchComplexity.COMPLEX,
                "topic": "Cryptocurrency regulation implications for financial institutions",
            },
        ]

        results = []

        for i, scenario in enumerate(scenarios):
            logger.info(
                f"📊 Adaptive scenario {i + 1}: {scenario['complexity'].value} complexity, {scenario['time_budget']}s budget"
            )

            # Create adaptive configuration
            config = create_adaptive_config(
                time_budget_seconds=scenario["time_budget"],
                complexity=scenario["complexity"],
            )

            agent = create_optimized_research_agent(
                openrouter_api_key=self.openrouter_api_key, persona="moderate"
            )

            start_time = time.time()

            result = await agent.research_comprehensive(
                topic=scenario["topic"],
                session_id=f"adaptive_{i + 1:03d}",
                time_budget_seconds=scenario["time_budget"],
                target_confidence=config.preset.target_confidence,
            )

            execution_time = time.time() - start_time

            scenario_result = {
                "scenario_id": i + 1,
                "complexity": scenario["complexity"].value,
                "time_budget": scenario["time_budget"],
                "actual_time": execution_time,
                "success": execution_time < scenario["time_budget"] * 1.1,  # 10% buffer
                "confidence": result.get("findings", {}).get("confidence_score", 0),
                "sources_processed": result.get("sources_analyzed", 0),
                "adaptations_used": result.get("optimization_metrics", {}).get(
                    "optimization_features_used", []
                ),
            }

            results.append(scenario_result)

            logger.info(
                f"✅ Adaptive scenario {i + 1} completed in {execution_time:.2f}s"
            )

        return {
            "scenario": "adaptive",
            "scenarios_tested": len(scenarios),
            "results": results,
            "overall_success": all(r["success"] for r in results),
        }

    async def example_5_optimization_comparison(self) -> dict[str, Any]:
        """
        Example 5: Compare optimized vs non-optimized research performance.

        Use case: Demonstrate the effectiveness of optimizations.
        """
        logger.info("📈 Example 5: Optimization Comparison")

        test_topic = "Amazon Web Services market share growth 2024"
        time_budget = 90.0

        results = {}

        # Test with optimizations enabled
        logger.info("🔧 Testing WITH optimizations...")

        optimized_agent = OptimizedDeepResearchAgent(
            openrouter_provider=OpenRouterProvider(self.openrouter_api_key),
            persona="moderate",
            optimization_enabled=True,
        )

        start_time = time.time()
        optimized_result = await optimized_agent.research_comprehensive(
            topic=test_topic,
            session_id="comparison_optimized",
            time_budget_seconds=time_budget,
            target_confidence=0.75,
        )
        optimized_time = time.time() - start_time

        results["optimized"] = {
            "execution_time": optimized_time,
            "success": optimized_time < time_budget,
            "confidence": optimized_result.get("findings", {}).get(
                "confidence_score", 0
            ),
            "sources_processed": optimized_result.get("sources_analyzed", 0),
            "optimization_features": optimized_result.get(
                "optimization_metrics", {}
            ).get("optimization_features_used", []),
        }

        # Test with optimizations disabled
        logger.info("🐌 Testing WITHOUT optimizations...")

        standard_agent = OptimizedDeepResearchAgent(
            openrouter_provider=OpenRouterProvider(self.openrouter_api_key),
            persona="moderate",
            optimization_enabled=False,
        )

        start_time = time.time()
        try:
            standard_result = await asyncio.wait_for(
                standard_agent.research_comprehensive(
                    topic=test_topic, session_id="comparison_standard", depth="standard"
                ),
                timeout=time_budget + 30,  # Give extra time for timeout demonstration
            )
            standard_time = time.time() - start_time

            results["standard"] = {
                "execution_time": standard_time,
                "success": standard_time < time_budget,
                "confidence": standard_result.get("findings", {}).get(
                    "confidence_score", 0
                ),
                "sources_processed": standard_result.get("sources_analyzed", 0),
                "timed_out": False,
            }

        except TimeoutError:
            standard_time = time_budget + 30
            results["standard"] = {
                "execution_time": standard_time,
                "success": False,
                "confidence": 0,
                "sources_processed": 0,
                "timed_out": True,
            }

        # Calculate improvement metrics
        time_improvement = (
            (
                results["standard"]["execution_time"]
                - results["optimized"]["execution_time"]
            )
            / results["standard"]["execution_time"]
            * 100
        )
        confidence_ratio = results["optimized"]["confidence"] / max(
            results["standard"]["confidence"], 0.01
        )

        results["comparison"] = {
            "time_improvement_pct": time_improvement,
            "optimized_faster": results["optimized"]["execution_time"]
            < results["standard"]["execution_time"],
            "confidence_ratio": confidence_ratio,
            "both_successful": results["optimized"]["success"]
            and results["standard"]["success"],
        }

        logger.info("📊 Optimization Results:")
        logger.info(
            f"   Optimized: {results['optimized']['execution_time']:.2f}s (success: {results['optimized']['success']})"
        )
        logger.info(
            f"   Standard: {results['standard']['execution_time']:.2f}s (success: {results['standard']['success']})"
        )
        logger.info(f"   Time improvement: {time_improvement:.1f}%")

        return results

    async def example_6_speed_optimized_models(self) -> dict[str, Any]:
        """
        Example 6: Test the new speed-optimized models (Gemini 2.5 Flash, GPT-4o Mini).

        Use case: Demonstrate 2-3x speed improvements with the fastest available models.
        """
        logger.info("🚀 Example 6: Speed-Optimized Models Test")

        speed_test_results = {}

        # Test Gemini 2.5 Flash (199 tokens/sec - fastest)
        logger.info("🔥 Testing Gemini 2.5 Flash (199 tokens/sec)...")
        provider = OpenRouterProvider(self.openrouter_api_key)

        gemini_llm = provider.get_llm(
            model_override="google/gemini-2.5-flash",
            task_type=TaskType.DEEP_RESEARCH,
            prefer_fast=True,
        )

        start_time = time.time()
        try:
            response = await gemini_llm.ainvoke(
                [
                    {
                        "role": "user",
                        "content": "Analyze Tesla's Q4 2024 performance in exactly 3 bullet points. Be concise and factual.",
                    }
                ]
            )
            gemini_time = time.time() - start_time

            # Safely handle content that could be string or list
            content_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
                if response.content
                else ""
            )
            speed_test_results["gemini_2_5_flash"] = {
                "execution_time": gemini_time,
                "tokens_per_second": len(content_text.split()) / gemini_time
                if gemini_time > 0
                else 0,
                "success": True,
                "response_quality": "high" if len(content_text) > 50 else "low",
            }
        except Exception as e:
            speed_test_results["gemini_2_5_flash"] = {
                "execution_time": 999,
                "success": False,
                "error": str(e),
            }

        # Test GPT-4o Mini (126 tokens/sec - excellent balance)
        logger.info("⚡ Testing GPT-4o Mini (126 tokens/sec)...")

        gpt_llm = provider.get_llm(
            model_override="openai/gpt-4o-mini",
            task_type=TaskType.MARKET_ANALYSIS,
            prefer_fast=True,
        )

        start_time = time.time()
        try:
            response = await gpt_llm.ainvoke(
                [
                    {
                        "role": "user",
                        "content": "Analyze Amazon's cloud services competitive position in exactly 3 bullet points. Be concise and factual.",
                    }
                ]
            )
            gpt_time = time.time() - start_time

            # Safely handle content that could be string or list
            content_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
                if response.content
                else ""
            )
            speed_test_results["gpt_4o_mini"] = {
                "execution_time": gpt_time,
                "tokens_per_second": len(content_text.split()) / gpt_time
                if gpt_time > 0
                else 0,
                "success": True,
                "response_quality": "high" if len(content_text) > 50 else "low",
            }
        except Exception as e:
            speed_test_results["gpt_4o_mini"] = {
                "execution_time": 999,
                "success": False,
                "error": str(e),
            }

        # Test Claude 3.5 Haiku (65.6 tokens/sec - old baseline)
        logger.info("🐌 Testing Claude 3.5 Haiku (65.6 tokens/sec - baseline)...")

        claude_llm = provider.get_llm(
            model_override="anthropic/claude-3.5-haiku",
            task_type=TaskType.QUICK_ANSWER,
            prefer_fast=True,
        )

        start_time = time.time()
        try:
            response = await claude_llm.ainvoke(
                [
                    {
                        "role": "user",
                        "content": "Analyze Microsoft's AI strategy in exactly 3 bullet points. Be concise and factual.",
                    }
                ]
            )
            claude_time = time.time() - start_time

            # Safely handle content that could be string or list
            content_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
                if response.content
                else ""
            )
            speed_test_results["claude_3_5_haiku"] = {
                "execution_time": claude_time,
                "tokens_per_second": len(content_text.split()) / claude_time
                if claude_time > 0
                else 0,
                "success": True,
                "response_quality": "high" if len(content_text) > 50 else "low",
            }
        except Exception as e:
            speed_test_results["claude_3_5_haiku"] = {
                "execution_time": 999,
                "success": False,
                "error": str(e),
            }

        # Calculate speed improvements
        baseline_time = speed_test_results.get("claude_3_5_haiku", {}).get(
            "execution_time", 10
        )

        if speed_test_results["gemini_2_5_flash"]["success"]:
            gemini_improvement = (
                (
                    baseline_time
                    - speed_test_results["gemini_2_5_flash"]["execution_time"]
                )
                / baseline_time
                * 100
            )
        else:
            gemini_improvement = 0

        if speed_test_results["gpt_4o_mini"]["success"]:
            gpt_improvement = (
                (baseline_time - speed_test_results["gpt_4o_mini"]["execution_time"])
                / baseline_time
                * 100
            )
        else:
            gpt_improvement = 0

        # Test emergency model selection
        emergency_models = ModelSelectionStrategy.get_model_priority(
            time_remaining=20.0,
            task_type=TaskType.DEEP_RESEARCH,
            complexity=ResearchComplexity.MODERATE,
        )

        logger.info("📊 Speed Test Results:")
        logger.info(
            f"   Gemini 2.5 Flash: {speed_test_results['gemini_2_5_flash']['execution_time']:.2f}s ({gemini_improvement:+.1f}% vs baseline)"
        )
        logger.info(
            f"   GPT-4o Mini: {speed_test_results['gpt_4o_mini']['execution_time']:.2f}s ({gpt_improvement:+.1f}% vs baseline)"
        )
        logger.info(
            f"   Claude 3.5 Haiku: {speed_test_results['claude_3_5_haiku']['execution_time']:.2f}s (baseline)"
        )
        logger.info(f"   Emergency models: {emergency_models[:2]}")

        return {
            "scenario": "speed_optimization",
            "models_tested": 3,
            "speed_results": speed_test_results,
            "improvements": {
                "gemini_2_5_flash_vs_baseline_pct": gemini_improvement,
                "gpt_4o_mini_vs_baseline_pct": gpt_improvement,
            },
            "emergency_models": emergency_models[:2],
            "success": all(
                result.get("success", False) for result in speed_test_results.values()
            ),
            "fastest_model": min(
                speed_test_results.items(),
                key=lambda x: x[1].get("execution_time", 999),
            )[0],
            "speed_optimization_effective": gemini_improvement > 30
            or gpt_improvement > 20,  # 30%+ or 20%+ improvement
        }

    def test_model_selection_strategy(self) -> dict[str, Any]:
        """Test the updated model selection strategy with speed-optimized models."""

        logger.info("🎯 Testing Model Selection Strategy...")

        test_scenarios = [
            {"time": 15, "task": TaskType.DEEP_RESEARCH, "desc": "Ultra Emergency"},
            {"time": 25, "task": TaskType.MARKET_ANALYSIS, "desc": "Emergency"},
            {"time": 45, "task": TaskType.TECHNICAL_ANALYSIS, "desc": "Fast"},
            {"time": 120, "task": TaskType.RESULT_SYNTHESIS, "desc": "Balanced"},
        ]

        strategy_results = {}

        for scenario in test_scenarios:
            models = ModelSelectionStrategy.get_model_priority(
                time_remaining=scenario["time"],
                task_type=scenario["task"],
                complexity=ResearchComplexity.MODERATE,
            )

            strategy_results[scenario["desc"].lower()] = {
                "time_budget": scenario["time"],
                "primary_model": models[0] if models else "None",
                "backup_models": models[1:3] if len(models) > 1 else [],
                "total_available": len(models),
                "uses_speed_optimized": any(
                    model in ["google/gemini-2.5-flash", "openai/gpt-4o-mini"]
                    for model in models[:2]
                ),
            }

            logger.info(
                f"   {scenario['desc']} ({scenario['time']}s): Primary = {models[0] if models else 'None'}"
            )

        return {
            "test_scenarios": len(test_scenarios),
            "strategy_results": strategy_results,
            "all_scenarios_use_speed_models": all(
                result["uses_speed_optimized"] for result in strategy_results.values()
            ),
            "success": True,
        }

    async def run_all_examples(self) -> dict[str, Any]:
        """Run all optimization examples and return combined results."""

        logger.info("🚀 Starting LLM Optimization Examples...")

        all_results = {}

        try:
            # Run each example
            all_results["emergency"] = await self.example_1_emergency_research()
            all_results["fast"] = await self.example_2_fast_research()
            all_results["balanced"] = await self.example_3_balanced_research()
            all_results["adaptive"] = await self.example_4_adaptive_research()
            all_results["comparison"] = await self.example_5_optimization_comparison()
            all_results[
                "speed_optimization"
            ] = await self.example_6_speed_optimized_models()
            all_results["model_strategy"] = self.test_model_selection_strategy()

            # Calculate overall success metrics
            successful_examples = sum(
                1
                for result in all_results.values()
                if result.get("success") or result.get("overall_success")
            )

            all_results["summary"] = {
                "total_examples": 7,  # Updated for new examples
                "successful_examples": successful_examples,
                "success_rate_pct": (successful_examples / 7) * 100,
                "optimization_effectiveness": "High"
                if successful_examples >= 6
                else "Moderate"
                if successful_examples >= 4
                else "Low",
                "speed_optimization_available": all_results.get(
                    "speed_optimization", {}
                ).get("success", False),
                "speed_improvement_demonstrated": all_results.get(
                    "speed_optimization", {}
                ).get("speed_optimization_effective", False),
            }

            logger.info(
                f"🎉 All examples completed! Success rate: {all_results['summary']['success_rate_pct']:.0f}%"
            )

        except Exception as e:
            logger.error(f"❌ Example execution failed: {e}")
            all_results["error"] = str(e)

        return all_results


async def main():
    """Main function to run optimization examples."""

    # Get OpenRouter API key
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.error("❌ OPENROUTER_API_KEY environment variable not set")
        return

    # Create examples instance
    examples = OptimizationExamples(openrouter_api_key)

    # Run all examples
    results = await examples.run_all_examples()

    # Print summary
    print("\n" + "=" * 80)
    print("LLM OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)

    if "summary" in results:
        summary = results["summary"]
        print(f"Total Examples: {summary['total_examples']}")
        print(f"Successful: {summary['successful_examples']}")
        print(f"Success Rate: {summary['success_rate_pct']:.0f}%")
        print(f"Effectiveness: {summary['optimization_effectiveness']}")

    if "comparison" in results and "comparison" in results["comparison"]:
        comp = results["comparison"]["comparison"]
        if comp.get("time_improvement_pct", 0) > 0:
            print(f"Speed Improvement: {comp['time_improvement_pct']:.1f}%")

    if "speed_optimization" in results and results["speed_optimization"].get("success"):
        speed_results = results["speed_optimization"]
        print(f"Fastest Model: {speed_results.get('fastest_model', 'Unknown')}")

        improvements = speed_results.get("improvements", {})
        if improvements.get("gemini_2_5_flash_vs_baseline_pct", 0) > 0:
            print(
                f"Gemini 2.5 Flash Speed Boost: {improvements['gemini_2_5_flash_vs_baseline_pct']:+.1f}%"
            )
        if improvements.get("gpt_4o_mini_vs_baseline_pct", 0) > 0:
            print(
                f"GPT-4o Mini Speed Boost: {improvements['gpt_4o_mini_vs_baseline_pct']:+.1f}%"
            )

    print("\nDetailed Results:")
    for example_name, result in results.items():
        if example_name not in ["summary", "error"]:
            if isinstance(result, dict):
                success = result.get("success") or result.get("overall_success")
                time_info = (
                    f"{result.get('actual_time', 0):.1f}s"
                    if "actual_time" in result
                    else "N/A"
                )
                print(
                    f"  {example_name.title()}: {'✅ SUCCESS' if success else '❌ FAILED'} ({time_info})"
                )

    print("=" * 80)


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
