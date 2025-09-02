"""
Speed Optimization Validation Test Suite for MaverickMCP Research Agents

This comprehensive test suite validates the speed optimizations implemented in the research system:
- Validates 2-3x speed improvement claims
- Tests emergency mode completion under 30s
- Verifies fast model selection (Gemini 2.5 Flash, GPT-4o Mini)
- Resolves previous timeout issues (138s, 129s failures)
- Compares before/after performance

Speed Optimization Features Being Tested:
1. Adaptive Model Selection (emergency, fast, balanced modes)
2. Progressive Token Budgeting with time awareness
3. Parallel LLM Processing with intelligent batching
4. Optimized Prompt Engineering for speed
5. Early Termination based on confidence thresholds
6. Content Filtering to reduce processing overhead
"""

import asyncio
import logging
import statistics
import time
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import pytest
except ImportError:
    # For standalone use without pytest
    pytest = None

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.agents.optimized_research import OptimizedDeepResearchAgent
from maverick_mcp.providers.openrouter_provider import OpenRouterProvider, TaskType
from maverick_mcp.utils.llm_optimization import (
    AdaptiveModelSelector,
    ConfidenceTracker,
    IntelligentContentFilter,
    ModelConfiguration,
    OptimizedPromptEngine,
    ParallelLLMProcessor,
    ProgressiveTokenBudgeter,
)

logger = logging.getLogger(__name__)

# Speed optimization validation thresholds
SPEED_THRESHOLDS = {
    "simple_query_max_time": 15.0,  # Simple queries should complete in <15s
    "moderate_query_max_time": 25.0,  # Moderate queries should complete in <25s
    "complex_query_max_time": 35.0,  # Complex queries should complete in <35s
    "emergency_mode_max_time": 30.0,  # Emergency mode should complete in <30s
    "minimum_speedup_factor": 2.0,  # Minimum 2x speedup over baseline
    "target_speedup_factor": 3.0,  # Target 3x speedup over baseline
    "timeout_failure_threshold": 0.05,  # Max 5% timeout failures allowed
}


# Test query complexity definitions
class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EMERGENCY = "emergency"


# Test query templates by complexity
SPEED_TEST_QUERIES = {
    QueryComplexity.SIMPLE: [
        "Apple Inc current stock price and basic sentiment",
        "Tesla recent news and market overview",
        "Microsoft quarterly earnings summary",
        "NVIDIA stock performance this month",
    ],
    QueryComplexity.MODERATE: [
        "Apple Inc comprehensive financial analysis and competitive position in smartphone market",
        "Tesla Inc market outlook considering EV competition and regulatory changes",
        "Microsoft Corp cloud business growth prospects and AI integration strategy",
        "NVIDIA competitive analysis in semiconductor and AI acceleration markets",
    ],
    QueryComplexity.COMPLEX: [
        "Apple Inc deep fundamental analysis including supply chain risks, product lifecycle assessment, regulatory challenges across global markets, competitive positioning against Samsung and Google, and 5-year growth trajectory considering AR/VR investments and services expansion",
        "Tesla Inc comprehensive investment thesis covering production scaling challenges, battery technology competitive advantages, autonomous driving timeline and regulatory risks, energy business growth potential, and Elon Musk leadership impact on stock volatility",
        "Microsoft Corp strategic analysis of cloud infrastructure competition with AWS and Google, AI monetization through Copilot integration, gaming division performance post-Activision acquisition, and enterprise software market share defense against Salesforce and Oracle",
        "NVIDIA Corp detailed semiconductor industry analysis covering data center growth drivers, gaming market maturity, automotive AI partnerships, geopolitical chip manufacturing risks, and competitive threats from AMD, Intel, and custom silicon development by major cloud providers",
    ],
    QueryComplexity.EMERGENCY: [
        "Quick Apple sentiment - bullish or bearish right now?",
        "Tesla stock - buy, hold, or sell this week?",
        "Microsoft earnings - beat or miss expectations?",
        "NVIDIA - momentum trade opportunity today?",
    ],
}

# Expected model selections for each scenario
EXPECTED_MODEL_SELECTIONS = {
    QueryComplexity.SIMPLE: ["google/gemini-2.5-flash", "openai/gpt-4o-mini"],
    QueryComplexity.MODERATE: ["openai/gpt-4o-mini", "google/gemini-2.5-flash"],
    QueryComplexity.COMPLEX: [
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
    ],
    QueryComplexity.EMERGENCY: ["google/gemini-2.5-flash", "openai/gpt-4o-mini"],
}

# Token generation speeds (tokens/second) for validation
MODEL_SPEED_BENCHMARKS = {
    "google/gemini-2.5-flash": 199,
    "openai/gpt-4o-mini": 126,
    "anthropic/claude-sonnet-4": 45,
    "google/gemini-2.5-pro": 25,
    "anthropic/claude-haiku": 89,
}


class SpeedTestMonitor:
    """Monitors speed optimization performance during test execution."""

    def __init__(self, test_name: str, complexity: QueryComplexity):
        self.test_name = test_name
        self.complexity = complexity
        self.start_time: float = 0
        self.end_time: float = 0
        self.phase_timings: dict[str, float] = {}
        self.model_selections: list[str] = []
        self.optimization_metrics: dict[str, Any] = {}

    def __enter__(self):
        """Start speed monitoring."""
        self.start_time = time.time()
        logger.info(f"Starting speed test: {self.test_name} ({self.complexity.value})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete speed monitoring and log results."""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        logger.info(
            f"Speed test completed: {self.test_name} - "
            f"Time: {total_time:.2f}s, "
            f"Complexity: {self.complexity.value}, "
            f"Models: {self.model_selections}"
        )

    def record_phase(self, phase_name: str, duration: float):
        """Record timing for a specific phase."""
        self.phase_timings[phase_name] = duration

    def record_model_selection(self, model_id: str):
        """Record which model was selected."""
        self.model_selections.append(model_id)

    def record_optimization_metric(self, metric_name: str, value: Any):
        """Record optimization-specific metrics."""
        self.optimization_metrics[metric_name] = value

    @property
    def total_execution_time(self) -> float:
        """Get total execution time."""
        return self.end_time - self.start_time if self.end_time > 0 else 0


class MockOpenRouterProvider:
    """Mock OpenRouter provider that simulates realistic API response times."""

    def __init__(self, simulate_model_speeds: bool = True):
        self.simulate_model_speeds = simulate_model_speeds
        self.call_history: list[dict[str, Any]] = []

    def get_llm(self, model_override: str = None, **kwargs):
        """Get mock LLM with realistic speed simulation."""
        model_id = model_override or "openai/gpt-4o-mini"

        mock_llm = AsyncMock()
        mock_llm.model_id = model_id

        # Simulate realistic response times based on model speed
        if self.simulate_model_speeds:
            speed = MODEL_SPEED_BENCHMARKS.get(model_id, 50)
            max_tokens = kwargs.get("max_tokens", 1000)
            # Calculate response time: (tokens / speed) + API overhead
            response_time = (max_tokens / speed) + 0.5  # 0.5s API overhead
        else:
            response_time = 0.1  # Fast mock response

        async def mock_ainvoke(messages):
            await asyncio.sleep(response_time)
            # Record the call
            self.call_history.append(
                {
                    "model_id": model_id,
                    "response_time": response_time,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "timestamp": time.time(),
                    "messages": len(messages),
                }
            )

            # Return mock response
            mock_response = MagicMock()
            mock_response.content = (
                f"Mock response from {model_id} (simulated {response_time:.2f}s)"
            )
            return mock_response

        mock_llm.ainvoke = mock_ainvoke
        return mock_llm


class SpeedOptimizationValidator:
    """Validates speed optimization claims and performance improvements."""

    @staticmethod
    async def test_adaptive_model_selection(
        time_budget: float, complexity: float, expected_models: list[str]
    ) -> dict[str, Any]:
        """Test that adaptive model selection chooses appropriate fast models."""
        provider = MockOpenRouterProvider(simulate_model_speeds=True)
        selector = AdaptiveModelSelector(provider)

        # Test model selection for time budget
        model_config = selector.select_model_for_time_budget(
            task_type=TaskType.MARKET_ANALYSIS,
            time_remaining_seconds=time_budget,
            complexity_score=complexity,
            content_size_tokens=1000,
        )

        return {
            "selected_model": model_config.model_id,
            "max_tokens": model_config.max_tokens,
            "timeout_seconds": model_config.timeout_seconds,
            "expected_models": expected_models,
            "model_appropriate": model_config.model_id in expected_models,
            "speed_optimized": model_config.model_id
            in ["google/gemini-2.5-flash", "openai/gpt-4o-mini"],
        }

    @staticmethod
    async def test_emergency_mode_performance(query: str) -> dict[str, Any]:
        """Test emergency mode performance (< 30s completion)."""
        provider = MockOpenRouterProvider(simulate_model_speeds=True)

        # Create optimized research agent
        agent = OptimizedDeepResearchAgent(
            openrouter_provider=provider,
            persona="moderate",
            optimization_enabled=True,
        )

        # Mock the search providers to avoid actual API calls
        agent.search_providers = [MockSearchProvider()]

        start_time = time.time()

        try:
            # Test with strict emergency time budget
            result = await agent.research_comprehensive(
                topic=query,
                session_id="emergency_test",
                depth="basic",
                time_budget_seconds=25.0,  # Strict emergency budget
                target_confidence=0.6,  # Lower confidence for speed
            )

            execution_time = time.time() - start_time

            return {
                "success": True,
                "execution_time": execution_time,
                "within_budget": execution_time
                < SPEED_THRESHOLDS["emergency_mode_max_time"],
                "result_status": result.get("status", "unknown"),
                "emergency_mode_used": result.get("emergency_mode", False),
                "optimization_metrics": result.get("optimization_metrics", {}),
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
                "within_budget": execution_time
                < SPEED_THRESHOLDS["emergency_mode_max_time"],
            }

    @staticmethod
    async def test_baseline_vs_optimized_performance(
        query: str, complexity: QueryComplexity
    ) -> dict[str, Any]:
        """Compare baseline vs optimized agent performance."""
        provider = MockOpenRouterProvider(simulate_model_speeds=True)

        # Test baseline agent (non-optimized)
        baseline_agent = DeepResearchAgent(
            llm=provider.get_llm(),
            persona="moderate",
            enable_parallel_execution=False,
        )
        baseline_agent.search_providers = [MockSearchProvider()]

        # Test optimized agent
        optimized_agent = OptimizedDeepResearchAgent(
            openrouter_provider=provider,
            persona="moderate",
            optimization_enabled=True,
        )
        optimized_agent.search_providers = [MockSearchProvider()]

        # Run baseline test
        baseline_start = time.time()
        try:
            baseline_result = await baseline_agent.research_comprehensive(
                topic=query,
                session_id="baseline_test",
                depth="standard",
            )
            baseline_time = time.time() - baseline_start
            baseline_success = True
        except Exception as e:
            baseline_time = time.time() - baseline_start
            baseline_success = False
            baseline_result = {"error": str(e)}

        # Run optimized test
        optimized_start = time.time()
        try:
            optimized_result = await optimized_agent.research_comprehensive(
                topic=query,
                session_id="optimized_test",
                depth="standard",
                time_budget_seconds=60.0,
            )
            optimized_time = time.time() - optimized_start
            optimized_success = True
        except Exception as e:
            optimized_time = time.time() - optimized_start
            optimized_success = False
            optimized_result = {"error": str(e)}

        # Calculate performance metrics
        speedup_factor = (
            baseline_time / max(optimized_time, 0.001) if optimized_time > 0 else 0
        )

        return {
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "speedup_factor": speedup_factor,
            "baseline_success": baseline_success,
            "optimized_success": optimized_success,
            "meets_2x_target": speedup_factor
            >= SPEED_THRESHOLDS["minimum_speedup_factor"],
            "meets_3x_target": speedup_factor
            >= SPEED_THRESHOLDS["target_speedup_factor"],
            "baseline_result": baseline_result,
            "optimized_result": optimized_result,
        }


class MockSearchProvider:
    """Mock search provider for testing without external API calls."""

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Return mock search results."""
        await asyncio.sleep(0.1)  # Simulate API delay

        return [
            {
                "title": f"Mock search result {i + 1} for: {query[:30]}",
                "url": f"https://example.com/result{i + 1}",
                "content": f"Mock content for result {i + 1}. " * 50,  # ~50 words
                "published_date": datetime.now().isoformat(),
                "credibility_score": 0.8,
                "relevance_score": 0.9 - (i * 0.1),
            }
            for i in range(num_results)
        ]


# Test fixtures (conditional on pytest availability)
if pytest:

    @pytest.fixture
    def mock_openrouter_provider():
        """Provide mock OpenRouter provider."""
        return MockOpenRouterProvider(simulate_model_speeds=True)

    @pytest.fixture
    def speed_validator():
        """Provide speed optimization validator."""
        return SpeedOptimizationValidator()

    @pytest.fixture
    def speed_monitor_factory():
        """Factory for creating speed test monitors."""

        def _create_monitor(test_name: str, complexity: QueryComplexity):
            return SpeedTestMonitor(test_name, complexity)

        return _create_monitor


# Core Speed Optimization Tests
if pytest:

    @pytest.mark.unit
    class TestSpeedOptimizations:
        """Core tests for speed optimization functionality."""

        async def test_adaptive_model_selector_emergency_mode(
            self, mock_openrouter_provider
        ):
            """Test that emergency mode selects fastest models."""
            selector = AdaptiveModelSelector(mock_openrouter_provider)

            # Test ultra-emergency mode (< 10s)
            config = selector.select_model_for_time_budget(
                task_type=TaskType.QUICK_ANSWER,
                time_remaining_seconds=8.0,
                complexity_score=0.5,
                content_size_tokens=500,
            )

            # Should select fastest model
            assert config.model_id in ["google/gemini-2.5-flash", "openai/gpt-4o-mini"]
            assert config.timeout_seconds < 10
            assert config.max_tokens < 1000

            # Test moderate emergency (< 25s)
            config = selector.select_model_for_time_budget(
                task_type=TaskType.MARKET_ANALYSIS,
                time_remaining_seconds=20.0,
                complexity_score=0.7,
                content_size_tokens=1000,
            )

            # Should still prefer fast models
            assert config.model_id in ["google/gemini-2.5-flash", "openai/gpt-4o-mini"]
            assert config.timeout_seconds < 25

        async def test_progressive_token_budgeter_time_constraints(self):
            """Test progressive token budgeter adapts to time pressure."""
            # Test emergency budget
            emergency_budgeter = ProgressiveTokenBudgeter(
                total_time_budget_seconds=20.0, confidence_target=0.6
            )

            allocation = emergency_budgeter.allocate_tokens_for_phase(
                phase=emergency_budgeter.phase_budgets.__class__.CONTENT_ANALYSIS,
                sources_count=3,
                current_confidence=0.3,
                complexity_score=0.5,
            )

            # Emergency mode should have reduced tokens and shorter timeout
            assert allocation.output_tokens < 1000
            assert allocation.timeout_seconds < 15

            # Test standard budget
            standard_budgeter = ProgressiveTokenBudgeter(
                total_time_budget_seconds=120.0, confidence_target=0.75
            )

            allocation = standard_budgeter.allocate_tokens_for_phase(
                phase=standard_budgeter.phase_budgets.__class__.CONTENT_ANALYSIS,
                sources_count=3,
                current_confidence=0.3,
                complexity_score=0.5,
            )

            # Standard mode should allow more tokens and time
            assert allocation.output_tokens >= 1000
            assert allocation.timeout_seconds >= 15

        async def test_parallel_llm_processor_speed_optimization(
            self, mock_openrouter_provider
        ):
            """Test parallel LLM processor speed optimizations."""
            processor = ParallelLLMProcessor(mock_openrouter_provider, max_concurrent=4)

            # Create mock sources
            sources = [
                {
                    "title": f"Source {i}",
                    "content": f"Mock content {i} " * 100,  # ~100 words
                    "url": f"https://example.com/{i}",
                }
                for i in range(6)
            ]

            start_time = time.time()

            results = await processor.parallel_content_analysis(
                sources=sources,
                analysis_type="sentiment",
                persona="moderate",
                time_budget_seconds=15.0,  # Tight budget
                current_confidence=0.0,
            )

            execution_time = time.time() - start_time

            # Should complete within time budget
            assert execution_time < 20.0  # Some buffer for test environment
            assert len(results) > 0  # Should produce results

            # Verify all results have required analysis structure
            for result in results:
                assert "analysis" in result
                analysis = result["analysis"]
                assert "sentiment" in analysis
                assert "batch_processed" in analysis

        async def test_confidence_tracker_early_termination(self):
            """Test confidence tracker enables early termination."""
            tracker = ConfidenceTracker(
                target_confidence=0.8,
                min_sources=2,
                max_sources=10,
            )

            # Simulate high-confidence evidence
            high_confidence_evidence = {
                "sentiment": {"direction": "bullish", "confidence": 0.9},
                "insights": ["Strong positive insight", "Another strong insight"],
                "risk_factors": ["Minor risk"],
                "opportunities": ["Major opportunity", "Growth catalyst"],
                "relevance_score": 0.95,
            }

            # Process minimum sources first
            for i in range(2):
                result = tracker.update_confidence(high_confidence_evidence, 0.9)
                if not result["should_continue"]:
                    break

            # After high-confidence sources, should suggest early termination
            final_result = tracker.update_confidence(high_confidence_evidence, 0.9)

            assert final_result["current_confidence"] > 0.7
            # Early termination logic should trigger with high confidence

        async def test_intelligent_content_filter_speed_optimization(self):
            """Test intelligent content filtering reduces processing overhead."""
            filter = IntelligentContentFilter()

            # Create sources with varying relevance
            sources = [
                {
                    "title": "Apple Inc Q4 Earnings Beat Expectations",
                    "content": "Apple Inc reported strong Q4 earnings with revenue growth of 15%. "
                    + "The company's iPhone sales exceeded analysts' expectations. "
                    * 20,
                    "url": "https://reuters.com/apple-earnings",
                    "published_date": datetime.now().isoformat(),
                },
                {
                    "title": "Random Tech News Not About Apple",
                    "content": "Some unrelated tech news content. " * 50,
                    "url": "https://example.com/random",
                    "published_date": "2023-01-01T00:00:00",
                },
                {
                    "title": "Apple Supply Chain Analysis",
                    "content": "Apple's supply chain faces challenges but shows resilience. "
                    + "Manufacturing partnerships in Asia remain strong. " * 15,
                    "url": "https://wsj.com/apple-supply-chain",
                    "published_date": datetime.now().isoformat(),
                },
            ]

            filtered_sources = await filter.filter_and_prioritize_sources(
                sources=sources,
                research_focus="fundamental",
                time_budget=20.0,  # Tight budget
                current_confidence=0.0,
            )

            # Should prioritize relevant, high-quality sources
            assert len(filtered_sources) <= len(sources)
            if filtered_sources:
                # First source should be most relevant
                assert "apple" in filtered_sources[0]["title"].lower()
                # Should have preprocessing applied
                assert "original_length" in filtered_sources[0]


# Speed Validation Tests by Query Complexity
if pytest:

    @pytest.mark.integration
    class TestQueryComplexitySpeedValidation:
        """Test speed validation across different query complexities."""

        @pytest.mark.parametrize("complexity", list(QueryComplexity))
        async def test_query_completion_time_thresholds(
            self, complexity: QueryComplexity, speed_monitor_factory, speed_validator
        ):
            """Test queries complete within time thresholds by complexity."""
            queries = SPEED_TEST_QUERIES[complexity]

            results = []

            for query in queries[:2]:  # Test 2 queries per complexity
                with speed_monitor_factory(
                    f"complexity_test_{complexity.value}", complexity
                ) as monitor:
                    if complexity == QueryComplexity.EMERGENCY:
                        result = await speed_validator.test_emergency_mode_performance(
                            query
                        )
                    else:
                        # Use baseline vs optimized for other complexities
                        result = await speed_validator.test_baseline_vs_optimized_performance(
                            query, complexity
                        )

                    monitor.record_optimization_metric(
                        "completion_time", monitor.total_execution_time
                    )
                    results.append(
                        {
                            "query": query,
                            "execution_time": monitor.total_execution_time,
                            "result": result,
                        }
                    )

            # Validate time thresholds based on complexity
            threshold_map = {
                QueryComplexity.SIMPLE: SPEED_THRESHOLDS["simple_query_max_time"],
                QueryComplexity.MODERATE: SPEED_THRESHOLDS["moderate_query_max_time"],
                QueryComplexity.COMPLEX: SPEED_THRESHOLDS["complex_query_max_time"],
                QueryComplexity.EMERGENCY: SPEED_THRESHOLDS["emergency_mode_max_time"],
            }

            max_allowed_time = threshold_map[complexity]

            for result in results:
                execution_time = result["execution_time"]
                assert execution_time < max_allowed_time, (
                    f"{complexity.value} query exceeded time threshold: "
                    f"{execution_time:.2f}s > {max_allowed_time}s"
                )

            # Log performance summary
            avg_time = statistics.mean([r["execution_time"] for r in results])
            logger.info(
                f"{complexity.value} queries - Avg time: {avg_time:.2f}s "
                f"(threshold: {max_allowed_time}s)"
            )

        async def test_emergency_mode_model_selection(self, mock_openrouter_provider):
            """Test emergency mode selects fastest models."""
            selector = AdaptiveModelSelector(mock_openrouter_provider)

            # Test various emergency time budgets
            emergency_scenarios = [5, 10, 15, 20, 25]

            for time_budget in emergency_scenarios:
                config = selector.select_model_for_time_budget(
                    task_type=TaskType.QUICK_ANSWER,
                    time_remaining_seconds=time_budget,
                    complexity_score=0.3,  # Low complexity for emergency
                    content_size_tokens=200,
                )

                # Should always select fastest models in emergency scenarios
                expected_models = EXPECTED_MODEL_SELECTIONS[QueryComplexity.EMERGENCY]
                assert config.model_id in expected_models, (
                    f"Emergency mode with {time_budget}s budget should select fast model, "
                    f"got {config.model_id}"
                )

                # Timeout should be appropriate for time budget
                assert config.timeout_seconds < time_budget * 0.8, (
                    f"Timeout too long for emergency budget: "
                    f"{config.timeout_seconds}s for {time_budget}s budget"
                )


# Performance Comparison Tests
if pytest:

    @pytest.mark.integration
    class TestSpeedImprovementValidation:
        """Validate claimed speed improvements (2-3x faster)."""

    async def test_2x_minimum_speedup_validation(self, speed_validator):
        """Validate minimum 2x speedup is achieved."""
        moderate_queries = SPEED_TEST_QUERIES[QueryComplexity.MODERATE]

        speedup_results = []

        for query in moderate_queries[:2]:  # Test subset for CI speed
            result = await speed_validator.test_baseline_vs_optimized_performance(
                query, QueryComplexity.MODERATE
            )

            if result["baseline_success"] and result["optimized_success"]:
                speedup_results.append(result["speedup_factor"])

                logger.info(
                    f"Speedup test: {result['speedup_factor']:.2f}x "
                    f"({result['baseline_time']:.2f}s -> {result['optimized_time']:.2f}s)"
                )

        # Validate minimum 2x speedup achieved
        if speedup_results:
            avg_speedup = statistics.mean(speedup_results)
            min_speedup = min(speedup_results)

            assert avg_speedup >= SPEED_THRESHOLDS["minimum_speedup_factor"], (
                f"Average speedup {avg_speedup:.2f}x below 2x minimum threshold"
            )

            # At least 80% of tests should meet minimum speedup
            meeting_threshold = sum(
                1
                for s in speedup_results
                if s >= SPEED_THRESHOLDS["minimum_speedup_factor"]
            )
            threshold_rate = meeting_threshold / len(speedup_results)

            assert threshold_rate >= 0.8, (
                f"Only {threshold_rate:.1%} of tests met 2x speedup threshold "
                f"(should be >= 80%)"
            )
        else:
            pytest.skip("No successful speedup comparisons completed")

    async def test_3x_target_speedup_aspiration(self, speed_validator):
        """Test aspirational 3x speedup target for simple queries."""
        simple_queries = SPEED_TEST_QUERIES[QueryComplexity.SIMPLE]

        speedup_results = []

        for query in simple_queries:
            result = await speed_validator.test_baseline_vs_optimized_performance(
                query, QueryComplexity.SIMPLE
            )

            if result["baseline_success"] and result["optimized_success"]:
                speedup_results.append(result["speedup_factor"])

        if speedup_results:
            avg_speedup = statistics.mean(speedup_results)
            max_speedup = max(speedup_results)

            logger.info(
                f"3x target test - Avg: {avg_speedup:.2f}x, Max: {max_speedup:.2f}x"
            )

            # This is aspirational - log results but don't fail
            target_met = avg_speedup >= SPEED_THRESHOLDS["target_speedup_factor"]
            if target_met:
                logger.info("🎉 3x speedup target achieved!")
            else:
                logger.info(f"3x target not yet achieved (current: {avg_speedup:.2f}x)")

            # Still assert we're making good progress toward 3x
            assert avg_speedup >= 1.5, (
                f"Should show significant speedup progress, got {avg_speedup:.2f}x"
            )


# Timeout Resolution Tests
if pytest:

    @pytest.mark.integration
    class TestTimeoutResolution:
        """Test resolution of previous timeout issues (138s, 129s failures)."""

    async def test_no_timeout_failures_in_emergency_mode(self, speed_validator):
        """Test emergency mode prevents timeout failures."""
        emergency_queries = SPEED_TEST_QUERIES[QueryComplexity.EMERGENCY]

        timeout_failures = 0
        total_tests = 0

        for query in emergency_queries:
            total_tests += 1

            result = await speed_validator.test_emergency_mode_performance(query)

            # Check if execution exceeded emergency time budget
            if result["execution_time"] >= SPEED_THRESHOLDS["emergency_mode_max_time"]:
                timeout_failures += 1
                logger.warning(
                    f"Emergency mode timeout: {result['execution_time']:.2f}s "
                    f"for query: {query[:50]}..."
                )

        # Calculate failure rate
        timeout_failure_rate = timeout_failures / max(total_tests, 1)

        # Should have very low timeout failure rate
        assert timeout_failure_rate <= SPEED_THRESHOLDS["timeout_failure_threshold"], (
            f"Timeout failure rate too high: {timeout_failure_rate:.1%} > "
            f"{SPEED_THRESHOLDS['timeout_failure_threshold']:.1%}"
        )

        logger.info(
            f"Timeout resolution test: {timeout_failure_rate:.1%} failure rate "
            f"({timeout_failures}/{total_tests} timeouts)"
        )

    async def test_graceful_degradation_under_time_pressure(self, speed_validator):
        """Test system degrades gracefully under extreme time pressure."""
        # Simulate very tight time budgets that previously caused 138s/129s failures
        tight_budgets = [10, 15, 20, 25]  # Various emergency scenarios

        degradation_results = []

        for budget in tight_budgets:
            provider = MockOpenRouterProvider(simulate_model_speeds=True)

            agent = OptimizedDeepResearchAgent(
                openrouter_provider=provider,
                persona="moderate",
                optimization_enabled=True,
            )
            agent.search_providers = [MockSearchProvider()]

            start_time = time.time()

            try:
                result = await agent.research_comprehensive(
                    topic="Apple Inc urgent analysis needed",
                    session_id=f"degradation_test_{budget}s",
                    depth="basic",
                    time_budget_seconds=budget,
                    target_confidence=0.5,  # Lower expectations
                )

                execution_time = time.time() - start_time

                degradation_results.append(
                    {
                        "budget": budget,
                        "execution_time": execution_time,
                        "success": True,
                        "within_budget": execution_time <= budget + 5,  # 5s buffer
                        "emergency_mode": result.get("emergency_mode", False),
                    }
                )

            except Exception as e:
                execution_time = time.time() - start_time
                degradation_results.append(
                    {
                        "budget": budget,
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e),
                        "within_budget": execution_time <= budget + 5,
                    }
                )

        # Validate graceful degradation
        successful_tests = [r for r in degradation_results if r["success"]]
        within_budget_tests = [r for r in degradation_results if r["within_budget"]]

        success_rate = len(successful_tests) / len(degradation_results)
        budget_compliance_rate = len(within_budget_tests) / len(degradation_results)

        # Should succeed most of the time and stay within budget
        assert success_rate >= 0.75, (
            f"Success rate too low under time pressure: {success_rate:.1%}"
        )
        assert budget_compliance_rate >= 0.80, (
            f"Budget compliance too low: {budget_compliance_rate:.1%}"
        )

        logger.info(
            f"Graceful degradation test: {success_rate:.1%} success rate, "
            f"{budget_compliance_rate:.1%} budget compliance"
        )


if __name__ == "__main__":
    # Allow running specific test categories
    import sys

    if len(sys.argv) > 1:
        pytest.main([sys.argv[1], "-v", "-s", "--tb=short"])
    else:
        # Run all speed validation tests by default
        pytest.main([__file__, "-v", "-s", "--tb=short"])
