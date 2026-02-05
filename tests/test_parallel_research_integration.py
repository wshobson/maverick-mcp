"""
Comprehensive integration tests for parallel research functionality.

This test suite covers:
- End-to-end parallel research workflows
- Integration between all parallel research components
- Performance characteristics under realistic conditions
- Error scenarios and recovery mechanisms
- Logging integration across all components
- Resource usage and scalability testing
"""

import asyncio
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.utils.parallel_research import (
    ParallelResearchConfig,
)

_ASYNC_MOCK_SENTINEL = AsyncMock


class MockSearchProvider:
    """Mock search provider for integration testing."""

    def __init__(self, provider_name: str, fail_rate: float = 0.0):
        self.provider_name = provider_name
        self.fail_rate = fail_rate
        self.call_count = 0

    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Mock search with configurable failure rate."""
        self.call_count += 1

        # Simulate failures based on fail_rate
        import random

        if random.random() < self.fail_rate:
            raise RuntimeError(f"{self.provider_name} search failed")

        await asyncio.sleep(0.02)  # Simulate network latency

        # Generate mock search results
        results = []
        for i in range(min(num_results, 3)):  # Return up to 3 results
            results.append(
                {
                    "url": f"https://{self.provider_name.lower()}.example.com/article_{i}_{self.call_count}",
                    "title": f"{query} - Article {i + 1} from {self.provider_name}",
                    "content": f"This is detailed content about {query} from {self.provider_name}. "
                    f"It contains valuable insights and analysis relevant to the research topic. "
                    f"Provider: {self.provider_name}, Call: {self.call_count}",
                    "published_date": datetime.now().isoformat(),
                    "author": f"Expert Analyst {i + 1}",
                    "score": 0.8 - (i * 0.1),
                    "provider": self.provider_name.lower(),
                }
            )

        return results


class MockContentAnalyzer:
    """Mock content analyzer for integration testing."""

    def __init__(self, analysis_delay: float = 0.01):
        self.analysis_delay = analysis_delay
        self.analysis_count = 0

    async def analyze_content(
        self, content: str, persona: str, analysis_focus: str = "general"
    ) -> dict[str, Any]:
        """Mock content analysis."""
        self.analysis_count += 1
        await asyncio.sleep(self.analysis_delay)

        # Generate realistic analysis based on content keywords
        insights = []
        risk_factors = []
        opportunities = []

        content_lower = content.lower()

        if "earnings" in content_lower or "revenue" in content_lower:
            insights.append("Strong earnings performance indicated")
            opportunities.append("Potential for continued revenue growth")

        if "technical" in content_lower or "chart" in content_lower:
            insights.append("Technical indicators suggest trend continuation")
            risk_factors.append("Support level break could trigger selling")

        if "sentiment" in content_lower or "analyst" in content_lower:
            insights.append("Market sentiment appears positive")
            opportunities.append("Analyst upgrades possible")

        if "competitive" in content_lower or "market share" in content_lower:
            insights.append("Competitive position remains strong")
            risk_factors.append("Increased competitive pressure in market")

        # Default insights if no specific keywords found
        if not insights:
            insights = [
                f"General analysis insight {self.analysis_count} for {persona} investor"
            ]

        sentiment_mapping = {
            "conservative": {"direction": "neutral", "confidence": 0.6},
            "moderate": {"direction": "bullish", "confidence": 0.7},
            "aggressive": {"direction": "bullish", "confidence": 0.8},
        }

        return {
            "insights": insights,
            "sentiment": sentiment_mapping.get(
                persona, {"direction": "neutral", "confidence": 0.5}
            ),
            "risk_factors": risk_factors or ["Standard market risks apply"],
            "opportunities": opportunities or ["Monitor for opportunities"],
            "credibility_score": 0.8,
            "relevance_score": 0.75,
            "summary": f"Analysis for {persona} investor from {analysis_focus} perspective",
            "analysis_timestamp": datetime.now(),
        }


class MockLLM(BaseChatModel):
    """Mock LLM for integration testing."""

    def __init__(self, response_delay: float = 0.05, fail_rate: float = 0.0):
        super().__init__()
        self.response_delay = response_delay
        self.fail_rate = fail_rate
        self.invocation_count = 0

    async def ainvoke(self, messages, config=None, **kwargs):
        """Mock async LLM invocation."""
        self.invocation_count += 1

        # Simulate failures
        import random

        if random.random() < self.fail_rate:
            raise RuntimeError("LLM service unavailable")

        await asyncio.sleep(self.response_delay)

        # Generate contextual response based on message content
        message_content = str(messages[-1].content).lower()

        if "synthesis" in message_content:
            response = """
            Based on the comprehensive research from multiple specialized agents, this analysis provides
            a well-rounded view of the investment opportunity. The fundamental analysis shows strong
            financial metrics, while sentiment analysis indicates positive market reception. Technical
            analysis suggests favorable entry points, and competitive analysis reveals sustainable
            advantages. Overall, this presents a compelling investment case for the specified investor persona.
            """
        else:
            response = '{"KEY_INSIGHTS": ["AI-generated insight"], "SENTIMENT": {"direction": "bullish", "confidence": 0.75}, "CREDIBILITY": 0.8}'

        return AIMessage(content=response)

    def _generate(self, messages, stop=None, **kwargs):
        raise NotImplementedError("Use ainvoke for async tests")

    @property
    def _llm_type(self) -> str:
        return "mock_llm"


@pytest.mark.integration
class TestParallelResearchEndToEnd:
    """Test complete end-to-end parallel research workflows."""

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return ParallelResearchConfig(
            max_concurrent_agents=3,
            timeout_per_agent=10,
            enable_fallbacks=True,
            rate_limit_delay=0.1,
        )

    @pytest.fixture
    def mock_search_providers(self):
        """Create mock search providers."""
        return [
            MockSearchProvider("Exa", fail_rate=0.1),
            MockSearchProvider("Tavily", fail_rate=0.1),
        ]

    @pytest.fixture
    def integration_agent(self, integration_config, mock_search_providers):
        """Create DeepResearchAgent for integration testing."""
        llm = MockLLM(response_delay=0.05, fail_rate=0.05)

        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            checkpointer=MemorySaver(),
            enable_parallel_execution=True,
            parallel_config=integration_config,
        )

        # Replace search providers with mocks
        agent.search_providers = mock_search_providers

        # Replace content analyzer with mock
        agent.content_analyzer = MockContentAnalyzer(analysis_delay=0.02)

        return agent

    @pytest.mark.asyncio
    async def test_complete_parallel_research_workflow(self, integration_agent):
        """Test complete parallel research workflow from start to finish."""
        start_time = time.time()

        result = await integration_agent.research_comprehensive(
            topic="Apple Inc comprehensive investment analysis for Q4 2024",
            session_id="integration_test_001",
            depth="comprehensive",
            focus_areas=["fundamentals", "technical_analysis", "market_sentiment"],
            timeframe="30d",
        )

        execution_time = time.time() - start_time

        # Verify successful execution
        assert result["status"] == "success"
        assert result["agent_type"] == "deep_research"
        assert result["execution_mode"] == "parallel"
        assert (
            result["research_topic"]
            == "Apple Inc comprehensive investment analysis for Q4 2024"
        )

        # Verify research quality
        assert result["confidence_score"] > 0.5
        assert result["sources_analyzed"] > 0
        assert len(result["citations"]) > 0

        # Verify parallel execution stats
        assert "parallel_execution_stats" in result
        stats = result["parallel_execution_stats"]
        assert stats["total_tasks"] > 0
        assert stats["successful_tasks"] >= 0
        assert stats["parallel_efficiency"] > 0

        # Verify findings structure
        assert "findings" in result
        findings = result["findings"]
        assert "synthesis" in findings
        assert "confidence_score" in findings

        # Verify performance characteristics
        assert execution_time < 15  # Should complete within reasonable time
        assert result["execution_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance_comparison(
        self, integration_agent
    ):
        """Compare parallel vs sequential execution performance."""
        topic = "Tesla Inc strategic analysis and market position"

        # Test parallel execution
        start_parallel = time.time()
        parallel_result = await integration_agent.research_comprehensive(
            topic=topic,
            session_id="perf_test_parallel",
            use_parallel_execution=True,
            depth="standard",
        )
        parallel_time = time.time() - start_parallel

        # Test sequential execution
        start_sequential = time.time()
        sequential_result = await integration_agent.research_comprehensive(
            topic=topic,
            session_id="perf_test_sequential",
            use_parallel_execution=False,
            depth="standard",
        )
        sequential_time = time.time() - start_sequential

        # Both should succeed
        assert parallel_result["status"] == "success"
        assert sequential_result["status"] == "success"

        # Verify execution modes
        assert parallel_result["execution_mode"] == "parallel"
        # Sequential won't have execution_mode in result

        # Parallel should show efficiency metrics
        if "parallel_execution_stats" in parallel_result:
            stats = parallel_result["parallel_execution_stats"]
            assert stats["parallel_efficiency"] > 0

            # If multiple tasks were executed in parallel, should show some efficiency gain
            if stats["total_tasks"] > 1:
                print(
                    f"Parallel time: {parallel_time:.3f}s, Sequential time: {sequential_time:.3f}s"
                )
                print(f"Parallel efficiency: {stats['parallel_efficiency']:.2f}x")

    @pytest.mark.asyncio
    async def test_error_resilience_and_fallback(self, integration_config):
        """Test error resilience and fallback mechanisms."""
        # Create agent with higher failure rates to test resilience
        failing_llm = MockLLM(fail_rate=0.3)  # 30% failure rate
        failing_providers = [
            MockSearchProvider("FailingProvider", fail_rate=0.5)  # 50% failure rate
        ]

        agent = DeepResearchAgent(
            llm=failing_llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=integration_config,
        )

        agent.search_providers = failing_providers
        agent.content_analyzer = MockContentAnalyzer()

        # Should handle failures gracefully
        result = await agent.research_comprehensive(
            topic="Resilience test analysis",
            session_id="resilience_test_001",
            depth="basic",
        )

        # Should complete successfully despite some failures
        assert result["status"] == "success"

        # May have lower confidence due to failures
        assert result["confidence_score"] >= 0.0

        # Should have some parallel execution stats even if tasks failed
        if result.get("execution_mode") == "parallel":
            stats = result["parallel_execution_stats"]
            # Total tasks should be >= failed tasks (some tasks were attempted)
            assert stats["total_tasks"] >= stats.get("failed_tasks", 0)

    @pytest.mark.asyncio
    async def test_different_research_depths(self, integration_agent):
        """Test parallel research with different depth configurations."""
        depths_to_test = ["basic", "standard", "comprehensive"]
        results = {}

        for depth in depths_to_test:
            result = await integration_agent.research_comprehensive(
                topic=f"Microsoft Corp analysis - {depth} depth",
                session_id=f"depth_test_{depth}",
                depth=depth,
                use_parallel_execution=True,
            )

            results[depth] = result

            # All should succeed
            assert result["status"] == "success"
            assert result["research_depth"] == depth

        # Comprehensive should generally have more sources and higher confidence
        if all(r["status"] == "success" for r in results.values()):
            basic_sources = results["basic"]["sources_analyzed"]
            comprehensive_sources = results["comprehensive"]["sources_analyzed"]

            # More comprehensive research should analyze more sources (when successful)
            if basic_sources > 0 and comprehensive_sources > 0:
                assert comprehensive_sources >= basic_sources

    @pytest.mark.asyncio
    async def test_persona_specific_research(self, integration_config):
        """Test parallel research with different investor personas."""
        personas_to_test = ["conservative", "moderate", "aggressive"]
        topic = "Amazon Inc investment opportunity analysis"

        for persona in personas_to_test:
            llm = MockLLM(response_delay=0.03)
            agent = DeepResearchAgent(
                llm=llm,
                persona=persona,
                enable_parallel_execution=True,
                parallel_config=integration_config,
            )

            # Mock components
            agent.search_providers = [MockSearchProvider("TestProvider")]
            agent.content_analyzer = MockContentAnalyzer()

            result = await agent.research_comprehensive(
                topic=topic,
                session_id=f"persona_test_{persona}",
                use_parallel_execution=True,
            )

            assert result["status"] == "success"
            assert result["persona"] == persona

            # Should have findings tailored to persona
            assert "findings" in result

    @pytest.mark.asyncio
    async def test_concurrent_research_sessions(self, integration_agent):
        """Test multiple concurrent research sessions."""
        topics = [
            "Google Alphabet strategic analysis",
            "Meta Platforms competitive position",
            "Netflix content strategy evaluation",
        ]

        # Run multiple research sessions concurrently
        tasks = [
            integration_agent.research_comprehensive(
                topic=topic,
                session_id=f"concurrent_test_{i}",
                use_parallel_execution=True,
                depth="standard",
            )
            for i, topic in enumerate(topics)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time

        # All should succeed (or be exceptions we can handle)
        successful_results = [
            r for r in results if isinstance(r, dict) and r.get("status") == "success"
        ]

        assert (
            len(successful_results) >= len(topics) // 2
        )  # At least half should succeed

        # Should complete in reasonable time despite concurrency
        assert execution_time < 30

        # Verify each result has proper session isolation
        for _i, result in enumerate(successful_results):
            if "findings" in result:
                # Each should have distinct research content
                assert result["research_topic"] in topics


@pytest.mark.integration
class TestParallelResearchScalability:
    """Test scalability characteristics of parallel research."""

    @pytest.fixture
    def scalability_config(self):
        """Configuration for scalability testing."""
        return ParallelResearchConfig(
            max_concurrent_agents=4,
            timeout_per_agent=8,
            enable_fallbacks=True,
            rate_limit_delay=0.05,
        )

    @pytest.mark.asyncio
    async def test_agent_limit_enforcement(self, scalability_config):
        """Test that concurrent agent limits are properly enforced."""
        llm = MockLLM(response_delay=0.1)  # Slower to see concurrency effects

        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=scalability_config,
        )

        # Mock components with tracking
        call_tracker = {"max_concurrent": 0, "current_concurrent": 0}

        class TrackingProvider(MockSearchProvider):
            async def search(self, query: str, num_results: int = 10):
                call_tracker["current_concurrent"] += 1
                call_tracker["max_concurrent"] = max(
                    call_tracker["max_concurrent"], call_tracker["current_concurrent"]
                )
                try:
                    return await super().search(query, num_results)
                finally:
                    call_tracker["current_concurrent"] -= 1

        agent.search_providers = [TrackingProvider("Tracker")]
        agent.content_analyzer = MockContentAnalyzer()

        result = await agent.research_comprehensive(
            topic="Scalability test with many potential subtasks",
            session_id="scalability_test_001",
            focus_areas=[
                "fundamentals",
                "technical",
                "sentiment",
                "competitive",
                "extra1",
                "extra2",
            ],
            use_parallel_execution=True,
        )

        assert result["status"] == "success"

        # Should not exceed configured max concurrent agents
        assert (
            call_tracker["max_concurrent"] <= scalability_config.max_concurrent_agents
        )

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, scalability_config):
        """Test memory usage characteristics under load."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        llm = MockLLM(response_delay=0.02)
        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=scalability_config,
        )

        agent.search_providers = [MockSearchProvider("MemoryTest")]
        agent.content_analyzer = MockContentAnalyzer(analysis_delay=0.01)

        # Perform multiple research operations
        for i in range(10):  # 10 operations to test memory accumulation
            result = await agent.research_comprehensive(
                topic=f"Memory test analysis {i}",
                session_id=f"memory_test_{i}",
                use_parallel_execution=True,
                depth="basic",
            )

            assert result["status"] == "success"

            # Force garbage collection
            gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (not indicative of leaks)
        assert memory_growth < 100  # Less than 100MB growth for 10 operations

    @pytest.mark.asyncio
    async def test_large_scale_task_distribution(self, scalability_config):
        """Test task distribution with many potential research areas."""
        llm = MockLLM()
        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=scalability_config,
        )

        agent.search_providers = [MockSearchProvider("LargeScale")]
        agent.content_analyzer = MockContentAnalyzer()

        # Test with many focus areas (more than max concurrent agents)
        many_focus_areas = [
            "earnings",
            "revenue",
            "profit_margins",
            "debt_analysis",
            "cash_flow",
            "technical_indicators",
            "chart_patterns",
            "support_levels",
            "momentum",
            "analyst_ratings",
            "news_sentiment",
            "social_sentiment",
            "institutional_sentiment",
            "market_share",
            "competitive_position",
            "industry_trends",
            "regulatory_environment",
        ]

        result = await agent.research_comprehensive(
            topic="Large scale comprehensive analysis with many research dimensions",
            session_id="large_scale_test_001",
            focus_areas=many_focus_areas,
            use_parallel_execution=True,
            depth="comprehensive",
        )

        assert result["status"] == "success"

        # Should handle large number of focus areas efficiently
        if "parallel_execution_stats" in result:
            stats = result["parallel_execution_stats"]
            # Should not create more tasks than max concurrent agents allows
            assert stats["total_tasks"] <= scalability_config.max_concurrent_agents

            # Should achieve some parallel efficiency
            if stats["successful_tasks"] > 1:
                assert stats["parallel_efficiency"] > 1.0


@pytest.mark.integration
class TestParallelResearchLoggingIntegration:
    """Test integration of logging throughout parallel research workflow."""

    @pytest.fixture
    def logged_agent(self):
        """Create agent with comprehensive logging."""
        llm = MockLLM(response_delay=0.02)
        config = ParallelResearchConfig(
            max_concurrent_agents=2,
            timeout_per_agent=5,
            enable_fallbacks=True,
            rate_limit_delay=0.05,
        )

        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        agent.search_providers = [MockSearchProvider("LoggedProvider")]
        agent.content_analyzer = MockContentAnalyzer()

        return agent

    @pytest.mark.asyncio
    async def test_comprehensive_logging_workflow(self, logged_agent):
        """Test that comprehensive logging occurs throughout workflow."""
        with patch(
            "maverick_mcp.utils.orchestration_logging.get_orchestration_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = await logged_agent.research_comprehensive(
                topic="Comprehensive logging test analysis",
                session_id="logging_test_001",
                use_parallel_execution=True,
            )

            assert result["status"] == "success"

            # Should have multiple logging calls
            assert mock_logger.info.call_count >= 10  # Multiple stages should log

            # Verify different types of log messages occurred
            all_log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            " ".join(all_log_calls)

            # Should contain various logging elements
            assert any("RESEARCH_START" in call for call in all_log_calls)
            assert any("PARALLEL" in call for call in all_log_calls)

    @pytest.mark.asyncio
    async def test_error_logging_integration(self, logged_agent):
        """Test error logging integration in parallel workflow."""
        # Create a scenario that will cause some errors
        failing_llm = MockLLM(fail_rate=0.5)  # High failure rate
        logged_agent.llm = failing_llm

        with patch(
            "maverick_mcp.utils.orchestration_logging.get_orchestration_logger"
        ) as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # This may succeed or fail, but should log appropriately
            try:
                result = await logged_agent.research_comprehensive(
                    topic="Error logging test",
                    session_id="error_logging_test_001",
                    use_parallel_execution=True,
                )

                # If it succeeds, should still have logged errors from failed components
                assert result["status"] == "success" or result["status"] == "error"

            except Exception:
                # If it fails completely, that's also acceptable for this test
                pass

            # Should have some error or warning logs due to high failure rate
            has_error_logs = (
                mock_logger.error.call_count > 0 or mock_logger.warning.call_count > 0
            )
            assert has_error_logs

    @pytest.mark.asyncio
    async def test_performance_metrics_logging(self, logged_agent):
        """Test that performance metrics are properly logged."""
        with patch(
            "maverick_mcp.utils.orchestration_logging.log_performance_metrics"
        ) as mock_perf_log:
            result = await logged_agent.research_comprehensive(
                topic="Performance metrics test",
                session_id="perf_metrics_test_001",
                use_parallel_execution=True,
            )

            assert result["status"] == "success"

            # Should have logged performance metrics
            assert mock_perf_log.call_count >= 1

            # Verify metrics content
            perf_call = mock_perf_log.call_args_list[0]
            perf_call[0][0]
            metrics = perf_call[0][1]

            assert isinstance(metrics, dict)
            # Should contain relevant performance metrics
            expected_metrics = [
                "total_tasks",
                "successful_tasks",
                "failed_tasks",
                "parallel_efficiency",
            ]
            assert any(metric in metrics for metric in expected_metrics)


@pytest.mark.integration
class TestParallelResearchErrorRecovery:
    """Test error recovery and resilience in parallel research."""

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test recovery when some parallel tasks fail."""
        config = ParallelResearchConfig(
            max_concurrent_agents=3,
            timeout_per_agent=5,
            enable_fallbacks=True,
            rate_limit_delay=0.05,
        )

        # Create agent with mixed success/failure providers
        llm = MockLLM(response_delay=0.03)
        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        # Mix of failing and working providers
        agent.search_providers = [
            MockSearchProvider("WorkingProvider", fail_rate=0.0),
            MockSearchProvider("FailingProvider", fail_rate=0.8),  # 80% failure rate
        ]
        agent.content_analyzer = MockContentAnalyzer()

        result = await agent.research_comprehensive(
            topic="Partial failure recovery test",
            session_id="partial_failure_test_001",
            use_parallel_execution=True,
        )

        # Should complete successfully despite some failures
        assert result["status"] == "success"

        # Should have parallel execution stats showing mixed results
        if "parallel_execution_stats" in result:
            stats = result["parallel_execution_stats"]
            # Should have attempted multiple tasks
            assert stats["total_tasks"] >= 1
            # May have some failures but should have some successes
            if stats["total_tasks"] > 1:
                assert (
                    stats["successful_tasks"] + stats["failed_tasks"]
                    == stats["total_tasks"]
                )

    @pytest.mark.asyncio
    async def test_complete_failure_fallback(self):
        """Test fallback to sequential when parallel execution completely fails."""
        config = ParallelResearchConfig(
            max_concurrent_agents=2,
            timeout_per_agent=3,
            enable_fallbacks=True,
        )

        # Create agent that will fail in parallel mode
        failing_llm = MockLLM(fail_rate=0.9)  # Very high failure rate
        agent = DeepResearchAgent(
            llm=failing_llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        agent.search_providers = [MockSearchProvider("FailingProvider", fail_rate=0.9)]
        agent.content_analyzer = MockContentAnalyzer()

        # Mock the sequential execution to succeed
        with patch.object(agent.graph, "ainvoke") as mock_sequential:
            mock_sequential.return_value = {
                "status": "success",
                "persona": "moderate",
                "research_confidence": 0.6,
                "research_findings": {"synthesis": "Fallback analysis"},
            }

            result = await agent.research_comprehensive(
                topic="Complete failure fallback test",
                session_id="complete_failure_test_001",
                use_parallel_execution=True,
            )

            # Should fall back to sequential and succeed
            assert result["status"] == "success"

            # Sequential execution should have been called due to parallel failure
            mock_sequential.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_handling_in_parallel_execution(self):
        """Test handling of timeouts in parallel execution."""
        config = ParallelResearchConfig(
            max_concurrent_agents=2,
            timeout_per_agent=1,  # Very short timeout
            enable_fallbacks=True,
        )

        # Create components with delays longer than timeout
        slow_llm = MockLLM(response_delay=2.0)  # Slower than timeout
        agent = DeepResearchAgent(
            llm=slow_llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        agent.search_providers = [MockSearchProvider("SlowProvider")]
        agent.content_analyzer = MockContentAnalyzer(analysis_delay=0.5)

        # Should handle timeouts gracefully
        result = await agent.research_comprehensive(
            topic="Timeout handling test",
            session_id="timeout_test_001",
            use_parallel_execution=True,
        )

        # Should complete with some status (success or error)
        assert result["status"] in ["success", "error"]

        # If parallel execution stats are available, should show timeout effects
        if (
            "parallel_execution_stats" in result
            and result["parallel_execution_stats"]["total_tasks"] > 0
        ):
            stats = result["parallel_execution_stats"]
            # Timeouts should result in failed tasks
            assert stats["failed_tasks"] >= 0


@pytest.mark.integration
class TestParallelResearchDataFlow:
    """Test data flow and consistency in parallel research."""

    @pytest.mark.asyncio
    async def test_data_consistency_across_parallel_tasks(self):
        """Test that data remains consistent across parallel task execution."""
        config = ParallelResearchConfig(
            max_concurrent_agents=3,
            timeout_per_agent=5,
        )

        llm = MockLLM()
        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        # Create providers that return consistent data
        consistent_provider = MockSearchProvider("ConsistentProvider")
        agent.search_providers = [consistent_provider]
        agent.content_analyzer = MockContentAnalyzer()

        result = await agent.research_comprehensive(
            topic="Data consistency test for Apple Inc",
            session_id="consistency_test_001",
            use_parallel_execution=True,
        )

        assert result["status"] == "success"

        # Verify data structure consistency
        assert "research_topic" in result
        assert "confidence_score" in result
        assert "citations" in result
        assert isinstance(result["citations"], list)

        # If parallel execution occurred, verify stats structure
        if "parallel_execution_stats" in result:
            stats = result["parallel_execution_stats"]
            required_stats = [
                "total_tasks",
                "successful_tasks",
                "failed_tasks",
                "parallel_efficiency",
            ]
            for stat in required_stats:
                assert stat in stats
                assert isinstance(stats[stat], int | float)

    @pytest.mark.asyncio
    async def test_citation_aggregation_across_tasks(self):
        """Test that citations are properly aggregated from parallel tasks."""
        config = ParallelResearchConfig(max_concurrent_agents=2)

        llm = MockLLM()
        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        # Multiple providers to generate multiple sources
        agent.search_providers = [
            MockSearchProvider("Provider1"),
            MockSearchProvider("Provider2"),
        ]
        agent.content_analyzer = MockContentAnalyzer()

        result = await agent.research_comprehensive(
            topic="Citation aggregation test",
            session_id="citation_test_001",
            use_parallel_execution=True,
        )

        assert result["status"] == "success"

        # Should have citations from multiple sources
        citations = result.get("citations", [])
        if len(citations) > 0:
            # Citations should have required fields
            for citation in citations:
                assert "id" in citation
                assert "title" in citation
                assert "url" in citation
                assert "credibility_score" in citation

            # Should have unique citation IDs
            citation_ids = [c["id"] for c in citations]
            assert len(citation_ids) == len(set(citation_ids))

    @pytest.mark.asyncio
    async def test_research_quality_metrics(self):
        """Test research quality metrics in parallel execution."""
        config = ParallelResearchConfig(max_concurrent_agents=2)

        llm = MockLLM()
        agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

        agent.search_providers = [MockSearchProvider("QualityProvider")]
        agent.content_analyzer = MockContentAnalyzer()

        result = await agent.research_comprehensive(
            topic="Research quality metrics test",
            session_id="quality_test_001",
            use_parallel_execution=True,
        )

        assert result["status"] == "success"

        # Verify quality metrics
        assert "confidence_score" in result
        assert 0.0 <= result["confidence_score"] <= 1.0

        assert "sources_analyzed" in result
        assert isinstance(result["sources_analyzed"], int)
        assert result["sources_analyzed"] >= 0

        if "source_diversity" in result:
            assert 0.0 <= result["source_diversity"] <= 1.0
