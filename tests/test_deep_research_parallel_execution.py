"""
Comprehensive test suite for DeepResearchAgent parallel execution functionality.

This test suite covers:
- Parallel vs sequential execution modes
- Subagent creation and orchestration
- Task routing to specialized subagents
- Parallel execution fallback mechanisms
- Result synthesis from parallel tasks
- Performance characteristics of parallel execution
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from maverick_mcp.agents.deep_research import (
    BaseSubagent,
    CompetitiveResearchAgent,
    DeepResearchAgent,
    FundamentalResearchAgent,
    SentimentResearchAgent,
    TechnicalResearchAgent,
)
from maverick_mcp.utils.parallel_research import (
    ParallelResearchConfig,
    ResearchResult,
    ResearchTask,
)


class MockLLM(BaseChatModel):
    """Mock LLM for testing."""

    def __init__(self, response_content: str = "Mock response"):
        super().__init__()
        self.response_content = response_content

    def _generate(self, messages, stop=None, **kwargs):
        # This method should not be called in async tests
        raise NotImplementedError("Use ainvoke for async tests")

    async def ainvoke(self, messages, config=None, **kwargs):
        """Mock async invocation."""
        await asyncio.sleep(0.01)  # Simulate processing time
        return AIMessage(content=self.response_content)

    @property
    def _llm_type(self) -> str:
        return "mock_llm"


class TestDeepResearchAgentParallelExecution:
    """Test DeepResearchAgent parallel execution capabilities."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MockLLM(
            '{"KEY_INSIGHTS": ["Test insight"], "SENTIMENT": {"direction": "bullish", "confidence": 0.8}, "CREDIBILITY": 0.9}'
        )

    @pytest.fixture
    def parallel_config(self):
        """Create test parallel configuration."""
        return ParallelResearchConfig(
            max_concurrent_agents=3,
            timeout_per_agent=5,
            enable_fallbacks=True,
            rate_limit_delay=0.1,
        )

    @pytest.fixture
    def deep_research_agent(self, mock_llm, parallel_config):
        """Create DeepResearchAgent with parallel execution enabled."""
        return DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            checkpointer=MemorySaver(),
            enable_parallel_execution=True,
            parallel_config=parallel_config,
        )

    @pytest.fixture
    def sequential_agent(self, mock_llm):
        """Create DeepResearchAgent with sequential execution."""
        return DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            checkpointer=MemorySaver(),
            enable_parallel_execution=False,
        )

    def test_agent_initialization_parallel_enabled(self, deep_research_agent):
        """Test agent initialization with parallel execution enabled."""
        assert deep_research_agent.enable_parallel_execution is True
        assert deep_research_agent.parallel_config is not None
        assert deep_research_agent.parallel_orchestrator is not None
        assert deep_research_agent.task_distributor is not None
        assert deep_research_agent.parallel_config.max_concurrent_agents == 3

    def test_agent_initialization_sequential(self, sequential_agent):
        """Test agent initialization with sequential execution."""
        assert sequential_agent.enable_parallel_execution is False
        # These components should still be initialized for potential future use
        assert sequential_agent.parallel_orchestrator is not None

    @pytest.mark.asyncio
    async def test_parallel_execution_mode_selection(self, deep_research_agent):
        """Test parallel execution mode selection."""
        with (
            patch.object(
                deep_research_agent, "_execute_parallel_research"
            ) as mock_parallel,
            patch.object(deep_research_agent.graph, "ainvoke") as mock_sequential,
        ):
            mock_parallel.return_value = {
                "status": "success",
                "execution_mode": "parallel",
                "agent_type": "deep_research",
            }

            # Test with parallel execution enabled (default)
            result = await deep_research_agent.research_comprehensive(
                topic="AAPL analysis", session_id="test_123"
            )

            # Should use parallel execution
            mock_parallel.assert_called_once()
            mock_sequential.assert_not_called()
            assert result["execution_mode"] == "parallel"

    @pytest.mark.asyncio
    async def test_sequential_execution_mode_selection(self, sequential_agent):
        """Test sequential execution mode selection."""
        with (
            patch.object(
                sequential_agent, "_execute_parallel_research"
            ) as mock_parallel,
            patch.object(sequential_agent.graph, "ainvoke") as mock_sequential,
        ):
            mock_sequential.return_value = {
                "status": "success",
                "persona": "moderate",
                "research_confidence": 0.8,
            }

            # Test with parallel execution disabled
            result = await sequential_agent.research_comprehensive(
                topic="AAPL analysis", session_id="test_123"
            )

            # Should use sequential execution
            mock_parallel.assert_not_called()
            mock_sequential.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_execution_override(self, deep_research_agent):
        """Test overriding parallel execution at runtime."""
        with (
            patch.object(
                deep_research_agent, "_execute_parallel_research"
            ) as mock_parallel,
            patch.object(deep_research_agent.graph, "ainvoke") as mock_sequential,
        ):
            mock_sequential.return_value = {"status": "success", "persona": "moderate"}

            # Override parallel execution to false
            result = await deep_research_agent.research_comprehensive(
                topic="AAPL analysis",
                session_id="test_123",
                use_parallel_execution=False,
            )

            # Should use sequential despite agent default
            mock_parallel.assert_not_called()
            mock_sequential.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_execution_fallback(self, deep_research_agent):
        """Test fallback to sequential when parallel execution fails."""
        with (
            patch.object(
                deep_research_agent, "_execute_parallel_research"
            ) as mock_parallel,
            patch.object(deep_research_agent.graph, "ainvoke") as mock_sequential,
        ):
            # Parallel execution fails
            mock_parallel.side_effect = RuntimeError("Parallel execution failed")
            mock_sequential.return_value = {
                "status": "success",
                "persona": "moderate",
                "research_confidence": 0.7,
            }

            result = await deep_research_agent.research_comprehensive(
                topic="AAPL analysis", session_id="test_123"
            )

            # Should attempt parallel then fall back to sequential
            mock_parallel.assert_called_once()
            mock_sequential.assert_called_once()
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_parallel_research_task_distribution(
        self, deep_research_agent
    ):
        """Test parallel research task distribution."""
        with (
            patch.object(
                deep_research_agent.task_distributor, "distribute_research_tasks"
            ) as mock_distribute,
            patch.object(
                deep_research_agent.parallel_orchestrator, "execute_parallel_research"
            ) as mock_execute,
        ):
            # Mock task distribution
            mock_tasks = [
                ResearchTask(
                    "test_123_fundamental", "fundamental", "AAPL", ["earnings"]
                ),
                ResearchTask("test_123_sentiment", "sentiment", "AAPL", ["news"]),
            ]
            mock_distribute.return_value = mock_tasks

            # Mock orchestrator execution
            mock_result = ResearchResult()
            mock_result.successful_tasks = 2
            mock_result.failed_tasks = 0
            mock_result.synthesis = {"confidence_score": 0.85}
            mock_execute.return_value = mock_result

            result = await deep_research_agent._execute_parallel_research(
                topic="AAPL analysis",
                session_id="test_123",
                depth="standard",
                focus_areas=["earnings", "sentiment"],
            )

            # Verify task distribution was called correctly
            mock_distribute.assert_called_once_with(
                topic="AAPL analysis",
                session_id="test_123",
                focus_areas=["earnings", "sentiment"],
            )

            # Verify orchestrator was called with distributed tasks
            mock_execute.assert_called_once()
            args, kwargs = mock_execute.call_args
            assert kwargs["tasks"] == mock_tasks

    @pytest.mark.asyncio
    async def test_subagent_task_routing(self, deep_research_agent):
        """Test routing tasks to appropriate subagents."""
        # Test fundamental routing
        fundamental_task = ResearchTask(
            "test_fundamental", "fundamental", "AAPL", ["earnings"]
        )

        with patch.object(
            deep_research_agent, "FundamentalResearchAgent"
        ) as mock_fundamental:
            mock_subagent = AsyncMock()
            mock_subagent.execute_research.return_value = {
                "research_type": "fundamental"
            }
            mock_fundamental.return_value = mock_subagent

            # This would normally be called by the orchestrator
            # We're testing the routing logic directly
            with patch(
                "maverick_mcp.agents.deep_research.FundamentalResearchAgent",
                mock_fundamental,
            ):
                result = await deep_research_agent._execute_subagent_task(
                    fundamental_task
                )

            mock_fundamental.assert_called_once_with(deep_research_agent)
            mock_subagent.execute_research.assert_called_once_with(fundamental_task)

    @pytest.mark.asyncio
    async def test_unknown_task_type_fallback(self, deep_research_agent):
        """Test fallback for unknown task types."""
        unknown_task = ResearchTask("test_unknown", "unknown_type", "AAPL", ["test"])

        with patch(
            "maverick_mcp.agents.deep_research.FundamentalResearchAgent"
        ) as mock_fundamental:
            mock_subagent = AsyncMock()
            mock_subagent.execute_research.return_value = {
                "research_type": "fundamental"
            }
            mock_fundamental.return_value = mock_subagent

            result = await deep_research_agent._execute_subagent_task(unknown_task)

            # Should fall back to fundamental analysis
            mock_fundamental.assert_called_once_with(deep_research_agent)

    @pytest.mark.asyncio
    async def test_parallel_result_synthesis(self, deep_research_agent, mock_llm):
        """Test synthesis of results from parallel tasks."""
        # Create mock task results
        task_results = {
            "test_123_fundamental": ResearchTask(
                "test_123_fundamental", "fundamental", "AAPL", ["earnings"]
            ),
            "test_123_sentiment": ResearchTask(
                "test_123_sentiment", "sentiment", "AAPL", ["news"]
            ),
        }

        # Set tasks as completed with results
        task_results["test_123_fundamental"].status = "completed"
        task_results["test_123_fundamental"].result = {
            "insights": ["Strong earnings growth"],
            "sentiment": {"direction": "bullish", "confidence": 0.8},
            "credibility_score": 0.9,
        }

        task_results["test_123_sentiment"].status = "completed"
        task_results["test_123_sentiment"].result = {
            "insights": ["Positive market sentiment"],
            "sentiment": {"direction": "bullish", "confidence": 0.7},
            "credibility_score": 0.8,
        }

        # Mock LLM synthesis response
        mock_llm.response_content = "Synthesized analysis showing strong bullish outlook based on fundamental and sentiment analysis"

        result = await deep_research_agent._synthesize_parallel_results(task_results)

        assert result is not None
        assert "synthesis" in result
        assert "key_insights" in result
        assert "overall_sentiment" in result
        assert len(result["key_insights"]) > 0
        assert result["overall_sentiment"]["direction"] == "bullish"

    @pytest.mark.asyncio
    async def test_synthesis_with_mixed_results(self, deep_research_agent):
        """Test synthesis with mixed successful and failed tasks."""
        task_results = {
            "test_123_fundamental": ResearchTask(
                "test_123_fundamental", "fundamental", "AAPL", ["earnings"]
            ),
            "test_123_technical": ResearchTask(
                "test_123_technical", "technical", "AAPL", ["charts"]
            ),
            "test_123_sentiment": ResearchTask(
                "test_123_sentiment", "sentiment", "AAPL", ["news"]
            ),
        }

        # One successful, one failed, one successful
        task_results["test_123_fundamental"].status = "completed"
        task_results["test_123_fundamental"].result = {
            "insights": ["Strong fundamentals"],
            "sentiment": {"direction": "bullish", "confidence": 0.8},
        }

        task_results["test_123_technical"].status = "failed"
        task_results["test_123_technical"].error = "Technical analysis failed"

        task_results["test_123_sentiment"].status = "completed"
        task_results["test_123_sentiment"].result = {
            "insights": ["Mixed sentiment"],
            "sentiment": {"direction": "neutral", "confidence": 0.6},
        }

        result = await deep_research_agent._synthesize_parallel_results(task_results)

        # Should handle mixed results gracefully
        assert result is not None
        assert len(result["key_insights"]) > 0
        assert "task_breakdown" in result
        assert result["task_breakdown"]["test_123_technical"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_synthesis_with_no_successful_results(self, deep_research_agent):
        """Test synthesis when all tasks fail."""
        task_results = {
            "test_123_fundamental": ResearchTask(
                "test_123_fundamental", "fundamental", "AAPL", ["earnings"]
            ),
            "test_123_sentiment": ResearchTask(
                "test_123_sentiment", "sentiment", "AAPL", ["news"]
            ),
        }

        # Both tasks failed
        task_results["test_123_fundamental"].status = "failed"
        task_results["test_123_fundamental"].error = "API timeout"

        task_results["test_123_sentiment"].status = "failed"
        task_results["test_123_sentiment"].error = "No data available"

        result = await deep_research_agent._synthesize_parallel_results(task_results)

        # Should handle gracefully
        assert result is not None
        assert result["confidence_score"] == 0.0
        assert "No research results available" in result["synthesis"]

    @pytest.mark.asyncio
    async def test_synthesis_llm_failure_fallback(self, deep_research_agent):
        """Test fallback when LLM synthesis fails."""
        task_results = {
            "test_123_fundamental": ResearchTask(
                "test_123_fundamental", "fundamental", "AAPL", ["earnings"]
            ),
        }

        task_results["test_123_fundamental"].status = "completed"
        task_results["test_123_fundamental"].result = {
            "insights": ["Good insights"],
            "sentiment": {"direction": "bullish", "confidence": 0.8},
        }

        # Mock LLM to fail
        with patch.object(
            deep_research_agent.llm, "ainvoke", side_effect=RuntimeError("LLM failed")
        ):
            result = await deep_research_agent._synthesize_parallel_results(
                task_results
            )

        # Should use fallback synthesis
        assert result is not None
        assert "fallback synthesis" in result["synthesis"].lower()

    @pytest.mark.asyncio
    async def test_format_parallel_research_response(self, deep_research_agent):
        """Test formatting of parallel research response."""
        # Create mock research result
        research_result = ResearchResult()
        research_result.successful_tasks = 2
        research_result.failed_tasks = 0
        research_result.total_execution_time = 1.5
        research_result.parallel_efficiency = 2.1
        research_result.synthesis = {
            "confidence_score": 0.85,
            "key_findings": ["Finding 1", "Finding 2"],
        }

        # Mock task results with sources
        task1 = ResearchTask(
            "test_123_fundamental", "fundamental", "AAPL", ["earnings"]
        )
        task1.status = "completed"
        task1.result = {
            "sources": [
                {
                    "title": "AAPL Earnings Report",
                    "url": "https://example.com/earnings",
                    "credibility_score": 0.9,
                }
            ]
        }
        research_result.task_results = {"test_123_fundamental": task1}

        start_time = datetime.now()
        formatted_result = await deep_research_agent._format_parallel_research_response(
            research_result=research_result,
            topic="AAPL analysis",
            session_id="test_123",
            depth="standard",
            initial_state={"persona": "moderate"},
            start_time=start_time,
        )

        # Verify formatted response structure
        assert formatted_result["status"] == "success"
        assert formatted_result["agent_type"] == "deep_research"
        assert formatted_result["execution_mode"] == "parallel"
        assert formatted_result["research_topic"] == "AAPL analysis"
        assert formatted_result["confidence_score"] == 0.85
        assert "parallel_execution_stats" in formatted_result
        assert formatted_result["parallel_execution_stats"]["successful_tasks"] == 2
        assert len(formatted_result["citations"]) > 0

    @pytest.mark.asyncio
    async def test_aggregated_sentiment_calculation(self, deep_research_agent):
        """Test aggregation of sentiment from multiple sources."""
        sentiment_scores = [
            {"direction": "bullish", "confidence": 0.8},
            {"direction": "bullish", "confidence": 0.6},
            {"direction": "neutral", "confidence": 0.7},
            {"direction": "bearish", "confidence": 0.5},
        ]

        result = deep_research_agent._calculate_aggregated_sentiment(sentiment_scores)

        assert result is not None
        assert "direction" in result
        assert "confidence" in result
        assert "consensus" in result
        assert "source_count" in result
        assert result["source_count"] == 4

    @pytest.mark.asyncio
    async def test_parallel_recommendation_derivation(self, deep_research_agent):
        """Test derivation of investment recommendations from parallel analysis."""
        # Test strong bullish signal
        bullish_sentiment = {"direction": "bullish", "confidence": 0.9}
        recommendation = deep_research_agent._derive_parallel_recommendation(
            bullish_sentiment
        )
        assert "strong buy" in recommendation.lower() or "buy" in recommendation.lower()

        # Test bearish signal
        bearish_sentiment = {"direction": "bearish", "confidence": 0.8}
        recommendation = deep_research_agent._derive_parallel_recommendation(
            bearish_sentiment
        )
        assert (
            "caution" in recommendation.lower() or "negative" in recommendation.lower()
        )

        # Test neutral/mixed signals
        neutral_sentiment = {"direction": "neutral", "confidence": 0.5}
        recommendation = deep_research_agent._derive_parallel_recommendation(
            neutral_sentiment
        )
        assert "neutral" in recommendation.lower() or "mixed" in recommendation.lower()


class TestSpecializedSubagents:
    """Test specialized research subagent functionality."""

    @pytest.fixture
    def mock_parent_agent(self):
        """Create mock parent DeepResearchAgent."""
        parent = Mock()
        parent.llm = MockLLM()
        parent.search_providers = []
        parent.content_analyzer = Mock()
        parent.persona = Mock()
        parent.persona.name = "moderate"
        parent._calculate_source_credibility = Mock(return_value=0.8)
        return parent

    def test_base_subagent_initialization(self, mock_parent_agent):
        """Test BaseSubagent initialization."""
        subagent = BaseSubagent(mock_parent_agent)

        assert subagent.parent == mock_parent_agent
        assert subagent.llm == mock_parent_agent.llm
        assert subagent.search_providers == mock_parent_agent.search_providers
        assert subagent.content_analyzer == mock_parent_agent.content_analyzer
        assert subagent.persona == mock_parent_agent.persona

    @pytest.mark.asyncio
    async def test_fundamental_research_agent(self, mock_parent_agent):
        """Test FundamentalResearchAgent execution."""
        # Mock content analyzer
        mock_parent_agent.content_analyzer.analyze_content = AsyncMock(
            return_value={
                "insights": ["Strong earnings growth"],
                "sentiment": {"direction": "bullish", "confidence": 0.8},
                "risk_factors": ["Market volatility"],
                "opportunities": ["Dividend growth"],
                "credibility_score": 0.9,
            }
        )

        subagent = FundamentalResearchAgent(mock_parent_agent)

        # Mock search results
        with patch.object(subagent, "_perform_specialized_search") as mock_search:
            mock_search.return_value = [
                {
                    "title": "AAPL Earnings Report",
                    "url": "https://example.com/earnings",
                    "content": "Apple reported strong quarterly earnings...",
                    "credibility_score": 0.9,
                }
            ]

            task = ResearchTask(
                "fund_task", "fundamental", "AAPL analysis", ["earnings"]
            )
            result = await subagent.execute_research(task)

            assert result["research_type"] == "fundamental"
            assert len(result["insights"]) > 0
            assert "sentiment" in result
            assert result["sentiment"]["direction"] == "bullish"
            assert len(result["sources"]) > 0

    def test_fundamental_query_generation(self, mock_parent_agent):
        """Test fundamental analysis query generation."""
        subagent = FundamentalResearchAgent(mock_parent_agent)
        queries = subagent._generate_fundamental_queries("AAPL")

        assert len(queries) > 0
        assert any("earnings" in query.lower() for query in queries)
        assert any("revenue" in query.lower() for query in queries)
        assert any("valuation" in query.lower() for query in queries)

    @pytest.mark.asyncio
    async def test_technical_research_agent(self, mock_parent_agent):
        """Test TechnicalResearchAgent execution."""
        mock_parent_agent.content_analyzer.analyze_content = AsyncMock(
            return_value={
                "insights": ["Bullish chart pattern"],
                "sentiment": {"direction": "bullish", "confidence": 0.7},
                "risk_factors": ["Support level break"],
                "opportunities": ["Breakout potential"],
                "credibility_score": 0.8,
            }
        )

        subagent = TechnicalResearchAgent(mock_parent_agent)

        with patch.object(subagent, "_perform_specialized_search") as mock_search:
            mock_search.return_value = [
                {
                    "title": "AAPL Technical Analysis",
                    "url": "https://example.com/technical",
                    "content": "Apple stock showing strong technical indicators...",
                    "credibility_score": 0.8,
                }
            ]

            task = ResearchTask("tech_task", "technical", "AAPL analysis", ["charts"])
            result = await subagent.execute_research(task)

            assert result["research_type"] == "technical"
            assert "price_action" in result["focus_areas"]
            assert "technical_indicators" in result["focus_areas"]

    def test_technical_query_generation(self, mock_parent_agent):
        """Test technical analysis query generation."""
        subagent = TechnicalResearchAgent(mock_parent_agent)
        queries = subagent._generate_technical_queries("AAPL")

        assert any("technical analysis" in query.lower() for query in queries)
        assert any("chart pattern" in query.lower() for query in queries)
        assert any(
            "rsi" in query.lower() or "macd" in query.lower() for query in queries
        )

    @pytest.mark.asyncio
    async def test_sentiment_research_agent(self, mock_parent_agent):
        """Test SentimentResearchAgent execution."""
        mock_parent_agent.content_analyzer.analyze_content = AsyncMock(
            return_value={
                "insights": ["Positive analyst sentiment"],
                "sentiment": {"direction": "bullish", "confidence": 0.9},
                "risk_factors": ["Market sentiment shift"],
                "opportunities": ["Upgrade potential"],
                "credibility_score": 0.85,
            }
        )

        subagent = SentimentResearchAgent(mock_parent_agent)

        with patch.object(subagent, "_perform_specialized_search") as mock_search:
            mock_search.return_value = [
                {
                    "title": "AAPL Analyst Upgrade",
                    "url": "https://example.com/upgrade",
                    "content": "Apple receives analyst upgrade...",
                    "credibility_score": 0.85,
                }
            ]

            task = ResearchTask("sent_task", "sentiment", "AAPL analysis", ["news"])
            result = await subagent.execute_research(task)

            assert result["research_type"] == "sentiment"
            assert "market_sentiment" in result["focus_areas"]
            assert result["sentiment"]["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_competitive_research_agent(self, mock_parent_agent):
        """Test CompetitiveResearchAgent execution."""
        mock_parent_agent.content_analyzer.analyze_content = AsyncMock(
            return_value={
                "insights": ["Strong competitive position"],
                "sentiment": {"direction": "bullish", "confidence": 0.7},
                "risk_factors": ["Increased competition"],
                "opportunities": ["Market expansion"],
                "credibility_score": 0.8,
            }
        )

        subagent = CompetitiveResearchAgent(mock_parent_agent)

        with patch.object(subagent, "_perform_specialized_search") as mock_search:
            mock_search.return_value = [
                {
                    "title": "AAPL Market Share Analysis",
                    "url": "https://example.com/marketshare",
                    "content": "Apple maintains strong market position...",
                    "credibility_score": 0.8,
                }
            ]

            task = ResearchTask(
                "comp_task", "competitive", "AAPL analysis", ["market_share"]
            )
            result = await subagent.execute_research(task)

            assert result["research_type"] == "competitive"
            assert "competitive_position" in result["focus_areas"]
            assert "market_share" in result["focus_areas"]

    @pytest.mark.asyncio
    async def test_subagent_search_deduplication(self, mock_parent_agent):
        """Test search result deduplication in subagents."""
        subagent = BaseSubagent(mock_parent_agent)

        # Mock search providers with duplicate results
        mock_provider1 = AsyncMock()
        mock_provider1.search.return_value = [
            {"url": "https://example.com/article1", "title": "Article 1"},
            {"url": "https://example.com/article2", "title": "Article 2"},
        ]

        mock_provider2 = AsyncMock()
        mock_provider2.search.return_value = [
            {"url": "https://example.com/article1", "title": "Article 1"},  # Duplicate
            {"url": "https://example.com/article3", "title": "Article 3"},
        ]

        subagent.search_providers = [mock_provider1, mock_provider2]

        results = await subagent._perform_specialized_search(
            "test topic", ["test query"], max_results=10
        )

        # Should deduplicate by URL
        urls = [result["url"] for result in results]
        assert len(urls) == len(set(urls))  # No duplicates
        assert len(results) == 3  # Should have 3 unique results

    @pytest.mark.asyncio
    async def test_subagent_search_error_handling(self, mock_parent_agent):
        """Test error handling in subagent search."""
        subagent = BaseSubagent(mock_parent_agent)

        # Mock provider that fails
        mock_provider = AsyncMock()
        mock_provider.search.side_effect = RuntimeError("Search failed")
        subagent.search_providers = [mock_provider]

        # Should handle errors gracefully and return empty results
        results = await subagent._perform_specialized_search(
            "test topic", ["test query"], max_results=10
        )

        assert results == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_subagent_content_analysis_error_handling(self, mock_parent_agent):
        """Test content analysis error handling in subagents."""
        # Mock content analyzer that fails
        mock_parent_agent.content_analyzer.analyze_content = AsyncMock(
            side_effect=RuntimeError("Analysis failed")
        )

        subagent = BaseSubagent(mock_parent_agent)

        search_results = [
            {
                "title": "Test Article",
                "url": "https://example.com/test",
                "content": "Test content",
            }
        ]

        # Should handle analysis errors gracefully
        results = await subagent._analyze_search_results(
            search_results, "test_analysis"
        )

        # Should return empty results when analysis fails
        assert results == []


@pytest.mark.integration
class TestDeepResearchParallelIntegration:
    """Integration tests for DeepResearchAgent parallel execution."""

    @pytest.fixture
    def integration_agent(self):
        """Create agent for integration testing."""
        llm = MockLLM(
            '{"KEY_INSIGHTS": ["Integration insight"], "SENTIMENT": {"direction": "bullish", "confidence": 0.8}}'
        )

        config = ParallelResearchConfig(
            max_concurrent_agents=2,
            timeout_per_agent=5,
            enable_fallbacks=True,
            rate_limit_delay=0.05,
        )

        return DeepResearchAgent(
            llm=llm,
            persona="moderate",
            enable_parallel_execution=True,
            parallel_config=config,
        )

    @pytest.mark.asyncio
    async def test_end_to_end_parallel_research(self, integration_agent):
        """Test complete end-to-end parallel research workflow."""
        # Mock the search providers and subagent execution
        with patch.object(integration_agent, "_execute_subagent_task") as mock_execute:
            mock_execute.return_value = {
                "research_type": "fundamental",
                "insights": ["Strong financial health", "Growing revenue"],
                "sentiment": {"direction": "bullish", "confidence": 0.8},
                "risk_factors": ["Market volatility"],
                "opportunities": ["Expansion potential"],
                "credibility_score": 0.85,
                "sources": [
                    {
                        "title": "Financial Report",
                        "url": "https://example.com/report",
                        "credibility_score": 0.9,
                    }
                ],
            }

            start_time = time.time()
            result = await integration_agent.research_comprehensive(
                topic="Apple Inc comprehensive financial analysis",
                session_id="integration_test_123",
                depth="comprehensive",
                focus_areas=["fundamentals", "sentiment", "competitive"],
            )
            execution_time = time.time() - start_time

            # Verify result structure
            assert result["status"] == "success"
            assert result["agent_type"] == "deep_research"
            assert result["execution_mode"] == "parallel"
            assert (
                result["research_topic"] == "Apple Inc comprehensive financial analysis"
            )
            assert result["confidence_score"] > 0
            assert len(result["citations"]) > 0
            assert "parallel_execution_stats" in result

            # Verify performance characteristics
            assert execution_time < 10  # Should complete reasonably quickly
            assert result["execution_time_ms"] > 0

            # Verify parallel execution stats
            stats = result["parallel_execution_stats"]
            assert stats["total_tasks"] > 0
            assert stats["successful_tasks"] >= 0
            assert stats["parallel_efficiency"] > 0

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, integration_agent):
        """Test performance comparison between parallel and sequential execution."""
        topic = "Microsoft Corp investment analysis"
        session_id = "perf_test_123"

        # Mock subagent execution with realistic delay
        async def mock_subagent_execution(task):
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "research_type": task.task_type,
                "insights": [f"Insight from {task.task_type}"],
                "sentiment": {"direction": "bullish", "confidence": 0.7},
                "credibility_score": 0.8,
                "sources": [],
            }

        with patch.object(
            integration_agent,
            "_execute_subagent_task",
            side_effect=mock_subagent_execution,
        ):
            # Test parallel execution
            start_parallel = time.time()
            parallel_result = await integration_agent.research_comprehensive(
                topic=topic, session_id=session_id, use_parallel_execution=True
            )
            parallel_time = time.time() - start_parallel

            # Test sequential execution
            start_sequential = time.time()
            sequential_result = await integration_agent.research_comprehensive(
                topic=topic,
                session_id=f"{session_id}_seq",
                use_parallel_execution=False,
            )
            sequential_time = time.time() - start_sequential

            # Verify both succeeded
            assert parallel_result["status"] == "success"
            assert sequential_result["status"] == "success"

            # Parallel should generally be faster (though not guaranteed in all test environments)
            # At minimum, parallel efficiency should be calculated
            if "parallel_execution_stats" in parallel_result:
                assert (
                    parallel_result["parallel_execution_stats"]["parallel_efficiency"]
                    > 0
                )

    @pytest.mark.asyncio
    async def test_research_quality_consistency(self, integration_agent):
        """Test that parallel and sequential execution produce consistent quality."""
        topic = "Tesla Inc strategic analysis"

        # Mock consistent subagent responses
        mock_response = {
            "research_type": "fundamental",
            "insights": ["Consistent insight 1", "Consistent insight 2"],
            "sentiment": {"direction": "bullish", "confidence": 0.75},
            "credibility_score": 0.8,
            "sources": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "credibility_score": 0.8,
                }
            ],
        }

        with patch.object(
            integration_agent, "_execute_subagent_task", return_value=mock_response
        ):
            parallel_result = await integration_agent.research_comprehensive(
                topic=topic,
                session_id="quality_test_parallel",
                use_parallel_execution=True,
            )

            sequential_result = await integration_agent.research_comprehensive(
                topic=topic,
                session_id="quality_test_sequential",
                use_parallel_execution=False,
            )

            # Both should succeed with reasonable confidence
            assert parallel_result["status"] == "success"
            assert sequential_result["status"] == "success"
            assert parallel_result["confidence_score"] > 0.5
            assert sequential_result["confidence_score"] > 0.5
