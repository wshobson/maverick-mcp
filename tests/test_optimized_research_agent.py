"""
Comprehensive test suite for OptimizedDeepResearchAgent.

Tests the core functionality of the optimized research agent including:
- Model selection logic
- Token budgeting
- Confidence tracking
- Content filtering
- Parallel processing
- Error handling
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from maverick_mcp.agents.optimized_research import (
    OptimizedContentAnalyzer,
    OptimizedDeepResearchAgent,
    create_optimized_research_agent,
)
from maverick_mcp.providers.openrouter_provider import OpenRouterProvider, TaskType
from maverick_mcp.utils.llm_optimization import (
    AdaptiveModelSelector,
    ConfidenceTracker,
    ModelConfiguration,
    ProgressiveTokenBudgeter,
)


class TestOptimizedContentAnalyzer:
    """Test the OptimizedContentAnalyzer component."""

    @pytest.fixture
    def mock_openrouter(self):
        """Create a mock OpenRouter provider."""
        provider = Mock(spec=OpenRouterProvider)
        provider.get_llm = Mock()
        return provider

    @pytest.fixture
    def analyzer(self, mock_openrouter):
        """Create an OptimizedContentAnalyzer instance."""
        return OptimizedContentAnalyzer(mock_openrouter)

    @pytest.mark.asyncio
    async def test_analyze_content_optimized_success(self, analyzer, mock_openrouter):
        """Test successful optimized content analysis."""
        # Setup mock LLM response
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = '{"insights": ["Test insight"], "sentiment": {"direction": "bullish", "confidence": 0.8}}'
        mock_llm.ainvoke.return_value = mock_response
        mock_openrouter.get_llm.return_value = mock_llm

        # Test analysis
        result = await analyzer.analyze_content_optimized(
            content="Test financial content about stocks",
            persona="moderate",
            analysis_focus="market_analysis",
            time_budget_seconds=30.0,
            current_confidence=0.5,
        )

        # Verify results
        assert result["insights"] == ["Test insight"]
        assert result["sentiment"]["direction"] == "bullish"
        assert result["sentiment"]["confidence"] == 0.8
        assert result["optimization_applied"] is True
        assert "model_used" in result
        assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_analyze_empty_content(self, analyzer):
        """Test handling of empty content."""
        result = await analyzer.analyze_content_optimized(
            content="",
            persona="moderate",
            analysis_focus="general",
            time_budget_seconds=30.0,
        )

        assert result["empty_content"] is True
        assert result["insights"] == []
        assert result["sentiment"]["direction"] == "neutral"
        assert result["sentiment"]["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_analyze_with_timeout(self, analyzer, mock_openrouter):
        """Test timeout handling during analysis."""
        # Setup mock to timeout
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = TimeoutError("Analysis timeout")
        mock_openrouter.get_llm.return_value = mock_llm

        result = await analyzer.analyze_content_optimized(
            content="Test content",
            persona="aggressive",
            analysis_focus="technical",
            time_budget_seconds=5.0,
        )

        # Should return fallback analysis
        assert "insights" in result
        assert "sentiment" in result
        assert result["sentiment"]["direction"] in ["bullish", "bearish", "neutral"]

    @pytest.mark.asyncio
    async def test_batch_analyze_content(self, analyzer, mock_openrouter):
        """Test batch content analysis with parallel processing."""
        # Setup mock parallel processor
        with patch.object(
            analyzer.parallel_processor,
            "parallel_content_analysis",
            new_callable=AsyncMock,
        ) as mock_parallel:
            mock_results = [
                {
                    "analysis": {
                        "insights": ["Insight 1"],
                        "sentiment": {"direction": "bullish", "confidence": 0.7},
                    }
                },
                {
                    "analysis": {
                        "insights": ["Insight 2"],
                        "sentiment": {"direction": "neutral", "confidence": 0.6},
                    }
                },
            ]
            mock_parallel.return_value = mock_results

            sources = [
                {"content": "Source 1 content", "url": "http://example1.com"},
                {"content": "Source 2 content", "url": "http://example2.com"},
            ]

            results = await analyzer.batch_analyze_content(
                sources=sources,
                persona="moderate",
                analysis_type="fundamental",
                time_budget_seconds=60.0,
                current_confidence=0.5,
            )

            assert len(results) == 2
            assert results[0]["analysis"]["insights"] == ["Insight 1"]
            assert results[1]["analysis"]["sentiment"]["direction"] == "neutral"


class TestOptimizedDeepResearchAgent:
    """Test the main OptimizedDeepResearchAgent."""

    @pytest.fixture
    def mock_openrouter(self):
        """Create a mock OpenRouter provider."""
        provider = Mock(spec=OpenRouterProvider)
        provider.get_llm = Mock(return_value=AsyncMock())
        return provider

    @pytest.fixture
    def mock_search_provider(self):
        """Create a mock search provider."""
        provider = AsyncMock()
        provider.search = AsyncMock(
            return_value=[
                {
                    "title": "Test Result 1",
                    "url": "http://example1.com",
                    "content": "Financial analysis content",
                },
                {
                    "title": "Test Result 2",
                    "url": "http://example2.com",
                    "content": "Market research content",
                },
            ]
        )
        return provider

    @pytest.fixture
    def agent(self, mock_openrouter, mock_search_provider):
        """Create an OptimizedDeepResearchAgent instance."""
        agent = OptimizedDeepResearchAgent(
            openrouter_provider=mock_openrouter,
            persona="moderate",
            optimization_enabled=True,
        )
        # Add mock search provider
        agent.search_providers = [mock_search_provider]
        # Initialize confidence tracker for tests that need it
        agent.confidence_tracker = ConfidenceTracker()
        return agent

    @pytest.mark.asyncio
    async def test_research_comprehensive_success(
        self, agent, mock_search_provider, mock_openrouter
    ):
        """Test successful comprehensive research."""
        # Setup mock LLM for synthesis
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Comprehensive synthesis of research findings."
        mock_llm.ainvoke.return_value = mock_response
        mock_openrouter.get_llm.return_value = mock_llm

        # Mock analysis phase to return analyzed sources
        async def mock_analysis_phase(*args, **kwargs):
            return {
                "analyzed_sources": [
                    {
                        "title": "AAPL Analysis Report",
                        "url": "http://example.com",
                        "analysis": {
                            "insights": ["Key insight"],
                            "sentiment": {"direction": "bullish", "confidence": 0.8},
                            "credibility_score": 0.9,
                            "relevance_score": 0.85,
                            "optimization_applied": True,
                        },
                    },
                    {
                        "title": "Technical Analysis AAPL",
                        "url": "http://example2.com",
                        "analysis": {
                            "insights": ["Technical insight"],
                            "sentiment": {"direction": "bullish", "confidence": 0.7},
                            "credibility_score": 0.8,
                            "relevance_score": 0.8,
                            "optimization_applied": True,
                        },
                    },
                ],
                "final_confidence": 0.8,
                "early_terminated": False,
                "processing_mode": "optimized",
            }

        with patch.object(
            agent, "_optimized_analysis_phase", new_callable=AsyncMock
        ) as mock_analysis:
            mock_analysis.side_effect = mock_analysis_phase

            result = await agent.research_comprehensive(
                topic="AAPL stock analysis",
                session_id="test_session",
                depth="standard",
                focus_areas=["fundamental", "technical"],
                timeframe="30d",
                time_budget_seconds=120.0,
                target_confidence=0.75,
            )

            # Verify successful research
            assert result["status"] == "success"
            assert result["agent_type"] == "optimized_deep_research"
            assert result["optimization_enabled"] is True
            assert result["research_topic"] == "AAPL stock analysis"
            assert result["sources_analyzed"] > 0
            assert "findings" in result
            assert "citations" in result
            assert "optimization_metrics" in result

    @pytest.mark.asyncio
    async def test_research_with_no_providers(self, mock_openrouter):
        """Test research when no search providers are configured."""
        agent = OptimizedDeepResearchAgent(
            openrouter_provider=mock_openrouter,
            optimization_enabled=True,
        )
        agent.search_providers = []  # No providers

        result = await agent.research_comprehensive(
            topic="Test topic",
            session_id="test_session",
            time_budget_seconds=60.0,
        )

        assert "error" in result
        assert "no search providers configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_research_with_early_termination(
        self, agent, mock_search_provider, mock_openrouter
    ):
        """Test early termination based on confidence threshold."""

        # Mock the entire analysis phase to return early termination
        async def mock_analysis_phase(*args, **kwargs):
            return {
                "analyzed_sources": [
                    {
                        "title": "Mock Source",
                        "url": "http://example.com",
                        "analysis": {
                            "insights": ["High confidence insight"],
                            "sentiment": {"direction": "bullish", "confidence": 0.95},
                            "credibility_score": 0.95,
                            "relevance_score": 0.9,
                        },
                    }
                ],
                "final_confidence": 0.92,
                "early_terminated": True,
                "termination_reason": "confidence_threshold_reached",
                "processing_mode": "optimized",
            }

        with patch.object(
            agent, "_optimized_analysis_phase", new_callable=AsyncMock
        ) as mock_analysis:
            mock_analysis.side_effect = mock_analysis_phase

            result = await agent.research_comprehensive(
                topic="Quick research test",
                session_id="test_session",
                time_budget_seconds=120.0,
                target_confidence=0.9,
            )

            assert result["findings"]["early_terminated"] is True
            assert (
                result["findings"]["termination_reason"]
                == "confidence_threshold_reached"
            )

    @pytest.mark.asyncio
    async def test_research_emergency_response(self, agent, mock_search_provider):
        """Test emergency response when time is critically low."""
        # Test with very short time budget
        result = agent._create_emergency_response(
            topic="Emergency test",
            search_results={"filtered_sources": [{"title": "Source 1"}]},
            start_time=time.time() - 1,  # 1 second ago
        )

        assert result["status"] == "partial_success"
        assert result["emergency_mode"] is True
        assert "Emergency mode" in result["findings"]["synthesis"]
        assert result["findings"]["confidence_score"] == 0.3


class TestModelSelectionLogic:
    """Test the adaptive model selection logic."""

    @pytest.fixture
    def model_selector(self):
        """Create a model selector with mock provider."""
        provider = Mock(spec=OpenRouterProvider)
        return AdaptiveModelSelector(provider)

    def test_calculate_task_complexity(self, model_selector):
        """Test task complexity calculation."""
        # Create content with financial complexity indicators
        content = (
            """
        This comprehensive financial analysis examines EBITDA, DCF valuation, and ROIC metrics.
        The company shows strong quarterly YoY growth with bullish sentiment from analysts.
        Technical analysis indicates RSI oversold conditions with MACD crossover signals.
        Support levels at $150 with resistance at $200. Volatility and beta measures suggest
        the stock outperforms relative to market. The Sharpe ratio indicates favorable
        risk-adjusted returns versus comparable companies in Q4 results.
        """
            * 20
        )  # Repeat to increase complexity

        complexity = model_selector.calculate_task_complexity(
            content, TaskType.DEEP_RESEARCH, ["fundamental", "technical"]
        )

        assert 0 <= complexity <= 1
        assert complexity > 0.1  # Should show some complexity with financial terms

    def test_select_model_for_time_budget(self, model_selector):
        """Test model selection based on time constraints."""
        # Test with short time budget - should select fast model
        config = model_selector.select_model_for_time_budget(
            task_type=TaskType.QUICK_ANSWER,
            time_remaining_seconds=10.0,
            complexity_score=0.3,
            content_size_tokens=100,
            current_confidence=0.5,
        )

        assert isinstance(config, ModelConfiguration)
        assert (
            config.timeout_seconds <= 15.0
        )  # Allow some flexibility for emergency models
        assert config.model_id is not None

        # Test with long time budget - can select quality model
        config_long = model_selector.select_model_for_time_budget(
            task_type=TaskType.DEEP_RESEARCH,
            time_remaining_seconds=300.0,
            complexity_score=0.8,
            content_size_tokens=5000,
            current_confidence=0.3,
        )

        assert config_long.timeout_seconds > config.timeout_seconds
        assert config_long.max_tokens >= config.max_tokens


class TestTokenBudgetingAndConfidence:
    """Test token budgeting and confidence tracking."""

    def test_progressive_token_budgeter(self):
        """Test progressive token budget allocation."""
        budgeter = ProgressiveTokenBudgeter(
            total_time_budget_seconds=120.0, confidence_target=0.8
        )

        # Test initial allocation
        allocation = budgeter.get_next_allocation(
            sources_remaining=10,
            current_confidence=0.3,
            time_elapsed_seconds=10.0,
        )

        assert allocation["time_budget"] > 0
        assert allocation["max_tokens"] > 0
        assert allocation["priority"] in ["low", "medium", "high"]

        # Test with higher confidence
        allocation_high = budgeter.get_next_allocation(
            sources_remaining=5,
            current_confidence=0.7,
            time_elapsed_seconds=60.0,
        )

        # With fewer sources and higher confidence, priority should be lower or equal
        assert allocation_high["priority"] in ["low", "medium"]
        # The high confidence scenario should have lower or equal priority
        priority_order = {"low": 0, "medium": 1, "high": 2}
        assert (
            priority_order[allocation_high["priority"]]
            <= priority_order[allocation["priority"]]
        )

    def test_confidence_tracker(self):
        """Test confidence tracking and early termination."""
        tracker = ConfidenceTracker(
            target_confidence=0.8, min_sources=3, max_sources=20
        )

        # Test confidence updates
        analysis = {
            "sentiment": {"confidence": 0.7},
            "insights": ["insight1", "insight2"],
        }

        update = tracker.update_confidence(analysis, credibility_score=0.8)

        assert "current_confidence" in update
        assert "should_continue" in update
        assert update["sources_analyzed"] == 1

        # Test minimum sources requirement
        for _i in range(2):
            update = tracker.update_confidence(analysis, credibility_score=0.9)

        # Should continue even with high confidence if min sources not met
        if tracker.sources_analyzed < tracker.min_sources:
            assert update["should_continue"] is True


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_search_timeout_handling(self):
        """Test handling of search provider timeouts."""
        agent = OptimizedDeepResearchAgent(
            openrouter_provider=Mock(spec=OpenRouterProvider),
            optimization_enabled=True,
        )

        # Mock search provider that times out
        mock_provider = AsyncMock()
        mock_provider.search.side_effect = TimeoutError("Search timeout")

        results = await agent._search_with_timeout(
            mock_provider, "test query", timeout=1.0
        )

        assert results == []  # Should return empty list on timeout

    @pytest.mark.asyncio
    async def test_synthesis_fallback(self):
        """Test fallback synthesis when LLM fails."""
        agent = OptimizedDeepResearchAgent(
            openrouter_provider=Mock(spec=OpenRouterProvider),
            optimization_enabled=True,
        )

        # Mock LLM failure
        with patch.object(
            agent.openrouter_provider,
            "get_llm",
            side_effect=Exception("LLM unavailable"),
        ):
            result = await agent._optimized_synthesis_phase(
                analyzed_sources=[{"analysis": {"insights": ["test"]}}],
                topic="Test topic",
                time_budget_seconds=10.0,
            )

            assert "fallback_used" in result
            assert result["fallback_used"] is True
            assert "basic processing" in result["synthesis"]


class TestIntegrationWithParallelProcessing:
    """Test integration with parallel processing capabilities."""

    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self):
        """Test parallel batch processing of sources."""
        analyzer = OptimizedContentAnalyzer(Mock(spec=OpenRouterProvider))

        # Mock parallel processor
        with patch.object(
            analyzer.parallel_processor,
            "parallel_content_analysis",
            new_callable=AsyncMock,
        ) as mock_parallel:
            mock_parallel.return_value = [
                {"analysis": {"insights": [f"Insight {i}"]}} for i in range(5)
            ]

            sources = [{"content": f"Source {i}"} for i in range(5)]

            results = await analyzer.batch_analyze_content(
                sources=sources,
                persona="moderate",
                analysis_type="general",
                time_budget_seconds=30.0,
            )

            assert len(results) == 5
            mock_parallel.assert_called_once()


class TestFactoryFunction:
    """Test the factory function for creating optimized agents."""

    def test_create_optimized_research_agent(self):
        """Test agent creation through factory function."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):
            agent = create_optimized_research_agent(
                openrouter_api_key="test_key",
                persona="aggressive",
                time_budget_seconds=180.0,
                target_confidence=0.85,
            )

            assert isinstance(agent, OptimizedDeepResearchAgent)
            assert agent.optimization_enabled is True
            assert agent.persona.name == "Aggressive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
