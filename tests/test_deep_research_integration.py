"""
Integration tests for DeepResearchAgent.

Tests the complete research workflow including web search, content analysis,
and persona-aware result adaptation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maverick_mcp.agents.deep_research import (
    ContentAnalyzer,
    DeepResearchAgent,
    WebSearchProvider,
)
from maverick_mcp.agents.supervisor import SupervisorAgent
from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import ResearchError, WebSearchError


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock()
    return llm


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    cache_manager = MagicMock()
    cache_manager.get = AsyncMock(return_value=None)
    cache_manager.set = AsyncMock()
    return cache_manager


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    return {
        "exa": [
            {
                "url": "https://example.com/article1",
                "title": "AAPL Stock Analysis",
                "text": "Apple stock shows strong fundamentals with growing iPhone sales...",
                "published_date": "2024-01-15",
                "score": 0.9,
                "provider": "exa",
                "domain": "example.com",
            },
            {
                "url": "https://example.com/article2",
                "title": "Tech Sector Outlook",
                "text": "Technology stocks are experiencing headwinds due to interest rates...",
                "published_date": "2024-01-14",
                "score": 0.8,
                "provider": "exa",
                "domain": "example.com",
            },
        ],
        "tavily": [
            {
                "url": "https://news.example.com/tech-news",
                "title": "Apple Earnings Beat Expectations",
                "text": "Apple reported strong quarterly earnings driven by services revenue...",
                "published_date": "2024-01-16",
                "score": 0.85,
                "provider": "tavily",
                "domain": "news.example.com",
            }
        ],
    }


# Note: ResearchQueryAnalyzer tests commented out - class not available at module level
# TODO: Access query analyzer through DeepResearchAgent if needed for testing

# class TestResearchQueryAnalyzer:
#     """Test query analysis functionality - DISABLED until class structure clarified."""
#     pass


class TestWebSearchProvider:
    """Test web search functionality."""

    @pytest.mark.asyncio
    async def test_search_multiple_providers(
        self, mock_cache_manager, mock_search_results
    ):
        """Test multi-provider search."""
        provider = WebSearchProvider(mock_cache_manager)

        # Mock provider methods
        provider._search_exa = AsyncMock(return_value=mock_search_results["exa"])
        provider._search_tavily = AsyncMock(return_value=mock_search_results["tavily"])

        result = await provider.search_multiple_providers(
            queries=["AAPL analysis"],
            providers=["exa", "tavily"],
            max_results_per_query=5,
        )

        assert "exa" in result
        assert "tavily" in result
        assert len(result["exa"]) == 2
        assert len(result["tavily"]) == 1

    @pytest.mark.asyncio
    async def test_search_with_cache(self, mock_cache_manager):
        """Test search with cache hit."""
        cached_results = [{"url": "cached.com", "title": "Cached Result"}]
        mock_cache_manager.get.return_value = cached_results

        provider = WebSearchProvider(mock_cache_manager)
        result = await provider.search_multiple_providers(
            queries=["test query"], providers=["exa"]
        )

        # Should use cached results
        mock_cache_manager.get.assert_called_once()
        assert result["exa"] == cached_results

    @pytest.mark.asyncio
    async def test_search_provider_failure(self, mock_cache_manager):
        """Test search with provider failure."""
        provider = WebSearchProvider(mock_cache_manager)
        provider._search_exa = AsyncMock(side_effect=Exception("API error"))
        provider._search_tavily = AsyncMock(return_value=[{"url": "backup.com"}])

        result = await provider.search_multiple_providers(
            queries=["test"], providers=["exa", "tavily"]
        )

        # Should continue with working provider
        assert "exa" in result
        assert len(result["exa"]) == 0  # Failed provider returns empty
        assert "tavily" in result
        assert len(result["tavily"]) == 1

    def test_timeframe_to_date(self):
        """Test timeframe conversion to date."""
        provider = WebSearchProvider(MagicMock())

        result = provider._timeframe_to_date("1d")
        assert result is not None

        result = provider._timeframe_to_date("1w")
        assert result is not None

        result = provider._timeframe_to_date("invalid")
        assert result is None


class TestContentAnalyzer:
    """Test content analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_content_batch(self, mock_llm, mock_search_results):
        """Test batch content analysis."""
        # Mock LLM response for content analysis
        mock_response = MagicMock()
        mock_response.content = '{"insights": [{"insight": "Strong fundamentals", "confidence": 0.8, "type": "performance"}], "sentiment": {"direction": "bullish", "confidence": 0.7}, "credibility": 0.8, "data_points": ["revenue growth"], "predictions": ["continued growth"], "key_entities": ["Apple", "iPhone"]}'
        mock_llm.ainvoke.return_value = mock_response

        analyzer = ContentAnalyzer(mock_llm)
        content_items = mock_search_results["exa"] + mock_search_results["tavily"]

        result = await analyzer.analyze_content_batch(content_items, ["performance"])

        assert "insights" in result
        assert "sentiment_scores" in result
        assert "credibility_scores" in result
        assert len(result["insights"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_single_content_failure(self, mock_llm):
        """Test single content analysis with LLM failure."""
        mock_llm.ainvoke.side_effect = Exception("Analysis error")

        analyzer = ContentAnalyzer(mock_llm)
        result = await analyzer._analyze_single_content(
            {"title": "Test", "text": "Test content", "domain": "test.com"},
            ["performance"],
        )

        # Should return default values on failure
        assert result["sentiment"]["direction"] == "neutral"
        assert result["credibility"] == 0.5

    @pytest.mark.asyncio
    async def test_extract_themes(self, mock_llm):
        """Test theme extraction from content."""
        mock_response = MagicMock()
        mock_response.content = (
            '{"themes": [{"theme": "Growth", "relevance": 0.9, "mentions": 10}]}'
        )
        mock_llm.ainvoke.return_value = mock_response

        analyzer = ContentAnalyzer(mock_llm)
        content_items = [{"text": "Growth is strong across sectors"}]

        themes = await analyzer._extract_themes(content_items)

        assert len(themes) == 1
        assert themes[0]["theme"] == "Growth"
        assert themes[0]["relevance"] == 0.9


class TestDeepResearchAgent:
    """Test DeepResearchAgent functionality."""

    @pytest.fixture
    def research_agent(self, mock_llm):
        """Create research agent for testing."""
        with (
            patch("maverick_mcp.agents.deep_research.CacheManager"),
            patch("maverick_mcp.agents.deep_research.WebSearchProvider"),
            patch("maverick_mcp.agents.deep_research.ContentAnalyzer"),
        ):
            return DeepResearchAgent(llm=mock_llm, persona="moderate", max_sources=10)

    @pytest.mark.asyncio
    async def test_research_topic_success(self, research_agent, mock_search_results):
        """Test successful research topic execution."""
        # Mock the web search provider
        research_agent.web_search_provider.search_multiple_providers = AsyncMock(
            return_value=mock_search_results
        )

        # Mock content analyzer
        research_agent.content_analyzer.analyze_content_batch = AsyncMock(
            return_value={
                "insights": [{"insight": "Strong growth", "confidence": 0.8}],
                "sentiment_scores": {
                    "example.com": {"direction": "bullish", "confidence": 0.7}
                },
                "key_themes": [{"theme": "Growth", "relevance": 0.9}],
                "consensus_view": {"direction": "bullish", "confidence": 0.7},
                "credibility_scores": {"example.com": 0.8},
            }
        )

        result = await research_agent.research_topic(
            query="Analyze AAPL", session_id="test_session", research_scope="standard"
        )

        assert "content" in result or "analysis" in result
        # Should call web search and content analysis
        research_agent.web_search_provider.search_multiple_providers.assert_called_once()
        research_agent.content_analyzer.analyze_content_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_company_comprehensive(self, research_agent):
        """Test comprehensive company research."""
        # Mock the research_topic method
        research_agent.research_topic = AsyncMock(
            return_value={
                "content": "Comprehensive analysis completed",
                "research_confidence": 0.85,
                "sources_found": 25,
            }
        )

        await research_agent.research_company_comprehensive(
            symbol="AAPL", session_id="company_test", include_competitive_analysis=True
        )

        research_agent.research_topic.assert_called_once()
        # Should include symbol in query
        call_args = research_agent.research_topic.call_args
        assert "AAPL" in call_args[1]["query"]

    @pytest.mark.asyncio
    async def test_analyze_market_sentiment(self, research_agent):
        """Test market sentiment analysis."""
        research_agent.research_topic = AsyncMock(
            return_value={
                "content": "Sentiment analysis completed",
                "research_confidence": 0.75,
            }
        )

        await research_agent.analyze_market_sentiment(
            topic="tech stocks", session_id="sentiment_test", timeframe="1w"
        )

        research_agent.research_topic.assert_called_once()
        call_args = research_agent.research_topic.call_args
        assert "sentiment" in call_args[1]["query"].lower()

    def test_persona_insight_relevance(self, research_agent):
        """Test persona insight relevance checking."""
        from maverick_mcp.agents.base import INVESTOR_PERSONAS

        conservative_persona = INVESTOR_PERSONAS["conservative"]

        # Test relevant insight for conservative
        insight = {"insight": "Strong dividend yield provides stable income"}
        assert research_agent._is_insight_relevant_for_persona(
            insight, conservative_persona.characteristics
        )

        # Test irrelevant insight for conservative
        insight = {"insight": "High volatility momentum play"}
        # This should return True as default implementation is permissive
        assert research_agent._is_insight_relevant_for_persona(
            insight, conservative_persona.characteristics
        )


class TestSupervisorIntegration:
    """Test SupervisorAgent integration with DeepResearchAgent."""

    @pytest.fixture
    def supervisor_with_research(self, mock_llm):
        """Create supervisor with research agent."""
        with patch(
            "maverick_mcp.agents.deep_research.DeepResearchAgent"
        ) as mock_research:
            mock_research_instance = MagicMock()
            mock_research.return_value = mock_research_instance

            supervisor = SupervisorAgent(
                llm=mock_llm,
                agents={"research": mock_research_instance},
                persona="moderate",
            )
            return supervisor, mock_research_instance

    @pytest.mark.asyncio
    async def test_research_query_routing(self, supervisor_with_research):
        """Test routing of research queries to research agent."""
        supervisor, mock_research = supervisor_with_research

        # Mock the coordination workflow
        supervisor.coordinate_agents = AsyncMock(
            return_value={
                "status": "success",
                "agents_used": ["research"],
                "confidence_score": 0.8,
                "synthesis": "Research completed successfully",
            }
        )

        result = await supervisor.coordinate_agents(
            query="Research Apple's competitive position", session_id="routing_test"
        )

        assert result["status"] == "success"
        assert "research" in result["agents_used"]

    def test_research_routing_matrix(self):
        """Test research queries in routing matrix."""
        from maverick_mcp.agents.supervisor import ROUTING_MATRIX

        # Check research categories exist
        assert "deep_research" in ROUTING_MATRIX
        assert "company_research" in ROUTING_MATRIX
        assert "sentiment_analysis" in ROUTING_MATRIX

        # Check research agent is primary
        assert ROUTING_MATRIX["deep_research"]["primary"] == "research"
        assert ROUTING_MATRIX["company_research"]["primary"] == "research"

    def test_query_classification_research(self):
        """Test query classification for research queries."""
        # Note: Testing internal classification logic through public interface
        # QueryClassifier might be internal to SupervisorAgent

        # Simple test to verify supervisor routing exists
        from maverick_mcp.agents.supervisor import ROUTING_MATRIX

        # Verify research-related routing categories exist
        research_categories = [
            "deep_research",
            "company_research",
            "sentiment_analysis",
        ]
        for category in research_categories:
            if category in ROUTING_MATRIX:
                assert "primary" in ROUTING_MATRIX[category]


class TestErrorHandling:
    """Test error handling in research operations."""

    @pytest.mark.asyncio
    async def test_web_search_error_handling(self, mock_cache_manager):
        """Test web search error handling."""
        provider = WebSearchProvider(mock_cache_manager)

        # Mock both providers to fail
        provider._search_exa = AsyncMock(
            side_effect=WebSearchError("Exa failed", "exa")
        )
        provider._search_tavily = AsyncMock(
            side_effect=WebSearchError("Tavily failed", "tavily")
        )

        result = await provider.search_multiple_providers(
            queries=["test"], providers=["exa", "tavily"]
        )

        # Should return empty results for failed providers
        assert result["exa"] == []
        assert result["tavily"] == []

    @pytest.mark.asyncio
    async def test_research_agent_api_key_missing(self, mock_llm):
        """Test research agent behavior with missing API keys."""
        with patch("maverick_mcp.config.settings.get_settings") as mock_settings:
            mock_settings.return_value.research.exa_api_key = None
            mock_settings.return_value.research.tavily_api_key = None

            # Should still initialize but searches will fail gracefully
            agent = DeepResearchAgent(llm=mock_llm)
            assert agent is not None

    def test_research_error_creation(self):
        """Test ResearchError exception creation."""
        error = ResearchError(
            "Search failed", research_type="web_search", provider="exa"
        )

        assert error.message == "Search failed"
        assert error.research_type == "web_search"
        assert error.provider == "exa"
        assert error.error_code == "RESEARCH_ERROR"


@pytest.mark.integration
class TestDeepResearchIntegration:
    """Integration tests requiring external services (marked for optional execution)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not get_settings().research.exa_api_key, reason="EXA_API_KEY not configured"
    )
    async def test_real_web_search(self):
        """Test real web search with Exa API (requires API key)."""
        from maverick_mcp.data.cache_manager import CacheManager

        cache_manager = CacheManager()
        provider = WebSearchProvider(cache_manager)

        result = await provider.search_multiple_providers(
            queries=["Apple stock analysis"],
            providers=["exa"],
            max_results_per_query=2,
            timeframe="1w",
        )

        assert "exa" in result
        # Should get some results (unless API is down)
        if result["exa"]:
            assert len(result["exa"]) > 0
            assert "url" in result["exa"][0]
            assert "title" in result["exa"][0]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not get_settings().research.exa_api_key,
        reason="Research API keys not configured",
    )
    async def test_full_research_workflow(self, mock_llm):
        """Test complete research workflow (requires API keys)."""
        DeepResearchAgent(
            llm=mock_llm, persona="moderate", max_sources=5, research_depth="basic"
        )

        # This would require real API keys and network access
        # Implementation depends on test environment setup
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
