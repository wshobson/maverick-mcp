"""
Tests for the new MCP tools in the agents router.

Tests the orchestrated_analysis, deep_research_financial, and
compare_multi_agent_analysis MCP tools for Claude Desktop integration.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maverick_mcp.api.routers.agents import (
    compare_multi_agent_analysis,
    deep_research_financial,
    get_or_create_agent,
    list_available_agents,
    orchestrated_analysis,
)


@pytest.fixture
def mock_supervisor_agent():
    """Mock SupervisorAgent for testing."""
    agent = MagicMock()
    agent.orchestrate_analysis = AsyncMock(
        return_value={
            "status": "success",
            "synthesis": "Comprehensive analysis completed",
            "agents_used": ["market", "research"],
            "execution_time_ms": 4500,
            "synthesis_confidence": 0.87,
            "agent_results": {
                "market": {"summary": "Strong momentum", "confidence": 0.85},
                "research": {"summary": "Solid fundamentals", "confidence": 0.88},
            },
            "key_recommendations": ["Focus on large-cap tech", "Monitor earnings"],
            "confidence": 0.87,
        }
    )
    return agent


@pytest.fixture
def mock_research_agent():
    """Mock DeepResearchAgent for testing."""
    agent = MagicMock()
    agent.conduct_research = AsyncMock(
        return_value={
            "status": "success",
            "research_findings": [
                {
                    "insight": "Strong revenue growth",
                    "confidence": 0.9,
                    "source": "earnings-report",
                },
                {
                    "insight": "Expanding market share",
                    "confidence": 0.85,
                    "source": "market-analysis",
                },
            ],
            "sources_analyzed": 42,
            "research_confidence": 0.88,
            "validation_checks_passed": 35,
            "total_sources_processed": 50,
            "content_summaries": [
                "Financial performance strong",
                "Market position improving",
            ],
            "citations": [
                {"url": "https://example.com/report1", "title": "Q3 Earnings Analysis"},
                {"url": "https://example.com/report2", "title": "Market Share Report"},
            ],
            "execution_time_ms": 6500,
        }
    )
    return agent


@pytest.fixture
def mock_market_agent():
    """Mock MarketAnalysisAgent for testing."""
    agent = MagicMock()
    agent.analyze_market = AsyncMock(
        return_value={
            "status": "success",
            "summary": "Top momentum stocks identified",
            "screened_symbols": ["AAPL", "MSFT", "NVDA"],
            "confidence": 0.82,
            "results": {
                "screening_scores": {"AAPL": 0.92, "MSFT": 0.88, "NVDA": 0.95},
                "sector_performance": {"Technology": 0.15, "Healthcare": 0.08},
            },
            "execution_time_ms": 2100,
        }
    )
    return agent


class TestOrchestratedAnalysis:
    """Test orchestrated_analysis MCP tool."""

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_success(self, mock_supervisor_agent):
        """Test successful orchestrated analysis."""
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(
                query="Analyze tech sector opportunities",
                persona="moderate",
                routing_strategy="llm_powered",
                max_agents=3,
                parallel_execution=True,
            )

            assert result["status"] == "success"
            assert result["agent_type"] == "supervisor_orchestrated"
            assert result["persona"] == "moderate"
            assert result["routing_strategy"] == "llm_powered"
            assert "agents_used" in result
            assert "synthesis_confidence" in result
            assert "execution_time_ms" in result

            mock_supervisor_agent.orchestrate_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_with_session_id(self, mock_supervisor_agent):
        """Test orchestrated analysis with provided session ID."""
        session_id = "test-session-123"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(
                query="Market analysis", session_id=session_id
            )

            assert result["session_id"] == session_id
            call_args = mock_supervisor_agent.orchestrate_analysis.call_args
            assert call_args[1]["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_generates_session_id(
        self, mock_supervisor_agent
    ):
        """Test orchestrated analysis generates session ID when not provided."""
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(query="Market analysis")

            assert "session_id" in result
            # Should be a valid UUID format
            uuid.UUID(result["session_id"])

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_error_handling(self, mock_supervisor_agent):
        """Test orchestrated analysis error handling."""
        mock_supervisor_agent.orchestrate_analysis.side_effect = Exception(
            "Orchestration failed"
        )

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(query="Test error handling")

            assert result["status"] == "error"
            assert result["agent_type"] == "supervisor_orchestrated"
            assert "error" in result
            assert "Orchestration failed" in result["error"]

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_persona_variations(
        self, mock_supervisor_agent
    ):
        """Test orchestrated analysis with different personas."""
        personas = ["conservative", "moderate", "aggressive", "day_trader"]

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            for persona in personas:
                result = await orchestrated_analysis(
                    query="Test persona", persona=persona
                )

                assert result["persona"] == persona
                # Check agent was created with correct persona
                call_args = mock_supervisor_agent.orchestrate_analysis.call_args
                assert call_args is not None


class TestDeepResearchFinancial:
    """Test deep_research_financial MCP tool."""

    @pytest.mark.asyncio
    async def test_deep_research_success(self, mock_research_agent):
        """Test successful deep research."""
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(
                research_topic="AAPL competitive analysis",
                persona="moderate",
                research_depth="comprehensive",
                focus_areas=["fundamentals", "competitive_landscape"],
                timeframe="90d",
            )

            assert result["status"] == "success"
            assert result["agent_type"] == "deep_research"
            assert result["research_topic"] == "AAPL competitive analysis"
            assert result["research_depth"] == "comprehensive"
            assert "fundamentals" in result["focus_areas"]
            assert "competitive_landscape" in result["focus_areas"]
            assert result["sources_analyzed"] == 42
            assert result["research_confidence"] == 0.88

            mock_research_agent.conduct_research.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_research_default_focus_areas(self, mock_research_agent):
        """Test deep research with default focus areas."""
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(
                research_topic="Tesla analysis",
                focus_areas=None,  # Should use defaults
            )

            expected_defaults = [
                "fundamentals",
                "market_sentiment",
                "competitive_landscape",
            ]
            assert result["focus_areas"] == expected_defaults

            call_args = mock_research_agent.conduct_research.call_args
            assert call_args[1]["focus_areas"] == expected_defaults

    @pytest.mark.asyncio
    async def test_deep_research_depth_variations(self, mock_research_agent):
        """Test deep research with different depth levels."""
        depth_levels = ["basic", "standard", "comprehensive", "exhaustive"]

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            for depth in depth_levels:
                result = await deep_research_financial(
                    research_topic="Test research", research_depth=depth
                )

                assert result["research_depth"] == depth

    @pytest.mark.asyncio
    async def test_deep_research_error_handling(self, mock_research_agent):
        """Test deep research error handling."""
        mock_research_agent.conduct_research.side_effect = Exception(
            "Research API failed"
        )

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(research_topic="Error test")

            assert result["status"] == "error"
            assert result["agent_type"] == "deep_research"
            assert "Research API failed" in result["error"]

    @pytest.mark.asyncio
    async def test_deep_research_timeframe_handling(self, mock_research_agent):
        """Test deep research with different timeframes."""
        timeframes = ["7d", "30d", "90d", "1y"]

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            for timeframe in timeframes:
                await deep_research_financial(
                    research_topic="Timeframe test", timeframe=timeframe
                )

                call_args = mock_research_agent.conduct_research.call_args
                assert call_args[1]["timeframe"] == timeframe


class TestCompareMultiAgentAnalysis:
    """Test compare_multi_agent_analysis MCP tool."""

    @pytest.mark.asyncio
    async def test_compare_agents_success(
        self, mock_market_agent, mock_supervisor_agent
    ):
        """Test successful multi-agent comparison."""

        def get_agent_mock(agent_type, persona):
            if agent_type == "market":
                return mock_market_agent
            elif agent_type == "supervisor":
                return mock_supervisor_agent
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=get_agent_mock,
        ):
            result = await compare_multi_agent_analysis(
                query="Analyze NVDA stock potential",
                agent_types=["market", "supervisor"],
                persona="moderate",
            )

            assert result["status"] == "success"
            assert result["persona"] == "moderate"
            assert "comparison" in result
            assert "market" in result["comparison"]
            assert "supervisor" in result["comparison"]
            assert "execution_times_ms" in result

            # Both agents should have been called
            mock_market_agent.analyze_market.assert_called_once()
            mock_supervisor_agent.orchestrate_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_agents_default_types(
        self, mock_market_agent, mock_supervisor_agent
    ):
        """Test comparison with default agent types."""

        def get_agent_mock(agent_type, persona):
            return (
                mock_market_agent if agent_type == "market" else mock_supervisor_agent
            )

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=get_agent_mock,
        ):
            result = await compare_multi_agent_analysis(
                query="Default comparison test",
                agent_types=None,  # Should use defaults
            )

            # Should use default agent types ["market", "supervisor"]
            assert "market" in result["agents_compared"]
            assert "supervisor" in result["agents_compared"]

    @pytest.mark.asyncio
    async def test_compare_agents_with_failure(
        self, mock_market_agent, mock_supervisor_agent
    ):
        """Test comparison with one agent failing."""
        mock_market_agent.analyze_market.side_effect = Exception("Market agent failed")

        def get_agent_mock(agent_type, persona):
            return (
                mock_market_agent if agent_type == "market" else mock_supervisor_agent
            )

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=get_agent_mock,
        ):
            result = await compare_multi_agent_analysis(
                query="Failure handling test", agent_types=["market", "supervisor"]
            )

            assert result["status"] == "success"  # Overall success
            assert "comparison" in result
            assert "error" in result["comparison"]["market"]
            assert result["comparison"]["market"]["status"] == "failed"
            # Supervisor should still succeed
            assert "summary" in result["comparison"]["supervisor"]

    @pytest.mark.asyncio
    async def test_compare_agents_session_id_handling(
        self, mock_market_agent, mock_supervisor_agent
    ):
        """Test session ID handling in agent comparison."""
        session_id = "compare-test-456"

        def get_agent_mock(agent_type, persona):
            return (
                mock_market_agent if agent_type == "market" else mock_supervisor_agent
            )

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=get_agent_mock,
        ):
            await compare_multi_agent_analysis(
                query="Session ID test", session_id=session_id
            )

            # Check session IDs were properly formatted for each agent
            market_call_args = mock_market_agent.analyze_market.call_args
            assert market_call_args[1]["session_id"] == f"{session_id}_market"

            supervisor_call_args = mock_supervisor_agent.orchestrate_analysis.call_args
            assert supervisor_call_args[1]["session_id"] == f"{session_id}_supervisor"


class TestGetOrCreateAgent:
    """Test agent factory function."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_create_supervisor_agent(self):
        """Test creating supervisor agent."""
        with patch("maverick_mcp.api.routers.agents.SupervisorAgent") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            agent = get_or_create_agent("supervisor", "moderate")

            assert agent == mock_instance
            mock_class.assert_called_once()

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "EXA_API_KEY": "exa-key",
            "TAVILY_API_KEY": "tavily-key",
        },
    )
    def test_create_deep_research_agent_with_api_keys(self):
        """Test creating deep research agent with API keys."""
        with patch("maverick_mcp.api.routers.agents.DeepResearchAgent") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            get_or_create_agent("deep_research", "moderate")

            # Should pass API keys to constructor
            call_args = mock_class.call_args
            assert call_args[1]["exa_api_key"] == "exa-key"
            assert call_args[1]["tavily_api_key"] == "tavily-key"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_create_deep_research_agent_without_api_keys(self):
        """Test creating deep research agent without optional API keys."""
        with patch("maverick_mcp.api.routers.agents.DeepResearchAgent") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            get_or_create_agent("deep_research", "moderate")

            # Should pass None for missing API keys
            call_args = mock_class.call_args
            assert call_args[1]["exa_api_key"] is None
            assert call_args[1]["tavily_api_key"] is None

    def test_agent_caching(self):
        """Test agent instance caching."""
        with patch("maverick_mcp.api.routers.agents.MarketAnalysisAgent") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            # First call should create agent
            agent1 = get_or_create_agent("market", "moderate")
            # Second call should return cached agent
            agent2 = get_or_create_agent("market", "moderate")

            assert agent1 == agent2 == mock_instance
            # Constructor should only be called once due to caching
            mock_class.assert_called_once()

    def test_different_personas_create_different_agents(self):
        """Test different personas create separate cached agents."""
        with patch("maverick_mcp.api.routers.agents.MarketAnalysisAgent") as mock_class:
            mock_class.return_value = MagicMock()

            agent_moderate = get_or_create_agent("market", "moderate")
            agent_aggressive = get_or_create_agent("market", "aggressive")

            # Should create separate instances for different personas
            assert agent_moderate != agent_aggressive
            assert mock_class.call_count == 2

    def test_invalid_agent_type(self):
        """Test handling of invalid agent type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_or_create_agent("invalid_agent_type", "moderate")


class TestListAvailableAgents:
    """Test list_available_agents MCP tool."""

    def test_list_available_agents_structure(self):
        """Test the structure of available agents list."""
        result = list_available_agents()

        assert result["status"] == "success"
        assert "agents" in result
        assert "orchestrated_tools" in result
        assert "features" in result

    def test_active_agents_listed(self):
        """Test that active agents are properly listed."""
        result = list_available_agents()
        agents = result["agents"]

        # Check new orchestrated agents
        assert "supervisor_orchestrated" in agents
        assert agents["supervisor_orchestrated"]["status"] == "active"
        assert (
            "Multi-agent orchestration"
            in agents["supervisor_orchestrated"]["description"]
        )

        assert "deep_research" in agents
        assert agents["deep_research"]["status"] == "active"
        assert (
            "comprehensive financial research"
            in agents["deep_research"]["description"].lower()
        )

    def test_orchestrated_tools_listed(self):
        """Test that orchestrated tools are listed."""
        result = list_available_agents()
        tools = result["orchestrated_tools"]

        assert "orchestrated_analysis" in tools
        assert "deep_research_financial" in tools
        assert "compare_multi_agent_analysis" in tools

    def test_personas_supported(self):
        """Test that all personas are supported."""
        result = list_available_agents()

        expected_personas = ["conservative", "moderate", "aggressive", "day_trader"]

        # Check supervisor agent supports all personas
        supervisor_personas = result["agents"]["supervisor_orchestrated"]["personas"]
        assert all(persona in supervisor_personas for persona in expected_personas)

        # Check research agent supports all personas
        research_personas = result["agents"]["deep_research"]["personas"]
        assert all(persona in research_personas for persona in expected_personas)

    def test_capabilities_documented(self):
        """Test that agent capabilities are documented."""
        result = list_available_agents()
        agents = result["agents"]

        # Supervisor capabilities
        supervisor_caps = agents["supervisor_orchestrated"]["capabilities"]
        assert "Intelligent query routing" in supervisor_caps
        assert "Multi-agent coordination" in supervisor_caps

        # Research capabilities
        research_caps = agents["deep_research"]["capabilities"]
        assert "Multi-provider web search" in research_caps
        assert "AI-powered content analysis" in research_caps

    def test_new_features_documented(self):
        """Test that new orchestration features are documented."""
        result = list_available_agents()
        features = result["features"]

        assert "multi_agent_orchestration" in features
        assert "web_search_research" in features
        assert "intelligent_routing" in features


@pytest.mark.integration
class TestAgentRouterIntegration:
    """Integration tests for agent router MCP tools."""

    @pytest.mark.asyncio
    async def test_end_to_end_orchestrated_workflow(self):
        """Test complete orchestrated analysis workflow."""
        # This would be a full integration test with real or more sophisticated mocks
        # Testing the complete flow: query -> classification -> agent execution -> synthesis
        pass

    @pytest.mark.asyncio
    async def test_research_agent_with_supervisor_integration(self):
        """Test research agent working with supervisor."""
        # Test how research agent integrates with supervisor routing
        pass

    @pytest.mark.asyncio
    async def test_error_propagation_across_agents(self):
        """Test how errors propagate through the orchestration system."""
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
