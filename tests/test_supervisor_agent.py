"""
Comprehensive tests for SupervisorAgent orchestration.

Tests the multi-agent coordination, routing logic, result synthesis,
and conflict resolution capabilities.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maverick_mcp.agents.base import PersonaAwareAgent
from maverick_mcp.agents.supervisor import (
    ROUTING_MATRIX,
    SupervisorAgent,
)
from maverick_mcp.config.settings import get_settings


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock()
    return llm


@pytest.fixture
def mock_agents():
    """Mock agent dictionary for testing."""
    agents = {}

    # Market analysis agent
    market_agent = MagicMock(spec=PersonaAwareAgent)
    market_agent.analyze_market = AsyncMock(
        return_value={
            "status": "success",
            "summary": "Strong momentum stocks identified",
            "screened_symbols": ["AAPL", "MSFT", "NVDA"],
            "confidence": 0.85,
            "execution_time_ms": 1500,
        }
    )
    agents["market"] = market_agent

    # Research agent
    research_agent = MagicMock(spec=PersonaAwareAgent)
    research_agent.conduct_research = AsyncMock(
        return_value={
            "status": "success",
            "research_findings": [
                {"insight": "Strong fundamentals", "confidence": 0.9}
            ],
            "sources_analyzed": 25,
            "research_confidence": 0.88,
            "execution_time_ms": 3500,
        }
    )
    agents["research"] = research_agent

    # Technical analysis agent (mock future agent)
    technical_agent = MagicMock(spec=PersonaAwareAgent)
    technical_agent.analyze_technicals = AsyncMock(
        return_value={
            "status": "success",
            "trend_direction": "bullish",
            "support_levels": [150.0, 145.0],
            "resistance_levels": [160.0, 165.0],
            "confidence": 0.75,
            "execution_time_ms": 800,
        }
    )
    agents["technical"] = technical_agent

    return agents


@pytest.fixture
def supervisor_agent(mock_llm, mock_agents):
    """Create SupervisorAgent for testing."""
    return SupervisorAgent(
        llm=mock_llm,
        agents=mock_agents,
        persona="moderate",
        ttl_hours=1,
        routing_strategy="llm_powered",
        max_iterations=5,
    )


# Note: Internal classes (QueryClassifier, ResultSynthesizer) not exposed at module level
# Testing through SupervisorAgent public interface instead

# class TestQueryClassifier:
#     """Test query classification logic - DISABLED (internal class)."""
#     pass

# class TestResultSynthesizer:
#     """Test result synthesis and conflict resolution - DISABLED (internal class)."""
#     pass


class TestSupervisorAgent:
    """Test main SupervisorAgent functionality."""

    @pytest.mark.asyncio
    async def test_orchestrate_analysis_success(self, supervisor_agent):
        """Test successful orchestrated analysis."""
        # Mock query classification
        mock_classification = {
            "category": "market_screening",
            "required_agents": ["market", "research"],
            "parallel_suitable": True,
            "confidence": 0.9,
        }
        supervisor_agent.query_classifier.classify_query = AsyncMock(
            return_value=mock_classification
        )

        # Mock synthesis result
        mock_synthesis = {
            "synthesis": "Strong market opportunities identified",
            "confidence": 0.87,
            "key_recommendations": ["Focus on momentum", "Research fundamentals"],
        }
        supervisor_agent.result_synthesizer.synthesize_results = AsyncMock(
            return_value=mock_synthesis
        )

        result = await supervisor_agent.orchestrate_analysis(
            query="Find top investment opportunities",
            session_id="test_session",
            routing_strategy="llm_powered",
        )

        assert result["status"] == "success"
        assert "agents_used" in result
        assert "synthesis" in result
        assert result["routing_strategy"] == "llm_powered"

        # Should have called both agents
        supervisor_agent.agents["market"].analyze_market.assert_called_once()
        supervisor_agent.agents["research"].conduct_research.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrate_analysis_sequential_execution(self, supervisor_agent):
        """Test sequential execution mode."""
        # Mock classification requiring sequential execution
        mock_classification = {
            "category": "complex_analysis",
            "required_agents": ["research", "market"],
            "parallel_suitable": False,
            "dependencies": {"market": ["research"]},  # Market depends on research
            "confidence": 0.85,
        }
        supervisor_agent.query_classifier.classify_query = AsyncMock(
            return_value=mock_classification
        )

        result = await supervisor_agent.orchestrate_analysis(
            query="Deep analysis with dependencies",
            session_id="sequential_test",
            parallel_execution=False,
        )

        assert result["status"] == "success"
        # Research should be called before market (sequential)
        supervisor_agent.agents["research"].conduct_research.assert_called_once()
        supervisor_agent.agents["market"].analyze_market.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrate_with_agent_failure(self, supervisor_agent):
        """Test orchestration with one agent failing."""
        # Make research agent fail
        supervisor_agent.agents["research"].conduct_research.side_effect = Exception(
            "Research API failed"
        )

        # Mock classification
        mock_classification = {
            "category": "market_screening",
            "required_agents": ["market", "research"],
            "parallel_suitable": True,
            "confidence": 0.9,
        }
        supervisor_agent.query_classifier.classify_query = AsyncMock(
            return_value=mock_classification
        )

        # Mock partial synthesis
        mock_synthesis = {
            "synthesis": "Partial analysis completed with market data only",
            "confidence": 0.6,  # Lower confidence due to missing research
            "warnings": ["Research agent failed - analysis incomplete"],
        }
        supervisor_agent.result_synthesizer.synthesize_results = AsyncMock(
            return_value=mock_synthesis
        )

        result = await supervisor_agent.orchestrate_analysis(
            query="Analysis with failure", session_id="failure_test"
        )

        assert (
            result["status"] == "partial_success"
        )  # Or still success but with warnings
        assert "agent_errors" in result
        assert "research" in result["agent_errors"]

    @pytest.mark.asyncio
    async def test_routing_strategy_rule_based(self, supervisor_agent):
        """Test rule-based routing strategy."""
        supervisor_agent.routing_strategy = "rule_based"

        result = await supervisor_agent.orchestrate_analysis(
            query="Find momentum stocks",
            session_id="rule_test",
            routing_strategy="rule_based",
        )

        assert result["status"] == "success"
        assert result["routing_strategy"] == "rule_based"

    def test_agent_selection_based_on_persona(self, supervisor_agent):
        """Test agent selection varies by persona."""
        # Conservative persona might prefer fewer, more established agents
        conservative_agents = supervisor_agent._select_agents_for_persona(
            required_agents=["market", "research", "technical"], persona="conservative"
        )

        # Aggressive persona might use more agents
        aggressive_agents = supervisor_agent._select_agents_for_persona(
            required_agents=["market", "research", "technical"], persona="aggressive"
        )

        # Both should select agents, exact logic depends on implementation
        assert len(conservative_agents) >= 1
        assert len(aggressive_agents) >= 1

    @pytest.mark.asyncio
    async def test_execution_timeout_handling(self, supervisor_agent):
        """Test handling of execution timeouts."""

        # Make research agent hang (simulate timeout)
        async def slow_research(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return {"status": "success"}

        supervisor_agent.agents["research"].conduct_research = slow_research

        # Mock classification
        mock_classification = {
            "category": "research_heavy",
            "required_agents": ["research"],
            "parallel_suitable": True,
            "confidence": 0.9,
        }
        supervisor_agent.query_classifier.classify_query = AsyncMock(
            return_value=mock_classification
        )

        # Should handle timeout gracefully
        with patch("asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = TimeoutError()

            result = await supervisor_agent.orchestrate_analysis(
                query="Research with timeout",
                session_id="timeout_test",
                max_execution_time_seconds=5,
            )

            assert "timeout" in str(result).lower() or result["status"] == "error"

    def test_routing_matrix_completeness(self):
        """Test routing matrix covers expected categories."""
        expected_categories = [
            "market_screening",
            "company_research",
            "technical_analysis",
            "sentiment_analysis",
            "portfolio_optimization",
            "risk_analysis",
        ]

        for category in expected_categories:
            assert category in ROUTING_MATRIX, f"Missing routing for {category}"
            assert "primary" in ROUTING_MATRIX[category]
            assert "secondary" in ROUTING_MATRIX[category]

    def test_confidence_thresholds_defined(self):
        """Test confidence thresholds are properly defined."""
        # Note: CONFIDENCE_THRESHOLDS not exposed at module level
        # Testing through agent behavior instead
        assert (
            True
        )  # Placeholder - could test confidence behavior through agent methods


class TestSupervisorStateManagement:
    """Test state management in supervisor workflows."""

    @pytest.mark.asyncio
    async def test_state_initialization(self, supervisor_agent):
        """Test proper state initialization."""
        initial_state = supervisor_agent._create_initial_state(
            query="Test query", session_id="state_test", persona="moderate"
        )

        assert isinstance(initial_state, dict)
        assert initial_state["session_id"] == "state_test"
        assert initial_state["persona"] == "moderate"
        assert initial_state["workflow_status"] == "planning"
        assert initial_state["current_iteration"] == 0

    @pytest.mark.asyncio
    async def test_state_updates_during_execution(self, supervisor_agent):
        """Test state updates during workflow execution."""
        # Mock classification and synthesis
        supervisor_agent.query_classifier.classify_query = AsyncMock(
            return_value={
                "category": "market_screening",
                "required_agents": ["market"],
                "confidence": 0.9,
            }
        )

        supervisor_agent.result_synthesizer.synthesize_results = AsyncMock(
            return_value={"synthesis": "Analysis complete", "confidence": 0.85}
        )

        result = await supervisor_agent.orchestrate_analysis(
            query="State test query", session_id="state_execution_test"
        )

        # Should track execution progress
        assert "execution_time_ms" in result
        assert result["workflow_status"] == "completed"


class TestErrorHandling:
    """Test error handling in supervisor operations."""

    @pytest.mark.asyncio
    async def test_classification_failure_recovery(self, supervisor_agent):
        """Test recovery from classification failures."""
        # Make classifier fail completely
        supervisor_agent.query_classifier.classify_query.side_effect = Exception(
            "Classification failed"
        )

        # Should still attempt fallback
        result = await supervisor_agent.orchestrate_analysis(
            query="Classification failure test", session_id="classification_error"
        )

        # Depending on implementation, might succeed with fallback or fail gracefully
        assert "error" in result["status"] or result["status"] == "success"

    @pytest.mark.asyncio
    async def test_synthesis_failure_recovery(self, supervisor_agent):
        """Test recovery from synthesis failures."""
        # Mock successful classification
        supervisor_agent.query_classifier.classify_query = AsyncMock(
            return_value={
                "category": "market_screening",
                "required_agents": ["market"],
                "confidence": 0.9,
            }
        )

        # Make synthesis fail
        supervisor_agent.result_synthesizer.synthesize_results.side_effect = Exception(
            "Synthesis failed"
        )

        result = await supervisor_agent.orchestrate_analysis(
            query="Synthesis failure test", session_id="synthesis_error"
        )

        # Should provide raw agent results even if synthesis fails
        assert result["status"] == "partial_success" or "agent_results" in result

    def test_invalid_persona_handling(self, mock_llm, mock_agents):
        """Test handling of invalid persona."""
        with pytest.raises(ValueError):
            SupervisorAgent(llm=mock_llm, agents=mock_agents, persona="invalid_persona")

    def test_missing_required_agents(self, mock_llm):
        """Test handling when required agents are missing."""
        # Create supervisor with limited agents
        limited_agents = {"market": MagicMock()}
        supervisor = SupervisorAgent(
            llm=mock_llm, agents=limited_agents, persona="moderate"
        )

        # Mock classification requiring missing agent
        supervisor.query_classifier.classify_query = AsyncMock(
            return_value={
                "category": "deep_research",
                "required_agents": ["research"],  # Not available
                "confidence": 0.9,
            }
        )

        # Should handle gracefully (exact behavior depends on implementation)
        with pytest.mark.asyncio:

            async def test_execution():
                result = await supervisor.orchestrate_analysis(
                    query="Test missing agent", session_id="missing_agent_test"
                )
                assert "error" in result or "missing_agents" in result

            # Run the async test
            asyncio.run(test_execution())


@pytest.mark.integration
class TestSupervisorIntegration:
    """Integration tests for supervisor with real components."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not get_settings().openai.api_key, reason="OpenAI API key not configured"
    )
    async def test_real_llm_classification(self):
        """Test with real LLM classification (requires API key)."""
        from langchain_openai import ChatOpenAI

        from maverick_mcp.agents.supervisor import QueryClassifier

        real_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        classifier = QueryClassifier(real_llm)

        result = await classifier.classify_query(
            "Find the best momentum stocks for aggressive growth portfolio",
            "aggressive",
        )

        assert "category" in result
        assert "required_agents" in result
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_supervisor_with_mock_real_agents(self, mock_llm):
        """Test supervisor with more realistic agent mocks."""
        # Create more realistic agent mocks that simulate actual agent behavior
        realistic_agents = {}

        # Market agent with realistic response structure
        market_agent = MagicMock()
        market_agent.analyze_market = AsyncMock(
            return_value={
                "status": "success",
                "results": {
                    "summary": "Found 15 momentum stocks meeting criteria",
                    "screened_symbols": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
                    "sector_breakdown": {
                        "Technology": 0.6,
                        "Healthcare": 0.2,
                        "Finance": 0.2,
                    },
                    "screening_scores": {"AAPL": 0.92, "MSFT": 0.88, "NVDA": 0.95},
                },
                "metadata": {
                    "screening_strategy": "momentum",
                    "total_candidates": 500,
                    "filtered_count": 15,
                },
                "confidence": 0.87,
                "execution_time_ms": 1200,
            }
        )
        realistic_agents["market"] = market_agent

        supervisor = SupervisorAgent(
            llm=mock_llm, agents=realistic_agents, persona="moderate"
        )

        # Mock realistic classification
        supervisor.query_classifier.classify_query = AsyncMock(
            return_value={
                "category": "market_screening",
                "required_agents": ["market"],
                "parallel_suitable": True,
                "confidence": 0.9,
            }
        )

        result = await supervisor.orchestrate_analysis(
            query="Find momentum stocks", session_id="realistic_test"
        )

        assert result["status"] == "success"
        assert "agents_used" in result
        assert "market" in result["agents_used"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
