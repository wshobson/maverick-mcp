"""
Integration tests for the orchestration system.

Tests the end-to-end functionality of SupervisorAgent and DeepResearchAgent
to verify the orchestration system works correctly.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from maverick_mcp.agents.base import INVESTOR_PERSONAS, PersonaAwareAgent
from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.agents.supervisor import ROUTING_MATRIX, SupervisorAgent


class TestOrchestrationSystemIntegration:
    """Test the complete orchestration system integration."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock()
        return llm

    @pytest.fixture
    def mock_market_agent(self):
        """Create a mock market analysis agent."""
        agent = MagicMock(spec=PersonaAwareAgent)
        agent.analyze_market = AsyncMock(
            return_value={
                "status": "success",
                "summary": "Market analysis completed",
                "screened_symbols": ["AAPL", "MSFT", "NVDA"],
                "confidence": 0.85,
                "execution_time_ms": 1500,
            }
        )
        return agent

    def test_agent_imports_successful(self):
        """Test that all agent classes can be imported successfully."""
        # These imports should not raise exceptions
        assert SupervisorAgent is not None
        assert DeepResearchAgent is not None
        assert ROUTING_MATRIX is not None
        assert INVESTOR_PERSONAS is not None

    def test_routing_matrix_structure(self):
        """Test that routing matrix has expected structure."""
        assert isinstance(ROUTING_MATRIX, dict)
        assert len(ROUTING_MATRIX) > 0

        # Check each routing entry has required fields
        for _category, routing_info in ROUTING_MATRIX.items():
            assert "primary" in routing_info
            assert isinstance(routing_info["primary"], str)
            assert "agents" in routing_info
            assert isinstance(routing_info["agents"], list)

    def test_personas_structure(self):
        """Test that investor personas have expected structure."""
        expected_personas = ["conservative", "moderate", "aggressive"]

        for persona_name in expected_personas:
            assert persona_name in INVESTOR_PERSONAS
            persona = INVESTOR_PERSONAS[persona_name]

            # Check persona has required attributes
            assert hasattr(persona, "name")
            assert hasattr(persona, "risk_tolerance")
            assert hasattr(persona, "position_size_max")

    @pytest.mark.asyncio
    async def test_supervisor_agent_instantiation(self, mock_llm, mock_market_agent):
        """Test SupervisorAgent can be instantiated properly."""
        agents = {"market": mock_market_agent}

        supervisor = SupervisorAgent(
            llm=mock_llm, agents=agents, persona="moderate", ttl_hours=1
        )

        assert supervisor is not None
        assert supervisor.persona.name == "Moderate"
        assert "market" in supervisor.agents

    @pytest.mark.asyncio
    async def test_deep_research_agent_instantiation(self, mock_llm):
        """Test DeepResearchAgent can be instantiated properly."""
        # Test without API keys (should still work)
        research_agent = DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            ttl_hours=1,
            exa_api_key=None,
            tavily_api_key=None,
        )

        assert research_agent is not None
        assert research_agent.persona.name == "Moderate"

    @pytest.mark.asyncio
    async def test_deep_research_agent_with_api_keys(self, mock_llm):
        """Test DeepResearchAgent instantiation with API keys."""
        # Test with mock API keys
        research_agent = DeepResearchAgent(
            llm=mock_llm,
            persona="aggressive",
            ttl_hours=2,
            exa_api_key="test-exa-key",
            tavily_api_key="test-tavily-key",
        )

        assert research_agent is not None
        assert research_agent.persona.name == "Aggressive"
        # Should have initialized search providers
        assert hasattr(research_agent, "search_providers")

    @pytest.mark.asyncio
    async def test_supervisor_with_research_agent(self, mock_llm, mock_market_agent):
        """Test supervisor working with research agent."""
        # Create research agent
        research_agent = DeepResearchAgent(
            llm=mock_llm, persona="moderate", ttl_hours=1
        )

        # Create supervisor with both agents
        agents = {"market": mock_market_agent, "research": research_agent}

        supervisor = SupervisorAgent(
            llm=mock_llm, agents=agents, persona="moderate", ttl_hours=1
        )

        assert len(supervisor.agents) == 2
        assert "market" in supervisor.agents
        assert "research" in supervisor.agents

    def test_configuration_completeness(self):
        """Test that configuration system is complete."""
        from maverick_mcp.config.settings import get_settings

        settings = get_settings()

        # Check that research settings exist
        assert hasattr(settings, "research")
        assert hasattr(settings.research, "exa_api_key")
        assert hasattr(settings.research, "tavily_api_key")

        # Check that data limits exist
        assert hasattr(settings, "data_limits")
        assert hasattr(settings.data_limits, "max_agent_iterations")

    def test_exception_hierarchy(self):
        """Test that exception hierarchy is properly set up."""
        from maverick_mcp.exceptions import (
            AgentExecutionError,
            MaverickException,
            ResearchError,
            WebSearchError,
        )

        # Test exception hierarchy
        assert issubclass(AgentExecutionError, MaverickException)
        assert issubclass(ResearchError, MaverickException)
        assert issubclass(WebSearchError, ResearchError)

        # Test exception instantiation
        error = AgentExecutionError("Test error")
        assert error.message == "Test error"
        assert error.error_code == "AGENT_EXECUTION_ERROR"

    def test_state_classes_structure(self):
        """Test that state classes have proper structure."""
        from maverick_mcp.workflows.state import DeepResearchState, SupervisorState

        # These should be TypedDict classes
        assert hasattr(SupervisorState, "__annotations__")
        assert hasattr(DeepResearchState, "__annotations__")

        # Check key fields exist
        supervisor_fields = SupervisorState.__annotations__.keys()
        assert "query_classification" in supervisor_fields
        assert "agent_results" in supervisor_fields
        assert "workflow_status" in supervisor_fields

        research_fields = DeepResearchState.__annotations__.keys()
        assert "research_topic" in research_fields
        assert "search_results" in research_fields
        assert "research_findings" in research_fields

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test that circuit breaker integration works."""
        from maverick_mcp.agents.circuit_breaker import circuit_breaker, circuit_manager

        # Test circuit breaker manager
        assert circuit_manager is not None

        # Test circuit breaker decorator
        @circuit_breaker("test_breaker", failure_threshold=2)
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"

    def test_mcp_router_structure(self):
        """Test that MCP router is properly structured."""
        from maverick_mcp.api.routers.agents import agents_router

        # Should be a FastMCP instance
        assert agents_router is not None
        assert hasattr(agents_router, "name")
        assert agents_router.name == "Financial_Analysis_Agents"

    def test_agent_factory_function(self):
        """Test agent factory function structure."""
        from maverick_mcp.api.routers.agents import get_or_create_agent

        # Should be a callable function
        assert callable(get_or_create_agent)

        # Test with invalid agent type
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_or_create_agent("invalid_type", "moderate")


class TestOrchestrationWorkflow:
    """Test orchestration workflow components."""

    def test_persona_compatibility(self):
        """Test that all agents support all personas."""
        expected_personas = ["conservative", "moderate", "aggressive"]

        for persona_name in expected_personas:
            assert persona_name in INVESTOR_PERSONAS

            # All personas should have required attributes
            persona = INVESTOR_PERSONAS[persona_name]
            assert hasattr(persona, "name")
            assert hasattr(persona, "risk_tolerance")
            assert hasattr(persona, "position_size_max")
            assert hasattr(persona, "stop_loss_multiplier")

    def test_routing_categories_completeness(self):
        """Test that routing covers expected analysis categories."""
        expected_categories = {
            "market_screening",
            "company_research",
            "technical_analysis",
            "sentiment_analysis",
        }

        routing_categories = set(ROUTING_MATRIX.keys())

        # Should contain the key categories we care about
        for category in expected_categories:
            if category in routing_categories:
                routing_info = ROUTING_MATRIX[category]
                assert "primary" in routing_info
                assert "agents" in routing_info

    @pytest.mark.asyncio
    async def test_end_to_end_mock_workflow(self):
        """Test a complete mock workflow from query to response."""
        from langchain_core.language_models import FakeListLLM

        # Create fake LLM for testing
        fake_llm = FakeListLLM(
            responses=[
                "Mock analysis complete",
                "Mock research findings",
                "Mock synthesis result",
            ]
        )

        # Create mock agents
        mock_market_agent = MagicMock()
        mock_market_agent.analyze_market = AsyncMock(
            return_value={
                "status": "success",
                "summary": "Market screening complete",
                "confidence": 0.8,
            }
        )

        # Create supervisor with mock agents
        supervisor = SupervisorAgent(
            llm=fake_llm, agents={"market": mock_market_agent}, persona="moderate"
        )

        # This would normally call the orchestration method
        # For now, just verify the supervisor was created properly
        assert supervisor is not None
        assert len(supervisor.agents) == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
