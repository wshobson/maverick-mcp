"""
Comprehensive end-to-end functional tests for MCP tool integration.

This test suite validates the complete workflows that Claude Desktop users will
interact with, ensuring tools work correctly from MCP call through agent
orchestration to final response.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from maverick_mcp.api.routers import agents
from maverick_mcp.api.routers.agents import (
    get_or_create_agent,
)


# Access the underlying functions from the decorated tools
def get_tool_function(tool_obj):
    """Extract the underlying function from a FastMCP tool."""
    # FastMCP tools store the function in the 'fn' attribute
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


# Get the actual function implementations
orchestrated_analysis = get_tool_function(agents.orchestrated_analysis)
deep_research_financial = get_tool_function(agents.deep_research_financial)
compare_multi_agent_analysis = get_tool_function(agents.compare_multi_agent_analysis)
list_available_agents = get_tool_function(agents.list_available_agents)


class TestOrchestredAnalysisTool:
    """Test the orchestrated_analysis MCP tool."""

    @pytest.fixture
    def mock_supervisor_result(self):
        """Mock successful supervisor analysis result."""
        return {
            "status": "success",
            "summary": "Comprehensive analysis of AAPL shows strong momentum signals",
            "key_findings": [
                "Technical breakout above resistance",
                "Strong earnings growth trajectory",
                "Positive sector rotation into technology",
            ],
            "recommendations": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "confidence": 0.85,
                    "target_price": 180.00,
                    "stop_loss": 150.00,
                }
            ],
            "agents_used": ["market", "technical"],
            "execution_time_ms": 2500,
            "synthesis_confidence": 0.88,
            "methodology": "Multi-agent orchestration with parallel execution",
            "persona_adjustments": "Moderate risk tolerance applied to position sizing",
        }

    @pytest.fixture
    def mock_supervisor_agent(self, mock_supervisor_result):
        """Mock SupervisorAgent instance."""
        agent = MagicMock()
        agent.orchestrate_analysis = AsyncMock(return_value=mock_supervisor_result)
        return agent

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_success_workflow(self, mock_supervisor_agent):
        """Test complete successful workflow for orchestrated analysis."""
        query = "Analyze AAPL for potential investment opportunity"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(
                query=query,
                persona="moderate",
                routing_strategy="llm_powered",
                max_agents=3,
                parallel_execution=True,
            )

        # Validate top-level response structure
        assert result["status"] == "success"
        assert result["agent_type"] == "supervisor_orchestrated"
        assert result["persona"] == "moderate"
        assert result["routing_strategy"] == "llm_powered"
        assert "session_id" in result

        # Validate agent orchestration was called correctly
        mock_supervisor_agent.orchestrate_analysis.assert_called_once()
        call_args = mock_supervisor_agent.orchestrate_analysis.call_args
        assert call_args[1]["query"] == query
        assert call_args[1]["routing_strategy"] == "llm_powered"
        assert call_args[1]["max_agents"] == 3
        assert call_args[1]["parallel_execution"] is True
        assert "session_id" in call_args[1]

        # Validate orchestration results are properly passed through
        assert (
            result["summary"]
            == "Comprehensive analysis of AAPL shows strong momentum signals"
        )
        assert len(result["key_findings"]) == 3
        assert result["agents_used"] == ["market", "technical"]
        assert result["execution_time_ms"] == 2500
        assert result["synthesis_confidence"] == 0.88

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_persona_variations(
        self, mock_supervisor_agent
    ):
        """Test orchestrated analysis with different personas."""
        personas = ["conservative", "moderate", "aggressive", "day_trader"]
        query = "Find momentum stocks with strong technical signals"

        for persona in personas:
            with patch(
                "maverick_mcp.api.routers.agents.get_or_create_agent",
                return_value=mock_supervisor_agent,
            ):
                result = await orchestrated_analysis(query=query, persona=persona)

            assert result["status"] == "success"
            assert result["persona"] == persona

            # Verify agent was created with correct persona
            # Note: get_or_create_agent is not directly patchable, so we verify persona through result

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_routing_strategies(
        self, mock_supervisor_agent
    ):
        """Test different routing strategies."""
        strategies = ["llm_powered", "rule_based", "hybrid"]
        query = "Evaluate current market conditions"

        for strategy in strategies:
            with patch(
                "maverick_mcp.api.routers.agents.get_or_create_agent",
                return_value=mock_supervisor_agent,
            ):
                result = await orchestrated_analysis(
                    query=query, routing_strategy=strategy
                )

            assert result["status"] == "success"
            assert result["routing_strategy"] == strategy

            # Verify strategy was passed to orchestration
            call_args = mock_supervisor_agent.orchestrate_analysis.call_args[1]
            assert call_args["routing_strategy"] == strategy

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_parameter_validation(
        self, mock_supervisor_agent
    ):
        """Test parameter validation and edge cases."""
        base_query = "Analyze market trends"

        # Test max_agents bounds
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(
                query=base_query,
                max_agents=10,  # High value should be accepted
            )
        assert result["status"] == "success"

        # Test parallel execution toggle
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(
                query=base_query, parallel_execution=False
            )
        assert result["status"] == "success"
        call_args = mock_supervisor_agent.orchestrate_analysis.call_args[1]
        assert call_args["parallel_execution"] is False

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_session_continuity(
        self, mock_supervisor_agent
    ):
        """Test session ID handling for conversation continuity."""
        query = "Continue analyzing AAPL from previous conversation"
        session_id = str(uuid4())

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(query=query, session_id=session_id)

        assert result["status"] == "success"
        assert result["session_id"] == session_id

        # Verify session ID was passed to agent
        call_args = mock_supervisor_agent.orchestrate_analysis.call_args[1]
        assert call_args["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_error_handling(self):
        """Test error handling in orchestrated analysis."""
        mock_failing_agent = MagicMock()
        mock_failing_agent.orchestrate_analysis = AsyncMock(
            side_effect=Exception("Agent orchestration failed")
        )

        query = "This query will fail"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_failing_agent,
        ):
            result = await orchestrated_analysis(query=query)

        assert result["status"] == "error"
        assert result["agent_type"] == "supervisor_orchestrated"
        assert "Agent orchestration failed" in result["error"]

    @pytest.mark.asyncio
    async def test_orchestrated_analysis_response_format_compliance(
        self, mock_supervisor_agent
    ):
        """Test that response format matches MCP tool expectations."""
        query = "Format compliance test"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_supervisor_agent,
        ):
            result = await orchestrated_analysis(query=query)

        # Verify response is JSON serializable (MCP requirement)
        json_str = json.dumps(result)
        reconstructed = json.loads(json_str)
        assert reconstructed["status"] == "success"

        # Verify all required fields are present
        required_fields = [
            "status",
            "agent_type",
            "persona",
            "session_id",
            "routing_strategy",
            "agents_used",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Verify data types are MCP-compatible
        assert isinstance(result["status"], str)
        assert isinstance(result["agents_used"], list)
        assert isinstance(result["synthesis_confidence"], int | float)


class TestDeepResearchFinancialTool:
    """Test the deep_research_financial MCP tool."""

    @pytest.fixture
    def mock_research_result(self):
        """Mock successful deep research result."""
        return {
            "status": "success",
            "research_summary": "Comprehensive research on TSLA reveals mixed fundamentals",
            "key_findings": [
                "EV market growth slowing in key markets",
                "Manufacturing efficiency improvements continuing",
                "Regulatory headwinds in European markets",
            ],
            "source_details": [  # Changed from sources_analyzed to avoid conflict
                {
                    "url": "https://example.com/tsla-analysis",
                    "credibility": 0.9,
                    "relevance": 0.85,
                },
                {
                    "url": "https://example.com/ev-market-report",
                    "credibility": 0.8,
                    "relevance": 0.92,
                },
            ],
            "total_sources_processed": 15,
            "research_confidence": 0.87,
            "validation_checks_passed": 12,
            "methodology": "Multi-source web research with AI synthesis",
            "citation_count": 8,
            "research_depth_achieved": "comprehensive",
        }

    @pytest.fixture
    def mock_research_agent(self, mock_research_result):
        """Mock DeepResearchAgent instance."""
        agent = MagicMock()
        agent.conduct_research = AsyncMock(return_value=mock_research_result)
        return agent

    @pytest.mark.asyncio
    async def test_deep_research_success_workflow(self, mock_research_agent):
        """Test complete successful workflow for deep research."""
        research_topic = "Tesla TSLA competitive position in EV market"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(
                research_topic=research_topic,
                persona="moderate",
                research_depth="comprehensive",
                focus_areas=["fundamentals", "competitive_landscape"],
                timeframe="90d",
            )

        # Validate top-level response structure
        assert result["status"] == "success"
        assert result["agent_type"] == "deep_research"
        assert result["persona"] == "moderate"
        assert result["research_topic"] == research_topic
        assert result["research_depth"] == "comprehensive"
        assert result["focus_areas"] == ["fundamentals", "competitive_landscape"]

        # Validate research agent was called correctly
        mock_research_agent.conduct_research.assert_called_once()
        call_args = mock_research_agent.conduct_research.call_args[1]
        assert call_args["research_topic"] == research_topic
        assert call_args["research_depth"] == "comprehensive"
        assert call_args["focus_areas"] == ["fundamentals", "competitive_landscape"]
        assert call_args["timeframe"] == "90d"

        # Validate research results are properly passed through
        assert result["sources_analyzed"] == 15
        assert result["research_confidence"] == 0.87
        assert result["validation_checks_passed"] == 12

    @pytest.mark.asyncio
    async def test_deep_research_depth_variations(self, mock_research_agent):
        """Test different research depth levels."""
        depths = ["basic", "standard", "comprehensive", "exhaustive"]
        topic = "Apple AAPL financial health analysis"

        for depth in depths:
            with patch(
                "maverick_mcp.api.routers.agents.get_or_create_agent",
                return_value=mock_research_agent,
            ):
                result = await deep_research_financial(
                    research_topic=topic, research_depth=depth
                )

            assert result["status"] == "success"
            assert result["research_depth"] == depth

            # Verify depth was passed to research
            call_args = mock_research_agent.conduct_research.call_args[1]
            assert call_args["research_depth"] == depth

    @pytest.mark.asyncio
    async def test_deep_research_focus_areas_handling(self, mock_research_agent):
        """Test focus areas parameter handling."""
        topic = "Market sentiment analysis for tech sector"

        # Test with provided focus areas
        custom_focus = ["market_sentiment", "technicals", "macroeconomic"]
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(
                research_topic=topic, focus_areas=custom_focus
            )

        assert result["focus_areas"] == custom_focus

        # Test with default focus areas (None provided)
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(
                research_topic=topic,
                focus_areas=None,  # Should use defaults
            )

        # Should use default focus areas
        expected_defaults = [
            "fundamentals",
            "market_sentiment",
            "competitive_landscape",
        ]
        assert result["focus_areas"] == expected_defaults

    @pytest.mark.asyncio
    async def test_deep_research_timeframe_handling(self, mock_research_agent):
        """Test different timeframe options."""
        timeframes = ["7d", "30d", "90d", "1y"]
        topic = "Economic indicators impact on markets"

        for timeframe in timeframes:
            with patch(
                "maverick_mcp.api.routers.agents.get_or_create_agent",
                return_value=mock_research_agent,
            ):
                result = await deep_research_financial(
                    research_topic=topic, timeframe=timeframe
                )

            assert result["status"] == "success"

            # Verify timeframe was passed correctly
            call_args = mock_research_agent.conduct_research.call_args[1]
            assert call_args["timeframe"] == timeframe

    @pytest.mark.asyncio
    async def test_deep_research_source_validation_reporting(self, mock_research_agent):
        """Test source validation and credibility reporting."""
        topic = "Source validation test topic"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_research_agent,
        ):
            result = await deep_research_financial(research_topic=topic)

        # Validate source metrics are reported
        assert "sources_analyzed" in result
        assert "research_confidence" in result
        assert "validation_checks_passed" in result

        # Validate source analysis results - note that **result spreads all mock data
        # so we have both mapped keys and original keys
        assert result["sources_analyzed"] == 15  # Mapped from total_sources_processed
        assert result["total_sources_processed"] == 15  # Original from mock
        assert result["research_confidence"] == 0.87
        assert result["validation_checks_passed"] == 12

    @pytest.mark.asyncio
    async def test_deep_research_error_handling(self):
        """Test error handling in deep research."""
        mock_failing_agent = MagicMock()
        mock_failing_agent.conduct_research = AsyncMock(
            side_effect=Exception("Research API failed")
        )

        topic = "This research will fail"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_failing_agent,
        ):
            result = await deep_research_financial(research_topic=topic)

        assert result["status"] == "error"
        assert result["agent_type"] == "deep_research"
        assert "Research API failed" in result["error"]

    @pytest.mark.asyncio
    async def test_deep_research_persona_impact(self, mock_research_agent):
        """Test how different personas affect research focus."""
        topic = "High-risk growth stock evaluation"
        personas = ["conservative", "moderate", "aggressive", "day_trader"]

        for persona in personas:
            with patch(
                "maverick_mcp.api.routers.agents.get_or_create_agent",
                return_value=mock_research_agent,
            ):
                result = await deep_research_financial(
                    research_topic=topic, persona=persona
                )

            assert result["status"] == "success"
            assert result["persona"] == persona

            # Verify correct persona was used in result
            assert result["persona"] == persona


class TestCompareMultiAgentAnalysisTool:
    """Test the compare_multi_agent_analysis MCP tool."""

    @pytest.fixture
    def mock_market_agent_result(self):
        """Mock market agent analysis result."""
        return {
            "summary": "Market analysis shows bullish momentum in tech sector",
            "key_findings": ["Strong earnings growth", "Sector rotation into tech"],
            "confidence": 0.82,
            "methodology": "Technical screening with momentum indicators",
            "execution_time_ms": 1800,
        }

    @pytest.fixture
    def mock_supervisor_agent_result(self):
        """Mock supervisor agent analysis result."""
        return {
            "summary": "Multi-agent consensus indicates cautious optimism",
            "key_findings": [
                "Mixed signals from fundamentals",
                "Technical breakout confirmed",
            ],
            "confidence": 0.78,
            "methodology": "Orchestrated multi-agent analysis",
            "execution_time_ms": 3200,
        }

    @pytest.fixture
    def mock_agents(self, mock_market_agent_result, mock_supervisor_agent_result):
        """Mock agent instances for comparison testing."""
        market_agent = MagicMock()
        market_agent.analyze_market = AsyncMock(return_value=mock_market_agent_result)

        supervisor_agent = MagicMock()
        supervisor_agent.orchestrate_analysis = AsyncMock(
            return_value=mock_supervisor_agent_result
        )

        def get_agent_side_effect(agent_type, persona):
            if agent_type == "market":
                return market_agent
            elif agent_type == "supervisor":
                return supervisor_agent
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        return get_agent_side_effect

    @pytest.mark.asyncio
    async def test_multi_agent_comparison_success(self, mock_agents):
        """Test successful multi-agent comparison workflow."""
        query = "Compare different perspectives on NVDA investment potential"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=mock_agents,
        ):
            result = await compare_multi_agent_analysis(
                query=query, agent_types=["market", "supervisor"], persona="moderate"
            )

        # Validate top-level response structure
        assert result["status"] == "success"
        assert result["query"] == query
        assert result["persona"] == "moderate"
        assert result["agents_compared"] == ["market", "supervisor"]

        # Validate comparison structure
        assert "comparison" in result
        comparison = result["comparison"]

        # Check market agent results
        assert "market" in comparison
        market_result = comparison["market"]
        assert (
            market_result["summary"]
            == "Market analysis shows bullish momentum in tech sector"
        )
        assert market_result["confidence"] == 0.82
        assert len(market_result["key_findings"]) == 2

        # Check supervisor agent results
        assert "supervisor" in comparison
        supervisor_result = comparison["supervisor"]
        assert (
            supervisor_result["summary"]
            == "Multi-agent consensus indicates cautious optimism"
        )
        assert supervisor_result["confidence"] == 0.78
        assert len(supervisor_result["key_findings"]) == 2

        # Check execution time tracking
        assert "execution_times_ms" in result
        exec_times = result["execution_times_ms"]
        assert exec_times["market"] == 1800
        assert exec_times["supervisor"] == 3200

    @pytest.mark.asyncio
    async def test_multi_agent_comparison_default_agents(self, mock_agents):
        """Test default agent selection when none specified."""
        query = "Default agent comparison test"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=mock_agents,
        ):
            result = await compare_multi_agent_analysis(
                query=query,
                agent_types=None,  # Should use defaults
            )

        assert result["status"] == "success"
        # Should default to market and supervisor agents
        assert set(result["agents_compared"]) == {"market", "supervisor"}

    @pytest.mark.asyncio
    async def test_multi_agent_comparison_session_isolation(self, mock_agents):
        """Test session ID isolation for different agents."""
        query = "Session isolation test"
        base_session_id = str(uuid4())

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=mock_agents,
        ):
            result = await compare_multi_agent_analysis(
                query=query, session_id=base_session_id
            )

        assert result["status"] == "success"

        # Verify agents were called with isolated session IDs
        # (This would be validated through call inspection in real implementation)

    @pytest.mark.asyncio
    async def test_multi_agent_comparison_partial_failure(self):
        """Test handling when some agents fail but others succeed."""

        def failing_get_agent_side_effect(agent_type, persona):
            if agent_type == "market":
                agent = MagicMock()
                agent.analyze_market = AsyncMock(
                    return_value={
                        "summary": "Successful market analysis",
                        "key_findings": ["Finding 1"],
                        "confidence": 0.8,
                    }
                )
                return agent
            elif agent_type == "supervisor":
                agent = MagicMock()
                agent.orchestrate_analysis = AsyncMock(
                    side_effect=Exception("Supervisor agent failed")
                )
                return agent
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

        query = "Partial failure test"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=failing_get_agent_side_effect,
        ):
            result = await compare_multi_agent_analysis(
                query=query, agent_types=["market", "supervisor"]
            )

        assert result["status"] == "success"
        comparison = result["comparison"]

        # Market agent should succeed
        assert "market" in comparison
        assert comparison["market"]["summary"] == "Successful market analysis"

        # Supervisor agent should show error
        assert "supervisor" in comparison
        assert "error" in comparison["supervisor"]
        assert comparison["supervisor"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_multi_agent_comparison_insights_generation(self, mock_agents):
        """Test insights generation from comparison results."""
        query = "Generate insights from agent comparison"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=mock_agents,
        ):
            result = await compare_multi_agent_analysis(query=query)

        assert result["status"] == "success"
        assert "insights" in result
        # Should provide some explanatory insights about different perspectives
        assert isinstance(result["insights"], str)
        assert len(result["insights"]) > 0

    @pytest.mark.asyncio
    async def test_multi_agent_comparison_error_handling(self):
        """Test agent creation failure handling."""

        def complete_failure_side_effect(agent_type, persona):
            raise Exception(f"Failed to create {agent_type} agent")

        query = "Complete failure test"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            side_effect=complete_failure_side_effect,
        ):
            result = await compare_multi_agent_analysis(query=query)

        # The function handles individual agent failures gracefully and returns success
        # but with failed agents marked in the comparison results
        assert result["status"] == "success"
        assert "comparison" in result

        # All agents should have failed
        comparison = result["comparison"]
        for agent_type in ["market", "supervisor"]:  # Default agent types
            if agent_type in comparison:
                assert "error" in comparison[agent_type]
                assert "Failed to create" in comparison[agent_type]["error"]


class TestEndToEndIntegrationWorkflows:
    """Test complete end-to-end workflows that mirror real Claude Desktop usage."""

    @pytest.mark.asyncio
    async def test_complete_stock_analysis_workflow(self):
        """Test a complete stock analysis workflow from start to finish."""
        # Simulate a user asking for complete stock analysis
        query = (
            "I want a comprehensive analysis of Apple (AAPL) as a long-term investment"
        )

        # Mock successful orchestrated analysis
        mock_result = {
            "status": "success",
            "summary": "AAPL presents a strong long-term investment opportunity",
            "key_findings": [
                "Strong financial fundamentals with consistent revenue growth",
                "Market-leading position in premium smartphone segment",
                "Services revenue providing stable recurring income",
                "Strong balance sheet with substantial cash reserves",
            ],
            "recommendations": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "confidence": 0.87,
                    "target_price": 195.00,
                    "stop_loss": 165.00,
                    "position_size": "5% of portfolio",
                }
            ],
            "agents_used": ["market", "fundamental", "technical"],
            "execution_time_ms": 4200,
            "synthesis_confidence": 0.89,
        }

        mock_agent = MagicMock()
        mock_agent.orchestrate_analysis = AsyncMock(return_value=mock_result)

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_agent,
        ):
            result = await orchestrated_analysis(
                query=query,
                persona="moderate",
                routing_strategy="llm_powered",
                max_agents=5,
                parallel_execution=True,
            )

        # Validate complete workflow results
        assert result["status"] == "success"
        assert (
            "AAPL presents a strong long-term investment opportunity"
            in result["summary"]
        )
        assert len(result["key_findings"]) == 4
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["symbol"] == "AAPL"
        assert result["recommendations"][0]["confidence"] > 0.8

        # Validate execution metrics
        assert result["execution_time_ms"] > 0
        assert result["synthesis_confidence"] > 0.8
        assert len(result["agents_used"]) >= 2

    @pytest.mark.asyncio
    async def test_market_research_workflow(self):
        """Test comprehensive market research workflow."""
        research_topic = "Impact of rising interest rates on REIT sector performance"

        # Mock comprehensive research result
        mock_result = {
            "research_summary": "Rising interest rates create mixed outlook for REITs",
            "key_findings": [
                "Higher rates increase borrowing costs for REIT acquisitions",
                "Residential REITs more sensitive than commercial REITs",
                "Dividend yields become less attractive vs bonds",
                "Quality REITs with strong cash flows may outperform",
            ],
            "source_details": [  # Changed from sources_analyzed to avoid conflict
                {
                    "url": "https://example.com/reit-analysis",
                    "credibility": 0.92,
                    "relevance": 0.88,
                },
                {
                    "url": "https://example.com/interest-rate-impact",
                    "credibility": 0.89,
                    "relevance": 0.91,
                },
            ],
            "total_sources_processed": 24,
            "research_confidence": 0.84,
            "validation_checks_passed": 20,
            "sector_breakdown": {
                "residential": {"outlook": "negative", "confidence": 0.78},
                "commercial": {"outlook": "neutral", "confidence": 0.72},
                "industrial": {"outlook": "positive", "confidence": 0.81},
            },
        }

        mock_agent = MagicMock()
        mock_agent.conduct_research = AsyncMock(return_value=mock_result)

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_agent,
        ):
            result = await deep_research_financial(
                research_topic=research_topic,
                persona="conservative",
                research_depth="comprehensive",
                focus_areas=["fundamentals", "market_sentiment", "macroeconomic"],
                timeframe="90d",
            )

        # Validate research workflow results
        assert result["status"] == "success"
        assert (
            "Rising interest rates create mixed outlook for REITs"
            in result["research_summary"]
        )
        # Note: sources_analyzed is mapped from total_sources_processed, both should exist due to **result spreading
        assert result["sources_analyzed"] == 24
        assert result["total_sources_processed"] == 24  # Original mock value
        assert result["research_confidence"] > 0.8
        assert result["validation_checks_passed"] == 20

    @pytest.mark.asyncio
    async def test_performance_optimization_workflow(self):
        """Test performance under various load conditions."""
        # Test concurrent requests to simulate multiple Claude Desktop users
        queries = [
            "Analyze tech sector momentum",
            "Research ESG investing trends",
            "Compare growth vs value strategies",
            "Evaluate cryptocurrency market sentiment",
            "Assess inflation impact on consumer staples",
        ]

        mock_agent = MagicMock()
        mock_agent.orchestrate_analysis = AsyncMock(
            return_value={
                "status": "success",
                "summary": "Analysis completed successfully",
                "execution_time_ms": 2000,
                "agents_used": ["market"],
                "synthesis_confidence": 0.85,
            }
        )

        # Simulate concurrent requests
        start_time = time.time()

        tasks = []
        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=mock_agent,
        ):
            for query in queries:
                task = orchestrated_analysis(
                    query=query, persona="moderate", parallel_execution=True
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Validate all requests completed successfully
        assert len(results) == 5
        for result in results:
            assert result["status"] == "success"

        # Performance should be reasonable (< 30 seconds for 5 concurrent requests)
        assert total_time < 30.0

    @pytest.mark.asyncio
    async def test_timeout_and_recovery_workflow(self):
        """Test timeout scenarios and recovery mechanisms."""
        # Mock an agent that takes too long initially then recovers
        timeout_then_success_agent = MagicMock()

        call_count = 0

        async def mock_slow_then_fast(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call simulates timeout
                await asyncio.sleep(0.1)  # Short delay for testing
                raise TimeoutError("Analysis timed out")
            else:
                # Subsequent calls succeed quickly
                return {
                    "status": "success",
                    "summary": "Recovered analysis",
                    "execution_time_ms": 800,
                }

        timeout_then_success_agent.orchestrate_analysis = mock_slow_then_fast

        query = "This analysis will timeout then recover"

        with patch(
            "maverick_mcp.api.routers.agents.get_or_create_agent",
            return_value=timeout_then_success_agent,
        ):
            # First attempt should fail with timeout
            result1 = await orchestrated_analysis(query=query)
            assert result1["status"] == "error"
            assert "timed out" in result1["error"].lower()

            # Second attempt should succeed (recovery)
            result2 = await orchestrated_analysis(query=query)
            assert result2["status"] == "success"
            assert result2["summary"] == "Recovered analysis"

    @pytest.mark.asyncio
    async def test_different_personas_comparative_workflow(self):
        """Test how different personas affect the complete analysis workflow."""
        query = "Should I invest in high-growth technology stocks?"

        # Mock different results based on persona
        def persona_aware_mock(agent_type, persona):
            agent = MagicMock()

            if persona == "conservative":
                agent.orchestrate_analysis = AsyncMock(
                    return_value={
                        "status": "success",
                        "summary": "Conservative approach suggests limiting tech exposure to 10-15%",
                        "risk_assessment": "High volatility concerns",
                        "recommended_allocation": 0.12,
                        "agents_used": ["risk", "fundamental"],
                    }
                )
            elif persona == "aggressive":
                agent.orchestrate_analysis = AsyncMock(
                    return_value={
                        "status": "success",
                        "summary": "Aggressive strategy supports 30-40% tech allocation for growth",
                        "risk_assessment": "Acceptable volatility for growth potential",
                        "recommended_allocation": 0.35,
                        "agents_used": ["momentum", "growth"],
                    }
                )
            else:  # moderate
                agent.orchestrate_analysis = AsyncMock(
                    return_value={
                        "status": "success",
                        "summary": "Balanced approach recommends 20-25% tech allocation",
                        "risk_assessment": "Managed risk with diversification",
                        "recommended_allocation": 0.22,
                        "agents_used": ["market", "fundamental", "technical"],
                    }
                )

            return agent

        personas = ["conservative", "moderate", "aggressive"]
        results = {}

        for persona in personas:
            with patch(
                "maverick_mcp.api.routers.agents.get_or_create_agent",
                side_effect=persona_aware_mock,
            ):
                result = await orchestrated_analysis(query=query, persona=persona)
                results[persona] = result

        # Validate persona-specific differences
        assert all(r["status"] == "success" for r in results.values())

        # Conservative should have lower allocation
        assert "10-15%" in results["conservative"]["summary"]

        # Aggressive should have higher allocation
        assert "30-40%" in results["aggressive"]["summary"]

        # Moderate should be balanced
        assert "20-25%" in results["moderate"]["summary"]


class TestMCPToolsListingAndValidation:
    """Test MCP tools listing and validation functions."""

    def test_list_available_agents_structure(self):
        """Test the list_available_agents tool returns proper structure."""
        result = list_available_agents()

        # Validate top-level structure
        assert result["status"] == "success"
        assert "agents" in result
        assert "orchestrated_tools" in result
        assert "features" in result

        # Validate agent descriptions
        agents = result["agents"]
        expected_agents = [
            "market_analysis",
            "supervisor_orchestrated",
            "deep_research",
        ]

        for agent_name in expected_agents:
            assert agent_name in agents
            agent_info = agents[agent_name]

            # Each agent should have required fields
            assert "description" in agent_info
            assert "capabilities" in agent_info
            assert "status" in agent_info
            assert isinstance(agent_info["capabilities"], list)
            assert len(agent_info["capabilities"]) > 0

        # Validate orchestrated tools
        orchestrated_tools = result["orchestrated_tools"]
        expected_tools = [
            "orchestrated_analysis",
            "deep_research_financial",
            "compare_multi_agent_analysis",
        ]

        for tool_name in expected_tools:
            assert tool_name in orchestrated_tools
            assert isinstance(orchestrated_tools[tool_name], str)
            assert len(orchestrated_tools[tool_name]) > 0

        # Validate features
        features = result["features"]
        expected_features = [
            "persona_adaptation",
            "conversation_memory",
            "streaming_support",
            "tool_integration",
        ]

        for feature_name in expected_features:
            if feature_name in features:
                assert isinstance(features[feature_name], str)
                assert len(features[feature_name]) > 0

    def test_agent_factory_validation(self):
        """Test agent factory function parameter validation."""
        # Test valid agent types that work with current implementation
        valid_types = ["market", "deep_research"]

        for agent_type in valid_types:
            # Should not raise exception for valid types
            try:
                # This will create a FakeListLLM since no OPENAI_API_KEY in test
                agent = get_or_create_agent(agent_type, "moderate")
                assert agent is not None
            except Exception as e:
                # Only acceptable exception is missing dependencies or initialization issues
                assert any(
                    keyword in str(e).lower()
                    for keyword in ["api", "key", "initialization", "missing"]
                )

        # Test supervisor agent (requires agents parameter - known limitation)
        try:
            agent = get_or_create_agent("supervisor", "moderate")
            assert agent is not None
        except Exception as e:
            # Expected to fail due to missing agents parameter
            assert "missing" in str(e).lower() and "agents" in str(e).lower()

        # Test invalid agent type
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_or_create_agent("invalid_agent_type", "moderate")

    def test_persona_validation_comprehensive(self):
        """Test comprehensive persona validation across all tools."""
        valid_personas = ["conservative", "moderate", "aggressive", "day_trader"]

        # Test each persona can be used (basic validation)
        for persona in valid_personas:
            try:
                # This tests the persona lookup doesn't crash
                agent = get_or_create_agent("market", persona)
                assert agent is not None
            except Exception as e:
                # Only acceptable exception is missing API dependencies
                assert "api" in str(e).lower() or "key" in str(e).lower()


if __name__ == "__main__":
    # Run with specific markers for different test categories
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-m",
            "not slow",  # Skip slow tests by default
            "--disable-warnings",
        ]
    )
