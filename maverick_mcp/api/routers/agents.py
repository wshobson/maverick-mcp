"""
Agent router for LangGraph-based financial analysis agents.

This router exposes the LangGraph agents as MCP tools while maintaining
compatibility with the existing infrastructure.
"""

import logging
import os
from typing import Any

from fastmcp import FastMCP

from maverick_mcp.agents.market_analysis import MarketAnalysisAgent

logger = logging.getLogger(__name__)

# Create the agents router
agents_router: FastMCP = FastMCP("Financial_Analysis_Agents")


# Cache for agent instances to avoid recreation
_agent_cache: dict[str, Any] = {}


def get_or_create_agent(agent_type: str, persona: str = "moderate") -> Any:
    """Get or create an agent instance with caching."""
    cache_key = f"{agent_type}:{persona}"

    if cache_key not in _agent_cache:
        # Create LLM based on settings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
        else:
            # Fallback for testing
            from langchain_core.language_models import FakeListLLM

            llm = FakeListLLM(
                responses=[
                    "Mock market analysis response",
                    "Mock technical analysis response",
                    "Mock risk analysis response",
                ]
            )
            logger.warning(
                "Using FakeListLLM - configure OPENAI_API_KEY for real analysis"
            )

        # Create agent based on type
        if agent_type == "market":
            _agent_cache[cache_key] = MarketAnalysisAgent(
                llm=llm, persona=persona, ttl_hours=1
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    return _agent_cache[cache_key]


async def analyze_market_with_agent(
    query: str,
    persona: str = "moderate",
    screening_strategy: str = "momentum",
    max_results: int = 20,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Analyze market using LangGraph agent with persona-aware recommendations.

    This tool uses advanced AI agents that adapt their analysis based on
    investor risk profiles (conservative, moderate, aggressive).

    Args:
        query: Market analysis query (e.g., "Find top momentum stocks")
        persona: Investor persona (conservative, moderate, aggressive)
        screening_strategy: Strategy to use (momentum, maverick, supply_demand_breakout)
        max_results: Maximum number of results
        session_id: Optional session ID for conversation continuity

    Returns:
        Persona-adjusted market analysis with recommendations
    """
    try:
        # Generate session ID if not provided
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        # Get or create agent
        agent = get_or_create_agent("market", persona)

        # Run analysis
        result = await agent.analyze_market(
            query=query,
            session_id=session_id,
            screening_strategy=screening_strategy,
            max_results=max_results,
        )

        return {
            "status": "success",
            "agent_type": "market_analysis",
            "persona": persona,
            "session_id": session_id,
            **result,
        }

    except Exception as e:
        logger.error(f"Error in market agent analysis: {str(e)}")
        return {"status": "error", "error": str(e), "agent_type": "market_analysis"}


async def get_agent_streaming_analysis(
    query: str,
    persona: str = "moderate",
    stream_mode: str = "updates",
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Get streaming market analysis with real-time updates.

    This demonstrates LangGraph's streaming capabilities. In a real
    implementation, this would return a streaming response.

    Args:
        query: Analysis query
        persona: Investor persona
        stream_mode: Streaming mode (updates, values, messages)
        session_id: Optional session ID

    Returns:
        Streaming configuration and initial results
    """
    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        agent = get_or_create_agent("market", persona)

        # For MCP compatibility, we'll collect streamed results
        # In a real implementation, this would be a streaming endpoint
        updates = []

        async for chunk in agent.stream_analysis(
            query=query, session_id=session_id, stream_mode=stream_mode
        ):
            updates.append(chunk)
            # Limit collected updates for demo
            if len(updates) >= 5:
                break

        return {
            "status": "success",
            "stream_mode": stream_mode,
            "persona": persona,
            "session_id": session_id,
            "updates_collected": len(updates),
            "sample_updates": updates[:3],
            "note": "Full streaming requires WebSocket or SSE endpoint",
        }

    except Exception as e:
        logger.error(f"Error in streaming analysis: {str(e)}")
        return {"status": "error", "error": str(e)}


def list_available_agents() -> dict[str, Any]:
    """
    List all available LangGraph agents and their capabilities.

    Returns:
        Information about available agents and personas
    """
    return {
        "status": "success",
        "agents": {
            "market_analysis": {
                "description": "Market screening and sector analysis",
                "personas": ["conservative", "moderate", "aggressive"],
                "capabilities": [
                    "Momentum screening",
                    "Sector rotation analysis",
                    "Market breadth indicators",
                    "Risk-adjusted recommendations",
                ],
                "streaming_modes": ["updates", "values", "messages", "debug"],
            },
            "technical_analysis": {
                "description": "Chart patterns and technical indicators",
                "status": "coming_soon",
            },
            "risk_management": {
                "description": "Position sizing and portfolio risk",
                "status": "coming_soon",
            },
            "portfolio_optimization": {
                "description": "Rebalancing and allocation",
                "status": "coming_soon",
            },
        },
        "features": {
            "persona_adaptation": "Agents adjust recommendations based on risk profile",
            "conversation_memory": "Maintains context within sessions",
            "streaming_support": "Real-time updates during analysis",
            "tool_integration": "Access to all MCP financial tools",
        },
    }


async def compare_personas_analysis(
    query: str, session_id: str | None = None
) -> dict[str, Any]:
    """
    Compare analysis across different investor personas.

    Runs the same query through conservative, moderate, and aggressive
    personas to show how recommendations differ.

    Args:
        query: Analysis query to run
        session_id: Optional session ID prefix

    Returns:
        Comparative analysis across all personas
    """
    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        results = {}

        for persona in ["conservative", "moderate", "aggressive"]:
            agent = get_or_create_agent("market", persona)

            # Run analysis for this persona
            result = await agent.analyze_market(
                query=query, session_id=f"{session_id}_{persona}", max_results=10
            )

            results[persona] = {
                "summary": result.get("results", {}).get("summary", ""),
                "top_picks": result.get("results", {}).get("screened_symbols", [])[:5],
                "risk_parameters": {
                    "risk_tolerance": agent.persona.risk_tolerance,
                    "max_position_size": f"{agent.persona.position_size_max * 100:.1f}%",
                    "stop_loss_multiplier": agent.persona.stop_loss_multiplier,
                },
            }

        return {
            "status": "success",
            "query": query,
            "comparison": results,
            "insights": "Notice how recommendations vary by risk profile",
        }

    except Exception as e:
        logger.error(f"Error in persona comparison: {str(e)}")
        return {"status": "error", "error": str(e)}
