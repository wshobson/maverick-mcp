"""
Agent router for LangGraph-based financial analysis agents.

This router exposes the LangGraph agents as MCP tools while maintaining
compatibility with the existing infrastructure.
"""

import asyncio
import logging
import os
from typing import Any

from fastmcp import FastMCP

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.agents.market_analysis import MarketAnalysisAgent
from maverick_mcp.agents.supervisor import SupervisorAgent

logger = logging.getLogger(__name__)

# Create the agents router
agents_router: FastMCP = FastMCP("Financial_Analysis_Agents")


# Cache for agent instances to avoid recreation
_agent_cache: dict[str, Any] = {}


def get_or_create_agent(agent_type: str, persona: str = "moderate") -> Any:
    """Get or create an agent instance with caching."""
    cache_key = f"{agent_type}:{persona}"

    if cache_key not in _agent_cache:
        try:
            # Import task-aware LLM factory
            from maverick_mcp.providers.llm_factory import get_llm
            from maverick_mcp.providers.openrouter_provider import TaskType

            # Map agent types to task types for optimal model selection
            task_mapping = {
                "market": TaskType.MARKET_ANALYSIS,
                "technical": TaskType.TECHNICAL_ANALYSIS,
                "supervisor": TaskType.MULTI_AGENT_ORCHESTRATION,
                "deep_research": TaskType.DEEP_RESEARCH,
            }

            task_type = task_mapping.get(agent_type, TaskType.GENERAL)

            # Get optimized LLM for this task
            llm = get_llm(task_type=task_type)

            # Create agent based on type
            if agent_type == "market":
                _agent_cache[cache_key] = MarketAnalysisAgent(
                    llm=llm, persona=persona, ttl_hours=1
                )
            elif agent_type == "supervisor":
                # Create mock agents for supervisor
                agents = {
                    "market": get_or_create_agent("market", persona),
                    "technical": None,  # Would be actual technical agent in full implementation
                }
                _agent_cache[cache_key] = SupervisorAgent(
                    llm=llm, agents=agents, persona=persona, ttl_hours=1
                )
            elif agent_type == "deep_research":
                # Get web search API keys from environment
                exa_api_key = os.getenv("EXA_API_KEY")

                agent = DeepResearchAgent(
                    llm=llm,
                    persona=persona,
                    ttl_hours=1,
                    exa_api_key=exa_api_key,
                )
                # Mark for initialization - will be initialized on first use
                agent._needs_initialization = True
                _agent_cache[cache_key] = agent
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        except Exception as e:
            logger.error(f"Failed to create {agent_type} agent: {e}")
            raise RuntimeError(
                f"Agent '{agent_type}' could not be initialized. "
                f"Ensure required API keys (OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY) "
                f"are configured. Error: {e}"
            ) from e

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

        # Run analysis with timeout to prevent Claude Desktop hanging
        result = await asyncio.wait_for(
            agent.analyze_market(
                query=query,
                session_id=session_id,
                screening_strategy=screening_strategy,
                max_results=max_results,
            ),
            timeout=25.0,
        )

        return {
            "status": "success",
            "agent_type": "market_analysis",
            "persona": persona,
            "session_id": session_id,
            **result,
        }

    except TimeoutError:
        logger.error("Market agent analysis timed out after 25s")
        return {
            "status": "error",
            "error": "Analysis timed out. Try a simpler query or check API key configuration.",
            "agent_type": "market_analysis",
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

        async def _collect_stream():
            async for chunk in agent.stream_analysis(
                query=query, session_id=session_id, stream_mode=stream_mode
            ):
                updates.append(chunk)
                if len(updates) >= 5:
                    break

        await asyncio.wait_for(_collect_stream(), timeout=25.0)

        return {
            "status": "success",
            "stream_mode": stream_mode,
            "persona": persona,
            "session_id": session_id,
            "updates_collected": len(updates),
            "sample_updates": updates[:3],
            "note": "Full streaming requires WebSocket or SSE endpoint",
        }

    except TimeoutError:
        logger.error("Streaming analysis timed out after 25s")
        return {
            "status": "error",
            "error": "Streaming analysis timed out. Try a simpler query.",
        }
    except Exception as e:
        logger.error(f"Error in streaming analysis: {str(e)}")
        return {"status": "error", "error": str(e)}


async def orchestrated_analysis(
    query: str,
    persona: str = "moderate",
    routing_strategy: str = "llm_powered",
    max_agents: int = 3,
    parallel_execution: bool = True,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Run orchestrated multi-agent analysis using the SupervisorAgent.

    This tool coordinates multiple specialized agents to provide comprehensive
    financial analysis. The supervisor intelligently routes queries to appropriate
    agents and synthesizes their results.

    Args:
        query: Financial analysis query
        persona: Investor persona (conservative, moderate, aggressive, day_trader)
        routing_strategy: How to route tasks (llm_powered, rule_based, hybrid)
        max_agents: Maximum number of agents to use
        parallel_execution: Whether to run agents in parallel
        session_id: Optional session ID for conversation continuity

    Returns:
        Orchestrated analysis with synthesized recommendations
    """
    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        # Get supervisor agent
        supervisor = get_or_create_agent("supervisor", persona)

        # Run orchestrated analysis with timeout
        result = await asyncio.wait_for(
            supervisor.coordinate_agents(
                query=query,
                session_id=session_id,
                routing_strategy=routing_strategy,
                max_agents=max_agents,
                parallel_execution=parallel_execution,
            ),
            timeout=25.0,
        )

        return {
            "status": "success",
            "agent_type": "supervisor_orchestrated",
            "persona": persona,
            "session_id": session_id,
            "routing_strategy": routing_strategy,
            "agents_used": result.get("agents_used", []),
            "execution_time_ms": result.get("execution_time_ms"),
            "synthesis_confidence": result.get("synthesis_confidence"),
            **result,
        }

    except TimeoutError:
        logger.error("Orchestrated analysis timed out after 25s")
        return {
            "status": "error",
            "error": "Orchestrated analysis timed out. Try reducing max_agents or simplifying the query.",
            "agent_type": "supervisor_orchestrated",
        }
    except Exception as e:
        logger.error(f"Error in orchestrated analysis: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "agent_type": "supervisor_orchestrated",
        }


async def deep_research_financial(
    research_topic: str,
    persona: str = "moderate",
    research_depth: str = "comprehensive",
    focus_areas: list[str] | None = None,
    timeframe: str = "30d",
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Conduct comprehensive financial research using web search and AI analysis.

    This tool performs deep research on financial topics, companies, or market
    trends using multiple web search providers and AI-powered content analysis.

    Args:
        research_topic: Main research topic (company, symbol, or market theme)
        persona: Investor persona affecting research focus
        research_depth: Depth level (basic, standard, comprehensive, exhaustive)
        focus_areas: Specific areas to focus on (e.g., ["fundamentals", "technicals"])
        timeframe: Time range for research (7d, 30d, 90d, 1y)
        session_id: Optional session ID for conversation continuity

    Returns:
        Comprehensive research report with validated sources and analysis
    """
    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        if focus_areas is None:
            focus_areas = ["fundamentals", "market_sentiment", "competitive_landscape"]

        # Get deep research agent
        researcher = get_or_create_agent("deep_research", persona)

        # Deep research gets a longer timeout since it's expected to take more time
        timeout = {"basic": 30.0, "standard": 60.0, "comprehensive": 120.0, "exhaustive": 180.0}
        research_timeout = timeout.get(research_depth, 60.0)

        result = await asyncio.wait_for(
            researcher.research_comprehensive(
                topic=research_topic,
                session_id=session_id,
                depth=research_depth,
                focus_areas=focus_areas,
                timeframe=timeframe,
            ),
            timeout=research_timeout,
        )

        return {
            "status": "success",
            "agent_type": "deep_research",
            "persona": persona,
            "session_id": session_id,
            "research_topic": research_topic,
            "research_depth": research_depth,
            "focus_areas": focus_areas,
            "sources_analyzed": result.get("total_sources_processed", 0),
            "research_confidence": result.get("research_confidence"),
            "validation_checks_passed": result.get("validation_checks_passed"),
            **result,
        }

    except TimeoutError:
        logger.error(f"Deep research timed out for topic: {research_topic}")
        return {
            "status": "error",
            "error": f"Research timed out. Try reducing research_depth from '{research_depth}' or narrowing focus_areas.",
            "agent_type": "deep_research",
        }
    except Exception as e:
        logger.error(f"Error in deep research: {str(e)}")
        return {"status": "error", "error": str(e), "agent_type": "deep_research"}


async def compare_multi_agent_analysis(
    query: str,
    agent_types: list[str] | None = None,
    persona: str = "moderate",
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Compare analysis results across multiple agent types.

    Runs the same query through different specialized agents to show how
    their approaches and insights differ, providing a multi-dimensional view.

    Args:
        query: Analysis query to run across multiple agents
        agent_types: List of agent types to compare (default: ["market", "supervisor"])
        persona: Investor persona for all agents
        session_id: Optional session ID prefix

    Returns:
        Comparative analysis showing different agent perspectives
    """
    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        if agent_types is None:
            agent_types = ["market", "supervisor"]

        results = {}
        execution_times = {}

        for agent_type in agent_types:
            try:
                agent = get_or_create_agent(agent_type, persona)

                # Run analysis based on agent type
                if agent_type == "market":
                    result = await agent.analyze_market(
                        query=query,
                        session_id=f"{session_id}_{agent_type}",
                        max_results=10,
                    )
                elif agent_type == "supervisor":
                    result = await agent.coordinate_agents(
                        query=query,
                        session_id=f"{session_id}_{agent_type}",
                        max_agents=2,
                    )
                else:
                    continue

                results[agent_type] = {
                    "summary": result.get("summary", ""),
                    "key_findings": result.get("key_findings", []),
                    "confidence": result.get("confidence", 0.0),
                    "methodology": result.get("methodology", f"{agent_type} analysis"),
                }
                execution_times[agent_type] = result.get("execution_time_ms", 0)

            except Exception as e:
                logger.warning(f"Error with {agent_type} agent: {str(e)}")
                results[agent_type] = {"error": str(e), "status": "failed"}

        return {
            "status": "success",
            "query": query,
            "persona": persona,
            "agents_compared": list(results.keys()),
            "comparison": results,
            "execution_times_ms": execution_times,
            "insights": "Each agent brings unique analytical perspectives and methodologies",
        }

    except Exception as e:
        logger.error(f"Error in multi-agent comparison: {str(e)}")
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
                "status": "active",
            },
            "supervisor_orchestrated": {
                "description": "Multi-agent orchestration and coordination",
                "personas": ["conservative", "moderate", "aggressive", "day_trader"],
                "capabilities": [
                    "Intelligent query routing",
                    "Multi-agent coordination",
                    "Result synthesis and conflict resolution",
                    "Parallel and sequential execution",
                    "Comprehensive analysis workflows",
                ],
                "routing_strategies": ["llm_powered", "rule_based", "hybrid"],
                "status": "active",
            },
            "deep_research": {
                "description": "Comprehensive financial research with web search",
                "personas": ["conservative", "moderate", "aggressive", "day_trader"],
                "capabilities": [
                    "Multi-provider web search",
                    "AI-powered content analysis",
                    "Source validation and credibility scoring",
                    "Citation and reference management",
                    "Comprehensive research reports",
                ],
                "research_depths": ["basic", "standard", "comprehensive", "exhaustive"],
                "focus_areas": [
                    "fundamentals",
                    "technicals",
                    "market_sentiment",
                    "competitive_landscape",
                ],
                "status": "active",
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
        "orchestrated_tools": {
            "orchestrated_analysis": "Coordinate multiple agents for comprehensive analysis",
            "deep_research_financial": "Conduct thorough research with web search",
            "compare_multi_agent_analysis": "Compare different agent perspectives",
        },
        "features": {
            "persona_adaptation": "Agents adjust recommendations based on risk profile",
            "conversation_memory": "Maintains context within sessions",
            "streaming_support": "Real-time updates during analysis",
            "tool_integration": "Access to all MCP financial tools",
            "multi_agent_orchestration": "Coordinate multiple specialized agents",
            "web_search_research": "AI-powered research with source validation",
            "intelligent_routing": "LLM-powered task routing and optimization",
        },
        "personas": ["conservative", "moderate", "aggressive", "day_trader"],
        "routing_strategies": ["llm_powered", "rule_based", "hybrid"],
        "research_depths": ["basic", "standard", "comprehensive", "exhaustive"],
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
