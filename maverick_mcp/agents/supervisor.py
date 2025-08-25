"""
SupervisorAgent implementation using 2025 LangGraph patterns.

Orchestrates multiple specialized agents with intelligent routing, result synthesis,
and conflict resolution for comprehensive financial analysis.
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from maverick_mcp.agents.base import INVESTOR_PERSONAS, PersonaAwareAgent
from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import AgentInitializationError
from maverick_mcp.memory.stores import ConversationStore
from maverick_mcp.workflows.state import SupervisorState

logger = logging.getLogger(__name__)
settings = get_settings()

# Query routing matrix for intelligent agent selection
ROUTING_MATRIX = {
    "market_screening": {
        "agents": ["market"],
        "primary": "market",
        "parallel": False,
        "confidence_threshold": 0.7,
        "synthesis_required": False,
    },
    "technical_analysis": {
        "agents": ["technical"],
        "primary": "technical",
        "parallel": False,
        "confidence_threshold": 0.8,
        "synthesis_required": False,
    },
    "stock_investment_decision": {
        "agents": ["market", "technical"],
        "primary": "technical",
        "parallel": True,
        "confidence_threshold": 0.85,
        "synthesis_required": True,
    },
    "portfolio_analysis": {
        "agents": ["market", "technical"],
        "primary": "market",
        "parallel": True,
        "confidence_threshold": 0.75,
        "synthesis_required": True,
    },
    "deep_research": {
        "agents": ["research"],  # Research agent handles comprehensive analysis
        "primary": "research",
        "parallel": False,
        "confidence_threshold": 0.9,
        "synthesis_required": False,  # Research agent provides complete analysis
    },
    "company_research": {
        "agents": ["research"],  # Dedicated company research
        "primary": "research",
        "parallel": False,
        "confidence_threshold": 0.85,
        "synthesis_required": False,
    },
    "sentiment_analysis": {
        "agents": ["research"],  # Market sentiment analysis
        "primary": "research",
        "parallel": False,
        "confidence_threshold": 0.8,
        "synthesis_required": False,
    },
    "risk_assessment": {
        "agents": ["market", "technical"],  # Future risk agent integration
        "primary": "market",
        "parallel": True,
        "confidence_threshold": 0.8,
        "synthesis_required": True,
    },
}


class QueryClassifier:
    """LLM-powered query classification with rule-based fallback."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def classify_query(self, query: str, persona: str) -> dict[str, Any]:
        """Classify query using LLM with structured output."""

        classification_prompt = f"""
        Analyze this financial query and classify it for multi-agent routing.

        Query: "{query}"
        Investor Persona: {persona}

        Classify into one of these categories:
        1. market_screening - Finding stocks, sector analysis, market breadth
        2. technical_analysis - Chart patterns, indicators, entry/exit points
        3. stock_investment_decision - Complete analysis of specific stock(s)
        4. portfolio_analysis - Portfolio optimization, risk assessment
        5. deep_research - Fundamental analysis, company research, news analysis
        6. risk_assessment - Position sizing, risk management, portfolio risk

        Consider the complexity and return classification with confidence.

        Return ONLY valid JSON in this exact format:
        {{
            "category": "category_name",
            "confidence": 0.85,
            "required_agents": ["agent1", "agent2"],
            "complexity": "simple",
            "estimated_execution_time_ms": 30000,
            "parallel_capable": true,
            "reasoning": "Brief explanation of classification"
        }}
        """

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a financial query classifier. Return only valid JSON."
                    ),
                    HumanMessage(content=classification_prompt),
                ]
            )

            # Parse LLM response
            import json

            classification = json.loads(response.content.strip())

            # Validate and enhance with routing matrix
            category = classification.get("category", "stock_investment_decision")
            routing_config = ROUTING_MATRIX.get(
                category, ROUTING_MATRIX["stock_investment_decision"]
            )

            return {
                **classification,
                "routing_config": routing_config,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using rule-based fallback")
            return self._rule_based_fallback(query, persona)

    def _rule_based_fallback(self, query: str, persona: str) -> dict[str, Any]:
        """Rule-based classification fallback."""
        query_lower = query.lower()

        # Simple keyword-based classification
        if any(
            word in query_lower for word in ["screen", "find stocks", "scan", "search"]
        ):
            category = "market_screening"
        elif any(
            word in query_lower
            for word in ["chart", "technical", "rsi", "macd", "pattern"]
        ):
            category = "technical_analysis"
        elif any(
            word in query_lower for word in ["portfolio", "allocation", "diversif"]
        ):
            category = "portfolio_analysis"
        elif any(
            word in query_lower
            for word in ["research", "fundamental", "news", "earnings"]
        ):
            category = "deep_research"
        elif any(
            word in query_lower
            for word in ["company", "business", "competitive", "industry"]
        ):
            category = "company_research"
        elif any(
            word in query_lower for word in ["sentiment", "opinion", "mood", "feeling"]
        ):
            category = "sentiment_analysis"
        elif any(
            word in query_lower for word in ["risk", "position size", "stop loss"]
        ):
            category = "risk_assessment"
        else:
            category = "stock_investment_decision"

        routing_config = ROUTING_MATRIX[category]

        return {
            "category": category,
            "confidence": 0.6,
            "required_agents": routing_config["agents"],
            "complexity": "moderate",
            "estimated_execution_time_ms": 60000,
            "parallel_capable": routing_config["parallel"],
            "reasoning": "Rule-based classification fallback",
            "routing_config": routing_config,
            "timestamp": datetime.now(),
        }


class ResultSynthesizer:
    """Synthesize results from multiple agents with conflict resolution."""

    def __init__(self, llm: BaseChatModel, persona):
        self.llm = llm
        self.persona = persona

    async def synthesize_results(
        self,
        agent_results: dict[str, Any],
        query_type: str,
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Synthesize final recommendation from agent results."""

        # Calculate agent weights based on query type and persona
        weights = self._calculate_agent_weights(query_type, agent_results)

        # Create synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            agent_results, weights, query_type, conflicts
        )

        # Use LLM to synthesize coherent response
        synthesis_response = await self.llm.ainvoke(
            [
                SystemMessage(content="You are a financial analysis synthesizer."),
                HumanMessage(content=synthesis_prompt),
            ]
        )

        return {
            "synthesis": synthesis_response.content,
            "weights_applied": weights,
            "conflicts_resolved": len(conflicts),
            "confidence_score": self._calculate_overall_confidence(
                agent_results, weights
            ),
            "contributing_agents": list(agent_results.keys()),
            "persona_alignment": self._assess_persona_alignment(
                synthesis_response.content
            ),
        }

    def _calculate_agent_weights(
        self, query_type: str, agent_results: dict
    ) -> dict[str, float]:
        """Calculate weights for agent results based on context."""
        base_weights = {
            "market_screening": {"market": 0.9, "technical": 0.1},
            "technical_analysis": {"market": 0.2, "technical": 0.8},
            "stock_investment_decision": {"market": 0.4, "technical": 0.6},
            "portfolio_analysis": {"market": 0.6, "technical": 0.4},
            "deep_research": {"research": 1.0},
            "company_research": {"research": 1.0},
            "sentiment_analysis": {"research": 1.0},
            "risk_assessment": {"market": 0.3, "technical": 0.3, "risk": 0.4},
        }

        weights = base_weights.get(query_type, {"market": 0.5, "technical": 0.5})

        # Adjust weights based on agent confidence scores
        for agent, base_weight in weights.items():
            if agent in agent_results:
                confidence = agent_results[agent].get("confidence_score", 0.5)
                weights[agent] = base_weight * (0.5 + confidence * 0.5)

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _build_synthesis_prompt(
        self,
        agent_results: dict[str, Any],
        weights: dict[str, float],
        query_type: str,
        conflicts: list[dict[str, Any]],
    ) -> str:
        """Build synthesis prompt for LLM."""

        prompt = f"""
        Synthesize a comprehensive financial analysis response from multiple specialized agents.

        Query Type: {query_type}
        Investor Persona: {self.persona.name} - {", ".join(self.persona.characteristics)}

        Agent Results:
        """

        for agent, result in agent_results.items():
            weight = weights.get(agent, 0.0)
            prompt += f"\n{agent.upper()} Agent (Weight: {weight:.2f}):\n"
            prompt += f"  - Confidence: {result.get('confidence_score', 0.5)}\n"
            prompt += (
                f"  - Analysis: {result.get('analysis', 'No analysis provided')}\n"
            )
            if "recommendations" in result:
                prompt += f"  - Recommendations: {result['recommendations']}\n"

        if conflicts:
            prompt += f"\nConflicts Detected ({len(conflicts)}):\n"
            for i, conflict in enumerate(conflicts, 1):
                prompt += f"{i}. {conflict}\n"

        prompt += f"""

        Please synthesize these results into a coherent, actionable response that:
        1. Weighs agent inputs according to their weights and confidence scores
        2. Resolves any conflicts using the {self.persona.name} investor perspective
        3. Provides clear, actionable recommendations aligned with {self.persona.name} characteristics
        4. Includes appropriate risk disclaimers
        5. Maintains professional, confident tone

        Focus on actionable insights for the {self.persona.name} investor profile.
        """

        return prompt

    def _calculate_overall_confidence(
        self, agent_results: dict, weights: dict[str, float]
    ) -> float:
        """Calculate weighted overall confidence score."""
        total_confidence = 0.0
        total_weight = 0.0

        for agent, weight in weights.items():
            if agent in agent_results:
                confidence = agent_results[agent].get("confidence_score", 0.5)
                total_confidence += confidence * weight
                total_weight += weight

        return total_confidence / total_weight if total_weight > 0 else 0.5

    def _assess_persona_alignment(self, synthesis_content: str) -> float:
        """Assess how well synthesis aligns with investor persona."""
        # Simple keyword-based alignment scoring
        persona_keywords = {
            "conservative": ["stable", "dividend", "low-risk", "preservation"],
            "moderate": ["balanced", "diversified", "moderate", "growth"],
            "aggressive": ["growth", "momentum", "high-return", "opportunity"],
        }

        keywords = persona_keywords.get(self.persona.name.lower(), [])
        content_lower = synthesis_content.lower()

        alignment_score = sum(1 for keyword in keywords if keyword in content_lower)
        return min(alignment_score / len(keywords) if keywords else 0.5, 1.0)


class SupervisorAgent(PersonaAwareAgent):
    """
    Multi-agent supervisor using 2025 LangGraph patterns.

    Orchestrates MarketAnalysisAgent, TechnicalAnalysisAgent, and future DeepResearchAgent
    with intelligent routing, result synthesis, and conflict resolution.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        agents: dict[str, PersonaAwareAgent],
        persona: str = "moderate",
        checkpointer: MemorySaver | None = None,
        ttl_hours: int = 1,
        routing_strategy: str = "llm_powered",
        synthesis_mode: str = "weighted",
        conflict_resolution: str = "confidence_based",
        max_iterations: int = 5,
    ):
        """Initialize supervisor with existing agent instances."""

        if not agents:
            raise AgentInitializationError(
                agent_type="SupervisorAgent",
                reason="No agents provided for supervision",
            )

        # Store agent references
        self.agents = agents
        self.market_agent = agents.get("market")
        self.technical_agent = agents.get("technical")
        self.research_agent = agents.get("research")  # DeepResearchAgent integration

        # Configuration
        self.routing_strategy = routing_strategy
        self.synthesis_mode = synthesis_mode
        self.conflict_resolution = conflict_resolution
        self.max_iterations = max_iterations

        # Ensure all agents use the same persona
        persona_obj = INVESTOR_PERSONAS.get(persona, INVESTOR_PERSONAS["moderate"])
        for agent in agents.values():
            if hasattr(agent, "persona"):
                agent.persona = persona_obj

        # Get supervisor-specific tools
        supervisor_tools = self._get_supervisor_tools()

        # Initialize base class
        super().__init__(
            llm=llm,
            tools=supervisor_tools,
            persona=persona,
            checkpointer=checkpointer or MemorySaver(),
            ttl_hours=ttl_hours,
        )

        # Initialize components
        self.conversation_store = ConversationStore(ttl_hours=ttl_hours)
        self.query_classifier = QueryClassifier(llm)
        self.result_synthesizer = ResultSynthesizer(llm, self.persona)

        logger.info(
            f"SupervisorAgent initialized with {len(agents)} agents: {list(agents.keys())}"
        )

    def get_state_schema(self) -> type:
        """Return SupervisorState schema."""
        return SupervisorState

    def _get_supervisor_tools(self) -> list[BaseTool]:
        """Get tools specific to supervision and coordination."""
        from langchain_core.tools import tool

        tools = []

        if self.market_agent:

            @tool
            async def query_market_agent(
                query: str,
                session_id: str,
                screening_strategy: str = "momentum",
                max_results: int = 20,
            ) -> dict[str, Any]:
                """Query the market analysis agent for stock screening and market analysis."""
                try:
                    return await self.market_agent.analyze_market(
                        query=query,
                        session_id=session_id,
                        screening_strategy=screening_strategy,
                        max_results=max_results,
                    )
                except Exception as e:
                    return {"error": f"Market agent error: {str(e)}"}

            tools.append(query_market_agent)

        if self.technical_agent:

            @tool
            async def query_technical_agent(
                symbol: str, timeframe: str = "1d", indicators: list[str] | None = None
            ) -> dict[str, Any]:
                """Query the technical analysis agent for chart analysis and indicators."""
                try:
                    if indicators is None:
                        indicators = ["sma_20", "rsi", "macd"]

                    return await self.technical_agent.analyze_stock(
                        symbol=symbol, timeframe=timeframe, indicators=indicators
                    )
                except Exception as e:
                    return {"error": f"Technical agent error: {str(e)}"}

            tools.append(query_technical_agent)

        if self.research_agent:

            @tool
            async def query_research_agent(
                query: str,
                session_id: str,
                research_scope: str = "comprehensive",
                max_sources: int = 50,
                timeframe: str = "1m",
            ) -> dict[str, Any]:
                """Query the deep research agent for comprehensive research and analysis."""
                try:
                    return await self.research_agent.research_topic(
                        query=query,
                        session_id=session_id,
                        research_scope=research_scope,
                        max_sources=max_sources,
                        timeframe=timeframe,
                    )
                except Exception as e:
                    return {"error": f"Research agent error: {str(e)}"}

            @tool
            async def analyze_company_research(
                symbol: str, session_id: str, include_competitive: bool = True
            ) -> dict[str, Any]:
                """Perform comprehensive company research and fundamental analysis."""
                try:
                    return await self.research_agent.research_company_comprehensive(
                        symbol=symbol,
                        session_id=session_id,
                        include_competitive_analysis=include_competitive,
                    )
                except Exception as e:
                    return {"error": f"Company research error: {str(e)}"}

            @tool
            async def analyze_market_sentiment_research(
                topic: str, session_id: str, timeframe: str = "1w"
            ) -> dict[str, Any]:
                """Analyze market sentiment using deep research capabilities."""
                try:
                    return await self.research_agent.analyze_market_sentiment(
                        topic=topic, session_id=session_id, timeframe=timeframe
                    )
                except Exception as e:
                    return {"error": f"Sentiment analysis error: {str(e)}"}

            tools.extend(
                [
                    query_research_agent,
                    analyze_company_research,
                    analyze_market_sentiment_research,
                ]
            )

        return tools

    def _build_graph(self):
        """Build supervisor graph with multi-agent coordination."""
        workflow = StateGraph(SupervisorState)

        # Core supervisor nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("create_execution_plan", self._create_execution_plan)
        workflow.add_node("route_to_agents", self._route_to_agents)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("resolve_conflicts", self._resolve_conflicts)
        workflow.add_node("synthesize_response", self._synthesize_response)

        # Agent invocation nodes
        if self.market_agent:
            workflow.add_node("invoke_market_agent", self._invoke_market_agent)
        if self.technical_agent:
            workflow.add_node("invoke_technical_agent", self._invoke_technical_agent)
        if self.research_agent:
            workflow.add_node("invoke_research_agent", self._invoke_research_agent)

        # Coordination nodes
        workflow.add_node("parallel_coordinator", self._parallel_coordinator)

        # Tool node
        if self.tools:
            from langgraph.prebuilt import ToolNode

            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)

        # Define workflow edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "create_execution_plan")
        workflow.add_edge("create_execution_plan", "route_to_agents")

        # Conditional routing based on execution plan
        workflow.add_conditional_edges(
            "route_to_agents",
            self._route_decision,
            {
                "market_only": "invoke_market_agent"
                if self.market_agent
                else "synthesize_response",
                "technical_only": "invoke_technical_agent"
                if self.technical_agent
                else "synthesize_response",
                "research_only": "invoke_research_agent"
                if self.research_agent
                else "synthesize_response",
                "parallel_execution": "parallel_coordinator",
                "use_tools": "tools" if self.tools else "synthesize_response",
                "synthesize": "synthesize_response",
            },
        )

        # Agent result collection
        if self.market_agent:
            workflow.add_edge("invoke_market_agent", "aggregate_results")
        if self.technical_agent:
            workflow.add_edge("invoke_technical_agent", "aggregate_results")
        if self.research_agent:
            workflow.add_edge("invoke_research_agent", "aggregate_results")

        workflow.add_edge("parallel_coordinator", "aggregate_results")

        if self.tools:
            workflow.add_edge("tools", "aggregate_results")

        # Conflict detection and resolution
        workflow.add_conditional_edges(
            "aggregate_results",
            self._check_conflicts,
            {"resolve": "resolve_conflicts", "synthesize": "synthesize_response"},
        )

        workflow.add_edge("resolve_conflicts", "synthesize_response")
        workflow.add_edge("synthesize_response", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # Workflow node implementations will continue...
    # (The rest of the implementation follows the same pattern)

    async def coordinate_agents(
        self, query: str, session_id: str, **kwargs
    ) -> dict[str, Any]:
        """
        Main entry point for multi-agent coordination.

        Args:
            query: User query requiring multiple agents
            session_id: Session identifier
            **kwargs: Additional parameters

        Returns:
            Coordinated response from multiple agents
        """
        start_time = datetime.now()

        # Initialize supervisor state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "persona": self.persona.name,
            "session_id": session_id,
            "timestamp": datetime.now(),
            "query_classification": {},
            "execution_plan": [],
            "current_subtask_index": 0,
            "routing_strategy": self.routing_strategy,
            "active_agents": [],
            "agent_results": {},
            "agent_confidence": {},
            "agent_execution_times": {},
            "agent_errors": {},
            "workflow_status": "planning",
            "parallel_execution": False,
            "dependency_graph": {},
            "max_iterations": self.max_iterations,
            "current_iteration": 0,
            "conflicts_detected": [],
            "conflict_resolution": {},
            "synthesis_weights": {},
            "final_recommendation_confidence": 0.0,
            "synthesis_mode": self.synthesis_mode,
            "total_execution_time_ms": 0.0,
            "agent_coordination_overhead_ms": 0.0,
            "synthesis_time_ms": 0.0,
            "cache_utilization": {},
            "api_calls_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            # Legacy fields initialized as None for backward compatibility
            "query_type": None,
            "subtasks": None,
            "current_subtask": None,
            "workflow_plan": None,
            "completed_steps": None,
            "pending_steps": None,
            "final_recommendations": None,
            "confidence_scores": None,
            "risk_warnings": None,
        }

        # Add any additional parameters
        initial_state.update(kwargs)

        # Execute supervision workflow
        try:
            result = await self.graph.ainvoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": session_id,
                        "checkpoint_ns": "supervisor",
                    }
                },
            )

            # Calculate total execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result["total_execution_time_ms"] = execution_time

            return self._format_supervisor_response(result)

        except Exception as e:
            logger.error(f"Error in supervisor coordination: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_execution_time_ms": (datetime.now() - start_time).total_seconds()
                * 1000,
                "agent_type": "supervisor",
            }

    def _format_supervisor_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Format supervisor response for consistent output."""
        return {
            "status": "success",
            "agent_type": "supervisor",
            "persona": result.get("persona"),
            "query_classification": result.get("query_classification", {}),
            "agents_used": result.get("active_agents", []),
            "synthesis": result.get("messages", [])[-1].content
            if result.get("messages")
            else "No synthesis available",
            "confidence_score": result.get("final_recommendation_confidence", 0.0),
            "execution_time_ms": result.get("total_execution_time_ms", 0.0),
            "conflicts_resolved": len(result.get("conflicts_detected", [])),
            "workflow_status": result.get("workflow_status", "completed"),
        }

    # Placeholder implementations for workflow nodes
    # These will be implemented based on the specific node logic

    async def _analyze_query(self, state: SupervisorState) -> Command:
        """Analyze query to determine routing strategy and requirements."""
        query = state["messages"][-1].content if state["messages"] else ""

        # Classify the query
        classification = await self.query_classifier.classify_query(
            query, state["persona"]
        )

        return Command(
            goto="create_execution_plan",
            update={
                "query_classification": classification,
                "workflow_status": "analyzing",
            },
        )

    async def _create_execution_plan(self, state: SupervisorState) -> Command:
        """Create execution plan based on query classification."""
        classification = state["query_classification"]

        # Create execution plan based on classification
        execution_plan = [
            {
                "task_id": "main_analysis",
                "agents": classification.get("required_agents", ["market"]),
                "parallel": classification.get("parallel_capable", False),
                "priority": 1,
            }
        ]

        return Command(
            goto="route_to_agents",
            update={"execution_plan": execution_plan, "workflow_status": "planning"},
        )

    async def _route_to_agents(self, state: SupervisorState) -> Command:
        """Route query to appropriate agents based on execution plan."""
        return Command(
            goto="parallel_execution", update={"workflow_status": "executing"}
        )

    async def _route_decision(self, state: SupervisorState) -> str:
        """Decide routing strategy based on state."""
        classification = state.get("query_classification", {})
        required_agents = classification.get("required_agents", ["market"])
        parallel = classification.get("parallel_capable", False)

        if len(required_agents) == 1:
            agent = required_agents[0]
            if agent == "market" and self.market_agent:
                return "market_only"
            elif agent == "technical" and self.technical_agent:
                return "technical_only"
            elif agent == "research" and self.research_agent:
                return "research_only"
        elif len(required_agents) > 1 and parallel:
            return "parallel_execution"

        return "synthesize"

    async def _parallel_coordinator(self, state: SupervisorState) -> Command:
        """Coordinate parallel execution of multiple agents."""
        # This would implement parallel agent coordination
        # For now, return to aggregation
        return Command(
            goto="aggregate_results", update={"workflow_status": "aggregating"}
        )

    async def _invoke_market_agent(self, state: SupervisorState) -> Command:
        """Invoke market analysis agent."""
        if not self.market_agent:
            return Command(
                goto="aggregate_results",
                update={"agent_errors": {"market": "Market agent not available"}},
            )

        try:
            query = state["messages"][-1].content if state["messages"] else ""
            result = await self.market_agent.analyze_market(
                query=query, session_id=state["session_id"]
            )

            return Command(
                goto="aggregate_results",
                update={
                    "agent_results": {"market": result},
                    "active_agents": ["market"],
                },
            )

        except Exception as e:
            return Command(
                goto="aggregate_results",
                update={
                    "agent_errors": {"market": str(e)},
                    "active_agents": ["market"],
                },
            )

    async def _invoke_technical_agent(self, state: SupervisorState) -> Command:
        """Invoke technical analysis agent."""
        if not self.technical_agent:
            return Command(
                goto="aggregate_results",
                update={"agent_errors": {"technical": "Technical agent not available"}},
            )

        # This would implement technical agent invocation
        return Command(
            goto="aggregate_results", update={"active_agents": ["technical"]}
        )

    async def _invoke_research_agent(self, state: SupervisorState) -> Command:
        """Invoke deep research agent (future implementation)."""
        if not self.research_agent:
            return Command(
                goto="aggregate_results",
                update={"agent_errors": {"research": "Research agent not available"}},
            )

        # Future implementation
        return Command(goto="aggregate_results", update={"active_agents": ["research"]})

    async def _aggregate_results(self, state: SupervisorState) -> Command:
        """Aggregate results from all agents."""
        return Command(
            goto="synthesize_response", update={"workflow_status": "synthesizing"}
        )

    def _check_conflicts(self, state: SupervisorState) -> str:
        """Check if there are conflicts between agent results."""
        conflicts = state.get("conflicts_detected", [])
        return "resolve" if conflicts else "synthesize"

    async def _resolve_conflicts(self, state: SupervisorState) -> Command:
        """Resolve conflicts between agent recommendations."""
        return Command(
            goto="synthesize_response",
            update={"conflict_resolution": {"strategy": "confidence_based"}},
        )

    async def _synthesize_response(self, state: SupervisorState) -> Command:
        """Synthesize final response from agent results."""
        agent_results = state.get("agent_results", {})
        conflicts = state.get("conflicts_detected", [])
        classification = state.get("query_classification", {})

        if agent_results:
            synthesis = await self.result_synthesizer.synthesize_results(
                agent_results=agent_results,
                query_type=classification.get("category", "stock_investment_decision"),
                conflicts=conflicts,
            )

            return Command(
                goto="__end__",
                update={
                    "final_recommendation_confidence": synthesis["confidence_score"],
                    "synthesis_weights": synthesis["weights_applied"],
                    "workflow_status": "completed",
                    "messages": state["messages"]
                    + [HumanMessage(content=synthesis["synthesis"])],
                },
            )
        else:
            return Command(
                goto="__end__",
                update={
                    "workflow_status": "completed",
                    "messages": state["messages"]
                    + [
                        HumanMessage(content="No agent results available for synthesis")
                    ],
                },
            )
