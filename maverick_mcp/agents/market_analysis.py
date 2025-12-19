"""
Market Analysis Agent using LangGraph best practices with professional features.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from maverick_mcp.agents.circuit_breaker import circuit_manager
from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import (
    AgentInitializationError,
    PersonaConfigurationError,
    ToolRegistrationError,
)
from maverick_mcp.langchain_tools import get_tool_registry
from maverick_mcp.memory import ConversationStore
from maverick_mcp.tools.risk_management import (
    PositionSizeTool,
    RiskMetricsTool,
    TechnicalStopsTool,
)
from maverick_mcp.tools.sentiment_analysis import (
    MarketBreadthTool,
    NewsSentimentTool,
    SectorSentimentTool,
)
from maverick_mcp.workflows.state import MarketAnalysisState

from .base import PersonaAwareAgent

logger = logging.getLogger(__name__)
settings = get_settings()


class MarketAnalysisAgent(PersonaAwareAgent):
    """
    Professional market analysis agent with advanced screening and risk assessment.

    Features:
    - Multi-strategy screening (momentum, mean reversion, breakout)
    - Sector rotation analysis
    - Market regime detection
    - Risk-adjusted recommendations
    - Real-time sentiment integration
    - Circuit breaker protection for API calls
    """

    VALID_PERSONAS = ["conservative", "moderate", "aggressive", "day_trader"]

    def __init__(
        self,
        llm,
        persona: str = "moderate",
        ttl_hours: int | None = None,
    ):
        """
        Initialize market analysis agent.

        Args:
            llm: Language model
            persona: Investor persona
            ttl_hours: Cache TTL in hours (uses config default if None)
            postgres_url: Optional PostgreSQL URL for checkpointing

        Raises:
            PersonaConfigurationError: If persona is invalid
            AgentInitializationError: If initialization fails
        """
        try:
            # Validate persona
            if persona.lower() not in self.VALID_PERSONAS:
                raise PersonaConfigurationError(
                    persona=persona, valid_personas=self.VALID_PERSONAS
                )

            # Store persona temporarily for tool configuration
            self._temp_persona = persona.lower()

            # Get comprehensive tool set
            tools = self._get_comprehensive_tools()

            if not tools:
                raise AgentInitializationError(
                    agent_type="MarketAnalysisAgent",
                    reason="No tools available for initialization",
                )

            # Use default TTL from config if not provided
            if ttl_hours is None:
                ttl_hours = settings.agent.conversation_cache_ttl_hours

            # Initialize with MemorySaver
            super().__init__(
                llm=llm,
                tools=tools,
                persona=persona,
                checkpointer=MemorySaver(),
                ttl_hours=ttl_hours,
            )

        except (PersonaConfigurationError, AgentInitializationError):
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MarketAnalysisAgent: {str(e)}")
            error = AgentInitializationError(
                agent_type="MarketAnalysisAgent",
                reason=str(e),
            )
            error.context["original_error"] = type(e).__name__
            raise error

        # Initialize conversation store
        self.conversation_store = ConversationStore(ttl_hours=ttl_hours)

        # Circuit breakers for external APIs
        self.circuit_breakers = {
            "screening": None,
            "sentiment": None,
            "market_data": None,
        }

    def _get_comprehensive_tools(self) -> list[BaseTool]:
        """Get comprehensive set of market analysis tools.

        Returns:
            List of configured tools

        Raises:
            ToolRegistrationError: If critical tools cannot be loaded
        """
        try:
            registry = get_tool_registry()
        except Exception as e:
            logger.error(f"Failed to get tool registry: {str(e)}")
            raise ToolRegistrationError(tool_name="registry", reason=str(e))

        # Core screening tools
        screening_tools = [
            registry.get_tool("get_maverick_stocks"),
            registry.get_tool("get_maverick_bear_stocks"),
            registry.get_tool("get_supply_demand_breakouts"),
            registry.get_tool("get_all_screening_recommendations"),
        ]

        # Technical analysis tools
        technical_tools = [
            registry.get_tool("get_technical_indicators"),
            registry.get_tool("calculate_support_resistance"),
            registry.get_tool("detect_chart_patterns"),
        ]

        # Market data tools
        market_tools = [
            registry.get_tool("get_market_movers"),
            registry.get_tool("get_sector_performance"),
            registry.get_tool("get_market_indices"),
        ]

        # Risk management tools (persona-aware)
        risk_tools = [
            PositionSizeTool(),
            TechnicalStopsTool(),
            RiskMetricsTool(),
        ]

        # Sentiment analysis tools
        sentiment_tools = [
            NewsSentimentTool(),
            MarketBreadthTool(),
            SectorSentimentTool(),
        ]

        # Combine all tools and filter None
        all_tools = (
            screening_tools
            + technical_tools
            + market_tools
            + risk_tools
            + sentiment_tools
        )
        tools = [t for t in all_tools if t is not None]

        # Configure persona for PersonaAwareTools
        for tool in tools:
            try:
                if hasattr(tool, "set_persona"):
                    tool.set_persona(self._temp_persona)
            except Exception as e:
                logger.warning(
                    f"Failed to set persona for tool {tool.__class__.__name__}: {str(e)}"
                )
                # Continue with other tools

        if not tools:
            logger.warning("No tools available for market analysis")
            return []

        logger.info(f"Loaded {len(tools)} tools for {self._temp_persona} persona")
        return tools

    def get_state_schema(self) -> type:
        """Return enhanced state schema for market analysis."""
        return MarketAnalysisState

    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for professional market analysis."""
        base_prompt = super()._build_system_prompt()

        market_prompt = f"""

You are a professional market analyst specializing in systematic screening and analysis.
Current market date: {datetime.now().strftime("%Y-%m-%d")}

## Core Responsibilities:

1. **Multi-Strategy Screening**:
   - Momentum: High RS stocks breaking out on volume
   - Mean Reversion: Oversold quality stocks at support
   - Breakout: Stocks clearing resistance with volume surge
   - Trend Following: Stocks in established uptrends

2. **Market Regime Analysis**:
   - Identify current market regime (bull/bear/sideways)
   - Analyze sector rotation patterns
   - Monitor breadth indicators and sentiment
   - Detect risk-on vs risk-off environments

3. **Risk-Adjusted Selection**:
   - Filter stocks by persona risk tolerance
   - Calculate position sizes using Kelly Criterion
   - Set appropriate stop losses using ATR
   - Consider correlation and portfolio heat

4. **Professional Reporting**:
   - Provide actionable entry/exit levels
   - Include risk/reward ratios
   - Highlight key catalysts and risks
   - Suggest portfolio allocation

## Screening Criteria by Persona:

**Conservative ({self.persona.name if self.persona.name == "Conservative" else "N/A"})**:
- Large-cap stocks (>$10B market cap)
- Dividend yield > 2%
- Low debt/equity < 1.5
- Beta < 1.2
- Established uptrends only

**Moderate ({self.persona.name if self.persona.name == "Moderate" else "N/A"})**:
- Mid to large-cap stocks (>$2B)
- Balanced growth/value metrics
- Moderate volatility accepted
- Mix of dividend and growth stocks

**Aggressive ({self.persona.name if self.persona.name == "Aggressive" else "N/A"})**:
- All market caps considered
- High growth rates prioritized
- Momentum and relative strength focus
- Higher volatility tolerated

**Day Trader ({self.persona.name if self.persona.name == "Day Trader" else "N/A"})**:
- High liquidity (>1M avg volume)
- Tight spreads (<0.1%)
- High ATR for movement
- Technical patterns emphasized

## Analysis Framework:

1. Start with market regime assessment
2. Identify leading/lagging sectors
3. Screen for stocks matching criteria
4. Apply technical analysis filters
5. Calculate risk metrics
6. Generate recommendations with specific levels

Remember to:
- Cite specific data points
- Explain your reasoning
- Highlight risks clearly
- Provide actionable insights
- Consider time horizon
"""

        return base_prompt + market_prompt

    def _build_graph(self):
        """Build enhanced graph with multiple analysis nodes."""
        workflow = StateGraph(MarketAnalysisState)

        # Add specialized nodes with unique names
        workflow.add_node("analyze_market_regime", self._analyze_market_regime)
        workflow.add_node("analyze_sectors", self._analyze_sectors)
        workflow.add_node("run_screening", self._run_screening)
        workflow.add_node("assess_risks", self._assess_risks)
        workflow.add_node("agent", self._agent_node)

        # Create tool node if tools available
        if self.tools:
            from langgraph.prebuilt import ToolNode

            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)

        # Define flow
        workflow.add_edge(START, "analyze_market_regime")
        workflow.add_edge("analyze_market_regime", "analyze_sectors")
        workflow.add_edge("analyze_sectors", "run_screening")
        workflow.add_edge("run_screening", "assess_risks")
        workflow.add_edge("assess_risks", "agent")

        if self.tools:
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "continue": "tools",
                    "end": END,
                },
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _analyze_market_regime(
        self, state: MarketAnalysisState
    ) -> dict[str, Any]:
        """Analyze current market regime."""
        try:
            # Use market breadth tool
            breadth_tool = next(
                (t for t in self.tools if t.name == "analyze_market_breadth"), None
            )

            if breadth_tool:
                circuit_breaker = await circuit_manager.get_or_create("market_data")

                async def get_breadth():
                    return await breadth_tool.ainvoke({"index": "SPY"})

                breadth_data = await circuit_breaker.call(get_breadth)

                # Parse results to determine regime
                # Handle both string and dict responses
                if isinstance(breadth_data, str):
                    # Try to extract sentiment from string response
                    if "Bullish" in breadth_data:
                        state["market_regime"] = "bullish"
                    elif "Bearish" in breadth_data:
                        state["market_regime"] = "bearish"
                    else:
                        state["market_regime"] = "neutral"
                elif (
                    isinstance(breadth_data, dict) and "market_breadth" in breadth_data
                ):
                    sentiment = breadth_data["market_breadth"].get(
                        "sentiment", "Neutral"
                    )
                    state["market_regime"] = sentiment.lower()
                else:
                    state["market_regime"] = "neutral"
            else:
                state["market_regime"] = "neutral"

        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            state["market_regime"] = "unknown"

        state["api_calls_made"] = state.get("api_calls_made", 0) + 1
        return {"market_regime": state.get("market_regime", "neutral")}

    async def _analyze_sectors(self, state: MarketAnalysisState) -> dict[str, Any]:
        """Analyze sector rotation patterns."""
        try:
            # Use sector sentiment tool
            sector_tool = next(
                (t for t in self.tools if t.name == "analyze_sector_sentiment"), None
            )

            if sector_tool:
                circuit_breaker = await circuit_manager.get_or_create("market_data")

                async def get_sectors():
                    return await sector_tool.ainvoke({})

                sector_data = await circuit_breaker.call(get_sectors)

                if "sector_rotation" in sector_data:
                    state["sector_rotation"] = sector_data["sector_rotation"]

                    # Extract leading sectors
                    leading = sector_data["sector_rotation"].get("leading_sectors", [])
                    if leading and state.get("sector_filter"):
                        # Prioritize screening in leading sectors
                        state["sector_filter"] = leading[0].get("name", "")

        except Exception as e:
            logger.error(f"Error analyzing sectors: {e}")

        state["api_calls_made"] = state.get("api_calls_made", 0) + 1
        return {"sector_rotation": state.get("sector_rotation", {})}

    async def _run_screening(self, state: MarketAnalysisState) -> dict[str, Any]:
        """Run multi-strategy screening."""
        try:
            # Determine which screening strategy based on market regime
            strategy = state.get("screening_strategy", "momentum")

            # Adjust strategy based on regime
            if state.get("market_regime") == "bearish" and strategy == "momentum":
                strategy = "mean_reversion"

            # Get appropriate screening tool
            tool_map = {
                "momentum": "get_maverick_stocks",
                "supply_demand_breakout": "get_supply_demand_breakouts",
                "bearish": "get_maverick_bear_stocks",
            }

            tool_name = tool_map.get(strategy, "get_maverick_stocks")
            screening_tool = next((t for t in self.tools if t.name == tool_name), None)

            if screening_tool:
                circuit_breaker = await circuit_manager.get_or_create("screening")

                async def run_screen():
                    return await screening_tool.ainvoke(
                        {"limit": state.get("max_results", 20)}
                    )

                results = await circuit_breaker.call(run_screen)

                if "stocks" in results:
                    symbols = [s.get("symbol") for s in results["stocks"]]
                    scores = {
                        s.get("symbol"): s.get("combined_score", 0)
                        for s in results["stocks"]
                    }

                    state["screened_symbols"] = symbols
                    state["screening_scores"] = scores
                    state["cache_hits"] += 1

        except Exception as e:
            logger.error(f"Error running screening: {e}")
            state["cache_misses"] += 1

        state["api_calls_made"] = state.get("api_calls_made", 0) + 1
        return {
            "screened_symbols": state.get("screened_symbols", []),
            "screening_scores": state.get("screening_scores", {}),
        }

    async def _assess_risks(self, state: MarketAnalysisState) -> dict[str, Any]:
        """Assess risks for screened symbols."""
        symbols = state.get("screened_symbols", [])[:5]  # Top 5 only

        if not symbols:
            return {}

        try:
            # Get risk metrics tool
            risk_tool = next(
                (t for t in self.tools if isinstance(t, RiskMetricsTool)), None
            )

            if risk_tool and len(symbols) > 1:
                # Calculate portfolio risk metrics
                risk_data = await risk_tool.ainvoke(
                    {"symbols": symbols, "lookback_days": 252}
                )

                # Store risk assessment
                state["conversation_context"]["risk_metrics"] = risk_data

        except Exception as e:
            logger.error(f"Error assessing risks: {e}")

        return {}

    async def analyze_market(
        self,
        query: str,
        session_id: str,
        screening_strategy: str = "momentum",
        max_results: int = 20,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Analyze market with specific screening parameters.

        Enhanced with caching, circuit breakers, and comprehensive analysis.
        """
        start_time = datetime.now()

        # Check cache first
        cached = self._check_enhanced_cache(query, session_id, screening_strategy)
        if cached:
            return cached

        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "persona": self.persona.name,
            "session_id": session_id,
            "screening_strategy": screening_strategy,
            "max_results": max_results,
            "timestamp": datetime.now(),
            "token_count": 0,
            "error": None,
            "analyzed_stocks": {},
            "key_price_levels": {},
            "last_analysis_time": {},
            "conversation_context": {},
            "execution_time_ms": None,
            "api_calls_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Update with any additional parameters
        initial_state.update(kwargs)

        # Run the analysis
        result = await self.ainvoke(query, session_id, initial_state=initial_state)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        result["execution_time_ms"] = execution_time

        # Extract and cache results
        analysis_results = self._extract_enhanced_results(result)

        # Create same cache key as used in _check_enhanced_cache
        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:8]
        cache_key = f"{screening_strategy}_{query_hash}"

        self.conversation_store.save_analysis(
            session_id=session_id,
            symbol=cache_key,
            analysis_type=f"{screening_strategy}_analysis",
            data=analysis_results,
        )

        return analysis_results

    def _check_enhanced_cache(
        self, query: str, session_id: str, strategy: str
    ) -> dict[str, Any] | None:
        """Check for cached analysis with strategy awareness."""
        # Create a hash of the query to use as cache key
        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:8]
        cache_key = f"{strategy}_{query_hash}"

        cached = self.conversation_store.get_analysis(
            session_id=session_id,
            symbol=cache_key,
            analysis_type=f"{strategy}_analysis",
        )

        if cached and cached.get("data"):
            # Check cache age based on strategy
            timestamp = datetime.fromisoformat(cached["timestamp"])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60

            # Different cache durations for different strategies
            cache_durations = {
                "momentum": 15,  # 15 minutes for fast-moving
                "trending": 60,  # 1 hour for trend following
                "mean_reversion": 30,  # 30 minutes
            }

            max_age = cache_durations.get(strategy, 30)

            if age_minutes < max_age:
                logger.info(f"Using cached {strategy} analysis")
                return cached["data"]  # type: ignore

        return None

    def _extract_enhanced_results(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract comprehensive results from agent output."""
        state = result.get("state", {})

        # Get final message content
        messages = result.get("messages", [])
        content = messages[-1].content if messages else ""

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "query_type": "professional_market_analysis",
            "execution_metrics": {
                "execution_time_ms": result.get("execution_time_ms", 0),
                "api_calls": state.get("api_calls_made", 0),
                "cache_hits": state.get("cache_hits", 0),
                "cache_misses": state.get("cache_misses", 0),
            },
            "market_analysis": {
                "regime": state.get("market_regime", "unknown"),
                "sector_rotation": state.get("sector_rotation", {}),
                "breadth": state.get("market_breadth", {}),
                "sentiment": state.get("sentiment_indicators", {}),
            },
            "screening_results": {
                "strategy": state.get("screening_strategy", "momentum"),
                "symbols": state.get("screened_symbols", [])[:20],
                "scores": state.get("screening_scores", {}),
                "count": len(state.get("screened_symbols", [])),
                "metadata": state.get("symbol_metadata", {}),
            },
            "risk_assessment": state.get("conversation_context", {}).get(
                "risk_metrics", {}
            ),
            "recommendations": {
                "summary": content,
                "persona_adjusted": True,
                "risk_level": self.persona.name,
                "position_sizing": f"Max {self.persona.position_size_max * 100:.1f}% per position",
            },
        }
