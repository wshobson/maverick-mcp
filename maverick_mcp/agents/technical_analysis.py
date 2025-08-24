"""
Technical Analysis Agent with pattern recognition and multi-timeframe analysis.
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from maverick_mcp.agents.circuit_breaker import circuit_manager
from maverick_mcp.langchain_tools import get_tool_registry
from maverick_mcp.memory import ConversationStore
from maverick_mcp.tools.risk_management import TechnicalStopsTool
from maverick_mcp.workflows.state import TechnicalAnalysisState

from .base import PersonaAwareAgent

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent(PersonaAwareAgent):
    """
    Professional technical analysis agent with pattern recognition.

    Features:
    - Chart pattern detection (head & shoulders, triangles, flags)
    - Multi-timeframe analysis
    - Indicator confluence scoring
    - Support/resistance clustering
    - Volume profile analysis
    - LLM-powered technical narratives
    """

    def __init__(
        self,
        llm,
        persona: str = "moderate",
        ttl_hours: int = 1,
    ):
        """
        Initialize technical analysis agent.

        Args:
            llm: Language model
            persona: Investor persona
            ttl_hours: Cache TTL in hours
            postgres_url: Optional PostgreSQL URL for checkpointing
        """
        # Store persona temporarily for tool configuration
        self._temp_persona = persona

        # Get technical analysis tools
        tools = self._get_technical_tools()

        # Initialize with MemorySaver
        super().__init__(
            llm=llm,
            tools=tools,
            persona=persona,
            checkpointer=MemorySaver(),
            ttl_hours=ttl_hours,
        )

        # Initialize conversation store
        self.conversation_store = ConversationStore(ttl_hours=ttl_hours)

    def _get_technical_tools(self) -> list[BaseTool]:
        """Get comprehensive technical analysis tools."""
        registry = get_tool_registry()

        # Core technical tools
        technical_tools = [
            registry.get_tool("get_technical_indicators"),
            registry.get_tool("calculate_support_resistance"),
            registry.get_tool("detect_chart_patterns"),
            registry.get_tool("calculate_moving_averages"),
            registry.get_tool("calculate_oscillators"),
        ]

        # Price action tools
        price_tools = [
            registry.get_tool("get_stock_price"),
            registry.get_tool("get_stock_history"),
            registry.get_tool("get_intraday_data"),
        ]

        # Volume analysis tools
        volume_tools = [
            registry.get_tool("analyze_volume_profile"),
            registry.get_tool("detect_volume_patterns"),
        ]

        # Risk tools
        risk_tools = [
            TechnicalStopsTool(),
        ]

        # Combine and filter
        all_tools = technical_tools + price_tools + volume_tools + risk_tools
        tools = [t for t in all_tools if t is not None]

        # Configure persona for PersonaAwareTools
        for tool in tools:
            if hasattr(tool, "set_persona"):
                tool.set_persona(self._temp_persona)

        if not tools:
            logger.warning("No technical tools available, using mock tools")
            tools = self._create_mock_tools()

        return tools

    def get_state_schema(self) -> type:
        """Return enhanced state schema for technical analysis."""
        return TechnicalAnalysisState

    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for technical analysis."""
        base_prompt = super()._build_system_prompt()

        technical_prompt = f"""

You are a professional technical analyst specializing in pattern recognition and multi-timeframe analysis.
Current date: {datetime.now().strftime("%Y-%m-%d")}

## Core Responsibilities:

1. **Pattern Recognition**:
   - Chart patterns: Head & Shoulders, Triangles, Flags, Wedges
   - Candlestick patterns: Doji, Hammer, Engulfing, etc.
   - Support/Resistance: Dynamic and static levels
   - Trend lines and channels

2. **Multi-Timeframe Analysis**:
   - Align signals across daily, hourly, and 5-minute charts
   - Identify confluences between timeframes
   - Spot divergences early
   - Time entries based on lower timeframe setups

3. **Indicator Analysis**:
   - Trend: Moving averages, ADX, MACD
   - Momentum: RSI, Stochastic, CCI
   - Volume: OBV, Volume Profile, VWAP
   - Volatility: Bollinger Bands, ATR, Keltner Channels

4. **Trade Setup Construction**:
   - Entry points with specific triggers
   - Stop loss placement using ATR or structure
   - Profit targets based on measured moves
   - Risk/Reward ratio calculation

## Analysis Framework by Persona:

**Conservative ({self.persona.name if self.persona.name == "Conservative" else "N/A"})**:
- Wait for confirmed patterns only
- Use wider stops above/below structure
- Target 1.5:1 risk/reward minimum
- Focus on daily/weekly timeframes

**Moderate ({self.persona.name if self.persona.name == "Moderate" else "N/A"})**:
- Balance pattern quality with opportunity
- Standard ATR-based stops
- Target 2:1 risk/reward
- Use daily/4H timeframes

**Aggressive ({self.persona.name if self.persona.name == "Aggressive" else "N/A"})**:
- Trade emerging patterns
- Tighter stops for larger positions
- Target 3:1+ risk/reward
- Include intraday timeframes

**Day Trader ({self.persona.name if self.persona.name == "Day Trader" else "N/A"})**:
- Focus on intraday patterns
- Use tick/volume charts
- Quick scalps with tight stops
- Multiple entries/exits

## Technical Analysis Process:

1. **Market Structure**: Identify trend direction and strength
2. **Key Levels**: Map support/resistance zones
3. **Pattern Search**: Scan for actionable patterns
4. **Indicator Confluence**: Check for agreement
5. **Volume Confirmation**: Validate with volume
6. **Risk Definition**: Calculate stops and targets
7. **Setup Quality**: Rate A+ to C based on confluence

Remember to:
- Be specific with price levels
- Explain pattern psychology
- Highlight invalidation levels
- Consider market context
- Provide clear action plans
"""

        return base_prompt + technical_prompt

    def _build_graph(self):
        """Build enhanced graph with technical analysis nodes."""
        workflow = StateGraph(TechnicalAnalysisState)

        # Add specialized nodes with unique names
        workflow.add_node("analyze_structure", self._analyze_structure)
        workflow.add_node("detect_patterns", self._detect_patterns)
        workflow.add_node("analyze_indicators", self._analyze_indicators)
        workflow.add_node("construct_trade_setup", self._construct_trade_setup)
        workflow.add_node("agent", self._agent_node)

        # Create tool node if tools available
        if self.tools:
            from langgraph.prebuilt import ToolNode

            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)

        # Define flow
        workflow.add_edge(START, "analyze_structure")
        workflow.add_edge("analyze_structure", "detect_patterns")
        workflow.add_edge("detect_patterns", "analyze_indicators")
        workflow.add_edge("analyze_indicators", "construct_trade_setup")
        workflow.add_edge("construct_trade_setup", "agent")

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

    async def _analyze_structure(self, state: TechnicalAnalysisState) -> dict[str, Any]:
        """Analyze market structure and identify key levels."""
        try:
            # Get support/resistance tool
            sr_tool = next(
                (t for t in self.tools if "support_resistance" in t.name), None
            )

            if sr_tool and state.get("symbol"):
                circuit_breaker = await circuit_manager.get_or_create("technical")

                async def get_levels():
                    return await sr_tool.ainvoke(
                        {
                            "symbol": state["symbol"],
                            "lookback_days": state.get("lookback_days", 20),
                        }
                    )

                levels_data = await circuit_breaker.call(get_levels)

                # Extract support/resistance levels
                if isinstance(levels_data, dict):
                    state["support_levels"] = levels_data.get("support_levels", [])
                    state["resistance_levels"] = levels_data.get(
                        "resistance_levels", []
                    )

                    # Determine trend based on structure
                    if levels_data.get("trend"):
                        state["trend_direction"] = levels_data["trend"]
                    else:
                        # Simple trend determination
                        current = state.get("current_price", 0)
                        ma_50 = levels_data.get("ma_50", current)
                        state["trend_direction"] = (
                            "bullish" if current > ma_50 else "bearish"
                        )

        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")

        state["api_calls_made"] = state.get("api_calls_made", 0) + 1
        return {
            "support_levels": state.get("support_levels", []),
            "resistance_levels": state.get("resistance_levels", []),
            "trend_direction": state.get("trend_direction", "neutral"),
        }

    async def _detect_patterns(self, state: TechnicalAnalysisState) -> dict[str, Any]:
        """Detect chart patterns."""
        try:
            # Get pattern detection tool
            pattern_tool = next((t for t in self.tools if "pattern" in t.name), None)

            if pattern_tool and state.get("symbol"):
                circuit_breaker = await circuit_manager.get_or_create("technical")

                async def detect():
                    return await pattern_tool.ainvoke(
                        {
                            "symbol": state["symbol"],
                            "timeframe": state.get("timeframe", "1d"),
                        }
                    )

                pattern_data = await circuit_breaker.call(detect)

                if isinstance(pattern_data, dict) and "patterns" in pattern_data:
                    patterns = pattern_data["patterns"]
                    state["patterns"] = patterns

                    # Calculate pattern confidence scores
                    pattern_confidence = {}
                    for pattern in patterns:
                        name = pattern.get("name", "Unknown")
                        confidence = pattern.get("confidence", 50)
                        pattern_confidence[name] = confidence

                    state["pattern_confidence"] = pattern_confidence

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        state["api_calls_made"] = state.get("api_calls_made", 0) + 1
        return {
            "patterns": state.get("patterns", []),
            "pattern_confidence": state.get("pattern_confidence", {}),
        }

    async def _analyze_indicators(
        self, state: TechnicalAnalysisState
    ) -> dict[str, Any]:
        """Analyze technical indicators."""
        try:
            # Get indicators tool
            indicators_tool = next(
                (t for t in self.tools if "technical_indicators" in t.name), None
            )

            if indicators_tool and state.get("symbol"):
                circuit_breaker = await circuit_manager.get_or_create("technical")

                indicators = state.get("indicators", ["RSI", "MACD", "BB"])

                async def get_indicators():
                    return await indicators_tool.ainvoke(
                        {
                            "symbol": state["symbol"],
                            "indicators": indicators,
                            "period": state.get("lookback_days", 20),
                        }
                    )

                indicator_data = await circuit_breaker.call(get_indicators)

                if isinstance(indicator_data, dict):
                    # Store indicator values
                    state["indicator_values"] = indicator_data.get("values", {})

                    # Generate indicator signals
                    signals = self._generate_indicator_signals(indicator_data)
                    state["indicator_signals"] = signals

                    # Check for divergences
                    divergences = self._check_divergences(
                        state.get("price_history", {}), indicator_data
                    )
                    state["divergences"] = divergences

        except Exception as e:
            logger.error(f"Error analyzing indicators: {e}")

        state["api_calls_made"] = state.get("api_calls_made", 0) + 1
        return {
            "indicator_values": state.get("indicator_values", {}),
            "indicator_signals": state.get("indicator_signals", {}),
            "divergences": state.get("divergences", []),
        }

    async def _construct_trade_setup(
        self, state: TechnicalAnalysisState
    ) -> dict[str, Any]:
        """Construct complete trade setup."""
        try:
            current_price = state.get("current_price", 0)

            if current_price > 0:
                # Calculate entry points based on patterns and levels
                entry_points = self._calculate_entry_points(state)
                state["entry_points"] = entry_points

                # Get stop loss recommendation
                stops_tool = next(
                    (t for t in self.tools if isinstance(t, TechnicalStopsTool)), None
                )

                if stops_tool:
                    stops_data = await stops_tool.ainvoke(
                        {
                            "symbol": state["symbol"],
                            "lookback_days": 20,
                        }
                    )

                    if isinstance(stops_data, dict):
                        stop_loss = stops_data.get(
                            "recommended_stop", current_price * 0.95
                        )
                    else:
                        stop_loss = current_price * 0.95
                else:
                    stop_loss = current_price * 0.95

                state["stop_loss"] = stop_loss

                # Calculate profit targets
                risk = current_price - stop_loss
                targets = [
                    current_price + (risk * 1.5),  # 1.5R
                    current_price + (risk * 2.0),  # 2R
                    current_price + (risk * 3.0),  # 3R
                ]
                state["profit_targets"] = targets

                # Calculate risk/reward
                state["risk_reward_ratio"] = 2.0  # Default target

                # Rate setup quality
                quality = self._rate_setup_quality(state)
                state["setup_quality"] = quality

                # Calculate confidence score
                confidence = self._calculate_confidence_score(state)
                state["confidence_score"] = confidence

        except Exception as e:
            logger.error(f"Error constructing trade setup: {e}")

        return {
            "entry_points": state.get("entry_points", []),
            "stop_loss": state.get("stop_loss", 0),
            "profit_targets": state.get("profit_targets", []),
            "risk_reward_ratio": state.get("risk_reward_ratio", 0),
            "setup_quality": state.get("setup_quality", "C"),
            "confidence_score": state.get("confidence_score", 0),
        }

    def _generate_indicator_signals(self, indicator_data: dict) -> dict[str, str]:
        """Generate buy/sell/hold signals from indicators."""
        signals = {}

        # RSI signals
        rsi = indicator_data.get("RSI", {}).get("value", 50)
        if rsi < 30:
            signals["RSI"] = "buy"
        elif rsi > 70:
            signals["RSI"] = "sell"
        else:
            signals["RSI"] = "hold"

        # MACD signals
        macd = indicator_data.get("MACD", {})
        if macd.get("histogram", 0) > 0 and macd.get("signal_cross", "") == "bullish":
            signals["MACD"] = "buy"
        elif macd.get("histogram", 0) < 0 and macd.get("signal_cross", "") == "bearish":
            signals["MACD"] = "sell"
        else:
            signals["MACD"] = "hold"

        return signals

    def _check_divergences(
        self, price_history: dict, indicator_data: dict
    ) -> list[dict[str, Any]]:
        """Check for price/indicator divergences."""
        divergences: list[dict[str, Any]] = []

        # Simplified divergence detection
        # In production, would use more sophisticated analysis

        return divergences

    def _calculate_entry_points(self, state: TechnicalAnalysisState) -> list[float]:
        """Calculate optimal entry points."""
        current_price = state.get("current_price", 0)
        support_levels = state.get("support_levels", [])
        patterns = state.get("patterns", [])

        entries = []

        # Pattern-based entries
        for pattern in patterns:
            if pattern.get("entry_price"):
                entries.append(pattern["entry_price"])

        # Support-based entries
        for support in support_levels:
            if support < current_price:
                # Entry just above support
                entries.append(support * 1.01)

        # Current price entry if momentum
        if state.get("trend_direction") == "bullish":
            entries.append(current_price)

        return sorted(set(entries))[:3]  # Top 3 unique entries

    def _rate_setup_quality(self, state: TechnicalAnalysisState) -> str:
        """Rate the quality of the trade setup."""
        score = 0

        # Pattern quality
        if state.get("patterns"):
            max_confidence = max(p.get("confidence", 0) for p in state["patterns"])
            if max_confidence > 80:
                score += 30
            elif max_confidence > 60:
                score += 20
            else:
                score += 10

        # Indicator confluence
        signals = state.get("indicator_signals", {})
        buy_signals = sum(1 for s in signals.values() if s == "buy")
        if buy_signals >= 3:
            score += 30
        elif buy_signals >= 2:
            score += 20
        else:
            score += 10

        # Risk/Reward
        rr = state.get("risk_reward_ratio", 0)
        if rr >= 3:
            score += 20
        elif rr >= 2:
            score += 15
        else:
            score += 5

        # Volume confirmation (would check in real implementation)
        score += 10

        # Market alignment (would check in real implementation)
        score += 10

        # Convert score to grade
        if score >= 85:
            return "A+"
        elif score >= 75:
            return "A"
        elif score >= 65:
            return "B"
        else:
            return "C"

    def _calculate_confidence_score(self, state: TechnicalAnalysisState) -> float:
        """Calculate overall confidence score for the setup."""
        factors = []

        # Pattern confidence
        if state.get("pattern_confidence"):
            factors.append(max(state["pattern_confidence"].values()) / 100)

        # Indicator agreement
        signals = state.get("indicator_signals", {})
        if signals:
            buy_count = sum(1 for s in signals.values() if s == "buy")
            factors.append(buy_count / len(signals))

        # Setup quality
        quality_scores = {"A+": 1.0, "A": 0.85, "B": 0.70, "C": 0.50}
        factors.append(quality_scores.get(state.get("setup_quality", "C"), 0.5))

        # Average confidence
        return round(sum(factors) / len(factors) * 100, 1) if factors else 50.0

    async def analyze_stock(
        self,
        symbol: str,
        timeframe: str = "1d",
        indicators: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform comprehensive technical analysis on a stock.

        Args:
            symbol: Stock symbol
            timeframe: Chart timeframe
            indicators: List of indicators to analyze
            **kwargs: Additional parameters

        Returns:
            Complete technical analysis with trade setup
        """
        start_time = datetime.now()

        # Default indicators
        if indicators is None:
            indicators = ["RSI", "MACD", "BB", "EMA", "VWAP"]

        # Prepare query
        query = f"Analyze {symbol} on {timeframe} timeframe with focus on patterns and trade setup"

        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators,
            "lookback_days": kwargs.get("lookback_days", 20),
            "pattern_detection": True,
            "multi_timeframe": kwargs.get("multi_timeframe", False),
            "persona": self.persona.name,
            "session_id": kwargs.get(
                "session_id", f"{symbol}_{datetime.now().timestamp()}"
            ),
            "timestamp": datetime.now(),
            "api_calls_made": 0,
        }

        # Run analysis
        result = await self.ainvoke(
            query, initial_state["session_id"], initial_state=initial_state
        )

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Extract results
        return self._format_analysis_results(result, execution_time)

    def _format_analysis_results(
        self, result: dict[str, Any], execution_time: float
    ) -> dict[str, Any]:
        """Format technical analysis results."""
        state = result.get("state", {})
        messages = result.get("messages", [])

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": execution_time,
            "symbol": state.get("symbol", ""),
            "analysis": {
                "market_structure": {
                    "trend": state.get("trend_direction", "neutral"),
                    "support_levels": state.get("support_levels", []),
                    "resistance_levels": state.get("resistance_levels", []),
                },
                "patterns": {
                    "detected": state.get("patterns", []),
                    "confidence": state.get("pattern_confidence", {}),
                },
                "indicators": {
                    "values": state.get("indicator_values", {}),
                    "signals": state.get("indicator_signals", {}),
                    "divergences": state.get("divergences", []),
                },
                "trade_setup": {
                    "entries": state.get("entry_points", []),
                    "stop_loss": state.get("stop_loss", 0),
                    "targets": state.get("profit_targets", []),
                    "risk_reward": state.get("risk_reward_ratio", 0),
                    "quality": state.get("setup_quality", "C"),
                    "confidence": state.get("confidence_score", 0),
                },
            },
            "recommendation": messages[-1].content if messages else "",
            "persona_adjusted": True,
            "risk_profile": self.persona.name,
        }

    def _create_mock_tools(self) -> list:
        """Create mock tools for testing."""
        from langchain_core.tools import tool

        @tool
        def mock_technical_indicators(symbol: str, indicators: list[str]) -> dict:
            """Mock technical indicators tool."""
            return {
                "RSI": {"value": 45, "trend": "neutral"},
                "MACD": {"histogram": 0.5, "signal_cross": "bullish"},
                "BB": {"upper": 150, "middle": 145, "lower": 140},
            }

        @tool
        def mock_support_resistance(symbol: str) -> dict:
            """Mock support/resistance tool."""
            return {
                "support_levels": [140, 135, 130],
                "resistance_levels": [150, 155, 160],
                "trend": "bullish",
            }

        return [mock_technical_indicators, mock_support_resistance]
