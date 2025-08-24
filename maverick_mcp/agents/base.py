"""
Base classes for persona-aware agents using LangGraph best practices.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class InvestorPersona(BaseModel):
    """Defines an investor persona with risk parameters."""

    name: str
    risk_tolerance: tuple[int, int] = Field(
        description="Risk tolerance range (min, max) on 0-100 scale"
    )
    position_size_max: float = Field(
        description="Maximum position size as percentage of portfolio"
    )
    stop_loss_multiplier: float = Field(
        description="Multiplier for stop loss calculation"
    )
    preferred_timeframe: str = Field(
        default="swing", description="Preferred trading timeframe: day, swing, position"
    )
    characteristics: list[str] = Field(
        default_factory=list, description="Key behavioral characteristics"
    )


# Predefined investor personas
INVESTOR_PERSONAS = {
    "conservative": InvestorPersona(
        name="Conservative",
        risk_tolerance=(
            settings.financial.risk_tolerance_conservative_min,
            settings.financial.risk_tolerance_conservative_max,
        ),
        position_size_max=settings.financial.max_position_size_conservative,
        stop_loss_multiplier=settings.financial.stop_loss_multiplier_conservative,
        preferred_timeframe="position",
        characteristics=[
            "Prioritizes capital preservation",
            "Focuses on dividend stocks",
            "Prefers established companies",
            "Long-term oriented",
        ],
    ),
    "moderate": InvestorPersona(
        name="Moderate",
        risk_tolerance=(
            settings.financial.risk_tolerance_moderate_min,
            settings.financial.risk_tolerance_moderate_max,
        ),
        position_size_max=settings.financial.max_position_size_moderate,
        stop_loss_multiplier=settings.financial.stop_loss_multiplier_moderate,
        preferred_timeframe="swing",
        characteristics=[
            "Balanced risk/reward approach",
            "Mix of growth and value",
            "Diversified portfolio",
            "Medium-term focus",
        ],
    ),
    "aggressive": InvestorPersona(
        name="Aggressive",
        risk_tolerance=(
            settings.financial.risk_tolerance_aggressive_min,
            settings.financial.risk_tolerance_aggressive_max,
        ),
        position_size_max=settings.financial.max_position_size_aggressive,
        stop_loss_multiplier=settings.financial.stop_loss_multiplier_aggressive,
        preferred_timeframe="day",
        characteristics=[
            "High risk tolerance",
            "Growth-focused",
            "Momentum trading",
            "Short-term opportunities",
        ],
    ),
    "day_trader": InvestorPersona(
        name="Day Trader",
        risk_tolerance=(
            settings.financial.risk_tolerance_day_trader_min,
            settings.financial.risk_tolerance_day_trader_max,
        ),
        position_size_max=settings.financial.max_position_size_day_trader,
        stop_loss_multiplier=settings.financial.stop_loss_multiplier_day_trader,
        preferred_timeframe="day",
        characteristics=[
            "Intraday positions only",
            "High-frequency trading",
            "Technical analysis focused",
            "Tight risk controls",
        ],
    ),
}


class BaseAgentState(TypedDict):
    """Base state for all persona-aware agents."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    persona: str
    session_id: str


class PersonaAwareTool(BaseTool):
    """Base class for tools that adapt to investor personas."""

    persona: InvestorPersona | None = None
    # State tracking
    last_analysis_time: dict[str, datetime] = {}
    analyzed_stocks: dict[str, dict] = {}
    key_price_levels: dict[str, dict] = {}
    # Cache settings
    cache_ttl: int = settings.agent.agent_cache_ttl_seconds

    def set_persona(self, persona: InvestorPersona) -> None:
        """Set the active investor persona."""
        self.persona = persona

    def adjust_for_risk(self, value: float, parameter_type: str) -> float:
        """Adjust a value based on the persona's risk profile."""
        if not self.persona:
            return value

        # Get average risk tolerance
        risk_avg = sum(self.persona.risk_tolerance) / 2
        risk_factor = risk_avg / 50  # Normalize to 1.0 at moderate risk

        # Adjust based on parameter type
        if parameter_type == "position_size":
            # Kelly Criterion-inspired sizing
            kelly_fraction = self._calculate_kelly_fraction(risk_factor)
            adjusted = value * kelly_fraction
            return min(adjusted, self.persona.position_size_max)
        elif parameter_type == "stop_loss":
            # ATR-based dynamic stops
            return value * self.persona.stop_loss_multiplier
        elif parameter_type == "profit_target":
            # Risk-adjusted targets
            return value * (2 - risk_factor)  # Conservative = lower targets
        elif parameter_type == "volatility_filter":
            # Volatility tolerance
            return value * (2 - risk_factor)  # Conservative = lower vol tolerance
        elif parameter_type == "time_horizon":
            # Holding period in days
            if self.persona.preferred_timeframe == "day":
                return 1
            elif self.persona.preferred_timeframe == "swing":
                return 5 * risk_factor  # 2.5-7.5 days
            else:  # position
                return 20 * risk_factor  # 10-30 days
        else:
            return value

    def _calculate_kelly_fraction(self, risk_factor: float) -> float:
        """Calculate position size using Kelly Criterion."""
        # Simplified Kelly: f = (p*b - q) / b
        # where p = win probability, b = win/loss ratio, q = loss probability
        # Using risk factor to adjust expected win rate
        win_probability = 0.45 + (0.1 * risk_factor)  # 45-55% base win rate
        win_loss_ratio = 2.0  # 2:1 reward/risk
        loss_probability = 1 - win_probability

        kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio

        # Apply safety factor (never use full Kelly)
        safety_factor = 0.25  # Use 25% of Kelly
        return max(0, kelly * safety_factor)

    def update_analysis_data(self, symbol: str, analysis_data: dict[str, Any]):
        """Update stored analysis data for a symbol."""
        symbol = symbol.upper()
        self.analyzed_stocks[symbol] = analysis_data
        self.last_analysis_time[symbol] = datetime.now()
        if "price_levels" in analysis_data:
            self.key_price_levels[symbol] = analysis_data["price_levels"]

    def get_stock_context(self, symbol: str) -> dict[str, Any]:
        """Get stored context for a symbol."""
        symbol = symbol.upper()
        return {
            "analysis": self.analyzed_stocks.get(symbol, {}),
            "last_analysis": self.last_analysis_time.get(symbol),
            "price_levels": self.key_price_levels.get(symbol, {}),
            "cache_expired": self._is_cache_expired(symbol),
        }

    def _is_cache_expired(self, symbol: str) -> bool:
        """Check if cached data has expired."""
        last_time = self.last_analysis_time.get(symbol.upper())
        if not last_time:
            return True

        age_seconds = (datetime.now() - last_time).total_seconds()
        return age_seconds > self.cache_ttl

    def _adjust_risk_parameters(self, params: dict) -> dict:
        """Adjust parameters based on risk profile."""
        if not self.persona:
            return params

        risk_factor = sum(self.persona.risk_tolerance) / 100  # 0.1-0.9 scale

        # Apply risk adjustments based on parameter names
        adjusted = {}
        for key, value in params.items():
            if isinstance(value, int | float):
                key_lower = key.lower()
                if any(term in key_lower for term in ["stop", "support", "risk"]):
                    # Wider stops/support for conservative, tighter for aggressive
                    adjusted[key] = value * (2 - risk_factor)
                elif any(
                    term in key_lower for term in ["resistance", "target", "profit"]
                ):
                    # Lower targets for conservative, higher for aggressive
                    adjusted[key] = value * risk_factor
                elif any(term in key_lower for term in ["size", "amount", "shares"]):
                    # Smaller positions for conservative, larger for aggressive
                    adjusted[key] = self.adjust_for_risk(value, "position_size")
                elif any(term in key_lower for term in ["volume", "liquidity"]):
                    # Higher liquidity requirements for conservative
                    adjusted[key] = value * (2 - risk_factor)
                elif any(term in key_lower for term in ["volatility", "atr", "std"]):
                    # Lower volatility tolerance for conservative
                    adjusted[key] = self.adjust_for_risk(value, "volatility_filter")
                else:
                    adjusted[key] = value
            else:
                adjusted[key] = value

        return adjusted

    def _validate_risk_levels(self, data: dict) -> bool:
        """Validate if the data meets the persona's risk criteria."""
        if not self.persona:
            return True

        min_risk, max_risk = self.persona.risk_tolerance

        # Extract risk metrics
        volatility = data.get("volatility", 0)
        beta = data.get("beta", 1.0)

        # Convert to risk score (0-100)
        volatility_score = min(100, volatility * 2)  # Assume 50% vol = 100 risk
        beta_score = abs(beta - 1) * 100  # Distance from market

        # Combined risk score
        risk_score = (volatility_score + beta_score) / 2

        if risk_score < min_risk or risk_score > max_risk:
            return False

        # Persona-specific validations
        if self.persona.name == "Conservative":
            # Additional checks for conservative investors
            if data.get("debt_to_equity", 0) > 1.5:
                return False
            if data.get("current_ratio", 0) < 1.5:
                return False
            if data.get("dividend_yield", 0) < 0.02:  # Prefer dividend stocks
                return False
        elif self.persona.name == "Day Trader":
            # Day traders need high liquidity
            if data.get("average_volume", 0) < 1_000_000:
                return False
            if data.get("spread_percentage", 0) > 0.1:  # Tight spreads only
                return False

        return True

    def format_for_persona(self, data: dict) -> dict:
        """Format output data based on persona preferences."""
        if not self.persona:
            return data

        formatted = data.copy()

        # Add persona-specific insights
        formatted["persona_insights"] = {
            "suitable_for_profile": self._validate_risk_levels(data),
            "risk_adjusted_parameters": self._adjust_risk_parameters(
                data.get("parameters", {})
            ),
            "recommended_timeframe": self.persona.preferred_timeframe,
            "max_position_size": self.persona.position_size_max,
        }

        # Add risk warnings if needed
        warnings = []
        if not self._validate_risk_levels(data):
            warnings.append(f"Risk profile outside {self.persona.name} parameters")

        if data.get("volatility", 0) > 50:
            warnings.append("High volatility - consider smaller position size")

        if warnings:
            formatted["risk_warnings"] = warnings

        return formatted


class PersonaAwareAgent(ABC):
    """
    Base class for agents that adapt behavior based on investor personas.

    This follows LangGraph best practices:
    - Uses StateGraph for workflow definition
    - Implements proper node/edge patterns
    - Supports native streaming modes
    - Uses TypedDict for state management
    """

    def __init__(
        self,
        llm,
        tools: list[BaseTool],
        persona: str = "moderate",
        checkpointer: MemorySaver | None = None,
        ttl_hours: int = 1,
    ):
        """
        Initialize a persona-aware agent.

        Args:
            llm: Language model to use
            tools: List of tools available to the agent
            persona: Investor persona name
            checkpointer: Optional checkpointer (defaults to MemorySaver)
            ttl_hours: Time-to-live for memory in hours
        """
        self.llm = llm
        self.tools = tools
        self.persona = INVESTOR_PERSONAS.get(persona, INVESTOR_PERSONAS["moderate"])
        self.ttl_hours = ttl_hours

        # Set up checkpointing
        if checkpointer is None:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = checkpointer

        # Configure tools with persona
        for tool in self.tools:
            if isinstance(tool, PersonaAwareTool):
                tool.set_persona(self.persona)

        # Build the graph
        self.graph = self._build_graph()

        # Track usage
        self.total_tokens = 0
        self.conversation_start = datetime.now()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create the graph builder
        workflow = StateGraph(self.get_state_schema())

        # Add the agent node
        workflow.add_node("agent", self._agent_node)

        # Create tool node if tools are available
        if self.tools:
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)

            # Add conditional edge from agent
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    # If agent returns tool calls, route to tools
                    "continue": "tools",
                    # Otherwise end
                    "end": END,
                },
            )

            # Add edge from tools back to agent
            workflow.add_edge("tools", "agent")
        else:
            # No tools, just end after agent
            workflow.add_edge("agent", END)

        # Set entry point
        workflow.add_edge(START, "agent")

        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    def _agent_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """The main agent node that processes messages."""
        messages = state["messages"]

        # Add system message if it's the first message
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            system_prompt = self._build_system_prompt()
            messages = [SystemMessage(content=system_prompt)] + messages

        # Call the LLM
        if self.tools:
            response = self.llm.bind_tools(self.tools).invoke(messages)
        else:
            response = self.llm.invoke(messages)

        # Track tokens (simplified)
        if hasattr(response, "content"):
            self.total_tokens += len(response.content) // 4

        # Return the response
        return {"messages": [response]}

    def _should_continue(self, state: dict[str, Any]) -> str:
        """Determine whether to continue to tools or end."""
        last_message = state["messages"][-1]

        # If the LLM makes a tool call, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        # Otherwise we're done
        return "end"

    def _build_system_prompt(self) -> str:
        """Build system prompt based on persona."""
        base_prompt = f"""You are a financial advisor configured for a {self.persona.name} investor profile.

Risk Parameters:
- Risk Tolerance: {self.persona.risk_tolerance[0]}-{self.persona.risk_tolerance[1]}/100
- Max Position Size: {self.persona.position_size_max * 100:.1f}% of portfolio
- Stop Loss Multiplier: {self.persona.stop_loss_multiplier}x
- Preferred Timeframe: {self.persona.preferred_timeframe}

Key Characteristics:
{chr(10).join(f"- {char}" for char in self.persona.characteristics)}

Always adjust your recommendations to match this risk profile. Be explicit about risk management."""

        return base_prompt

    @abstractmethod
    def get_state_schema(self) -> type:
        """
        Get the state schema for this agent.

        Subclasses should return their specific state schema.
        """
        return BaseAgentState

    async def ainvoke(self, query: str, session_id: str, **kwargs) -> dict[str, Any]:
        """
        Invoke the agent asynchronously.

        Args:
            query: User query
            session_id: Session identifier
            **kwargs: Additional parameters

        Returns:
            Agent response
        """
        config = {
            "configurable": {"thread_id": session_id, "persona": self.persona.name}
        }

        # Merge additional config
        if "config" in kwargs:
            config.update(kwargs["config"])

        # Run the graph
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=query)],
                "persona": self.persona.name,
                "session_id": session_id,
            },
            config=config,
        )

        return self._extract_response(result)

    def invoke(self, query: str, session_id: str, **kwargs) -> dict[str, Any]:
        """
        Invoke the agent synchronously.

        Args:
            query: User query
            session_id: Session identifier
            **kwargs: Additional parameters

        Returns:
            Agent response
        """
        config = {
            "configurable": {"thread_id": session_id, "persona": self.persona.name}
        }

        # Merge additional config
        if "config" in kwargs:
            config.update(kwargs["config"])

        # Run the graph
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "persona": self.persona.name,
                "session_id": session_id,
            },
            config=config,
        )

        return self._extract_response(result)

    async def astream(
        self, query: str, session_id: str, stream_mode: str = "values", **kwargs
    ):
        """
        Stream agent responses asynchronously.

        Args:
            query: User query
            session_id: Session identifier
            stream_mode: Streaming mode (values, updates, messages, custom, debug)
            **kwargs: Additional parameters

        Yields:
            Streamed chunks based on mode
        """
        config = {
            "configurable": {"thread_id": session_id, "persona": self.persona.name}
        }

        # Merge additional config
        if "config" in kwargs:
            config.update(kwargs["config"])

        # Stream the graph
        async for chunk in self.graph.astream(
            {
                "messages": [HumanMessage(content=query)],
                "persona": self.persona.name,
                "session_id": session_id,
            },
            config=config,
            stream_mode=stream_mode,
        ):
            yield chunk

    def stream(
        self, query: str, session_id: str, stream_mode: str = "values", **kwargs
    ):
        """
        Stream agent responses synchronously.

        Args:
            query: User query
            session_id: Session identifier
            stream_mode: Streaming mode (values, updates, messages, custom, debug)
            **kwargs: Additional parameters

        Yields:
            Streamed chunks based on mode
        """
        config = {
            "configurable": {"thread_id": session_id, "persona": self.persona.name}
        }

        # Merge additional config
        if "config" in kwargs:
            config.update(kwargs["config"])

        # Stream the graph
        yield from self.graph.stream(
            {
                "messages": [HumanMessage(content=query)],
                "persona": self.persona.name,
                "session_id": session_id,
            },
            config=config,
            stream_mode=stream_mode,
        )

    def _extract_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract the final response from graph execution."""
        messages = result.get("messages", [])

        if not messages:
            return {"content": "No response generated", "status": "error"}

        # Get the last AI message
        last_message = messages[-1]

        return {
            "content": last_message.content
            if hasattr(last_message, "content")
            else str(last_message),
            "status": "success",
            "persona": self.persona.name,
            "message_count": len(messages),
            "session_id": result.get("session_id", ""),
        }

    def get_risk_adjusted_params(
        self, base_params: dict[str, float]
    ) -> dict[str, float]:
        """Adjust parameters based on persona risk profile."""
        adjusted = {}

        for key, value in base_params.items():
            if "size" in key.lower() or "position" in key.lower():
                adjusted[key] = self.adjust_for_risk(value, "position_size")
            elif "stop" in key.lower():
                adjusted[key] = self.adjust_for_risk(value, "stop_loss")
            elif "target" in key.lower() or "profit" in key.lower():
                adjusted[key] = self.adjust_for_risk(value, "profit_target")
            else:
                adjusted[key] = value

        return adjusted

    def adjust_for_risk(self, value: float, parameter_type: str) -> float:
        """Adjust a value based on the persona's risk profile."""
        # Get average risk tolerance
        risk_avg = sum(self.persona.risk_tolerance) / 2
        risk_factor = risk_avg / 50  # Normalize to 1.0 at moderate risk

        # Adjust based on parameter type
        if parameter_type == "position_size":
            return min(value * risk_factor, self.persona.position_size_max)
        elif parameter_type == "stop_loss":
            return value * self.persona.stop_loss_multiplier
        elif parameter_type == "profit_target":
            return value * (2 - risk_factor)  # Conservative = lower targets
        else:
            return value
