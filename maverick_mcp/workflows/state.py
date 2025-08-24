"""
State definitions for LangGraph workflows using TypedDict pattern.
"""

from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class BaseAgentState(TypedDict):
    """Base state for all agents with comprehensive tracking."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    persona: str
    timestamp: datetime
    token_count: int
    error: str | None

    # Enhanced tracking
    analyzed_stocks: dict[str, dict[str, Any]]  # symbol -> analysis data
    key_price_levels: dict[str, dict[str, float]]  # symbol -> support/resistance
    last_analysis_time: dict[str, datetime]  # symbol -> timestamp
    conversation_context: dict[str, Any]  # Additional context

    # Performance tracking
    execution_time_ms: float | None
    api_calls_made: int
    cache_hits: int
    cache_misses: int


class MarketAnalysisState(BaseAgentState):
    """State for market analysis workflows."""

    # Screening parameters
    screening_strategy: str  # maverick, trending, momentum, mean_reversion
    sector_filter: str | None
    min_volume: float | None
    min_price: float | None
    max_results: int

    # Enhanced filters
    min_market_cap: float | None
    max_pe_ratio: float | None
    min_momentum_score: int | None
    volatility_filter: float | None

    # Results
    screened_symbols: list[str]
    screening_scores: dict[str, float]
    sector_performance: dict[str, float]
    market_breadth: dict[str, Any]

    # Enhanced results
    symbol_metadata: dict[str, dict[str, Any]]  # symbol -> metadata
    sector_rotation: dict[str, Any]  # sector rotation analysis
    market_regime: str  # bull, bear, sideways
    sentiment_indicators: dict[str, float]

    # Analysis cache
    analyzed_sectors: set[str]
    last_screen_time: datetime | None
    cache_expiry: datetime | None


class TechnicalAnalysisState(BaseAgentState):
    """State for technical analysis workflows with enhanced tracking."""

    # Analysis parameters
    symbol: str
    timeframe: str  # 1d, 1h, 5m, 15m, 30m
    lookback_days: int
    indicators: list[str]

    # Enhanced parameters
    pattern_detection: bool
    fibonacci_levels: bool
    volume_analysis: bool
    multi_timeframe: bool

    # Price data
    price_history: dict[str, Any]
    current_price: float
    volume: float

    # Enhanced price data
    vwap: float
    average_volume: float
    relative_volume: float
    spread_percentage: float

    # Technical results
    support_levels: list[float]
    resistance_levels: list[float]
    patterns: list[dict[str, Any]]
    indicator_values: dict[str, float]
    trend_direction: str  # bullish, bearish, neutral

    # Enhanced technical results
    pattern_confidence: dict[str, float]  # pattern -> confidence score
    indicator_signals: dict[str, str]  # indicator -> signal (buy/sell/hold)
    divergences: list[dict[str, Any]]  # price/indicator divergences
    market_structure: dict[str, Any]  # higher highs, lower lows, etc.

    # Trade setup
    entry_points: list[float]
    stop_loss: float
    profit_targets: list[float]
    risk_reward_ratio: float

    # Enhanced trade setup
    position_size_shares: int
    position_size_value: float
    expected_holding_period: int  # days
    confidence_score: float  # 0-100
    setup_quality: str  # A+, A, B, C


class RiskManagementState(BaseAgentState):
    """State for risk management workflows with comprehensive tracking."""

    # Account parameters
    account_size: float
    risk_per_trade: float  # percentage
    max_portfolio_heat: float  # percentage

    # Enhanced account parameters
    buying_power: float
    margin_used: float
    cash_available: float
    portfolio_leverage: float

    # Position parameters
    symbol: str
    entry_price: float
    stop_loss_price: float

    # Enhanced position parameters
    position_type: str  # long, short
    time_stop_days: int | None
    trailing_stop_percent: float | None
    scale_in_levels: list[float]
    scale_out_levels: list[float]

    # Calculations
    position_size: int
    position_value: float
    risk_amount: float
    portfolio_heat: float

    # Enhanced calculations
    kelly_fraction: float
    optimal_f: float
    risk_units: float  # position risk in "R" units
    expected_value: float
    risk_adjusted_return: float

    # Portfolio context
    open_positions: list[dict[str, Any]]
    total_exposure: float
    correlation_matrix: dict[str, dict[str, float]]

    # Enhanced portfolio context
    sector_exposure: dict[str, float]
    asset_class_exposure: dict[str, float]
    geographic_exposure: dict[str, float]
    factor_exposure: dict[str, float]  # value, growth, momentum, etc.

    # Risk metrics
    sharpe_ratio: float | None
    max_drawdown: float | None
    win_rate: float | None

    # Enhanced risk metrics
    sortino_ratio: float | None
    calmar_ratio: float | None
    var_95: float | None  # Value at Risk
    cvar_95: float | None  # Conditional VaR
    beta_to_market: float | None
    correlation_to_market: float | None


class PortfolioState(BaseAgentState):
    """State for portfolio optimization workflows."""

    # Portfolio composition
    holdings: list[dict[str, Any]]  # symbol, shares, cost_basis, current_value
    cash_balance: float
    total_value: float

    # Performance metrics
    returns: dict[str, float]  # period -> return percentage
    benchmark_comparison: dict[str, float]
    attribution: dict[str, float]  # contribution by position

    # Optimization parameters
    target_allocation: dict[str, float]
    rebalance_threshold: float
    tax_aware: bool

    # Recommendations
    rebalance_trades: list[dict[str, Any]]
    new_positions: list[dict[str, Any]]
    exit_positions: list[str]

    # Risk analysis
    portfolio_beta: float
    diversification_score: float
    concentration_risk: dict[str, float]


class SupervisorState(BaseAgentState):
    """State for supervisor agent coordinating multiple agents."""

    # Query routing
    query_type: str  # screening, analysis, risk, portfolio
    subtasks: list[dict[str, Any]]
    current_subtask: int

    # Agent coordination
    active_agents: list[str]
    agent_results: dict[str, Any]

    # Workflow control
    workflow_plan: list[str]
    completed_steps: list[str]
    pending_steps: list[str]

    # Aggregated results
    final_recommendations: list[dict[str, Any]]
    confidence_scores: dict[str, float]
    risk_warnings: list[str]
