"""
State definitions for LangGraph workflows using TypedDict pattern.
"""

from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


def take_latest_status(current: str, new: str) -> str:
    """Reducer function that takes the latest status update."""
    return new if new else current


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
    """Enhanced state for supervisor agent coordinating multiple agents."""

    # Query routing and classification
    query_classification: dict[str, Any]  # Query type, complexity, required agents
    execution_plan: list[dict[str, Any]]  # Subtasks with dependencies and timing
    current_subtask_index: int  # Current execution position
    routing_strategy: str  # "llm_powered", "rule_based", "hybrid"

    # Agent coordination
    active_agents: list[str]  # Currently active agent names
    agent_results: dict[str, dict[str, Any]]  # Results from each agent
    agent_confidence: dict[str, float]  # Confidence scores per agent
    agent_execution_times: dict[str, float]  # Execution times per agent
    agent_errors: dict[str, str | None]  # Errors from agents

    # Workflow control
    workflow_status: (
        str  # "planning", "executing", "aggregating", "synthesizing", "completed"
    )
    parallel_execution: bool  # Whether to run agents in parallel
    dependency_graph: dict[str, list[str]]  # Task dependencies
    max_iterations: int  # Maximum iterations to prevent loops
    current_iteration: int  # Current iteration count

    # Result synthesis and conflict resolution
    conflicts_detected: list[dict[str, Any]]  # Conflicts between agent results
    conflict_resolution: dict[str, Any]  # How conflicts were resolved
    synthesis_weights: dict[str, float]  # Weights applied to agent results
    final_recommendation_confidence: float  # Overall confidence in final result
    synthesis_mode: str  # "weighted", "consensus", "priority"

    # Performance and monitoring
    total_execution_time_ms: float  # Total workflow execution time
    agent_coordination_overhead_ms: float  # Time spent coordinating agents
    synthesis_time_ms: float  # Time spent synthesizing results
    cache_utilization: dict[str, int]  # Cache usage per agent

    # Legacy fields for backward compatibility
    query_type: str | None  # Legacy field - use query_classification instead
    subtasks: list[dict[str, Any]] | None  # Legacy field - use execution_plan instead
    current_subtask: int | None  # Legacy field - use current_subtask_index instead
    workflow_plan: list[str] | None  # Legacy field
    completed_steps: list[str] | None  # Legacy field
    pending_steps: list[str] | None  # Legacy field
    final_recommendations: list[dict[str, Any]] | None  # Legacy field
    confidence_scores: (
        dict[str, float] | None
    )  # Legacy field - use agent_confidence instead
    risk_warnings: list[str] | None  # Legacy field


class DeepResearchState(BaseAgentState):
    """State for deep research workflows with web search and content analysis."""

    # Research parameters
    research_topic: str  # Main research topic or symbol
    research_depth: str  # basic, standard, comprehensive, exhaustive
    focus_areas: list[str]  # Specific focus areas for research
    timeframe: str  # Time range for research (7d, 30d, 90d, 1y)

    # Search and query management
    search_queries: list[str]  # Generated search queries
    search_results: list[dict[str, Any]]  # Raw search results from providers
    search_providers_used: list[str]  # Which providers were used
    search_metadata: dict[str, Any]  # Search execution metadata

    # Content analysis
    analyzed_content: list[dict[str, Any]]  # Content with AI analysis
    content_summaries: list[str]  # Summaries of analyzed content
    key_themes: list[str]  # Extracted themes from content
    content_quality_scores: dict[str, float]  # Quality scores for content

    # Source management and validation
    validated_sources: list[dict[str, Any]]  # Sources that passed validation
    rejected_sources: list[dict[str, Any]]  # Sources that failed validation
    source_credibility_scores: dict[str, float]  # Credibility score per source URL
    source_diversity_score: float  # Diversity metric for sources
    duplicate_sources_removed: int  # Count of duplicates removed

    # Research findings and analysis
    research_findings: list[dict[str, Any]]  # Core research findings
    sentiment_analysis: dict[str, Any]  # Overall sentiment analysis
    risk_assessment: dict[str, Any]  # Risk factors and assessment
    opportunity_analysis: dict[str, Any]  # Investment opportunities identified
    competitive_landscape: dict[str, Any]  # Competitive analysis if applicable

    # Citations and references
    citations: list[dict[str, Any]]  # Properly formatted citations
    reference_urls: list[str]  # All referenced URLs
    source_attribution: dict[str, str]  # Finding -> source mapping

    # Research workflow status
    research_status: Annotated[
        str, take_latest_status
    ]  # planning, searching, analyzing, validating, synthesizing, completed
    research_confidence: float  # Overall confidence in research (0-1)
    validation_checks_passed: int  # Number of validation checks passed
    fact_validation_results: list[dict[str, Any]]  # Results from fact-checking

    # Performance and metrics
    search_execution_time_ms: float  # Time spent on searches
    analysis_execution_time_ms: float  # Time spent on content analysis
    validation_execution_time_ms: float  # Time spent on validation
    synthesis_execution_time_ms: float  # Time spent on synthesis
    total_sources_processed: int  # Total number of sources processed
    api_rate_limits_hit: int  # Number of rate limit encounters

    # Research quality indicators
    source_age_distribution: dict[str, int]  # Age distribution of sources
    geographic_coverage: list[str]  # Geographic regions covered
    publication_types: dict[str, int]  # Types of publications analyzed
    author_expertise_scores: dict[str, float]  # Author expertise assessments

    # Specialized research areas
    fundamental_analysis_data: dict[str, Any]  # Fundamental analysis results
    technical_context: dict[str, Any]  # Technical analysis context if relevant
    macro_economic_factors: list[str]  # Macro factors identified
    regulatory_considerations: list[str]  # Regulatory issues identified

    # Research iteration and refinement
    research_iterations: int  # Number of research iterations performed
    query_refinements: list[dict[str, Any]]  # Query refinement history
    research_gaps_identified: list[str]  # Areas needing more research
    follow_up_research_suggestions: list[str]  # Suggestions for additional research

    # Parallel execution tracking
    parallel_tasks: dict[str, dict[str, Any]]  # task_id -> task info
    parallel_results: dict[str, dict[str, Any]]  # task_id -> results
    parallel_execution_enabled: bool  # Whether parallel execution is enabled
    concurrent_agents_count: int  # Number of agents running concurrently
    parallel_efficiency_score: float  # Parallel vs sequential execution efficiency
    task_distribution_strategy: str  # How tasks were distributed

    # Subagent specialization results
    fundamental_research_results: dict[
        str, Any
    ]  # Results from fundamental analysis agent
    technical_research_results: dict[str, Any]  # Results from technical analysis agent
    sentiment_research_results: dict[str, Any]  # Results from sentiment analysis agent
    competitive_research_results: dict[
        str, Any
    ]  # Results from competitive analysis agent

    # Cross-agent synthesis
    consensus_findings: list[dict[str, Any]]  # Findings agreed upon by multiple agents
    conflicting_findings: list[dict[str, Any]]  # Findings where agents disagree
    confidence_weighted_analysis: dict[
        str, Any
    ]  # Analysis weighted by agent confidence
    multi_agent_synthesis_quality: float  # Quality score for multi-agent synthesis


class BacktestingWorkflowState(BaseAgentState):
    """State for intelligent backtesting workflows with market regime analysis."""

    # Input parameters
    symbol: str  # Stock symbol to backtest
    start_date: str  # Start date for analysis (YYYY-MM-DD)
    end_date: str  # End date for analysis (YYYY-MM-DD)
    initial_capital: float  # Starting capital for backtest
    requested_strategy: str | None  # User-requested strategy (optional)

    # Market regime analysis
    market_regime: str  # bull, bear, sideways, volatile, low_volume
    regime_confidence: float  # Confidence in regime detection (0-1)
    regime_indicators: dict[str, float]  # Supporting indicators for regime
    regime_analysis_time_ms: float  # Time spent on regime analysis
    volatility_percentile: float  # Current volatility vs historical
    trend_strength: float  # Strength of current trend (-1 to 1)

    # Market conditions context
    market_conditions: dict[str, Any]  # Overall market environment
    sector_performance: dict[str, float]  # Sector relative performance
    correlation_to_market: float  # Stock correlation to broad market
    volume_profile: dict[str, float]  # Volume characteristics
    support_resistance_levels: list[float]  # Key price levels

    # Strategy selection process
    candidate_strategies: list[dict[str, Any]]  # List of potential strategies
    strategy_rankings: dict[str, float]  # Strategy -> fitness score
    selected_strategies: list[str]  # Final selected strategies for testing
    strategy_selection_reasoning: str  # Why these strategies were chosen
    strategy_selection_confidence: float  # Confidence in selection (0-1)

    # Parameter optimization
    optimization_config: dict[str, Any]  # Optimization configuration
    parameter_grids: dict[str, dict[str, list]]  # Strategy -> parameter grid
    optimization_results: dict[str, dict[str, Any]]  # Strategy -> optimization results
    best_parameters: dict[str, dict[str, Any]]  # Strategy -> best parameters
    optimization_time_ms: float  # Time spent on optimization
    optimization_iterations: int  # Number of parameter combinations tested

    # Validation and robustness
    walk_forward_results: dict[str, dict[str, Any]]  # Strategy -> WF results
    monte_carlo_results: dict[str, dict[str, Any]]  # Strategy -> MC results
    out_of_sample_performance: dict[str, dict[str, float]]  # OOS metrics
    robustness_score: dict[str, float]  # Strategy -> robustness score (0-1)
    validation_warnings: list[str]  # Validation warnings and concerns

    # Final recommendations
    final_strategy_ranking: list[dict[str, Any]]  # Ranked strategy recommendations
    recommended_strategy: str  # Top recommended strategy
    recommended_parameters: dict[str, Any]  # Recommended parameter set
    recommendation_confidence: float  # Overall confidence (0-1)
    risk_assessment: dict[str, Any]  # Risk analysis of recommendation

    # Performance metrics aggregation
    comparative_metrics: dict[str, dict[str, float]]  # Strategy -> metrics
    benchmark_comparison: dict[str, float]  # Comparison to buy-and-hold
    risk_adjusted_performance: dict[str, float]  # Strategy -> risk-adj returns
    drawdown_analysis: dict[str, dict[str, float]]  # Drawdown characteristics

    # Workflow status and control
    workflow_status: Annotated[
        str, take_latest_status
    ]  # analyzing_regime, selecting_strategies, optimizing, validating, completed
    current_step: str  # Current workflow step for progress tracking
    steps_completed: list[str]  # Completed workflow steps
    total_execution_time_ms: float  # Total workflow execution time

    # Error handling and recovery
    errors_encountered: list[dict[str, Any]]  # Errors with context
    fallback_strategies_used: list[str]  # Fallback strategies activated
    data_quality_issues: list[str]  # Data quality concerns identified

    # Caching and performance
    cached_results: dict[str, Any]  # Cached intermediate results
    cache_hit_rate: float  # Cache effectiveness
    api_calls_made: int  # Number of external API calls

    # Advanced analysis features
    regime_transition_analysis: dict[str, Any]  # Analysis of regime changes
    multi_timeframe_analysis: dict[str, dict[str, Any]]  # Analysis across timeframes
    correlation_analysis: dict[str, float]  # Inter-asset correlations
    macroeconomic_context: dict[str, Any]  # Macro environment factors
