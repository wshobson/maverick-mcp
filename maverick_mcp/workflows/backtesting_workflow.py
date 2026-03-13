"""
Intelligent Backtesting Workflow using LangGraph.

This workflow orchestrates market regime analysis, strategy selection, parameter optimization,
and validation to provide intelligent, confidence-scored backtesting recommendations.

Includes replanning capability: when intermediate steps produce low-confidence results,
the workflow can retry with adjusted parameters before falling back to the safety net.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from maverick_mcp.workflows.agents import (
    MarketAnalyzerAgent,
    OptimizerAgent,
    StrategySelectorAgent,
    ValidatorAgent,
)
from maverick_mcp.workflows.state import BacktestingWorkflowState

logger = logging.getLogger(__name__)

# Default maximum retries per step
DEFAULT_MAX_RETRIES = 2

# How much to extend the lookback window on each market regime retry (days)
_REGIME_RETRY_LOOKBACK_EXTENSION_DAYS = 180

# Lowered confidence thresholds used when broadening strategy selection on retry
_STRATEGY_RETRY_RELAXED_CONFIDENCE_FLOOR = 0.10


class BacktestingWorkflow:
    """Intelligent backtesting workflow orchestrator."""

    def __init__(
        self,
        market_analyzer: MarketAnalyzerAgent | None = None,
        strategy_selector: StrategySelectorAgent | None = None,
        optimizer: OptimizerAgent | None = None,
        validator: ValidatorAgent | None = None,
    ):
        """Initialize backtesting workflow.

        Args:
            market_analyzer: Market regime analysis agent
            strategy_selector: Strategy selection agent
            optimizer: Parameter optimization agent
            validator: Results validation agent
        """
        self.market_analyzer = market_analyzer or MarketAnalyzerAgent()
        self.strategy_selector = strategy_selector or StrategySelectorAgent()
        self.optimizer = optimizer or OptimizerAgent()
        self.validator = validator or ValidatorAgent()

        # Build the workflow graph
        self.workflow = self._build_workflow_graph()

        logger.info("BacktestingWorkflow initialized")

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with replanning paths.

        The graph supports three replanning loops:
        1. After market regime analysis: retry with extended data (max 2 retries)
        2. After strategy selection: retry with broadened parameters (max 1 retry)
        3. After validation: re-optimize with relaxed constraints before fallback
        """
        # Define the workflow graph
        workflow = StateGraph(BacktestingWorkflowState)

        # Add nodes for each step
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("analyze_market_regime", self._analyze_market_regime_node)
        workflow.add_node("retry_market_regime", self._retry_market_regime_node)
        workflow.add_node("select_strategies", self._select_strategies_node)
        workflow.add_node(
            "retry_strategy_selection", self._retry_strategy_selection_node
        )
        workflow.add_node("optimize_parameters", self._optimize_parameters_node)
        workflow.add_node("validate_results", self._validate_results_node)
        workflow.add_node("reoptimize_relaxed", self._reoptimize_relaxed_node)
        workflow.add_node("finalize_workflow", self._finalize_workflow)

        # Define the workflow flow
        workflow.set_entry_point("initialize")

        # Sequential workflow with conditional routing and replanning edges
        workflow.add_edge("initialize", "analyze_market_regime")

        # After market regime analysis: continue, retry with extended data, or fallback
        workflow.add_conditional_edges(
            "analyze_market_regime",
            self._route_after_market_analysis,
            {
                "continue": "select_strategies",
                "retry": "retry_market_regime",
                "fallback": "finalize_workflow",
            },
        )

        # Retry market regime feeds back into the same routing decision
        workflow.add_conditional_edges(
            "retry_market_regime",
            self._route_after_market_analysis,
            {
                "continue": "select_strategies",
                "retry": "retry_market_regime",
                "fallback": "finalize_workflow",
            },
        )

        # After strategy selection: continue, retry with broadened params, or fallback
        workflow.add_conditional_edges(
            "select_strategies",
            self._route_after_strategy_selection,
            {
                "continue": "optimize_parameters",
                "retry": "retry_strategy_selection",
                "fallback": "finalize_workflow",
            },
        )

        # Retry strategy selection feeds back into the same routing decision
        workflow.add_conditional_edges(
            "retry_strategy_selection",
            self._route_after_strategy_selection,
            {
                "continue": "optimize_parameters",
                "retry": "retry_strategy_selection",
                "fallback": "finalize_workflow",
            },
        )

        # After optimization: continue or fallback (unchanged)
        workflow.add_conditional_edges(
            "optimize_parameters",
            self._should_proceed_after_optimization,
            {
                "continue": "validate_results",
                "fallback": "finalize_workflow",
            },
        )

        # After validation: finalize, or re-optimize with relaxed constraints
        workflow.add_conditional_edges(
            "validate_results",
            self._route_after_validation,
            {
                "finalize": "finalize_workflow",
                "reoptimize": "reoptimize_relaxed",
            },
        )

        # Re-optimization with relaxed constraints goes to validation then finalize
        workflow.add_edge("reoptimize_relaxed", "validate_results")

        workflow.add_edge("finalize_workflow", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Retry / replanning nodes
    # ------------------------------------------------------------------

    async def _retry_market_regime_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Retry market regime analysis with extended historical data.

        Extends the start_date further into the past so the market analyzer
        has more data points to work with, which can improve regime detection
        confidence.
        """
        retry_key = "market_regime"
        current_retries = state["retry_count"].get(retry_key, 0)
        state["retry_count"][retry_key] = current_retries + 1

        logger.info(
            f"Replanning market regime analysis for {state['symbol']} "
            f"(retry {state['retry_count'][retry_key]}/{state['max_retries']}): "
            f"extending lookback period"
        )

        # Extend start_date to provide more historical data
        try:
            original_start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            original_start = datetime.now() - timedelta(days=365)

        extended_start = original_start - timedelta(
            days=_REGIME_RETRY_LOOKBACK_EXTENSION_DAYS
        )
        state["start_date"] = extended_start.strftime("%Y-%m-%d")

        # Clear previous regime results so the analyzer starts fresh
        state["market_regime"] = "unknown"
        state["regime_confidence"] = 0.0
        state["regime_indicators"] = {}

        # Record the replanning event
        state["fallback_strategies_used"].append(
            f"market_regime_retry_{state['retry_count'][retry_key]}"
        )

        # Re-run market regime analysis with extended data
        return await self.market_analyzer.analyze_market_regime(state)

    async def _retry_strategy_selection_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Retry strategy selection with broadened selection parameters.

        Lowers the minimum fitness threshold and increases the candidate pool
        so that strategies which were previously filtered out can be considered.
        """
        retry_key = "strategy_selection"
        current_retries = state["retry_count"].get(retry_key, 0)
        state["retry_count"][retry_key] = current_retries + 1

        logger.info(
            f"Replanning strategy selection for {state['symbol']} "
            f"(retry {state['retry_count'][retry_key]}/1): "
            f"broadening selection parameters"
        )

        # Clear previous strategy selection results
        state["selected_strategies"] = []
        state["strategy_rankings"] = {}
        state["candidate_strategies"] = []
        state["strategy_selection_confidence"] = 0.0
        state["strategy_selection_reasoning"] = ""

        # If regime confidence is very low, treat it as a neutral/unknown regime
        # which gives more balanced fitness scores across all strategies
        if state["regime_confidence"] < 0.3:
            state["market_regime"] = "unknown"
            logger.info(
                f"Reset market regime to 'unknown' for broader strategy consideration "
                f"(original confidence was {state['regime_confidence']:.2f})"
            )

        # Record the replanning event
        state["fallback_strategies_used"].append(
            f"strategy_selection_retry_{state['retry_count'][retry_key]}"
        )

        # Re-run strategy selection (the selector will use the updated state)
        return await self.strategy_selector.select_strategies(state)

    async def _reoptimize_relaxed_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Re-optimize with relaxed constraints after validation failure.

        When validation produces low robustness scores, this node loosens the
        optimization constraints (e.g., accepting higher drawdowns, lower win
        rates) and re-runs the optimizer before falling back.
        """
        retry_key = "validation_reoptimize"
        current_retries = state["retry_count"].get(retry_key, 0)
        state["retry_count"][retry_key] = current_retries + 1

        logger.info(
            "Replanning optimization for %s with relaxed constraints "
            "(attempt %d/%d): relaxing validation thresholds after low robustness scores",
            state["symbol"],
            state["retry_count"][retry_key],
            state["max_retries"],
        )

        # Relax the optimization config to allow more trades through
        optimization_config = state["optimization_config"]
        if optimization_config:
            relaxed_config = dict(optimization_config)
            # Increase drawdown tolerance by 50%
            if "max_drawdown_limit" in relaxed_config:
                relaxed_config["max_drawdown_limit"] = min(
                    relaxed_config["max_drawdown_limit"] * 1.5, 0.5
                )
            # Lower the minimum trades threshold
            if "min_trades" in relaxed_config:
                relaxed_config["min_trades"] = max(
                    int(relaxed_config["min_trades"] * 0.6), 3
                )
            state["optimization_config"] = relaxed_config

        # Clear previous optimization and validation results
        state["best_parameters"] = {}
        state["optimization_results"] = {}
        state["walk_forward_results"] = {}
        state["monte_carlo_results"] = {}
        state["out_of_sample_performance"] = {}
        state["robustness_score"] = {}
        state["validation_warnings"] = []
        state["final_strategy_ranking"] = []
        state["recommended_strategy"] = ""
        state["recommended_parameters"] = {}
        state["recommendation_confidence"] = 0.0

        # Record the replanning event
        state["fallback_strategies_used"].append(
            f"validation_reoptimize_{state['retry_count'][retry_key]}"
        )

        # Re-run optimization with relaxed constraints
        return await self.optimizer.optimize_parameters(state)

    # ------------------------------------------------------------------
    # Routing / conditional edge functions
    # ------------------------------------------------------------------

    def _route_after_market_analysis(self, state: BacktestingWorkflowState) -> str:
        """Route after market regime analysis: continue, retry, or fallback.

        Retry logic (max retries from state.max_retries, default 2):
        - If there are errors in the market regime analysis step, retry with
          extended data before giving up.
        - If the regime is unknown with very low confidence, retry to get more
          data points.
        - After max retries, fall back to finalize_workflow.
        """
        retry_key = "market_regime"
        current_retries = state["retry_count"].get(retry_key, 0)
        max_retries = state["max_retries"]

        has_errors = state["errors_encountered"] and any(
            "market_regime_analysis" in err.get("step", "")
            for err in state["errors_encountered"]
        )
        is_low_confidence = (
            state["market_regime"] == "unknown" and state["regime_confidence"] < 0.1
        )

        if has_errors or is_low_confidence:
            if current_retries < max_retries:
                logger.info(
                    f"Market regime analysis produced low-confidence results "
                    f"(regime={state['market_regime']}, confidence={state['regime_confidence']:.2f}). "
                    f"Retrying ({current_retries + 1}/{max_retries})."
                )
                return "retry"
            else:
                logger.warning(
                    f"Market regime analysis failed after {current_retries} retries. "
                    f"Falling back to finalize."
                )
                return "fallback"

        return "continue"

    def _route_after_strategy_selection(self, state: BacktestingWorkflowState) -> str:
        """Route after strategy selection: continue, retry, or fallback.

        Retry logic (max 1 retry):
        - If no strategies meet the minimum fitness criteria, broaden
          selection parameters and retry once.
        - If strategy selection confidence is too low, retry once.
        """
        retry_key = "strategy_selection"
        current_retries = state["retry_count"].get(retry_key, 0)
        # Strategy selection gets at most 1 retry
        max_strategy_retries = min(1, state["max_retries"])

        no_strategies = not state["selected_strategies"]
        low_confidence = (
            state["strategy_selection_confidence"]
            < _STRATEGY_RETRY_RELAXED_CONFIDENCE_FLOOR * 2
        )

        if no_strategies or low_confidence:
            if current_retries < max_strategy_retries:
                logger.info(
                    f"Strategy selection insufficient "
                    f"(strategies={len(state['selected_strategies'])}, "
                    f"confidence={state['strategy_selection_confidence']:.2f}). "
                    f"Retrying with broadened parameters ({current_retries + 1}/{max_strategy_retries})."
                )
                return "retry"
            else:
                logger.warning(
                    f"Strategy selection failed after {current_retries} retries. "
                    f"Falling back to finalize."
                )
                return "fallback"

        return "continue"

    def _route_after_validation(self, state: BacktestingWorkflowState) -> str:
        """Route after validation: finalize or re-optimize with relaxed constraints.

        If validation produces low robustness scores across all strategies,
        retry optimization with relaxed constraints before accepting the result.
        Only retries once.
        """
        retry_key = "validation_reoptimize"
        current_retries = state["retry_count"].get(retry_key, 0)
        max_validation_retries = min(1, state["max_retries"])

        # Check if validation produced acceptable results
        robustness_score = state["robustness_score"]
        if robustness_score:
            max_robustness = max(robustness_score.values())
            if max_robustness >= 0.5:
                # At least one strategy passed validation adequately
                return "finalize"

            # Low robustness across all strategies
            if current_retries < max_validation_retries:
                logger.info(
                    "Validation produced low robustness scores "
                    "(max=%.2f). Re-optimizing with relaxed constraints (%d/%d).",
                    max_robustness,
                    current_retries + 1,
                    max_validation_retries,
                )
                return "reoptimize"

            logger.warning(
                "Validation still below threshold after re-optimization. "
                "Proceeding to finalize with best available results."
            )

        return "finalize"

    # ------------------------------------------------------------------
    # Original node methods (preserved)
    # ------------------------------------------------------------------

    async def run_intelligent_backtest(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
        requested_strategy: str | None = None,
    ) -> dict[str, Any]:
        """Run intelligent backtesting workflow.

        Args:
            symbol: Stock symbol to analyze
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today
            initial_capital: Starting capital for backtest
            requested_strategy: User-requested strategy (optional)

        Returns:
            Comprehensive backtesting results with recommendations
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting intelligent backtest workflow for {symbol}")

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            # Initialize workflow state
            initial_state = self._create_initial_state(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                requested_strategy=requested_strategy,
            )

            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Convert state to results dictionary
            results = self._format_results(final_state)

            # Add execution metadata
            total_execution_time = (datetime.now() - start_time).total_seconds() * 1000
            results["execution_metadata"] = {
                "total_execution_time_ms": total_execution_time,
                "workflow_completed": final_state["workflow_status"] == "completed",
                "steps_completed": final_state["steps_completed"],
                "errors_encountered": final_state["errors_encountered"],
                "fallback_strategies_used": final_state["fallback_strategies_used"],
                "retry_counts": final_state["retry_count"],
            }

            logger.info(
                "Intelligent backtest completed for %s in %.0fms: "
                "%s recommended with %.1f%% confidence",
                symbol,
                total_execution_time,
                final_state["recommended_strategy"],
                final_state["recommendation_confidence"] * 100,
            )

            return results

        except Exception as e:
            logger.error(f"Intelligent backtest failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "execution_metadata": {
                    "total_execution_time_ms": (
                        datetime.now() - start_time
                    ).total_seconds()
                    * 1000,
                    "workflow_completed": False,
                },
            }

    def _create_initial_state(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        requested_strategy: str | None,
    ) -> BacktestingWorkflowState:
        """Create initial workflow state."""
        return BacktestingWorkflowState(
            # Base agent state
            messages=[
                HumanMessage(content=f"Analyze backtesting opportunities for {symbol}")
            ],
            session_id=f"backtest_{symbol}_{datetime.now().isoformat()}",
            persona="intelligent_backtesting_agent",
            timestamp=datetime.now(),
            token_count=0,
            error=None,
            analyzed_stocks={},
            key_price_levels={},
            last_analysis_time={},
            conversation_context={},
            execution_time_ms=None,
            api_calls_made=0,
            cache_hits=0,
            cache_misses=0,
            # Input parameters
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            requested_strategy=requested_strategy,
            # Market regime analysis (initialized)
            market_regime="unknown",
            regime_confidence=0.0,
            regime_indicators={},
            regime_analysis_time_ms=0.0,
            volatility_percentile=0.0,
            trend_strength=0.0,
            market_conditions={},
            sector_performance={},
            correlation_to_market=0.0,
            volume_profile={},
            support_resistance_levels=[],
            # Strategy selection (initialized)
            candidate_strategies=[],
            strategy_rankings={},
            selected_strategies=[],
            strategy_selection_reasoning="",
            strategy_selection_confidence=0.0,
            # Parameter optimization (initialized)
            optimization_config={},
            parameter_grids={},
            optimization_results={},
            best_parameters={},
            optimization_time_ms=0.0,
            optimization_iterations=0,
            # Validation (initialized)
            walk_forward_results={},
            monte_carlo_results={},
            out_of_sample_performance={},
            robustness_score={},
            validation_warnings=[],
            # Final recommendations (initialized)
            final_strategy_ranking=[],
            recommended_strategy="",
            recommended_parameters={},
            recommendation_confidence=0.0,
            risk_assessment={},
            # Performance metrics (initialized)
            comparative_metrics={},
            benchmark_comparison={},
            risk_adjusted_performance={},
            drawdown_analysis={},
            # Workflow control (initialized)
            workflow_status="initializing",
            current_step="initialization",
            steps_completed=[],
            total_execution_time_ms=0.0,
            # Replanning control (initialized)
            retry_count={},
            max_retries=DEFAULT_MAX_RETRIES,
            # Error handling (initialized)
            errors_encountered=[],
            fallback_strategies_used=[],
            data_quality_issues=[],
            # Caching (initialized)
            cached_results={},
            cache_hit_rate=0.0,
            # Advanced analysis (initialized)
            regime_transition_analysis={},
            multi_timeframe_analysis={},
            correlation_analysis={},
            macroeconomic_context={},
        )

    async def _initialize_workflow(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Initialize the workflow and validate inputs."""
        logger.info("Initializing backtesting workflow for %s", state["symbol"])

        # Validate inputs
        if not state["symbol"]:
            state["errors_encountered"].append(
                {
                    "step": "initialization",
                    "error": "Symbol is required",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            state["workflow_status"] = "failed"
            return state

        # Ensure retry_count and max_retries are initialized
        if not state.get("retry_count"):
            state["retry_count"] = {}
        if not state.get("max_retries"):
            state["max_retries"] = DEFAULT_MAX_RETRIES

        # Update workflow state
        state["workflow_status"] = "analyzing_regime"
        state["current_step"] = "initialization_completed"
        state["steps_completed"].append("initialization")

        logger.info("Workflow initialized for %s", state["symbol"])
        return state

    async def _analyze_market_regime_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Market regime analysis node."""
        return await self.market_analyzer.analyze_market_regime(state)

    async def _select_strategies_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Strategy selection node."""
        return await self.strategy_selector.select_strategies(state)

    async def _optimize_parameters_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Parameter optimization node."""
        return await self.optimizer.optimize_parameters(state)

    async def _validate_results_node(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Results validation node."""
        return await self.validator.validate_strategies(state)

    async def _finalize_workflow(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Finalize the workflow and prepare results."""
        if state["workflow_status"] != "completed":
            # Handle incomplete workflow
            if not state["recommended_strategy"] and state["best_parameters"]:
                # Select first available strategy as fallback
                fallback = list(state["best_parameters"].keys())[0]
                state["recommended_strategy"] = fallback
                state["recommended_parameters"] = state["best_parameters"][fallback]
                state["recommendation_confidence"] = 0.3
                state["fallback_strategies_used"].append("incomplete_workflow_fallback")

        state["current_step"] = "workflow_finalized"
        logger.info("Workflow finalized for %s", state["symbol"])
        return state

    def _should_proceed_after_optimization(
        self, state: BacktestingWorkflowState
    ) -> str:
        """Decide whether to proceed after optimization."""
        if not state["best_parameters"]:
            return "fallback"
        return "continue"

    def _format_results(self, state: BacktestingWorkflowState) -> dict[str, Any]:
        """Format final results for output."""
        return {
            "symbol": state["symbol"],
            "period": {
                "start_date": state["start_date"],
                "end_date": state["end_date"],
                "initial_capital": state["initial_capital"],
            },
            "market_analysis": {
                "regime": state["market_regime"],
                "regime_confidence": state["regime_confidence"],
                "regime_indicators": state["regime_indicators"],
                "volatility_percentile": state["volatility_percentile"],
                "trend_strength": state["trend_strength"],
                "market_conditions": state["market_conditions"],
                "support_resistance_levels": state["support_resistance_levels"],
            },
            "strategy_selection": {
                "selected_strategies": state["selected_strategies"],
                "strategy_rankings": state["strategy_rankings"],
                "selection_reasoning": state["strategy_selection_reasoning"],
                "selection_confidence": state["strategy_selection_confidence"],
                "candidate_strategies": state["candidate_strategies"],
            },
            "optimization": {
                "optimization_config": state["optimization_config"],
                "best_parameters": state["best_parameters"],
                "optimization_iterations": state["optimization_iterations"],
                "optimization_time_ms": state["optimization_time_ms"],
            },
            "validation": {
                "robustness_scores": state["robustness_score"],
                "validation_warnings": state["validation_warnings"],
                "out_of_sample_performance": state["out_of_sample_performance"],
            },
            "recommendation": {
                "recommended_strategy": state["recommended_strategy"],
                "recommended_parameters": state["recommended_parameters"],
                "recommendation_confidence": state["recommendation_confidence"],
                "final_strategy_ranking": state["final_strategy_ranking"],
                "risk_assessment": state["risk_assessment"],
            },
            "performance_analysis": {
                "comparative_metrics": state["comparative_metrics"],
                "benchmark_comparison": state["benchmark_comparison"],
                "risk_adjusted_performance": state["risk_adjusted_performance"],
            },
        }

    async def run_quick_analysis(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 10000.0,
    ) -> dict[str, Any]:
        """Run quick analysis with market regime detection and basic strategy recommendations.

        This is a faster alternative that skips parameter optimization and validation
        for rapid insights.

        Args:
            symbol: Stock symbol to analyze
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital

        Returns:
            Quick analysis results with strategy recommendations
        """
        start_time = datetime.now()

        try:
            logger.info(f"Running quick analysis for {symbol}")

            # Set default dates
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            # Create initial state
            state = self._create_initial_state(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                requested_strategy=None,
            )

            # Run market analysis
            state = await self.market_analyzer.analyze_market_regime(state)

            # Run strategy selection
            if state["market_regime"] != "unknown" or state["regime_confidence"] > 0.3:
                state = await self.strategy_selector.select_strategies(state)

            # Format quick results
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "symbol": symbol,
                "analysis_type": "quick_analysis",
                "market_regime": {
                    "regime": state["market_regime"],
                    "confidence": state["regime_confidence"],
                    "trend_strength": state["trend_strength"],
                    "volatility_percentile": state["volatility_percentile"],
                },
                "recommended_strategies": state["selected_strategies"][:3],  # Top 3
                "strategy_fitness": {
                    strategy: state["strategy_rankings"].get(strategy, 0)
                    for strategy in state["selected_strategies"][:3]
                },
                "market_conditions": state["market_conditions"],
                "selection_reasoning": state["strategy_selection_reasoning"],
                "execution_time_ms": execution_time,
                "data_quality": {
                    "errors": len(state["errors_encountered"]),
                    "warnings": state["data_quality_issues"],
                },
            }

        except Exception as e:
            logger.error(f"Quick analysis failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "analysis_type": "quick_analysis",
                "error": str(e),
                "execution_time_ms": (datetime.now() - start_time).total_seconds()
                * 1000,
            }

    def get_workflow_status(self, state: BacktestingWorkflowState) -> dict[str, Any]:
        """Get current workflow status and progress."""
        total_steps = 5  # initialize, analyze, select, optimize, validate
        completed_steps = len(state["steps_completed"])

        return {
            "workflow_status": state["workflow_status"],
            "current_step": state["current_step"],
            "progress_percentage": (completed_steps / total_steps) * 100,
            "steps_completed": state["steps_completed"],
            "errors_count": len(state["errors_encountered"]),
            "warnings_count": len(state["validation_warnings"]),
            "execution_time_ms": state["total_execution_time_ms"],
            "recommended_strategy": state["recommended_strategy"] or "TBD",
            "recommendation_confidence": state["recommendation_confidence"],
            "retry_counts": state["retry_count"],
        }
