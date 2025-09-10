"""
Intelligent Backtesting Workflow using LangGraph.

This workflow orchestrates market regime analysis, strategy selection, parameter optimization,
and validation to provide intelligent, confidence-scored backtesting recommendations.
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
        """Build the LangGraph workflow."""
        # Define the workflow graph
        workflow = StateGraph(BacktestingWorkflowState)

        # Add nodes for each step
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("analyze_market_regime", self._analyze_market_regime_node)
        workflow.add_node("select_strategies", self._select_strategies_node)
        workflow.add_node("optimize_parameters", self._optimize_parameters_node)
        workflow.add_node("validate_results", self._validate_results_node)
        workflow.add_node("finalize_workflow", self._finalize_workflow)

        # Define the workflow flow
        workflow.set_entry_point("initialize")

        # Sequential workflow with conditional routing
        workflow.add_edge("initialize", "analyze_market_regime")
        workflow.add_conditional_edges(
            "analyze_market_regime",
            self._should_proceed_after_market_analysis,
            {
                "continue": "select_strategies",
                "fallback": "finalize_workflow",
            },
        )
        workflow.add_conditional_edges(
            "select_strategies",
            self._should_proceed_after_strategy_selection,
            {
                "continue": "optimize_parameters",
                "fallback": "finalize_workflow",
            },
        )
        workflow.add_conditional_edges(
            "optimize_parameters",
            self._should_proceed_after_optimization,
            {
                "continue": "validate_results",
                "fallback": "finalize_workflow",
            },
        )
        workflow.add_edge("validate_results", "finalize_workflow")
        workflow.add_edge("finalize_workflow", END)

        return workflow.compile()

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
                "workflow_completed": final_state.workflow_status == "completed",
                "steps_completed": final_state.steps_completed,
                "errors_encountered": final_state.errors_encountered,
                "fallback_strategies_used": final_state.fallback_strategies_used,
            }

            logger.info(
                f"Intelligent backtest completed for {symbol} in {total_execution_time:.0f}ms: "
                f"{final_state.recommended_strategy} recommended with {final_state.recommendation_confidence:.1%} confidence"
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
        logger.info(f"Initializing backtesting workflow for {state.symbol}")

        # Validate inputs
        if not state.symbol:
            state.errors_encountered.append(
                {
                    "step": "initialization",
                    "error": "Symbol is required",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            state.workflow_status = "failed"
            return state

        # Update workflow state
        state.workflow_status = "analyzing_regime"
        state.current_step = "initialization_completed"
        state.steps_completed.append("initialization")

        logger.info(f"Workflow initialized for {state.symbol}")
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
        if state.workflow_status != "completed":
            # Handle incomplete workflow
            if not state.recommended_strategy and state.best_parameters:
                # Select first available strategy as fallback
                state.recommended_strategy = list(state.best_parameters.keys())[0]
                state.recommended_parameters = state.best_parameters[
                    state.recommended_strategy
                ]
                state.recommendation_confidence = 0.3
                state.fallback_strategies_used.append("incomplete_workflow_fallback")

        state.current_step = "workflow_finalized"
        logger.info(f"Workflow finalized for {state.symbol}")
        return state

    def _should_proceed_after_market_analysis(
        self, state: BacktestingWorkflowState
    ) -> str:
        """Decide whether to proceed after market analysis."""
        if state.errors_encountered and any(
            "market_regime_analysis" in err.get("step", "")
            for err in state.errors_encountered
        ):
            return "fallback"
        if state.market_regime == "unknown" and state.regime_confidence < 0.1:
            return "fallback"
        return "continue"

    def _should_proceed_after_strategy_selection(
        self, state: BacktestingWorkflowState
    ) -> str:
        """Decide whether to proceed after strategy selection."""
        if not state.selected_strategies:
            return "fallback"
        if state.strategy_selection_confidence < 0.2:
            return "fallback"
        return "continue"

    def _should_proceed_after_optimization(
        self, state: BacktestingWorkflowState
    ) -> str:
        """Decide whether to proceed after optimization."""
        if not state.best_parameters:
            return "fallback"
        return "continue"

    def _format_results(self, state: BacktestingWorkflowState) -> dict[str, Any]:
        """Format final results for output."""
        return {
            "symbol": state.symbol,
            "period": {
                "start_date": state.start_date,
                "end_date": state.end_date,
                "initial_capital": state.initial_capital,
            },
            "market_analysis": {
                "regime": state.market_regime,
                "regime_confidence": state.regime_confidence,
                "regime_indicators": state.regime_indicators,
                "volatility_percentile": state.volatility_percentile,
                "trend_strength": state.trend_strength,
                "market_conditions": state.market_conditions,
                "support_resistance_levels": state.support_resistance_levels,
            },
            "strategy_selection": {
                "selected_strategies": state.selected_strategies,
                "strategy_rankings": state.strategy_rankings,
                "selection_reasoning": state.strategy_selection_reasoning,
                "selection_confidence": state.strategy_selection_confidence,
                "candidate_strategies": state.candidate_strategies,
            },
            "optimization": {
                "optimization_config": state.optimization_config,
                "best_parameters": state.best_parameters,
                "optimization_iterations": state.optimization_iterations,
                "optimization_time_ms": state.optimization_time_ms,
            },
            "validation": {
                "robustness_scores": state.robustness_score,
                "validation_warnings": state.validation_warnings,
                "out_of_sample_performance": state.out_of_sample_performance,
            },
            "recommendation": {
                "recommended_strategy": state.recommended_strategy,
                "recommended_parameters": state.recommended_parameters,
                "recommendation_confidence": state.recommendation_confidence,
                "final_strategy_ranking": state.final_strategy_ranking,
                "risk_assessment": state.risk_assessment,
            },
            "performance_analysis": {
                "comparative_metrics": state.comparative_metrics,
                "benchmark_comparison": state.benchmark_comparison,
                "risk_adjusted_performance": state.risk_adjusted_performance,
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
        completed_steps = len(state.steps_completed)

        return {
            "workflow_status": state.workflow_status,
            "current_step": state.current_step,
            "progress_percentage": (completed_steps / total_steps) * 100,
            "steps_completed": state.steps_completed,
            "errors_count": len(state.errors_encountered),
            "warnings_count": len(state.validation_warnings),
            "execution_time_ms": state.total_execution_time_ms,
            "recommended_strategy": state.recommended_strategy or "TBD",
            "recommendation_confidence": state.recommendation_confidence,
        }
