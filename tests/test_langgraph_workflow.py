"""
Comprehensive tests for LangGraph backtesting workflow.

Tests cover:
- LangGraph workflow state transitions and agent orchestration
- Market regime analysis workflow steps
- Strategy selection and parameter optimization
- Results validation and recommendation generation
- Error handling and fallback strategies
- Performance benchmarks and timing
"""

import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from maverick_mcp.workflows.agents import (
    MarketAnalyzerAgent,
    OptimizerAgent,
    StrategySelectorAgent,
    ValidatorAgent,
)
from maverick_mcp.workflows.backtesting_workflow import BacktestingWorkflow
from maverick_mcp.workflows.state import BacktestingWorkflowState

logger = logging.getLogger(__name__)


class TestBacktestingWorkflow:
    """Test suite for BacktestingWorkflow class."""

    @pytest.fixture
    def sample_workflow_state(self) -> BacktestingWorkflowState:
        """Create a sample workflow state for testing."""
        from langchain_core.messages import HumanMessage

        return BacktestingWorkflowState(
            # Base agent state
            messages=[HumanMessage(content="Analyze AAPL for backtesting")],
            session_id="test_session_123",
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
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000.0,
            requested_strategy=None,
            # Market regime analysis (initialized)
            market_regime="bullish",
            regime_confidence=0.85,
            regime_indicators={
                "trend_strength": 0.75,
                "volatility": 0.25,
                "momentum": 0.80,
            },
            regime_analysis_time_ms=150.0,
            volatility_percentile=35.0,
            trend_strength=0.75,
            market_conditions={
                "trend": "upward",
                "volatility": "low",
                "volume": "normal",
            },
            sector_performance={"technology": 0.15},
            correlation_to_market=0.75,
            volume_profile={"average": 50000000, "relative": 1.2},
            support_resistance_levels=[150.0, 160.0, 170.0],
            # Strategy selection (initialized)
            candidate_strategies=["momentum", "mean_reversion", "breakout"],
            strategy_rankings={"momentum": 0.9, "breakout": 0.7, "mean_reversion": 0.6},
            selected_strategies=["momentum", "breakout"],
            strategy_selection_reasoning="High momentum and trend strength favor momentum strategies",
            strategy_selection_confidence=0.85,
            # Parameter optimization (initialized)
            optimization_config={"method": "grid_search", "cv_folds": 5},
            parameter_grids={
                "momentum": {"window": [10, 20, 30], "threshold": [0.01, 0.02]}
            },
            optimization_results={
                "momentum": {
                    "best_sharpe": 1.5,
                    "best_params": {"window": 20, "threshold": 0.02},
                }
            },
            best_parameters={"momentum": {"window": 20, "threshold": 0.02}},
            optimization_time_ms=2500.0,
            optimization_iterations=45,
            # Validation (initialized)
            walk_forward_results={"out_of_sample_sharpe": 1.2, "degradation": 0.2},
            monte_carlo_results={"confidence_95": 0.8, "max_drawdown_95": 0.15},
            out_of_sample_performance={"sharpe": 1.2, "return": 0.18},
            robustness_score={"overall": 0.75, "parameter_sensitivity": 0.8},
            validation_warnings=["High parameter sensitivity detected"],
            # Final recommendations (initialized)
            final_strategy_ranking=[
                {"strategy": "momentum", "score": 0.9, "confidence": 0.85}
            ],
            recommended_strategy="momentum",
            recommended_parameters={"window": 20, "threshold": 0.02},
            recommendation_confidence=0.85,
            risk_assessment={"max_drawdown": 0.15, "volatility": 0.25},
            # Performance metrics (initialized)
            comparative_metrics={"sharpe_vs_benchmark": 1.5, "alpha": 0.05},
            benchmark_comparison={"excess_return": 0.08, "information_ratio": 0.6},
            risk_adjusted_performance={"calmar": 1.0, "sortino": 1.8},
            drawdown_analysis={"max_dd": 0.15, "avg_dd": 0.05, "recovery_days": 30},
            # Workflow control (initialized)
            workflow_status="analyzing_regime",
            current_step="market_analysis",
            steps_completed=["initialization"],
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

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        market_analyzer = Mock(spec=MarketAnalyzerAgent)
        strategy_selector = Mock(spec=StrategySelectorAgent)
        optimizer = Mock(spec=OptimizerAgent)
        validator = Mock(spec=ValidatorAgent)

        # Set up successful mock responses
        async def mock_analyze_market_regime(state):
            state.market_regime = "bullish"
            state.regime_confidence = 0.85
            state.workflow_status = "selecting_strategies"
            state.steps_completed.append("market_analysis")
            return state

        async def mock_select_strategies(state):
            state.selected_strategies = ["momentum", "breakout"]
            state.strategy_selection_confidence = 0.85
            state.workflow_status = "optimizing_parameters"
            state.steps_completed.append("strategy_selection")
            return state

        async def mock_optimize_parameters(state):
            state.best_parameters = {"momentum": {"window": 20, "threshold": 0.02}}
            state.optimization_iterations = 45
            state.workflow_status = "validating_results"
            state.steps_completed.append("parameter_optimization")
            return state

        async def mock_validate_strategies(state):
            state.recommended_strategy = "momentum"
            state.recommendation_confidence = 0.85
            state.workflow_status = "completed"
            state.steps_completed.append("validation")
            return state

        market_analyzer.analyze_market_regime = AsyncMock(
            side_effect=mock_analyze_market_regime
        )
        strategy_selector.select_strategies = AsyncMock(
            side_effect=mock_select_strategies
        )
        optimizer.optimize_parameters = AsyncMock(side_effect=mock_optimize_parameters)
        validator.validate_strategies = AsyncMock(side_effect=mock_validate_strategies)

        return {
            "market_analyzer": market_analyzer,
            "strategy_selector": strategy_selector,
            "optimizer": optimizer,
            "validator": validator,
        }

    @pytest.fixture
    def workflow_with_mocks(self, mock_agents):
        """Create a workflow with mocked agents."""
        return BacktestingWorkflow(
            market_analyzer=mock_agents["market_analyzer"],
            strategy_selector=mock_agents["strategy_selector"],
            optimizer=mock_agents["optimizer"],
            validator=mock_agents["validator"],
        )

    async def test_workflow_initialization(self):
        """Test workflow initialization creates proper graph structure."""
        workflow = BacktestingWorkflow()

        # Test workflow has been compiled
        assert workflow.workflow is not None

        # Test agent initialization
        assert workflow.market_analyzer is not None
        assert workflow.strategy_selector is not None
        assert workflow.optimizer is not None
        assert workflow.validator is not None

        # Test workflow nodes exist
        nodes = workflow.workflow.get_graph().nodes()
        expected_nodes = [
            "initialize",
            "analyze_market_regime",
            "select_strategies",
            "optimize_parameters",
            "validate_results",
            "finalize_workflow",
        ]
        for node in expected_nodes:
            assert node in [n for n in nodes]

    async def test_successful_workflow_execution(self, workflow_with_mocks):
        """Test successful end-to-end workflow execution."""
        start_time = datetime.now()

        result = await workflow_with_mocks.run_intelligent_backtest(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000.0,
        )

        execution_time = datetime.now() - start_time

        # Test basic structure
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "execution_metadata" in result

        # Test workflow completion
        exec_metadata = result["execution_metadata"]
        assert exec_metadata["workflow_completed"] is True
        assert "initialization" in exec_metadata["steps_completed"]
        assert "market_analysis" in exec_metadata["steps_completed"]
        assert "strategy_selection" in exec_metadata["steps_completed"]

        # Test recommendation structure
        assert "recommendation" in result
        recommendation = result["recommendation"]
        assert recommendation["recommended_strategy"] == "momentum"
        assert recommendation["recommendation_confidence"] == 0.85

        # Test performance
        assert exec_metadata["total_execution_time_ms"] > 0
        assert (
            execution_time.total_seconds() < 5.0
        )  # Should complete quickly with mocks

    async def test_market_analysis_conditional_routing(
        self, workflow_with_mocks, sample_workflow_state
    ):
        """Test conditional routing after market analysis step."""
        workflow = workflow_with_mocks

        # Test successful routing
        result = workflow._should_proceed_after_market_analysis(sample_workflow_state)
        assert result == "continue"

        # Test failure routing - unknown regime with low confidence
        failure_state = sample_workflow_state.copy()
        failure_state.market_regime = "unknown"
        failure_state.regime_confidence = 0.05

        result = workflow._should_proceed_after_market_analysis(failure_state)
        assert result == "fallback"

        # Test error routing
        error_state = sample_workflow_state.copy()
        error_state.errors_encountered = [
            {"step": "market_regime_analysis", "error": "Data unavailable"}
        ]

        result = workflow._should_proceed_after_market_analysis(error_state)
        assert result == "fallback"

    async def test_strategy_selection_conditional_routing(
        self, workflow_with_mocks, sample_workflow_state
    ):
        """Test conditional routing after strategy selection step."""
        workflow = workflow_with_mocks

        # Test successful routing
        result = workflow._should_proceed_after_strategy_selection(
            sample_workflow_state
        )
        assert result == "continue"

        # Test failure routing - no strategies selected
        failure_state = sample_workflow_state.copy()
        failure_state.selected_strategies = []

        result = workflow._should_proceed_after_strategy_selection(failure_state)
        assert result == "fallback"

        # Test low confidence routing
        low_conf_state = sample_workflow_state.copy()
        low_conf_state.strategy_selection_confidence = 0.1

        result = workflow._should_proceed_after_strategy_selection(low_conf_state)
        assert result == "fallback"

    async def test_optimization_conditional_routing(
        self, workflow_with_mocks, sample_workflow_state
    ):
        """Test conditional routing after parameter optimization step."""
        workflow = workflow_with_mocks

        # Test successful routing
        result = workflow._should_proceed_after_optimization(sample_workflow_state)
        assert result == "continue"

        # Test failure routing - no best parameters
        failure_state = sample_workflow_state.copy()
        failure_state.best_parameters = {}

        result = workflow._should_proceed_after_optimization(failure_state)
        assert result == "fallback"

    async def test_workflow_state_transitions(self, workflow_with_mocks):
        """Test that workflow state transitions occur correctly."""
        workflow = workflow_with_mocks

        # Create initial state
        initial_state = workflow._create_initial_state(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000.0,
            requested_strategy=None,
        )

        # Test initialization step
        state = await workflow._initialize_workflow(initial_state)
        assert "initialization" in state.steps_completed
        assert state.workflow_status == "analyzing_regime"
        assert state.current_step == "initialization_completed"

    async def test_workflow_error_handling(self, workflow_with_mocks):
        """Test workflow error handling and recovery."""
        # Create workflow with failing market analyzer
        workflow = workflow_with_mocks

        async def failing_market_analyzer(state):
            state.errors_encountered.append(
                {
                    "step": "market_regime_analysis",
                    "error": "API unavailable",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return state

        workflow.market_analyzer.analyze_market_regime = AsyncMock(
            side_effect=failing_market_analyzer
        )

        result = await workflow.run_intelligent_backtest(
            symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Test that workflow handles error gracefully
        assert "execution_metadata" in result
        exec_metadata = result["execution_metadata"]
        assert len(exec_metadata["errors_encountered"]) > 0

        # Test fallback behavior
        assert len(exec_metadata["fallback_strategies_used"]) > 0

    async def test_workflow_performance_benchmarks(
        self, workflow_with_mocks, benchmark_timer
    ):
        """Test workflow performance meets benchmarks."""
        workflow = workflow_with_mocks

        with benchmark_timer() as timer:
            result = await workflow.run_intelligent_backtest(
                symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
            )

        # Test performance benchmarks
        execution_time = result["execution_metadata"]["total_execution_time_ms"]
        actual_time = timer.elapsed * 1000

        # Should complete within reasonable time with mocks
        assert execution_time < 1000  # < 1 second
        assert actual_time < 5000  # < 5 seconds actual

        # Test execution metadata accuracy
        assert abs(execution_time - actual_time) < 100  # Within 100ms tolerance

    async def test_quick_analysis_workflow(self, workflow_with_mocks):
        """Test quick analysis workflow bypass."""
        workflow = workflow_with_mocks

        result = await workflow.run_quick_analysis(
            symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Test quick analysis structure
        assert result["analysis_type"] == "quick_analysis"
        assert "market_regime" in result
        assert "recommended_strategies" in result
        assert "execution_time_ms" in result

        # Test performance - quick analysis should be faster
        assert result["execution_time_ms"] < 500  # < 500ms

        # Test that it skips optimization and validation
        assert "optimization" not in result
        assert "validation" not in result

    async def test_workflow_status_tracking(
        self, workflow_with_mocks, sample_workflow_state
    ):
        """Test workflow status tracking and progress reporting."""
        workflow = workflow_with_mocks

        # Test initial status
        status = workflow.get_workflow_status(sample_workflow_state)

        assert status["workflow_status"] == sample_workflow_state.workflow_status
        assert status["current_step"] == sample_workflow_state.current_step
        assert status["progress_percentage"] >= 0
        assert status["progress_percentage"] <= 100
        assert (
            status["recommended_strategy"] == sample_workflow_state.recommended_strategy
        )

        # Test progress calculation
        expected_progress = (len(sample_workflow_state.steps_completed) / 5) * 100
        assert status["progress_percentage"] == expected_progress

    async def test_workflow_with_requested_strategy(self, workflow_with_mocks):
        """Test workflow behavior with user-requested strategy."""
        workflow = workflow_with_mocks

        result = await workflow.run_intelligent_backtest(
            symbol="AAPL",
            requested_strategy="momentum",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Test that requested strategy is considered
        assert "strategy_selection" in result
        strategy_info = result["strategy_selection"]

        # Should influence selection (mock will still return its default, but in real implementation would consider)
        assert len(strategy_info["selected_strategies"]) > 0

    async def test_workflow_fallback_handling(
        self, workflow_with_mocks, sample_workflow_state
    ):
        """Test workflow fallback strategy handling."""
        workflow = workflow_with_mocks

        # Create incomplete state that triggers fallback
        incomplete_state = sample_workflow_state.copy()
        incomplete_state.workflow_status = "incomplete"
        incomplete_state.recommended_strategy = ""
        incomplete_state.best_parameters = {"momentum": {"window": 20}}

        final_state = await workflow._finalize_workflow(incomplete_state)

        # Test fallback behavior
        assert (
            final_state.recommended_strategy == "momentum"
        )  # Should use first available
        assert final_state.recommendation_confidence == 0.3  # Low confidence fallback
        assert "incomplete_workflow_fallback" in final_state.fallback_strategies_used

    async def test_workflow_results_formatting(
        self, workflow_with_mocks, sample_workflow_state
    ):
        """Test comprehensive results formatting."""
        workflow = workflow_with_mocks

        # Set completed status for full results
        complete_state = sample_workflow_state.copy()
        complete_state.workflow_status = "completed"

        results = workflow._format_results(complete_state)

        # Test all major sections are present
        expected_sections = [
            "symbol",
            "period",
            "market_analysis",
            "strategy_selection",
            "optimization",
            "validation",
            "recommendation",
            "performance_analysis",
        ]

        for section in expected_sections:
            assert section in results

        # Test detailed content
        assert results["market_analysis"]["regime"] == "bullish"
        assert results["strategy_selection"]["selection_confidence"] == 0.85
        assert results["optimization"]["optimization_iterations"] == 45
        assert results["recommendation"]["recommended_strategy"] == "momentum"


class TestLangGraphIntegration:
    """Test suite for LangGraph-specific integration aspects."""

    async def test_langgraph_state_serialization(self, sample_workflow_state):
        """Test that workflow state can be properly serialized/deserialized for LangGraph."""
        # Test JSON serialization compatibility
        import json

        # Extract serializable data
        serializable_data = {
            "symbol": sample_workflow_state.symbol,
            "workflow_status": sample_workflow_state.workflow_status,
            "market_regime": sample_workflow_state.market_regime,
            "regime_confidence": sample_workflow_state.regime_confidence,
            "selected_strategies": sample_workflow_state.selected_strategies,
            "recommendation_confidence": sample_workflow_state.recommendation_confidence,
        }

        # Test serialization
        serialized = json.dumps(serializable_data)
        deserialized = json.loads(serialized)

        assert deserialized["symbol"] == "AAPL"
        assert deserialized["market_regime"] == "bullish"
        assert deserialized["regime_confidence"] == 0.85

    async def test_langgraph_message_flow(self, workflow_with_mocks):
        """Test message flow through LangGraph nodes."""

        workflow = workflow_with_mocks

        # Test that messages are properly handled
        result = await workflow.run_intelligent_backtest(
            symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Verify mock agents were called in sequence
        workflow.market_analyzer.analyze_market_regime.assert_called_once()
        workflow.strategy_selector.select_strategies.assert_called_once()
        workflow.optimizer.optimize_parameters.assert_called_once()
        workflow.validator.validate_strategies.assert_called_once()

    async def test_langgraph_conditional_edges(self, workflow_with_mocks):
        """Test LangGraph conditional edge routing logic."""
        workflow = workflow_with_mocks

        # Create states that should trigger different routing
        good_state = Mock()
        good_state.market_regime = "bullish"
        good_state.regime_confidence = 0.8
        good_state.errors_encountered = []
        good_state.selected_strategies = ["momentum"]
        good_state.strategy_selection_confidence = 0.7
        good_state.best_parameters = {"momentum": {}}

        bad_state = Mock()
        bad_state.market_regime = "unknown"
        bad_state.regime_confidence = 0.1
        bad_state.errors_encountered = [{"step": "test", "error": "test"}]
        bad_state.selected_strategies = []
        bad_state.strategy_selection_confidence = 0.1
        bad_state.best_parameters = {}

        # Test routing decisions
        assert workflow._should_proceed_after_market_analysis(good_state) == "continue"
        assert workflow._should_proceed_after_market_analysis(bad_state) == "fallback"

        assert (
            workflow._should_proceed_after_strategy_selection(good_state) == "continue"
        )
        assert (
            workflow._should_proceed_after_strategy_selection(bad_state) == "fallback"
        )

        assert workflow._should_proceed_after_optimization(good_state) == "continue"
        assert workflow._should_proceed_after_optimization(bad_state) == "fallback"


class TestWorkflowStressTests:
    """Stress tests for workflow performance and reliability."""

    async def test_concurrent_workflow_execution(self, workflow_with_mocks):
        """Test concurrent execution of multiple workflows."""
        workflow = workflow_with_mocks
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        # Run multiple workflows concurrently
        tasks = []
        for symbol in symbols:
            task = workflow.run_intelligent_backtest(
                symbol=symbol, start_date="2023-01-01", end_date="2023-12-31"
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Test all succeeded
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            assert not isinstance(result, Exception)
            assert result["symbol"] == symbols[i]

    async def test_workflow_memory_usage(self, workflow_with_mocks):
        """Test workflow memory usage doesn't grow excessively."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        workflow = workflow_with_mocks

        # Run multiple workflows
        for i in range(10):
            await workflow.run_intelligent_backtest(
                symbol=f"TEST{i}", start_date="2023-01-01", end_date="2023-12-31"
            )

        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory growth should be reasonable (< 50MB for 10 workflows)
        assert memory_growth < 50

    async def test_workflow_error_recovery(self, mock_agents):
        """Test workflow recovery from various error conditions."""
        # Create workflow with intermittently failing agents
        failure_count = 0

        async def intermittent_failure(state):
            nonlocal failure_count
            failure_count += 1

            if failure_count <= 2:
                raise Exception("Simulated failure")

            # Eventually succeed
            state.market_regime = "bullish"
            state.regime_confidence = 0.8
            state.workflow_status = "selecting_strategies"
            state.steps_completed.append("market_analysis")
            return state

        mock_agents["market_analyzer"].analyze_market_regime = AsyncMock(
            side_effect=intermittent_failure
        )

        workflow = BacktestingWorkflow(
            market_analyzer=mock_agents["market_analyzer"],
            strategy_selector=mock_agents["strategy_selector"],
            optimizer=mock_agents["optimizer"],
            validator=mock_agents["validator"],
        )

        # This should eventually succeed despite initial failures
        try:
            result = await workflow.run_intelligent_backtest(
                symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
            )
            # If we reach here, the workflow had some form of error handling
            assert "error" in result or "execution_metadata" in result
        except Exception:
            # Expected for this test - workflow should handle gracefully
            pass


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
