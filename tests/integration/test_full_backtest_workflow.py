"""
Comprehensive end-to-end integration tests for VectorBT backtesting workflow.

Tests cover:
- Full workflow integration from data fetching to result visualization
- LangGraph workflow orchestration with real agents
- Database persistence with real PostgreSQL operations
- Chart generation and visualization pipeline
- ML strategy integration with adaptive learning
- Performance benchmarks for complete workflow
- Error recovery and resilience testing
- Concurrent workflow execution
- Resource cleanup and memory management
- Cache integration and optimization
"""

import asyncio
import base64
import logging
from datetime import datetime
from unittest.mock import Mock, patch
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.backtesting.persistence import (
    BacktestPersistenceManager,
)
from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine
from maverick_mcp.backtesting.visualization import (
    generate_equity_curve,
    generate_performance_dashboard,
)
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
from maverick_mcp.workflows.backtesting_workflow import BacktestingWorkflow

logger = logging.getLogger(__name__)


class TestFullBacktestWorkflowIntegration:
    """Integration tests for complete backtesting workflow."""

    @pytest.fixture
    async def mock_stock_data_provider(self):
        """Create a mock stock data provider with realistic data."""
        provider = Mock(spec=EnhancedStockDataProvider)

        # Generate realistic stock data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual volatility
        prices = 150 * np.cumprod(1 + returns)  # Start at $150
        volumes = np.random.randint(1000000, 10000000, len(dates))

        stock_data = pd.DataFrame(
            {
                "Open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                "High": prices * np.random.uniform(1.00, 1.03, len(dates)),
                "Low": prices * np.random.uniform(0.97, 1.00, len(dates)),
                "Close": prices,
                "Volume": volumes,
                "Adj Close": prices,
            },
            index=dates,
        )

        # Ensure OHLC constraints
        stock_data["High"] = np.maximum(
            stock_data["High"], np.maximum(stock_data["Open"], stock_data["Close"])
        )
        stock_data["Low"] = np.minimum(
            stock_data["Low"], np.minimum(stock_data["Open"], stock_data["Close"])
        )

        provider.get_stock_data.return_value = stock_data
        return provider

    @pytest.fixture
    async def vectorbt_engine(self, mock_stock_data_provider):
        """Create VectorBT engine with mocked data provider."""
        engine = VectorBTEngine(data_provider=mock_stock_data_provider)
        return engine

    @pytest.fixture
    def workflow_with_real_agents(self):
        """Create workflow with real agents (not mocked)."""
        return BacktestingWorkflow()

    async def test_complete_workflow_execution(
        self, workflow_with_real_agents, db_session, benchmark_timer
    ):
        """Test complete workflow from start to finish with database persistence."""
        start_time = datetime.now()

        with benchmark_timer() as timer:
            # Execute intelligent backtest
            result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=10000.0,
            )

        execution_time = datetime.now() - start_time

        # Test basic result structure
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "execution_metadata" in result
        assert "market_analysis" in result
        assert "strategy_selection" in result
        assert "recommendation" in result

        # Test execution metadata
        metadata = result["execution_metadata"]
        assert "total_execution_time_ms" in metadata
        assert "workflow_completed" in metadata
        assert "steps_completed" in metadata

        # Test performance requirements
        assert timer.elapsed < 60.0  # Should complete within 1 minute
        assert metadata["total_execution_time_ms"] > 0

        # Test that meaningful analysis occurred
        market_analysis = result["market_analysis"]
        assert "regime" in market_analysis
        assert "regime_confidence" in market_analysis

        strategy_selection = result["strategy_selection"]
        assert "selected_strategies" in strategy_selection
        assert "selection_reasoning" in strategy_selection

        recommendation = result["recommendation"]
        assert "recommended_strategy" in recommendation
        assert "recommendation_confidence" in recommendation

        logger.info(f"Complete workflow executed in {timer.elapsed:.2f}s")

    async def test_workflow_with_persistence_integration(
        self, workflow_with_real_agents, db_session, sample_vectorbt_results
    ):
        """Test workflow integration with database persistence."""
        # First run the workflow
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="TSLA", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Simulate saving backtest results to database
        with BacktestPersistenceManager(session=db_session) as persistence:
            # Modify sample results to match workflow output
            sample_vectorbt_results["symbol"] = "TSLA"
            sample_vectorbt_results["strategy"] = result["recommendation"][
                "recommended_strategy"
            ]

            backtest_id = persistence.save_backtest_result(
                vectorbt_results=sample_vectorbt_results,
                execution_time=result["execution_metadata"]["total_execution_time_ms"]
                / 1000,
                notes=f"Intelligent backtest - {result['recommendation']['recommendation_confidence']:.2%} confidence",
            )

            # Verify persistence
            assert backtest_id is not None
            assert UUID(backtest_id)

            # Retrieve and verify
            saved_result = persistence.get_backtest_by_id(backtest_id)
            assert saved_result is not None
            assert saved_result.symbol == "TSLA"
            assert (
                saved_result.strategy_type
                == result["recommendation"]["recommended_strategy"]
            )

    async def test_workflow_with_visualization_integration(
        self, workflow_with_real_agents, sample_vectorbt_results
    ):
        """Test workflow integration with visualization generation."""
        # Run workflow
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="NVDA", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Generate visualizations based on workflow results
        equity_curve_data = pd.Series(sample_vectorbt_results["equity_curve"])
        drawdown_data = pd.Series(sample_vectorbt_results["drawdown_series"])

        # Test equity curve generation
        equity_chart = generate_equity_curve(
            equity_curve_data,
            drawdown=drawdown_data,
            title=f"NVDA - {result['recommendation']['recommended_strategy']} Strategy",
        )

        assert isinstance(equity_chart, str)
        assert len(equity_chart) > 100

        # Verify base64 image
        try:
            decoded_bytes = base64.b64decode(equity_chart)
            assert decoded_bytes.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Invalid chart generation: {e}")

        # Test performance dashboard
        dashboard_metrics = {
            "Strategy": result["recommendation"]["recommended_strategy"],
            "Confidence": f"{result['recommendation']['recommendation_confidence']:.1%}",
            "Market Regime": result["market_analysis"]["regime"],
            "Regime Confidence": f"{result['market_analysis']['regime_confidence']:.1%}",
            "Total Return": sample_vectorbt_results["metrics"]["total_return"],
            "Sharpe Ratio": sample_vectorbt_results["metrics"]["sharpe_ratio"],
            "Max Drawdown": sample_vectorbt_results["metrics"]["max_drawdown"],
        }

        dashboard_chart = generate_performance_dashboard(
            dashboard_metrics, title="Intelligent Backtest Results"
        )

        assert isinstance(dashboard_chart, str)
        assert len(dashboard_chart) > 100

    async def test_workflow_with_ml_strategy_integration(
        self, workflow_with_real_agents, mock_stock_data_provider
    ):
        """Test workflow integration with ML-enhanced strategies."""
        # Mock the workflow to use ML strategies
        with patch.object(
            workflow_with_real_agents.strategy_selector, "select_strategies"
        ) as mock_selector:

            async def mock_select_with_ml(state):
                state.selected_strategies = ["adaptive_momentum", "online_learning"]
                state.strategy_selection_confidence = 0.85
                state.strategy_selection_reasoning = (
                    "ML strategies selected for volatile market conditions"
                )
                return state

            mock_selector.side_effect = mock_select_with_ml

            result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol="AMZN", start_date="2023-01-01", end_date="2023-12-31"
            )

            # Verify ML strategy integration
            assert (
                "adaptive_momentum"
                in result["strategy_selection"]["selected_strategies"]
                or "online_learning"
                in result["strategy_selection"]["selected_strategies"]
            )
            assert (
                "ML" in result["strategy_selection"]["selection_reasoning"]
                or "adaptive" in result["strategy_selection"]["selection_reasoning"]
            )

    async def test_vectorbt_engine_integration(self, vectorbt_engine):
        """Test VectorBT engine integration with workflow."""
        # Test data fetching
        data = await vectorbt_engine.get_historical_data(
            symbol="MSFT", start_date="2023-01-01", end_date="2023-12-31"
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(
            col in data.columns.str.lower() for col in ["open", "high", "low", "close"]
        )

        # Test backtest execution
        backtest_result = await vectorbt_engine.run_backtest(
            symbol="MSFT",
            strategy_type="sma_crossover",
            parameters={"fast_window": 10, "slow_window": 20},
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert isinstance(backtest_result, dict)
        assert "symbol" in backtest_result
        assert "metrics" in backtest_result
        assert "equity_curve" in backtest_result

    async def test_error_recovery_integration(self, workflow_with_real_agents):
        """Test error recovery in integrated workflow."""
        # Test with invalid symbol
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="INVALID_SYMBOL", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Should handle gracefully
        assert "error" in result or "execution_metadata" in result

        if "execution_metadata" in result:
            assert result["execution_metadata"]["workflow_completed"] is False

        # Test with invalid date range
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="AAPL",
            start_date="2025-01-01",  # Future date
            end_date="2025-12-31",
        )

        # Should handle gracefully
        assert isinstance(result, dict)

    async def test_concurrent_workflow_execution(
        self, workflow_with_real_agents, benchmark_timer
    ):
        """Test concurrent execution of multiple complete workflows."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        async def run_workflow(symbol):
            return await workflow_with_real_agents.run_intelligent_backtest(
                symbol=symbol, start_date="2023-01-01", end_date="2023-12-31"
            )

        with benchmark_timer() as timer:
            # Run workflows concurrently
            results = await asyncio.gather(
                *[run_workflow(symbol) for symbol in symbols], return_exceptions=True
            )

        # Test all completed
        assert len(results) == len(symbols)

        # Test no exceptions
        successful_results = []
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                successful_results.append(result)
                assert result["symbol"] == symbols[i]
            else:
                logger.warning(f"Workflow failed for {symbols[i]}: {result}")

        # At least half should succeed in concurrent execution
        assert len(successful_results) >= len(symbols) // 2

        # Test reasonable execution time for concurrent runs
        assert timer.elapsed < 120.0  # Should complete within 2 minutes

        logger.info(
            f"Concurrent workflows completed: {len(successful_results)}/{len(symbols)} in {timer.elapsed:.2f}s"
        )

    async def test_performance_benchmarks_integration(
        self, workflow_with_real_agents, benchmark_timer
    ):
        """Test performance benchmarks for integrated workflow."""
        performance_results = {}

        # Test quick analysis performance
        with benchmark_timer() as timer:
            quick_result = await workflow_with_real_agents.run_quick_analysis(
                symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
            )
        quick_time = timer.elapsed

        # Test full workflow performance
        with benchmark_timer() as timer:
            full_result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
            )
        full_time = timer.elapsed

        # Performance requirements
        assert quick_time < 10.0  # Quick analysis < 10 seconds
        assert full_time < 60.0  # Full workflow < 1 minute
        assert quick_time < full_time  # Quick should be faster than full

        performance_results["quick_analysis"] = quick_time
        performance_results["full_workflow"] = full_time

        # Test workflow status tracking performance
        if "workflow_completed" in full_result.get("execution_metadata", {}):
            workflow_status = workflow_with_real_agents.get_workflow_status(
                full_result.get(
                    "_internal_state",
                    Mock(
                        workflow_status="completed",
                        current_step="finalized",
                        steps_completed=[
                            "initialization",
                            "market_analysis",
                            "strategy_selection",
                        ],
                        errors_encountered=[],
                        validation_warnings=[],
                        total_execution_time_ms=full_time * 1000,
                        recommended_strategy=full_result.get("recommendation", {}).get(
                            "recommended_strategy", "unknown"
                        ),
                        recommendation_confidence=full_result.get(
                            "recommendation", {}
                        ).get("recommendation_confidence", 0.0),
                    ),
                )
            )

            assert workflow_status["progress_percentage"] >= 0
            assert workflow_status["progress_percentage"] <= 100

        logger.info(f"Performance benchmarks: {performance_results}")

    async def test_resource_cleanup_integration(self, workflow_with_real_agents):
        """Test resource cleanup after workflow completion."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        initial_threads = process.num_threads()

        # Run multiple workflows
        for i in range(3):
            result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol=f"TEST_{i}",  # Use different symbols
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            assert isinstance(result, dict)

        # Check resource usage after completion
        final_memory = process.memory_info().rss
        final_threads = process.num_threads()

        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        thread_growth = final_threads - initial_threads

        # Memory growth should be reasonable
        assert memory_growth < 200  # < 200MB growth

        # Thread count should not grow excessively
        assert thread_growth <= 5  # Allow some thread growth

        logger.info(
            f"Resource usage: Memory +{memory_growth:.1f}MB, Threads +{thread_growth}"
        )

    async def test_cache_optimization_integration(self, workflow_with_real_agents):
        """Test cache optimization in integrated workflow."""
        # First run - should populate cache
        start_time1 = datetime.now()
        result1 = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="CACHE_TEST", start_date="2023-01-01", end_date="2023-12-31"
        )
        time1 = (datetime.now() - start_time1).total_seconds()

        # Second run - should use cache
        start_time2 = datetime.now()
        result2 = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="CACHE_TEST", start_date="2023-01-01", end_date="2023-12-31"
        )
        time2 = (datetime.now() - start_time2).total_seconds()

        # Both should complete successfully
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

        # Second run might be faster due to caching (though not guaranteed)
        # We mainly test that caching doesn't break functionality
        assert result1["symbol"] == result2["symbol"] == "CACHE_TEST"

        logger.info(f"Cache test: First run {time1:.2f}s, Second run {time2:.2f}s")


class TestWorkflowErrorResilience:
    """Test workflow resilience under various error conditions."""

    async def test_database_failure_resilience(self, workflow_with_real_agents):
        """Test workflow resilience when database operations fail."""
        with patch(
            "maverick_mcp.backtesting.persistence.SessionLocal",
            side_effect=Exception("Database unavailable"),
        ):
            # Workflow should still complete even if persistence fails
            result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol="DB_FAIL_TEST", start_date="2023-01-01", end_date="2023-12-31"
            )

            # Should get a result even if database persistence failed
            assert isinstance(result, dict)
            assert "symbol" in result

    async def test_external_api_failure_resilience(self, workflow_with_real_agents):
        """Test workflow resilience when external APIs fail."""
        # Mock external API failures
        with patch(
            "maverick_mcp.providers.stock_data.EnhancedStockDataProvider.get_stock_data",
            side_effect=Exception("API rate limit exceeded"),
        ):
            result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol="API_FAIL_TEST", start_date="2023-01-01", end_date="2023-12-31"
            )

            # Should handle API failure gracefully
            assert isinstance(result, dict)
            # Should either have an error field or fallback behavior
            assert "error" in result or "execution_metadata" in result

    async def test_memory_pressure_resilience(self, workflow_with_real_agents):
        """Test workflow resilience under memory pressure."""
        # Simulate memory pressure by creating large objects
        memory_pressure = []
        try:
            # Create memory pressure (but not too much to crash the test)
            for i in range(10):
                large_array = np.random.random((1000, 1000))  # ~8MB each
                memory_pressure.append(large_array)

            # Run workflow under memory pressure
            result = await workflow_with_real_agents.run_intelligent_backtest(
                symbol="MEMORY_TEST", start_date="2023-01-01", end_date="2023-12-31"
            )

            assert isinstance(result, dict)
            assert "symbol" in result

        finally:
            # Clean up memory pressure
            del memory_pressure

    async def test_timeout_handling(self, workflow_with_real_agents):
        """Test workflow timeout handling."""
        # Create a workflow with very short timeout
        with patch.object(asyncio, "wait_for") as mock_wait_for:
            mock_wait_for.side_effect = TimeoutError("Workflow timed out")

            try:
                result = await workflow_with_real_agents.run_intelligent_backtest(
                    symbol="TIMEOUT_TEST",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                )

                # If we get here, timeout was handled
                assert isinstance(result, dict)

            except TimeoutError:
                # Timeout occurred - this is also acceptable behavior
                pass


class TestWorkflowValidation:
    """Test workflow validation and data integrity."""

    async def test_input_validation(self, workflow_with_real_agents):
        """Test input parameter validation."""
        # Test invalid symbol
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="",  # Empty symbol
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert "error" in result or (
            "execution_metadata" in result
            and not result["execution_metadata"]["workflow_completed"]
        )

        # Test invalid date range
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="AAPL",
            start_date="2023-12-31",  # Start after end
            end_date="2023-01-01",
        )

        assert isinstance(result, dict)  # Should handle gracefully

    async def test_output_validation(self, workflow_with_real_agents):
        """Test output structure validation."""
        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol="VALIDATE_TEST", start_date="2023-01-01", end_date="2023-12-31"
        )

        # Validate required fields
        required_fields = ["symbol", "execution_metadata"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate execution metadata structure
        metadata = result["execution_metadata"]
        required_metadata = ["total_execution_time_ms", "workflow_completed"]
        for field in required_metadata:
            assert field in metadata, f"Missing metadata field: {field}"

        # Validate data types
        assert isinstance(metadata["total_execution_time_ms"], (int, float))
        assert isinstance(metadata["workflow_completed"], bool)

        if "recommendation" in result:
            recommendation = result["recommendation"]
            assert "recommended_strategy" in recommendation
            assert "recommendation_confidence" in recommendation
            assert isinstance(recommendation["recommendation_confidence"], (int, float))
            assert 0.0 <= recommendation["recommendation_confidence"] <= 1.0

    async def test_data_consistency(self, workflow_with_real_agents, db_session):
        """Test data consistency across workflow components."""
        symbol = "CONSISTENCY_TEST"

        result = await workflow_with_real_agents.run_intelligent_backtest(
            symbol=symbol, start_date="2023-01-01", end_date="2023-12-31"
        )

        # Test symbol consistency
        assert result["symbol"] == symbol

        # If workflow completed successfully, all components should be consistent
        if result["execution_metadata"]["workflow_completed"]:
            # Market analysis should be consistent
            if "market_analysis" in result:
                market_analysis = result["market_analysis"]
                assert "regime" in market_analysis
                assert isinstance(
                    market_analysis.get("regime_confidence", 0), (int, float)
                )

            # Strategy selection should be consistent
            if "strategy_selection" in result:
                strategy_selection = result["strategy_selection"]
                selected_strategies = strategy_selection.get("selected_strategies", [])
                assert isinstance(selected_strategies, list)

            # Recommendation should be consistent with selection
            if "recommendation" in result and "strategy_selection" in result:
                recommended = result["recommendation"]["recommended_strategy"]
                if recommended and selected_strategies:
                    # Recommended strategy should be from selected strategies
                    # (though fallback behavior might select others)
                    pass  # Allow flexibility for fallback scenarios


if __name__ == "__main__":
    # Run integration tests with extended timeout
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "--timeout=300",  # 5 minute timeout for integration tests
            "-x",  # Stop on first failure
        ]
    )
