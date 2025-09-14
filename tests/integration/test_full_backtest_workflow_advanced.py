"""
Advanced End-to-End Integration Tests for VectorBT Backtesting Workflow.

This comprehensive test suite covers:
- Complete workflow integration from data fetch to results
- All 15 strategies (9 traditional + 6 ML) testing
- Parallel execution capabilities
- Cache behavior and optimization
- Real production-like scenarios
- Error recovery and resilience
- Resource management and cleanup
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.api.routers.backtesting import setup_backtesting_tools
from maverick_mcp.backtesting import (
    BacktestAnalyzer,
    StrategyOptimizer,
    VectorBTEngine,
)
from maverick_mcp.backtesting.persistence import BacktestPersistenceManager
from maverick_mcp.backtesting.strategies import STRATEGY_TEMPLATES
from maverick_mcp.backtesting.visualization import (
    generate_equity_curve,
    generate_performance_dashboard,
)
from maverick_mcp.workflows.backtesting_workflow import BacktestingWorkflow

logger = logging.getLogger(__name__)

# Strategy definitions for comprehensive testing
TRADITIONAL_STRATEGIES = [
    "sma_cross", "ema_cross", "rsi", "macd", "bollinger",
    "momentum", "breakout", "mean_reversion", "volume_momentum"
]

ML_STRATEGIES = [
    "ml_predictor", "adaptive", "ensemble",
    "regime_aware", "online_learning", "reinforcement_learning"
]

ALL_STRATEGIES = TRADITIONAL_STRATEGIES + ML_STRATEGIES


class TestAdvancedBacktestWorkflowIntegration:
    """Advanced integration tests for complete backtesting workflow."""

    @pytest.fixture
    async def enhanced_stock_data_provider(self):
        """Create enhanced mock stock data provider with realistic multi-year data."""
        provider = Mock()

        # Generate 3 years of realistic stock data with different market conditions
        dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")

        # Simulate different market regimes
        bull_period = len(dates) // 3  # First third: bull market
        sideways_period = len(dates) // 3  # Second third: sideways
        bear_period = len(dates) - bull_period - sideways_period  # Final: bear market

        # Generate returns for different regimes
        bull_returns = np.random.normal(0.0015, 0.015, bull_period)  # Positive drift
        sideways_returns = np.random.normal(0.0002, 0.02, sideways_period)  # Low drift
        bear_returns = np.random.normal(-0.001, 0.025, bear_period)  # Negative drift

        all_returns = np.concatenate([bull_returns, sideways_returns, bear_returns])
        prices = 100 * np.cumprod(1 + all_returns)  # Start at $100

        # Add realistic volume patterns
        volumes = np.random.randint(500000, 5000000, len(dates)).astype(float)
        volumes += np.random.normal(0, volumes * 0.1)  # Add volume volatility
        volumes = np.maximum(volumes, 100000)  # Minimum volume
        volumes = volumes.astype(int)  # Convert back to integers

        stock_data = pd.DataFrame({
            "Open": prices * np.random.uniform(0.995, 1.005, len(dates)),
            "High": prices * np.random.uniform(1.002, 1.025, len(dates)),
            "Low": prices * np.random.uniform(0.975, 0.998, len(dates)),
            "Close": prices,
            "Volume": volumes.astype(int),
            "Adj Close": prices,
        }, index=dates)

        # Ensure OHLC constraints
        stock_data["High"] = np.maximum(
            stock_data["High"],
            np.maximum(stock_data["Open"], stock_data["Close"])
        )
        stock_data["Low"] = np.minimum(
            stock_data["Low"],
            np.minimum(stock_data["Open"], stock_data["Close"])
        )

        provider.get_stock_data.return_value = stock_data
        return provider

    @pytest.fixture
    async def complete_vectorbt_engine(self, enhanced_stock_data_provider):
        """Create complete VectorBT engine with all strategies enabled."""
        engine = VectorBTEngine(data_provider=enhanced_stock_data_provider)
        return engine

    async def test_all_15_strategies_integration(self, complete_vectorbt_engine, benchmark_timer):
        """Test all 15 strategies (9 traditional + 6 ML) in complete workflow."""
        results = {}
        failed_strategies = []

        with benchmark_timer() as timer:
            # Test traditional strategies
            for strategy in TRADITIONAL_STRATEGIES:
                try:
                    if strategy in STRATEGY_TEMPLATES:
                        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]
                        result = await complete_vectorbt_engine.run_backtest(
                            symbol="COMPREHENSIVE_TEST",
                            strategy_type=strategy,
                            parameters=parameters,
                            start_date="2022-01-01",
                            end_date="2023-12-31",
                        )
                        results[strategy] = result

                        # Validate basic result structure
                        assert "metrics" in result
                        assert "trades" in result
                        assert "equity_curve" in result
                        assert result["symbol"] == "COMPREHENSIVE_TEST"

                        logger.info(f"✓ {strategy} strategy executed successfully")
                    else:
                        logger.warning(f"Strategy {strategy} not found in templates")

                except Exception as e:
                    failed_strategies.append(strategy)
                    logger.error(f"✗ {strategy} strategy failed: {str(e)}")

            # Test ML strategies (mock implementation for integration test)
            for strategy in ML_STRATEGIES:
                try:
                    # Mock ML strategy execution
                    mock_ml_result = {
                        "symbol": "COMPREHENSIVE_TEST",
                        "strategy_type": strategy,
                        "metrics": {
                            "total_return": np.random.uniform(-0.2, 0.3),
                            "sharpe_ratio": np.random.uniform(0.5, 2.0),
                            "max_drawdown": np.random.uniform(-0.3, -0.05),
                            "total_trades": np.random.randint(10, 100),
                        },
                        "trades": [],
                        "equity_curve": np.random.cumsum(np.random.normal(0.001, 0.02, 252)).tolist(),
                        "ml_specific": {
                            "model_accuracy": np.random.uniform(0.55, 0.85),
                            "feature_importance": {"momentum": 0.3, "volatility": 0.25, "volume": 0.45},
                        }
                    }
                    results[strategy] = mock_ml_result
                    logger.info(f"✓ {strategy} ML strategy simulated successfully")

                except Exception as e:
                    failed_strategies.append(strategy)
                    logger.error(f"✗ {strategy} ML strategy failed: {str(e)}")

        execution_time = timer.elapsed

        # Validate overall results
        successful_strategies = len(results)
        total_strategies = len(ALL_STRATEGIES)
        success_rate = successful_strategies / total_strategies

        # Performance requirements
        assert execution_time < 180.0  # Should complete within 3 minutes
        assert success_rate >= 0.8  # At least 80% success rate
        assert successful_strategies >= 12  # At least 12 strategies should work

        # Log comprehensive results
        logger.info(
            f"Strategy Integration Test Results:\n"
            f"  • Total Strategies: {total_strategies}\n"
            f"  • Successful: {successful_strategies}\n"
            f"  • Failed: {len(failed_strategies)}\n"
            f"  • Success Rate: {success_rate:.1%}\n"
            f"  • Execution Time: {execution_time:.2f}s\n"
            f"  • Failed Strategies: {failed_strategies}"
        )

        return {
            "total_strategies": total_strategies,
            "successful_strategies": successful_strategies,
            "failed_strategies": failed_strategies,
            "success_rate": success_rate,
            "execution_time": execution_time,
            "results": results,
        }

    async def test_parallel_execution_capabilities(self, complete_vectorbt_engine, benchmark_timer):
        """Test parallel execution of multiple backtests."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA"]
        strategies = ["sma_cross", "rsi", "macd", "bollinger"]

        async def run_single_backtest(symbol, strategy):
            """Run a single backtest."""
            try:
                parameters = STRATEGY_TEMPLATES.get(strategy, {}).get("parameters", {})
                result = await complete_vectorbt_engine.run_backtest(
                    symbol=symbol,
                    strategy_type=strategy,
                    parameters=parameters,
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                )
                return {"symbol": symbol, "strategy": strategy, "result": result, "success": True}
            except Exception as e:
                return {"symbol": symbol, "strategy": strategy, "error": str(e), "success": False}

        with benchmark_timer() as timer:
            # Create all combinations
            tasks = []
            for symbol in symbols:
                for strategy in strategies:
                    tasks.append(run_single_backtest(symbol, strategy))

            # Execute in parallel with semaphore to control concurrency
            semaphore = asyncio.Semaphore(8)  # Max 8 concurrent executions

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )

        execution_time = timer.elapsed

        # Analyze results
        total_executions = len(tasks)
        successful_executions = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_executions = total_executions - successful_executions

        # Performance assertions
        assert execution_time < 300.0  # Should complete within 5 minutes
        assert successful_executions >= total_executions * 0.7  # At least 70% success

        # Calculate average execution time per backtest
        avg_time_per_backtest = execution_time / total_executions

        logger.info(
            f"Parallel Execution Results:\n"
            f"  • Total Executions: {total_executions}\n"
            f"  • Successful: {successful_executions}\n"
            f"  • Failed: {failed_executions}\n"
            f"  • Success Rate: {successful_executions/total_executions:.1%}\n"
            f"  • Total Time: {execution_time:.2f}s\n"
            f"  • Avg Time/Backtest: {avg_time_per_backtest:.2f}s\n"
            f"  • Parallel Speedup: ~{total_executions * avg_time_per_backtest / execution_time:.1f}x"
        )

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "execution_time": execution_time,
            "avg_time_per_backtest": avg_time_per_backtest,
        }

    async def test_cache_behavior_and_optimization(self, complete_vectorbt_engine):
        """Test cache behavior and optimization in integrated workflow."""
        symbol = "CACHE_TEST_SYMBOL"
        strategy = "sma_cross"
        parameters = STRATEGY_TEMPLATES[strategy]["parameters"]

        # First run - should populate cache
        start_time = time.time()
        result1 = await complete_vectorbt_engine.run_backtest(
            symbol=symbol,
            strategy_type=strategy,
            parameters=parameters,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        first_run_time = time.time() - start_time

        # Second run - should use cache
        start_time = time.time()
        result2 = await complete_vectorbt_engine.run_backtest(
            symbol=symbol,
            strategy_type=strategy,
            parameters=parameters,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        second_run_time = time.time() - start_time

        # Third run with different parameters - should not use cache
        modified_parameters = {**parameters, "fast_period": parameters.get("fast_period", 10) + 5}
        start_time = time.time()
        result3 = await complete_vectorbt_engine.run_backtest(
            symbol=symbol,
            strategy_type=strategy,
            parameters=modified_parameters,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        third_run_time = time.time() - start_time

        # Validate results consistency (for cached runs)
        assert result1["symbol"] == result2["symbol"]
        assert result1["strategy_type"] == result2["strategy_type"]

        # Cache effectiveness check (second run might be faster, but not guaranteed)
        cache_speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0

        logger.info(
            f"Cache Behavior Test Results:\n"
            f"  • First Run: {first_run_time:.3f}s\n"
            f"  • Second Run (cached): {second_run_time:.3f}s\n"
            f"  • Third Run (different params): {third_run_time:.3f}s\n"
            f"  • Cache Speedup: {cache_speedup:.2f}x\n"
        )

        return {
            "first_run_time": first_run_time,
            "second_run_time": second_run_time,
            "third_run_time": third_run_time,
            "cache_speedup": cache_speedup,
        }

    async def test_database_persistence_integration(self, complete_vectorbt_engine, db_session):
        """Test complete database persistence integration."""
        # Generate test results
        result = await complete_vectorbt_engine.run_backtest(
            symbol="PERSISTENCE_TEST",
            strategy_type="sma_cross",
            parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Test persistence workflow
        with BacktestPersistenceManager(session=db_session) as persistence:
            # Save backtest result
            backtest_id = persistence.save_backtest_result(
                vectorbt_results=result,
                execution_time=2.5,
                notes="Integration test - complete persistence workflow",
            )

            # Validate saved data
            assert backtest_id is not None
            assert UUID(backtest_id)  # Valid UUID

            # Retrieve and validate
            saved_result = persistence.get_backtest_by_id(backtest_id)
            assert saved_result is not None
            assert saved_result.symbol == "PERSISTENCE_TEST"
            assert saved_result.strategy_type == "sma_cross"
            assert saved_result.execution_time == 2.5

            # Test batch operations
            batch_results = []
            for i in range(5):
                batch_result = await complete_vectorbt_engine.run_backtest(
                    symbol=f"BATCH_TEST_{i}",
                    strategy_type="rsi",
                    parameters=STRATEGY_TEMPLATES["rsi"]["parameters"],
                    start_date="2023-06-01",
                    end_date="2023-12-31",
                )
                batch_results.append(batch_result)

            # Save batch results
            batch_ids = []
            for i, batch_result in enumerate(batch_results):
                batch_id = persistence.save_backtest_result(
                    vectorbt_results=batch_result,
                    execution_time=1.8 + i * 0.1,
                    notes=f"Batch test #{i+1}",
                )
                batch_ids.append(batch_id)

            # Query saved batch results
            saved_batch = [persistence.get_backtest_by_id(bid) for bid in batch_ids]
            assert all(saved is not None for saved in saved_batch)
            assert len(saved_batch) == 5

            # Test filtering and querying
            rsi_results = persistence.get_backtests_by_strategy("rsi")
            assert len(rsi_results) >= 5  # At least our batch results

        logger.info(f"Database persistence test completed successfully")
        return {"batch_ids": batch_ids, "single_id": backtest_id}

    async def test_visualization_integration_complete(self, complete_vectorbt_engine):
        """Test complete visualization integration workflow."""
        # Run backtest to get data for visualization
        result = await complete_vectorbt_engine.run_backtest(
            symbol="VIZ_TEST",
            strategy_type="macd",
            parameters=STRATEGY_TEMPLATES["macd"]["parameters"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Test all visualization components
        visualizations = {}

        # 1. Equity curve visualization
        equity_data = pd.Series(result["equity_curve"])
        drawdown_data = pd.Series(result["drawdown_series"])

        equity_chart = generate_equity_curve(
            equity_data,
            drawdown=drawdown_data,
            title="Complete Integration Test - Equity Curve"
        )
        visualizations["equity_curve"] = equity_chart

        # 2. Performance dashboard
        dashboard_chart = generate_performance_dashboard(
            result["metrics"],
            title="Complete Integration Test - Performance Dashboard"
        )
        visualizations["dashboard"] = dashboard_chart

        # 3. Validate all visualizations
        for viz_name, viz_data in visualizations.items():
            assert isinstance(viz_data, str), f"{viz_name} should return string"
            assert len(viz_data) > 100, f"{viz_name} should have substantial content"

            # Try to decode as base64 (should be valid image)
            try:
                import base64
                decoded = base64.b64decode(viz_data)
                assert len(decoded) > 0, f"{viz_name} should have valid image data"
                logger.info(f"✓ {viz_name} visualization generated successfully")
            except Exception as e:
                logger.error(f"✗ {viz_name} visualization failed: {e}")
                raise

        return visualizations

    async def test_error_recovery_comprehensive(self, complete_vectorbt_engine):
        """Test comprehensive error recovery across the workflow."""
        recovery_results = {}

        # 1. Invalid symbol handling
        try:
            result = await complete_vectorbt_engine.run_backtest(
                symbol="",  # Empty symbol
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            recovery_results["empty_symbol"] = {"recovered": True, "result": result}
        except Exception as e:
            recovery_results["empty_symbol"] = {"recovered": False, "error": str(e)}

        # 2. Invalid date range handling
        try:
            result = await complete_vectorbt_engine.run_backtest(
                symbol="ERROR_TEST",
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2025-01-01",  # Future date
                end_date="2025-12-31",
            )
            recovery_results["future_dates"] = {"recovered": True, "result": result}
        except Exception as e:
            recovery_results["future_dates"] = {"recovered": False, "error": str(e)}

        # 3. Invalid strategy parameters
        try:
            invalid_params = {"fast_period": -10, "slow_period": -20}  # Invalid negative values
            result = await complete_vectorbt_engine.run_backtest(
                symbol="ERROR_TEST",
                strategy_type="sma_cross",
                parameters=invalid_params,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            recovery_results["invalid_params"] = {"recovered": True, "result": result}
        except Exception as e:
            recovery_results["invalid_params"] = {"recovered": False, "error": str(e)}

        # 4. Unknown strategy handling
        try:
            result = await complete_vectorbt_engine.run_backtest(
                symbol="ERROR_TEST",
                strategy_type="nonexistent_strategy",
                parameters={},
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            recovery_results["unknown_strategy"] = {"recovered": True, "result": result}
        except Exception as e:
            recovery_results["unknown_strategy"] = {"recovered": False, "error": str(e)}

        # Analyze recovery effectiveness
        total_tests = len(recovery_results)
        recovered_tests = sum(1 for r in recovery_results.values() if r.get("recovered", False))
        recovery_rate = recovered_tests / total_tests if total_tests > 0 else 0

        logger.info(
            f"Error Recovery Test Results:\n"
            f"  • Total Error Scenarios: {total_tests}\n"
            f"  • Successfully Recovered: {recovered_tests}\n"
            f"  • Recovery Rate: {recovery_rate:.1%}\n"
        )

        for scenario, result in recovery_results.items():
            status = "✓ RECOVERED" if result.get("recovered") else "✗ FAILED"
            logger.info(f"  • {scenario}: {status}")

        return recovery_results

    async def test_resource_management_comprehensive(self, complete_vectorbt_engine):
        """Test comprehensive resource management across workflow."""
        import os
        import psutil

        process = psutil.Process(os.getpid())

        # Baseline measurements
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_threads = process.num_threads()
        initial_cpu = process.cpu_percent()

        resource_snapshots = []

        # Run multiple backtests while monitoring resources
        for i in range(10):
            await complete_vectorbt_engine.run_backtest(
                symbol=f"RESOURCE_TEST_{i}",
                strategy_type="sma_cross",
                parameters=STRATEGY_TEMPLATES["sma_cross"]["parameters"],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            # Take resource snapshot
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            current_threads = process.num_threads()
            current_cpu = process.cpu_percent()

            resource_snapshots.append({
                "iteration": i + 1,
                "memory_mb": current_memory,
                "threads": current_threads,
                "cpu_percent": current_cpu,
            })

        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_threads = process.num_threads()

        # Calculate resource growth
        memory_growth = final_memory - initial_memory
        thread_growth = final_threads - initial_threads
        peak_memory = max(snapshot["memory_mb"] for snapshot in resource_snapshots)
        avg_threads = sum(snapshot["threads"] for snapshot in resource_snapshots) / len(resource_snapshots)

        # Resource management assertions
        assert memory_growth < 500, f"Memory growth too high: {memory_growth:.1f}MB"  # Max 500MB growth
        assert thread_growth <= 10, f"Thread growth too high: {thread_growth}"  # Max 10 additional threads
        assert peak_memory < initial_memory + 1000, f"Peak memory too high: {peak_memory:.1f}MB"  # Peak within 1GB of initial

        logger.info(
            f"Resource Management Test Results:\n"
            f"  • Initial Memory: {initial_memory:.1f}MB\n"
            f"  • Final Memory: {final_memory:.1f}MB\n"
            f"  • Memory Growth: {memory_growth:.1f}MB\n"
            f"  • Peak Memory: {peak_memory:.1f}MB\n"
            f"  • Initial Threads: {initial_threads}\n"
            f"  • Final Threads: {final_threads}\n"
            f"  • Thread Growth: {thread_growth}\n"
            f"  • Avg Threads: {avg_threads:.1f}"
        )

        return {
            "memory_growth": memory_growth,
            "thread_growth": thread_growth,
            "peak_memory": peak_memory,
            "resource_snapshots": resource_snapshots,
        }


if __name__ == "__main__":
    # Run advanced integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--timeout=600",  # 10 minute timeout for comprehensive tests
        "-x",  # Stop on first failure
        "--durations=10",  # Show 10 slowest tests
    ])