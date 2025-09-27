"""
Comprehensive tests for backtest persistence layer.

Tests cover:
- PostgreSQL persistence layer with comprehensive database operations
- BacktestResult, BacktestTrade, OptimizationResult, and WalkForwardTest models
- Database CRUD operations with proper error handling
- Performance comparison and ranking functionality
- Backtest result caching and retrieval optimization
- Database constraint validation and data integrity
- Concurrent access and transaction handling
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from maverick_mcp.backtesting.persistence import (
    BacktestPersistenceError,
    BacktestPersistenceManager,
    find_best_strategy_for_symbol,
    get_recent_backtests,
    save_vectorbt_results,
)
from maverick_mcp.data.models import (
    BacktestResult,
    BacktestTrade,
    OptimizationResult,
    WalkForwardTest,
)


class TestBacktestPersistenceManager:
    """Test suite for BacktestPersistenceManager class."""

    @pytest.fixture
    def sample_vectorbt_results(self) -> dict[str, Any]:
        """Create sample VectorBT results for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        equity_curve = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
        drawdown_series = np.minimum(
            0, equity_curve - np.maximum.accumulate(equity_curve)
        )

        return {
            "symbol": "AAPL",
            "strategy": "momentum_crossover",
            "parameters": {
                "fast_window": 10,
                "slow_window": 20,
                "signal_threshold": 0.02,
            },
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000.0,
            "metrics": {
                "total_return": 0.15,
                "annualized_return": 0.18,
                "sharpe_ratio": 1.25,
                "sortino_ratio": 1.45,
                "calmar_ratio": 1.10,
                "max_drawdown": -0.08,
                "max_drawdown_duration": 45,
                "volatility": 0.16,
                "downside_volatility": 0.12,
                "total_trades": 24,
                "winning_trades": 14,
                "losing_trades": 10,
                "win_rate": 0.583,
                "profit_factor": 1.35,
                "average_win": 0.045,
                "average_loss": -0.025,
                "largest_win": 0.12,
                "largest_loss": -0.08,
                "final_value": 11500.0,
                "peak_value": 12100.0,
                "beta": 1.05,
                "alpha": 0.03,
            },
            "equity_curve": equity_curve.tolist(),
            "drawdown_series": drawdown_series.tolist(),
            "trades": [
                {
                    "entry_date": "2023-01-15",
                    "entry_price": 150.0,
                    "entry_time": "2023-01-15T09:30:00",
                    "exit_date": "2023-01-25",
                    "exit_price": 155.0,
                    "exit_time": "2023-01-25T16:00:00",
                    "position_size": 100,
                    "direction": "long",
                    "pnl": 500.0,
                    "pnl_percent": 0.033,
                    "mae": -150.0,
                    "mfe": 600.0,
                    "duration_days": 10,
                    "duration_hours": 6.5,
                    "exit_reason": "take_profit",
                    "fees_paid": 2.0,
                    "slippage_cost": 1.0,
                },
                {
                    "entry_date": "2023-02-01",
                    "entry_price": 160.0,
                    "entry_time": "2023-02-01T10:00:00",
                    "exit_date": "2023-02-10",
                    "exit_price": 156.0,
                    "exit_time": "2023-02-10T15:30:00",
                    "position_size": 100,
                    "direction": "long",
                    "pnl": -400.0,
                    "pnl_percent": -0.025,
                    "mae": -500.0,
                    "mfe": 200.0,
                    "duration_days": 9,
                    "duration_hours": 5.5,
                    "exit_reason": "stop_loss",
                    "fees_paid": 2.0,
                    "slippage_cost": 1.0,
                },
            ],
        }

    @pytest.fixture
    def persistence_manager(self, db_session: Session):
        """Create a persistence manager with test database session."""
        return BacktestPersistenceManager(session=db_session)

    def test_persistence_manager_context_manager(self, db_session: Session):
        """Test persistence manager as context manager."""
        with BacktestPersistenceManager(session=db_session) as manager:
            assert manager.session == db_session
            assert not manager._owns_session

        # Test with auto-session creation (mocked)
        with patch(
            "maverick_mcp.backtesting.persistence.SessionLocal"
        ) as mock_session_local:
            mock_session = Mock(spec=Session)
            mock_session_local.return_value = mock_session

            with BacktestPersistenceManager() as manager:
                assert manager.session == mock_session
                assert manager._owns_session
                mock_session.commit.assert_called_once()
                mock_session.close.assert_called_once()

    def test_save_backtest_result_success(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test successful backtest result saving."""
        backtest_id = persistence_manager.save_backtest_result(
            vectorbt_results=sample_vectorbt_results,
            execution_time=2.5,
            notes="Test backtest run",
        )

        # Test return value
        assert isinstance(backtest_id, str)
        assert UUID(backtest_id)  # Valid UUID

        # Test database record
        result = (
            persistence_manager.session.query(BacktestResult)
            .filter(BacktestResult.backtest_id == UUID(backtest_id))
            .first()
        )

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.strategy_type == "momentum_crossover"
        assert result.total_return == Decimal("0.15")
        assert result.sharpe_ratio == Decimal("1.25")
        assert result.total_trades == 24
        assert result.execution_time_seconds == Decimal("2.5")
        assert result.notes == "Test backtest run"

        # Test trades were saved
        trades = (
            persistence_manager.session.query(BacktestTrade)
            .filter(BacktestTrade.backtest_id == UUID(backtest_id))
            .all()
        )

        assert len(trades) == 2
        assert trades[0].symbol == "AAPL"
        assert trades[0].pnl == Decimal("500.0")
        assert trades[1].pnl == Decimal("-400.0")

    def test_save_backtest_result_validation_error(self, persistence_manager):
        """Test backtest saving with validation errors."""
        # Missing required fields
        invalid_results = {"symbol": "", "strategy": ""}

        with pytest.raises(BacktestPersistenceError) as exc_info:
            persistence_manager.save_backtest_result(invalid_results)

        assert "Symbol and strategy type are required" in str(exc_info.value)

    def test_save_backtest_result_database_error(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test backtest saving with database errors."""
        with patch.object(
            persistence_manager.session, "add", side_effect=SQLAlchemyError("DB Error")
        ):
            with pytest.raises(BacktestPersistenceError) as exc_info:
                persistence_manager.save_backtest_result(sample_vectorbt_results)

            assert "Failed to save backtest" in str(exc_info.value)

    def test_get_backtest_by_id(self, persistence_manager, sample_vectorbt_results):
        """Test retrieval of backtest by ID."""
        # Save a backtest first
        backtest_id = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Retrieve it
        result = persistence_manager.get_backtest_by_id(backtest_id)

        assert result is not None
        assert str(result.backtest_id) == backtest_id
        assert result.symbol == "AAPL"
        assert result.strategy_type == "momentum_crossover"

        # Test non-existent ID
        fake_id = str(uuid4())
        result = persistence_manager.get_backtest_by_id(fake_id)
        assert result is None

        # Test invalid UUID format
        result = persistence_manager.get_backtest_by_id("invalid-uuid")
        assert result is None

    def test_get_backtests_by_symbol(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test retrieval of backtests by symbol."""
        # Save multiple backtests for same symbol
        sample_vectorbt_results["strategy"] = "momentum_v1"
        backtest_id1 = persistence_manager.save_backtest_result(sample_vectorbt_results)

        sample_vectorbt_results["strategy"] = "momentum_v2"
        backtest_id2 = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Save backtest for different symbol
        sample_vectorbt_results["symbol"] = "GOOGL"
        sample_vectorbt_results["strategy"] = "momentum_v1"
        backtest_id3 = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Test retrieval by symbol
        aapl_results = persistence_manager.get_backtests_by_symbol("AAPL")
        assert len(aapl_results) == 2
        assert all(result.symbol == "AAPL" for result in aapl_results)
        assert backtest_id1 != backtest_id2
        assert backtest_id3 not in {backtest_id1, backtest_id2}
        retrieved_ids = {str(result.backtest_id) for result in aapl_results}
        assert {backtest_id1, backtest_id2}.issubset(retrieved_ids)

        # Test with strategy filter
        aapl_v1_results = persistence_manager.get_backtests_by_symbol(
            "AAPL", "momentum_v1"
        )
        assert len(aapl_v1_results) == 1
        assert aapl_v1_results[0].strategy_type == "momentum_v1"

        # Test with limit
        limited_results = persistence_manager.get_backtests_by_symbol("AAPL", limit=1)
        assert len(limited_results) == 1

        # Test non-existent symbol
        empty_results = persistence_manager.get_backtests_by_symbol("NONEXISTENT")
        assert len(empty_results) == 0

    def test_get_best_performing_strategies(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test retrieval of best performing strategies."""
        # Create multiple backtests with different performance
        strategies_performance = [
            (
                "momentum",
                {"sharpe_ratio": 1.5, "total_return": 0.2, "total_trades": 15},
            ),
            (
                "mean_reversion",
                {"sharpe_ratio": 1.8, "total_return": 0.15, "total_trades": 20},
            ),
            (
                "breakout",
                {"sharpe_ratio": 0.8, "total_return": 0.25, "total_trades": 10},
            ),
            (
                "momentum_v2",
                {"sharpe_ratio": 2.0, "total_return": 0.3, "total_trades": 25},
            ),
        ]

        backtest_ids = []
        for strategy, metrics in strategies_performance:
            sample_vectorbt_results["strategy"] = strategy
            sample_vectorbt_results["metrics"].update(metrics)
            backtest_id = persistence_manager.save_backtest_result(
                sample_vectorbt_results
            )
            backtest_ids.append(backtest_id)

        # Test best by Sharpe ratio (default)
        best_sharpe = persistence_manager.get_best_performing_strategies(
            "sharpe_ratio", limit=3
        )
        assert len(best_sharpe) == 3
        assert best_sharpe[0].strategy_type == "momentum_v2"  # Highest Sharpe
        assert best_sharpe[1].strategy_type == "mean_reversion"  # Second highest
        assert best_sharpe[0].sharpe_ratio > best_sharpe[1].sharpe_ratio

        # Test best by total return
        best_return = persistence_manager.get_best_performing_strategies(
            "total_return", limit=2
        )
        assert len(best_return) == 2
        assert best_return[0].strategy_type == "momentum_v2"  # Highest return

        # Test minimum trades filter
        high_volume = persistence_manager.get_best_performing_strategies(
            "sharpe_ratio", min_trades=20
        )
        assert len(high_volume) == 2  # Only momentum_v2 and mean_reversion
        assert all(result.total_trades >= 20 for result in high_volume)

    def test_compare_strategies(self, persistence_manager, sample_vectorbt_results):
        """Test strategy comparison functionality."""
        # Create backtests to compare
        strategies = ["momentum", "mean_reversion", "breakout"]
        backtest_ids = []

        for i, strategy in enumerate(strategies):
            sample_vectorbt_results["strategy"] = strategy
            sample_vectorbt_results["metrics"]["sharpe_ratio"] = 1.0 + i * 0.5
            sample_vectorbt_results["metrics"]["total_return"] = 0.1 + i * 0.05
            sample_vectorbt_results["metrics"]["max_drawdown"] = -0.05 - i * 0.02
            backtest_id = persistence_manager.save_backtest_result(
                sample_vectorbt_results
            )
            backtest_ids.append(backtest_id)

        # Test comparison
        comparison = persistence_manager.compare_strategies(backtest_ids)

        assert "backtests" in comparison
        assert "rankings" in comparison
        assert "summary" in comparison
        assert len(comparison["backtests"]) == 3

        # Test rankings
        assert "sharpe_ratio" in comparison["rankings"]
        sharpe_rankings = comparison["rankings"]["sharpe_ratio"]
        assert len(sharpe_rankings) == 3
        assert sharpe_rankings[0]["rank"] == 1  # Best rank
        assert sharpe_rankings[0]["value"] > sharpe_rankings[1]["value"]

        # Test max_drawdown ranking (lower is better)
        assert "max_drawdown" in comparison["rankings"]
        dd_rankings = comparison["rankings"]["max_drawdown"]
        assert (
            dd_rankings[0]["value"] > dd_rankings[-1]["value"]
        )  # Less negative is better

        # Test summary
        summary = comparison["summary"]
        assert summary["total_backtests"] == 3
        assert "date_range" in summary

    def test_save_optimization_results(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test saving parameter optimization results."""
        # Save parent backtest first
        backtest_id = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Create optimization results
        optimization_results = [
            {
                "parameters": {"window": 10, "threshold": 0.01},
                "objective_value": 1.2,
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "win_rate": 0.6,
                "profit_factor": 1.3,
                "total_trades": 20,
                "rank": 1,
            },
            {
                "parameters": {"window": 20, "threshold": 0.02},
                "objective_value": 1.5,
                "total_return": 0.18,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.06,
                "win_rate": 0.65,
                "profit_factor": 1.4,
                "total_trades": 18,
                "rank": 2,
            },
        ]

        # Save optimization results
        count = persistence_manager.save_optimization_results(
            backtest_id=backtest_id,
            optimization_results=optimization_results,
            objective_function="sharpe_ratio",
        )

        assert count == 2

        # Verify saved results
        opt_results = (
            persistence_manager.session.query(OptimizationResult)
            .filter(OptimizationResult.backtest_id == UUID(backtest_id))
            .all()
        )

        assert len(opt_results) == 2
        assert opt_results[0].objective_function == "sharpe_ratio"
        assert opt_results[0].parameters == {"window": 10, "threshold": 0.01}
        assert opt_results[0].objective_value == Decimal("1.2")

    def test_save_walk_forward_test(self, persistence_manager, sample_vectorbt_results):
        """Test saving walk-forward validation results."""
        # Save parent backtest first
        backtest_id = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Create walk-forward test data
        walk_forward_data = {
            "window_size_months": 6,
            "step_size_months": 1,
            "training_start": "2023-01-01",
            "training_end": "2023-06-30",
            "test_period_start": "2023-07-01",
            "test_period_end": "2023-07-31",
            "optimal_parameters": {"window": 15, "threshold": 0.015},
            "training_performance": 1.3,
            "out_of_sample_return": 0.12,
            "out_of_sample_sharpe": 1.1,
            "out_of_sample_drawdown": -0.05,
            "out_of_sample_trades": 8,
            "performance_ratio": 0.85,
            "degradation_factor": 0.15,
            "is_profitable": True,
            "is_statistically_significant": True,
        }

        # Save walk-forward test
        wf_id = persistence_manager.save_walk_forward_test(
            backtest_id, walk_forward_data
        )

        assert isinstance(wf_id, str)
        assert UUID(wf_id)

        # Verify saved result
        wf_test = (
            persistence_manager.session.query(WalkForwardTest)
            .filter(WalkForwardTest.walk_forward_id == UUID(wf_id))
            .first()
        )

        assert wf_test is not None
        assert wf_test.parent_backtest_id == UUID(backtest_id)
        assert wf_test.window_size_months == 6
        assert wf_test.out_of_sample_sharpe == Decimal("1.1")
        assert wf_test.is_profitable is True

    def test_get_backtest_performance_summary(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test performance summary generation."""
        # Create backtests with different dates and performance
        base_date = datetime.utcnow()

        # Recent backtests (within 30 days)
        for i in range(3):
            sample_vectorbt_results["strategy"] = f"momentum_v{i + 1}"
            sample_vectorbt_results["metrics"]["total_return"] = 0.1 + i * 0.05
            sample_vectorbt_results["metrics"]["sharpe_ratio"] = 1.0 + i * 0.3
            sample_vectorbt_results["metrics"]["win_rate"] = 0.5 + i * 0.1

            with patch(
                "maverick_mcp.data.models.BacktestResult.backtest_date",
                base_date - timedelta(days=i * 10),
            ):
                persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Old backtest (outside 30 days)
        sample_vectorbt_results["strategy"] = "old_strategy"
        with patch(
            "maverick_mcp.data.models.BacktestResult.backtest_date",
            base_date - timedelta(days=45),
        ):
            persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Get summary
        summary = persistence_manager.get_backtest_performance_summary(days_back=30)

        assert "period" in summary
        assert summary["total_backtests"] == 3  # Only recent ones
        assert "performance_metrics" in summary

        metrics = summary["performance_metrics"]
        assert "average_return" in metrics
        assert "best_return" in metrics
        assert "worst_return" in metrics
        assert "average_sharpe" in metrics

        # Test strategy and symbol breakdowns
        assert "strategy_breakdown" in summary
        assert len(summary["strategy_breakdown"]) == 3
        assert "symbol_breakdown" in summary
        assert "AAPL" in summary["symbol_breakdown"]

    def test_delete_backtest(self, persistence_manager, sample_vectorbt_results):
        """Test backtest deletion with cascading."""
        # Save backtest with trades
        backtest_id = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Verify it exists
        result = persistence_manager.get_backtest_by_id(backtest_id)
        assert result is not None

        trades = (
            persistence_manager.session.query(BacktestTrade)
            .filter(BacktestTrade.backtest_id == UUID(backtest_id))
            .all()
        )
        assert len(trades) > 0

        # Delete backtest
        deleted = persistence_manager.delete_backtest(backtest_id)
        assert deleted is True

        # Verify deletion
        result = persistence_manager.get_backtest_by_id(backtest_id)
        assert result is None

        # Test non-existent deletion
        fake_id = str(uuid4())
        deleted = persistence_manager.delete_backtest(fake_id)
        assert deleted is False

    def test_safe_decimal_conversion(self):
        """Test safe decimal conversion utility."""
        from maverick_mcp.backtesting.persistence import BacktestPersistenceManager

        # Test valid conversions
        assert BacktestPersistenceManager._safe_decimal(123) == Decimal("123")
        assert BacktestPersistenceManager._safe_decimal(123.45) == Decimal("123.45")
        assert BacktestPersistenceManager._safe_decimal("456.78") == Decimal("456.78")
        assert BacktestPersistenceManager._safe_decimal(Decimal("789.01")) == Decimal(
            "789.01"
        )

        # Test None and invalid values
        assert BacktestPersistenceManager._safe_decimal(None) is None
        assert BacktestPersistenceManager._safe_decimal("invalid") is None
        assert BacktestPersistenceManager._safe_decimal([1, 2, 3]) is None


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_save_vectorbt_results_function(
        self, db_session: Session, sample_vectorbt_results
    ):
        """Test save_vectorbt_results convenience function."""
        with patch(
            "maverick_mcp.backtesting.persistence.get_persistence_manager"
        ) as mock_factory:
            mock_manager = Mock(spec=BacktestPersistenceManager)
            mock_manager.save_backtest_result.return_value = "test-uuid-123"
            mock_manager.__enter__ = Mock(return_value=mock_manager)
            mock_manager.__exit__ = Mock(return_value=None)
            mock_factory.return_value = mock_manager

            result = save_vectorbt_results(
                vectorbt_results=sample_vectorbt_results,
                execution_time=2.5,
                notes="Test run",
            )

            assert result == "test-uuid-123"
            mock_manager.save_backtest_result.assert_called_once_with(
                sample_vectorbt_results, 2.5, "Test run"
            )

    def test_get_recent_backtests_function(self, db_session: Session):
        """Test get_recent_backtests convenience function."""
        with patch(
            "maverick_mcp.backtesting.persistence.get_persistence_manager"
        ) as mock_factory:
            mock_manager = Mock(spec=BacktestPersistenceManager)
            mock_session = Mock(spec=Session)
            mock_query = Mock()

            mock_manager.session = mock_session
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.all.return_value = ["result1", "result2"]

            mock_manager.__enter__ = Mock(return_value=mock_manager)
            mock_manager.__exit__ = Mock(return_value=None)
            mock_factory.return_value = mock_manager

            results = get_recent_backtests("AAPL", days=7)

            assert results == ["result1", "result2"]
            mock_session.query.assert_called_once_with(BacktestResult)

    def test_find_best_strategy_for_symbol_function(self, db_session: Session):
        """Test find_best_strategy_for_symbol convenience function."""
        with patch(
            "maverick_mcp.backtesting.persistence.get_persistence_manager"
        ) as mock_factory:
            mock_manager = Mock(spec=BacktestPersistenceManager)
            mock_best_result = Mock(spec=BacktestResult)

            mock_manager.get_best_performing_strategies.return_value = [
                mock_best_result
            ]
            mock_manager.get_backtests_by_symbol.return_value = [mock_best_result]
            mock_manager.__enter__ = Mock(return_value=mock_manager)
            mock_manager.__exit__ = Mock(return_value=None)
            mock_factory.return_value = mock_manager

            result = find_best_strategy_for_symbol("AAPL", "sharpe_ratio")

            assert result == mock_best_result
            mock_manager.get_backtests_by_symbol.assert_called_once_with(
                "AAPL", limit=1000
            )


class TestPersistenceStressTests:
    """Stress tests for persistence layer performance and reliability."""

    def test_bulk_insert_performance(
        self, persistence_manager, sample_vectorbt_results, benchmark_timer
    ):
        """Test bulk insert performance with many backtests."""
        backtest_count = 50

        with benchmark_timer() as timer:
            for i in range(backtest_count):
                sample_vectorbt_results["symbol"] = f"STOCK{i:03d}"
                sample_vectorbt_results["strategy"] = (
                    f"strategy_{i % 5}"  # 5 different strategies
                )
                persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Should complete within reasonable time
        assert timer.elapsed < 30.0  # < 30 seconds for 50 backtests

        # Verify all were saved
        all_results = persistence_manager.session.query(BacktestResult).count()
        assert all_results == backtest_count

    def test_concurrent_access_handling(
        self, db_session: Session, sample_vectorbt_results
    ):
        """Test handling of concurrent database access."""
        import queue
        import threading

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def save_backtest(thread_id):
            try:
                # Each thread gets its own session
                with BacktestPersistenceManager() as manager:
                    modified_results = sample_vectorbt_results.copy()
                    modified_results["symbol"] = f"THREAD{thread_id}"
                    backtest_id = manager.save_backtest_result(modified_results)
                    results_queue.put(backtest_id)
            except Exception as e:
                error_queue.put(f"Thread {thread_id}: {e}")

        # Create multiple threads
        threads = []
        thread_count = 5

        for i in range(thread_count):
            thread = threading.Thread(target=save_backtest, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout per thread

        # Check results
        assert error_queue.empty(), f"Errors occurred: {list(error_queue.queue)}"
        assert results_queue.qsize() == thread_count

        # Verify all backtests were saved with unique IDs
        saved_ids = []
        while not results_queue.empty():
            saved_ids.append(results_queue.get())

        assert len(saved_ids) == thread_count
        assert len(set(saved_ids)) == thread_count  # All unique

    def test_large_result_handling(self, persistence_manager, sample_vectorbt_results):
        """Test handling of large backtest results."""
        # Create large equity curve and drawdown series (1 year of minute data)
        large_data_size = 365 * 24 * 60  # ~525k data points

        sample_vectorbt_results["equity_curve"] = list(range(large_data_size))
        sample_vectorbt_results["drawdown_series"] = [
            -i / 1000 for i in range(large_data_size)
        ]

        # Also add many trades
        sample_vectorbt_results["trades"] = []
        for i in range(1000):  # 1000 trades
            trade = {
                "entry_date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "entry_price": 100 + (i % 100),
                "exit_date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "exit_price": 101 + (i % 100),
                "position_size": 100,
                "direction": "long",
                "pnl": i % 100 - 50,
                "pnl_percent": (i % 100 - 50) / 1000,
                "duration_days": i % 30 + 1,
                "exit_reason": "time_exit",
            }
            sample_vectorbt_results["trades"].append(trade)

        # Should handle large data without issues
        backtest_id = persistence_manager.save_backtest_result(sample_vectorbt_results)

        assert backtest_id is not None

        # Verify retrieval works
        result = persistence_manager.get_backtest_by_id(backtest_id)
        assert result is not None
        assert result.data_points == large_data_size

        # Verify trades were saved
        trades = (
            persistence_manager.session.query(BacktestTrade)
            .filter(BacktestTrade.backtest_id == UUID(backtest_id))
            .count()
        )
        assert trades == 1000

    def test_database_constraint_validation(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test database constraint validation and error handling."""
        # Save first backtest
        backtest_id1 = persistence_manager.save_backtest_result(sample_vectorbt_results)

        # Try to save with same UUID (should be prevented by unique constraint)
        with patch("uuid.uuid4", return_value=UUID(backtest_id1)):
            # This should handle the constraint violation gracefully
            try:
                backtest_id2 = persistence_manager.save_backtest_result(
                    sample_vectorbt_results
                )
                # If it succeeds, it should have generated a different UUID
                assert backtest_id2 != backtest_id1
            except BacktestPersistenceError:
                # Or it should raise a proper persistence error
                pass

    def test_memory_usage_with_large_datasets(
        self, persistence_manager, sample_vectorbt_results
    ):
        """Test memory usage doesn't grow excessively with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and save multiple large backtests
        for i in range(10):
            large_results = sample_vectorbt_results.copy()
            large_results["symbol"] = f"LARGE{i}"
            large_results["equity_curve"] = list(range(10000))  # 10k data points each
            large_results["drawdown_series"] = [-j / 1000 for j in range(10000)]

            persistence_manager.save_backtest_result(large_results)

        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory growth should be reasonable (< 100MB for 10 large backtests)
        assert memory_growth < 100


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "-x"])
