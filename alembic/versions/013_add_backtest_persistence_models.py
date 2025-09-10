"""Add backtest persistence models

Revision ID: 013_add_backtest_persistence_models
Revises: fix_database_integrity_issues
Create Date: 2025-01-16 12:00:00.000000

This migration adds comprehensive backtesting persistence models:
1. BacktestResult - Main backtest results with comprehensive metrics
2. BacktestTrade - Individual trade records from backtests
3. OptimizationResult - Parameter optimization results
4. WalkForwardTest - Walk-forward validation test results
5. BacktestPortfolio - Portfolio-level backtests with multiple symbols

All tables include proper indexes for common query patterns and foreign key
relationships for data integrity.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "013_add_backtest_persistence_models"
down_revision = "fix_database_integrity_issues"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create BacktestResult table
    op.create_table(
        "mcp_backtest_results",
        sa.Column("backtest_id", sa.Uuid(), nullable=False, primary_key=True),
        # Basic metadata
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("strategy_type", sa.String(length=50), nullable=False),
        sa.Column("backtest_date", sa.DateTime(timezone=True), nullable=False),
        # Date range and setup
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column(
            "initial_capital",
            sa.Numeric(precision=15, scale=2),
            server_default="10000.0",
        ),
        # Trading costs
        sa.Column("fees", sa.Numeric(precision=6, scale=4), server_default="0.001"),
        sa.Column("slippage", sa.Numeric(precision=6, scale=4), server_default="0.001"),
        # Strategy parameters
        sa.Column("parameters", sa.JSON()),
        # Performance metrics
        sa.Column("total_return", sa.Numeric(precision=10, scale=4)),
        sa.Column("annualized_return", sa.Numeric(precision=10, scale=4)),
        sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("sortino_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("calmar_ratio", sa.Numeric(precision=8, scale=4)),
        # Risk metrics
        sa.Column("max_drawdown", sa.Numeric(precision=8, scale=4)),
        sa.Column("max_drawdown_duration", sa.Integer()),
        sa.Column("volatility", sa.Numeric(precision=8, scale=4)),
        sa.Column("downside_volatility", sa.Numeric(precision=8, scale=4)),
        # Trade statistics
        sa.Column("total_trades", sa.Integer(), server_default="0"),
        sa.Column("winning_trades", sa.Integer(), server_default="0"),
        sa.Column("losing_trades", sa.Integer(), server_default="0"),
        sa.Column("win_rate", sa.Numeric(precision=5, scale=4)),
        # P&L statistics
        sa.Column("profit_factor", sa.Numeric(precision=8, scale=4)),
        sa.Column("average_win", sa.Numeric(precision=12, scale=4)),
        sa.Column("average_loss", sa.Numeric(precision=12, scale=4)),
        sa.Column("largest_win", sa.Numeric(precision=12, scale=4)),
        sa.Column("largest_loss", sa.Numeric(precision=12, scale=4)),
        # Portfolio values
        sa.Column("final_portfolio_value", sa.Numeric(precision=15, scale=2)),
        sa.Column("peak_portfolio_value", sa.Numeric(precision=15, scale=2)),
        # Market analysis
        sa.Column("beta", sa.Numeric(precision=8, scale=4)),
        sa.Column("alpha", sa.Numeric(precision=8, scale=4)),
        # Time series data
        sa.Column("equity_curve", sa.JSON()),
        sa.Column("drawdown_series", sa.JSON()),
        # Execution metadata
        sa.Column("execution_time_seconds", sa.Numeric(precision=8, scale=3)),
        sa.Column("data_points", sa.Integer()),
        # Status and notes
        sa.Column("status", sa.String(length=20), server_default="completed"),
        sa.Column("error_message", sa.Text()),
        sa.Column("notes", sa.Text()),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes for BacktestResult
    op.create_index(
        "mcp_backtest_results_symbol_idx", "mcp_backtest_results", ["symbol"]
    )
    op.create_index(
        "mcp_backtest_results_strategy_idx", "mcp_backtest_results", ["strategy_type"]
    )
    op.create_index(
        "mcp_backtest_results_date_idx", "mcp_backtest_results", ["backtest_date"]
    )
    op.create_index(
        "mcp_backtest_results_sharpe_idx", "mcp_backtest_results", ["sharpe_ratio"]
    )
    op.create_index(
        "mcp_backtest_results_total_return_idx",
        "mcp_backtest_results",
        ["total_return"],
    )
    op.create_index(
        "mcp_backtest_results_symbol_strategy_idx",
        "mcp_backtest_results",
        ["symbol", "strategy_type"],
    )

    # Create BacktestTrade table
    op.create_table(
        "mcp_backtest_trades",
        sa.Column("trade_id", sa.Uuid(), nullable=False, primary_key=True),
        sa.Column("backtest_id", sa.Uuid(), nullable=False),
        # Trade identification
        sa.Column("trade_number", sa.Integer(), nullable=False),
        # Entry details
        sa.Column("entry_date", sa.Date(), nullable=False),
        sa.Column("entry_price", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True)),
        # Exit details
        sa.Column("exit_date", sa.Date()),
        sa.Column("exit_price", sa.Numeric(precision=12, scale=4)),
        sa.Column("exit_time", sa.DateTime(timezone=True)),
        # Position details
        sa.Column("position_size", sa.Numeric(precision=15, scale=2)),
        sa.Column("direction", sa.String(length=5), nullable=False),
        # P&L
        sa.Column("pnl", sa.Numeric(precision=12, scale=4)),
        sa.Column("pnl_percent", sa.Numeric(precision=8, scale=4)),
        # Risk metrics
        sa.Column("mae", sa.Numeric(precision=8, scale=4)),  # Maximum Adverse Excursion
        sa.Column(
            "mfe", sa.Numeric(precision=8, scale=4)
        ),  # Maximum Favorable Excursion
        # Duration
        sa.Column("duration_days", sa.Integer()),
        sa.Column("duration_hours", sa.Numeric(precision=8, scale=2)),
        # Exit details
        sa.Column("exit_reason", sa.String(length=50)),
        sa.Column("fees_paid", sa.Numeric(precision=10, scale=4), server_default="0"),
        sa.Column(
            "slippage_cost", sa.Numeric(precision=10, scale=4), server_default="0"
        ),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ["backtest_id"], ["mcp_backtest_results.backtest_id"], ondelete="CASCADE"
        ),
    )

    # Create indexes for BacktestTrade
    op.create_index(
        "mcp_backtest_trades_backtest_idx", "mcp_backtest_trades", ["backtest_id"]
    )
    op.create_index(
        "mcp_backtest_trades_entry_date_idx", "mcp_backtest_trades", ["entry_date"]
    )
    op.create_index(
        "mcp_backtest_trades_exit_date_idx", "mcp_backtest_trades", ["exit_date"]
    )
    op.create_index("mcp_backtest_trades_pnl_idx", "mcp_backtest_trades", ["pnl"])
    op.create_index(
        "mcp_backtest_trades_backtest_entry_idx",
        "mcp_backtest_trades",
        ["backtest_id", "entry_date"],
    )

    # Create OptimizationResult table
    op.create_table(
        "mcp_optimization_results",
        sa.Column("optimization_id", sa.Uuid(), nullable=False, primary_key=True),
        sa.Column("backtest_id", sa.Uuid(), nullable=False),
        # Optimization metadata
        sa.Column("optimization_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("parameter_set", sa.Integer(), nullable=False),
        # Parameters and results
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("objective_function", sa.String(length=50)),
        sa.Column("objective_value", sa.Numeric(precision=12, scale=6)),
        # Key metrics
        sa.Column("total_return", sa.Numeric(precision=10, scale=4)),
        sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("max_drawdown", sa.Numeric(precision=8, scale=4)),
        sa.Column("win_rate", sa.Numeric(precision=5, scale=4)),
        sa.Column("profit_factor", sa.Numeric(precision=8, scale=4)),
        sa.Column("total_trades", sa.Integer()),
        # Ranking
        sa.Column("rank", sa.Integer()),
        # Statistical significance
        sa.Column("is_statistically_significant", sa.Boolean(), server_default="false"),
        sa.Column("p_value", sa.Numeric(precision=8, scale=6)),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ["backtest_id"], ["mcp_backtest_results.backtest_id"], ondelete="CASCADE"
        ),
    )

    # Create indexes for OptimizationResult
    op.create_index(
        "mcp_optimization_results_backtest_idx",
        "mcp_optimization_results",
        ["backtest_id"],
    )
    op.create_index(
        "mcp_optimization_results_param_set_idx",
        "mcp_optimization_results",
        ["parameter_set"],
    )
    op.create_index(
        "mcp_optimization_results_objective_idx",
        "mcp_optimization_results",
        ["objective_value"],
    )

    # Create WalkForwardTest table
    op.create_table(
        "mcp_walk_forward_tests",
        sa.Column("walk_forward_id", sa.Uuid(), nullable=False, primary_key=True),
        sa.Column("parent_backtest_id", sa.Uuid(), nullable=False),
        # Test configuration
        sa.Column("test_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_size_months", sa.Integer(), nullable=False),
        sa.Column("step_size_months", sa.Integer(), nullable=False),
        # Time periods
        sa.Column("training_start", sa.Date(), nullable=False),
        sa.Column("training_end", sa.Date(), nullable=False),
        sa.Column("test_period_start", sa.Date(), nullable=False),
        sa.Column("test_period_end", sa.Date(), nullable=False),
        # Training results
        sa.Column("optimal_parameters", sa.JSON()),
        sa.Column("training_performance", sa.Numeric(precision=10, scale=4)),
        # Out-of-sample results
        sa.Column("out_of_sample_return", sa.Numeric(precision=10, scale=4)),
        sa.Column("out_of_sample_sharpe", sa.Numeric(precision=8, scale=4)),
        sa.Column("out_of_sample_drawdown", sa.Numeric(precision=8, scale=4)),
        sa.Column("out_of_sample_trades", sa.Integer()),
        # Performance analysis
        sa.Column("performance_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("degradation_factor", sa.Numeric(precision=8, scale=4)),
        # Validation
        sa.Column("is_profitable", sa.Boolean()),
        sa.Column("is_statistically_significant", sa.Boolean(), server_default="false"),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ["parent_backtest_id"],
            ["mcp_backtest_results.backtest_id"],
            ondelete="CASCADE",
        ),
    )

    # Create indexes for WalkForwardTest
    op.create_index(
        "mcp_walk_forward_tests_parent_idx",
        "mcp_walk_forward_tests",
        ["parent_backtest_id"],
    )
    op.create_index(
        "mcp_walk_forward_tests_period_idx",
        "mcp_walk_forward_tests",
        ["test_period_start"],
    )
    op.create_index(
        "mcp_walk_forward_tests_performance_idx",
        "mcp_walk_forward_tests",
        ["out_of_sample_return"],
    )

    # Create BacktestPortfolio table
    op.create_table(
        "mcp_backtest_portfolios",
        sa.Column("portfolio_backtest_id", sa.Uuid(), nullable=False, primary_key=True),
        # Portfolio identification
        sa.Column("portfolio_name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text()),
        # Test metadata
        sa.Column("backtest_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        # Portfolio composition
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("weights", sa.JSON()),
        sa.Column("rebalance_frequency", sa.String(length=20)),
        # Portfolio parameters
        sa.Column(
            "initial_capital",
            sa.Numeric(precision=15, scale=2),
            server_default="100000.0",
        ),
        sa.Column("max_positions", sa.Integer()),
        sa.Column("position_sizing_method", sa.String(length=50)),
        # Risk management
        sa.Column("portfolio_stop_loss", sa.Numeric(precision=6, scale=4)),
        sa.Column("max_sector_allocation", sa.Numeric(precision=5, scale=4)),
        sa.Column("correlation_threshold", sa.Numeric(precision=5, scale=4)),
        # Performance metrics
        sa.Column("total_return", sa.Numeric(precision=10, scale=4)),
        sa.Column("annualized_return", sa.Numeric(precision=10, scale=4)),
        sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("sortino_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("max_drawdown", sa.Numeric(precision=8, scale=4)),
        sa.Column("volatility", sa.Numeric(precision=8, scale=4)),
        # Portfolio-specific metrics
        sa.Column("diversification_ratio", sa.Numeric(precision=8, scale=4)),
        sa.Column("concentration_index", sa.Numeric(precision=8, scale=4)),
        sa.Column("turnover_rate", sa.Numeric(precision=8, scale=4)),
        # References and time series
        sa.Column("component_backtest_ids", sa.JSON()),
        sa.Column("portfolio_equity_curve", sa.JSON()),
        sa.Column("portfolio_weights_history", sa.JSON()),
        # Status
        sa.Column("status", sa.String(length=20), server_default="completed"),
        sa.Column("notes", sa.Text()),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes for BacktestPortfolio
    op.create_index(
        "mcp_backtest_portfolios_name_idx",
        "mcp_backtest_portfolios",
        ["portfolio_name"],
    )
    op.create_index(
        "mcp_backtest_portfolios_date_idx", "mcp_backtest_portfolios", ["backtest_date"]
    )
    op.create_index(
        "mcp_backtest_portfolios_return_idx",
        "mcp_backtest_portfolios",
        ["total_return"],
    )


def downgrade() -> None:
    # Drop tables in reverse order (due to foreign key constraints)
    op.drop_table("mcp_backtest_portfolios")
    op.drop_table("mcp_walk_forward_tests")
    op.drop_table("mcp_optimization_results")
    op.drop_table("mcp_backtest_trades")
    op.drop_table("mcp_backtest_results")
