"""The five legacy backtesting tables, bound to this domain's own `MetaData`.

Ports the five tables defined in `maverick_mcp/data/models.py`
(`BacktestResult` -> `mcp_backtest_results`, `BacktestTrade` ->
`mcp_backtest_trades`, `OptimizationResult` -> `mcp_optimization_results`,
`WalkForwardTest` -> `mcp_walk_forward_tests`, `BacktestPortfolio` ->
`mcp_backtest_portfolios`) as plain SQLAlchemy Core `Table` objects --
mirroring `maverick/portfolio/data.py`'s Core-table style rather than the
legacy declarative-ORM-class style, since this domain has no ORM
relationships to preserve. Same table names as the legacy schema so an
existing database carries over. `METADATA` is this module's own -- it never
imports the legacy `maverick_mcp.data.models.Base`.

## FK behavior (pinned by test)

The legacy `mcp_backtest_trades.backtest_id` (and the optimization/
walk-forward FKs) declare `ForeignKey(...)` with no `ondelete=` clause --
the *ORM* cascade (`cascade="all, delete-orphan"` on the relationship)
handles cleanup only when deleting through the ORM session; the physical
constraint itself has no `ON DELETE` action, which is RESTRICT/NO ACTION
under SQLite (with `PRAGMA foreign_keys=ON`, which
`maverick.platform.db.create_engine_from_settings` always enables for
SQLite). Since this module is Core-table-based (no ORM relationships) and
the `store` package doesn't port `delete_backtest`, the FKs here declare no
`ondelete=` either -- matching the legacy DDL exactly rather than upgrading
to `ON DELETE CASCADE`, which the legacy schema never actually had at the
constraint level. `test_deleting_backtest_result_with_trades_is_restricted`
in `tests/backtesting/test_store.py` pins this: deleting a
`mcp_backtest_results` row that still has child trades raises
`IntegrityError`.
"""

from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    Uuid,
)

from maverick.backtesting.store._decimal import now

METADATA = MetaData()

BACKTEST_RESULTS = Table(
    "mcp_backtest_results",
    METADATA,
    Column("backtest_id", Uuid, primary_key=True),
    Column("symbol", String(10), nullable=False),
    Column("strategy_type", String(50), nullable=False),
    Column("backtest_date", DateTime(timezone=True), nullable=False, default=now),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False),
    Column("initial_capital", Numeric(15, 2), default=Decimal("10000.0")),
    Column("fees", Numeric(6, 4), default=Decimal("0.001")),
    Column("slippage", Numeric(6, 4), default=Decimal("0.001")),
    Column("parameters", JSON),
    Column("total_return", Numeric(10, 4)),
    Column("annualized_return", Numeric(10, 4)),
    Column("sharpe_ratio", Numeric(8, 4)),
    Column("sortino_ratio", Numeric(8, 4)),
    Column("calmar_ratio", Numeric(8, 4)),
    Column("max_drawdown", Numeric(8, 4)),
    Column("max_drawdown_duration", Integer),
    Column("volatility", Numeric(8, 4)),
    Column("downside_volatility", Numeric(8, 4)),
    Column("total_trades", Integer, default=0),
    Column("winning_trades", Integer, default=0),
    Column("losing_trades", Integer, default=0),
    Column("win_rate", Numeric(5, 4)),
    Column("profit_factor", Numeric(8, 4)),
    Column("average_win", Numeric(12, 4)),
    Column("average_loss", Numeric(12, 4)),
    Column("largest_win", Numeric(12, 4)),
    Column("largest_loss", Numeric(12, 4)),
    Column("final_portfolio_value", Numeric(15, 2)),
    Column("peak_portfolio_value", Numeric(15, 2)),
    Column("beta", Numeric(8, 4)),
    Column("alpha", Numeric(8, 4)),
    Column("equity_curve", JSON),
    Column("drawdown_series", JSON),
    Column("execution_time_seconds", Numeric(8, 3)),
    Column("data_points", Integer),
    Column("status", String(20), default="completed"),
    Column("error_message", Text),
    Column("notes", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=now,
        onupdate=now,
    ),
    Index("mcp_backtest_results_symbol_idx", "symbol"),
    Index("mcp_backtest_results_strategy_idx", "strategy_type"),
    Index("mcp_backtest_results_date_idx", "backtest_date"),
    Index("mcp_backtest_results_sharpe_idx", "sharpe_ratio"),
    Index("mcp_backtest_results_total_return_idx", "total_return"),
    Index("mcp_backtest_results_symbol_strategy_idx", "symbol", "strategy_type"),
)

BACKTEST_TRADES = Table(
    "mcp_backtest_trades",
    METADATA,
    Column("trade_id", Uuid, primary_key=True),
    Column(
        "backtest_id",
        Uuid,
        ForeignKey("mcp_backtest_results.backtest_id"),
        nullable=False,
    ),
    Column("trade_number", Integer, nullable=False),
    Column("entry_date", Date, nullable=False),
    Column("entry_price", Numeric(12, 4), nullable=False),
    Column("entry_time", DateTime(timezone=True)),
    Column("exit_date", Date),
    Column("exit_price", Numeric(12, 4)),
    Column("exit_time", DateTime(timezone=True)),
    Column("position_size", Numeric(15, 2)),
    Column("direction", String(5), nullable=False),
    Column("pnl", Numeric(12, 4)),
    Column("pnl_percent", Numeric(8, 4)),
    Column("mae", Numeric(8, 4)),
    Column("mfe", Numeric(8, 4)),
    Column("duration_days", Integer),
    Column("duration_hours", Numeric(8, 2)),
    Column("exit_reason", String(50)),
    Column("fees_paid", Numeric(10, 4), default=0),
    Column("slippage_cost", Numeric(10, 4), default=0),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=now,
        onupdate=now,
    ),
    Index("mcp_backtest_trades_backtest_idx", "backtest_id"),
    Index("mcp_backtest_trades_entry_date_idx", "entry_date"),
    Index("mcp_backtest_trades_exit_date_idx", "exit_date"),
    Index("mcp_backtest_trades_pnl_idx", "pnl"),
    Index("mcp_backtest_trades_backtest_entry_idx", "backtest_id", "entry_date"),
)

OPTIMIZATION_RESULTS = Table(
    "mcp_optimization_results",
    METADATA,
    Column("optimization_id", Uuid, primary_key=True),
    Column(
        "backtest_id",
        Uuid,
        ForeignKey("mcp_backtest_results.backtest_id"),
        nullable=False,
    ),
    Column("optimization_date", DateTime(timezone=True), default=now),
    Column("parameter_set", Integer, nullable=False),
    Column("parameters", JSON, nullable=False),
    Column("objective_function", String(50)),
    Column("objective_value", Numeric(12, 6)),
    Column("total_return", Numeric(10, 4)),
    Column("sharpe_ratio", Numeric(8, 4)),
    Column("max_drawdown", Numeric(8, 4)),
    Column("win_rate", Numeric(5, 4)),
    Column("profit_factor", Numeric(8, 4)),
    Column("total_trades", Integer),
    Column("rank", Integer),
    Column("is_statistically_significant", Boolean, default=False),
    Column("p_value", Numeric(8, 6)),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=now,
        onupdate=now,
    ),
    Index("mcp_optimization_results_backtest_idx", "backtest_id"),
    Index("mcp_optimization_results_param_set_idx", "parameter_set"),
    Index("mcp_optimization_results_objective_idx", "objective_value"),
)

WALK_FORWARD_TESTS = Table(
    "mcp_walk_forward_tests",
    METADATA,
    Column("walk_forward_id", Uuid, primary_key=True),
    Column(
        "parent_backtest_id",
        Uuid,
        ForeignKey("mcp_backtest_results.backtest_id"),
        nullable=False,
    ),
    Column("test_date", DateTime(timezone=True), default=now),
    Column("window_size_months", Integer, nullable=False),
    Column("step_size_months", Integer, nullable=False),
    Column("training_start", Date, nullable=False),
    Column("training_end", Date, nullable=False),
    Column("test_period_start", Date, nullable=False),
    Column("test_period_end", Date, nullable=False),
    Column("optimal_parameters", JSON),
    Column("training_performance", Numeric(10, 4)),
    Column("out_of_sample_return", Numeric(10, 4)),
    Column("out_of_sample_sharpe", Numeric(8, 4)),
    Column("out_of_sample_drawdown", Numeric(8, 4)),
    Column("out_of_sample_trades", Integer),
    Column("performance_ratio", Numeric(8, 4)),
    Column("degradation_factor", Numeric(8, 4)),
    Column("is_profitable", Boolean),
    Column("is_statistically_significant", Boolean, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=now,
        onupdate=now,
    ),
    Index("mcp_walk_forward_tests_parent_idx", "parent_backtest_id"),
    Index("mcp_walk_forward_tests_period_idx", "test_period_start"),
    Index("mcp_walk_forward_tests_performance_idx", "out_of_sample_return"),
)

BACKTEST_PORTFOLIOS = Table(
    "mcp_backtest_portfolios",
    METADATA,
    Column("portfolio_backtest_id", Uuid, primary_key=True),
    Column("portfolio_name", String(100), nullable=False),
    Column("description", Text),
    Column("backtest_date", DateTime(timezone=True), default=now),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False),
    Column("symbols", JSON, nullable=False),
    Column("weights", JSON),
    Column("rebalance_frequency", String(20)),
    Column("initial_capital", Numeric(15, 2), default=Decimal("100000.0")),
    Column("max_positions", Integer),
    Column("position_sizing_method", String(50)),
    Column("portfolio_stop_loss", Numeric(6, 4)),
    Column("max_sector_allocation", Numeric(5, 4)),
    Column("correlation_threshold", Numeric(5, 4)),
    Column("total_return", Numeric(10, 4)),
    Column("annualized_return", Numeric(10, 4)),
    Column("sharpe_ratio", Numeric(8, 4)),
    Column("sortino_ratio", Numeric(8, 4)),
    Column("max_drawdown", Numeric(8, 4)),
    Column("volatility", Numeric(8, 4)),
    Column("diversification_ratio", Numeric(8, 4)),
    Column("concentration_index", Numeric(8, 4)),
    Column("turnover_rate", Numeric(8, 4)),
    Column("component_backtest_ids", JSON),
    Column("portfolio_equity_curve", JSON),
    Column("portfolio_weights_history", JSON),
    Column("status", String(20), default="completed"),
    Column("notes", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, default=now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=now,
        onupdate=now,
    ),
    Index("mcp_backtest_portfolios_name_idx", "portfolio_name"),
    Index("mcp_backtest_portfolios_date_idx", "backtest_date"),
    Index("mcp_backtest_portfolios_return_idx", "total_return"),
)
