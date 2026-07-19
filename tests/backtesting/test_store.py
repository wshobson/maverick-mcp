"""Tests for maverick.backtesting.store.

Persistence layer on the platform db seam. Uses a tmp SQLite database per
test (via `platform.db.create_engine_from_settings` + `ensure_schema`),
matching the pattern in `tests/portfolio/test_data.py` and
`tests/market_data/test_data.py`.
"""

import uuid
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from maverick.backtesting.store import (
    BACKTEST_RESULTS,
    BACKTEST_TRADES,
    METADATA,
    BacktestPortfolioInput,
    OptimizationResultInput,
    WalkForwardTestInput,
    get_backtest_by_id,
    get_backtests_by_symbol,
    get_optimization_results,
    get_trades_for_backtest,
    get_walk_forward_tests,
    list_backtest_portfolios,
    save_backtest_portfolio,
    save_backtest_result,
    save_optimization_results,
    save_walk_forward_test,
)
from maverick.backtesting.types import BacktestMetrics, BacktestResult, TradeRecord
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)


@pytest.fixture
def factory(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/backtesting.db", use_pooling=True
    )
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return sessionmaker(bind=engine)


def _metrics(**overrides) -> BacktestMetrics:
    fields = {
        "total_return": 0.2534,
        "annual_return": 0.1892,
        "sharpe_ratio": 1.2345,
        "sortino_ratio": 1.5678,
        "calmar_ratio": 0.9876,
        "max_drawdown": -0.1234,
        "win_rate": 0.5556,
        "profit_factor": 1.8765,
        "expectancy": 12.34,
        "total_trades": 20,
        "winning_trades": 11,
        "losing_trades": 9,
        "avg_win": 150.25,
        "avg_loss": -80.5,
        "best_trade": 320.75,
        "worst_trade": -150.0,
        "avg_duration": 5.5,
        "kelly_criterion": 0.15,
        "recovery_factor": 2.1,
        "risk_reward_ratio": 1.9,
    }
    fields.update(overrides)
    return BacktestMetrics(**fields)


def _trade(**overrides) -> TradeRecord:
    fields = {
        "entry_date": "2026-01-05",
        "exit_date": "2026-01-10",
        "entry_price": 100.0,
        "exit_price": 105.5,
        "size": 10.0,
        "pnl": 55.0,
        "return": 0.055,
        "duration": "5 days",
    }
    fields.update(overrides)
    return TradeRecord(**fields)


def _result(**overrides) -> BacktestResult:
    fields = {
        "symbol": "aapl",
        "strategy": "sma_cross",
        "parameters": {"fast_period": 10, "slow_period": 50},
        "metrics": _metrics(),
        "trades": [_trade()],
        "equity_curve": {"2026-01-01": 10000.0, "2026-01-31": 10250.0},
        "drawdown_series": {"2026-01-01": 0.0, "2026-01-31": -0.05},
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "initial_capital": 10000.0,
    }
    fields.update(overrides)
    return BacktestResult(**fields)


# -- save_backtest_result / get_backtest_by_id -------------------------------


def test_save_backtest_result_returns_a_uuid(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())

    assert isinstance(backtest_id, uuid.UUID)


def test_save_backtest_result_uppercases_symbol(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result(symbol="aapl"))

    with session_scope(factory) as session:
        record = get_backtest_by_id(session, backtest_id)

    assert record is not None
    assert record.symbol == "AAPL"


def test_get_backtest_by_id_round_trips_core_fields(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(
            session, _result(), execution_time=3.5, notes="test run"
        )

    with session_scope(factory) as session:
        record = get_backtest_by_id(session, backtest_id)

    assert record is not None
    assert record.backtest_id == backtest_id
    assert record.symbol == "AAPL"
    assert record.strategy_type == "sma_cross"
    assert record.start_date == date(2026, 1, 1)
    assert record.end_date == date(2026, 1, 31)
    assert record.parameters == {"fast_period": 10, "slow_period": 50}
    assert record.equity_curve == {"2026-01-01": 10000.0, "2026-01-31": 10250.0}
    assert record.drawdown_series == {"2026-01-01": 0.0, "2026-01-31": -0.05}
    assert record.total_trades == 20
    assert record.winning_trades == 11
    assert record.losing_trades == 9
    assert record.data_points == 2
    assert record.notes == "test run"
    assert record.status == "completed"


def test_get_backtest_by_id_returns_none_when_not_found(factory):
    with session_scope(factory) as session:
        record = get_backtest_by_id(session, uuid.uuid4())

    assert record is None


def test_decimal_metrics_round_trip_exactly(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(
            session,
            _result(
                metrics=_metrics(
                    total_return=0.2534,
                    sharpe_ratio=1.2345,
                    max_drawdown=-0.1234,
                    win_rate=0.5556,
                    profit_factor=1.8765,
                )
            ),
        )

    with session_scope(factory) as session:
        record = get_backtest_by_id(session, backtest_id)

    assert record.total_return == Decimal("0.2534")
    assert record.sharpe_ratio == Decimal("1.2345")
    assert record.max_drawdown == Decimal("-0.1234")
    assert record.win_rate == Decimal("0.5556")
    assert record.profit_factor == Decimal("1.8765")


def test_execution_time_seconds_is_none_when_not_provided(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())

    with session_scope(factory) as session:
        record = get_backtest_by_id(session, backtest_id)

    assert record.execution_time_seconds is None


# -- trades ------------------------------------------------------------------


def test_save_backtest_result_saves_trades(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(
            session,
            _result(trades=[_trade(entry_price=100.0), _trade(entry_price=110.0)]),
        )

    with session_scope(factory) as session:
        trades = get_trades_for_backtest(session, backtest_id)

    assert len(trades) == 2
    assert {t.entry_price for t in trades} == {Decimal("100.0"), Decimal("110.0")}


def test_save_backtest_result_with_no_trades_saves_none(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result(trades=[]))

    with session_scope(factory) as session:
        trades = get_trades_for_backtest(session, backtest_id)

    assert trades == []


def test_trade_defaults_direction_to_long(factory):
    """TradeRecord has no `direction` field; the legacy `_save_trades` default
    (`trade.get("direction", "long")`) applies here too -- see module docstring."""
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result(trades=[_trade()]))

    with session_scope(factory) as session:
        trades = get_trades_for_backtest(session, backtest_id)

    assert trades[0].direction == "long"


def test_trade_fields_round_trip(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(
            session,
            _result(
                trades=[
                    _trade(
                        entry_date="2026-02-01",
                        exit_date="2026-02-06",
                        entry_price=200.25,
                        exit_price=210.75,
                        size=5.0,
                        pnl=52.5,
                        **{"return": 0.0525},
                    )
                ]
            ),
        )

    with session_scope(factory) as session:
        trades = get_trades_for_backtest(session, backtest_id)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.backtest_id == backtest_id
    assert trade.trade_number == 1
    assert trade.entry_date == date(2026, 2, 1)
    assert trade.exit_date == date(2026, 2, 6)
    assert trade.entry_price == Decimal("200.25")
    assert trade.exit_price == Decimal("210.75")
    assert trade.position_size == Decimal("5.0")
    assert trade.pnl == Decimal("52.5")
    assert trade.pnl_percent == Decimal("0.0525")


def test_trade_numbers_are_sequential_in_input_order(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(
            session,
            _result(
                trades=[
                    _trade(entry_date="2026-01-01", exit_date="2026-01-02"),
                    _trade(entry_date="2026-01-03", exit_date="2026-01-04"),
                    _trade(entry_date="2026-01-05", exit_date="2026-01-06"),
                ]
            ),
        )

    with session_scope(factory) as session:
        trades = get_trades_for_backtest(session, backtest_id)

    assert [t.trade_number for t in trades] == [1, 2, 3]


# -- get_backtests_by_symbol --------------------------------------------------


def test_get_backtests_by_symbol_filters_by_symbol(factory):
    with session_scope(factory) as session:
        save_backtest_result(session, _result(symbol="AAPL"))
        save_backtest_result(session, _result(symbol="MSFT"))

    with session_scope(factory) as session:
        results = get_backtests_by_symbol(session, "AAPL")

    assert len(results) == 1
    assert results[0].symbol == "AAPL"


def test_get_backtests_by_symbol_uppercases_query_symbol(factory):
    with session_scope(factory) as session:
        save_backtest_result(session, _result(symbol="AAPL"))

    with session_scope(factory) as session:
        results = get_backtests_by_symbol(session, "aapl")

    assert len(results) == 1


def test_get_backtests_by_symbol_filters_by_strategy_type(factory):
    with session_scope(factory) as session:
        save_backtest_result(session, _result(symbol="AAPL", strategy="sma_cross"))
        save_backtest_result(session, _result(symbol="AAPL", strategy="rsi"))

    with session_scope(factory) as session:
        results = get_backtests_by_symbol(session, "AAPL", strategy_type="rsi")

    assert len(results) == 1
    assert results[0].strategy_type == "rsi"


def test_get_backtests_by_symbol_respects_limit(factory):
    with session_scope(factory) as session:
        for _ in range(5):
            save_backtest_result(session, _result(symbol="AAPL"))

    with session_scope(factory) as session:
        results = get_backtests_by_symbol(session, "AAPL", limit=2)

    assert len(results) == 2


def test_get_backtests_by_symbol_returns_empty_for_unknown_symbol(factory):
    with session_scope(factory) as session:
        results = get_backtests_by_symbol(session, "NOPE")

    assert results == []


# -- save_optimization_results / get_optimization_results -------------------


def test_save_optimization_results_returns_count(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())
        count = save_optimization_results(
            session,
            backtest_id,
            [
                OptimizationResultInput(parameters={"fast": 5}, sharpe_ratio=1.1),
                OptimizationResultInput(parameters={"fast": 10}, sharpe_ratio=1.5),
            ],
        )

    assert count == 2


def test_save_optimization_results_with_empty_list_returns_zero(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())
        count = save_optimization_results(session, backtest_id, [])

    assert count == 0


def test_optimization_results_round_trip(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())
        save_optimization_results(
            session,
            backtest_id,
            [
                OptimizationResultInput(
                    parameters={"fast": 5, "slow": 20},
                    objective_value=1.85,
                    total_return=0.35,
                    sharpe_ratio=1.85,
                    max_drawdown=-0.1,
                    win_rate=0.6,
                    profit_factor=2.1,
                    total_trades=15,
                    rank=1,
                    is_statistically_significant=True,
                    p_value=0.02,
                )
            ],
            objective_function="sharpe_ratio",
        )

    with session_scope(factory) as session:
        results = get_optimization_results(session, backtest_id)

    assert len(results) == 1
    row = results[0]
    assert row.backtest_id == backtest_id
    assert row.parameters == {"fast": 5, "slow": 20}
    assert row.objective_function == "sharpe_ratio"
    assert row.objective_value == Decimal("1.85")
    assert row.sharpe_ratio == Decimal("1.85")
    assert row.rank == 1
    assert row.is_statistically_significant is True
    assert row.p_value == Decimal("0.02")


def test_optimization_result_defaults_rank_to_position_when_absent(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())
        save_optimization_results(
            session,
            backtest_id,
            [
                OptimizationResultInput(parameters={"fast": 5}),
                OptimizationResultInput(parameters={"fast": 10}),
            ],
        )

    with session_scope(factory) as session:
        results = get_optimization_results(session, backtest_id)

    assert sorted(r.rank for r in results) == [1, 2]


def test_optimization_result_input_requires_parameters():
    with pytest.raises(ValidationError):
        OptimizationResultInput()  # type: ignore[call-arg]


# -- save_walk_forward_test / get_walk_forward_tests -------------------------


def _walk_forward_input(**overrides) -> WalkForwardTestInput:
    fields = {
        "window_size_months": 12,
        "step_size_months": 3,
        "training_start": date(2025, 1, 1),
        "training_end": date(2025, 12, 31),
        "test_period_start": date(2026, 1, 1),
        "test_period_end": date(2026, 3, 31),
        "optimal_parameters": {"fast": 10},
        "training_performance": 0.15,
        "out_of_sample_return": 0.08,
        "out_of_sample_sharpe": 1.2,
        "out_of_sample_drawdown": -0.05,
        "out_of_sample_trades": 8,
        "performance_ratio": 0.53,
        "degradation_factor": 0.47,
        "is_profitable": True,
        "is_statistically_significant": False,
    }
    fields.update(overrides)
    return WalkForwardTestInput(**fields)


def test_save_walk_forward_test_returns_a_uuid(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())
        walk_forward_id = save_walk_forward_test(
            session, backtest_id, _walk_forward_input()
        )

    assert isinstance(walk_forward_id, uuid.UUID)


def test_walk_forward_test_round_trips(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result())
        save_walk_forward_test(session, backtest_id, _walk_forward_input())

    with session_scope(factory) as session:
        tests = get_walk_forward_tests(session, backtest_id)

    assert len(tests) == 1
    test = tests[0]
    assert test.parent_backtest_id == backtest_id
    assert test.window_size_months == 12
    assert test.step_size_months == 3
    assert test.training_start == date(2025, 1, 1)
    assert test.training_end == date(2025, 12, 31)
    assert test.test_period_start == date(2026, 1, 1)
    assert test.test_period_end == date(2026, 3, 31)
    assert test.optimal_parameters == {"fast": 10}
    assert test.out_of_sample_return == Decimal("0.08")
    assert test.is_profitable is True
    assert test.is_statistically_significant is False


def test_walk_forward_test_input_requires_date_fields():
    with pytest.raises(ValidationError):
        WalkForwardTestInput(window_size_months=12, step_size_months=3)  # type: ignore[call-arg]


# -- save_backtest_portfolio / list_backtest_portfolios ----------------------


def _portfolio_input(**overrides) -> BacktestPortfolioInput:
    fields = {
        "portfolio_name": "Core Holdings",
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 6, 30),
        "symbols": ["AAPL", "MSFT"],
        "weights": {"AAPL": 0.6, "MSFT": 0.4},
        "initial_capital": 100000.0,
        "total_return": 0.12,
        "sharpe_ratio": 1.4,
    }
    fields.update(overrides)
    return BacktestPortfolioInput(**fields)


def test_save_backtest_portfolio_returns_a_uuid(factory):
    with session_scope(factory) as session:
        portfolio_id = save_backtest_portfolio(session, _portfolio_input())

    assert isinstance(portfolio_id, uuid.UUID)


def test_backtest_portfolio_round_trips(factory):
    with session_scope(factory) as session:
        save_backtest_portfolio(session, _portfolio_input())

    with session_scope(factory) as session:
        portfolios = list_backtest_portfolios(session)

    assert len(portfolios) == 1
    portfolio = portfolios[0]
    assert portfolio.portfolio_name == "Core Holdings"
    assert portfolio.symbols == ["AAPL", "MSFT"]
    assert portfolio.weights == {"AAPL": 0.6, "MSFT": 0.4}
    assert portfolio.initial_capital == Decimal("100000.0")
    assert portfolio.total_return == Decimal("0.12")
    assert portfolio.sharpe_ratio == Decimal("1.4")
    assert portfolio.status == "completed"


def test_list_backtest_portfolios_filters_by_name(factory):
    with session_scope(factory) as session:
        save_backtest_portfolio(session, _portfolio_input(portfolio_name="Alpha"))
        save_backtest_portfolio(session, _portfolio_input(portfolio_name="Beta"))

    with session_scope(factory) as session:
        portfolios = list_backtest_portfolios(session, portfolio_name="Beta")

    assert len(portfolios) == 1
    assert portfolios[0].portfolio_name == "Beta"


def test_list_backtest_portfolios_respects_limit(factory):
    with session_scope(factory) as session:
        for i in range(5):
            save_backtest_portfolio(
                session, _portfolio_input(portfolio_name=f"Portfolio {i}")
            )

    with session_scope(factory) as session:
        portfolios = list_backtest_portfolios(session, limit=2)

    assert len(portfolios) == 2


def test_list_backtest_portfolios_returns_empty_when_none_saved(factory):
    with session_scope(factory) as session:
        portfolios = list_backtest_portfolios(session)

    assert portfolios == []


def test_backtest_portfolio_input_requires_symbols():
    with pytest.raises(ValidationError):
        BacktestPortfolioInput(
            portfolio_name="X",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
        )  # type: ignore[call-arg]


# -- FK integrity: trades -> results (RESTRICT, matching the legacy DDL) -----


def test_deleting_backtest_result_with_trades_is_restricted(factory):
    """The legacy `mcp_backtest_trades.backtest_id` FK declares no
    `ondelete=` (see module docstring's "FK behavior" note) -- with SQLite
    FK enforcement on, that's RESTRICT/NO ACTION, not CASCADE. Deleting a
    parent row while a child trade still references it must raise
    IntegrityError -- this is the opposite behavior of
    `maverick/portfolio/data.py`'s `pf_positions` (`ON DELETE CASCADE`), and
    the difference is deliberate, not an oversight."""
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result(trades=[_trade()]))

    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.execute(
                sa_delete(BACKTEST_RESULTS).where(
                    BACKTEST_RESULTS.c.backtest_id == backtest_id
                )
            )

    # The result and its trade both survive the failed delete attempt.
    with session_scope(factory) as session:
        result_count = session.execute(
            select(func.count())
            .select_from(BACKTEST_RESULTS)
            .where(BACKTEST_RESULTS.c.backtest_id == backtest_id)
        ).scalar_one()
        trade_count = session.execute(
            select(func.count())
            .select_from(BACKTEST_TRADES)
            .where(BACKTEST_TRADES.c.backtest_id == backtest_id)
        ).scalar_one()

    assert result_count == 1
    assert trade_count == 1


def test_deleting_backtest_result_without_trades_succeeds(factory):
    with session_scope(factory) as session:
        backtest_id = save_backtest_result(session, _result(trades=[]))

    with session_scope(factory) as session:
        session.execute(
            sa_delete(BACKTEST_RESULTS).where(
                BACKTEST_RESULTS.c.backtest_id == backtest_id
            )
        )

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count())
            .select_from(BACKTEST_RESULTS)
            .where(BACKTEST_RESULTS.c.backtest_id == backtest_id)
        ).scalar_one()

    assert count == 0


# -- degraded input: missing required fields raise validation errors --------


def test_backtest_result_payload_requires_symbol():
    fields = _result().model_dump()
    del fields["symbol"]
    with pytest.raises(ValidationError):
        BacktestResult(**fields)


def test_trade_record_requires_entry_price():
    fields = _trade().model_dump(by_alias=True)
    del fields["entry_price"]
    with pytest.raises(ValidationError):
        TradeRecord(**fields)
