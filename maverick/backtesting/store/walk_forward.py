"""Save/fetch operations for `mcp_walk_forward_tests`.

Ports `BacktestPersistenceManager.save_walk_forward_test`.
`get_walk_forward_tests` is an addition needed to round-trip-test what this
module saves (see `maverick/backtesting/store/__init__.py` for the full
ported-vs-dropped rationale).
"""

import uuid

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from maverick.backtesting.store._decimal import read_decimal, to_decimal
from maverick.backtesting.store.models import (
    WalkForwardTestInput,
    WalkForwardTestRecord,
)
from maverick.backtesting.store.tables import WALK_FORWARD_TESTS


def save_walk_forward_test(
    session: Session,
    parent_backtest_id: uuid.UUID,
    data: WalkForwardTestInput,
) -> uuid.UUID:
    """Save a walk-forward validation test row. Returns the generated `walk_forward_id`."""
    walk_forward_id = uuid.uuid4()
    session.execute(
        insert(WALK_FORWARD_TESTS).values(
            walk_forward_id=walk_forward_id,
            parent_backtest_id=parent_backtest_id,
            window_size_months=data.window_size_months,
            step_size_months=data.step_size_months,
            training_start=data.training_start,
            training_end=data.training_end,
            test_period_start=data.test_period_start,
            test_period_end=data.test_period_end,
            optimal_parameters=data.optimal_parameters,
            training_performance=to_decimal(data.training_performance),
            out_of_sample_return=to_decimal(data.out_of_sample_return),
            out_of_sample_sharpe=to_decimal(data.out_of_sample_sharpe),
            out_of_sample_drawdown=to_decimal(data.out_of_sample_drawdown),
            out_of_sample_trades=data.out_of_sample_trades,
            performance_ratio=to_decimal(data.performance_ratio),
            degradation_factor=to_decimal(data.degradation_factor),
            is_profitable=data.is_profitable,
            is_statistically_significant=data.is_statistically_significant,
        )
    )
    return walk_forward_id


def get_walk_forward_tests(
    session: Session, parent_backtest_id: uuid.UUID
) -> list[WalkForwardTestRecord]:
    """Fetch every walk-forward test row for `parent_backtest_id`, ordered by test period."""
    rows = session.execute(
        select(WALK_FORWARD_TESTS)
        .where(WALK_FORWARD_TESTS.c.parent_backtest_id == parent_backtest_id)
        .order_by(WALK_FORWARD_TESTS.c.test_period_start)
    ).all()
    return [
        WalkForwardTestRecord(
            walk_forward_id=row.walk_forward_id,
            parent_backtest_id=row.parent_backtest_id,
            window_size_months=row.window_size_months,
            step_size_months=row.step_size_months,
            training_start=row.training_start,
            training_end=row.training_end,
            test_period_start=row.test_period_start,
            test_period_end=row.test_period_end,
            optimal_parameters=row.optimal_parameters,
            training_performance=read_decimal(row.training_performance),
            out_of_sample_return=read_decimal(row.out_of_sample_return),
            out_of_sample_sharpe=read_decimal(row.out_of_sample_sharpe),
            out_of_sample_drawdown=read_decimal(row.out_of_sample_drawdown),
            out_of_sample_trades=row.out_of_sample_trades,
            performance_ratio=read_decimal(row.performance_ratio),
            degradation_factor=read_decimal(row.degradation_factor),
            is_profitable=row.is_profitable,
            is_statistically_significant=bool(row.is_statistically_significant),
        )
        for row in rows
    ]
