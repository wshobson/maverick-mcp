"""Save/fetch operations for `mcp_optimization_results`.

Ports `BacktestPersistenceManager.save_optimization_results`.
`get_optimization_results` is an addition needed to round-trip-test what
this module saves (see `maverick/backtesting/store/__init__.py` for the
full ported-vs-dropped rationale).
"""

import uuid

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from maverick.backtesting.store._decimal import read_decimal, to_decimal
from maverick.backtesting.store.models import (
    OptimizationResultInput,
    OptimizationResultRecord,
)
from maverick.backtesting.store.tables import OPTIMIZATION_RESULTS


def save_optimization_results(
    session: Session,
    backtest_id: uuid.UUID,
    results: list[OptimizationResultInput],
    objective_function: str = "sharpe_ratio",
) -> int:
    """Save a batch of optimization-run rows for `backtest_id`. Returns the count saved."""
    if not results:
        return 0

    session.execute(
        insert(OPTIMIZATION_RESULTS),
        [
            {
                "optimization_id": uuid.uuid4(),
                "backtest_id": backtest_id,
                "parameter_set": i,
                "parameters": result.parameters,
                "objective_function": objective_function,
                "objective_value": to_decimal(result.objective_value),
                "total_return": to_decimal(result.total_return),
                "sharpe_ratio": to_decimal(result.sharpe_ratio),
                "max_drawdown": to_decimal(result.max_drawdown),
                "win_rate": to_decimal(result.win_rate),
                "profit_factor": to_decimal(result.profit_factor),
                "total_trades": result.total_trades,
                # Matches legacy: `rank=result.get("rank", i)`.
                "rank": result.rank if result.rank is not None else i,
                "is_statistically_significant": result.is_statistically_significant,
                "p_value": to_decimal(result.p_value),
            }
            for i, result in enumerate(results, 1)
        ],
    )
    return len(results)


def get_optimization_results(
    session: Session, backtest_id: uuid.UUID
) -> list[OptimizationResultRecord]:
    """Fetch every optimization-run row for `backtest_id`, ranked best-first."""
    rows = session.execute(
        select(OPTIMIZATION_RESULTS)
        .where(OPTIMIZATION_RESULTS.c.backtest_id == backtest_id)
        .order_by(OPTIMIZATION_RESULTS.c.rank)
    ).all()
    return [
        OptimizationResultRecord(
            optimization_id=row.optimization_id,
            backtest_id=row.backtest_id,
            parameter_set=row.parameter_set,
            parameters=row.parameters,
            objective_function=row.objective_function,
            objective_value=read_decimal(row.objective_value),
            total_return=read_decimal(row.total_return),
            sharpe_ratio=read_decimal(row.sharpe_ratio),
            max_drawdown=read_decimal(row.max_drawdown),
            win_rate=read_decimal(row.win_rate),
            profit_factor=read_decimal(row.profit_factor),
            total_trades=row.total_trades,
            rank=row.rank,
            is_statistically_significant=bool(row.is_statistically_significant),
            p_value=read_decimal(row.p_value),
        )
        for row in rows
    ]
