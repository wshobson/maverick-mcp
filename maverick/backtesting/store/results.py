"""Save/fetch operations for `mcp_backtest_results` and `mcp_backtest_trades`.

Ports `BacktestPersistenceManager.save_backtest_result` (+ its
`_save_trades` helper), `get_backtest_by_id`, and
`get_backtests_by_symbol` from `maverick_mcp/backtesting/persistence.py`.
`get_trades_for_backtest` is an addition needed to round-trip-test the
trades this module saves (see `maverick/backtesting/store/__init__.py` for
the full ported-vs-dropped rationale).
"""

import uuid
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import desc, insert, select
from sqlalchemy.orm import Session

from maverick.backtesting.store._decimal import read_decimal, to_decimal
from maverick.backtesting.store.models import BacktestResultRecord, BacktestTradeRecord
from maverick.backtesting.store.tables import BACKTEST_RESULTS, BACKTEST_TRADES
from maverick.backtesting.types import BacktestResult as BacktestResultPayload
from maverick.backtesting.types import TradeRecord


def _trade_values(
    backtest_id: uuid.UUID, index: int, trade: TradeRecord
) -> dict[str, Any]:
    return {
        "trade_id": uuid.uuid4(),
        "backtest_id": backtest_id,
        "trade_number": index,
        "entry_date": date.fromisoformat(trade.entry_date),
        "entry_price": to_decimal(trade.entry_price),
        "exit_date": date.fromisoformat(trade.exit_date),
        "exit_price": to_decimal(trade.exit_price),
        "position_size": to_decimal(trade.size),
        # TradeRecord has no `direction` field; default to "long" matching
        # the legacy `_save_trades`'s `trade.get("direction", "long")`.
        "direction": "long",
        "pnl": to_decimal(trade.pnl),
        "pnl_percent": to_decimal(trade.return_),
    }


def save_backtest_result(
    session: Session,
    result: BacktestResultPayload,
    *,
    execution_time: float | None = None,
    notes: str | None = None,
) -> uuid.UUID:
    """Save a backtest result and its trades. Returns the generated `backtest_id`.

    See the `store` package docstring's "Fields left NULL rather than
    guessed" note for which legacy columns this doesn't populate.
    """
    backtest_id = uuid.uuid4()
    metrics = result.metrics

    session.execute(
        insert(BACKTEST_RESULTS).values(
            backtest_id=backtest_id,
            symbol=result.symbol.upper(),
            strategy_type=result.strategy,
            start_date=date.fromisoformat(result.start_date),
            end_date=date.fromisoformat(result.end_date),
            initial_capital=to_decimal(result.initial_capital),
            parameters=result.parameters,
            total_return=to_decimal(metrics.total_return),
            sharpe_ratio=to_decimal(metrics.sharpe_ratio),
            sortino_ratio=to_decimal(metrics.sortino_ratio),
            calmar_ratio=to_decimal(metrics.calmar_ratio),
            max_drawdown=to_decimal(metrics.max_drawdown),
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            win_rate=to_decimal(metrics.win_rate),
            profit_factor=to_decimal(metrics.profit_factor),
            equity_curve=result.equity_curve,
            drawdown_series=result.drawdown_series,
            execution_time_seconds=to_decimal(execution_time),
            data_points=len(result.equity_curve),
            notes=notes,
        )
    )

    if result.trades:
        session.execute(
            insert(BACKTEST_TRADES),
            [
                _trade_values(backtest_id, i, trade)
                for i, trade in enumerate(result.trades, 1)
            ],
        )

    return backtest_id


def _row_to_result_record(row: Any) -> BacktestResultRecord:
    return BacktestResultRecord(
        backtest_id=row.backtest_id,
        symbol=row.symbol,
        strategy_type=row.strategy_type,
        backtest_date=row.backtest_date,
        start_date=row.start_date,
        end_date=row.end_date,
        initial_capital=read_decimal(row.initial_capital),
        parameters=row.parameters,
        total_return=read_decimal(row.total_return),
        annualized_return=read_decimal(row.annualized_return),
        sharpe_ratio=read_decimal(row.sharpe_ratio),
        sortino_ratio=read_decimal(row.sortino_ratio),
        calmar_ratio=read_decimal(row.calmar_ratio),
        max_drawdown=read_decimal(row.max_drawdown),
        total_trades=row.total_trades,
        winning_trades=row.winning_trades,
        losing_trades=row.losing_trades,
        win_rate=read_decimal(row.win_rate),
        profit_factor=read_decimal(row.profit_factor),
        equity_curve=row.equity_curve,
        drawdown_series=row.drawdown_series,
        execution_time_seconds=read_decimal(row.execution_time_seconds),
        data_points=row.data_points,
        status=row.status,
        notes=row.notes,
    )


def get_backtest_by_id(
    session: Session, backtest_id: uuid.UUID
) -> BacktestResultRecord | None:
    """Fetch a single backtest result by id, or `None` if not found."""
    row = session.execute(
        select(BACKTEST_RESULTS).where(BACKTEST_RESULTS.c.backtest_id == backtest_id)
    ).first()
    return _row_to_result_record(row) if row is not None else None


def get_backtests_by_symbol(
    session: Session,
    symbol: str,
    strategy_type: str | None = None,
    limit: int = 10,
) -> list[BacktestResultRecord]:
    """Fetch backtest results for `symbol`, optionally filtered by `strategy_type`.

    Ordered most-recent-first, matching the legacy
    `BacktestPersistenceManager.get_backtests_by_symbol`.
    """
    query = select(BACKTEST_RESULTS).where(BACKTEST_RESULTS.c.symbol == symbol.upper())
    if strategy_type is not None:
        query = query.where(BACKTEST_RESULTS.c.strategy_type == strategy_type)
    query = query.order_by(desc(BACKTEST_RESULTS.c.backtest_date)).limit(limit)

    rows = session.execute(query).all()
    return [_row_to_result_record(row) for row in rows]


def get_trades_for_backtest(
    session: Session, backtest_id: uuid.UUID
) -> list[BacktestTradeRecord]:
    """Fetch every trade for `backtest_id`, ordered by entry date then trade number."""
    rows = session.execute(
        select(BACKTEST_TRADES)
        .where(BACKTEST_TRADES.c.backtest_id == backtest_id)
        .order_by(BACKTEST_TRADES.c.entry_date, BACKTEST_TRADES.c.trade_number)
    ).all()
    return [
        BacktestTradeRecord(
            trade_id=row.trade_id,
            backtest_id=row.backtest_id,
            trade_number=row.trade_number,
            entry_date=row.entry_date,
            # entry_price is `nullable=False`; read directly rather than via
            # `read_decimal` so the type checker sees a plain `Decimal`, not
            # `Decimal | None`.
            entry_price=Decimal(str(row.entry_price)),
            exit_date=row.exit_date,
            exit_price=read_decimal(row.exit_price),
            position_size=read_decimal(row.position_size),
            direction=row.direction,
            pnl=read_decimal(row.pnl),
            pnl_percent=read_decimal(row.pnl_percent),
        )
        for row in rows
    ]
