"""Save/list operations for `mcp_backtest_portfolios`.

Neither a "save" nor a "list" operation existed as a
`BacktestPersistenceManager` method in the legacy code -- the table has a
model (`BacktestPortfolio`) but no persistence-manager wrapper around it.
`list_backtest_portfolios` ports the model's own
`BacktestPortfolio.get_portfolio_backtests` classmethod;
`save_backtest_portfolio` is new, following the same insert pattern as the
other `save_*` functions in this package (see
`maverick/backtesting/store/__init__.py`'s "Ported vs. dropped operations"
section).
"""

import uuid

from sqlalchemy import desc, insert, select
from sqlalchemy.orm import Session

from maverick.backtesting.store._decimal import read_decimal, to_decimal
from maverick.backtesting.store.models import (
    BacktestPortfolioInput,
    BacktestPortfolioRecord,
)
from maverick.backtesting.store.tables import BACKTEST_PORTFOLIOS


def save_backtest_portfolio(
    session: Session, portfolio: BacktestPortfolioInput
) -> uuid.UUID:
    """Save a portfolio-level backtest row. Returns the generated `portfolio_backtest_id`."""
    portfolio_backtest_id = uuid.uuid4()
    session.execute(
        insert(BACKTEST_PORTFOLIOS).values(
            portfolio_backtest_id=portfolio_backtest_id,
            portfolio_name=portfolio.portfolio_name,
            description=portfolio.description,
            start_date=portfolio.start_date,
            end_date=portfolio.end_date,
            symbols=portfolio.symbols,
            weights=portfolio.weights,
            rebalance_frequency=portfolio.rebalance_frequency,
            initial_capital=to_decimal(portfolio.initial_capital),
            max_positions=portfolio.max_positions,
            position_sizing_method=portfolio.position_sizing_method,
            portfolio_stop_loss=to_decimal(portfolio.portfolio_stop_loss),
            max_sector_allocation=to_decimal(portfolio.max_sector_allocation),
            correlation_threshold=to_decimal(portfolio.correlation_threshold),
            total_return=to_decimal(portfolio.total_return),
            annualized_return=to_decimal(portfolio.annualized_return),
            sharpe_ratio=to_decimal(portfolio.sharpe_ratio),
            sortino_ratio=to_decimal(portfolio.sortino_ratio),
            max_drawdown=to_decimal(portfolio.max_drawdown),
            volatility=to_decimal(portfolio.volatility),
            diversification_ratio=to_decimal(portfolio.diversification_ratio),
            concentration_index=to_decimal(portfolio.concentration_index),
            turnover_rate=to_decimal(portfolio.turnover_rate),
            component_backtest_ids=portfolio.component_backtest_ids,
            portfolio_equity_curve=portfolio.portfolio_equity_curve,
            portfolio_weights_history=portfolio.portfolio_weights_history,
            status=portfolio.status,
            notes=portfolio.notes,
        )
    )
    return portfolio_backtest_id


def list_backtest_portfolios(
    session: Session,
    portfolio_name: str | None = None,
    limit: int = 10,
) -> list[BacktestPortfolioRecord]:
    """List portfolio-level backtests, optionally filtered by name, most-recent-first.

    Matches the legacy `BacktestPortfolio.get_portfolio_backtests` classmethod.
    """
    query = select(BACKTEST_PORTFOLIOS)
    if portfolio_name is not None:
        query = query.where(BACKTEST_PORTFOLIOS.c.portfolio_name == portfolio_name)
    query = query.order_by(desc(BACKTEST_PORTFOLIOS.c.backtest_date)).limit(limit)

    rows = session.execute(query).all()
    return [
        BacktestPortfolioRecord(
            portfolio_backtest_id=row.portfolio_backtest_id,
            portfolio_name=row.portfolio_name,
            description=row.description,
            backtest_date=row.backtest_date,
            start_date=row.start_date,
            end_date=row.end_date,
            symbols=row.symbols,
            weights=row.weights,
            rebalance_frequency=row.rebalance_frequency,
            initial_capital=read_decimal(row.initial_capital),
            total_return=read_decimal(row.total_return),
            sharpe_ratio=read_decimal(row.sharpe_ratio),
            max_drawdown=read_decimal(row.max_drawdown),
            status=row.status,
            notes=row.notes,
        )
        for row in rows
    ]
