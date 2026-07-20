"""Persistent trade-journal storage. Third layer (sibling of data/ledger/
watchlist): imports types only.

Table names (`journal_entries`, `strategy_performance`) and every column
are ported verbatim from the legacy `maverick_mcp.services.journal.models`
schema (`JournalEntry`, `StrategyPerformance`) so a pre-existing database
carries its rows forward with zero migration -- same rationale as
`watchlist.py`'s module docstring. Columns are `Float`, not `Numeric`:
legacy declared `entry_price`/`exit_price`/`shares`/`pnl`/`r_multiple` as
plain `Mapped[float]`, never `Numeric`, so this port keeps `Float` rather
than "upgrading" to a Decimal-backed column type -- doing so would make
`ensure_schema` skip DDL against a legacy-shaped database (exact
carry-over scenario) while every insert here now assumes a column shape
that was never created.

`journal_entries` carries `created_at`/`updated_at` from legacy's
`TimestampMixin` (`DateTime(timezone=True) NOT NULL`, Python-side defaults
only, no `server_default`) -- required, not cosmetic, per the same
carry-over reasoning as `watchlist.py`. `strategy_performance` has NO such
columns: legacy's `StrategyPerformance` model extends only `Base`, not
`TimestampMixin` -- adding them here would break inserts against a
pre-existing legacy-shaped `strategy_performance` table that lacks them.

`tags` filtering and `limit` slicing are done in the service tier
(`service_journal.py`), not here, matching legacy's own `JournalService`:
JSON "contains" queries vary too much across backends to push into SQL,
and `limit` is applied only after the (Python-side) tag filter.
"""

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import (
    JSON,
    Column,
    CursorResult,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    insert,
    select,
    update,
)
from sqlalchemy.orm import Session

from maverick.portfolio.types import JournalEntryPayload, StrategyPerformancePayload

METADATA = MetaData()


def _now() -> datetime:
    return datetime.now(UTC)


# `created_at`/`updated_at` on JOURNAL_ENTRIES mirror legacy's
# `TimestampMixin` exactly: Python-side `default`/`onupdate` callables, no
# `server_default`. STRATEGY_PERFORMANCE deliberately has neither column
# (see module docstring).

JOURNAL_ENTRIES = Table(
    "journal_entries",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False, index=True),
    Column("side", String(10), nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("exit_price", Float, nullable=True),
    Column("shares", Float, nullable=False),
    Column("entry_date", DateTime(timezone=True), nullable=False, default=_now),
    Column("exit_date", DateTime(timezone=True), nullable=True),
    Column("rationale", Text, nullable=True),
    Column("tags", JSON, nullable=False, default=list),
    Column("pnl", Float, nullable=True),
    Column("r_multiple", Float, nullable=True),
    Column("notes", Text, nullable=True),
    Column("status", String(10), nullable=False, default="open"),
    Column("created_at", DateTime(timezone=True), nullable=False, default=_now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=_now,
        onupdate=_now,
    ),
)

STRATEGY_PERFORMANCE = Table(
    "strategy_performance",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("strategy_tag", String(100), nullable=False, index=True, unique=True),
    Column("period", String(20), nullable=False, default="all_time"),
    Column("win_count", Integer, nullable=False, default=0),
    Column("loss_count", Integer, nullable=False, default=0),
    Column("total_pnl", Float, nullable=False, default=0.0),
    Column("avg_win", Float, nullable=False, default=0.0),
    Column("avg_loss", Float, nullable=False, default=0.0),
    Column("expectancy", Float, nullable=False, default=0.0),
    Column("profit_factor", Float, nullable=False, default=0.0),
)


def _inserted_id(result: CursorResult) -> int:
    """Narrow `inserted_primary_key` (typed `Sequence | None`) to the
    autoincrement id `int` -- always present for a successful single-row
    insert against a table with an `Integer` primary key."""
    primary_key = result.inserted_primary_key
    assert primary_key is not None
    return primary_key[0]


def _row_to_entry(row: Any) -> JournalEntryPayload:
    return JournalEntryPayload(
        id=row.id,
        symbol=row.symbol,
        side=row.side,
        entry_price=row.entry_price,
        exit_price=row.exit_price,
        shares=row.shares,
        entry_date=row.entry_date.isoformat(),
        exit_date=row.exit_date.isoformat() if row.exit_date else None,
        rationale=row.rationale,
        tags=list(row.tags) if row.tags else [],
        pnl=row.pnl,
        r_multiple=row.r_multiple,
        notes=row.notes,
        status=row.status,
    )


def insert_trade(
    session: Session,
    *,
    symbol: str,
    side: str,
    entry_price: float,
    shares: float,
    entry_date: datetime,
    rationale: str | None,
    tags: list[str],
    notes: str | None,
) -> JournalEntryPayload:
    """Insert a new open trade row (`symbol` uppercased, `side` lowercased --
    matches legacy `JournalService.add_trade`)."""
    result = cast(
        CursorResult,
        session.execute(
            insert(JOURNAL_ENTRIES).values(
                symbol=symbol.upper(),
                side=side.lower(),
                entry_price=entry_price,
                shares=shares,
                entry_date=entry_date,
                rationale=rationale,
                tags=tags,
                notes=notes,
                status="open",
            )
        ),
    )
    entry_id = _inserted_id(result)
    row = session.execute(
        select(JOURNAL_ENTRIES).where(JOURNAL_ENTRIES.c.id == entry_id)
    ).one()
    return _row_to_entry(row)


def read_trade(session: Session, entry_id: int) -> JournalEntryPayload | None:
    """Fetch a single journal entry by primary key, or `None` if not found."""
    row = session.execute(
        select(JOURNAL_ENTRIES).where(JOURNAL_ENTRIES.c.id == entry_id)
    ).one_or_none()
    return _row_to_entry(row) if row is not None else None


def update_trade_close(
    session: Session,
    entry_id: int,
    exit_price: float,
    exit_date: datetime,
    pnl: float,
    notes: str | None,
) -> JournalEntryPayload:
    """Mark an entry closed with the caller-computed `pnl`. The caller
    (`service_journal.py`) has already validated the entry exists, is open,
    and merged `notes` with any existing notes -- this only writes."""
    session.execute(
        update(JOURNAL_ENTRIES)
        .where(JOURNAL_ENTRIES.c.id == entry_id)
        .values(
            exit_price=exit_price,
            exit_date=exit_date,
            pnl=pnl,
            status="closed",
            notes=notes,
        )
    )
    row = session.execute(
        select(JOURNAL_ENTRIES).where(JOURNAL_ENTRIES.c.id == entry_id)
    ).one()
    return _row_to_entry(row)


def read_trades(
    session: Session, symbol: str | None, status: str | None
) -> list[JournalEntryPayload]:
    """Query journal entries by `symbol`/`status`, newest-entry-date first.
    No `limit` here and no `strategy_tag` filter -- matches legacy: both
    are applied in Python by the caller, after this full result set."""
    query = select(JOURNAL_ENTRIES)
    if symbol is not None:
        query = query.where(JOURNAL_ENTRIES.c.symbol == symbol.upper())
    if status is not None:
        query = query.where(JOURNAL_ENTRIES.c.status == status)
    query = query.order_by(JOURNAL_ENTRIES.c.entry_date.desc())
    rows = session.execute(query).all()
    return [_row_to_entry(row) for row in rows]


def read_closed_trades(session: Session) -> list[JournalEntryPayload]:
    """Every closed trade, for strategy-performance recomputation (tag
    filtering happens in Python -- matches legacy `StrategyTracker`)."""
    rows = session.execute(
        select(JOURNAL_ENTRIES).where(JOURNAL_ENTRIES.c.status == "closed")
    ).all()
    return [_row_to_entry(row) for row in rows]


def _row_to_performance(row: Any) -> StrategyPerformancePayload:
    return StrategyPerformancePayload(
        strategy_tag=row.strategy_tag,
        period=row.period,
        win_count=row.win_count,
        loss_count=row.loss_count,
        total_pnl=row.total_pnl,
        avg_win=row.avg_win,
        avg_loss=row.avg_loss,
        expectancy=row.expectancy,
        profit_factor=row.profit_factor,
    )


def upsert_strategy_performance(
    session: Session,
    strategy_tag: str,
    *,
    period: str,
    win_count: int,
    loss_count: int,
    total_pnl: float,
    avg_win: float,
    avg_loss: float,
    expectancy: float,
    profit_factor: float,
) -> StrategyPerformancePayload:
    """Upsert the aggregated row for `strategy_tag` (unique constraint on
    `strategy_tag` makes this an insert-or-update, matching legacy)."""
    existing = session.execute(
        select(STRATEGY_PERFORMANCE.c.id).where(
            STRATEGY_PERFORMANCE.c.strategy_tag == strategy_tag
        )
    ).scalar_one_or_none()

    values = {
        "period": period,
        "win_count": win_count,
        "loss_count": loss_count,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }
    if existing is None:
        session.execute(
            insert(STRATEGY_PERFORMANCE).values(strategy_tag=strategy_tag, **values)
        )
    else:
        session.execute(
            update(STRATEGY_PERFORMANCE)
            .where(STRATEGY_PERFORMANCE.c.id == existing)
            .values(**values)
        )

    row = session.execute(
        select(STRATEGY_PERFORMANCE).where(
            STRATEGY_PERFORMANCE.c.strategy_tag == strategy_tag
        )
    ).one()
    return _row_to_performance(row)


def read_strategy_performance(
    session: Session, strategy_tag: str
) -> StrategyPerformancePayload | None:
    row = session.execute(
        select(STRATEGY_PERFORMANCE).where(
            STRATEGY_PERFORMANCE.c.strategy_tag == strategy_tag
        )
    ).one_or_none()
    return _row_to_performance(row) if row is not None else None


def read_strategy_performance_ranked(
    session: Session,
) -> list[StrategyPerformancePayload]:
    """Every strategy's performance, ranked by expectancy descending --
    matches legacy `StrategyTracker.compare_strategies`."""
    rows = session.execute(
        select(STRATEGY_PERFORMANCE).order_by(STRATEGY_PERFORMANCE.c.expectancy.desc())
    ).all()
    return [_row_to_performance(row) for row in rows]
