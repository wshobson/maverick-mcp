"""Persistent price-bar storage. Third layer: imports config and types."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from maverick.market_data.types import PRICE_COLUMNS, HistoryState

METADATA = MetaData()

MD_STOCKS = Table(
    "md_stocks",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False, unique=True, index=True),
    Column("company_name", String(255), nullable=True),
    Column("history_refreshed_at", DateTime(timezone=True), nullable=True),
    Column("history_generation", Integer, nullable=True),
)

MD_PRICE_BARS = Table(
    "md_price_bars",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("stock_id", Integer, ForeignKey("md_stocks.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("open", Numeric(12, 4), nullable=False),
    Column("high", Numeric(12, 4), nullable=False),
    Column("low", Numeric(12, 4), nullable=False),
    Column("close", Numeric(12, 4), nullable=False),
    Column("volume", BigInteger, nullable=False),
    UniqueConstraint("stock_id", "date", name="md_price_bars_stock_date_unique"),
)


def _to_date(value: pd.Timestamp) -> date:
    """Normalize a DataFrame index entry to a tz-naive ``date``."""
    timestamp = value.tz_localize(None) if value.tzinfo is not None else value
    return timestamp.date()


def _empty_price_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="Date").as_unit("ns")
    data = {
        col: pd.Series(dtype="int64" if col == "Volume" else "float64")
        for col in PRICE_COLUMNS
    }
    return pd.DataFrame(data, index=index)


def _find_stock_id(session: Session, symbol: str) -> int | None:
    return session.execute(
        select(MD_STOCKS.c.id).where(MD_STOCKS.c.symbol == symbol)
    ).scalar_one_or_none()


def get_or_create_stock(session: Session, symbol: str) -> int:
    """Return the stock id, using conflict-safe insertion for supported backends."""
    stock_id = _find_stock_id(session, symbol)
    if stock_id is not None:
        return stock_id

    dialect = session.get_bind().dialect.name
    if dialect in ("sqlite", "postgresql"):
        statement = (sqlite_insert if dialect == "sqlite" else pg_insert)(MD_STOCKS)
        session.execute(
            statement.values(symbol=symbol).on_conflict_do_nothing(
                index_elements=[MD_STOCKS.c.symbol]
            )
        )
    else:
        try:
            with session.begin_nested():
                session.execute(insert(MD_STOCKS).values(symbol=symbol))
        except IntegrityError:
            if _find_stock_id(session, symbol) is None:
                raise

    stock_id = _find_stock_id(session, symbol)
    if stock_id is None:
        raise RuntimeError(f"Failed to create or find stock row for symbol {symbol!r}")
    return stock_id


def list_symbols(session: Session) -> list[str]:
    """Return every symbol registered in ``md_stocks``, alphabetically.

    The public entry point other domains (e.g. screening's default universe)
    use to enumerate known symbols without reaching into ``MD_STOCKS`` directly.
    """
    return list(
        session.execute(
            select(MD_STOCKS.c.symbol).distinct().order_by(MD_STOCKS.c.symbol)
        ).scalars()
    )


def read_price_range(
    session: Session, symbol: str, start: date, end: date
) -> pd.DataFrame:
    """Read cached price bars for ``symbol`` between ``start`` and ``end``, inclusive."""
    stock_id = _find_stock_id(session, symbol)
    if stock_id is None:
        return _empty_price_frame()

    rows = session.execute(
        select(
            MD_PRICE_BARS.c.date,
            MD_PRICE_BARS.c.open,
            MD_PRICE_BARS.c.high,
            MD_PRICE_BARS.c.low,
            MD_PRICE_BARS.c.close,
            MD_PRICE_BARS.c.volume,
        )
        .where(
            MD_PRICE_BARS.c.stock_id == stock_id,
            MD_PRICE_BARS.c.date >= start,
            MD_PRICE_BARS.c.date <= end,
        )
        .order_by(MD_PRICE_BARS.c.date)
    ).all()

    if not rows:
        return _empty_price_frame()

    # pandas 3 infers `s` from `date` objects; pin ns so the reader's index
    # dtype is stable across pandas versions and independent of how a caller
    # built the frame it compares against.
    index = pd.DatetimeIndex(
        [pd.Timestamp(row.date) for row in rows], name="Date"
    ).as_unit("ns")
    return pd.DataFrame(
        {
            "Open": [float(row.open) for row in rows],
            "High": [float(row.high) for row in rows],
            "Low": [float(row.low) for row in rows],
            "Close": [float(row.close) for row in rows],
            "Volume": [int(row.volume) for row in rows],
        },
        index=index,
    )


def write_price_bars(session: Session, symbol: str, df: pd.DataFrame) -> int:
    """Upsert bars; return the number of distinct supplied dates, including updates.

    SQLite/PostgreSQL resolve concurrent date conflicts in the database.
    Duplicate input dates use the last supplied bar. Financial values reach
    Numeric columns through Decimal rather than float arithmetic.
    """
    if df.empty:
        return 0
    stock_id = get_or_create_stock(session, symbol)
    rows = {}
    for ts, row in df.iterrows():
        bar_date = _to_date(cast(pd.Timestamp, ts))
        rows[bar_date] = {
            "stock_id": stock_id,
            "date": bar_date,
            **{name.lower(): Decimal(str(row[name])) for name in PRICE_COLUMNS[:-1]},
            "volume": int(row["Volume"]),
        }
    dialect = session.get_bind().dialect.name
    if dialect in ("sqlite", "postgresql"):
        statement = (sqlite_insert if dialect == "sqlite" else pg_insert)(MD_PRICE_BARS)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[MD_PRICE_BARS.c.stock_id, MD_PRICE_BARS.c.date],
                set_={
                    name.lower(): statement.excluded[name.lower()]
                    for name in PRICE_COLUMNS
                },
            ),
            [rows[day] for day in sorted(rows)],
        )
    else:
        raise ValueError("Price history storage requires SQLite or PostgreSQL")
    return len(rows)


def lock_history_state(session: Session, symbol: str) -> HistoryState:
    """Read stored coverage under the stock row lock (SQLite uses BEGIN IMMEDIATE)."""
    stock_id = get_or_create_stock(session, symbol)
    refreshed_at = session.execute(
        select(MD_STOCKS.c.history_refreshed_at)
        .where(MD_STOCKS.c.id == stock_id)
        .with_for_update()
    ).scalar_one()
    if refreshed_at is not None and refreshed_at.tzinfo is None:
        refreshed_at = refreshed_at.replace(tzinfo=UTC)
    dates = tuple(
        session.execute(
            select(MD_PRICE_BARS.c.date)
            .where(MD_PRICE_BARS.c.stock_id == stock_id)
            .order_by(MD_PRICE_BARS.c.date)
        ).scalars()
    )
    return HistoryState(stock_id, refreshed_at, dates)


def reserve_history_generation(session: Session, stock_id: int) -> int:
    """Reserve a monotonically increasing generation in the caller's transaction."""
    return session.execute(
        update(MD_STOCKS)
        .where(MD_STOCKS.c.id == stock_id)
        .values(history_generation=func.coalesce(MD_STOCKS.c.history_generation, 0) + 1)
        .returning(MD_STOCKS.c.history_generation)
    ).scalar_one()


def commit_history_snapshot(
    session: Session,
    symbol: str,
    generation: int,
    frame: pd.DataFrame,
    refreshed_at: datetime | None,
) -> bool:
    """Atomically accept only the latest reserved generation and its entire snapshot."""
    stock_id = session.execute(
        update(MD_STOCKS)
        .where(
            MD_STOCKS.c.symbol == symbol, MD_STOCKS.c.history_generation == generation
        )
        .values(history_refreshed_at=refreshed_at)
        .returning(MD_STOCKS.c.id)
    ).scalar_one_or_none()
    if stock_id is None:
        return False
    write_price_bars(session, symbol, frame)
    return True


def cached_date_range(session: Session, symbol: str) -> tuple[date, date] | None:
    """Return the ``(min, max)`` cached date span for ``symbol``, or ``None``."""
    stock_id = _find_stock_id(session, symbol)
    if stock_id is None:
        return None

    min_date, max_date = session.execute(
        select(func.min(MD_PRICE_BARS.c.date), func.max(MD_PRICE_BARS.c.date)).where(
            MD_PRICE_BARS.c.stock_id == stock_id
        )
    ).one()

    if min_date is None or max_date is None:
        return None
    return (min_date, max_date)
