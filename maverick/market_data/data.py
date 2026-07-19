"""Persistent price-bar storage. Third layer: imports config and types."""

from datetime import date

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
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
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from maverick.market_data.types import PRICE_COLUMNS

METADATA = MetaData()

MD_STOCKS = Table(
    "md_stocks",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(20), nullable=False, unique=True, index=True),
    Column("company_name", String(255), nullable=True),
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
    index = pd.DatetimeIndex([], name="Date")
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
    """Return the ``md_stocks`` row id for ``symbol``, creating it if absent.

    Guards against a concurrent first-create race: if another session's
    insert for the same ``symbol`` commits between this session's check and
    its own insert, the unique-constraint violation raises ``IntegrityError``
    here. That's caught, the insert is rolled back to a savepoint taken just
    before it (so the outer transaction stays usable), and the winner's row
    is re-selected instead.
    """
    stock_id = _find_stock_id(session, symbol)
    if stock_id is not None:
        return stock_id

    try:
        with session.begin_nested():
            session.execute(insert(MD_STOCKS).values(symbol=symbol))
            session.flush()
    except IntegrityError:
        stock_id = _find_stock_id(session, symbol)
        if stock_id is None:
            raise
        return stock_id

    stock_id = _find_stock_id(session, symbol)
    if stock_id is None:
        raise RuntimeError(f"Failed to create or find stock row for symbol {symbol!r}")
    return stock_id


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

    index = pd.DatetimeIndex([pd.Timestamp(row.date) for row in rows], name="Date")
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
    """Insert new price bars for ``symbol``, skipping dates already cached.

    Accepts a yfinance-cased OHLCV DataFrame (``Open``/``High``/``Low``/
    ``Close``/``Volume``) indexed by date. Returns the count of newly
    inserted rows; existing dates are left untouched.
    """
    if df.empty:
        return 0

    stock_id = get_or_create_stock(session, symbol)

    incoming = [(ts, _to_date(ts)) for ts in df.index]
    incoming_dates = [bar_date for _, bar_date in incoming]
    existing_dates = set(
        session.execute(
            select(MD_PRICE_BARS.c.date).where(
                MD_PRICE_BARS.c.stock_id == stock_id,
                MD_PRICE_BARS.c.date.in_(incoming_dates),
            )
        ).scalars()
    )

    rows_to_insert = []
    for ts, bar_date in incoming:
        if bar_date in existing_dates:
            continue
        row = df.loc[ts]
        rows_to_insert.append(
            {
                "stock_id": stock_id,
                "date": bar_date,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )
        existing_dates.add(bar_date)

    if not rows_to_insert:
        return 0

    session.execute(insert(MD_PRICE_BARS), rows_to_insert)
    return len(rows_to_insert)


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
