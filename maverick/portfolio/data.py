"""Persistent portfolio storage. Third layer (sibling of ledger): imports config and types.

No math lives here -- these functions store and retrieve exactly what the
ledger computed. Two conventions worth calling out:

* Decimal round-trip exactness: `shares`/`average_cost_basis`/`total_cost`
  are bound and read back explicitly via `Decimal(str(...))` rather than
  relying on the driver's native Decimal handling, so the 8/4/4-place
  precision the columns declare survives a write-then-read cycle exactly,
  independent of backend.
* `purchase_date` is an opaque ISO 8601 string on `PositionPayload` (see
  `ledger.py`), but the column is a real `DateTime(timezone=True)`. SQLite
  (this project's only tested backend) drops tzinfo on read, so writes
  normalize to UTC first and reads reattach UTC tzinfo. That preserves the
  exact instant (`==` on aware datetimes is offset-independent) but not the
  original UTC offset's string form.

`pf_positions.portfolio_id`'s `ON DELETE CASCADE` relies on SQLite FK
enforcement being turned on for the engine in use -- that's a platform-seam
concern (`maverick.platform.db.create_engine_from_settings`), not this
module's; see its docstring for the policy.
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from sqlalchemy import (
    Column,
    CursorResult,
    DateTime,
    ForeignKey,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from maverick.portfolio.types import PositionPayload

METADATA = MetaData()

PF_PORTFOLIOS = Table(
    "pf_portfolios",
    METADATA,
    Column("id", Uuid, primary_key=True, default=uuid.uuid4),
    Column("user_id", String(100), nullable=False, index=True),
    Column("name", String(200), nullable=False),
    UniqueConstraint("user_id", "name", name="pf_portfolios_user_name_unique"),
)

PF_POSITIONS = Table(
    "pf_positions",
    METADATA,
    Column("id", Uuid, primary_key=True, default=uuid.uuid4),
    Column(
        "portfolio_id",
        Uuid,
        ForeignKey("pf_portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("ticker", String(20), nullable=False, index=True),
    Column("shares", Numeric(20, 8), nullable=False),
    Column("average_cost_basis", Numeric(12, 4), nullable=False),
    Column("total_cost", Numeric(20, 4), nullable=False),
    Column("purchase_date", DateTime(timezone=True), nullable=False),
    Column("notes", Text, nullable=True),
    UniqueConstraint(
        "portfolio_id", "ticker", name="pf_positions_portfolio_ticker_unique"
    ),
)


def _find_portfolio_id(session: Session, user_id: str, name: str) -> uuid.UUID | None:
    return session.execute(
        select(PF_PORTFOLIOS.c.id).where(
            PF_PORTFOLIOS.c.user_id == user_id, PF_PORTFOLIOS.c.name == name
        )
    ).scalar_one_or_none()


def get_or_create_portfolio(session: Session, user_id: str, name: str) -> uuid.UUID:
    """Return the ``pf_portfolios`` row id for ``(user_id, name)``, creating it if absent.

    Idempotent: repeat calls with the same pair return the same id and never
    create a duplicate row (the unique constraint enforces this even under a
    concurrent first-create race, mirroring `market_data.data.get_or_create_stock`).
    """
    portfolio_id = _find_portfolio_id(session, user_id, name)
    if portfolio_id is not None:
        return portfolio_id

    try:
        with session.begin_nested():
            session.execute(insert(PF_PORTFOLIOS).values(user_id=user_id, name=name))
            session.flush()
    except IntegrityError:
        portfolio_id = _find_portfolio_id(session, user_id, name)
        if portfolio_id is None:
            raise
        return portfolio_id

    portfolio_id = _find_portfolio_id(session, user_id, name)
    if portfolio_id is None:
        raise RuntimeError(
            f"Failed to create or find portfolio row for ({user_id!r}, {name!r})"
        )
    return portfolio_id


def _purchase_date_to_datetime(value: str) -> datetime:
    """Parse `PositionPayload.purchase_date` and normalize to UTC-aware."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _datetime_to_purchase_date(value: datetime) -> str:
    """Inverse of `_purchase_date_to_datetime`: reattach UTC tzinfo and format.

    SQLite drops tzinfo on read (the DBAPI returns a naive datetime), but the
    stored value was normalized to UTC on write, so reattaching UTC here is
    correct rather than assumed.
    """
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.isoformat()


def _position_values(position: PositionPayload) -> dict[str, object]:
    return {
        "ticker": position.ticker,
        "shares": Decimal(str(position.shares)),
        "average_cost_basis": Decimal(str(position.average_cost_basis)),
        "total_cost": Decimal(str(position.total_cost)),
        "purchase_date": _purchase_date_to_datetime(position.purchase_date),
        "notes": position.notes,
    }


def _row_to_position(row) -> PositionPayload:  # noqa: ANN001
    return PositionPayload(
        ticker=row.ticker,
        shares=Decimal(str(row.shares)),
        average_cost_basis=Decimal(str(row.average_cost_basis)),
        total_cost=Decimal(str(row.total_cost)),
        purchase_date=_datetime_to_purchase_date(row.purchase_date),
        notes=row.notes,
    )


def read_positions(session: Session, portfolio_id: uuid.UUID) -> list[PositionPayload]:
    """Return every position for ``portfolio_id``, ordered by ticker."""
    rows = session.execute(
        select(PF_POSITIONS)
        .where(PF_POSITIONS.c.portfolio_id == portfolio_id)
        .order_by(PF_POSITIONS.c.ticker)
    ).all()
    return [_row_to_position(row) for row in rows]


def upsert_position(
    session: Session, portfolio_id: uuid.UUID, position: PositionPayload
) -> None:
    """Insert ``position`` for ``portfolio_id``, or update it if that ticker
    already exists in the portfolio (no duplicate rows)."""
    existing_id = session.execute(
        select(PF_POSITIONS.c.id).where(
            PF_POSITIONS.c.portfolio_id == portfolio_id,
            PF_POSITIONS.c.ticker == position.ticker,
        )
    ).scalar_one_or_none()

    values = _position_values(position)
    if existing_id is None:
        session.execute(
            insert(PF_POSITIONS).values(portfolio_id=portfolio_id, **values)
        )
    else:
        session.execute(
            update(PF_POSITIONS)
            .where(PF_POSITIONS.c.id == existing_id)
            .values(**values)
        )


def delete_position(session: Session, portfolio_id: uuid.UUID, ticker: str) -> bool:
    """Delete the position for ``ticker`` in ``portfolio_id``. Returns whether a row was removed."""
    result = cast(
        CursorResult,
        session.execute(
            delete(PF_POSITIONS).where(
                PF_POSITIONS.c.portfolio_id == portfolio_id,
                PF_POSITIONS.c.ticker == ticker,
            )
        ),
    )
    return result.rowcount > 0


def clear_positions(session: Session, portfolio_id: uuid.UUID) -> int:
    """Delete every position in ``portfolio_id``. Returns the count removed."""
    result = cast(
        CursorResult,
        session.execute(
            delete(PF_POSITIONS).where(PF_POSITIONS.c.portfolio_id == portfolio_id)
        ),
    )
    return result.rowcount
