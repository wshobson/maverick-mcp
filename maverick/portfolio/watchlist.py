"""Persistent watchlist storage. Third layer (sibling of data/ledger):
imports types only.

Table names (`watchlists`, `watchlist_items`) are ported verbatim from the
legacy `maverick_mcp.services.watchlist.models` schema instead of this
domain's `pf_`-prefixed convention. Both stacks resolve `DATABASE_URL`/
`POSTGRES_URL` identically (`maverick.platform.config._resolve_database_url`
and the legacy `maverick_mcp.data.models` module apply the same env-var
precedence), so a deployment that has set that variable has ONE physical
database underneath old and new code. Reusing the legacy table/column names
lets existing watchlist rows survive the cutover with no migration step.

Column shapes mirror the legacy `Watchlist`/`WatchlistItem` models exactly,
including `created_at`/`updated_at` from legacy's `TimestampMixin` (`DateTime
(timezone=True) NOT NULL`, Python-side defaults only, no `server_default`)
-- required, not cosmetic: against a pre-existing legacy-shaped database
(the exact carry-over scenario this module exists for), `ensure_schema`
sees the tables already present and skips DDL, so every INSERT here must
already satisfy those NOT NULL columns or it fails outright.

Two more behaviors read as bugs but are the real, shipped semantics this
module ports faithfully:

* `watchlist_items.watchlist_id` is a plain integer, not a foreign key --
  legacy never constrained it, so adding an item under a nonexistent
  watchlist id silently succeeds here too (no existence check).
* There is no unique constraint on `(watchlist_id, symbol)` -- legacy's
  `add_to_watchlist` never deduplicated, so repeat calls for the same
  symbol create additional rows rather than updating one.

`remove_item` deviates from the legacy service in one place: it returns
whether a row was actually deleted rather than blindly reporting success,
matching this package's own `data.py.delete_position` convention.
"""

from datetime import UTC, datetime
from typing import cast

from sqlalchemy import (
    Column,
    CursorResult,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from maverick.portfolio.types import WatchlistItemPayload, WatchlistPayload

METADATA = MetaData()


def _now() -> datetime:
    return datetime.now(UTC)


# `created_at`/`updated_at` columns below mirror legacy's `TimestampMixin`
# exactly: Python-side `default`/`onupdate` callables, no `server_default`.
# SQLAlchemy Core applies these automatically on insert()/update() for any
# column not given an explicit value -- no changes needed at the call sites.

WATCHLISTS = Table(
    "watchlists",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), unique=True, nullable=False),
    Column("description", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=_now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=_now,
        onupdate=_now,
    ),
)

WATCHLIST_ITEMS = Table(
    "watchlist_items",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("watchlist_id", Integer, nullable=False, index=True),
    Column("symbol", String(10), nullable=False, index=True),
    Column("added_at", DateTime(timezone=True), nullable=False),
    Column("notes", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=_now),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        default=_now,
        onupdate=_now,
    ),
)


def _inserted_id(result: CursorResult) -> int:
    """Narrow `inserted_primary_key` (typed `Sequence | None`) to the
    autoincrement id `int` -- always present for a successful single-row
    insert against a table with an `Integer` primary key."""
    primary_key = result.inserted_primary_key
    assert primary_key is not None
    return primary_key[0]


def _find_watchlist_id_by_name(session: Session, name: str) -> int | None:
    return session.execute(
        select(WATCHLISTS.c.id).where(WATCHLISTS.c.name == name)
    ).scalar_one_or_none()


def create_watchlist(
    session: Session, name: str, description: str | None
) -> WatchlistPayload:
    """Insert a new watchlist row.

    Raises `ValueError` if `name` is already taken (`watchlists.name` is
    unique, matching legacy). Uses a savepoint so the IntegrityError only
    rolls back this insert, leaving the caller's session usable. The
    IntegrityError is narrowed the same way `data.py::get_or_create_stock`
    does: only re-raised as the friendly `ValueError` if a row with `name`
    now actually exists -- any other integrity failure (e.g. a NOT NULL
    violation from an unexpected schema) propagates as-is instead of being
    mislabeled as a duplicate name.
    """
    try:
        with session.begin_nested():
            result = cast(
                CursorResult,
                session.execute(
                    insert(WATCHLISTS).values(name=name, description=description)
                ),
            )
            session.flush()
    except IntegrityError as exc:
        if _find_watchlist_id_by_name(session, name) is not None:
            raise ValueError(f"Watchlist name {name!r} already exists") from exc
        raise

    watchlist_id = _inserted_id(result)
    return WatchlistPayload(id=watchlist_id, name=name, description=description)


def add_item(
    session: Session, watchlist_id: int, symbol: str, notes: str | None
) -> WatchlistItemPayload:
    """Insert a new watchlist item row (`symbol` is uppercased, matching
    legacy). No existence check on `watchlist_id` and no dedup on
    `(watchlist_id, symbol)` -- matches legacy: repeat calls for the same
    symbol create additional rows."""
    symbol = symbol.upper()
    added_at = datetime.now(UTC)
    result = cast(
        CursorResult,
        session.execute(
            insert(WATCHLIST_ITEMS).values(
                watchlist_id=watchlist_id,
                symbol=symbol,
                added_at=added_at,
                notes=notes,
            )
        ),
    )
    item_id = _inserted_id(result)
    return WatchlistItemPayload(
        id=item_id,
        watchlist_id=watchlist_id,
        symbol=symbol,
        added_at=added_at.isoformat(),
        notes=notes,
    )


def remove_item(session: Session, watchlist_id: int, symbol: str) -> bool:
    """Delete every row matching `(watchlist_id, symbol.upper())`. Returns
    whether any row was removed."""
    result = cast(
        CursorResult,
        session.execute(
            delete(WATCHLIST_ITEMS).where(
                WATCHLIST_ITEMS.c.watchlist_id == watchlist_id,
                WATCHLIST_ITEMS.c.symbol == symbol.upper(),
            )
        ),
    )
    return result.rowcount > 0


def _row_to_item(row) -> WatchlistItemPayload:  # noqa: ANN001
    return WatchlistItemPayload(
        id=row.id,
        watchlist_id=row.watchlist_id,
        symbol=row.symbol,
        added_at=row.added_at.isoformat() if row.added_at else None,
        notes=row.notes,
    )


def read_items(session: Session, watchlist_id: int) -> list[WatchlistItemPayload]:
    """Return every item for `watchlist_id`, ordered by insertion (`id`).
    An unknown `watchlist_id` returns an empty list, not an error -- legacy
    never checked watchlist existence before reading items either."""
    rows = session.execute(
        select(WATCHLIST_ITEMS)
        .where(WATCHLIST_ITEMS.c.watchlist_id == watchlist_id)
        .order_by(WATCHLIST_ITEMS.c.id)
    ).all()
    return [_row_to_item(row) for row in rows]
