"""Tests for maverick.portfolio.watchlist.

Persistence layer for the `watchlists`/`watchlist_items` tables (names
ported verbatim from the legacy schema). Uses a tmp SQLite database per
test, matching `tests/portfolio/test_data.py`'s pattern.
"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.portfolio.watchlist import (
    METADATA,
    WATCHLIST_ITEMS,
    WATCHLISTS,
    add_item,
    create_watchlist,
    read_items,
    remove_item,
)


@pytest.fixture
def factory(tmp_path):
    settings = DatabaseSettings(
        url=f"sqlite:///{tmp_path}/watchlist.db", use_pooling=True
    )
    engine = create_engine_from_settings(settings)
    ensure_schema(engine, METADATA)
    return sessionmaker(bind=engine)


# -- create_watchlist ---------------------------------------------------


def test_create_watchlist_inserts_row_and_returns_payload(factory):
    with session_scope(factory) as session:
        result = create_watchlist(session, "Tech Movers", "High-beta tech names")

    assert result.id is not None
    assert result.name == "Tech Movers"
    assert result.description == "High-beta tech names"

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(WATCHLISTS)
        ).scalar_one()
    assert count == 1


def test_create_watchlist_allows_none_description(factory):
    with session_scope(factory) as session:
        result = create_watchlist(session, "No Description", None)

    assert result.description is None


def test_create_watchlist_rejects_duplicate_name(factory):
    with session_scope(factory) as session:
        create_watchlist(session, "Dup", None)

    with pytest.raises(ValueError, match="already exists"):
        with session_scope(factory) as session:
            create_watchlist(session, "Dup", None)


def test_create_watchlist_duplicate_name_leaves_session_usable(factory):
    """The savepoint rollback on IntegrityError must not poison the whole
    session -- a subsequent operation on the same session should still work."""
    with session_scope(factory) as session:
        create_watchlist(session, "Dup", None)
        with pytest.raises(ValueError, match="already exists"):
            create_watchlist(session, "Dup", None)
        # Session is still usable for a different, valid operation.
        create_watchlist(session, "Not A Dup", None)

    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(WATCHLISTS)
        ).scalar_one()
    assert count == 2


def test_create_watchlist_not_null_violation_is_not_mislabeled_as_duplicate(factory):
    """A NOT NULL violation (here: an explicitly `None` name, bypassing the
    `str` type hint at runtime) must propagate as the original
    `IntegrityError`, not be mislabeled as "already exists" by the
    duplicate-name mapping. `_find_watchlist_id_by_name(session, None)`
    generates `WHERE name IS NULL`, which correctly matches nothing (`name`
    is NOT NULL), so the guard falls through and re-raises."""
    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            create_watchlist(session, None, "desc")  # type: ignore[arg-type]


# -- created_at/updated_at: legacy TimestampMixin carry-over --------------


def test_watchlist_columns_are_a_superset_of_legacy_timestamp_columns():
    """`created_at`/`updated_at` must be present on both tables -- legacy's
    `TimestampMixin` puts NOT NULL versions of both on every watchlist
    table. Omitting them would break inserts against a pre-existing
    legacy-shaped database (see the module docstring and
    `test_watchlist_operations_carry_over_against_a_preexisting_legacy_database`
    in `tests/portfolio/test_service.py`)."""
    watchlists_columns = {c.name for c in WATCHLISTS.columns}
    items_columns = {c.name for c in WATCHLIST_ITEMS.columns}
    assert {"created_at", "updated_at"} <= watchlists_columns
    assert {"created_at", "updated_at"} <= items_columns


def test_create_watchlist_populates_created_at_and_updated_at(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "Timestamps", None).id

    with session_scope(factory) as session:
        row = session.execute(
            select(WATCHLISTS.c.created_at, WATCHLISTS.c.updated_at).where(
                WATCHLISTS.c.id == watchlist_id
            )
        ).one()
    assert row.created_at is not None
    assert row.updated_at is not None


def test_add_item_populates_created_at_and_updated_at(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "Timestamps", None).id
        add_item(session, watchlist_id, "AAPL", None)

    with session_scope(factory) as session:
        row = session.execute(
            select(WATCHLIST_ITEMS.c.created_at, WATCHLIST_ITEMS.c.updated_at)
        ).one()
    assert row.created_at is not None
    assert row.updated_at is not None


# -- add_item: duplicate-add semantics matching legacy -------------------


def test_add_item_inserts_row_and_returns_payload(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        result = add_item(session, watchlist_id, "aapl", "Watching for breakout")

    assert result.id is not None
    assert result.watchlist_id == watchlist_id
    assert result.symbol == "AAPL"
    assert result.notes == "Watching for breakout"
    assert result.added_at is not None


def test_add_item_uppercases_symbol(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        result = add_item(session, watchlist_id, "msft", None)

    assert result.symbol == "MSFT"


def test_add_item_duplicate_symbol_creates_two_rows_not_deduped(factory):
    """Legacy `WatchlistItem` has no unique constraint on (watchlist_id,
    symbol) -- repeat adds of the same symbol are additional rows, not an
    update. This port preserves that exactly."""
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        add_item(session, watchlist_id, "AAPL", "first note")
        add_item(session, watchlist_id, "AAPL", "second note")

    with session_scope(factory) as session:
        items = read_items(session, watchlist_id)

    assert len(items) == 2
    assert {item.notes for item in items} == {"first note", "second note"}
    assert all(item.symbol == "AAPL" for item in items)


def test_add_item_succeeds_for_nonexistent_watchlist_id(factory):
    """Legacy never validated `watchlist_id` existence (no FK constraint);
    this port matches that -- an orphan item is created silently."""
    with session_scope(factory) as session:
        result = add_item(session, 999999, "AAPL", None)

    assert result.watchlist_id == 999999
    with session_scope(factory) as session:
        count = session.execute(
            select(func.count()).select_from(WATCHLIST_ITEMS)
        ).scalar_one()
    assert count == 1


# -- remove_item -----------------------------------------------------------


def test_remove_item_returns_true_and_removes_matching_row(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        add_item(session, watchlist_id, "AAPL", None)

    with session_scope(factory) as session:
        removed = remove_item(session, watchlist_id, "AAPL")

    assert removed is True
    with session_scope(factory) as session:
        assert read_items(session, watchlist_id) == []


def test_remove_item_is_case_insensitive(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        add_item(session, watchlist_id, "AAPL", None)

    with session_scope(factory) as session:
        removed = remove_item(session, watchlist_id, "aapl")

    assert removed is True


def test_remove_item_returns_false_when_no_matching_row(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id

    with session_scope(factory) as session:
        removed = remove_item(session, watchlist_id, "NOPE")

    assert removed is False


def test_remove_item_removes_all_duplicate_rows_for_that_symbol(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        add_item(session, watchlist_id, "AAPL", "one")
        add_item(session, watchlist_id, "AAPL", "two")

    with session_scope(factory) as session:
        removed = remove_item(session, watchlist_id, "AAPL")

    assert removed is True
    with session_scope(factory) as session:
        assert read_items(session, watchlist_id) == []


def test_remove_item_only_removes_from_target_watchlist(factory):
    with session_scope(factory) as session:
        watchlist_a = create_watchlist(session, "List A", None).id
        watchlist_b = create_watchlist(session, "List B", None).id
        add_item(session, watchlist_a, "AAPL", None)
        add_item(session, watchlist_b, "AAPL", None)

    with session_scope(factory) as session:
        remove_item(session, watchlist_a, "AAPL")

    with session_scope(factory) as session:
        items_a = read_items(session, watchlist_a)
        items_b = read_items(session, watchlist_b)
    assert items_a == []
    assert len(items_b) == 1


# -- read_items --------------------------------------------------------


def test_read_items_returns_empty_list_for_unknown_watchlist_id(factory):
    with session_scope(factory) as session:
        items = read_items(session, 999999)
    assert items == []


def test_read_items_orders_by_insertion(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        add_item(session, watchlist_id, "MSFT", None)
        add_item(session, watchlist_id, "AAPL", None)

    with session_scope(factory) as session:
        items = read_items(session, watchlist_id)

    assert [item.symbol for item in items] == ["MSFT", "AAPL"]


def test_read_items_preserves_none_notes(factory):
    with session_scope(factory) as session:
        watchlist_id = create_watchlist(session, "My List", None).id
        add_item(session, watchlist_id, "AAPL", None)

    with session_scope(factory) as session:
        items = read_items(session, watchlist_id)

    assert items[0].notes is None
