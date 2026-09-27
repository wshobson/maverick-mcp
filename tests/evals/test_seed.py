"""The data states `evals/tool_surface/seed.py` builds (offline, local SQLite)."""

from pathlib import Path

import pytest
from sqlalchemy import Table, func, select
from sqlalchemy.orm import sessionmaker

from evals.tool_surface import seed
from maverick.market_data.data import list_symbols
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import create_engine_from_settings, session_scope
from maverick.portfolio.data import PF_POSITIONS
from maverick.portfolio.journal import JOURNAL_ENTRIES
from maverick.portfolio.watchlist import WATCHLIST_ITEMS
from maverick.screening.data import SCR_RESULTS

SCREENING_SYMBOLS = [
    "ANET",
    "AVGO",
    "CVS",
    "INTC",
    "META",
    "NFLX",
    "NKE",
    "NVDA",
    "PLTR",
]


def _read(path: Path) -> dict[str, object]:
    engine = create_engine_from_settings(DatabaseSettings(url=f"sqlite:///{path}"))
    try:
        with session_scope(sessionmaker(bind=engine)) as session:

            def count(table: Table) -> int:
                return session.execute(select(func.count()).select_from(table)).one()[0]

            return {
                "symbols": list_symbols(session),
                "screening_rows": count(SCR_RESULTS),
                "positions": count(PF_POSITIONS),
                "watchlist_items": count(WATCHLIST_ITEMS),
                "journal_trades": count(JOURNAL_ENTRIES),
            }
    finally:
        engine.dispose()


def test_empty_state_has_no_rows(tmp_path: Path) -> None:
    state = _read(seed.build("empty", tmp_path / "empty.db"))

    assert state == {
        "symbols": [],
        "screening_rows": 0,
        "positions": 0,
        "watchlist_items": 0,
        "journal_trades": 0,
    }


@pytest.mark.parametrize("data_state", ["seeded", "edge"])
def test_seeded_states_register_a_universe_instead_of_a_snapshot(
    tmp_path: Path, data_state: str
) -> None:
    state = _read(seed.build(data_state, tmp_path / f"{data_state}.db"))

    assert state == {
        "symbols": SCREENING_SYMBOLS,
        "screening_rows": 0,
        "positions": 5,
        "watchlist_items": 3,
        "journal_trades": 4,
    }
