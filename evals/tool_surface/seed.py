"""Build one SQLite database per data state, offline, through maverick's own
data functions and services. No network access and no model calls.

- empty: every domain's schema, no rows.
- seeded: "My Portfolio" (5 positions), nine symbols for the screener to
  screen, the "Tech leaders" watchlist, and four journal trades.
- edge: the same rows as seeded; its cases differ in the query, not the data.

Positions are written through the portfolio data layer and ledger rather
than `PortfolioService.add_position`, which looks up each sector over the
network. The screener's universe is the symbols in `md_stocks`, so the nine
symbols are registered there through the market-data data layer, with no
price bars and no screening snapshot: a trace that runs the screens computes
them from live data.
"""

import asyncio
from decimal import Decimal
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from maverick.market_data import data as market_data
from maverick.platform.config import DatabaseSettings
from maverick.platform.db import (
    create_engine_from_settings,
    ensure_schema,
    session_scope,
)
from maverick.portfolio import data as portfolio_data
from maverick.portfolio import journal, service_watchlist, watchlist
from maverick.portfolio.ledger import add_shares
from maverick.portfolio.service_journal import JournalService
from maverick.screening import data as screening_data

STATES = ("empty", "seeded", "edge")

# ticker, shares, average cost, purchase date, sector
POSITIONS = (
    ("AAPL", "50", "198.40", "2025-11-14", "Technology"),
    ("MSFT", "30", "412.10", "2025-12-03", "Technology"),
    ("NVDA", "120", "131.75", "2025-10-21", "Technology"),
    ("JPM", "40", "244.60", "2026-02-10", "Financial Services"),
    ("XOM", "60", "112.30", "2026-03-18", "Energy"),
)
WATCHLIST = ("Tech leaders", "Large-cap tech I follow", ("AAPL", "MSFT", "NVDA"))
# symbol, entry price, shares, entry date, tag, rationale, (exit price, exit date)
TRADES = (
    ("TSLA", "238.50", "20", "2026-08-11", "breakout", "Base breakout on volume", None),
    ("TSLA", "262.00", "15", "2026-09-08", "pullback", "Added on the 20-day", None),
    ("AMD", "161.20", "40", "2026-09-02", "momentum", "Relative strength leader", None),
    (
        "NVDA",
        "118.00",
        "50",
        "2026-06-15",
        "breakout",
        "Earnings gap hold",
        ("131.40", "2026-08-20"),
    ),
)
# The screener's universe: registered in `md_stocks`, never priced here.
SCREENING_SYMBOLS = (
    "NVDA",
    "AVGO",
    "META",
    "NFLX",
    "PLTR",
    "ANET",
    "INTC",
    "NKE",
    "CVS",
)


def _write_rows(engine: Engine) -> None:
    factory = sessionmaker(bind=engine)
    with session_scope(factory) as session:
        portfolio_id = portfolio_data.get_or_create_portfolio(
            session, "default", "My Portfolio"
        )
        for ticker, shares, cost, bought, sector in POSITIONS:
            position = add_shares(
                None, ticker, Decimal(shares), Decimal(cost), bought, None, sector
            )
            portfolio_data.upsert_position(session, portfolio_id, position)
        for symbol in SCREENING_SYMBOLS:
            market_data.get_or_create_stock(session, symbol)


async def _write_services(engine: Engine) -> None:
    factory = sessionmaker(bind=engine)
    name, description, symbols = WATCHLIST
    created = await service_watchlist.create_watchlist(
        engine, factory, name, description
    )
    for symbol in symbols:
        await service_watchlist.add_item(engine, factory, created.id, symbol, None)

    journal_service = JournalService(engine)
    for symbol, price, shares, entered, tag, rationale, close in TRADES:
        trade = await journal_service.add_trade(
            symbol, "long", Decimal(price), Decimal(shares), entered, rationale, [tag]
        )
        if close is not None:
            await journal_service.close_trade(trade.id, Decimal(close[0]), close[1])


def build(state: str, path: Path) -> Path:
    """Create the database for `state` at `path` and return `path`."""
    if state not in STATES:
        raise ValueError(f"unknown data state {state!r}; expected one of {STATES}")
    engine = create_engine_from_settings(DatabaseSettings(url=f"sqlite:///{path}"))
    try:
        for metadata in (
            market_data.METADATA,
            screening_data.METADATA,
            portfolio_data.METADATA,
            watchlist.METADATA,
            journal.METADATA,
        ):
            ensure_schema(engine, metadata)
        if state != "empty":
            _write_rows(engine)
            asyncio.run(_write_services(engine))
    finally:
        engine.dispose()
    return path


def build_all(directory: Path) -> dict[str, Path]:
    """Build every data state's database under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    return {state: build(state, directory / f"{state}.db") for state in STATES}
