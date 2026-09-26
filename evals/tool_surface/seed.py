"""Build one SQLite database per data state, offline, through maverick's own
data functions and services. No network access and no model calls.

- empty: every domain's schema, no rows.
- seeded: "My Portfolio" (5 positions), a fixture snapshot for all three
  screens, the "Tech leaders" watchlist, and four journal trades.
- edge: the same rows as seeded; its cases differ in the query, not the data.

Positions are written through the portfolio data layer and ledger rather
than `PortfolioService.add_position`, which looks up each sector over the
network. The screening rows are fixtures written through the screening data
layer; the real screener never runs.
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
from maverick.screening.screens import _build_reason
from maverick.screening.types import ScreeningResult, ScreenName

STATES = ("empty", "seeded", "edge")
SNAPSHOT_DATE = "2026-09-25"

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
# symbol, close, rsi14, volume, 30-day average volume
BULLISH = (
    ("NVDA", 181.20, 64.1, 312e6, 190e6),
    ("AVGO", 342.75, 61.8, 28e6, 21e6),
    ("META", 781.30, 58.9, 16e6, 14e6),
    ("NFLX", 1214.50, 55.2, 3.1e6, 3.4e6),
    ("PLTR", 176.40, 83.6, 95e6, 70e6),
    ("ANET", 141.90, 62.7, 9.8e6, 8.9e6),
)
# symbol, close, rsi14, macd, macd signal, volume, 30-day average volume
BEARISH = (
    ("INTC", 21.15, 27.4, -0.62, -0.41, 88e6, 61e6),
    ("NKE", 64.80, 36.9, -1.12, -0.88, 14e6, 12e6),
    ("CVS", 58.30, 38.2, -0.35, -0.47, 9e6, 10e6),
)
# The supply/demand screen keeps a symbol only when all of these hold.
TREND_FLAGS = (
    "close_above_sma150",
    "close_above_sma200",
    "sma150_above_sma200",
    "sma200_rising",
    "sma50_above_sma150",
    "sma50_above_sma200",
    "close_above_sma50",
)
# symbol, close, volume, 30-day average volume, 252-day high
SUPPLY_DEMAND = (
    ("AVGO", 342.75, 28e6, 21e6, 351.00),
    ("NVDA", 181.20, 312e6, 190e6, 195.60),
    ("ANET", 141.90, 9.8e6, 8.9e6, 149.20),
    ("PLTR", 176.40, 95e6, 70e6, 190.10),
)


def _result(
    screen: ScreenName,
    symbol: str,
    close: float,
    flags: dict[str, bool],
    score: int,
    indicators: dict[str, float | None],
    momentum: float | None = None,
) -> ScreeningResult:
    return ScreeningResult(
        symbol=symbol,
        screen=screen,
        date_analyzed=SNAPSHOT_DATE,
        close=close,
        combined_score=score,
        momentum_score=momentum,
        indicators={"close": close, **indicators},
        flags=flags,
        reason=_build_reason(screen, flags),
    )


def _bullish() -> list[ScreeningResult]:
    rows = []
    for symbol, close, rsi14, volume, avg in BULLISH:
        flags = {
            "close_above_sma50": True,
            "close_above_sma150": True,
            "close_above_sma200": True,
            "ma_aligned": True,
            "volume_surge": volume > 1.5 * avg,
            "rsi_not_overbought": rsi14 < 80,
        }
        score = 100 + 10 * flags["volume_surge"] + 10 * flags["rsi_not_overbought"]
        sma = {"sma50": close * 0.94, "sma150": close * 0.87, "sma200": close * 0.82}
        extra = {"rsi14": rsi14, "volume": volume, "avg_volume_30d": avg}
        rows.append(_result("bullish", symbol, close, flags, score, sma | extra))
    return rows


def _bearish() -> list[ScreeningResult]:
    rows = []
    for symbol, close, rsi14, macd, signal, volume, avg in BEARISH:
        flags = {
            "close_below_sma50": True,
            "close_below_sma200": True,
            "rsi_oversold": rsi14 < 30,
            "rsi_weak": 30 <= rsi14 < 40,
            "macd_bearish": macd < signal,
            "volume_decline": volume > 1.2 * avg,
            "atr_contraction": False,
        }
        weights = {"rsi_oversold": 15, "rsi_weak": 10, "macd_bearish": 15}
        score = 40 + sum(w for k, w in weights.items() if flags[k])
        score += 20 * flags["volume_decline"]
        indicators = {
            "sma50": close * 1.08,
            "sma200": close * 1.21,
            "rsi14": rsi14,
            "macd": macd,
            "macd_signal": signal,
            "volume": volume,
            "avg_volume_30d": avg,
        }
        rows.append(_result("bearish", symbol, close, flags, score, indicators))
    return rows


def _supply_demand() -> list[ScreeningResult]:
    rows = []
    for symbol, close, volume, avg, high in SUPPLY_DEMAND:
        flags = dict.fromkeys(TREND_FLAGS, True)
        flags["volume_surge"] = volume > 1.2 * avg
        flags["near_52w_high"] = close > 0.75 * high
        score = 50 + 25 * flags["volume_surge"] + 25 * flags["near_52w_high"]
        sma200 = close * 0.82
        momentum = min(100.0, max(0.0, (close / sma200 - 1) * 100 * 5))
        indicators = {
            "sma50": close * 0.94,
            "sma150": close * 0.87,
            "sma200": sma200,
            "volume": volume,
            "avg_volume_30d": avg,
            "high_252d": high,
        }
        rows.append(
            _result("supply_demand", symbol, close, flags, score, indicators, momentum)
        )
    return rows


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
        snapshots: dict[ScreenName, list[ScreeningResult]] = {
            "bullish": _bullish(),
            "bearish": _bearish(),
            "supply_demand": _supply_demand(),
        }
        for screen, rows in snapshots.items():
            screening_data.replace_screen_snapshot(session, screen, SNAPSHOT_DATE, rows)


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
