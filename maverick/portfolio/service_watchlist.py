"""Watchlist orchestration, split out of `service.py` to stay under the
package's 500-line file-size cap (`tests/structure/test_harness_rules.py`).

Same layer as `service.py` in the layers contract (`service : service_risk :
service_watchlist` -- non-independent, siblings may import each other).
Unlike `service_risk.py` (pure orchestration over positions/prices the
caller already read), this module owns its own DB session scoping directly
-- it mirrors `service.py`'s own CRUD shape (`add_position`/
`remove_position`) instead, since watchlist rows are independent of
`pf_portfolios`/`pf_positions` and need no position read first.
"""

import asyncio
from datetime import UTC, datetime

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from maverick.market_data.service import MarketDataService
from maverick.platform.db import ensure_schema, read_only_session_scope, session_scope
from maverick.platform.telemetry import get_logger
from maverick.portfolio import watchlist
from maverick.portfolio.types import (
    WatchlistBrief,
    WatchlistBriefItem,
    WatchlistItemPayload,
    WatchlistPayload,
    WatchlistRemoveResult,
)

logger = get_logger(__name__)

_QUOTE_CONCURRENCY = 4


async def _ensure_watchlist_schema(engine: Engine) -> None:
    """Own schema-readiness check (no `service.py`-style cached flag):
    `ensure_schema` already memoizes per engine internally, so the only
    cost after the first real call is a thread hop."""
    await asyncio.to_thread(ensure_schema, engine, watchlist.METADATA)


async def create_watchlist(
    engine: Engine,
    session_factory: sessionmaker[Session],
    name: str,
    description: str | None,
) -> WatchlistPayload:
    await _ensure_watchlist_schema(engine)

    def _write() -> WatchlistPayload:
        with session_scope(session_factory) as session:
            return watchlist.create_watchlist(session, name, description)

    return await asyncio.to_thread(_write)


async def add_item(
    engine: Engine,
    session_factory: sessionmaker[Session],
    watchlist_id: int,
    symbol: str,
    notes: str | None,
) -> WatchlistItemPayload:
    await _ensure_watchlist_schema(engine)

    def _write() -> WatchlistItemPayload:
        with session_scope(session_factory) as session:
            return watchlist.add_item(session, watchlist_id, symbol, notes)

    return await asyncio.to_thread(_write)


async def remove_item(
    engine: Engine,
    session_factory: sessionmaker[Session],
    watchlist_id: int,
    symbol: str,
) -> WatchlistRemoveResult:
    await _ensure_watchlist_schema(engine)

    def _write() -> bool:
        with session_scope(session_factory) as session:
            return watchlist.remove_item(session, watchlist_id, symbol)

    removed = await asyncio.to_thread(_write)
    return WatchlistRemoveResult(
        watchlist_id=watchlist_id, symbol=symbol.upper(), removed=removed
    )


async def _fetch_quote_prices(
    market_data: MarketDataService, symbols: list[str]
) -> dict[str, float]:
    """Fetch quotes concurrently (Semaphore(4)); a failed quote is logged
    and simply absent from the returned dict -- never fatal. Mirrors
    `service.py`'s `_fetch_quote_prices`, but returns `float`: watchlist
    items carry no cost basis, so there is no Decimal P&L math to protect.
    """
    semaphore = asyncio.Semaphore(_QUOTE_CONCURRENCY)

    async def _fetch(symbol: str) -> tuple[str, float | None]:
        async with semaphore:
            try:
                quote = await market_data.get_quote(symbol)
            except Exception:
                logger.warning(
                    "watchlist: failed to fetch quote for %s, leaving price None",
                    symbol,
                    exc_info=True,
                )
                return symbol, None
        return symbol, quote.price

    fetched = await asyncio.gather(*(_fetch(symbol) for symbol in symbols))
    return {symbol: price for symbol, price in fetched if price is not None}


def _days_on_watchlist(added_at: str | None, now: datetime) -> int | None:
    if added_at is None:
        return None
    added = datetime.fromisoformat(added_at)
    if added.tzinfo is None:
        added = added.replace(tzinfo=UTC)
    return (now - added).days


async def brief(
    engine: Engine,
    session_factory: sessionmaker[Session],
    market_data: MarketDataService,
    watchlist_id: int,
) -> WatchlistBrief:
    await _ensure_watchlist_schema(engine)

    def _read() -> list[WatchlistItemPayload]:
        with read_only_session_scope(session_factory) as session:
            return watchlist.read_items(session, watchlist_id)

    items = await asyncio.to_thread(_read)
    prices = await _fetch_quote_prices(market_data, [item.symbol for item in items])
    now = datetime.now(UTC)

    brief_items = [
        WatchlistBriefItem(
            symbol=item.symbol,
            days_on_watchlist=_days_on_watchlist(item.added_at, now),
            notes=item.notes,
            current_price=prices.get(item.symbol),
        )
        for item in items
    ]
    return WatchlistBrief(
        watchlist_id=watchlist_id, count=len(brief_items), items=brief_items
    )
