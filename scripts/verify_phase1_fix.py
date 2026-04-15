#!/usr/bin/env python
"""End-to-end behavioural probe for the Phase 1 stale-data fix.

Implements Stage 2 verification step #2 from
``docs/audit/2026-04-14-mcp-audit-roadmap.md``. The audit's primary
hypothesis was that ``PriceCache.bulk_insert_price_data``'s insert-or-skip
semantics made any row with ``(stock_id, date)`` immortal — so a
provisional mid-session bar written by one call could never be corrected
by a later call, and the user saw "days-old data".

The fix is an upsert. This script proves the fix is alive **on the
running database** — not just in unit tests — by:

1. Reading the current ``close_price`` for a ticker+date from
   ``PriceCache``.
2. Clobbering it to a sentinel value (``9999.99``) via raw SQL.
3. Invoking the provider's ``get_stock_data`` path, which will
   re-fetch yfinance and upsert back.
4. Re-reading the row and asserting the sentinel is **gone**.

Exit 0 on a verified fix. Exit 1 on a failing probe (sentinel survived
→ upsert not working → bug not fixed in production).

Usage:
    uv run python scripts/verify_phase1_fix.py [--ticker AAPL]
    uv run python scripts/verify_phase1_fix.py --ticker SPY --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from sqlalchemy import text

from maverick_mcp.data.models import PriceCache, SessionLocal, Stock

_SENTINEL = 9999.99


def _find_recent_cached_row(
    session, ticker: str
) -> tuple[date, float] | None:
    """Return (date, close_price) of the most recent cached bar for the ticker, or None."""
    stock = session.query(Stock).filter(Stock.ticker_symbol == ticker.upper()).one_or_none()
    if stock is None:
        return None

    recent_cutoff = date.today() - timedelta(days=30)
    row = (
        session.query(PriceCache)
        .filter(
            PriceCache.stock_id == stock.stock_id,
            PriceCache.date >= recent_cutoff,
        )
        .order_by(PriceCache.date.desc())
        .first()
    )
    if row is None:
        return None
    return row.date, float(row.close_price)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to probe (default: AAPL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect state without clobbering — useful as a smoke check.",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()

    with SessionLocal() as session:
        existing = _find_recent_cached_row(session, ticker)
        if existing is None:
            print(
                f"SKIP: no cached rows for {ticker} in the last 30 days. "
                f"Run the provider at least once to populate PriceCache before probing."
            )
            return 0

        target_date, original_close = existing
        print(
            f"Found cached row: {ticker} @ {target_date} close={original_close}"
        )

        if args.dry_run:
            print("DRY RUN: skipping clobber + refetch.")
            return 0

        # 1. Clobber the row via raw SQL so we're bypassing the provider.
        session.execute(
            text(
                "UPDATE mcp_price_cache SET close_price = :sentinel "
                "WHERE date = :target_date "
                "AND stock_id = (SELECT stock_id FROM mcp_stocks WHERE ticker_symbol = :ticker)"
            ),
            {"sentinel": _SENTINEL, "target_date": target_date, "ticker": ticker},
        )
        session.commit()
        print(f"CLOBBERED {ticker} @ {target_date} close -> {_SENTINEL}")

        # 2. Invoke the provider with ``use_cache=False`` so the smart-cache
        #    short-circuit is bypassed and yfinance is always hit. The
        #    returned DataFrame flows back into ``_cache_price_data`` which
        #    calls ``bulk_insert_price_data`` — that is exactly the upsert
        #    path under test. Without ``use_cache=False`` we're at yfinance's
        #    mercy for whether the guard's chosen date happens to overlap
        #    with the clobbered date (and yfinance can return empty for
        #    single-day requests right at a session boundary).
        from maverick_mcp.providers.stock_data import EnhancedStockDataProvider

        provider = EnhancedStockDataProvider()
        # Request a multi-day range that includes target_date. A range of
        # at least 5 calendar days works across weekends and holidays.
        start = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
        fetched = provider.get_stock_data(
            ticker, start_date=start, end_date=end, use_cache=False
        )
        if fetched is None or fetched.empty:
            print(
                f"SKIP: yfinance returned no data for {ticker} in {start}..{end}. "
                f"Cannot verify upsert without upstream data; try again later "
                f"or pick a more-liquid ticker with --ticker."
            )
            # Restore the clobbered value so the cache isn't left poisoned
            # by a SKIP run.
            session.execute(
                text(
                    "UPDATE mcp_price_cache SET close_price = :orig "
                    "WHERE date = :target_date "
                    "AND stock_id = (SELECT stock_id FROM mcp_stocks WHERE ticker_symbol = :ticker)"
                ),
                {
                    "orig": original_close,
                    "target_date": target_date,
                    "ticker": ticker,
                },
            )
            session.commit()
            return 0

        # Persist the fetch into PriceCache — ``use_cache=False`` bypasses
        # the smart-cache on the read path but still writes through on the
        # way back (see EnhancedStockDataProvider._cache_price_data).
        from maverick_mcp.data.models import bulk_insert_price_data

        bulk_insert_price_data(session, ticker, fetched)

        # 3. Re-read the row.
        session.expire_all()
        after = _find_recent_cached_row(session, ticker)
        if after is None:
            print("FAIL: row disappeared after refetch — unexpected.")
            return 1

        _, refetched_close = after
        print(
            f"After refetch: {ticker} @ {target_date} close={refetched_close}"
        )

        if abs(refetched_close - _SENTINEL) < 1e-6:
            print(
                "FAIL: sentinel value survived the refetch. The upsert fix "
                "is NOT active in this deployment — stale data will persist."
            )
            return 1

        print(
            "OK: sentinel was overwritten by upsert. Phase 1 fix is "
            "active end-to-end."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
