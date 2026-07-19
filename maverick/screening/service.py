"""Screening business logic. Fourth layer: imports data, screens, config, and types."""

import asyncio
from collections.abc import Callable
from datetime import date, timedelta
from time import perf_counter

import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from maverick.market_data.data import list_symbols
from maverick.market_data.service import MarketDataService
from maverick.platform.db import ensure_schema, read_only_session_scope, session_scope
from maverick.platform.telemetry import get_logger
from maverick.screening.config import ScreeningSettings, get_screening_settings
from maverick.screening.data import (
    METADATA,
    read_by_criteria,
    read_latest_all,
    read_top,
    replace_screen_snapshot,
)
from maverick.screening.screens import score_bearish, score_bullish, score_supply_demand
from maverick.screening.types import (
    AllScreeningResults,
    ScreeningCriteria,
    ScreeningResult,
    ScreenName,
    ScreenRun,
)

logger = get_logger(__name__)

_SCREEN_FUNCS = {
    "bullish": score_bullish,
    "bearish": score_bearish,
    "supply_demand": score_supply_demand,
}
_ALL_SCREENS: tuple[ScreenName, ...] = ("bullish", "bearish", "supply_demand")
_HISTORY_CONCURRENCY = 4
# The supply_demand rubric checks sma200 against its value 22 bars earlier,
# which needs sma200's own 200-bar warmup plus 21 more rows -- 221 non-NaN
# rows minimum. A 400-calendar-day window comfortably covers that even
# through weekends/holidays (roughly 270-280 NYSE trading days), so every
# screen gets a window generous enough for the strictest of the three.
_HISTORY_WINDOW_DAYS = 400


class ScreeningService:
    """Domain service: screen queries (thin reads over `data.py`) and screen
    compute (`run_screen`/`run_all_screens`, fetching price history through
    the injected `MarketDataService` and scoring it via the pure rubrics in
    `screens.py`).

    Owns the `scr_results` schema, created lazily (on first async call, not
    in `__init__`) since construction itself should never touch the database.
    """

    def __init__(
        self,
        engine: Engine,
        market_data: MarketDataService,
        settings: ScreeningSettings | None = None,
        universe_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self._engine = engine
        self._market_data = market_data
        self._settings = settings or get_screening_settings()
        self._session_factory = sessionmaker(bind=engine)
        self._universe_fn = universe_fn or self._default_universe
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    @property
    def settings(self) -> ScreeningSettings:
        return self._settings

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            await asyncio.to_thread(ensure_schema, self._engine, METADATA)
            self._schema_ready = True

    def _default_universe(self) -> list[str]:
        with read_only_session_scope(self._session_factory) as session:
            return list_symbols(session)

    def _clamp_limit(self, limit: int | None) -> int:
        resolved = limit if limit is not None else self._settings.default_limit
        return min(resolved, self._settings.max_limit)

    # -- query methods ----------------------------------------------------

    async def get_bullish(
        self, limit: int | None = None, min_score: int | None = None
    ) -> list[ScreeningResult]:
        await self._ensure_schema()
        resolved_limit = self._clamp_limit(limit)

        def _read() -> list[ScreeningResult]:
            with read_only_session_scope(self._session_factory) as session:
                return read_top(
                    session, "bullish", resolved_limit, min_combined_score=min_score
                )

        return await asyncio.to_thread(_read)

    async def get_bearish(
        self, limit: int | None = None, min_score: int | None = None
    ) -> list[ScreeningResult]:
        await self._ensure_schema()
        resolved_limit = self._clamp_limit(limit)

        def _read() -> list[ScreeningResult]:
            with read_only_session_scope(self._session_factory) as session:
                return read_top(
                    session, "bearish", resolved_limit, min_combined_score=min_score
                )

        return await asyncio.to_thread(_read)

    async def get_supply_demand(
        self, limit: int | None = None, min_momentum_score: float | None = None
    ) -> list[ScreeningResult]:
        await self._ensure_schema()
        resolved_limit = self._clamp_limit(limit)

        def _read() -> list[ScreeningResult]:
            with read_only_session_scope(self._session_factory) as session:
                return read_top(
                    session,
                    "supply_demand",
                    resolved_limit,
                    min_momentum_score=min_momentum_score,
                )

        return await asyncio.to_thread(_read)

    async def get_all(self) -> AllScreeningResults:
        await self._ensure_schema()

        def _read() -> AllScreeningResults:
            with read_only_session_scope(self._session_factory) as session:
                return read_latest_all(session)

        return await asyncio.to_thread(_read)

    async def get_by_criteria(
        self, criteria: ScreeningCriteria, limit: int | None = None
    ) -> list[ScreeningResult]:
        await self._ensure_schema()
        resolved_limit = self._clamp_limit(limit)

        def _read() -> list[ScreeningResult]:
            with read_only_session_scope(self._session_factory) as session:
                return read_by_criteria(session, criteria, resolved_limit)

        return await asyncio.to_thread(_read)

    # -- compute ------------------------------------------------------------

    async def _fetch_universe(
        self, symbols: list[str]
    ) -> tuple[dict[str, pd.DataFrame], int]:
        """Fetch each symbol's price history exactly once.

        Returns the frames keyed by symbol (only symbols whose fetch
        succeeded) and the count of symbols whose fetch failed.
        """
        semaphore = asyncio.Semaphore(_HISTORY_CONCURRENCY)
        today = date.today()
        history_start = today - timedelta(days=_HISTORY_WINDOW_DAYS)

        async def _fetch(symbol: str) -> tuple[str, pd.DataFrame | None]:
            async with semaphore:
                try:
                    frame = await self._market_data.get_price_history(
                        symbol, history_start, None
                    )
                except Exception:
                    logger.warning(
                        "screening: failed to fetch history for %s, skipping",
                        symbol,
                        exc_info=True,
                    )
                    return symbol, None
            return symbol, frame

        fetched = await asyncio.gather(*(_fetch(symbol) for symbol in symbols))
        frames = {symbol: frame for symbol, frame in fetched if frame is not None}
        failed = len(symbols) - len(frames)
        return frames, failed

    async def _run_sweep(
        self, screens: tuple[ScreenName, ...]
    ) -> dict[ScreenName, ScreenRun]:
        """One universe fetch, then every screen in `screens` scores each frame.

        Fetches each symbol's price history exactly once regardless of how
        many screens are requested, then applies every requested rubric to
        every fetched frame. Each screen's snapshot is gated independently
        (via `replace_screen_snapshot`) but against the same shared
        successful-fetch count -- see `_persist_if_safe`.
        """
        await self._ensure_schema()
        start_time = perf_counter()

        symbols = await asyncio.to_thread(self._universe_fn)
        symbols = symbols[: self._settings.universe_max]

        frames, failed = await self._fetch_universe(symbols)
        date_analyzed = date.today().isoformat()

        runs: dict[ScreenName, ScreenRun] = {}
        for screen in screens:
            screen_fn = _SCREEN_FUNCS[screen]
            qualifying = [
                result
                for symbol, frame in frames.items()
                if (result := screen_fn(symbol, frame, self._settings)) is not None
            ]

            await self._persist_if_safe(screen, date_analyzed, qualifying, len(frames))

            runs[screen] = ScreenRun(
                screen=screen,
                symbols_screened=len(symbols),
                symbols_qualified=len(qualifying),
                symbols_failed=failed,
                date_analyzed=date_analyzed,
                duration_seconds=perf_counter() - start_time,
            )
        return runs

    async def _persist_if_safe(
        self,
        screen: ScreenName,
        date_analyzed: str,
        qualifying: list[ScreeningResult],
        fetched_ok: int,
    ) -> None:
        """Replace `screen`'s snapshot unless the sweep fetched zero symbols.

        `replace_screen_snapshot` is destructive (delete-then-insert against
        `date_analyzed`), so a run where every fetch failed must not touch
        the table: writing zero rows for today would wipe out a prior good
        snapshot already dated today, leaving the screen with no data at
        all. Skipping preserves whatever snapshot was already persisted.
        """
        if fetched_ok == 0:
            logger.warning(
                "screening.%s: zero symbols fetched successfully, skipping "
                "snapshot replace and preserving the prior snapshot",
                screen,
            )
            return

        def _persist() -> None:
            with session_scope(self._session_factory) as session:
                replace_screen_snapshot(session, screen, date_analyzed, qualifying)

        await asyncio.to_thread(_persist)

    async def run_screen(self, screen: ScreenName) -> ScreenRun:
        if screen not in _SCREEN_FUNCS:
            raise ValueError(f"Unknown screen: {screen!r}")
        runs = await self._run_sweep((screen,))
        return runs[screen]

    async def run_all_screens(self) -> dict[ScreenName, ScreenRun]:
        return await self._run_sweep(_ALL_SCREENS)
