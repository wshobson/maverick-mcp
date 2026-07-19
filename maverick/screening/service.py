"""Screening business logic. Fourth layer: imports data, screens, config, and types."""

import asyncio
from collections.abc import Callable
from datetime import date, timedelta
from time import perf_counter

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

    async def run_screen(self, screen: ScreenName) -> ScreenRun:
        screen_fn = _SCREEN_FUNCS.get(screen)
        if screen_fn is None:
            raise ValueError(f"Unknown screen: {screen!r}")

        await self._ensure_schema()
        start_time = perf_counter()

        symbols = await asyncio.to_thread(self._universe_fn)
        symbols = symbols[: self._settings.universe_max]

        semaphore = asyncio.Semaphore(_HISTORY_CONCURRENCY)
        today = date.today()
        history_start = today - timedelta(days=_HISTORY_WINDOW_DAYS)

        async def _score(symbol: str) -> ScreeningResult | None:
            async with semaphore:
                try:
                    frame = await self._market_data.get_price_history(
                        symbol, history_start, None
                    )
                except Exception:
                    logger.warning(
                        "screening.%s: failed to fetch history for %s, skipping",
                        screen,
                        symbol,
                        exc_info=True,
                    )
                    return None
            return screen_fn(symbol, frame, self._settings)

        scored = await asyncio.gather(*(_score(symbol) for symbol in symbols))
        qualifying = [result for result in scored if result is not None]

        date_analyzed = today.isoformat()

        def _persist() -> None:
            with session_scope(self._session_factory) as session:
                replace_screen_snapshot(session, screen, date_analyzed, qualifying)

        await asyncio.to_thread(_persist)

        return ScreenRun(
            screen=screen,
            symbols_screened=len(symbols),
            symbols_qualified=len(qualifying),
            date_analyzed=date_analyzed,
            duration_seconds=perf_counter() - start_time,
        )

    async def run_all_screens(self) -> dict[ScreenName, ScreenRun]:
        return {screen: await self.run_screen(screen) for screen in _ALL_SCREENS}
