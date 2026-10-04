"""Market data business logic. Fourth layer: imports data, fetchers, config, and types."""

import asyncio
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from maverick.market_data.config import MarketDataSettings, get_market_data_settings
from maverick.market_data.data import (
    METADATA,
    commit_history_snapshot,
    lock_history_state,
    read_price_range,
    reserve_history_generation,
)
from maverick.market_data.fetchers import MoverFetcher, YFinanceFetcher, info_price
from maverick.market_data.types import (
    PRICE_COLUMNS,
    CompanyInfo,
    Fundamentals,
    HistoryRefresh,
    IndexQuote,
    MarketNumbers,
    MarketOverview,
    Mover,
    Quote,
    TradingStats,
    Volatility,
    fear_level_from_vix,
)
from maverick.platform.cache import Cache, generate_cache_key
from maverick.platform.db import ensure_schema, read_only_session_scope, session_scope

_MOVER_KINDS = ("gainers", "losers", "most_active")
_DEFAULT_HISTORY_LOOKBACK_DAYS = 365
_HISTORY_REFRESH_INTERVAL = timedelta(hours=24)
_MARKET_TIMEZONE = ZoneInfo("America/New_York")
_CALENDAR_BY_SUFFIX = {
    "L": "LSE",
    "T": "JPX",
    "TO": "TSX",
    "AX": "ASX",
    "HK": "HKEX",
    "DE": "XETR",
}


def _default_calendar(symbol: str) -> Any:
    """Resolve supported Yahoo exchange suffixes; US/class-share symbols use NYSE."""
    import pandas_market_calendars as mcal

    _, separator, suffix = symbol.rpartition(".")
    if not separator or suffix in {"A", "B"}:
        return mcal.get_calendar("NYSE")
    if suffix not in _CALENDAR_BY_SUFFIX:
        raise ValueError(
            f"Unsupported daily-history calendar for {symbol} (.{suffix}); "
            "a matching trading calendar is required. Quotes and fundamentals "
            "remain available."
        )
    return mcal.get_calendar(_CALENDAR_BY_SUFFIX[suffix])


def _to_plain_date(value: Any) -> date:
    """Normalize a calendar-schedule index entry to a tz-naive ``date``."""
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_localize(None)
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # A plain-callable calendar could hand back any date-like value; a real
    # NaT can't occur here (calendar schedules never contain one), so the
    # cast just resolves pandas' broader stub return type.
    return cast(date, pd.Timestamp(value).date())


def _quote_from_info(symbol: str, info: dict[str, Any]) -> Quote:
    price = info_price(info)
    if price is None:
        raise ValueError(
            f"No quote data for {symbol}; it may be delisted or not a valid ticker."
        )
    prev_close = info.get("previousClose") or price
    change = price - prev_close
    change_percent = (change / prev_close * 100) if prev_close else 0.0
    volume = info.get("volume") or info.get("regularMarketVolume") or 0
    return Quote(
        symbol=symbol,
        price=float(price),
        change=float(change),
        change_percent=float(change_percent),
        volume=int(volume),
        timestamp=datetime.now(UTC).isoformat(),
    )


def _fundamentals_from_info(symbol: str, info: dict[str, Any]) -> Fundamentals:
    """Map `yf.info` into `Fundamentals`, mirroring the legacy get_stock_info shape."""
    return Fundamentals(
        symbol=symbol,
        company=CompanyInfo(
            name=info.get("longName", info.get("shortName")),
            sector=info.get("sector"),
            industry=info.get("industry"),
            website=info.get("website"),
            description=info.get("longBusinessSummary"),
        ),
        market_data=MarketNumbers(
            current_price=info.get("currentPrice", info.get("regularMarketPrice")),
            market_cap=info.get("marketCap"),
            enterprise_value=info.get("enterpriseValue"),
            shares_outstanding=info.get("sharesOutstanding"),
            float_shares=info.get("floatShares"),
        ),
        valuation={
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
        },
        financials={
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
        },
        trading=TradingStats(
            avg_volume=info.get("averageVolume"),
            avg_volume_10d=info.get("averageVolume10days"),
            beta=info.get("beta"),
            week_52_high=info.get("fiftyTwoWeekHigh"),
            week_52_low=info.get("fiftyTwoWeekLow"),
        ),
    )


def _mover_from_dict(item: dict[str, Any]) -> Mover:
    return Mover(
        symbol=item["symbol"],
        price=float(item.get("price") or 0.0),
        change=float(item.get("change") or 0.0),
        change_percent=float(item.get("change_percent") or 0.0),
        volume=int(item.get("volume") or 0),
    )


def _index_quote_from_frame(
    symbol: str, name: str, frame: pd.DataFrame | None
) -> IndexQuote | None:
    """Build an `IndexQuote` from a >=2-row `Close` history, or `None` if unavailable."""
    if frame is None or frame.empty or len(frame) < 2 or "Close" not in frame.columns:
        return None
    prev_close = float(frame["Close"].iloc[0])
    current = float(frame["Close"].iloc[-1])
    change = current - prev_close
    change_percent = (change / prev_close * 100) if prev_close else 0.0
    return IndexQuote(
        name=name,
        symbol=symbol,
        price=round(current, 2),
        change=round(change, 2),
        change_percent=round(change_percent, 2),
    )


class MarketDataService:
    """Domain service: price history, quotes, fundamentals, and market overview.

    Owns the `md_stocks`/`md_price_bars` schema (created lazily on
    construction) and the trading-day-aware price-history cache. Short-TTL
    payloads (quotes, market overview) go through the injected `platform.
    cache.Cache`; price bars go through `platform.db` directly since they
    need permanent, range-queryable storage rather than a TTL blob cache.
    """

    def __init__(
        self,
        engine: Engine,
        cache: Cache,
        yf: YFinanceFetcher,
        movers: MoverFetcher,
        settings: MarketDataSettings | None = None,
        calendar: Any = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._yf = yf
        self._movers = movers
        self._settings = settings or get_market_data_settings()
        self._calendar = calendar
        self._clock = clock or (lambda: datetime.now(UTC))
        self._session_factory = sessionmaker(bind=engine)
        ensure_schema(engine, METADATA)

    @property
    def settings(self) -> MarketDataSettings:
        return self._settings

    # -- price history --------------------------------------------------

    def _trading_days(self, start: date, end: date, calendar: Any) -> list[date]:
        """Resolve exchange trading days in `[start, end]` via the chosen calendar.

        Accepts either an object exposing `schedule(start_date, end_date) ->
        DataFrame` (the real pandas-market-calendars shape, and its fakes)
        or a plain callable returning an iterable of trading days.
        """
        schedule_fn = getattr(calendar, "schedule", None)
        if schedule_fn is not None:
            schedule = schedule_fn(start_date=start, end_date=end)
            raw_days = list(schedule.index)
        else:
            raw_days = list(calendar(start, end))
        return sorted({_to_plain_date(day) for day in raw_days})

    def _read_range(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        with read_only_session_scope(self._session_factory) as session:
            return read_price_range(session, symbol, start, end)

    def _prepare_history(
        self, symbol: str, requested: list[date], now: datetime, today: date
    ) -> HistoryRefresh | None:
        with session_scope(self._session_factory, sqlite_immediate=True) as session:
            state = lock_history_state(session, symbol)
            stale = state.refreshed_at is None or not (
                timedelta(0) <= now - state.refreshed_at < _HISTORY_REFRESH_INTERVAL
            )
            missing = not set(requested) <= set(state.dates)
            current = today in requested or today in state.dates
            if not (stale or missing or current):
                return None
            generation = reserve_history_generation(session, state.stock_id)
            return HistoryRefresh(
                generation,
                min((requested[0], *state.dates)),
                max((requested[-1], *state.dates)),
                state.dates,
            )

    @staticmethod
    def _validate_history(
        symbol: str,
        frame: pd.DataFrame,
        refresh: HistoryRefresh,
        sessions: list[date],
        today: date,
    ) -> None:
        """Require all stored dates and the listed session suffix on one basis.

        Leading gaps can precede listing. Missing stored/interior/completed
        trailing sessions are partial responses; today's uncompleted bar may
        be absent, but an already stored current bar cannot disappear.
        """
        message = (
            f"Incomplete price history refresh for {symbol}; cached snapshot preserved"
        )
        if frame.empty or not set(PRICE_COLUMNS) <= set(frame.columns):
            raise ValueError(message)
        dates = [_to_plain_date(ts) for ts in frame.index]
        supplied = set(dates)
        first = min(dates)
        required = set(refresh.cached_dates) | {
            day for day in sessions if first <= day < today
        }
        if (
            len(supplied) != len(dates)
            or not required <= supplied
            or not supplied <= set(sessions) | set(refresh.cached_dates)
        ):
            raise ValueError(message)
        try:
            finite = all(
                math.isfinite(float(value))
                for name in PRICE_COLUMNS
                for value in frame[name]
            )
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(message)

    def _commit_history(
        self,
        symbol: str,
        refresh: HistoryRefresh,
        frame: pd.DataFrame,
        refreshed_at: datetime | None,
    ) -> bool:
        with session_scope(self._session_factory, sqlite_immediate=True) as session:
            return commit_history_snapshot(
                session, symbol, refresh.generation, frame, refreshed_at
            )

    async def get_price_history(
        self, symbol: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        """Refresh full adjusted snapshots on expansion, provisional bars, or 24h age.

        Reserve a generation before provider I/O, then commit only a complete
        response that has not been superseded. Failures leave the prior bars
        and freshness intact. A current market-date bar stays provisional
        until a later date's refresh, conservatively including after hours.
        """
        symbol = symbol.upper()
        now = self._clock().astimezone(UTC)
        calendar = (
            self._calendar
            if self._calendar is not None
            else await asyncio.to_thread(_default_calendar, symbol)
        )
        today = now.astimezone(getattr(calendar, "tz", _MARKET_TIMEZONE)).date()
        resolved_end = end or today
        resolved_start = start or (
            resolved_end - timedelta(days=_DEFAULT_HISTORY_LOOKBACK_DAYS)
        )
        requested = (
            await asyncio.to_thread(
                self._trading_days, resolved_start, min(resolved_end, today), calendar
            )
            if resolved_start <= min(resolved_end, today)
            else []
        )
        if requested:
            refresh = await asyncio.to_thread(
                self._prepare_history, symbol, requested, now, today
            )
            if refresh is not None:
                sessions = await asyncio.to_thread(
                    self._trading_days, refresh.start, refresh.end, calendar
                )
                frame = await self._yf.history(
                    symbol, refresh.start, refresh.end + timedelta(days=1)
                )
                if frame.empty and not refresh.cached_dates:
                    # An entirely pre-listing range can legitimately be empty.
                    # Leave freshness unset so later requests can retry.
                    return await asyncio.to_thread(
                        self._read_range, symbol, resolved_start, resolved_end
                    )
                self._validate_history(symbol, frame, refresh, sessions, today)
                accepted = await asyncio.to_thread(
                    self._commit_history,
                    symbol,
                    refresh,
                    frame,
                    None if today in sessions else now,
                )
                if not accepted:
                    raise ValueError(
                        "Price history refresh superseded by a newer request; retry"
                    )
        return await asyncio.to_thread(
            self._read_range, symbol, resolved_start, resolved_end
        )

    # -- quotes -----------------------------------------------------------

    async def get_quote(self, symbol: str) -> Quote:
        symbol = symbol.upper()
        cache_key = generate_cache_key("md_quote", symbol=symbol)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return Quote(**cached)

        info = await self._yf.info(symbol)
        quote = _quote_from_info(symbol, info)
        await self._cache.set(
            cache_key, quote.model_dump(), ttl=self._settings.quote_ttl_seconds
        )
        return quote

    # -- fundamentals -------------------------------------------------------

    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        symbol = symbol.upper()
        info = await self._yf.info(symbol)
        return _fundamentals_from_info(symbol, info)

    # -- market overview ------------------------------------------------------

    async def get_indices_summary(self) -> dict[str, IndexQuote]:
        symbols = list(self._settings.indices.keys())
        frames = await self._yf.batch_history(symbols, period="2d")
        result: dict[str, IndexQuote] = {}
        for symbol, name in self._settings.indices.items():
            quote = _index_quote_from_frame(symbol, name, frames.get(symbol))
            if quote is not None:
                result[symbol] = quote
        return result

    async def get_sector_performance(self) -> dict[str, float]:
        """Sector performance keyed by sector *name* (settings.sector_etfs is symbol-keyed)."""
        symbols = list(self._settings.sector_etfs.keys())
        frames = await self._yf.batch_history(symbols, period="2d")
        result: dict[str, float] = {}
        for symbol, name in self._settings.sector_etfs.items():
            quote = _index_quote_from_frame(symbol, name, frames.get(symbol))
            if quote is not None:
                result[name] = quote.change_percent
        return result

    async def get_movers(self, kind: str, limit: int) -> list[Mover]:
        if kind not in _MOVER_KINDS:
            raise ValueError(f"Unknown mover kind: {kind!r}")
        raw = await getattr(self._movers, kind)(limit)
        return [_mover_from_dict(item) for item in raw]

    async def get_market_overview(self) -> MarketOverview:
        """Compose indices, sectors, and movers; VIX read explicitly from `^VIX`.

        The legacy tool read `change_percent` off the wrong dict and always
        reported "low" fear. This reads the `^VIX` entry from the indices
        summary explicitly, so `fear_level` reflects the actual VIX level.
        """
        cache_key = generate_cache_key("md_market_overview")
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return MarketOverview(**cached)

        indices = await self.get_indices_summary()
        sectors = await self.get_sector_performance()
        limit = self._settings.mover_limit_default
        top_gainers = await self.get_movers("gainers", limit)
        top_losers = await self.get_movers("losers", limit)

        vix_quote = indices.get("^VIX")
        vix = vix_quote.price if vix_quote is not None else None
        vix_change_percent = vix_quote.change_percent if vix_quote is not None else None

        overview = MarketOverview(
            indices=indices,
            sectors=sectors,
            top_gainers=top_gainers,
            top_losers=top_losers,
            volatility=Volatility(
                vix=vix,
                vix_change_percent=vix_change_percent,
                fear_level=fear_level_from_vix(vix),
            ),
            last_updated=datetime.now(UTC).isoformat(),
        )

        await self._cache.set(
            cache_key, overview.model_dump(), ttl=self._settings.overview_ttl_seconds
        )
        return overview

    # -- cache management -------------------------------------------------

    async def clear_cache(self, symbol: str | None = None) -> int:
        """Clear the quote cache for `symbol`, or every quote plus the market
        overview cache when `symbol` is `None`. Returns the number of cache
        entries removed."""
        if symbol is not None:
            pattern = generate_cache_key("md_quote", symbol=symbol.upper()) + "*"
            return await self._cache.delete_pattern(pattern)

        quote_pattern = generate_cache_key("md_quote", symbol="*")
        quote_count = await self._cache.delete_pattern(quote_pattern)
        overview_count = await self._cache.delete_pattern(
            generate_cache_key("md_market_overview")
        )
        return quote_count + overview_count
