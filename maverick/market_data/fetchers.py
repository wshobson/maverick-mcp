"""External market data fetchers. Third layer: imports config and types."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pandas as pd

from maverick.market_data.config import MarketDataSettings, get_market_data_settings
from maverick.platform.config import HttpSettings
from maverick.platform.http import get_breaker, request_with_retry

_YFINANCE_RETRIES = 2
_CAPITAL_COMPANION_RETRIES = 2

HistoryFn = Callable[..., pd.DataFrame]
InfoFn = Callable[[str], dict[str, Any]]
DownloadFn = Callable[..., dict[str, pd.DataFrame]]
MoverTierFn = Callable[[str, int], list[dict[str, Any]]]
ExternalClientFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


def _default_history_fn(
    symbol: str, start: Any, end: Any, interval: str = "1d"
) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(symbol).history(start=start, end=end, interval=interval)


def _default_info_fn(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    return yf.Ticker(symbol).info


def _default_download_fn(
    symbols: list[str], period: str = "1d"
) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(symbols, period=period, group_by="ticker", threads=True)
    if len(symbols) == 1:
        return {symbols[0]: raw}
    return {
        symbol: raw[symbol]
        for symbol in symbols
        if symbol in raw.columns.get_level_values(0)
    }


def _strip_tz(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` with a tz-naive index, copying only if a tz is present."""
    index = frame.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        frame = frame.copy()
        frame.index = index.tz_localize(None)
    return frame


class YFinanceFetcher:
    """Injectable yfinance wrapper with circuit-breaker and retry resilience.

    The default `*_fn` bindings lazily `import yfinance` only when actually
    invoked, so constructing this class (or importing this module) never
    imports yfinance — tests inject fakes and never hit the real bindings.
    """

    def __init__(
        self,
        history_fn: HistoryFn | None = None,
        info_fn: InfoFn | None = None,
        download_fn: DownloadFn | None = None,
        *,
        breaker_name: str = "yfinance",
        http_settings: HttpSettings | None = None,
    ) -> None:
        self._history_fn = history_fn or _default_history_fn
        self._info_fn = info_fn or _default_info_fn
        self._download_fn = download_fn or _default_download_fn
        self._breaker_name = breaker_name
        self._http_settings = http_settings

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        breaker = get_breaker(self._breaker_name, self._http_settings)

        async def _attempt() -> Any:
            last_exc: Exception | None = None
            for attempt in range(_YFINANCE_RETRIES + 1):
                try:
                    return await asyncio.to_thread(fn, *args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= _YFINANCE_RETRIES:
                        raise
            assert last_exc is not None  # pragma: no cover - loop always raises
            raise last_exc

        return await breaker.call(_attempt)

    async def history(
        self, symbol: str, start: Any, end: Any, interval: str = "1d"
    ) -> pd.DataFrame:
        frame = await self._call(self._history_fn, symbol, start, end, interval)
        return _strip_tz(frame)

    async def batch_history(
        self, symbols: list[str], period: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        frames = await self._call(self._download_fn, symbols, period)
        return {symbol: _strip_tz(frame) for symbol, frame in frames.items()}

    async def info(self, symbol: str) -> dict[str, Any]:
        return await self._call(self._info_fn, symbol)


class MoverFetcher:
    """Three-tier mover lookup: Capital Companion -> finviz -> yfinance batch.

    Each tier is an injected callable. A raised exception or an empty result
    from a tier falls through to the next one; exhausting all tiers (or
    having none configured) returns `[]`. The Capital Companion tier is only
    attempted when both `external_client` is injected and the settings key
    is configured.
    """

    def __init__(
        self,
        external_client: ExternalClientFn | None = None,
        finviz_fn: MoverTierFn | None = None,
        batch_quote_fn: MoverTierFn | None = None,
        *,
        settings: MarketDataSettings | None = None,
    ) -> None:
        self._external_client = external_client
        self._finviz_fn = finviz_fn
        self._batch_quote_fn = batch_quote_fn
        self._settings = settings or get_market_data_settings()

    async def _from_external(self, kind: str, limit: int) -> list[dict[str, Any]]:
        if self._external_client is None:
            return []
        if self._settings.capital_companion_api_key is None:
            return []
        try:
            return await self._external_client(kind, limit)
        except Exception:
            return []

    async def _from_finviz(self, kind: str, limit: int) -> list[dict[str, Any]]:
        if self._finviz_fn is None:
            return []
        try:
            return await asyncio.to_thread(self._finviz_fn, kind, limit)
        except Exception:
            return []

    async def _from_batch_quote(self, kind: str, limit: int) -> list[dict[str, Any]]:
        if self._batch_quote_fn is None:
            return []
        try:
            return await asyncio.to_thread(self._batch_quote_fn, kind, limit)
        except Exception:
            return []

    async def _movers(self, kind: str, limit: int) -> list[dict[str, Any]]:
        for tier in (self._from_external, self._from_finviz, self._from_batch_quote):
            result = await tier(kind, limit)
            if result:
                return result
        return []

    async def gainers(self, limit: int) -> list[dict[str, Any]]:
        return await self._movers("gainers", limit)

    async def losers(self, limit: int) -> list[dict[str, Any]]:
        return await self._movers("losers", limit)

    async def most_active(self, limit: int) -> list[dict[str, Any]]:
        return await self._movers("most_active", limit)


async def fetch_capital_companion(
    client: httpx.AsyncClient, endpoint: str, api_key: str
) -> list[dict[str, Any]]:
    """Fetch a list payload from the Capital Companion API, retrying transient failures."""
    response = await request_with_retry(
        client,
        "GET",
        endpoint,
        retries=_CAPITAL_COMPANION_RETRIES,
        backoff_base=0.0,
        headers={"X-API-KEY": api_key},
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []
