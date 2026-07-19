"""Technical business logic. Fourth layer: imports indicators, analysis, config, and types.

`TechnicalService` owns frame preparation: it fetches OHLCV history from the
injected `MarketDataService`, computes every indicator via `indicators.py`,
merges the results onto the OHLCV frame under the column names documented as
the "prepared DataFrame" contract in `analysis.py`'s module docstring
(including the macd `signal`/`histogram` -> `macd_signal`/`macd_hist`
renames), and only then calls the appropriate `analyze_*` function(s).

Every method guards against two related failure shapes before calling any
`analyze_*` function: an empty fetch (no price history at all) and a "NaN
tail" (the most recent bar is missing data). Both raise `ValueError` naming
the ticker; callers never see a partially-valid typed payload built over
missing data. The guard always checks the raw `Close` column specifically,
not just each method's own indicator column(s): EMA/Wilder-smoothed
indicators (`rsi`, `macd`) can carry a *stale* non-NaN value forward through
a NaN input (pandas `ewm` skips NaN inputs rather than propagating them),
while rolling-window indicators (`sma`, `bollinger`, `stochastic`) reliably
NaN out. Checking `Close` directly closes that gap uniformly instead of
depending on each indicator's own NaN-propagation quirks.

This guard is also what keeps `analyze_trend`'s NaN-safe scoring quirk from
leaking into service responses: with too little history, `sma_long` (the
longest warmup of any trend input, at `settings.sma_long_period` bars) stays
NaN, and `get_trend`/`get_full_analysis` raise before `analyze_trend` ever
runs -- so a resting score of 0 (which reads as "bearish", see
`analyze_trend`'s docstring) never gets constructed from missing data and
returned as if it were a real bearish reading. `get_full_analysis` uses this
same `sma_long` check, which also guarantees `Close` is real before
`FullTechnicalAnalysis.current_price` (a non-optional field) is built.
"""

import asyncio
from datetime import date, timedelta
from typing import Any

import pandas as pd

from maverick.market_data.service import MarketDataService
from maverick.technical import indicators
from maverick.technical.analysis import (
    analyze_bollinger,
    analyze_macd,
    analyze_rsi,
    analyze_stochastic,
    analyze_trend,
    analyze_volume,
    generate_outlook,
    support_resistance,
)
from maverick.technical.config import TechnicalSettings, get_technical_settings
from maverick.technical.types import (
    BollingerAnalysis,
    FullTechnicalAnalysis,
    LevelsResult,
    MACDAnalysis,
    RSIAnalysis,
    StochasticAnalysis,
    TrendAnalysis,
    VolumeAnalysis,
)

# Mirrors `maverick.screening.service`'s `_HISTORY_WINDOW_DAYS`: 400 calendar
# days comfortably covers the 200-bar `sma_long` warmup (the longest of any
# indicator this domain computes) even through weekends/holidays. Every
# fetch uses at least this many calendar days regardless of the caller's
# `days` override, which only *extends* the window, never shrinks it below
# the warmup floor.
_MIN_HISTORY_CALENDAR_DAYS = 400


def _prepare_frame(raw: pd.DataFrame, settings: TechnicalSettings) -> pd.DataFrame:
    """Merge every indicator this domain computes onto `raw` OHLCV, renamed
    per the "prepared DataFrame" contract documented in `analysis.py`."""
    if raw.empty:
        return raw

    df = raw.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["rsi"] = indicators.rsi(close, settings.rsi_period)

    macd_df = indicators.macd(
        close,
        settings.macd_fast_period,
        settings.macd_slow_period,
        settings.macd_signal_period,
    )
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["signal"]
    df["macd_hist"] = macd_df["histogram"]

    stoch_df = indicators.stochastic(
        high,
        low,
        close,
        settings.stoch_k_period,
        settings.stoch_d_period,
        settings.stoch_smooth_k,
    )
    df["stoch_k"] = stoch_df["k"]
    df["stoch_d"] = stoch_df["d"]

    bb_df = indicators.bollinger(
        close, settings.bollinger_period, settings.bollinger_std
    )
    df["bb_mid"] = bb_df["mid"]
    df["bb_upper"] = bb_df["upper"]
    df["bb_lower"] = bb_df["lower"]

    df["sma_short"] = indicators.sma(close, settings.sma_short_period)
    df["sma_long"] = indicators.sma(close, settings.sma_long_period)
    df["ema"] = indicators.ema(close, settings.ema_period)
    df["adx"] = indicators.adx(high, low, close, settings.adx_period)

    return df


def _require(
    df: pd.DataFrame, ticker: str, columns: tuple[str, ...], label: str
) -> None:
    """Raise a clear `ValueError` unless every column in `columns` has a
    non-NaN value on the most recent bar -- the "insufficient history (or
    NaN tail)" guard every service method applies before analyzing.
    `columns` always includes `"Close"` at the call sites below; see this
    module's docstring for why that specific check is load-bearing."""
    incomplete = df.empty or any(
        column not in df.columns or pd.isna(df[column].iloc[-1]) for column in columns
    )
    if incomplete:
        raise ValueError(
            f"Insufficient price history for {ticker!r} to compute {label}: "
            "not enough bars were fetched, or the most recent bar is missing "
            "data."
        )


class TechnicalService:
    """Domain service: fetches price history via the injected
    `MarketDataService`, prepares indicator-annotated frames (see
    `_prepare_frame`), and runs the pure rubrics in `analysis.py`.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        settings: TechnicalSettings | None = None,
    ) -> None:
        self._market_data = market_data
        self._settings = settings or get_technical_settings()

    @property
    def settings(self) -> TechnicalSettings:
        return self._settings

    async def _run(self, coro: Any) -> Any:
        """Apply `settings.analysis_timeout_seconds` to `coro`, translating
        a timeout into the same clear `ValueError` shape every other
        service failure uses."""
        try:
            return await asyncio.wait_for(
                coro, timeout=self._settings.analysis_timeout_seconds
            )
        except TimeoutError as exc:
            raise ValueError(
                "Technical analysis timed out after "
                f"{self._settings.analysis_timeout_seconds}s"
            ) from exc

    async def _prepared_frame(
        self, ticker: str, days: int | None, settings: TechnicalSettings
    ) -> pd.DataFrame:
        """Fetch a calendar-padded history window and merge every indicator
        column onto it (see `_prepare_frame`)."""
        end = date.today()
        requested = days if days is not None else settings.default_days
        calendar_days = max(requested, _MIN_HISTORY_CALENDAR_DAYS)
        start = end - timedelta(days=calendar_days)
        raw = await self._market_data.get_price_history(ticker, start, end)
        return _prepare_frame(raw, settings)

    # -- single-indicator getters ------------------------------------------

    async def get_rsi(
        self, ticker: str, days: int | None = None, period: int | None = None
    ) -> RSIAnalysis:
        settings = self._settings
        if period is not None:
            settings = settings.model_copy(update={"rsi_period": period})

        async def _impl() -> RSIAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "rsi"), "RSI")
            return analyze_rsi(df, settings)

        return await self._run(_impl())

    async def get_macd(
        self,
        ticker: str,
        days: int | None = None,
        fast_period: int | None = None,
        slow_period: int | None = None,
        signal_period: int | None = None,
    ) -> MACDAnalysis:
        overrides: dict[str, int] = {}
        if fast_period is not None:
            overrides["macd_fast_period"] = fast_period
        if slow_period is not None:
            overrides["macd_slow_period"] = slow_period
        if signal_period is not None:
            overrides["macd_signal_period"] = signal_period
        settings = (
            self._settings.model_copy(update=overrides) if overrides else self._settings
        )

        async def _impl() -> MACDAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "macd", "macd_signal", "macd_hist"), "MACD")
            return analyze_macd(df, settings)

        return await self._run(_impl())

    async def get_bollinger(
        self,
        ticker: str,
        days: int | None = None,
        period: int | None = None,
        std: float | None = None,
    ) -> BollingerAnalysis:
        overrides: dict[str, Any] = {}
        if period is not None:
            overrides["bollinger_period"] = period
        if std is not None:
            overrides["bollinger_std"] = std
        settings = (
            self._settings.model_copy(update=overrides) if overrides else self._settings
        )

        async def _impl() -> BollingerAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(
                df,
                ticker,
                ("Close", "bb_mid", "bb_upper", "bb_lower"),
                "Bollinger Bands",
            )
            return analyze_bollinger(df, settings)

        return await self._run(_impl())

    async def get_stochastic(
        self,
        ticker: str,
        days: int | None = None,
        k_period: int | None = None,
        d_period: int | None = None,
        smooth_k: int | None = None,
    ) -> StochasticAnalysis:
        overrides: dict[str, int] = {}
        if k_period is not None:
            overrides["stoch_k_period"] = k_period
        if d_period is not None:
            overrides["stoch_d_period"] = d_period
        if smooth_k is not None:
            overrides["stoch_smooth_k"] = smooth_k
        settings = (
            self._settings.model_copy(update=overrides) if overrides else self._settings
        )

        async def _impl() -> StochasticAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "stoch_k", "stoch_d"), "Stochastic")
            return analyze_stochastic(df, settings)

        return await self._run(_impl())

    async def get_trend(self, ticker: str, days: int | None = None) -> TrendAnalysis:
        settings = self._settings

        async def _impl() -> TrendAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "sma_long"), "trend")
            return analyze_trend(df, settings)

        return await self._run(_impl())

    async def get_support_resistance(
        self, ticker: str, days: int | None = None
    ) -> LevelsResult:
        settings = self._settings

        async def _impl() -> LevelsResult:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "High", "Low"), "support/resistance levels")
            return support_resistance(df, settings)

        return await self._run(_impl())

    async def get_volume(self, ticker: str, days: int | None = None) -> VolumeAnalysis:
        settings = self._settings

        async def _impl() -> VolumeAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "Volume"), "volume")
            return analyze_volume(df, settings)

        return await self._run(_impl())

    # -- composite ------------------------------------------------------------

    async def get_full_analysis(
        self, ticker: str, days: int | None = None
    ) -> FullTechnicalAnalysis:
        settings = self._settings

        async def _impl() -> FullTechnicalAnalysis:
            df = await self._prepared_frame(ticker, days, settings)
            _require(df, ticker, ("Close", "sma_long"), "full technical analysis")

            trend = analyze_trend(df, settings)
            rsi = analyze_rsi(df, settings)
            macd = analyze_macd(df, settings)
            stochastic = analyze_stochastic(df, settings)
            bollinger = analyze_bollinger(df, settings)
            volume = analyze_volume(df, settings)
            levels = support_resistance(df, settings)
            outlook = generate_outlook(trend, rsi, macd, stochastic)

            return FullTechnicalAnalysis(
                ticker=ticker.upper(),
                current_price=round(float(df["Close"].iloc[-1]), 2),
                trend=trend,
                outlook=outlook,
                rsi=rsi,
                macd=macd,
                stochastic=stochastic,
                bollinger=bollinger,
                volume=volume,
                levels=levels,
                analysis_metadata={
                    "bars_analyzed": len(df),
                    "as_of": df.index[-1].strftime("%Y-%m-%d"),
                },
            )

        return await self._run(_impl())
