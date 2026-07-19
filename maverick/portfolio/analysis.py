"""Portfolio market-data-driven analyses: correlation, comparison, and
risk-adjusted position sizing.

Sits below `service.py` and does not touch persistence (`data.py`) or the
session/engine boundary at all -- every function here takes an
already-resolved, already-validated list of tickers (or a single ticker)
and fans out to `MarketDataService` and `maverick.technical.indicators`.
`service.py` owns portfolio auto-fill (reading `pf_positions` when the
caller omits explicit tickers) and the ledger-derived existing-position
P&L block, stamping both onto the results returned here via `model_copy`
after the fact -- so every function below returns a plain analysis of
exactly what it was given, nothing portfolio-aware baked in.

Two settings-driven history windows recur below and are documented once
here rather than at each call site:

* `correlation_analysis`/`compare_tickers` treat their `days` parameter as
  a *trading-day* count (it is echoed back as `period_days` and drives
  `correlation_min_rows`/ranking semantics), but
  `market_data.get_price_history` takes calendar dates. The calendar fetch
  window is padded by `settings.history_pad_calendar_days` on top of
  `days` so that even a small `days` request still has enough calendar
  days behind it to yield `days` actual trading rows; the fetched series
  is then trimmed to its most recent `days` rows via `.tail(days)`.
* `risk_adjusted_analysis` has no `days` parameter (ATR(20) just needs a
  reliable warmup), so it reuses the same `history_pad_calendar_days`
  constant as a fixed calendar-day lookback, mirroring the screening
  domain's use of an identical 400-day constant for the same warmup
  purpose.
"""

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from maverick.market_data.service import MarketDataService
from maverick.platform.telemetry import get_logger
from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.types import ComparisonResult, CorrelationResult, RiskAnalysis
from maverick.technical.indicators import atr as compute_atr
from maverick.technical.indicators import rsi as compute_rsi

logger = get_logger(__name__)

_HISTORY_CONCURRENCY = 4
_TRADING_DAYS_PER_YEAR = 252
_HIGH_CORRELATION_THRESHOLD = 0.7
_HEDGE_CORRELATION_THRESHOLD = -0.3
_ATR_PERIOD = 20


def _classify_trend(close: pd.Series) -> tuple[int, str]:
    """Simple 0-3 trend score from price vs. two window-relative SMAs plus
    a first-half/second-half momentum check.

    This intentionally does not port the legacy 7-point sma50/ema21/sma200
    rubric, which needs a >=200-row warmup that a `compare_tickers` window
    (default 90 trading days) will rarely have -- that rubric would just
    return all-NaN inputs here. SMA lengths degrade to the available
    window (`min(10, n)` / `min(30, n)`) so short windows still produce a
    score instead of NaN.
    """
    n = len(close)
    if n < 2:
        return 0, "Neutral"

    price = float(close.iloc[-1])
    sma_short = float(close.tail(min(10, n)).mean())
    sma_long = float(close.tail(min(30, n)).mean())
    half = n // 2
    first_half_avg = float(close.iloc[:half].mean()) if half > 0 else price
    second_half_avg = float(close.iloc[half:].mean())

    score = 0
    if price > sma_short:
        score += 1
    if sma_short > sma_long:
        score += 1
    if second_half_avg > first_half_avg:
        score += 1

    description = "Uptrend" if score >= 2 else "Downtrend" if score == 0 else "Neutral"
    return score, description


async def _fetch_frames(
    market_data: MarketDataService, tickers: list[str], days: int, pad_days: int
) -> dict[str, pd.DataFrame]:
    """Fetch each ticker's OHLCV history once, calendar-padded, trimmed to
    its most recent `days` rows. A failed or empty fetch is simply absent
    from the returned dict -- never fatal to the whole batch."""
    start = date.today() - timedelta(days=days + pad_days)
    semaphore = asyncio.Semaphore(_HISTORY_CONCURRENCY)

    async def _fetch(ticker: str) -> tuple[str, pd.DataFrame | None]:
        async with semaphore:
            try:
                frame = await market_data.get_price_history(ticker, start, None)
            except Exception:
                logger.warning(
                    "portfolio: failed to fetch history for %s, skipping",
                    ticker,
                    exc_info=True,
                )
                return ticker, None
        return ticker, frame

    fetched = await asyncio.gather(*(_fetch(ticker) for ticker in tickers))
    return {
        ticker: frame.tail(days)
        for ticker, frame in fetched
        if frame is not None and not frame.empty
    }


async def correlation_analysis(
    market_data: MarketDataService,
    settings: PortfolioSettings,
    tickers: list[str],
    days: int | None,
) -> CorrelationResult:
    """Correlation matrix, high-correlation/hedge pairs, and a
    diversification score for `tickers` (already normalized, len >= 2).
    `portfolio_context` is always `None` here -- the caller stamps it on."""
    resolved_days = days if days is not None else settings.correlation_default_days

    frames = await _fetch_frames(
        market_data, tickers, resolved_days, settings.history_pad_calendar_days
    )
    closes = {
        ticker: frame["Close"]
        for ticker, frame in frames.items()
        if "Close" in frame.columns
    }
    if len(closes) < 2:
        raise ValueError(
            f"Insufficient valid price data (need 2+ tickers, got {len(closes)})"
        )

    used_tickers = list(closes.keys())
    returns_df = pd.DataFrame(closes).pct_change().dropna()

    if len(returns_df) < settings.correlation_min_rows:
        raise ValueError(
            "Insufficient data points for correlation analysis: need at least "
            f"{settings.correlation_min_rows}, got {len(returns_df)}"
        )

    corr = returns_df.corr()
    matrix = {
        row: {col: float(corr.loc[row, col]) for col in used_tickers}
        for row in used_tickers
    }

    high_correlation_pairs: list[dict[str, Any]] = []
    hedges: list[dict[str, Any]] = []
    for i in range(len(used_tickers)):
        for j in range(i + 1, len(used_tickers)):
            a, b = used_tickers[i], used_tickers[j]
            pair_corr = float(corr.loc[a, b])
            if pair_corr > _HIGH_CORRELATION_THRESHOLD:
                high_correlation_pairs.append(
                    {
                        "pair": (a, b),
                        "correlation": round(pair_corr, 3),
                        "interpretation": "High positive correlation",
                    }
                )
            elif pair_corr < _HEDGE_CORRELATION_THRESHOLD:
                hedges.append(
                    {
                        "pair": (a, b),
                        "correlation": round(pair_corr, 3),
                        "interpretation": "Negative correlation (potential hedge)",
                    }
                )

    off_diagonal = corr.values[corr.values != 1]
    avg_correlation = float(off_diagonal.mean()) if len(off_diagonal) > 0 else 0.0
    diversification_score = round((1 - avg_correlation) * 100, 1)
    recommendation = (
        "Well diversified"
        if avg_correlation < 0.3
        else "Moderately diversified"
        if avg_correlation < 0.5
        else "Consider adding uncorrelated assets"
    )

    return CorrelationResult(
        matrix=matrix,
        high_correlation_pairs=high_correlation_pairs,
        hedges=hedges,
        average_correlation=round(avg_correlation, 3),
        diversification_score=diversification_score,
        recommendation=recommendation,
        period_days=resolved_days,
        data_points=len(returns_df),
        portfolio_context=None,
    )


def _compare_one(frame: pd.DataFrame) -> dict[str, Any]:
    close = frame["Close"]
    current_price = float(close.iloc[-1])
    start_price = float(close.iloc[0])
    price_change_pct = (
        ((current_price - start_price) / start_price) * 100 if start_price else 0.0
    )

    returns = close.pct_change().dropna()
    volatility = (
        float(returns.std() * (_TRADING_DAYS_PER_YEAR**0.5) * 100)
        if len(returns) > 1
        else 0.0
    )

    rsi_series = compute_rsi(close)
    latest_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else float("nan")
    current_rsi = None if pd.isna(latest_rsi) else float(latest_rsi)
    rsi_signal = (
        "unavailable"
        if current_rsi is None
        else "overbought"
        if current_rsi > 70
        else "oversold"
        if current_rsi < 30
        else "neutral"
    )

    trend_strength, trend_description = _classify_trend(close)

    volume = frame["Volume"] if "Volume" in frame.columns else None
    if volume is not None and len(volume) >= 22 and volume.iloc[-22] > 0:
        volume_change_pct = float((volume.iloc[-1] / volume.iloc[-22] - 1) * 100)
    else:
        volume_change_pct = 0.0
    current_volume = int(volume.iloc[-1]) if volume is not None else 0
    avg_volume = int(volume.mean()) if volume is not None else 0
    volume_trend = (
        "Increasing"
        if volume_change_pct > 20
        else "Decreasing"
        if volume_change_pct < -20
        else "Stable"
    )

    return {
        "current_price": current_price,
        "performance": {
            "price_change_pct": round(price_change_pct, 2),
            "period_high": float(frame["High"].max()),
            "period_low": float(frame["Low"].min()),
            "volatility_annual": round(volatility, 2),
        },
        "technical": {
            "rsi": current_rsi,
            "rsi_signal": rsi_signal,
            "trend_strength": trend_strength,
            "trend_description": trend_description,
        },
        "volume": {
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "volume_change_pct": round(volume_change_pct, 2),
            "volume_trend": volume_trend,
        },
    }


async def compare_tickers(
    market_data: MarketDataService,
    settings: PortfolioSettings,
    tickers: list[str],
    days: int | None,
) -> ComparisonResult:
    """Side-by-side performance/technical/volume comparison for `tickers`
    (already normalized, len >= 2). `portfolio_context` is always `None`
    here -- the caller stamps it on."""
    resolved_days = days if days is not None else settings.compare_default_days

    frames = await _fetch_frames(
        market_data, tickers, resolved_days, settings.history_pad_calendar_days
    )
    missing = [t for t in tickers if t not in frames]
    if missing:
        raise ValueError(f"Insufficient price data for: {', '.join(missing)}")

    comparison: dict[str, dict[str, Any]] = {
        ticker: _compare_one(frames[ticker]) for ticker in tickers
    }

    perf_sorted = sorted(
        tickers,
        key=lambda t: comparison[t]["performance"]["price_change_pct"],
        reverse=True,
    )
    trend_sorted = sorted(
        tickers,
        key=lambda t: comparison[t]["technical"]["trend_strength"],
        reverse=True,
    )
    for i, ticker in enumerate(perf_sorted):
        comparison[ticker]["rankings"] = {
            "performance_rank": i + 1,
            "trend_rank": trend_sorted.index(ticker) + 1,
        }

    return ComparisonResult(
        comparison=comparison,
        best_performer=perf_sorted[0],
        strongest_trend=trend_sorted[0],
        period_days=resolved_days,
        as_of=datetime.now(UTC).isoformat(),
        portfolio_context=None,
    )


async def risk_adjusted_analysis(
    market_data: MarketDataService,
    settings: PortfolioSettings,
    ticker: str,
    risk_level: float,
) -> RiskAnalysis:
    """ATR-based position sizing/stop/target for `ticker` (already
    normalized). `existing_position` is always `None` here -- the caller
    computes and stamps that block on via the ledger."""
    start = date.today() - timedelta(days=settings.history_pad_calendar_days)
    try:
        frame = await market_data.get_price_history(ticker, start, None)
    except Exception as exc:
        raise ValueError(f"Unable to fetch price data for {ticker}: {exc}") from exc

    if frame.empty or not {"High", "Low", "Close"}.issubset(frame.columns):
        raise ValueError(f"Insufficient data for {ticker}")

    atr_series = compute_atr(
        frame["High"], frame["Low"], frame["Close"], period=_ATR_PERIOD
    )
    latest_atr = atr_series.iloc[-1] if len(atr_series) > 0 else float("nan")
    if pd.isna(latest_atr):
        raise ValueError(f"Insufficient history to compute ATR for {ticker}")

    current_atr = float(latest_atr)
    current_price = float(frame["Close"].iloc[-1])
    risk_factor = risk_level / 100
    account_size = settings.risk_account_size

    position_sizing = {
        "suggested_position_size": round(account_size * 0.01 * risk_factor, 2),
        "max_shares": (
            int((account_size * 0.01 * risk_factor) / current_price)
            if current_price
            else 0
        ),
        "position_value": round(account_size * 0.01 * risk_factor, 2),
        "percent_of_portfolio": round(1 * risk_factor, 2),
    }
    stop_loss = {
        "stop_loss": round(current_price - (current_atr * (2 - risk_factor)), 2),
        "stop_loss_percent": (
            round(((current_atr * (2 - risk_factor)) / current_price) * 100, 2)
            if current_price
            else 0.0
        ),
        "max_risk_amount": round(account_size * 0.01 * risk_factor, 2),
    }
    entry_strategy = {
        "immediate_entry": round(current_price, 2),
        "scale_in_levels": [
            round(current_price, 2),
            round(current_price - (current_atr * 0.5), 2),
            round(current_price - current_atr, 2),
        ],
    }
    targets = {
        "price_target": round(current_price + (current_atr * 3 * risk_factor), 2),
        "profit_potential": round(current_atr * 3 * risk_factor, 2),
        "risk_reward_ratio": round(3 * risk_factor, 2),
    }
    analysis = {
        "confidence_score": round(70 * risk_factor, 2),
        "strategy_type": (
            "aggressive"
            if risk_level > 70
            else "moderate"
            if risk_level > 30
            else "conservative"
        ),
        "time_horizon": (
            "short-term"
            if risk_level > 70
            else "medium-term"
            if risk_level > 30
            else "long-term"
        ),
    }

    return RiskAnalysis(
        ticker=ticker,
        current_price=round(current_price, 2),
        atr=round(current_atr, 2),
        risk_level=risk_level,
        position_sizing=position_sizing,
        stop_loss=stop_loss,
        entry_strategy=entry_strategy,
        targets=targets,
        analysis=analysis,
        existing_position=None,
    )
