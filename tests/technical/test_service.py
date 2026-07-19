"""Tests for maverick.technical.service.

The "engineered bullish frame" (`_bullish_frame`) is a deterministic
sawtooth-uptrend-plus-tail price series, hand-verified by running the real
`indicators.py`/`analysis.py` functions over it while designing this fixture
(see the module docstring on the exact construction). Every numeric
assertion below against that frame is that verified ground truth, not a
guess -- this is what lets round-trip tests assert exact labels/values
rather than "one of these outcomes".
"""

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from maverick.technical.config import TechnicalSettings
from maverick.technical.service import TechnicalService


class StubMarketData:
    """Async fake matching `MarketDataService.get_price_history`'s surface."""

    def __init__(self, frame: pd.DataFrame, delay: float = 0.0) -> None:
        self._frame = frame
        self._delay = delay
        self.calls: list[tuple[str, date | None, date | None]] = []

    async def get_price_history(
        self, symbol: str, start: date | None, end: date | None
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._frame


def _sawtooth(n: int, start: float, up_step: float, down_step: float) -> list[float]:
    closes = [start]
    for i in range(1, n):
        step = -down_step if i % 4 == 3 else up_step
        closes.append(closes[-1] + step)
    return closes


def _frame_from_closes(closes: list[float], last_volume: float = 1_000_000.0):
    n = len(closes)
    close_arr = np.array(closes, dtype=float)
    volumes = np.full(n, 1_000_000.0)
    volumes[-1] = last_volume
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": close_arr - 0.1,
            "High": close_arr + 0.6,
            "Low": close_arr - 0.6,
            "Close": close_arr,
            "Volume": volumes,
        },
        index=index,
    )


def _bullish_frame() -> pd.DataFrame:
    """247-row engineered frame: a moderate sawtooth uptrend (240 * 0.4 up /
    0.6 down cycles) followed by a 3-bar 0.4-step continuation tail and a
    volume spike on the last bar. Against `TechnicalSettings()` defaults
    this produces (hand-verified):

    - RSI: 69.91, signal "bullish" (just under the 70 overbought line).
    - MACD: macd=1.08, signal_line=1.05, histogram=0.02, "bullish" with a
      "bullish crossover detected".
    - Stochastic: k=81.15, d=77.56, "bullish" with a "bullish crossover
      detected".
    - Bollinger: upper=137.57, middle=135.6, lower=133.63, current=137.4,
      "above middle band", "stable" volatility.
    - Volume: current=1,800,000 vs a 1,080,000 average (ratio 1.67, "above
      average"), "bullish (high volume on up move)".
    - Trend: score 7/7, direction "bullish".
    - Outlook: "strongly bullish" (trend's 2 + RSI's 1 + MACD's 1 already
      clears the >= 4 "strongly" band before stochastic's contribution).
    """
    closes = _sawtooth(244, start=100.0, up_step=0.4, down_step=0.6)
    last = closes[-1]
    for _ in range(3):
        last += 0.4
        closes.append(last)
    return _frame_from_closes(closes, last_volume=1_800_000.0)


def _short_frame(n: int = 50) -> pd.DataFrame:
    """Enough bars for RSI (needs 14) but short of the 200-bar `sma_long`
    warmup `get_trend`/`get_full_analysis` require."""
    closes = _sawtooth(n, start=100.0, up_step=0.4, down_step=0.6)
    return _frame_from_closes(closes)


def _very_short_frame(n: int = 5) -> pd.DataFrame:
    """Short of even RSI's own 14-bar warmup."""
    closes = _sawtooth(n, start=100.0, up_step=0.4, down_step=0.6)
    return _frame_from_closes(closes)


def _nan_tail_frame() -> pd.DataFrame:
    """The bullish frame with its last bar's `Close` blanked out -- plenty
    of history, but the most recent bar is unusable."""
    df = _bullish_frame().copy()
    df.loc[df.index[-1], "Close"] = np.nan
    return df


SETTINGS = TechnicalSettings()


def _service(frame: pd.DataFrame, settings: TechnicalSettings | None = None):
    return TechnicalService(StubMarketData(frame), settings=settings)


# ---------------------------------------------------------------------------
# happy path: exact labels/values over the engineered bullish frame
# ---------------------------------------------------------------------------


async def test_get_rsi_bullish_frame():
    result = await _service(_bullish_frame()).get_rsi("AAPL")

    assert result.current == 69.91
    assert result.period == 14
    assert result.signal == "bullish"
    assert "69.91" in result.description


async def test_get_macd_bullish_frame():
    result = await _service(_bullish_frame()).get_macd("AAPL")

    assert result.macd == 1.08
    assert result.signal_line == 1.05
    assert result.histogram == 0.02
    assert result.indicator_signal == "bullish"
    assert result.crossover == "bullish crossover detected"


async def test_get_stochastic_bullish_frame():
    result = await _service(_bullish_frame()).get_stochastic("AAPL")

    assert result.k == 81.15
    assert result.d == 77.56
    assert result.signal == "bullish"
    assert result.crossover == "bullish crossover detected"


async def test_get_bollinger_bullish_frame():
    result = await _service(_bullish_frame()).get_bollinger("AAPL")

    assert result.upper == 137.57
    assert result.middle == 135.6
    assert result.lower == 133.63
    assert result.current_price == 137.4
    assert result.position == "above middle band"
    assert result.volatility == "stable"


async def test_get_volume_bullish_frame():
    result = await _service(_bullish_frame()).get_volume("AAPL")

    assert result.current == 1_800_000.0
    assert result.average == 1_080_000.0
    assert result.ratio == 1.67
    assert result.description == "above average"
    assert result.signal == "bullish (high volume on up move)"


async def test_get_trend_bullish_frame():
    result = await _service(_bullish_frame()).get_trend("AAPL")

    assert result.score == 7
    assert result.direction == "bullish"
    assert result.adx == 33.78


async def test_get_support_resistance_bullish_frame():
    result = await _service(_bullish_frame()).get_support_resistance("AAPL")

    assert result.support == [123.66, 130.53, 132.0]
    assert result.resistance == [138.0, 144.27, 151.14]


async def test_get_full_analysis_bullish_frame():
    result = await _service(_bullish_frame()).get_full_analysis("aapl")

    assert result.ticker == "AAPL"
    assert result.current_price == 137.4
    assert result.trend.score == 7
    assert result.trend.direction == "bullish"
    assert result.outlook == "strongly bullish"
    assert result.rsi.signal == "bullish"
    assert result.macd.indicator_signal == "bullish"
    assert result.stochastic.signal == "bullish"
    assert result.bollinger.position == "above middle band"
    assert result.volume.signal == "bullish (high volume on up move)"
    assert result.levels.support == [123.66, 130.53, 132.0]
    assert result.analysis_metadata["bars_analyzed"] == 247
    assert result.analysis_metadata["as_of"] == "2020-12-10"


# ---------------------------------------------------------------------------
# fetch window: calendar-padded for the 200-bar warmup
# ---------------------------------------------------------------------------


async def test_default_fetch_window_is_at_least_400_calendar_days():
    market_data = StubMarketData(_bullish_frame())
    service = TechnicalService(market_data)

    await service.get_rsi("AAPL")

    assert len(market_data.calls) == 1
    _, start, end = market_data.calls[0]
    assert (end - start).days >= 400


async def test_small_days_override_is_floored_at_400_calendar_days():
    market_data = StubMarketData(_bullish_frame())
    service = TechnicalService(market_data)

    await service.get_rsi("AAPL", days=10)

    _, start, end = market_data.calls[0]
    assert (end - start).days >= 400


async def test_large_days_override_extends_the_window():
    market_data = StubMarketData(_bullish_frame())
    service = TechnicalService(market_data)

    await service.get_rsi("AAPL", days=900)

    _, start, end = market_data.calls[0]
    assert (end - start).days >= 900


# ---------------------------------------------------------------------------
# period overrides
# ---------------------------------------------------------------------------


async def test_get_rsi_period_override_propagates_to_result():
    result = await _service(_bullish_frame()).get_rsi("AAPL", period=10)

    assert result.period == 10


async def test_get_macd_period_overrides_do_not_raise():
    result = await _service(_bullish_frame()).get_macd(
        "AAPL", fast_period=5, slow_period=15, signal_period=4
    )

    assert result.indicator_signal != "unavailable"


# ---------------------------------------------------------------------------
# insufficient history / NaN tail -> ValueError
# ---------------------------------------------------------------------------


async def test_get_rsi_succeeds_on_short_frame_with_enough_bars_for_rsi():
    result = await _service(_short_frame()).get_rsi("AAPL")

    assert result.signal != "unavailable"


async def test_get_trend_raises_on_short_frame_missing_sma_long_warmup():
    with pytest.raises(ValueError, match="AAPL"):
        await _service(_short_frame()).get_trend("AAPL")


async def test_get_full_analysis_raises_on_short_frame():
    with pytest.raises(ValueError, match="Insufficient price history"):
        await _service(_short_frame()).get_full_analysis("AAPL")


async def test_get_rsi_raises_on_very_short_frame():
    with pytest.raises(ValueError, match="Insufficient price history"):
        await _service(_very_short_frame()).get_rsi("AAPL")


async def test_get_rsi_raises_on_empty_frame():
    with pytest.raises(ValueError, match="Insufficient price history"):
        await _service(pd.DataFrame()).get_rsi("AAPL")


@pytest.mark.parametrize(
    "method",
    [
        "get_rsi",
        "get_macd",
        "get_bollinger",
        "get_stochastic",
        "get_trend",
        "get_support_resistance",
        "get_volume",
        "get_full_analysis",
    ],
)
async def test_every_method_raises_on_nan_tail_close(method: str):
    service = _service(_nan_tail_frame())

    with pytest.raises(ValueError, match="Insufficient price history"):
        await getattr(service, method)("AAPL")


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


async def test_slow_fetch_raises_value_error_not_hang():
    market_data = StubMarketData(_bullish_frame(), delay=0.2)
    service = TechnicalService(
        market_data, settings=TechnicalSettings(analysis_timeout_seconds=0.01)
    )

    with pytest.raises(ValueError, match="timed out"):
        await service.get_rsi("AAPL")
