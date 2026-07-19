"""Tests for maverick.screening.screens.

The rubric IS the contract: every qualifying fixture below asserts an
*exact* combined_score, derived by hand from which criteria fire in that
specific hand-crafted frame (documented inline). Frames are built with
plain arithmetic (linear/sawtooth price series) -- no network, no
randomness.
"""

import numpy as np
import pandas as pd

from maverick.screening.config import ScreeningSettings
from maverick.screening.screens import score_bearish, score_bullish, score_supply_demand


def _build_frame(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    """Build a yfinance-cased, tz-naive OHLCV frame from a close/volume series.

    High/Low/Open are derived with a small fixed offset around each close so
    every row has a sane (non-degenerate) daily range for ATR purposes.
    """
    n = len(closes)
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
    close_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": close_arr - 0.1,
            "High": close_arr + 0.5,
            "Low": close_arr - 0.5,
            "Close": close_arr,
            "Volume": np.array(volumes, dtype=float),
        },
        index=index,
    )


def _sawtooth_uptrend(
    n: int, start: float, up_step: float, down_step: float
) -> list[float]:
    """A three-up-one-down sawtooth series: net upward drift with periodic
    pullbacks, so RSI stays away from the 100 ceiling a purely monotonic
    series would hit."""
    closes = [start]
    for i in range(1, n):
        cycle = i % 4
        step = -down_step if cycle == 3 else up_step
        closes.append(closes[-1] + step)
    return closes


SETTINGS = ScreeningSettings()


class TestScoreBullish:
    def test_strong_uptrend_qualifies_with_exact_score(self):
        """A 260-row sawtooth uptrend (net +1.0/-1.2 per 4-day cycle) with a
        volume spike on the final day and RSI kept moderate by the pullbacks.

        Hand-verified criteria for this exact frame (computed once against
        the real sma/rsi indicator functions while designing the fixture):
          close=216.0  sma50=205.8  sma150=183.3  sma200=172.05  rsi14=68.35
          volume=2,500,000  avg_volume_30d=1,050,000 (ratio 2.38x)

        All six criteria fire:
          close>sma50   (216.0 > 205.8)    -> +25
          close>sma150  (216.0 > 183.3)    -> +25
          close>sma200  (216.0 > 172.05)   -> +25
          ma_aligned    (205.8>183.3>172.05) -> +25
          volume_surge  (2.38x > 1.5x)     -> +10
          rsi<80        (68.35 < 80)       -> +10
        Total = 25*4 + 10 + 10 = 120.
        """
        n = 260
        closes = _sawtooth_uptrend(n, start=100.0, up_step=1.0, down_step=1.2)
        volumes = [1_000_000.0] * n
        volumes[-1] = 2_500_000.0
        df = _build_frame(closes, volumes)

        result = score_bullish("TEST", df, SETTINGS)

        assert result is not None
        assert result.symbol == "TEST"
        assert result.screen == "bullish"
        assert result.combined_score == 120
        assert result.close == closes[-1]
        assert result.momentum_score is None
        assert result.flags == {
            "close_above_sma50": True,
            "close_above_sma150": True,
            "close_above_sma200": True,
            "ma_aligned": True,
            "volume_surge": True,
            "rsi_not_overbought": True,
        }
        assert result.indicators["close"] == closes[-1]
        assert "SMA50" in result.reason or "close above SMA50" in result.reason

    def test_one_criterion_short_of_threshold_returns_none(self):
        """A 260-row frame engineered so only 3 of 6 criteria fire, landing
        one full 25-point criterion below bullish_min_score (50).

        Construction: a long, gentle sawtooth decline (210 days, net -1.3
        + 1.0 per 4-day cycle) followed by a shorter, gentler sawtooth
        rally (50 days, net +0.9 - 0.8 per 4-day cycle). The rally lifts
        the close back above the fast SMA50 but not the slower SMA150 /
        SMA200 (still anchored by the much higher pre-decline prices), and
        the sawtooth keeps RSI out of overbought territory.

        Hand-verified criteria for this exact frame:
          close=121.2  sma50=109.35  sma150=124.967  sma200=140.512
          rsi14=76.165  volume ratio=1.935x

          close>sma50   (121.2 > 109.35)   -> +25   (fires)
          close>sma150  (121.2 > 124.967)  -> +0    (does NOT fire)
          close>sma200  (121.2 > 140.512)  -> +0    (does NOT fire)
          ma_aligned    (109.35 > 124.967? no) -> +0 (does NOT fire)
          volume_surge  (1.935x > 1.5x)    -> +10  (fires)
          rsi<80        (76.165 < 80)      -> +10  (fires)
        Total = 25 + 10 + 10 = 45, which is < 50 (bullish_min_score).
        Flipping just the next 25-point criterion (close>sma150) would
        push this to 70 and comfortably qualify -- so 45 sits exactly one
        criterion short of the line.
        """
        n = 260
        decline_days = 210
        closes: list[float] = []
        price = 250.0
        for i in range(decline_days):
            cycle = i % 4
            step = 1.0 if cycle == 3 else -1.3
            price += step
            closes.append(price)
        for i in range(n - decline_days):
            cycle = i % 4
            step = -0.8 if cycle == 3 else 0.9
            price += step
            closes.append(price)
        volumes = [1_000_000.0] * n
        volumes[-1] = 2_000_000.0
        df = _build_frame(closes, volumes)

        result = score_bullish("TEST", df, SETTINGS)

        assert result is None

    def test_history_too_short_returns_none(self):
        """A 100-row frame is below min_history_days (200 by default)."""
        n = 100
        closes = _sawtooth_uptrend(n, start=100.0, up_step=1.0, down_step=1.2)
        volumes = [1_000_000.0] * n
        df = _build_frame(closes, volumes)

        assert score_bullish("TEST", df, SETTINGS) is None
        assert score_bearish("TEST", df, SETTINGS) is None
        assert score_supply_demand("TEST", df, SETTINGS) is None


class TestScoreBearish:
    def test_bear_setup_qualifies_with_exact_score(self):
        """A 260-row frame: a sawtoothed decline for 220 days (net -1.4+1.0
        per 4-day cycle, daily range 4.0) followed by a steeper 40-day
        grind down (step -1.6/day) whose daily range shrinks to 0.4 for
        the last 20 days -- simulating a volatility-contracting sell-off.

        Hand-verified criteria for this exact frame:
          close=60.0  sma50=98.3  sma200=167.6  rsi14=0.907
          macd=-10.7055  signal=-10.4861
          atr14=3.3629  atr_avg_20d=5.0141
          volume ratio=1.5 (last-day volume 1,500,000 vs avg 1,016,666.7)
          prior_close=61.6 (today's close 60.0 is a down day)

        All seven criteria fire:
          close<sma50    (60.0 < 98.3)              -> +20
          close<sma200   (60.0 < 167.6)              -> +20
          rsi<30         (0.907 < 30)                -> +15
          macd<signal    (-10.7055 < -10.4861)       -> +15
          volume_decline (1.5x > 1.2x AND down day)  -> +20
          atr_contraction (3.3629 < 0.8*5.0141=4.011) -> +10
        Total = 20 + 20 + 15 + 15 + 20 + 10 = 100.
        """
        n = 260
        closes: list[float] = []
        ranges: list[float] = []
        price = 300.0
        for i in range(220):
            cycle = i % 4
            step = 1.0 if cycle == 3 else -1.4
            price += step
            closes.append(price)
            ranges.append(4.0)
        for _i in range(220, 240):
            price += -1.6
            closes.append(price)
            ranges.append(4.0)
        for _i in range(240, n):
            price += -1.6
            closes.append(price)
            ranges.append(0.4)

        close_arr = np.array(closes, dtype=float)
        range_arr = np.array(ranges, dtype=float)
        index = pd.date_range("2020-01-01", periods=n, freq="B", name="Date")
        volumes = [1_000_000.0] * n
        volumes[-1] = 1_500_000.0
        df = pd.DataFrame(
            {
                "Open": close_arr + range_arr / 2,
                "High": close_arr + range_arr,
                "Low": close_arr - range_arr,
                "Close": close_arr,
                "Volume": np.array(volumes, dtype=float),
            },
            index=index,
        )

        result = score_bearish("TEST", df, SETTINGS)

        assert result is not None
        assert result.symbol == "TEST"
        assert result.screen == "bearish"
        assert result.combined_score == 100
        assert result.close == closes[-1]
        assert result.momentum_score is None
        assert result.flags == {
            "close_below_sma50": True,
            "close_below_sma200": True,
            "rsi_oversold": True,
            "rsi_weak": False,
            "macd_bearish": True,
            "volume_decline": True,
            "atr_contraction": True,
        }

    def test_history_too_short_returns_none(self):
        n = 100
        closes = [200.0 - i for i in range(n)]
        volumes = [1_000_000.0] * n
        df = _build_frame(closes, volumes)

        assert score_bearish("TEST", df, SETTINGS) is None


class TestScoreSupplyDemand:
    def test_qualifying_frame_has_exact_breakout_strength(self):
        """Reuses the same strong-uptrend fixture as the bullish test: a
        260-row sawtooth uptrend qualifies the supply/demand gate too.

        Hand-verified criteria for this exact frame:
          close=216.0  sma50=205.8  sma150=183.3  sma200=172.05
          sma200 22 bars ago=162.6  high_252d=217.7
          volume=2,500,000 vs avg_volume_30d=1,050,000 (ratio 2.38x)

        All seven gate criteria fire (ALL required):
          close>sma150        (216.0 > 183.3)
          close>sma200        (216.0 > 172.05)
          sma150>sma200       (183.3 > 172.05)
          sma200_rising       (172.05 > 162.6, 22 bars back)
          sma50>sma150        (205.8 > 183.3)
          sma50>sma200        (205.8 > 172.05)
          close>sma50         (216.0 > 205.8)
        -> gated in.

        breakout_strength = 50 base
          + 25 (volume 2.38x > 1.2x)
          + 25 (close 216.0 within 25% of 252d-high 217.7,
                since 216.0 > 217.7 * 0.75 = 163.275)
          = 100.

        momentum_score = min(100, max(0, (216.0/172.05 - 1) * 100 * 5))
                        = min(100, 127.7...) = 100.0 (clipped).
        """
        n = 260
        closes = _sawtooth_uptrend(n, start=100.0, up_step=1.0, down_step=1.2)
        volumes = [1_000_000.0] * n
        volumes[-1] = 2_500_000.0
        df = _build_frame(closes, volumes)

        result = score_supply_demand("TEST", df, SETTINGS)

        assert result is not None
        assert result.symbol == "TEST"
        assert result.screen == "supply_demand"
        assert result.combined_score == 100
        assert result.momentum_score == 100.0
        assert result.flags == {
            "close_above_sma150": True,
            "close_above_sma200": True,
            "sma150_above_sma200": True,
            "sma200_rising": True,
            "sma50_above_sma150": True,
            "sma50_above_sma200": True,
            "close_above_sma50": True,
            "volume_surge": True,
            "near_52w_high": True,
        }

    def test_failing_gate_returns_none(self):
        """A flat/declining frame fails every gate criterion (close never
        clears any SMA), so it must return None even with 260 rows."""
        n = 260
        closes = [200.0 - i * 0.3 for i in range(n)]
        volumes = [1_000_000.0] * n
        df = _build_frame(closes, volumes)

        assert score_supply_demand("TEST", df, SETTINGS) is None

    def test_history_too_short_returns_none(self):
        n = 100
        closes = _sawtooth_uptrend(n, start=100.0, up_step=1.0, down_step=1.2)
        volumes = [1_000_000.0] * n
        df = _build_frame(closes, volumes)

        assert score_supply_demand("TEST", df, SETTINGS) is None
