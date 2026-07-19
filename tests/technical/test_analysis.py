"""Tests for maverick.technical.analysis.

Threshold/crossover/labeling cases are ported from
`tests/core/test_technical_analysis.py` (the legacy behavioral reference),
adapted to this module's typed outputs and "prepared DataFrame" column
contract (see `maverick/technical/analysis.py`'s module docstring). Exact
labels are asserted throughout, rather than the legacy suite's looser
"one of these outcomes" checks, per the porting mandate.

`indicators.py` is used here (never inside `analysis.py` itself -- the
"Technical layers are forward-only" contract forbids that import) to build
realistic prepared frames for the NaN-warmup-tolerance tests at the bottom
of this file.
"""

import numpy as np
import pandas as pd
import pytest

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
from maverick.technical.config import TechnicalSettings
from maverick.technical.types import (
    MACDAnalysis,
    RSIAnalysis,
    StochasticAnalysis,
    TrendAnalysis,
)

SETTINGS = TechnicalSettings()


# -- analyze_rsi --------------------------------------------------------


class TestAnalyzeRSI:
    def test_overbought(self):
        df = pd.DataFrame({"rsi": [50, 55, 52, 65, 60, 70, 68, 75, 72, 80]})
        result = analyze_rsi(df, SETTINGS)

        assert result.current == 80.0
        assert result.period == 14
        assert result.signal == "overbought"
        assert "overbought" in result.description

    def test_oversold(self):
        df = pd.DataFrame({"rsi": [50, 40, 30, 25, 20]})
        result = analyze_rsi(df, SETTINGS)

        assert result.current == 20.0
        assert result.signal == "oversold"

    def test_bullish(self):
        df = pd.DataFrame({"rsi": [50, 55, 60]})
        result = analyze_rsi(df, SETTINGS)

        assert result.current == 60.0
        assert result.signal == "bullish"

    def test_bearish(self):
        df = pd.DataFrame({"rsi": [50, 45, 40]})
        result = analyze_rsi(df, SETTINGS)

        assert result.current == 40.0
        assert result.signal == "bearish"

    def test_empty_dataframe(self):
        result = analyze_rsi(pd.DataFrame(), SETTINGS)

        assert result.current is None
        assert result.period == 14
        assert result.signal == "unavailable"

    def test_missing_rsi_column(self):
        df = pd.DataFrame({"Close": [100, 105, 110]})
        result = analyze_rsi(df, SETTINGS)

        assert result.current is None
        assert result.signal == "unavailable"

    def test_nan_current_value(self):
        df = pd.DataFrame({"rsi": [50, 55, np.nan]})
        result = analyze_rsi(df, SETTINGS)

        assert result.current is None
        assert result.signal == "unavailable"


# -- analyze_macd ---------------------------------------------------------


class TestAnalyzeMACD:
    def test_bearish_signal(self):
        df = pd.DataFrame(
            {
                "macd": [1.5, 2.0, 2.5, 3.0, 2.8],
                "macd_signal": [1.0, 1.8, 2.2, 2.7, 3.2],
                "macd_hist": [0.5, 0.2, 0.3, 0.3, -0.4],
            }
        )
        result = analyze_macd(df, SETTINGS)

        assert result.macd == 2.8
        assert result.signal_line == 3.2
        assert result.histogram == -0.4
        assert result.indicator_signal == "bearish"

    def test_bullish_crossover_detected(self):
        df = pd.DataFrame(
            {
                "macd": [1.0, 2.0],
                "macd_signal": [1.5, 1.0],
                "macd_hist": [-0.5, 1.0],
            }
        )
        result = analyze_macd(df, SETTINGS)

        assert result.crossover == "bullish crossover detected"
        assert result.indicator_signal == "bullish"

    def test_bearish_crossover_detected(self):
        df = pd.DataFrame(
            {
                "macd": [2.0, 1.0],
                "macd_signal": [1.5, 2.0],
                "macd_hist": [0.5, -1.0],
            }
        )
        result = analyze_macd(df, SETTINGS)

        assert result.crossover == "bearish crossover detected"
        assert result.indicator_signal == "bearish"

    def test_no_crossover_with_two_rows(self):
        df = pd.DataFrame(
            {
                "macd": [1.0, 1.2],
                "macd_signal": [2.0, 2.2],
                "macd_hist": [-1.0, -1.0],
            }
        )
        result = analyze_macd(df, SETTINGS)

        assert result.crossover == "no recent crossover"

    def test_crossover_requires_two_rows(self):
        df = pd.DataFrame({"macd": [2.0], "macd_signal": [1.0], "macd_hist": [1.0]})
        result = analyze_macd(df, SETTINGS)

        assert result.crossover == "no recent crossover"

    def test_missing_data(self):
        df = pd.DataFrame(
            {"macd": [np.nan], "macd_signal": [np.nan], "macd_hist": [np.nan]}
        )
        result = analyze_macd(df, SETTINGS)

        assert result.macd is None
        assert result.indicator_signal == "unavailable"
        assert result.crossover == "unavailable"


# -- analyze_stochastic -----------------------------------------------------


class TestAnalyzeStochastic:
    def test_bearish_signal(self):
        df = pd.DataFrame(
            {
                "stoch_k": [20, 30, 40, 50, 60],
                "stoch_d": [25, 35, 45, 55, 65],
            }
        )
        result = analyze_stochastic(df, SETTINGS)

        assert result.k == 60.0
        assert result.d == 65.0
        assert result.signal == "bearish"

    def test_overbought(self):
        df = pd.DataFrame({"stoch_k": [85], "stoch_d": [83]})
        result = analyze_stochastic(df, SETTINGS)

        assert result.signal == "overbought"

    def test_oversold(self):
        df = pd.DataFrame({"stoch_k": [15], "stoch_d": [18]})
        result = analyze_stochastic(df, SETTINGS)

        assert result.signal == "oversold"

    def test_bullish_crossover_detected(self):
        df = pd.DataFrame({"stoch_k": [30, 45], "stoch_d": [40, 35]})
        result = analyze_stochastic(df, SETTINGS)

        assert result.crossover == "bullish crossover detected"
        assert result.signal == "bullish"

    def test_missing_data(self):
        df = pd.DataFrame({"stoch_k": [np.nan], "stoch_d": [np.nan]})
        result = analyze_stochastic(df, SETTINGS)

        assert result.k is None
        assert result.d is None
        assert result.signal == "unavailable"
        assert result.crossover == "unavailable"


# -- analyze_bollinger ------------------------------------------------------


class TestAnalyzeBollinger:
    def test_above_middle_band(self):
        df = pd.DataFrame(
            {
                "Close": [100, 105, 110, 108, 112],
                "bb_upper": [115, 116, 117, 116, 118],
                "bb_lower": [85, 86, 87, 86, 88],
                "bb_mid": [100, 101, 102, 101, 103],
            }
        )
        result = analyze_bollinger(df, SETTINGS)

        assert result.upper == 118.0
        assert result.middle == 103.0
        assert result.lower == 88.0
        assert result.current_price == 112.0
        assert result.position == "above middle band"
        assert "bullish" in result.description

    def test_above_upper_band(self):
        df = pd.DataFrame(
            {"Close": [120], "bb_upper": [115], "bb_lower": [85], "bb_mid": [100]}
        )
        result = analyze_bollinger(df, SETTINGS)

        assert result.position == "above upper band"
        assert "overbought" in result.description

    def test_below_lower_band(self):
        df = pd.DataFrame(
            {"Close": [80], "bb_upper": [115], "bb_lower": [85], "bb_mid": [100]}
        )
        result = analyze_bollinger(df, SETTINGS)

        assert result.position == "below lower band"
        assert "oversold" in result.description

    def test_volatility_contracting(self):
        df = pd.DataFrame(
            {
                "Close": [100, 100, 100, 100, 100],
                "bb_upper": [110, 108, 106, 104, 102],
                "bb_lower": [90, 92, 94, 96, 98],
                "bb_mid": [100, 100, 100, 100, 100],
            }
        )
        result = analyze_bollinger(df, SETTINGS)

        assert result.volatility == "contracting (potential breakout ahead)"

    def test_volatility_expanding(self):
        df = pd.DataFrame(
            {
                "Close": [100, 100, 100, 100, 100],
                "bb_upper": [102, 104, 106, 108, 110],
                "bb_lower": [98, 96, 94, 92, 90],
                "bb_mid": [100, 100, 100, 100, 100],
            }
        )
        result = analyze_bollinger(df, SETTINGS)

        assert result.volatility == "expanding (increased volatility)"

    def test_unavailable_when_bands_missing(self):
        df = pd.DataFrame({"Close": [100]})
        result = analyze_bollinger(df, SETTINGS)

        assert result.upper is None
        assert result.current_price is None
        assert result.position == "unavailable"
        assert result.volatility == "unavailable"


# -- analyze_volume -----------------------------------------------------


class TestAnalyzeVolume:
    def test_high_volume_up_move(self):
        df = pd.DataFrame(
            {
                "Volume": [1_000_000] * 9 + [2_000_000],
                "Close": [100] * 9 + [105],
            }
        )
        result = analyze_volume(df, SETTINGS)

        assert result.current == 2_000_000.0
        assert result.ratio == round(2_000_000 / 1_100_000, 2)
        assert result.description == "above average"
        assert result.signal == "bullish (high volume on up move)"

    def test_high_volume_down_move(self):
        df = pd.DataFrame(
            {
                "Volume": [1_000_000] * 9 + [2_000_000],
                "Close": [105] * 9 + [100],
            }
        )
        result = analyze_volume(df, SETTINGS)

        assert result.description == "above average"
        assert result.signal == "bearish (high volume on down move)"

    def test_low_volume(self):
        df = pd.DataFrame(
            {
                "Volume": [1_000_000] * 9 + [600_000],
                "Close": [100] * 10,
            }
        )
        result = analyze_volume(df, SETTINGS)

        assert result.ratio < 0.7
        assert result.description == "below average"
        assert result.signal == "weak conviction"

    def test_insufficient_data_uses_full_mean(self):
        df = pd.DataFrame({"Volume": [1_000_000], "Close": [100]})
        result = analyze_volume(df, SETTINGS)

        assert result.current == 1_000_000.0
        assert result.average == 1_000_000.0
        assert result.ratio == 1.0
        assert result.signal == "neutral"

    def test_nan_volume_is_unavailable(self):
        df = pd.DataFrame({"Volume": [np.nan], "Close": [100]})
        result = analyze_volume(df, SETTINGS)

        assert result.current is None
        assert result.signal == "unavailable"

    def test_missing_volume_column(self):
        df = pd.DataFrame({"Close": [100]})
        result = analyze_volume(df, SETTINGS)

        assert result.current is None
        assert result.signal == "unavailable"


# -- analyze_trend ------------------------------------------------------


class TestAnalyzeTrend:
    def test_full_bullish_score(self):
        df = pd.DataFrame(
            {
                "Close": [110],
                "sma_short": [100],
                "ema": [105],
                "sma_long": [95],
                "rsi": [60],
                "macd": [1.0],
                "adx": [30],
            }
        )
        result = analyze_trend(df, SETTINGS)

        assert result.score == 7
        assert result.direction == "bullish"
        assert result.adx == 30.0

    def test_zero_score_on_downtrend_data(self):
        df = pd.DataFrame(
            {
                "Close": [90],
                "sma_short": [100],
                "ema": [95],
                "sma_long": [105],
                "rsi": [40],
                "macd": [-1.0],
                "adx": [10],
            }
        )
        result = analyze_trend(df, SETTINGS)

        assert result.score == 0
        assert result.direction == "bearish"

    def test_neutral_band_score(self):
        df = pd.DataFrame(
            {
                "Close": [110],
                "sma_short": [100],  # close > sma_short: +1
                "ema": [115],  # close > ema: false
                "sma_long": [90],  # ema > sma_short: +1 ; sma_short > sma_long: +1
                "rsi": [60],  # +1
                "macd": [-1.0],  # false
                "adx": [10],  # false
            }
        )
        result = analyze_trend(df, SETTINGS)

        assert result.score == 4
        assert result.direction == "neutral"

    def test_empty_dataframe_scores_zero(self):
        result = analyze_trend(pd.DataFrame(), SETTINGS)

        assert result.score == 0
        assert result.direction == "bearish"
        assert result.adx is None

    def test_missing_indicator_columns_scores_zero(self):
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})
        result = analyze_trend(df, SETTINGS)

        assert result.score == 0
        assert result.adx is None


# -- support_resistance -----------------------------------------------------


class TestSupportResistance:
    def test_small_dataset_uses_whole_frame(self):
        df = pd.DataFrame(
            {
                "High": [105, 110, 108],
                "Low": [95, 100, 98],
                "Close": [100, 105, 103],
            }
        )
        result = support_resistance(df, SETTINGS)

        assert result.support == [92.7, 95.0, 97.85]
        assert result.resistance == [108.15, 110.0, 113.3]

    def test_lookback_window_excludes_older_bars(self):
        # sr_lookback defaults to 30: the first 5 rows carry an extreme
        # High/Low that must NOT leak into the window-based levels.
        df = pd.DataFrame(
            {
                "High": [200.0] * 5 + [110.0] * 30,
                "Low": [10.0] * 5 + [95.0] * 30,
                "Close": [100.0] * 35,
            }
        )
        result = support_resistance(df, SETTINGS)

        assert result.support == [90.0, 95.0]
        assert result.resistance == [105.0, 110.0]

    def test_missing_columns_returns_empty_levels(self):
        df = pd.DataFrame({"Close": [100, 101]})
        result = support_resistance(df, SETTINGS)

        assert result.support == []
        assert result.resistance == []

    def test_empty_dataframe_returns_empty_levels(self):
        result = support_resistance(pd.DataFrame(), SETTINGS)

        assert result.support == []
        assert result.resistance == []


# -- generate_outlook -----------------------------------------------------


def _trend(score: int) -> TrendAnalysis:
    direction = "bullish" if score >= 5 else "bearish" if score <= 2 else "neutral"
    return TrendAnalysis(score=score, direction=direction, adx=None)


def _rsi(signal: str) -> RSIAnalysis:
    return RSIAnalysis(current=50.0, period=14, signal=signal, description="x")


def _macd(
    indicator_signal: str, crossover: str = "no recent crossover"
) -> MACDAnalysis:
    return MACDAnalysis(
        macd=0.0,
        signal_line=0.0,
        histogram=0.0,
        indicator_signal=indicator_signal,
        crossover=crossover,
        description="x",
    )


def _stoch(signal: str) -> StochasticAnalysis:
    return StochasticAnalysis(
        k=50.0, d=50.0, signal=signal, crossover="no recent crossover", description="x"
    )


class TestGenerateOutlook:
    def test_strongly_bullish(self):
        outlook = generate_outlook(
            _trend(7), _rsi("oversold"), _macd("bullish"), _stoch("oversold")
        )

        assert outlook == "strongly bullish"

    def test_moderately_bullish(self):
        outlook = generate_outlook(
            _trend(3), _rsi("bullish"), _macd("neutral"), _stoch("neutral")
        )

        assert outlook == "moderately bullish"

    def test_strongly_bearish(self):
        outlook = generate_outlook(
            _trend(0), _rsi("bearish"), _macd("bearish"), _stoch("bearish")
        )

        assert outlook == "strongly bearish"

    def test_moderately_bearish(self):
        outlook = generate_outlook(
            _trend(2), _rsi("neutral"), _macd("neutral"), _stoch("bearish")
        )

        assert outlook == "moderately bearish"

    def test_neutral(self):
        outlook = generate_outlook(
            _trend(3), _rsi("neutral"), _macd("neutral"), _stoch("neutral")
        )

        assert outlook == "neutral"

    def test_macd_crossover_alone_contributes_a_bullish_signal(self):
        outlook = generate_outlook(
            _trend(3),
            _rsi("neutral"),
            _macd("neutral", crossover="bullish crossover detected"),
            _stoch("neutral"),
        )

        assert outlook == "moderately bullish"

    def test_fixed_trend_branch_now_fires_for_a_strongly_trending_frame(self):
        """Regression test for the decision-logged behavior change.

        A frame engineered so every `analyze_trend` check passes (score 7)
        must make the trend score count toward the outlook, even when
        RSI/MACD/stochastic are all neutral. Under the legacy
        `generate_outlook`, `trend` was compared as a string
        (`trend == "uptrend"`) against `analyze_trend(df)`'s **integer**
        score -- a comparison that is always False -- so this exact
        scenario produced "neutral" in production. This test would FAIL
        against that legacy behavior.
        """
        df = pd.DataFrame(
            {
                "Close": [110],
                "sma_short": [100],
                "ema": [105],
                "sma_long": [95],
                "rsi": [60],
                "macd": [1.0],
                "adx": [30],
            }
        )
        trend = analyze_trend(df, SETTINGS)
        assert trend.score == 7  # sanity: every check in the rubric passed

        outlook = generate_outlook(
            trend, _rsi("neutral"), _macd("neutral"), _stoch("neutral")
        )

        assert outlook != "neutral"
        assert "bullish" in outlook


# -- NaN-warmup tolerance -------------------------------------------------


def _engineered_prepared_frame(n: int) -> pd.DataFrame:
    """A long, steadily-rising OHLCV frame with every indicator column
    this module reads, computed via `indicators.py` (never imported by
    `analysis.py` itself -- only by this test module). Warmup periods
    (sma_long=200 needs 200 rows, etc.) leave real NaN runs at the head of
    several columns, but with n=220 the tail is fully valid -- exactly the
    "history has NaN warmups, current row doesn't" shape these tests
    exercise.
    """
    close = pd.Series(np.linspace(100.0, 180.0, n))
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series(np.linspace(1_000_000, 1_500_000, n))

    macd_df = indicators.macd(
        close,
        fast=SETTINGS.macd_fast_period,
        slow=SETTINGS.macd_slow_period,
        signal=SETTINGS.macd_signal_period,
    )
    stoch_df = indicators.stochastic(
        high,
        low,
        close,
        k=SETTINGS.stoch_k_period,
        d=SETTINGS.stoch_d_period,
        smooth_k=SETTINGS.stoch_smooth_k,
    )
    bb_df = indicators.bollinger(
        close, length=SETTINGS.bollinger_period, std=SETTINGS.bollinger_std
    )

    df = pd.DataFrame(
        {
            "Close": close,
            "High": high,
            "Low": low,
            "Volume": volume,
            "rsi": indicators.rsi(close, period=SETTINGS.rsi_period),
            "macd": macd_df["macd"],
            "macd_signal": macd_df["signal"],
            "macd_hist": macd_df["histogram"],
            "stoch_k": stoch_df["k"],
            "stoch_d": stoch_df["d"],
            "bb_mid": bb_df["mid"],
            "bb_upper": bb_df["upper"],
            "bb_lower": bb_df["lower"],
            "sma_short": indicators.sma(close, SETTINGS.sma_short_period),
            "sma_long": indicators.sma(close, SETTINGS.sma_long_period),
            "ema": indicators.ema(close, SETTINGS.ema_period),
            "adx": indicators.adx(high, low, close, length=SETTINGS.adx_period),
        }
    )
    # Sanity: real warmup NaNs exist in history, but not on the last row --
    # this is the exact shape these tests are asserting tolerance for.
    assert df["sma_long"].iloc[:190].isna().all()
    assert df.iloc[-1].notna().all()
    return df


@pytest.fixture(scope="module")
def warm_frame() -> pd.DataFrame:
    return _engineered_prepared_frame(220)


class TestNaNWarmupTolerance:
    def test_rsi_tolerates_history_warmup(self, warm_frame):
        result = analyze_rsi(warm_frame, SETTINGS)

        assert result.current is not None
        assert result.signal != "unavailable"

    def test_macd_tolerates_history_warmup(self, warm_frame):
        result = analyze_macd(warm_frame, SETTINGS)

        assert result.macd is not None
        assert result.indicator_signal != "unavailable"
        assert result.crossover != "unavailable"

    def test_stochastic_tolerates_history_warmup(self, warm_frame):
        result = analyze_stochastic(warm_frame, SETTINGS)

        assert result.k is not None
        assert result.signal != "unavailable"

    def test_bollinger_tolerates_history_warmup(self, warm_frame):
        result = analyze_bollinger(warm_frame, SETTINGS)

        assert result.upper is not None
        assert result.position != "unavailable"

    def test_trend_tolerates_history_warmup(self, warm_frame):
        result = analyze_trend(warm_frame, SETTINGS)

        assert 0 <= result.score <= 7
        assert result.direction in ("bullish", "bearish", "neutral")
        assert result.adx is not None

    def test_volume_unaffected_by_indicator_warmup(self, warm_frame):
        result = analyze_volume(warm_frame, SETTINGS)

        assert result.current is not None
        assert result.signal != "unavailable"

    def test_support_resistance_unaffected_by_indicator_warmup(self, warm_frame):
        result = support_resistance(warm_frame, SETTINGS)

        assert len(result.support) > 0
        assert len(result.resistance) > 0
