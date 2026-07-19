"""Tests for maverick.technical.types."""

import pytest
from pydantic import ValidationError

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

# -- RSIAnalysis ------------------------------------------------------------


def _make_rsi(**overrides) -> RSIAnalysis:
    fields = {
        "current": 65.5,
        "period": 14,
        "signal": "bullish",
        "description": "RSI is currently at 65.5, indicating bullish conditions.",
    }
    fields.update(overrides)
    return RSIAnalysis(**fields)


def test_rsi_analysis_round_trips_through_model_dump():
    rsi = _make_rsi()
    data = rsi.model_dump()
    assert data["current"] == 65.5
    assert data["period"] == 14
    assert RSIAnalysis(**data) == rsi


def test_rsi_analysis_tolerates_none_current():
    rsi = _make_rsi(current=None, signal="unavailable", description="unavailable")
    assert rsi.current is None


def test_rsi_analysis_requires_period():
    with pytest.raises(ValidationError):
        RSIAnalysis.model_validate(
            {"current": 65.5, "signal": "bullish", "description": "x"}
        )


# -- MACDAnalysis -------------------------------------------------------


def _make_macd(**overrides) -> MACDAnalysis:
    fields = {
        "macd": 1.23,
        "signal_line": 0.98,
        "histogram": 0.25,
        "indicator_signal": "bullish",
        "crossover": "bullish crossover detected",
        "description": "MACD is bullish with bullish crossover detected.",
    }
    fields.update(overrides)
    return MACDAnalysis(**fields)


def test_macd_analysis_round_trips_through_model_dump():
    macd = _make_macd()
    data = macd.model_dump()
    assert data["macd"] == 1.23
    assert data["signal_line"] == 0.98
    assert data["histogram"] == 0.25
    assert MACDAnalysis(**data) == macd


def test_macd_analysis_tolerates_none_macd_signal_line_and_histogram():
    macd = _make_macd(
        macd=None,
        signal_line=None,
        histogram=None,
        indicator_signal="unavailable",
        crossover="unavailable",
        description="MACD data not available (insufficient data points)",
    )
    assert macd.macd is None
    assert macd.signal_line is None
    assert macd.histogram is None


# -- StochasticAnalysis ---------------------------------------------------


def _make_stochastic(**overrides) -> StochasticAnalysis:
    fields = {
        "k": 82.0,
        "d": 78.5,
        "signal": "overbought",
        "crossover": "no recent crossover",
        "description": "Stochastic Oscillator is overbought with no recent crossover.",
    }
    fields.update(overrides)
    return StochasticAnalysis(**fields)


def test_stochastic_analysis_round_trips_through_model_dump():
    stoch = _make_stochastic()
    data = stoch.model_dump()
    assert data["k"] == 82.0
    assert data["d"] == 78.5
    assert StochasticAnalysis(**data) == stoch


def test_stochastic_analysis_tolerates_none_k_and_d():
    stoch = _make_stochastic(
        k=None, d=None, signal="unavailable", crossover="unavailable"
    )
    assert stoch.k is None
    assert stoch.d is None


# -- BollingerAnalysis ------------------------------------------------------


def _make_bollinger(**overrides) -> BollingerAnalysis:
    fields = {
        "upper": 195.0,
        "middle": 190.0,
        "lower": 185.0,
        "current_price": 192.5,
        "position": "above middle band",
        "volatility": "stable",
        "description": "Price is above middle band, indicating bullish conditions.",
    }
    fields.update(overrides)
    return BollingerAnalysis(**fields)


def test_bollinger_analysis_round_trips_through_model_dump():
    bollinger = _make_bollinger()
    data = bollinger.model_dump()
    assert data["upper"] == 195.0
    assert data["current_price"] == 192.5
    assert BollingerAnalysis(**data) == bollinger


def test_bollinger_analysis_tolerates_none_bands_and_price():
    bollinger = _make_bollinger(
        upper=None,
        middle=None,
        lower=None,
        current_price=None,
        position="unavailable",
        volatility="unavailable",
    )
    assert bollinger.upper is None
    assert bollinger.middle is None
    assert bollinger.lower is None
    assert bollinger.current_price is None


# -- VolumeAnalysis -----------------------------------------------------


def _make_volume(**overrides) -> VolumeAnalysis:
    fields = {
        "current": 1_500_000.0,
        "average": 1_000_000.0,
        "ratio": 1.5,
        "description": "above average",
        "signal": "bullish (high volume on up move)",
    }
    fields.update(overrides)
    return VolumeAnalysis(**fields)


def test_volume_analysis_round_trips_through_model_dump():
    volume = _make_volume()
    data = volume.model_dump()
    assert data["current"] == 1_500_000.0
    assert data["ratio"] == 1.5
    assert VolumeAnalysis(**data) == volume


def test_volume_analysis_tolerates_none_current_average_and_ratio():
    volume = _make_volume(
        current=None,
        average=None,
        ratio=None,
        description="unavailable",
        signal="unavailable",
    )
    assert volume.current is None
    assert volume.average is None
    assert volume.ratio is None


# -- TrendAnalysis --------------------------------------------------------


def _make_trend(**overrides) -> TrendAnalysis:
    fields = {"score": 6, "direction": "bullish", "adx": 28.5}
    fields.update(overrides)
    return TrendAnalysis(**fields)


def test_trend_analysis_round_trips_through_model_dump():
    trend = _make_trend()
    data = trend.model_dump()
    assert data["score"] == 6
    assert data["direction"] == "bullish"
    assert TrendAnalysis(**data) == trend


def test_trend_analysis_tolerates_none_adx():
    trend = _make_trend(adx=None)
    assert trend.adx is None


# -- LevelsResult -----------------------------------------------------------


def test_levels_result_round_trips_through_model_dump():
    levels = LevelsResult(support=[180.0, 175.5], resistance=[195.0, 200.0])
    data = levels.model_dump()
    assert data["support"] == [180.0, 175.5]
    assert LevelsResult(**data) == levels


def test_levels_result_allows_empty_lists():
    levels = LevelsResult(support=[], resistance=[])
    assert levels.support == []
    assert levels.resistance == []


# -- FullTechnicalAnalysis: composition --------------------------------


def _make_full_analysis(**overrides) -> FullTechnicalAnalysis:
    fields = {
        "ticker": "AAPL",
        "current_price": 192.5,
        "trend": _make_trend(),
        "outlook": "moderately bullish",
        "rsi": _make_rsi(),
        "macd": _make_macd(),
        "stochastic": _make_stochastic(),
        "bollinger": _make_bollinger(),
        "volume": _make_volume(),
        "levels": LevelsResult(support=[180.0], resistance=[200.0]),
        "analysis_metadata": {"days": 365, "source": "market_data"},
    }
    fields.update(overrides)
    return FullTechnicalAnalysis(**fields)


def test_full_technical_analysis_composes_all_sub_models():
    full = _make_full_analysis()
    assert full.ticker == "AAPL"
    assert full.current_price == 192.5
    assert full.trend.score == 6
    assert full.outlook == "moderately bullish"
    assert full.rsi.current == 65.5
    assert full.macd.macd == 1.23
    assert full.stochastic.k == 82.0
    assert full.bollinger.upper == 195.0
    assert full.volume.current == 1_500_000.0
    assert full.levels.support == [180.0]
    assert full.analysis_metadata == {"days": 365, "source": "market_data"}


def test_full_technical_analysis_round_trips_through_model_dump():
    full = _make_full_analysis()
    data = full.model_dump()
    restored = FullTechnicalAnalysis.model_validate(data)
    assert restored == full
    assert restored.rsi == full.rsi
    assert restored.macd == full.macd
    assert restored.levels == full.levels


def test_full_technical_analysis_requires_trend_submodel():
    with pytest.raises(ValidationError):
        FullTechnicalAnalysis.model_validate(
            {
                "ticker": "AAPL",
                "current_price": 192.5,
                "outlook": "neutral",
                "rsi": _make_rsi().model_dump(),
                "macd": _make_macd().model_dump(),
                "stochastic": _make_stochastic().model_dump(),
                "bollinger": _make_bollinger().model_dump(),
                "volume": _make_volume().model_dump(),
                "levels": LevelsResult(support=[], resistance=[]).model_dump(),
                "analysis_metadata": {},
            }
        )
