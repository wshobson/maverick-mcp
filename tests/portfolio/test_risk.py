"""Tests for maverick.portfolio.risk -- pure functions, hand-computable fixtures.

Every expected number below is derived by hand (or by mirrored symmetric
construction, documented inline) from the same arithmetic the functions
under test perform, not by re-running the function and asserting on its
own output.
"""

import pandas as pd
import pytest

from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.risk import (
    check_position_risk,
    classify_regime,
    compute_dashboard,
    generate_alerts,
    regime_adjusted_size,
)
from maverick.portfolio.types import PositionExposure

_SETTINGS = PortfolioSettings()


def _exposure(
    symbol: str,
    shares: float,
    cost_basis: float,
    current_price: float,
    sector: str = "Unknown",
) -> PositionExposure:
    return PositionExposure(
        symbol=symbol,
        shares=shares,
        cost_basis=cost_basis,
        current_price=current_price,
        sector=sector,
    )


# ---------------------------------------------------------------------------
# compute_dashboard
# ---------------------------------------------------------------------------


def test_compute_dashboard_empty_positions_returns_zeroed_dashboard():
    result = compute_dashboard([], _SETTINGS)

    assert result.total_value == 0.0
    assert result.sector_concentration == {}
    assert result.max_sector_pct == 0.0
    assert result.portfolio_var_95 == 0.0
    assert result.portfolio_var_99 == 0.0
    assert result.total_pnl == 0.0
    assert result.position_count == 0


def test_compute_dashboard_single_position_var_and_pnl():
    # value = 10 * 150 = 1500; pnl = (150-100)*10 = 500
    # weight = 1.0 -> std = sqrt((1.0*0.02)^2) = 0.02
    # var_95 = 1.645 * 0.02 * 1500 = 49.35; var_99 = 2.326 * 0.02 * 1500 = 69.78
    positions = [_exposure("AAPL", 10, 100, 150, sector="Tech")]

    result = compute_dashboard(positions, _SETTINGS)

    assert result.total_value == 1500.0
    assert result.total_pnl == 500.0
    assert result.sector_concentration == {"Tech": 1.0}
    assert result.max_sector_pct == 1.0
    assert result.portfolio_var_95 == pytest.approx(49.35)
    assert result.portfolio_var_99 == pytest.approx(69.78)
    assert result.position_count == 1


def test_compute_dashboard_two_sectors_concentration_split():
    # Tech: 10*150 = 1500; Financial: 10*100 = 1000; total = 2500
    positions = [
        _exposure("AAPL", 10, 100, 150, sector="Tech"),
        _exposure("JPM", 10, 100, 100, sector="Financial"),
    ]

    result = compute_dashboard(positions, _SETTINGS)

    assert result.total_value == 2500.0
    assert result.sector_concentration == {"Tech": 0.6, "Financial": 0.4}
    assert result.max_sector_pct == 0.6
    assert result.total_pnl == 500.0  # (150-100)*10 + 0*10


# ---------------------------------------------------------------------------
# check_position_risk
# ---------------------------------------------------------------------------


def test_check_position_risk_new_ticker_not_previously_held():
    result = check_position_risk([], "aapl", 10, 100, _SETTINGS)

    assert result.current.total_value == 0.0
    assert result.current.position_count == 0
    assert result.projected.total_value == 1000.0
    assert result.projected.position_count == 1
    assert result.new_position.ticker == "AAPL"
    assert result.new_position.shares == 10
    assert result.new_position.price == 100
    assert result.new_position.position_value == 1000.0
    assert result.new_position.pct_of_projected_portfolio == 1.0


def test_check_position_risk_merges_into_existing_holding():
    # existing 10 @ cost 100; add 10 @ 200 -> avg cost (1000+2000)/20 = 150
    # projected value = 20 * 200 = 4000; pnl = (200-150)*20 = 1000
    existing = [_exposure("AAPL", 10, 100, 100)]

    result = check_position_risk(existing, "AAPL", 10, 200, _SETTINGS)

    assert result.projected.total_value == 4000.0
    assert result.projected.total_pnl == 1000.0
    assert result.projected.position_count == 1  # merged, not appended
    assert result.new_position.position_value == 2000.0
    assert result.new_position.pct_of_projected_portfolio == 0.5


# ---------------------------------------------------------------------------
# regime_adjusted_size
# ---------------------------------------------------------------------------


def test_regime_adjusted_size_bull_full_risk():
    # risk_amount = 100000 * 0.02 = 2000; risk_per_share = 5; shares = 400
    result = regime_adjusted_size(100000, 50, 45, 2.0, "bull", _SETTINGS)

    assert result.regime_multiplier == 1.0
    assert result.adjusted_risk_pct == 2.0
    assert result.risk_amount == 2000.0
    assert result.shares == 400
    assert result.position_value == 20000.0
    assert result.regime == "bull"


def test_regime_adjusted_size_bear_halves_risk():
    # risk_amount = 100000 * 0.01 = 1000; risk_per_share = 5; shares = 200
    result = regime_adjusted_size(100000, 50, 45, 2.0, "bear", _SETTINGS)

    assert result.regime_multiplier == 0.5
    assert result.adjusted_risk_pct == 1.0
    assert result.risk_amount == 1000.0
    assert result.shares == 200
    assert result.position_value == 10000.0


def test_regime_adjusted_size_choppy_and_transitional_are_75_pct():
    choppy = regime_adjusted_size(100000, 50, 45, 2.0, "choppy", _SETTINGS)
    transitional = regime_adjusted_size(100000, 50, 45, 2.0, "transitional", _SETTINGS)

    assert choppy.regime_multiplier == 0.75
    assert transitional.regime_multiplier == 0.75


def test_regime_adjusted_size_unknown_regime_defaults_to_full_risk():
    result = regime_adjusted_size(100000, 50, 45, 2.0, "sideways", _SETTINGS)

    assert result.regime_multiplier == 1.0
    assert result.regime == "sideways"


def test_regime_adjusted_size_zero_risk_per_share_returns_zero_shares():
    result = regime_adjusted_size(100000, 50, 50, 2.0, "bull", _SETTINGS)

    assert result.shares == 0
    assert result.position_value == 0.0


# ---------------------------------------------------------------------------
# generate_alerts
# ---------------------------------------------------------------------------


def test_generate_alerts_empty_positions_returns_no_alerts():
    assert generate_alerts([], _SETTINGS) == []


def test_generate_alerts_well_diversified_no_thresholds_breached():
    # 6 equal positions across 6 sectors: each is 1/6 = 16.67% of the
    # portfolio -- below both the 20% position-size and 30% sector warn
    # thresholds. cost_basis == current_price -> no drawdown either.
    positions = [_exposure(f"T{i}", 1, 100, 100, sector=f"S{i}") for i in range(6)]

    assert generate_alerts(positions, _SETTINGS) == []


def test_generate_alerts_sector_concentration_critical_and_warning():
    # Tech = 1500/2500 = 60% (> 50% critical); Financial = 1000/2500 = 40%
    # (> 30% warn, <= 50% critical).
    positions = [
        _exposure("AAPL", 10, 150, 150, sector="Tech"),
        _exposure("JPM", 10, 100, 100, sector="Financial"),
    ]

    alerts = generate_alerts(positions, _SETTINGS)

    concentration = {
        a.details["sector"]: a for a in alerts if a.alert_type == "concentration"
    }
    assert concentration["Tech"].severity == "critical"
    assert concentration["Financial"].severity == "warning"


def test_generate_alerts_oversized_position_warning():
    # AAPL = 1500/2500 = 60% > 20% position warn threshold.
    positions = [
        _exposure("AAPL", 10, 150, 150, sector="Tech"),
        _exposure("JPM", 10, 100, 100, sector="Financial"),
    ]

    alerts = generate_alerts(positions, _SETTINGS)

    sizing = [a for a in alerts if a.alert_type == "sizing"]
    assert any(a.details["ticker"] == "AAPL" for a in sizing)


def test_generate_alerts_drawdown_warning():
    # value = 800, cost = 1000 -> loss_pct = 0.20 > 10% threshold.
    positions = [_exposure("AAPL", 10, 100, 80, sector="Tech")]

    alerts = generate_alerts(positions, _SETTINGS)

    drawdown = [a for a in alerts if a.alert_type == "drawdown"]
    assert len(drawdown) == 1
    assert drawdown[0].details["loss_pct"] == pytest.approx(0.20)
    assert drawdown[0].severity == "warning"


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------


def _flat_series(n: int = 60, value: float = 100.0) -> pd.Series:
    return pd.Series([value] * n)


def _uptrend_series(n: int = 60, start: float = 100.0) -> pd.Series:
    return pd.Series([start + i for i in range(n)])


def _downtrend_series(n: int = 60, start: float = 200.0) -> pd.Series:
    return pd.Series([start - i for i in range(n)])


def test_classify_regime_flat_prices_low_vix_is_transitional():
    # Flat SMAs tie -> sma_cross = -1.0 (trend_score -0.5, "bear" vote);
    # vix=15 -> vol_score 0.8 ("bull"); momentum/breadth both 0/neutral.
    # composite = 0.35*-0.5 + 0.25*0.8 = 0.025 -> confidence 0.025 < 0.45.
    result = classify_regime(_flat_series(), vix_level=15.0)

    assert result["regime"] == "transitional"
    assert result["confidence"] == pytest.approx(0.025, abs=1e-4)
    assert result["drivers"]["trend"] == pytest.approx(-0.5)
    assert result["drivers"]["volatility"] == pytest.approx(0.8)
    assert result["votes"]["momentum"] == "neutral"
    assert result["votes"]["breadth"] == "neutral"


def test_classify_regime_strong_uptrend_low_vix_is_bull():
    result = classify_regime(_uptrend_series(), vix_level=10.0)

    assert result["regime"] == "bull"
    assert result["drivers"]["trend"] == pytest.approx(1.0)
    assert result["drivers"]["volatility"] == pytest.approx(0.8)
    assert result["drivers"]["momentum"] == pytest.approx(0.6711, abs=1e-4)
    assert result["confidence"] == pytest.approx(0.7178, abs=1e-4)
    assert result["votes"]["trend"] == "bull"
    assert result["votes"]["volatility"] == "bull"
    assert result["votes"]["momentum"] == "bull"


def test_classify_regime_strong_downtrend_high_vix_is_bear():
    result = classify_regime(_downtrend_series(), vix_level=35.0)

    assert result["regime"] == "bear"
    assert result["drivers"]["trend"] == pytest.approx(-1.0)
    assert result["drivers"]["volatility"] == pytest.approx(-1.0)
    assert result["confidence"] == pytest.approx(0.7656, abs=1e-4)
    assert result["votes"]["trend"] == "bear"
    assert result["votes"]["volatility"] == "bear"


def test_classify_regime_short_series_trend_and_momentum_neutral():
    # Fewer than long_window+1 (51) rows -> trend neutral; fewer than
    # momentum_window+1 (11) rows -> momentum neutral too.
    result = classify_regime(_flat_series(n=5), vix_level=20.0)

    assert result["drivers"]["trend"] == 0.0
    assert result["votes"]["trend"] == "neutral"
    assert result["drivers"]["momentum"] == 0.0
    assert result["votes"]["momentum"] == "neutral"


def test_classify_regime_breadth_ratio_bullish_and_bearish():
    bullish = classify_regime(_flat_series(n=5), vix_level=20.0, breadth_ratio=0.8)
    bearish = classify_regime(_flat_series(n=5), vix_level=20.0, breadth_ratio=0.2)

    assert bullish["votes"]["breadth"] == "bull"
    assert bullish["drivers"]["breadth"] > 0
    assert bearish["votes"]["breadth"] == "bear"
    assert bearish["drivers"]["breadth"] < 0
