"""Unit tests for RiskService portfolio analytics and alerting."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.database.base import Base
from maverick_mcp.services.risk.service import RiskService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    _Session = sessionmaker(bind=engine)
    session = _Session()
    yield session
    session.close()


@pytest.fixture
def service(db_session):
    return RiskService(db_session=db_session)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def _make_positions():
    """Three-position portfolio across two sectors."""
    return [
        {
            "symbol": "AAPL",
            "shares": 10,
            "cost_basis": 150.0,
            "current_price": 170.0,
            "sector": "Technology",
        },
        {
            "symbol": "MSFT",
            "shares": 5,
            "cost_basis": 280.0,
            "current_price": 300.0,
            "sector": "Technology",
        },
        {
            "symbol": "JPM",
            "shares": 8,
            "cost_basis": 140.0,
            "current_price": 145.0,
            "sector": "Financials",
        },
    ]


def _make_balanced_positions():
    """Balanced portfolio — no alerts should fire."""
    return [
        {
            "symbol": "AAPL",
            "shares": 5,
            "cost_basis": 150.0,
            "current_price": 155.0,
            "sector": "Technology",
        },
        {
            "symbol": "JPM",
            "shares": 5,
            "cost_basis": 140.0,
            "current_price": 145.0,
            "sector": "Financials",
        },
        {
            "symbol": "PG",
            "shares": 5,
            "cost_basis": 130.0,
            "current_price": 133.0,
            "sector": "Consumer Staples",
        },
        {
            "symbol": "XOM",
            "shares": 5,
            "cost_basis": 100.0,
            "current_price": 102.0,
            "sector": "Energy",
        },
    ]


# ---------------------------------------------------------------------------
# compute_dashboard tests
# ---------------------------------------------------------------------------


def test_compute_dashboard_basic(service):
    positions = _make_positions()
    result = service.compute_dashboard(positions)

    # total_value: 10*170 + 5*300 + 8*145 = 1700 + 1500 + 1160 = 4360
    assert result["total_value"] == pytest.approx(4360.0, rel=1e-4)
    assert result["position_count"] == 3

    # Sector concentration
    conc = result["sector_concentration"]
    assert "Technology" in conc
    assert "Financials" in conc

    tech_pct = (1700 + 1500) / 4360
    fin_pct = 1160 / 4360
    assert conc["Technology"] == pytest.approx(tech_pct, rel=1e-3)
    assert conc["Financials"] == pytest.approx(fin_pct, rel=1e-3)

    assert result["max_sector_pct"] == pytest.approx(tech_pct, rel=1e-3)


def test_compute_dashboard_empty(service):
    result = service.compute_dashboard([])
    assert result["total_value"] == 0.0
    assert result["position_count"] == 0
    assert result["portfolio_var_95"] == 0.0


def test_compute_dashboard_total_pnl(service):
    positions = _make_positions()
    result = service.compute_dashboard(positions)

    # pnl: (170-150)*10 + (300-280)*5 + (145-140)*8 = 200 + 100 + 40 = 340
    assert result["total_pnl"] == pytest.approx(340.0, rel=1e-4)


# ---------------------------------------------------------------------------
# regime_adjusted_size tests
# ---------------------------------------------------------------------------


def test_regime_adjusted_size_bull(service):
    result = service.get_regime_adjusted_size(
        account_size=100_000,
        entry_price=100.0,
        stop_loss=95.0,
        risk_pct=2.0,
        regime="bull",
    )
    # multiplier = 1.0; risk_amount = 2000; risk_per_share = 5; shares = 400
    assert result["regime_multiplier"] == pytest.approx(1.0)
    assert result["risk_amount"] == pytest.approx(2000.0)
    assert result["shares"] == 400
    assert result["position_value"] == pytest.approx(40_000.0)


def test_regime_adjusted_size_bear(service):
    result = service.get_regime_adjusted_size(
        account_size=100_000,
        entry_price=100.0,
        stop_loss=95.0,
        risk_pct=2.0,
        regime="bear",
    )
    # multiplier = 0.5; risk_amount = 1000; risk_per_share = 5; shares = 200
    assert result["regime_multiplier"] == pytest.approx(0.5)
    assert result["risk_amount"] == pytest.approx(1000.0)
    assert result["shares"] == 200


def test_regime_adjusted_size_choppy(service):
    result = service.get_regime_adjusted_size(
        account_size=100_000,
        entry_price=100.0,
        stop_loss=95.0,
        risk_pct=2.0,
        regime="choppy",
    )
    # multiplier = 0.75; risk_amount = 1500; risk_per_share = 5; shares = 300
    assert result["regime_multiplier"] == pytest.approx(0.75)
    assert result["risk_amount"] == pytest.approx(1500.0)
    assert result["shares"] == 300


# ---------------------------------------------------------------------------
# generate_alerts tests
# ---------------------------------------------------------------------------


def test_generate_alerts_concentration(service):
    """Sector concentration > 30% should trigger a warning alert."""
    positions = _make_positions()
    # Technology sector is ~73% → expect a warning (possibly critical)
    alerts = service.generate_alerts(positions, portfolio_name="test")
    alert_types = [a.alert_type for a in alerts]
    severities = [a.severity for a in alerts]

    assert "concentration" in alert_types
    # Tech at ~73% is above 50% critical threshold
    assert "critical" in severities


def test_generate_alerts_no_issues(service):
    """Balanced portfolio across four sectors should produce no concentration alerts."""
    positions = _make_balanced_positions()
    alerts = service.generate_alerts(positions, portfolio_name="balanced")

    concentration_alerts = [a for a in alerts if a.alert_type == "concentration"]
    assert len(concentration_alerts) == 0


def test_generate_alerts_single_position_oversized(service):
    """A single position > 20% of portfolio should trigger a sizing warning."""
    positions = [
        {
            "symbol": "AAPL",
            "shares": 100,
            "cost_basis": 150.0,
            "current_price": 170.0,
            "sector": "Technology",
        },
        {
            "symbol": "JPM",
            "shares": 5,
            "cost_basis": 140.0,
            "current_price": 145.0,
            "sector": "Financials",
        },
    ]
    alerts = service.generate_alerts(positions, portfolio_name="test")
    sizing_alerts = [a for a in alerts if a.alert_type == "sizing"]
    assert len(sizing_alerts) >= 1


# ---------------------------------------------------------------------------
# check_position_risk tests
# ---------------------------------------------------------------------------


def test_check_position_risk(service):
    """Adding a new position should change projected metrics."""
    positions = _make_balanced_positions()
    current = service.compute_dashboard(positions)

    result = service.check_position_risk(
        portfolio_positions=positions,
        new_ticker="NVDA",
        new_shares=10,
        new_price=500.0,
    )

    assert "current" in result
    assert "projected" in result
    assert "new_position" in result

    # Projected value should be higher than current
    assert result["projected"]["total_value"] > result["current"]["total_value"]

    # New position metadata
    assert result["new_position"]["ticker"] == "NVDA"
    assert result["new_position"]["shares"] == 10
    assert result["new_position"]["position_value"] == pytest.approx(5000.0)


def test_check_position_risk_adds_to_existing(service):
    """Adding shares to an existing ticker should merge, not duplicate."""
    positions = _make_balanced_positions()  # AAPL is already there

    result = service.check_position_risk(
        portfolio_positions=positions,
        new_ticker="AAPL",
        new_shares=5,
        new_price=155.0,
    )

    # Projected position count should remain the same (merged)
    assert result["projected"]["position_count"] == len(positions)
