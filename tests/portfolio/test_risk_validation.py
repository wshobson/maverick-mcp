"""Reject invalid prospective positions before producing a risk dashboard."""

from typing import Any

import pytest

from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.risk import check_position_risk, regime_adjusted_size


@pytest.mark.parametrize("field", ["account_size", "entry_price", "stop_loss"])
@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), -float("inf")])
def test_regime_sizing_rejects_invalid_positive_inputs(field, value):
    arguments: dict[str, Any] = {
        "account_size": 10000,
        "entry_price": 100,
        "stop_loss": 95,
        "risk_pct": 2,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        regime_adjusted_size(**arguments, regime="bull", settings=PortfolioSettings())


@pytest.mark.parametrize("risk_pct", [-1, float("nan"), float("inf"), -float("inf")])
def test_regime_sizing_rejects_invalid_risk_percentage(risk_pct):
    with pytest.raises(ValueError, match="risk_pct"):
        regime_adjusted_size(10000, 100, 95, risk_pct, "bull", PortfolioSettings())


def test_regime_sizing_keeps_zero_risk_and_equal_stop_contract():
    settings = PortfolioSettings()
    assert regime_adjusted_size(10000, 100, 95, 0, "bull", settings).shares == 0
    assert regime_adjusted_size(10000, 100, 100, 2, "bull", settings).shares == 0


@pytest.mark.parametrize("field", ["new_shares", "new_price"])
@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), -float("inf")])
def test_pretrade_risk_rejects_invalid_position_inputs(field, value):
    arguments: dict[str, Any] = {"new_shares": 1, "new_price": 100}
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        check_position_risk([], "AAPL", **arguments, settings=PortfolioSettings())
