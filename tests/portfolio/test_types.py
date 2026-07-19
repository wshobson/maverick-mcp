"""Tests for maverick.portfolio.types."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from maverick.portfolio.types import (
    ComparisonResult,
    CorrelationResult,
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionPayload,
    PositionWithPrice,
    RemoveResult,
    RiskAnalysis,
)


def _make_position(**overrides) -> PositionPayload:
    fields = {
        "ticker": "AAPL",
        "shares": Decimal("10.0001"),
        "average_cost_basis": Decimal("150.25"),
        "total_cost": Decimal("1502.75"),
        "purchase_date": "2026-01-15",
        "notes": "Long-term hold",
    }
    fields.update(overrides)
    return PositionPayload(**fields)


# -- PositionPayload: Decimal preservation --------------------------------


def test_position_payload_preserves_decimal_shares_exactly():
    position = _make_position(shares=Decimal("10.0001"))
    assert position.shares == Decimal("10.0001")
    assert str(position.shares) == "10.0001"


def test_position_payload_model_validate_preserves_decimal_no_float_drift():
    data = {
        "ticker": "AAPL",
        "shares": Decimal("10.0001"),
        "average_cost_basis": Decimal("150.2500"),
        "total_cost": Decimal("1502.75"),
        "purchase_date": "2026-01-15",
        "notes": None,
    }
    position = PositionPayload.model_validate(data)
    assert position.shares == Decimal("10.0001")
    assert position.average_cost_basis == Decimal("150.2500")
    assert position.total_cost == Decimal("1502.75")
    # Exact string form is preserved -- no binary float round-trip drift.
    assert str(position.shares) == "10.0001"
    assert str(position.average_cost_basis) == "150.2500"


def test_position_payload_notes_optional():
    position = _make_position(notes=None)
    assert position.notes is None


def test_position_payload_round_trips_through_model_dump():
    position = _make_position()
    data = position.model_dump()
    assert data["shares"] == Decimal("10.0001")
    restored = PositionPayload.model_validate(data)
    assert restored == position
    assert restored.shares == Decimal("10.0001")


def test_position_payload_model_dump_json_mode_serializes_decimal_as_string():
    """Documents the JSON boundary: mode="json" turns Decimal into str, not float.

    This is the observed pydantic v2 behavior -- Decimal fields serialize to
    their exact string representation under mode="json", not a float. This
    is what preserves precision across the tools JSON boundary.
    """
    position = _make_position()
    json_data = position.model_dump(mode="json")
    assert json_data["shares"] == "10.0001"
    assert isinstance(json_data["shares"], str)
    assert json_data["average_cost_basis"] == "150.25"
    assert isinstance(json_data["average_cost_basis"], str)
    assert json_data["total_cost"] == "1502.75"
    assert isinstance(json_data["total_cost"], str)


# -- PositionWithPrice ------------------------------------------------------


def test_position_with_price_extends_position_payload_fields():
    position = PositionWithPrice(
        ticker="AAPL",
        shares=Decimal("10.0001"),
        average_cost_basis=Decimal("150.25"),
        total_cost=Decimal("1502.75"),
        purchase_date="2026-01-15",
        notes=None,
        current_price=175.50,
        current_value=1755.02,
        unrealized_pnl=252.27,
        unrealized_pnl_percent=16.79,
    )
    assert position.shares == Decimal("10.0001")
    assert position.current_price == 175.50
    assert position.current_value == 1755.02
    assert position.unrealized_pnl == 252.27
    assert position.unrealized_pnl_percent == 16.79


def test_position_with_price_allows_none_price_fields_when_price_unavailable():
    position = PositionWithPrice(
        ticker="AAPL",
        shares=Decimal("10"),
        average_cost_basis=Decimal("150.25"),
        total_cost=Decimal("1502.50"),
        purchase_date="2026-01-15",
        notes=None,
        current_price=None,
        current_value=None,
        unrealized_pnl=None,
        unrealized_pnl_percent=None,
    )
    assert position.current_price is None
    assert position.current_value is None
    assert position.unrealized_pnl is None
    assert position.unrealized_pnl_percent is None


def test_position_with_price_round_trips_through_model_dump():
    position = PositionWithPrice(
        ticker="AAPL",
        shares=Decimal("10.0001"),
        average_cost_basis=Decimal("150.25"),
        total_cost=Decimal("1502.75"),
        purchase_date="2026-01-15",
        notes=None,
        current_price=175.50,
        current_value=1755.02,
        unrealized_pnl=252.27,
        unrealized_pnl_percent=16.79,
    )
    data = position.model_dump()
    restored = PositionWithPrice.model_validate(data)
    assert restored == position
    assert restored.shares == Decimal("10.0001")


# -- PortfolioMetrics ---------------------------------------------------


def test_portfolio_metrics_preserves_decimal_total_invested():
    metrics = PortfolioMetrics(
        total_invested=Decimal("1502.75"),
        total_value=1755.02,
        total_pnl=252.27,
        total_pnl_percent=16.79,
        position_count=1,
    )
    assert metrics.total_invested == Decimal("1502.75")
    assert str(metrics.total_invested) == "1502.75"


def test_portfolio_metrics_allows_none_when_prices_unavailable():
    metrics = PortfolioMetrics(
        total_invested=Decimal("1502.75"),
        total_value=None,
        total_pnl=None,
        total_pnl_percent=None,
        position_count=1,
    )
    assert metrics.total_value is None
    assert metrics.total_pnl is None
    assert metrics.total_pnl_percent is None


def test_portfolio_metrics_round_trips_through_model_dump():
    metrics = PortfolioMetrics(
        total_invested=Decimal("1502.75"),
        total_value=1755.02,
        total_pnl=252.27,
        total_pnl_percent=16.79,
        position_count=1,
    )
    data = metrics.model_dump()
    restored = PortfolioMetrics.model_validate(data)
    assert restored == metrics
    assert restored.total_invested == Decimal("1502.75")


# -- PortfolioSnapshot: composition --------------------------------------


def test_portfolio_snapshot_composes_positions_and_metrics():
    position = PositionWithPrice(
        ticker="AAPL",
        shares=Decimal("10.0001"),
        average_cost_basis=Decimal("150.25"),
        total_cost=Decimal("1502.75"),
        purchase_date="2026-01-15",
        notes=None,
        current_price=175.50,
        current_value=1755.02,
        unrealized_pnl=252.27,
        unrealized_pnl_percent=16.79,
    )
    metrics = PortfolioMetrics(
        total_invested=Decimal("1502.75"),
        total_value=1755.02,
        total_pnl=252.27,
        total_pnl_percent=16.79,
        position_count=1,
    )
    snapshot = PortfolioSnapshot(
        user_id="default",
        name="My Portfolio",
        positions=[position],
        metrics=metrics,
        as_of="2026-07-19T00:00:00+00:00",
    )
    assert snapshot.user_id == "default"
    assert snapshot.name == "My Portfolio"
    assert snapshot.positions == [position]
    assert snapshot.positions[0].shares == Decimal("10.0001")
    assert snapshot.metrics == metrics
    assert snapshot.metrics.total_invested == Decimal("1502.75")
    assert snapshot.as_of == "2026-07-19T00:00:00+00:00"


def test_portfolio_snapshot_allows_empty_positions():
    metrics = PortfolioMetrics(
        total_invested=Decimal("0"),
        total_value=None,
        total_pnl=None,
        total_pnl_percent=None,
        position_count=0,
    )
    snapshot = PortfolioSnapshot(
        user_id="default",
        name="My Portfolio",
        positions=[],
        metrics=metrics,
        as_of="2026-07-19T00:00:00+00:00",
    )
    assert snapshot.positions == []


def test_portfolio_snapshot_round_trips_through_model_dump():
    position = PositionWithPrice(
        ticker="AAPL",
        shares=Decimal("10.0001"),
        average_cost_basis=Decimal("150.25"),
        total_cost=Decimal("1502.75"),
        purchase_date="2026-01-15",
        notes=None,
        current_price=175.50,
        current_value=1755.02,
        unrealized_pnl=252.27,
        unrealized_pnl_percent=16.79,
    )
    metrics = PortfolioMetrics(
        total_invested=Decimal("1502.75"),
        total_value=1755.02,
        total_pnl=252.27,
        total_pnl_percent=16.79,
        position_count=1,
    )
    snapshot = PortfolioSnapshot(
        user_id="default",
        name="My Portfolio",
        positions=[position],
        metrics=metrics,
        as_of="2026-07-19T00:00:00+00:00",
    )
    data = snapshot.model_dump()
    restored = PortfolioSnapshot.model_validate(data)
    assert restored == snapshot
    assert restored.positions[0].shares == Decimal("10.0001")
    assert restored.metrics.total_invested == Decimal("1502.75")


def test_portfolio_snapshot_model_dump_json_mode_serializes_nested_decimal_as_string():
    position = _make_position()
    position_with_price = PositionWithPrice(
        **position.model_dump(),
        current_price=None,
        current_value=None,
        unrealized_pnl=None,
        unrealized_pnl_percent=None,
    )
    metrics = PortfolioMetrics(
        total_invested=Decimal("1502.75"),
        total_value=None,
        total_pnl=None,
        total_pnl_percent=None,
        position_count=1,
    )
    snapshot = PortfolioSnapshot(
        user_id="default",
        name="My Portfolio",
        positions=[position_with_price],
        metrics=metrics,
        as_of="2026-07-19T00:00:00+00:00",
    )
    json_data = snapshot.model_dump(mode="json")
    assert json_data["positions"][0]["shares"] == "10.0001"
    assert json_data["metrics"]["total_invested"] == "1502.75"


# -- RemoveResult ---------------------------------------------------------


def test_remove_result_preserves_decimal_shares_removed():
    result = RemoveResult(
        ticker="AAPL",
        shares_removed=Decimal("5.0001"),
        position_fully_closed=False,
    )
    assert result.shares_removed == Decimal("5.0001")
    assert result.position_fully_closed is False


def test_remove_result_full_close():
    result = RemoveResult(
        ticker="AAPL",
        shares_removed=Decimal("10"),
        position_fully_closed=True,
    )
    assert result.position_fully_closed is True


def test_remove_result_round_trips_through_model_dump():
    result = RemoveResult(
        ticker="AAPL",
        shares_removed=Decimal("5.0001"),
        position_fully_closed=False,
    )
    data = result.model_dump()
    restored = RemoveResult.model_validate(data)
    assert restored == result
    assert restored.shares_removed == Decimal("5.0001")


# -- ComparisonResult (advisory floats) ------------------------------------


def test_comparison_result_holds_per_ticker_metrics_and_rankings():
    result = ComparisonResult(
        comparison={
            "AAPL": {
                "current_price": 175.50,
                "performance": {"price_change_pct": 5.2},
                "technical": {"rsi": 65.0, "trend_strength": 6},
                "rankings": {"performance_rank": 1, "trend_rank": 1},
            },
            "MSFT": {
                "current_price": 410.0,
                "performance": {"price_change_pct": 2.1},
                "technical": {"rsi": 55.0, "trend_strength": 4},
                "rankings": {"performance_rank": 2, "trend_rank": 2},
            },
        },
        best_performer="AAPL",
        strongest_trend="AAPL",
        period_days=90,
        as_of="2026-07-19T00:00:00+00:00",
        portfolio_context=None,
    )
    assert result.comparison["AAPL"]["current_price"] == 175.50
    assert result.best_performer == "AAPL"
    assert result.strongest_trend == "AAPL"
    assert result.period_days == 90
    assert result.as_of == "2026-07-19T00:00:00+00:00"
    assert result.portfolio_context is None


def test_comparison_result_optional_portfolio_context():
    result = ComparisonResult(
        comparison={},
        best_performer="AAPL",
        strongest_trend="AAPL",
        period_days=90,
        as_of="2026-07-19T00:00:00+00:00",
        portfolio_context={"using_portfolio": True, "portfolio_name": "My Portfolio"},
    )
    assert result.portfolio_context == {
        "using_portfolio": True,
        "portfolio_name": "My Portfolio",
    }


def test_comparison_result_round_trips_through_model_dump():
    result = ComparisonResult(
        comparison={"AAPL": {"current_price": 175.50}},
        best_performer="AAPL",
        strongest_trend="AAPL",
        period_days=90,
        as_of="2026-07-19T00:00:00+00:00",
        portfolio_context=None,
    )
    data = result.model_dump()
    assert ComparisonResult.model_validate(data) == result


# -- CorrelationResult (advisory floats) -----------------------------------


def test_correlation_result_holds_matrix_and_pairs():
    result = CorrelationResult(
        matrix={
            "AAPL": {"AAPL": 1.0, "MSFT": 0.82},
            "MSFT": {"AAPL": 0.82, "MSFT": 1.0},
        },
        high_correlation_pairs=[
            {"pair": ("AAPL", "MSFT"), "correlation": 0.82},
        ],
        hedges=[],
        average_correlation=0.82,
        diversification_score=18.0,
        recommendation="Consider adding uncorrelated assets",
        period_days=252,
        data_points=200,
    )
    assert result.matrix["AAPL"]["MSFT"] == 0.82
    assert result.high_correlation_pairs[0]["correlation"] == 0.82
    assert result.hedges == []
    assert result.average_correlation == 0.82
    assert result.diversification_score == 18.0
    assert result.recommendation == "Consider adding uncorrelated assets"
    assert result.period_days == 252
    assert result.data_points == 200
    assert result.portfolio_context is None


def test_correlation_result_optional_portfolio_context():
    result = CorrelationResult(
        matrix={"AAPL": {"AAPL": 1.0}},
        high_correlation_pairs=[],
        hedges=[],
        average_correlation=0.1,
        diversification_score=90.0,
        recommendation="Well diversified",
        period_days=252,
        data_points=200,
        portfolio_context={"using_portfolio": True, "portfolio_name": "My Portfolio"},
    )
    assert result.portfolio_context == {
        "using_portfolio": True,
        "portfolio_name": "My Portfolio",
    }


def test_correlation_result_round_trips_through_model_dump():
    result = CorrelationResult(
        matrix={"AAPL": {"AAPL": 1.0}},
        high_correlation_pairs=[],
        hedges=[{"pair": ("AAPL", "TLT"), "correlation": -0.4}],
        average_correlation=-0.4,
        diversification_score=70.0,
        recommendation="Well diversified",
        period_days=252,
        data_points=200,
    )
    data = result.model_dump()
    assert CorrelationResult.model_validate(data) == result


# -- RiskAnalysis (advisory floats) ----------------------------------------


def test_risk_analysis_holds_sizing_stop_entry_targets():
    result = RiskAnalysis(
        ticker="AAPL",
        current_price=175.50,
        atr=3.25,
        risk_level=50.0,
        position_sizing={"suggested_position_size": 500.0, "max_shares": 2},
        stop_loss={"stop_loss": 170.0, "stop_loss_percent": 3.1},
        entry_strategy={"immediate_entry": 175.50, "scale_in_levels": [175.50, 172.0]},
        targets={"price_target": 190.0, "risk_reward_ratio": 3.0},
        existing_position=None,
    )
    assert result.ticker == "AAPL"
    assert result.current_price == 175.50
    assert result.atr == 3.25
    assert result.risk_level == 50.0
    assert result.position_sizing["max_shares"] == 2
    assert result.stop_loss["stop_loss"] == 170.0
    assert result.entry_strategy["immediate_entry"] == 175.50
    assert result.targets["risk_reward_ratio"] == 3.0
    assert result.analysis is None
    assert result.existing_position is None


def test_risk_analysis_optional_analysis_block():
    result = RiskAnalysis(
        ticker="AAPL",
        current_price=175.50,
        atr=3.25,
        risk_level=50.0,
        position_sizing={},
        stop_loss={},
        entry_strategy={},
        targets={},
        analysis={"confidence_score": 35.0, "strategy_type": "moderate"},
        existing_position=None,
    )
    assert result.analysis == {"confidence_score": 35.0, "strategy_type": "moderate"}


def test_risk_analysis_optional_existing_position_block():
    result = RiskAnalysis(
        ticker="AAPL",
        current_price=175.50,
        atr=3.25,
        risk_level=50.0,
        position_sizing={},
        stop_loss={},
        entry_strategy={},
        targets={},
        existing_position={
            "shares_owned": 10.0,
            "average_cost_basis": 150.25,
            "unrealized_pnl": 252.27,
        },
    )
    assert result.existing_position is not None
    assert result.existing_position["shares_owned"] == 10.0


def test_risk_analysis_round_trips_through_model_dump():
    result = RiskAnalysis(
        ticker="AAPL",
        current_price=175.50,
        atr=3.25,
        risk_level=50.0,
        position_sizing={"suggested_position_size": 500.0},
        stop_loss={"stop_loss": 170.0},
        entry_strategy={"immediate_entry": 175.50},
        targets={"price_target": 190.0},
        analysis={"confidence_score": 35.0},
        existing_position=None,
    )
    data = result.model_dump()
    assert RiskAnalysis.model_validate(data) == result


# -- Validation: required fields ------------------------------------------


def test_position_payload_requires_ticker():
    with pytest.raises(ValidationError):
        PositionPayload.model_validate(
            {
                "shares": Decimal("1"),
                "average_cost_basis": Decimal("1"),
                "total_cost": Decimal("1"),
                "purchase_date": "2026-01-15",
                "notes": None,
            }
        )


# -- Construction validation: ported from tests/domain/test_portfolio_entities.py
# (TestPosition.test_position_rejects_{zero,negative}_shares,
# test_position_rejects_zero_cost_basis, test_position_rejects_negative_total_cost).
# The legacy dataclass raised ValueError with a specific message; PositionPayload's
# Field(gt=0) constraints raise pydantic's ValidationError instead, so these assert
# on the pydantic error rather than the legacy message text.


def test_position_payload_rejects_zero_shares():
    with pytest.raises(ValidationError):
        _make_position(shares=Decimal("0"))


def test_position_payload_rejects_negative_shares():
    with pytest.raises(ValidationError):
        _make_position(shares=Decimal("-10"))


def test_position_payload_rejects_zero_cost_basis():
    with pytest.raises(ValidationError):
        _make_position(average_cost_basis=Decimal("0"))


def test_position_payload_rejects_negative_total_cost():
    with pytest.raises(ValidationError):
        _make_position(total_cost=Decimal("-1500.00"))
