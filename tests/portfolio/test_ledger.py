"""Unit tests for maverick.portfolio.ledger: pure Decimal position math.

Ports the legacy behavior matrix from tests/domain/test_portfolio_entities.py
(Position/Portfolio dataclasses) onto the new functional API operating on
PositionPayload. Scope note: the legacy Portfolio aggregate's list-management
behavior (find-by-ticker, add/remove within a collection, to_dict, portfolio
creation/clearing) is orchestration that belongs to service.py (Task 6), not
to these four pure functions, so those specific legacy tests do not have a
direct analog here. Every legacy Position-level behavior (averaging,
quantization, earliest-date-wins, partial/full close, current-value P&L,
fractional shares, validation on non-positive inputs) and every
Portfolio-level aggregation behavior (total invested, portfolio metrics,
fallback pricing, multi-position winner/loser) is ported below.

All monetary/quantity assertions are exact Decimal comparisons. No
pytest.approx anywhere in this file.
"""

from decimal import Decimal

import pytest

from maverick.portfolio.ledger import (
    add_shares,
    portfolio_metrics,
    position_value,
    remove_shares,
)
from maverick.portfolio.types import PositionPayload


def _position(
    ticker: str = "AAPL",
    shares: str = "10",
    average_cost_basis: str = "150.00",
    total_cost: str = "1500.00",
    purchase_date: str = "2026-01-01",
    notes: str | None = None,
) -> PositionPayload:
    return PositionPayload(
        ticker=ticker,
        shares=Decimal(shares),
        average_cost_basis=Decimal(average_cost_basis),
        total_cost=Decimal(total_cost),
        purchase_date=purchase_date,
        notes=notes,
    )


class TestAddSharesNewPosition:
    """add_shares(None, ...) creates a fresh position."""

    def test_creates_new_position(self):
        pos = add_shares(None, "AAPL", Decimal("10"), Decimal("150.00"), "2026-01-01")

        assert pos.ticker == "AAPL"
        assert pos.shares == Decimal("10")
        assert pos.average_cost_basis == Decimal("150.00")
        assert pos.total_cost == Decimal("1500.00")
        assert pos.purchase_date == "2026-01-01"

    def test_normalizes_ticker_to_uppercase(self):
        pos = add_shares(None, "aapl", Decimal("10"), Decimal("150.00"), "2026-01-01")

        assert pos.ticker == "AAPL"

    def test_carries_notes(self):
        pos = add_shares(
            None,
            "AAPL",
            Decimal("10"),
            Decimal("150.00"),
            "2026-01-01",
            notes="Long-term hold",
        )

        assert pos.notes == "Long-term hold"

    def test_rejects_zero_shares(self):
        with pytest.raises(ValueError, match="Shares to add must be positive"):
            add_shares(None, "AAPL", Decimal("0"), Decimal("150.00"), "2026-01-01")

    def test_rejects_negative_shares(self):
        with pytest.raises(ValueError, match="Shares to add must be positive"):
            add_shares(None, "AAPL", Decimal("-10"), Decimal("150.00"), "2026-01-01")

    def test_rejects_zero_price(self):
        with pytest.raises(ValueError, match="Price must be positive"):
            add_shares(None, "AAPL", Decimal("10"), Decimal("0"), "2026-01-01")

    def test_rejects_negative_price(self):
        with pytest.raises(ValueError, match="Price must be positive"):
            add_shares(None, "AAPL", Decimal("10"), Decimal("-150.00"), "2026-01-01")


class TestAddSharesExistingPosition:
    """add_shares(position, ...) averages into an existing position."""

    def test_averages_cost_basis(self):
        pos = _position(shares="10", average_cost_basis="150.00", total_cost="1500.00")

        pos = add_shares(pos, "AAPL", Decimal("10"), Decimal("170.00"), "2026-01-02")

        assert pos.shares == Decimal("20")
        assert pos.average_cost_basis == Decimal("160.0000")
        assert pos.total_cost == Decimal("3200.00")

    def test_uses_stored_total_cost_not_shares_times_basis(self):
        # total_cost is carried forward from the stored value plus the new
        # lot's cost, never recomputed as shares * average_cost_basis.
        pos = _position(shares="10", average_cost_basis="150.00", total_cost="1500.00")

        pos = add_shares(pos, "AAPL", Decimal("5"), Decimal("100.00"), "2026-01-02")

        assert pos.total_cost == Decimal("2000.00")  # 1500.00 + (5 * 100.00)

    def test_updates_purchase_date_to_earlier_date(self):
        pos = _position(purchase_date="2026-01-15")

        pos = add_shares(pos, "AAPL", Decimal("10"), Decimal("170.00"), "2026-01-01")

        assert pos.purchase_date == "2026-01-01"

    def test_keeps_purchase_date_when_new_date_is_later(self):
        pos = _position(purchase_date="2026-01-01")

        pos = add_shares(pos, "AAPL", Decimal("10"), Decimal("170.00"), "2026-01-15")

        assert pos.purchase_date == "2026-01-01"

    def test_ignores_notes_param_and_preserves_existing_notes(self):
        pos = _position(notes="Original notes")

        pos = add_shares(
            pos,
            "AAPL",
            Decimal("10"),
            Decimal("170.00"),
            "2026-01-02",
            notes="New notes",
        )

        assert pos.notes == "Original notes"

    def test_rejects_zero_shares(self):
        pos = _position()

        with pytest.raises(ValueError, match="Shares to add must be positive"):
            add_shares(pos, "AAPL", Decimal("0"), Decimal("170.00"), "2026-01-02")

    def test_rejects_negative_shares(self):
        pos = _position()

        with pytest.raises(ValueError, match="Shares to add must be positive"):
            add_shares(pos, "AAPL", Decimal("-5"), Decimal("170.00"), "2026-01-02")

    def test_rejects_zero_price(self):
        pos = _position()

        with pytest.raises(ValueError, match="Price must be positive"):
            add_shares(pos, "AAPL", Decimal("10"), Decimal("0"), "2026-01-02")

    def test_rejects_negative_price(self):
        pos = _position()

        with pytest.raises(ValueError, match="Price must be positive"):
            add_shares(pos, "AAPL", Decimal("10"), Decimal("-170.00"), "2026-01-02")

    def test_average_cost_basis_quantizes_half_up_not_half_even(self):
        # 1.0002 / 4 = 0.25005 exactly -> ROUND_HALF_UP gives 0.2501;
        # ROUND_HALF_EVEN (Python/Decimal's default) would give 0.2500.
        # This proves the ledger explicitly forces ROUND_HALF_UP.
        pos = _position(
            shares="3",
            average_cost_basis="0.3333",
            total_cost="1.0000",
        )

        pos = add_shares(pos, "AAPL", Decimal("1"), Decimal("0.0002"), "2026-01-02")

        assert pos.shares == Decimal("4")
        assert pos.total_cost == Decimal("1.0002")
        assert pos.average_cost_basis == Decimal("0.2501")


class TestRemoveShares:
    def test_partial_removal_keeps_basis(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        updated, result = remove_shares(pos, Decimal("10"))

        assert updated is not None
        assert updated.shares == Decimal("10")
        assert updated.average_cost_basis == Decimal("160.00")  # unchanged
        assert updated.total_cost == Decimal("1600.00")
        assert result.ticker == "AAPL"
        assert result.shares_removed == Decimal("10")
        assert result.position_fully_closed is False

    def test_full_removal_exact_amount_closes_position(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        updated, result = remove_shares(pos, Decimal("20"))

        assert updated is None
        assert result.ticker == "AAPL"
        assert result.shares_removed == Decimal("20")
        assert result.position_fully_closed is True

    def test_removing_more_than_held_closes_position(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        updated, result = remove_shares(pos, Decimal("25"))

        assert updated is None
        # only what was actually held is reported as removed
        assert result.shares_removed == Decimal("20")
        assert result.position_fully_closed is True

    def test_none_shares_closes_entire_position(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        updated, result = remove_shares(pos, None)

        assert updated is None
        assert result.shares_removed == Decimal("20")
        assert result.position_fully_closed is True

    def test_rejects_zero_shares(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        with pytest.raises(ValueError, match="Shares to remove must be positive"):
            remove_shares(pos, Decimal("0"))

    def test_rejects_negative_shares(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        with pytest.raises(ValueError, match="Shares to remove must be positive"):
            remove_shares(pos, Decimal("-5"))

    def test_partial_removal_preserves_notes_and_date(self):
        pos = _position(
            shares="20",
            average_cost_basis="160.00",
            total_cost="3200.00",
            purchase_date="2026-01-01",
            notes="keep me",
        )

        updated, _result = remove_shares(pos, Decimal("5"))

        assert updated is not None
        assert updated.purchase_date == "2026-01-01"
        assert updated.notes == "keep me"


class TestPositionValue:
    def test_current_value_with_gain(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        value, pnl, pnl_percent = position_value(pos, Decimal("175.50"))

        assert value == Decimal("3510.00")
        assert pnl == Decimal("310.00")
        assert pnl_percent == Decimal("9.69")

    def test_current_value_with_loss(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        value, pnl, pnl_percent = position_value(pos, Decimal("145.00"))

        assert value == Decimal("2900.00")
        assert pnl == Decimal("-300.00")
        assert pnl_percent == Decimal("-9.38")

    def test_current_value_unchanged(self):
        pos = _position(shares="20", average_cost_basis="160.00", total_cost="3200.00")

        value, pnl, pnl_percent = position_value(pos, Decimal("160.00"))

        assert value == Decimal("3200.00")
        assert pnl == Decimal("0.00")
        assert pnl_percent == Decimal("0.00")

    def test_fractional_shares(self):
        pos = _position(
            shares="10.5",
            average_cost_basis="150.25",
            total_cost="1577.625",
        )

        value, pnl, pnl_percent = position_value(pos, Decimal("175.50"))

        assert value == Decimal("1842.75")
        assert pnl == Decimal("265.13")
        assert pnl_percent == Decimal("16.81")

    def test_zero_cost_is_safe_and_returns_zero_percent(self):
        # PositionPayload has no field validators, so a directly constructed
        # zero-total_cost payload is a valid input the ledger must not
        # divide-by-zero on.
        pos = _position(shares="10", average_cost_basis="0.00", total_cost="0.00")

        value, pnl, pnl_percent = position_value(pos, Decimal("50.00"))

        assert value == Decimal("500.00")
        assert pnl == Decimal("500.00")
        assert pnl_percent == Decimal("0.00")

    def test_pnl_percent_quantizes_half_up_not_half_even(self):
        # 1.00 / 800.00 * 100 = 0.125 exactly -> ROUND_HALF_UP gives 0.13;
        # ROUND_HALF_EVEN would give 0.12.
        pos = _position(shares="1", average_cost_basis="800.00", total_cost="800.00")

        value, pnl, pnl_percent = position_value(pos, Decimal("801.00"))

        assert value == Decimal("801.00")
        assert pnl == Decimal("1.00")
        assert pnl_percent == Decimal("0.13")


class TestPortfolioMetrics:
    def test_empty_positions(self):
        metrics = portfolio_metrics([], {})

        assert metrics.total_invested == Decimal("0")
        assert metrics.total_value == 0.0
        assert metrics.total_pnl == 0.0
        assert metrics.total_pnl_percent == 0.00
        assert metrics.position_count == 0

    def test_aggregates_two_positions(self):
        positions = [
            _position(
                ticker="AAPL",
                shares="10",
                average_cost_basis="150.00",
                total_cost="1500.00",
            ),
            _position(
                ticker="MSFT",
                shares="5",
                average_cost_basis="300.00",
                total_cost="1500.00",
            ),
        ]
        prices = {"AAPL": Decimal("175.50"), "MSFT": Decimal("320.00")}

        metrics = portfolio_metrics(positions, prices)

        assert metrics.total_invested == Decimal("3000.00")
        assert metrics.total_value == 3355.00
        assert metrics.total_pnl == 355.00
        assert metrics.total_pnl_percent == 11.83
        assert metrics.position_count == 2

    def test_falls_back_to_cost_basis_when_price_missing(self):
        positions = [
            _position(
                ticker="AAPL",
                shares="10",
                average_cost_basis="150.00",
                total_cost="1500.00",
            ),
        ]

        metrics = portfolio_metrics(positions, {})

        assert metrics.total_value == 1500.00
        assert metrics.total_pnl == 0.0

    def test_multiple_positions_with_different_performance(self):
        positions = [
            _position(
                ticker="NVDA",
                shares="5",
                average_cost_basis="450.00",
                total_cost="2250.00",
            ),
            _position(
                ticker="MARA",
                shares="50",
                average_cost_basis="18.50",
                total_cost="925.00",
            ),
        ]
        prices = {"NVDA": Decimal("520.00"), "MARA": Decimal("13.50")}

        metrics = portfolio_metrics(positions, prices)

        assert metrics.total_pnl == 100.00  # 350 (NVDA) - 250 (MARA)
        assert metrics.position_count == 2

    def test_total_invested_sums_stored_total_cost(self):
        positions = [
            _position(
                ticker="AAPL",
                shares="10",
                average_cost_basis="150.00",
                total_cost="1500.00",
            ),
            _position(
                ticker="MSFT",
                shares="5",
                average_cost_basis="300.00",
                total_cost="1500.00",
            ),
        ]

        metrics = portfolio_metrics(positions, {})

        assert metrics.total_invested == Decimal("3000.00")
