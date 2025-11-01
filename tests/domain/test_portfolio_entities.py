"""
Unit tests for portfolio domain entities.

Tests the pure business logic of Position and Portfolio entities without
any database or infrastructure dependencies.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from maverick_mcp.domain.portfolio import Portfolio, Position


class TestPosition:
    """Test suite for Position value object."""

    def test_position_creation(self):
        """Test creating a valid position."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=datetime.now(UTC),
        )

        assert pos.ticker == "AAPL"
        assert pos.shares == Decimal("10")
        assert pos.average_cost_basis == Decimal("150.00")
        assert pos.total_cost == Decimal("1500.00")

    def test_position_normalizes_ticker(self):
        """Test that ticker is normalized to uppercase."""
        pos = Position(
            ticker="aapl",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=datetime.now(UTC),
        )

        assert pos.ticker == "AAPL"

    def test_position_rejects_zero_shares(self):
        """Test that positions cannot have zero shares."""
        with pytest.raises(ValueError, match="Shares must be positive"):
            Position(
                ticker="AAPL",
                shares=Decimal("0"),
                average_cost_basis=Decimal("150.00"),
                total_cost=Decimal("1500.00"),
                purchase_date=datetime.now(UTC),
            )

    def test_position_rejects_negative_shares(self):
        """Test that positions cannot have negative shares."""
        with pytest.raises(ValueError, match="Shares must be positive"):
            Position(
                ticker="AAPL",
                shares=Decimal("-10"),
                average_cost_basis=Decimal("150.00"),
                total_cost=Decimal("1500.00"),
                purchase_date=datetime.now(UTC),
            )

    def test_position_rejects_zero_cost_basis(self):
        """Test that positions cannot have zero cost basis."""
        with pytest.raises(ValueError, match="Average cost basis must be positive"):
            Position(
                ticker="AAPL",
                shares=Decimal("10"),
                average_cost_basis=Decimal("0"),
                total_cost=Decimal("1500.00"),
                purchase_date=datetime.now(UTC),
            )

    def test_position_rejects_negative_total_cost(self):
        """Test that positions cannot have negative total cost."""
        with pytest.raises(ValueError, match="Total cost must be positive"):
            Position(
                ticker="AAPL",
                shares=Decimal("10"),
                average_cost_basis=Decimal("150.00"),
                total_cost=Decimal("-1500.00"),
                purchase_date=datetime.now(UTC),
            )

    def test_add_shares_averages_cost_basis(self):
        """Test that adding shares correctly averages the cost basis."""
        # Start with 10 shares @ $150
        pos = Position(
            ticker="AAPL",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=datetime.now(UTC),
        )

        # Add 10 shares @ $170
        pos = pos.add_shares(Decimal("10"), Decimal("170.00"), datetime.now(UTC))

        # Should have 20 shares @ $160 average
        assert pos.shares == Decimal("20")
        assert pos.average_cost_basis == Decimal("160.0000")
        assert pos.total_cost == Decimal("3200.00")

    def test_add_shares_updates_purchase_date(self):
        """Test that adding shares updates purchase date to earliest."""
        later_date = datetime.now(UTC)
        earlier_date = later_date - timedelta(days=30)

        pos = Position(
            ticker="AAPL",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=later_date,
        )

        pos = pos.add_shares(Decimal("10"), Decimal("170.00"), earlier_date)

        assert pos.purchase_date == earlier_date

    def test_add_shares_rejects_zero_shares(self):
        """Test that adding zero shares raises error."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="Shares to add must be positive"):
            pos.add_shares(Decimal("0"), Decimal("170.00"), datetime.now(UTC))

    def test_add_shares_rejects_zero_price(self):
        """Test that adding shares at zero price raises error."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="Price must be positive"):
            pos.add_shares(Decimal("10"), Decimal("0"), datetime.now(UTC))

    def test_remove_shares_partial(self):
        """Test removing part of a position."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        pos = pos.remove_shares(Decimal("10"))

        assert pos is not None
        assert pos.shares == Decimal("10")
        assert pos.average_cost_basis == Decimal("160.00")  # Unchanged
        assert pos.total_cost == Decimal("1600.00")

    def test_remove_shares_full(self):
        """Test removing entire position returns None."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        result = pos.remove_shares(Decimal("20"))

        assert result is None

    def test_remove_shares_more_than_held(self):
        """Test removing more shares than held closes position."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        result = pos.remove_shares(Decimal("25"))

        assert result is None

    def test_remove_shares_rejects_zero(self):
        """Test that removing zero shares raises error."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        with pytest.raises(ValueError, match="Shares to remove must be positive"):
            pos.remove_shares(Decimal("0"))

    def test_calculate_current_value_with_gain(self):
        """Test calculating current value with unrealized gain."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        metrics = pos.calculate_current_value(Decimal("175.50"))

        assert metrics["current_value"] == Decimal("3510.00")
        assert metrics["unrealized_pnl"] == Decimal("310.00")
        assert metrics["pnl_percentage"] == Decimal("9.69")

    def test_calculate_current_value_with_loss(self):
        """Test calculating current value with unrealized loss."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        metrics = pos.calculate_current_value(Decimal("145.00"))

        assert metrics["current_value"] == Decimal("2900.00")
        assert metrics["unrealized_pnl"] == Decimal("-300.00")
        assert metrics["pnl_percentage"] == Decimal("-9.38")

    def test_calculate_current_value_unchanged(self):
        """Test calculating current value when price unchanged."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("20"),
            average_cost_basis=Decimal("160.00"),
            total_cost=Decimal("3200.00"),
            purchase_date=datetime.now(UTC),
        )

        metrics = pos.calculate_current_value(Decimal("160.00"))

        assert metrics["current_value"] == Decimal("3200.00")
        assert metrics["unrealized_pnl"] == Decimal("0.00")
        assert metrics["pnl_percentage"] == Decimal("0.00")

    def test_fractional_shares(self):
        """Test that fractional shares are supported."""
        pos = Position(
            ticker="AAPL",
            shares=Decimal("10.5"),
            average_cost_basis=Decimal("150.25"),
            total_cost=Decimal("1577.625"),
            purchase_date=datetime.now(UTC),
        )

        assert pos.shares == Decimal("10.5")
        metrics = pos.calculate_current_value(Decimal("175.50"))
        assert metrics["current_value"] == Decimal("1842.75")

    def test_to_dict(self):
        """Test converting position to dictionary."""
        date = datetime.now(UTC)
        pos = Position(
            ticker="AAPL",
            shares=Decimal("10"),
            average_cost_basis=Decimal("150.00"),
            total_cost=Decimal("1500.00"),
            purchase_date=date,
            notes="Long-term hold",
        )

        result = pos.to_dict()

        assert result["ticker"] == "AAPL"
        assert result["shares"] == 10.0
        assert result["average_cost_basis"] == 150.0
        assert result["total_cost"] == 1500.0
        assert result["purchase_date"] == date.isoformat()
        assert result["notes"] == "Long-term hold"


class TestPortfolio:
    """Test suite for Portfolio aggregate root."""

    def test_portfolio_creation(self):
        """Test creating an empty portfolio."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        assert portfolio.portfolio_id == "test-id"
        assert portfolio.user_id == "default"
        assert portfolio.name == "My Portfolio"
        assert len(portfolio.positions) == 0

    def test_add_position_new(self):
        """Test adding a new position."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].ticker == "AAPL"
        assert portfolio.positions[0].shares == Decimal("10")

    def test_add_position_existing_averages(self):
        """Test that adding to existing position averages cost basis."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        # First purchase
        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        # Second purchase
        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("170.00"),
            date=datetime.now(UTC),
        )

        assert len(portfolio.positions) == 1  # Still one position
        assert portfolio.positions[0].shares == Decimal("20")
        assert portfolio.positions[0].average_cost_basis == Decimal("160.0000")

    def test_add_position_case_insensitive(self):
        """Test that ticker matching is case-insensitive."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="aapl",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("170.00"),
            date=datetime.now(UTC),
        )

        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].ticker == "AAPL"

    def test_remove_position_partial(self):
        """Test partially removing a position."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("20"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        result = portfolio.remove_position("AAPL", Decimal("10"))

        assert result is True
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].shares == Decimal("10")

    def test_remove_position_full(self):
        """Test fully removing a position."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("20"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        result = portfolio.remove_position("AAPL")

        assert result is True
        assert len(portfolio.positions) == 0

    def test_remove_position_nonexistent(self):
        """Test removing non-existent position returns False."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        result = portfolio.remove_position("AAPL")

        assert result is False

    def test_get_position(self):
        """Test getting a position by ticker."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        pos = portfolio.get_position("AAPL")

        assert pos is not None
        assert pos.ticker == "AAPL"

    def test_get_position_case_insensitive(self):
        """Test that get_position is case-insensitive."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        pos = portfolio.get_position("aapl")

        assert pos is not None
        assert pos.ticker == "AAPL"

    def test_get_position_nonexistent(self):
        """Test getting non-existent position returns None."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        pos = portfolio.get_position("AAPL")

        assert pos is None

    def test_get_total_invested(self):
        """Test calculating total capital invested."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        portfolio.add_position(
            ticker="MSFT",
            shares=Decimal("5"),
            price=Decimal("300.00"),
            date=datetime.now(UTC),
        )

        total = portfolio.get_total_invested()

        assert total == Decimal("3000.00")

    def test_calculate_portfolio_metrics(self):
        """Test calculating comprehensive portfolio metrics."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        portfolio.add_position(
            ticker="MSFT",
            shares=Decimal("5"),
            price=Decimal("300.00"),
            date=datetime.now(UTC),
        )

        current_prices = {
            "AAPL": Decimal("175.50"),
            "MSFT": Decimal("320.00"),
        }

        metrics = portfolio.calculate_portfolio_metrics(current_prices)

        assert metrics["total_value"] == 3355.0  # (10 * 175.50) + (5 * 320)
        assert metrics["total_invested"] == 3000.0
        assert metrics["total_pnl"] == 355.0
        assert metrics["total_pnl_percentage"] == 11.83
        assert metrics["position_count"] == 2
        assert len(metrics["positions"]) == 2

    def test_calculate_portfolio_metrics_uses_fallback_price(self):
        """Test that missing prices fall back to cost basis."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        # No current price provided
        metrics = portfolio.calculate_portfolio_metrics({})

        # Should use cost basis as current price
        assert metrics["total_value"] == 1500.0
        assert metrics["total_pnl"] == 0.0

    def test_clear_all_positions(self):
        """Test clearing all positions."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        portfolio.add_position(
            ticker="MSFT",
            shares=Decimal("5"),
            price=Decimal("300.00"),
            date=datetime.now(UTC),
        )

        portfolio.clear_all_positions()

        assert len(portfolio.positions) == 0

    def test_to_dict(self):
        """Test converting portfolio to dictionary."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        portfolio.add_position(
            ticker="AAPL",
            shares=Decimal("10"),
            price=Decimal("150.00"),
            date=datetime.now(UTC),
        )

        result = portfolio.to_dict()

        assert result["portfolio_id"] == "test-id"
        assert result["user_id"] == "default"
        assert result["name"] == "My Portfolio"
        assert result["position_count"] == 1
        assert result["total_invested"] == 1500.0
        assert len(result["positions"]) == 1

    def test_multiple_positions_with_different_performance(self):
        """Test portfolio with positions having different performance."""
        portfolio = Portfolio(
            portfolio_id="test-id",
            user_id="default",
            name="My Portfolio",
        )

        # Winner
        portfolio.add_position(
            ticker="NVDA",
            shares=Decimal("5"),
            price=Decimal("450.00"),
            date=datetime.now(UTC),
        )

        # Loser
        portfolio.add_position(
            ticker="MARA",
            shares=Decimal("50"),
            price=Decimal("18.50"),
            date=datetime.now(UTC),
        )

        current_prices = {
            "NVDA": Decimal("520.00"),  # +15.6%
            "MARA": Decimal("13.50"),  # -27.0%
        }

        metrics = portfolio.calculate_portfolio_metrics(current_prices)

        # Check individual positions
        nvda_pos = next(p for p in metrics["positions"] if p["ticker"] == "NVDA")
        mara_pos = next(p for p in metrics["positions"] if p["ticker"] == "MARA")

        assert nvda_pos["unrealized_pnl"] == 350.0  # (520 - 450) * 5
        assert mara_pos["unrealized_pnl"] == -250.0  # (13.50 - 18.50) * 50

        # Overall portfolio
        assert metrics["total_pnl"] == 100.0  # 350 - 250
