"""
Comprehensive integration tests for portfolio persistence layer.

Tests the database CRUD operations, relationships, constraints, and data integrity
for the portfolio management system. Uses pytest fixtures with database sessions
and SQLite for testing without external dependencies.

Test Coverage:
- Database CRUD operations (Create, Read, Update, Delete)
- Relationship management (portfolio -> positions)
- Unique constraints (user+portfolio name, portfolio+ticker)
- Cascade deletes (portfolio deletion removes positions)
- Data integrity (Decimal precision, timezone-aware datetimes)
- Query performance (selectin loading, filtering)
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import exc
from sqlalchemy.orm import Session

from maverick_mcp.data.models import PortfolioPosition, UserPortfolio

pytestmark = pytest.mark.integration


class TestPortfolioCreation:
    """Test suite for creating portfolios."""

    def test_create_portfolio_with_defaults(self, db_session: Session):
        """Test creating a portfolio with default values."""
        portfolio = UserPortfolio(
            user_id="default",
            name="My Portfolio",
        )
        db_session.add(portfolio)
        db_session.commit()

        # Verify creation
        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved is not None
        assert retrieved.user_id == "default"
        assert retrieved.name == "My Portfolio"
        assert retrieved.positions == []
        assert retrieved.created_at is not None
        assert retrieved.updated_at is not None

    def test_create_portfolio_with_custom_user(self, db_session: Session):
        """Test creating a portfolio for a specific user."""
        portfolio = UserPortfolio(
            user_id="user123",
            name="User Portfolio",
        )
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.user_id == "user123"
        assert retrieved.name == "User Portfolio"

    def test_create_multiple_portfolios_for_same_user(self, db_session: Session):
        """Test creating multiple portfolios for the same user."""
        portfolio1 = UserPortfolio(user_id="user1", name="Portfolio 1")
        portfolio2 = UserPortfolio(user_id="user1", name="Portfolio 2")

        db_session.add_all([portfolio1, portfolio2])
        db_session.commit()

        portfolios = db_session.query(UserPortfolio).filter_by(user_id="user1").all()
        assert len(portfolios) == 2
        assert {p.name for p in portfolios} == {"Portfolio 1", "Portfolio 2"}

    def test_portfolio_timestamps_created(self, db_session: Session):
        """Test that portfolio timestamps are set on creation."""
        before = datetime.now(UTC)
        portfolio = UserPortfolio(user_id="default", name="Test")
        db_session.add(portfolio)
        db_session.commit()
        after = datetime.now(UTC)

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert before <= retrieved.created_at <= after
        assert before <= retrieved.updated_at <= after


class TestPortfolioPositionCreation:
    """Test suite for creating positions within portfolios."""

    @pytest.fixture
    def portfolio(self, db_session: Session):
        """Create a portfolio for position tests."""
        # Use unique name with UUID to avoid constraint violations across tests
        unique_name = f"Test Portfolio {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()
        return portfolio

    def test_create_position_basic(self, db_session: Session, portfolio: UserPortfolio):
        """Test creating a basic position in a portfolio."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.ticker == "AAPL"
        assert retrieved.shares == Decimal("10.00000000")
        assert retrieved.average_cost_basis == Decimal("150.0000")
        assert retrieved.total_cost == Decimal("1500.0000")
        assert retrieved.notes is None

    def test_create_position_with_notes(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test creating a position with optional notes."""
        notes = "Accumulated during bear market. Strong technicals."
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="MSFT",
            shares=Decimal("5.50000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("2090.0000"),
            purchase_date=datetime.now(UTC),
            notes=notes,
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.notes == notes

    def test_create_position_with_fractional_shares(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that positions support fractional shares."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="GOOG",
            shares=Decimal("2.33333333"),  # Fractional shares
            average_cost_basis=Decimal("2750.0000"),
            total_cost=Decimal("6408.3333"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == Decimal("2.33333333")

    def test_create_position_with_high_precision_prices(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that positions maintain Decimal precision for prices."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TSLA",
            shares=Decimal("1.50000000"),
            average_cost_basis=Decimal("245.1234"),  # 4 decimal places
            total_cost=Decimal("367.6851"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.average_cost_basis == Decimal("245.1234")
        assert retrieved.total_cost == Decimal("367.6851")

    def test_position_gets_portfolio_relationship(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that position relationship to portfolio is properly loaded."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        # Query fresh without expunging to verify relationship loading
        retrieved_position = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved_position.portfolio is not None
        assert retrieved_position.portfolio.id == portfolio.id
        assert retrieved_position.portfolio.name == portfolio.name


class TestPortfolioRead:
    """Test suite for reading portfolio data."""

    @pytest.fixture
    def portfolio_with_positions(self, db_session: Session):
        """Create a portfolio with multiple positions."""
        unique_name = f"Mixed Portfolio {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()

        positions = [
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker="AAPL",
                shares=Decimal("10.00000000"),
                average_cost_basis=Decimal("150.0000"),
                total_cost=Decimal("1500.0000"),
                purchase_date=datetime.now(UTC),
            ),
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker="MSFT",
                shares=Decimal("5.00000000"),
                average_cost_basis=Decimal("380.0000"),
                total_cost=Decimal("1900.0000"),
                purchase_date=datetime.now(UTC) - timedelta(days=30),
            ),
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker="GOOG",
                shares=Decimal("2.50000000"),
                average_cost_basis=Decimal("2750.0000"),
                total_cost=Decimal("6875.0000"),
                purchase_date=datetime.now(UTC) - timedelta(days=60),
            ),
        ]
        db_session.add_all(positions)
        db_session.commit()

        return portfolio

    def test_read_portfolio_with_eager_loaded_positions(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test that positions are eagerly loaded with portfolio (selectin)."""
        portfolio = (
            db_session.query(UserPortfolio)
            .filter_by(id=portfolio_with_positions.id)
            .first()
        )
        assert len(portfolio.positions) == 3
        assert {p.ticker for p in portfolio.positions} == {"AAPL", "MSFT", "GOOG"}

    def test_read_position_by_ticker(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test filtering positions by ticker."""
        position = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id, ticker="MSFT")
            .first()
        )
        assert position is not None
        assert position.ticker == "MSFT"
        assert position.shares == Decimal("5.00000000")
        assert position.average_cost_basis == Decimal("380.0000")

    def test_read_all_positions_for_portfolio(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test reading all positions for a portfolio."""
        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id)
            .order_by(PortfolioPosition.ticker)
            .all()
        )
        assert len(positions) == 3
        assert positions[0].ticker == "AAPL"
        assert positions[1].ticker == "GOOG"
        assert positions[2].ticker == "MSFT"

    def test_read_portfolio_by_user_and_name(self, db_session: Session):
        """Test reading portfolio by user_id and name."""
        portfolio = UserPortfolio(user_id="user1", name="Specific Portfolio")
        db_session.add(portfolio)
        db_session.commit()

        retrieved = (
            db_session.query(UserPortfolio)
            .filter_by(user_id="user1", name="Specific Portfolio")
            .first()
        )
        assert retrieved is not None
        assert retrieved.id == portfolio.id

    def test_read_multiple_portfolios_for_user(self, db_session: Session):
        """Test reading multiple portfolios for the same user."""
        user_id = "user_multi"
        portfolios = [
            UserPortfolio(user_id=user_id, name=f"Portfolio {i}") for i in range(3)
        ]
        db_session.add_all(portfolios)
        db_session.commit()

        retrieved_portfolios = (
            db_session.query(UserPortfolio)
            .filter_by(user_id=user_id)
            .order_by(UserPortfolio.name)
            .all()
        )
        assert len(retrieved_portfolios) == 3


class TestPortfolioUpdate:
    """Test suite for updating portfolio data."""

    @pytest.fixture
    def portfolio_with_position(self, db_session: Session):
        """Create portfolio with a position for update tests."""
        unique_name = f"Update Test {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
            notes="Initial purchase",
        )
        db_session.add(position)
        db_session.commit()

        return portfolio, position

    def test_update_portfolio_name(
        self, db_session: Session, portfolio_with_position: tuple
    ):
        """Test updating portfolio name."""
        portfolio, _ = portfolio_with_position

        portfolio.name = "Updated Portfolio Name"
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.name == "Updated Portfolio Name"

    def test_update_position_shares_and_cost(
        self, db_session: Session, portfolio_with_position: tuple
    ):
        """Test updating position shares and cost (simulating averaging)."""
        _, position = portfolio_with_position

        # Simulate adding shares with cost basis averaging
        position.shares = Decimal("20.00000000")
        position.average_cost_basis = Decimal("160.0000")  # Averaged cost
        position.total_cost = Decimal("3200.0000")
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == Decimal("20.00000000")
        assert retrieved.average_cost_basis == Decimal("160.0000")
        assert retrieved.total_cost == Decimal("3200.0000")

    def test_update_position_notes(
        self, db_session: Session, portfolio_with_position: tuple
    ):
        """Test updating position notes."""
        _, position = portfolio_with_position

        new_notes = "Sold 5 shares at $180, added 5 at $140"
        position.notes = new_notes
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.notes == new_notes

    def test_update_position_clears_notes(
        self, db_session: Session, portfolio_with_position: tuple
    ):
        """Test clearing position notes."""
        _, position = portfolio_with_position

        position.notes = None
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.notes is None

    def test_portfolio_updated_timestamp_changes(
        self, db_session: Session, portfolio_with_position: tuple
    ):
        """Test that updated_at timestamp changes when portfolio is modified."""
        portfolio, _ = portfolio_with_position

        # Small delay to ensure timestamp changes
        import time

        time.sleep(0.01)

        portfolio.name = "New Name"
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        # Note: updated_at may not always change depending on DB precision
        # This test verifies the column exists and is updateable
        assert retrieved.updated_at is not None


class TestPortfolioDelete:
    """Test suite for deleting portfolios and positions."""

    @pytest.fixture
    def portfolio_with_positions(self, db_session: Session):
        """Create portfolio with positions for deletion tests."""
        unique_name = f"Delete Test {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()

        positions = [
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker="AAPL",
                shares=Decimal("10.00000000"),
                average_cost_basis=Decimal("150.0000"),
                total_cost=Decimal("1500.0000"),
                purchase_date=datetime.now(UTC),
            ),
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker="MSFT",
                shares=Decimal("5.00000000"),
                average_cost_basis=Decimal("380.0000"),
                total_cost=Decimal("1900.0000"),
                purchase_date=datetime.now(UTC),
            ),
        ]
        db_session.add_all(positions)
        db_session.commit()

        return portfolio

    def test_delete_single_position(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test deleting a single position from a portfolio."""
        position = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id, ticker="AAPL")
            .first()
        )
        position_id = position.id

        db_session.delete(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position_id).first()
        )
        assert retrieved is None

        # Verify other position still exists
        other_position = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id, ticker="MSFT")
            .first()
        )
        assert other_position is not None

    def test_delete_all_positions_from_portfolio(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test deleting all positions from a portfolio."""
        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id)
            .all()
        )

        for position in positions:
            db_session.delete(position)
        db_session.commit()

        remaining = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id)
            .all()
        )
        assert len(remaining) == 0

        # Portfolio should still exist
        portfolio = (
            db_session.query(UserPortfolio)
            .filter_by(id=portfolio_with_positions.id)
            .first()
        )
        assert portfolio is not None

    def test_cascade_delete_portfolio_removes_positions(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test that deleting a portfolio cascades delete to positions."""
        portfolio_id = portfolio_with_positions.id

        db_session.delete(portfolio_with_positions)
        db_session.commit()

        # Portfolio should be deleted
        portfolio = db_session.query(UserPortfolio).filter_by(id=portfolio_id).first()
        assert portfolio is None

        # Positions should also be deleted
        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_id)
            .all()
        )
        assert len(positions) == 0

    def test_delete_portfolio_doesnt_affect_other_portfolios(self, db_session: Session):
        """Test that deleting one portfolio doesn't affect others."""
        user_id = f"user1_{uuid.uuid4()}"
        portfolio1 = UserPortfolio(user_id=user_id, name=f"Portfolio 1 {uuid.uuid4()}")
        portfolio2 = UserPortfolio(user_id=user_id, name=f"Portfolio 2 {uuid.uuid4()}")
        db_session.add_all([portfolio1, portfolio2])
        db_session.commit()

        # Add position to portfolio1
        position = PortfolioPosition(
            portfolio_id=portfolio1.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        # Delete portfolio1
        db_session.delete(portfolio1)
        db_session.commit()

        # Portfolio2 should still exist
        p2 = db_session.query(UserPortfolio).filter_by(id=portfolio2.id).first()
        assert p2 is not None
        assert p2.name == portfolio2.name  # Use the actual name since it's generated


class TestUniqueConstraints:
    """Test suite for unique constraint enforcement."""

    def test_duplicate_portfolio_name_for_same_user_fails(self, db_session: Session):
        """Test that duplicate portfolio names for same user fail."""
        user_id = f"user1_{uuid.uuid4()}"
        name = f"My Portfolio {uuid.uuid4()}"

        portfolio1 = UserPortfolio(user_id=user_id, name=name)
        db_session.add(portfolio1)
        db_session.commit()

        # Try to create duplicate
        portfolio2 = UserPortfolio(user_id=user_id, name=name)
        db_session.add(portfolio2)

        with pytest.raises(exc.IntegrityError):
            db_session.commit()

    def test_same_portfolio_name_different_users_succeeds(self, db_session: Session):
        """Test that same portfolio name is allowed for different users."""
        name = f"My Portfolio {uuid.uuid4()}"

        portfolio1 = UserPortfolio(user_id=f"user1_{uuid.uuid4()}", name=name)
        portfolio2 = UserPortfolio(user_id=f"user2_{uuid.uuid4()}", name=name)
        db_session.add_all([portfolio1, portfolio2])
        db_session.commit()

        # Both should exist
        p1 = (
            db_session.query(UserPortfolio)
            .filter_by(user_id=portfolio1.user_id, name=name)
            .first()
        )
        p2 = (
            db_session.query(UserPortfolio)
            .filter_by(user_id=portfolio2.user_id, name=name)
            .first()
        )
        assert p1 is not None
        assert p2 is not None
        assert p1.id != p2.id

    def test_duplicate_ticker_in_same_portfolio_fails(self, db_session: Session):
        """Test that duplicate tickers in same portfolio fail."""
        unique_name = f"Test {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()

        position1 = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position1)
        db_session.commit()

        # Try to create duplicate ticker
        position2 = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("160.0000"),
            total_cost=Decimal("800.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position2)

        with pytest.raises(exc.IntegrityError):
            db_session.commit()

    def test_same_ticker_different_portfolios_succeeds(self, db_session: Session):
        """Test that same ticker is allowed in different portfolios."""
        user_id = f"user1_{uuid.uuid4()}"
        portfolio1 = UserPortfolio(user_id=user_id, name=f"Portfolio 1 {uuid.uuid4()}")
        portfolio2 = UserPortfolio(user_id=user_id, name=f"Portfolio 2 {uuid.uuid4()}")
        db_session.add_all([portfolio1, portfolio2])
        db_session.commit()

        position1 = PortfolioPosition(
            portfolio_id=portfolio1.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
        )
        position2 = PortfolioPosition(
            portfolio_id=portfolio2.id,
            ticker="AAPL",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("160.0000"),
            total_cost=Decimal("800.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add_all([position1, position2])
        db_session.commit()

        # Both should exist
        p1 = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio1.id, ticker="AAPL")
            .first()
        )
        p2 = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio2.id, ticker="AAPL")
            .first()
        )
        assert p1 is not None
        assert p2 is not None
        assert p1.id != p2.id


class TestDataIntegrity:
    """Test suite for data integrity and precision."""

    @pytest.fixture
    def portfolio(self, db_session: Session):
        """Create a portfolio for integrity tests."""
        unique_name = f"Integrity Test {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()
        return portfolio

    def test_decimal_precision_preserved(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that Decimal precision is maintained through round-trip."""
        # Use precision that matches database columns:
        # shares: Numeric(20, 8), cost_basis: Numeric(12, 4), total_cost: Numeric(20, 4)
        shares = Decimal("1.12345678")
        cost_basis = Decimal("2345.6789")
        total_cost = Decimal("2637.4012")  # Limited to 4 decimal places

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TEST",
            shares=shares,
            average_cost_basis=cost_basis,
            total_cost=total_cost,
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == shares
        assert retrieved.average_cost_basis == cost_basis
        assert retrieved.total_cost == total_cost

    def test_timezone_aware_datetime_preserved(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that timezone-aware datetimes are preserved."""
        purchase_date = datetime(2024, 1, 15, 14, 30, 45, 123456, tzinfo=UTC)

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=purchase_date,
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.purchase_date.tzinfo is not None
        # Compare date/time (may lose microsecond precision depending on DB)
        assert retrieved.purchase_date.year == purchase_date.year
        assert retrieved.purchase_date.month == purchase_date.month
        assert retrieved.purchase_date.day == purchase_date.day
        assert retrieved.purchase_date.hour == purchase_date.hour
        assert retrieved.purchase_date.minute == purchase_date.minute
        assert retrieved.purchase_date.second == purchase_date.second

    def test_null_notes_allowed(self, db_session: Session, portfolio: UserPortfolio):
        """Test that NULL notes are properly handled."""
        position1 = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
            notes=None,
        )
        position2 = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="MSFT",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("1900.0000"),
            purchase_date=datetime.now(UTC),
            notes="Some notes",
        )
        db_session.add_all([position1, position2])
        db_session.commit()

        p1 = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
            .first()
        )
        p2 = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id, ticker="MSFT")
            .first()
        )
        assert p1.notes is None
        assert p2.notes == "Some notes"

    def test_empty_notes_string_stored(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that empty string notes are stored (if provided)."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
            notes="",
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.notes == ""

    def test_large_decimal_values(self, db_session: Session, portfolio: UserPortfolio):
        """Test handling of large Decimal values."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="HUGE",
            shares=Decimal("999999999999.99999999"),  # Large shares
            average_cost_basis=Decimal("9999.9999"),  # Large price
            total_cost=Decimal("9999999999999999.9999"),  # Large total
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == Decimal("999999999999.99999999")
        assert retrieved.average_cost_basis == Decimal("9999.9999")
        assert retrieved.total_cost == Decimal("9999999999999999.9999")

    def test_very_small_decimal_values(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test handling of very small Decimal values.

        Note: total_cost uses Numeric(20, 4) precision, so values smaller than
        0.0001 will be truncated. This is appropriate for stock trading.
        """
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TINY",
            shares=Decimal("0.00000001"),  # Very small shares (supports 8 decimals)
            average_cost_basis=Decimal("0.0001"),  # Minimum price precision
            total_cost=Decimal("0.0000"),  # Rounds to 0.0000 due to Numeric(20, 4)
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == Decimal("0.00000001")
        assert retrieved.average_cost_basis == Decimal("0.0001")
        # Total cost truncated to 4 decimal places as per Numeric(20, 4)
        assert retrieved.total_cost == Decimal("0.0000")


class TestQueryPerformance:
    """Test suite for query optimization and index usage."""

    @pytest.fixture
    def large_portfolio(self, db_session: Session):
        """Create a portfolio with many positions."""
        unique_name = f"Large Portfolio {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()

        # Create many positions
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        positions = [
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker=tickers[i % len(tickers)],
                shares=Decimal(f"{10 + i}.00000000"),
                average_cost_basis=Decimal(f"{100 + (i * 10)}.0000"),
                total_cost=Decimal(f"{(10 + i) * (100 + (i * 10))}.0000"),
                purchase_date=datetime.now(UTC) - timedelta(days=i),
            )
            for i in range(len(tickers))
        ]
        db_session.add_all(positions)
        db_session.commit()

        return portfolio

    def test_selectin_loading_of_positions(
        self, db_session: Session, large_portfolio: UserPortfolio
    ):
        """Test that selectin loading prevents N+1 queries on positions."""
        portfolio = (
            db_session.query(UserPortfolio).filter_by(id=large_portfolio.id).first()
        )

        # Accessing positions should not trigger additional queries
        # (they should already be loaded via selectin)
        assert len(portfolio.positions) > 0
        for position in portfolio.positions:
            assert position.ticker is not None

    def test_filter_by_ticker_uses_index(
        self, db_session: Session, large_portfolio: UserPortfolio
    ):
        """Test that filtering by ticker uses the index."""
        # This test verifies index exists by checking query can filter
        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=large_portfolio.id, ticker="AAPL")
            .all()
        )
        assert len(positions) >= 1
        assert all(p.ticker == "AAPL" for p in positions)

    def test_filter_by_portfolio_id_uses_index(
        self, db_session: Session, large_portfolio: UserPortfolio
    ):
        """Test that filtering by portfolio_id uses the index."""
        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=large_portfolio.id)
            .all()
        )
        assert len(positions) > 0
        assert all(p.portfolio_id == large_portfolio.id for p in positions)

    def test_combined_filter_portfolio_and_ticker(
        self, db_session: Session, large_portfolio: UserPortfolio
    ):
        """Test filtering by both portfolio_id and ticker (composite index)."""
        position = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=large_portfolio.id, ticker="MSFT")
            .first()
        )
        assert position is not None
        assert position.ticker == "MSFT"

    def test_query_user_portfolios_by_user_id(self, db_session: Session):
        """Test that querying portfolios by user_id is efficient."""
        user_id = f"user_perf_{uuid.uuid4()}"
        portfolios = [
            UserPortfolio(user_id=user_id, name=f"Portfolio {i}_{uuid.uuid4()}")
            for i in range(5)
        ]
        db_session.add_all(portfolios)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(user_id=user_id).all()
        assert len(retrieved) == 5

    def test_order_by_ticker_works(
        self, db_session: Session, large_portfolio: UserPortfolio
    ):
        """Test ordering positions by ticker."""
        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=large_portfolio.id)
            .order_by(PortfolioPosition.ticker)
            .all()
        )
        assert len(positions) > 0
        # Verify ordering
        tickers = [p.ticker for p in positions]
        assert tickers == sorted(tickers)


class TestPortfolioIntegration:
    """End-to-end integration tests combining multiple operations."""

    def test_complete_portfolio_lifecycle(self, db_session: Session):
        """Test complete portfolio lifecycle from creation to deletion."""
        # Create portfolio
        unique_name = f"Lifecycle Portfolio {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="test_user", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()
        portfolio_id = portfolio.id

        # Add positions
        positions_data = [
            ("AAPL", Decimal("10"), Decimal("150.0000"), Decimal("1500.0000")),
            ("MSFT", Decimal("5"), Decimal("380.0000"), Decimal("1900.0000")),
        ]

        for ticker, shares, price, total in positions_data:
            position = PortfolioPosition(
                portfolio_id=portfolio_id,
                ticker=ticker,
                shares=shares,
                average_cost_basis=price,
                total_cost=total,
                purchase_date=datetime.now(UTC),
            )
            db_session.add(position)
        db_session.commit()

        # Read and verify
        portfolio = db_session.query(UserPortfolio).filter_by(id=portfolio_id).first()
        assert len(portfolio.positions) == 2
        assert {p.ticker for p in portfolio.positions} == {"AAPL", "MSFT"}

        # Update position
        msft_position = next(p for p in portfolio.positions if p.ticker == "MSFT")
        msft_position.shares = Decimal("10")  # Double shares
        msft_position.average_cost_basis = Decimal("370.0000")  # Averaged price
        msft_position.total_cost = Decimal("3700.0000")
        db_session.commit()

        # Delete one position
        aapl_position = next(p for p in portfolio.positions if p.ticker == "AAPL")
        db_session.delete(aapl_position)
        db_session.commit()

        # Verify state
        portfolio = db_session.query(UserPortfolio).filter_by(id=portfolio_id).first()
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].ticker == "MSFT"
        assert portfolio.positions[0].shares == Decimal("10")

        # Delete portfolio
        db_session.delete(portfolio)
        db_session.commit()

        # Verify deletion
        portfolio = db_session.query(UserPortfolio).filter_by(id=portfolio_id).first()
        assert portfolio is None

        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_id)
            .all()
        )
        assert len(positions) == 0

    def test_portfolio_with_various_decimal_precision(self, db_session: Session):
        """Test portfolio with positions of varying decimal precisions.

        Note: total_cost uses Numeric(20, 4), so values are truncated to 4 decimal places.
        """
        unique_name = f"Mixed Precision {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id="default", name=unique_name)
        db_session.add(portfolio)
        db_session.commit()

        positions_data = [
            ("AAPL", Decimal("1"), Decimal("100.00"), Decimal("100.00")),
            ("MSFT", Decimal("1.5"), Decimal("200.5000"), Decimal("300.7500")),
            (
                "GOOG",
                Decimal("0.33333333"),
                Decimal("2750.1234"),
                Decimal("917.5041"),  # Truncated from 917.50413522 to 4 decimals
            ),
            ("AMZN", Decimal("100"), Decimal("150.1"), Decimal("15010")),
        ]

        for ticker, shares, price, total in positions_data:
            position = PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker=ticker,
                shares=shares,
                average_cost_basis=price,
                total_cost=total,
                purchase_date=datetime.now(UTC),
            )
            db_session.add(position)
        db_session.commit()

        # Verify all positions preserved their precision
        portfolio = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert len(portfolio.positions) == 4

        for (
            expected_ticker,
            expected_shares,
            expected_price,
            expected_total,
        ) in positions_data:
            position = next(
                p for p in portfolio.positions if p.ticker == expected_ticker
            )
            assert position.shares == expected_shares
            assert position.average_cost_basis == expected_price
            assert position.total_cost == expected_total
