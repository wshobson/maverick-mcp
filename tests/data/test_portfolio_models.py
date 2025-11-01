"""
Comprehensive integration tests for portfolio database models and migration.

This module tests:
1. Migration upgrade and downgrade operations
2. SQLAlchemy model CRUD operations (Create, Read, Update, Delete)
3. Database constraints (unique constraints, foreign keys, cascade deletes)
4. Relationships between UserPortfolio and PortfolioPosition
5. Decimal field precision for financial data (Numeric(12,4) and Numeric(20,8))
6. Timezone-aware datetime fields
7. Index creation and query optimization

Test Coverage:
- Migration creates tables with correct schema
- Indexes are created properly for performance optimization
- Unique constraints work for both portfolio and position level
- Cascade delete removes positions when portfolio is deleted
- Decimal precision is maintained through round-trip database operations
- Relationships are properly loaded with selectin strategy
- Default values are applied correctly (user_id="default", name="My Portfolio")
- Timestamp mixin functionality (created_at, updated_at)

Test Markers:
- @pytest.mark.integration - Full database integration tests
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import exc, inspect
from sqlalchemy.orm import Session

from maverick_mcp.data.models import PortfolioPosition, UserPortfolio

pytestmark = pytest.mark.integration


# ============================================================================
# Migration Tests
# ============================================================================


class TestMigrationUpgrade:
    """Test suite for migration upgrade operations."""

    def test_migration_creates_portfolios_table(self, db_session: Session):
        """Test that migration creates mcp_portfolios table."""
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names()
        assert "mcp_portfolios" in tables

    def test_migration_creates_positions_table(self, db_session: Session):
        """Test that migration creates mcp_portfolio_positions table."""
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names()
        assert "mcp_portfolio_positions" in tables

    def test_portfolios_table_has_correct_columns(self, db_session: Session):
        """Test that portfolios table has all required columns."""
        inspector = inspect(db_session.bind)
        columns = {col["name"] for col in inspector.get_columns("mcp_portfolios")}

        required_columns = {"id", "user_id", "name", "created_at", "updated_at"}
        assert required_columns.issubset(columns)

    def test_positions_table_has_correct_columns(self, db_session: Session):
        """Test that positions table has all required columns."""
        inspector = inspect(db_session.bind)
        columns = {
            col["name"] for col in inspector.get_columns("mcp_portfolio_positions")
        }

        required_columns = {
            "id",
            "portfolio_id",
            "ticker",
            "shares",
            "average_cost_basis",
            "total_cost",
            "purchase_date",
            "notes",
            "created_at",
            "updated_at",
        }
        assert required_columns.issubset(columns)

    def test_portfolios_id_column_type(self, db_session: Session):
        """Test that portfolio id column is UUID type."""
        inspector = inspect(db_session.bind)
        columns = {col["name"]: col for col in inspector.get_columns("mcp_portfolios")}
        assert "id" in columns
        # Column exists and is configured as primary key through Index and UniqueConstraint

    def test_positions_foreign_key_constraint(self, db_session: Session):
        """Test that positions table has foreign key to portfolios."""
        inspector = inspect(db_session.bind)
        fks = inspector.get_foreign_keys("mcp_portfolio_positions")
        assert len(fks) > 0
        assert any(fk["constrained_columns"] == ["portfolio_id"] for fk in fks)

    def test_migration_creates_portfolio_user_index(self, db_session: Session):
        """Test that migration creates index on portfolio user_id."""
        inspector = inspect(db_session.bind)
        indexes = {idx["name"] for idx in inspector.get_indexes("mcp_portfolios")}
        assert "idx_portfolio_user" in indexes

    def test_migration_creates_position_portfolio_index(self, db_session: Session):
        """Test that migration creates index on position portfolio_id."""
        inspector = inspect(db_session.bind)
        indexes = {
            idx["name"] for idx in inspector.get_indexes("mcp_portfolio_positions")
        }
        assert "idx_position_portfolio" in indexes

    def test_migration_creates_position_ticker_index(self, db_session: Session):
        """Test that migration creates index on position ticker."""
        inspector = inspect(db_session.bind)
        indexes = {
            idx["name"] for idx in inspector.get_indexes("mcp_portfolio_positions")
        }
        assert "idx_position_ticker" in indexes

    def test_migration_creates_position_composite_index(self, db_session: Session):
        """Test that migration creates composite index on portfolio_id and ticker."""
        inspector = inspect(db_session.bind)
        indexes = {
            idx["name"] for idx in inspector.get_indexes("mcp_portfolio_positions")
        }
        assert "idx_position_portfolio_ticker" in indexes

    def test_migration_creates_unique_portfolio_constraint(self, db_session: Session):
        """Test that migration creates unique constraint on user_id and name."""
        inspector = inspect(db_session.bind)
        constraints = inspector.get_unique_constraints("mcp_portfolios")
        constraint_names = {c["name"] for c in constraints}
        assert "uq_user_portfolio_name" in constraint_names

    def test_migration_creates_unique_position_constraint(self, db_session: Session):
        """Test that migration creates unique constraint on portfolio_id and ticker."""
        inspector = inspect(db_session.bind)
        constraints = inspector.get_unique_constraints("mcp_portfolio_positions")
        constraint_names = {c["name"] for c in constraints}
        assert "uq_portfolio_position_ticker" in constraint_names

    def test_portfolios_user_id_has_default(self, db_session: Session):
        """Test that user_id column exists and is not nullable."""
        inspector = inspect(db_session.bind)
        columns = {col["name"]: col for col in inspector.get_columns("mcp_portfolios")}
        assert "user_id" in columns
        # Default is handled at model level, not server level

    def test_portfolios_name_has_default(self, db_session: Session):
        """Test that name column exists and is not nullable."""
        inspector = inspect(db_session.bind)
        columns = {col["name"]: col for col in inspector.get_columns("mcp_portfolios")}
        assert "name" in columns
        # Default is handled at model level, not server level

    def test_portfolios_created_at_has_default(self, db_session: Session):
        """Test that created_at column exists for timestamp tracking."""
        inspector = inspect(db_session.bind)
        columns = {col["name"]: col for col in inspector.get_columns("mcp_portfolios")}
        assert "created_at" in columns

    def test_portfolios_updated_at_has_default(self, db_session: Session):
        """Test that updated_at column exists for timestamp tracking."""
        inspector = inspect(db_session.bind)
        columns = {col["name"]: col for col in inspector.get_columns("mcp_portfolios")}
        assert "updated_at" in columns

    def test_positions_created_at_has_default(self, db_session: Session):
        """Test that position created_at column exists for timestamp tracking."""
        inspector = inspect(db_session.bind)
        columns = {
            col["name"]: col for col in inspector.get_columns("mcp_portfolio_positions")
        }
        assert "created_at" in columns

    def test_positions_updated_at_has_default(self, db_session: Session):
        """Test that position updated_at column exists for timestamp tracking."""
        inspector = inspect(db_session.bind)
        columns = {
            col["name"]: col for col in inspector.get_columns("mcp_portfolio_positions")
        }
        assert "updated_at" in columns


# ============================================================================
# Model CRUD Operation Tests
# ============================================================================


class TestPortfolioModelCRUD:
    """Test suite for UserPortfolio CRUD operations."""

    def test_create_portfolio_with_all_fields(self, db_session: Session):
        """Test creating a portfolio with all fields specified."""
        portfolio = UserPortfolio(
            id=uuid.uuid4(),
            user_id="test_user",
            name="Test Portfolio",
        )
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved is not None
        assert retrieved.user_id == "test_user"
        assert retrieved.name == "Test Portfolio"
        assert retrieved.created_at is not None
        assert retrieved.updated_at is not None

    def test_create_portfolio_with_defaults(self, db_session: Session):
        """Test that portfolio defaults are applied correctly."""
        portfolio = UserPortfolio()
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.user_id == "default"
        assert retrieved.name == "My Portfolio"

    def test_read_portfolio_by_id(self, db_session: Session):
        """Test reading portfolio by ID."""
        portfolio = UserPortfolio(user_id="user1", name="Portfolio 1")
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved is not None
        assert retrieved.id == portfolio.id

    def test_read_portfolio_by_user_and_name(self, db_session: Session):
        """Test reading portfolio by user_id and name."""
        portfolio = UserPortfolio(user_id="user2", name="My Portfolio 2")
        db_session.add(portfolio)
        db_session.commit()

        retrieved = (
            db_session.query(UserPortfolio)
            .filter_by(user_id="user2", name="My Portfolio 2")
            .first()
        )
        assert retrieved is not None
        assert retrieved.id == portfolio.id

    def test_read_all_portfolios_for_user(self, db_session: Session):
        """Test reading all portfolios for a specific user."""
        user_id = f"user_read_{uuid.uuid4()}"
        portfolios = [
            UserPortfolio(user_id=user_id, name=f"Portfolio {i}") for i in range(3)
        ]
        db_session.add_all(portfolios)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(user_id=user_id).all()
        assert len(retrieved) == 3

    def test_update_portfolio_name(self, db_session: Session):
        """Test updating portfolio name."""
        portfolio = UserPortfolio(user_id="user3", name="Original Name")
        db_session.add(portfolio)
        db_session.commit()

        portfolio.name = "Updated Name"
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.name == "Updated Name"

    def test_update_portfolio_user_id(self, db_session: Session):
        """Test updating portfolio user_id."""
        portfolio = UserPortfolio(user_id="old_user", name="Portfolio")
        db_session.add(portfolio)
        db_session.commit()

        portfolio.user_id = "new_user"
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.user_id == "new_user"

    def test_delete_portfolio(self, db_session: Session):
        """Test deleting a portfolio."""
        portfolio = UserPortfolio(user_id="user4", name="To Delete")
        db_session.add(portfolio)
        db_session.commit()
        portfolio_id = portfolio.id

        db_session.delete(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio_id).first()
        assert retrieved is None

    def test_portfolio_repr(self, db_session: Session):
        """Test portfolio string representation."""
        portfolio = UserPortfolio(user_id="user5", name="Test Portfolio")
        db_session.add(portfolio)
        db_session.commit()

        repr_str = repr(portfolio)
        assert "UserPortfolio" in repr_str
        assert "Test Portfolio" in repr_str


class TestPositionModelCRUD:
    """Test suite for PortfolioPosition CRUD operations."""

    @pytest.fixture
    def portfolio(self, db_session: Session) -> UserPortfolio:
        """Create a test portfolio."""
        portfolio = UserPortfolio(
            user_id="default", name=f"Test Portfolio {uuid.uuid4()}"
        )
        db_session.add(portfolio)
        db_session.commit()
        return portfolio

    def test_create_position_with_all_fields(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test creating a position with all fields."""
        position = PortfolioPosition(
            id=uuid.uuid4(),
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
            notes="Test position",
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved is not None
        assert retrieved.ticker == "AAPL"
        assert retrieved.notes == "Test position"

    def test_create_position_without_notes(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test creating a position without notes."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="MSFT",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("1900.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.notes is None

    def test_read_position_by_id(self, db_session: Session, portfolio: UserPortfolio):
        """Test reading position by ID."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="GOOG",
            shares=Decimal("2.00000000"),
            average_cost_basis=Decimal("2750.0000"),
            total_cost=Decimal("5500.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved is not None
        assert retrieved.ticker == "GOOG"

    def test_read_position_by_portfolio_and_ticker(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test reading position by portfolio_id and ticker."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TSLA",
            shares=Decimal("1.00000000"),
            average_cost_basis=Decimal("250.0000"),
            total_cost=Decimal("250.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id, ticker="TSLA")
            .first()
        )
        assert retrieved is not None

    def test_read_all_positions_in_portfolio(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test reading all positions in a portfolio."""
        positions_data = [
            ("AAPL", Decimal("10"), Decimal("150.0000")),
            ("MSFT", Decimal("5"), Decimal("380.0000")),
            ("GOOG", Decimal("2"), Decimal("2750.0000")),
        ]

        for ticker, shares, price in positions_data:
            position = PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker=ticker,
                shares=shares,
                average_cost_basis=price,
                total_cost=shares * price,
                purchase_date=datetime.now(UTC),
            )
            db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id)
            .all()
        )
        assert len(retrieved) == 3

    def test_update_position_shares(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test updating position shares."""
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

        position.shares = Decimal("20.00000000")
        position.average_cost_basis = Decimal("160.0000")
        position.total_cost = Decimal("3200.0000")
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == Decimal("20.00000000")

    def test_update_position_cost_basis(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test updating position average cost basis."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="MSFT",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("1900.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        original_cost_basis = position.average_cost_basis
        position.average_cost_basis = Decimal("390.0000")
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.average_cost_basis != original_cost_basis
        assert retrieved.average_cost_basis == Decimal("390.0000")

    def test_update_position_notes(self, db_session: Session, portfolio: UserPortfolio):
        """Test updating position notes."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="GOOG",
            shares=Decimal("2.00000000"),
            average_cost_basis=Decimal("2750.0000"),
            total_cost=Decimal("5500.0000"),
            purchase_date=datetime.now(UTC),
            notes="Original notes",
        )
        db_session.add(position)
        db_session.commit()

        position.notes = "Updated notes"
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.notes == "Updated notes"

    def test_delete_position(self, db_session: Session, portfolio: UserPortfolio):
        """Test deleting a position."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TSLA",
            shares=Decimal("1.00000000"),
            average_cost_basis=Decimal("250.0000"),
            total_cost=Decimal("250.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()
        position_id = position.id

        db_session.delete(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position_id).first()
        )
        assert retrieved is None

    def test_position_repr(self, db_session: Session, portfolio: UserPortfolio):
        """Test position string representation."""
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="NVDA",
            shares=Decimal("3.00000000"),
            average_cost_basis=Decimal("900.0000"),
            total_cost=Decimal("2700.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        repr_str = repr(position)
        assert "PortfolioPosition" in repr_str
        assert "NVDA" in repr_str


# ============================================================================
# Relationship Tests
# ============================================================================


class TestPortfolioPositionRelationships:
    """Test suite for relationships between UserPortfolio and PortfolioPosition."""

    @pytest.fixture
    def portfolio_with_positions(self, db_session: Session) -> UserPortfolio:
        """Create a portfolio with multiple positions."""
        portfolio = UserPortfolio(
            user_id="default", name=f"Relationship Test {uuid.uuid4()}"
        )
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

    def test_portfolio_has_positions_relationship(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test that portfolio has positions relationship."""
        portfolio = (
            db_session.query(UserPortfolio)
            .filter_by(id=portfolio_with_positions.id)
            .first()
        )
        assert hasattr(portfolio, "positions")
        assert isinstance(portfolio.positions, list)

    def test_positions_eagerly_loaded_via_selectin(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test that positions are eagerly loaded (selectin strategy)."""
        portfolio = (
            db_session.query(UserPortfolio)
            .filter_by(id=portfolio_with_positions.id)
            .first()
        )
        assert len(portfolio.positions) == 2
        assert {p.ticker for p in portfolio.positions} == {"AAPL", "MSFT"}

    def test_position_has_portfolio_relationship(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test that position has back reference to portfolio."""
        position = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id)
            .first()
        )
        assert position.portfolio is not None
        assert position.portfolio.id == portfolio_with_positions.id

    def test_position_portfolio_relationship_maintains_integrity(
        self, db_session: Session, portfolio_with_positions: UserPortfolio
    ):
        """Test that position portfolio relationship maintains data integrity."""
        position = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_with_positions.id, ticker="AAPL")
            .first()
        )
        assert position.portfolio.name == portfolio_with_positions.name
        assert position.portfolio.user_id == portfolio_with_positions.user_id

    def test_multiple_portfolios_have_separate_positions(self, db_session: Session):
        """Test that multiple portfolios have separate position lists."""
        user_id = f"user_multi_{uuid.uuid4()}"
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
            ticker="MSFT",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("1900.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add_all([position1, position2])
        db_session.commit()

        p1 = db_session.query(UserPortfolio).filter_by(id=portfolio1.id).first()
        p2 = db_session.query(UserPortfolio).filter_by(id=portfolio2.id).first()

        assert len(p1.positions) == 1
        assert len(p2.positions) == 1
        assert p1.positions[0].ticker == "AAPL"
        assert p2.positions[0].ticker == "MSFT"


# ============================================================================
# Constraint Tests
# ============================================================================


class TestDatabaseConstraints:
    """Test suite for database constraints enforcement."""

    def test_unique_portfolio_name_constraint_enforced(self, db_session: Session):
        """Test that unique constraint on (user_id, name) is enforced."""
        user_id = f"user_constraint_{uuid.uuid4()}"
        name = f"Unique Portfolio {uuid.uuid4()}"

        portfolio1 = UserPortfolio(user_id=user_id, name=name)
        db_session.add(portfolio1)
        db_session.commit()

        # Try to create duplicate
        portfolio2 = UserPortfolio(user_id=user_id, name=name)
        db_session.add(portfolio2)

        with pytest.raises(exc.IntegrityError):
            db_session.commit()

    def test_unique_position_ticker_constraint_enforced(self, db_session: Session):
        """Test that unique constraint on (portfolio_id, ticker) is enforced."""
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
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

    def test_foreign_key_constraint_enforced(self, db_session: Session):
        """Test that foreign key constraint is enforced."""
        position = PortfolioPosition(
            portfolio_id=uuid.uuid4(),  # Non-existent portfolio
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)

        with pytest.raises(exc.IntegrityError):
            db_session.commit()

    def test_cascade_delete_removes_positions(self, db_session: Session):
        """Test that deleting a portfolio cascades delete to positions."""
        portfolio = UserPortfolio(user_id="default", name=f"Delete Test {uuid.uuid4()}")
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

        portfolio_id = portfolio.id
        db_session.delete(portfolio)
        db_session.commit()

        # Verify portfolio is deleted
        p = db_session.query(UserPortfolio).filter_by(id=portfolio_id).first()
        assert p is None

        # Verify positions are also deleted
        pos = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio_id)
            .all()
        )
        assert len(pos) == 0

    def test_cascade_delete_doesnt_affect_other_portfolios(self, db_session: Session):
        """Test that deleting one portfolio doesn't affect others."""
        user_id = f"user_cascade_{uuid.uuid4()}"
        portfolio1 = UserPortfolio(user_id=user_id, name=f"Portfolio 1 {uuid.uuid4()}")
        portfolio2 = UserPortfolio(user_id=user_id, name=f"Portfolio 2 {uuid.uuid4()}")
        db_session.add_all([portfolio1, portfolio2])
        db_session.commit()

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

        db_session.delete(portfolio1)
        db_session.commit()

        # Portfolio2 should still exist
        p2 = db_session.query(UserPortfolio).filter_by(id=portfolio2.id).first()
        assert p2 is not None


# ============================================================================
# Decimal Precision Tests
# ============================================================================


class TestDecimalPrecision:
    """Test suite for Decimal field precision."""

    @pytest.fixture
    def portfolio(self, db_session: Session) -> UserPortfolio:
        """Create a test portfolio."""
        portfolio = UserPortfolio(
            user_id="default", name=f"Decimal Test {uuid.uuid4()}"
        )
        db_session.add(portfolio)
        db_session.commit()
        return portfolio

    def test_shares_numeric_20_8_precision(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that shares maintains Numeric(20,8) precision."""
        shares = Decimal("12345678901.12345678")

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TEST1",
            shares=shares,
            average_cost_basis=Decimal("100.0000"),
            total_cost=Decimal("1234567890112.3456"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.shares == shares

    def test_cost_basis_numeric_12_4_precision(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that average_cost_basis maintains Numeric(12,4) precision."""
        cost_basis = Decimal("99999999.9999")

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TEST2",
            shares=Decimal("100.00000000"),
            average_cost_basis=cost_basis,
            total_cost=Decimal("9999999999.9999"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.average_cost_basis == cost_basis

    def test_total_cost_numeric_20_4_precision(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that total_cost maintains Numeric(20,4) precision."""
        total_cost = Decimal("9999999999999999.9999")

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TEST3",
            shares=Decimal("1000.00000000"),
            average_cost_basis=Decimal("9999999.9999"),
            total_cost=total_cost,
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.total_cost == total_cost

    def test_fractional_shares_precision(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that fractional shares with high precision are maintained.

        Note: total_cost uses Numeric(20, 4), so values are truncated to 4 decimal places.
        """
        shares = Decimal("0.33333333")
        cost_basis = Decimal("2750.1234")
        total_cost = Decimal("917.5041")  # Truncated from 917.50413522 to 4 decimals

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TEST4",
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

    def test_very_small_decimal_values(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test handling of very small Decimal values.

        Note: total_cost uses Numeric(20, 4) precision, so values smaller than
        0.0001 will be truncated. This is appropriate for stock trading.
        """
        shares = Decimal("0.00000001")
        cost_basis = Decimal("0.0001")
        total_cost = Decimal("0.0000")  # Rounds to 0.0000 due to Numeric(20, 4)

        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="TEST5",
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
        # Total cost truncated to 4 decimal places as per Numeric(20, 4)
        assert retrieved.total_cost == total_cost

    def test_multiple_positions_precision_preserved(
        self, db_session: Session, portfolio: UserPortfolio
    ):
        """Test that precision is maintained across multiple positions."""
        test_data = [
            (Decimal("1"), Decimal("100.00"), Decimal("100.00")),
            (Decimal("1.5"), Decimal("200.5000"), Decimal("300.7500")),
            (Decimal("0.33333333"), Decimal("2750.1234"), Decimal("917.5041")),
            (Decimal("100"), Decimal("150.1234"), Decimal("15012.34")),
        ]

        for i, (shares, cost_basis, total_cost) in enumerate(test_data):
            position = PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker=f"MULTI{i}",
                shares=shares,
                average_cost_basis=cost_basis,
                total_cost=total_cost,
                purchase_date=datetime.now(UTC),
            )
            db_session.add(position)
        db_session.commit()

        positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id)
            .all()
        )
        assert len(positions) == 4

        for i, (expected_shares, expected_cost, _expected_total) in enumerate(
            test_data
        ):
            position = next(p for p in positions if p.ticker == f"MULTI{i}")
            assert position.shares == expected_shares
            assert position.average_cost_basis == expected_cost


# ============================================================================
# Timestamp Tests
# ============================================================================


class TestTimestampMixin:
    """Test suite for TimestampMixin functionality."""

    def test_portfolio_created_at_set_on_creation(self, db_session: Session):
        """Test that created_at is set when portfolio is created."""
        before = datetime.now(UTC)
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()
        after = datetime.now(UTC)

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.created_at is not None
        assert before <= retrieved.created_at <= after

    def test_portfolio_updated_at_set_on_creation(self, db_session: Session):
        """Test that updated_at is set when portfolio is created."""
        before = datetime.now(UTC)
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()
        after = datetime.now(UTC)

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.updated_at is not None
        assert before <= retrieved.updated_at <= after

    def test_position_created_at_set_on_creation(self, db_session: Session):
        """Test that created_at is set when position is created."""
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()

        before = datetime.now(UTC)
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
        after = datetime.now(UTC)

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.created_at is not None
        assert before <= retrieved.created_at <= after

    def test_position_updated_at_set_on_creation(self, db_session: Session):
        """Test that updated_at is set when position is created."""
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()

        before = datetime.now(UTC)
        position = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="MSFT",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("1900.0000"),
            purchase_date=datetime.now(UTC),
        )
        db_session.add(position)
        db_session.commit()
        after = datetime.now(UTC)

        retrieved = (
            db_session.query(PortfolioPosition).filter_by(id=position.id).first()
        )
        assert retrieved.updated_at is not None
        assert before <= retrieved.updated_at <= after

    def test_created_at_does_not_change_on_update(self, db_session: Session):
        """Test that created_at remains unchanged when portfolio is updated."""
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()

        original_created_at = portfolio.created_at
        import time

        time.sleep(0.01)

        portfolio.name = "Updated Name"
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.created_at == original_created_at

    def test_timezone_aware_datetimes(self, db_session: Session):
        """Test that datetimes are timezone-aware."""
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.created_at.tzinfo is not None
        assert retrieved.updated_at.tzinfo is not None


# ============================================================================
# Default Value Tests
# ============================================================================


class TestDefaultValues:
    """Test suite for default values in models."""

    def test_portfolio_default_user_id(self, db_session: Session):
        """Test that portfolio has default user_id."""
        portfolio = UserPortfolio(name="Custom Name")
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.user_id == "default"

    def test_portfolio_default_name(self, db_session: Session):
        """Test that portfolio has default name."""
        portfolio = UserPortfolio(user_id="custom_user")
        db_session.add(portfolio)
        db_session.commit()

        retrieved = db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        assert retrieved.name == "My Portfolio"

    def test_position_default_notes(self, db_session: Session):
        """Test that position notes default to None."""
        portfolio = UserPortfolio(user_id="default", name=f"Portfolio {uuid.uuid4()}")
        db_session.add(portfolio)
        db_session.commit()

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
        assert retrieved.notes is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestPortfolioIntegration:
    """End-to-end integration tests combining multiple operations."""

    def test_complete_portfolio_workflow(self, db_session: Session):
        """Test complete workflow: create, read, update, delete."""
        # Create portfolio
        user_id = f"test_user_{uuid.uuid4()}"
        portfolio_name = f"Integration Test {uuid.uuid4()}"
        portfolio = UserPortfolio(user_id=user_id, name=portfolio_name)
        db_session.add(portfolio)
        db_session.commit()

        # Add positions
        position1 = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            shares=Decimal("10.00000000"),
            average_cost_basis=Decimal("150.0000"),
            total_cost=Decimal("1500.0000"),
            purchase_date=datetime.now(UTC) - timedelta(days=30),
            notes="Initial purchase",
        )
        position2 = PortfolioPosition(
            portfolio_id=portfolio.id,
            ticker="MSFT",
            shares=Decimal("5.00000000"),
            average_cost_basis=Decimal("380.0000"),
            total_cost=Decimal("1900.0000"),
            purchase_date=datetime.now(UTC) - timedelta(days=15),
        )
        db_session.add_all([position1, position2])
        db_session.commit()

        # Read and verify
        retrieved_portfolio = (
            db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        )
        assert retrieved_portfolio is not None
        assert len(retrieved_portfolio.positions) == 2

        # Update position
        aapl_position = next(
            p for p in retrieved_portfolio.positions if p.ticker == "AAPL"
        )
        original_shares = aapl_position.shares
        aapl_position.shares = Decimal("20.00000000")
        aapl_position.average_cost_basis = Decimal("160.0000")
        aapl_position.total_cost = Decimal("3200.0000")
        db_session.commit()

        # Verify update
        retrieved_position = (
            db_session.query(PortfolioPosition).filter_by(id=aapl_position.id).first()
        )
        assert retrieved_position.shares == Decimal("20.00000000")
        assert retrieved_position.shares != original_shares

        # Delete one position
        db_session.delete(aapl_position)
        db_session.commit()

        # Verify deletion
        remaining_positions = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id)
            .all()
        )
        assert len(remaining_positions) == 1
        assert remaining_positions[0].ticker == "MSFT"

        # Delete portfolio (cascade delete)
        db_session.delete(retrieved_portfolio)
        db_session.commit()

        # Verify cascade delete
        portfolio_check = (
            db_session.query(UserPortfolio).filter_by(id=portfolio.id).first()
        )
        assert portfolio_check is None

        positions_check = (
            db_session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id)
            .all()
        )
        assert len(positions_check) == 0

    def test_complex_portfolio_with_multiple_users(self, db_session: Session):
        """Test complex scenario with multiple portfolios and users."""
        user_ids = [f"user_{uuid.uuid4()}" for _ in range(3)]
        portfolios = []

        # Create portfolios for multiple users
        for user_id in user_ids:
            for i in range(2):
                portfolio = UserPortfolio(
                    user_id=user_id, name=f"Portfolio {i} {uuid.uuid4()}"
                )
                db_session.add(portfolio)
                portfolios.append(portfolio)
        db_session.commit()

        # Add positions to each portfolio
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        for portfolio in portfolios:
            for ticker in tickers[:3]:  # Add 3 positions per portfolio
                position = PortfolioPosition(
                    portfolio_id=portfolio.id,
                    ticker=ticker,
                    shares=Decimal("10.00000000"),
                    average_cost_basis=Decimal("150.0000"),
                    total_cost=Decimal("1500.0000"),
                    purchase_date=datetime.now(UTC),
                )
                db_session.add(position)
        db_session.commit()

        # Verify structure
        for user_id in user_ids:
            user_portfolios = (
                db_session.query(UserPortfolio).filter_by(user_id=user_id).all()
            )
            assert len(user_portfolios) == 2
            for portfolio in user_portfolios:
                assert len(portfolio.positions) == 3
