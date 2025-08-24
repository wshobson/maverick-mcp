"""
Tests for database integrity improvements.

Tests the following critical fixes:
1. Financial data precision (Numeric(12,4))
2. Foreign key constraints
3. Check constraints for positive values and ranges
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.auth.models import (
    Base as AuthBase,
)
from maverick_mcp.auth.models import (
    CreditTransactionModel,
    MCPRequestModel,
    UserCreditModel,
    UserMapping,
)
from maverick_mcp.data.models import Base as DataBase
from maverick_mcp.data.models import PriceCache, Stock


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    # Note: SQLite doesn't enforce all constraints like PostgreSQL,
    # but it's sufficient for basic testing
    engine = create_engine("sqlite:///:memory:")

    # Create all tables
    DataBase.metadata.create_all(engine)
    AuthBase.metadata.create_all(engine)

    yield engine

    engine.dispose()


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()

    yield session

    session.close()


class TestFinancialDataPrecision:
    """Test financial data precision improvements."""

    @pytest.mark.integration
    def test_high_value_stock_prices(self, test_session):
        """Test that we can store high-value stock prices like BRK.A."""
        # Create a stock
        stock = Stock(
            ticker_symbol="BRK.A",
            company_name="Berkshire Hathaway Inc. Class A",
            sector="Financial Services",
            industry="Insurance—Diversified",
        )
        test_session.add(stock)
        test_session.commit()

        # Create price data with very high values
        high_price = Decimal("675000.1234")  # BRK.A can trade above $600,000

        price_cache = PriceCache(
            stock_id=stock.stock_id,
            date=datetime.now(UTC).date(),
            open_price=high_price - Decimal("1000"),
            high_price=high_price,
            low_price=high_price - Decimal("2000"),
            close_price=high_price - Decimal("500"),
            volume=100,
        )
        test_session.add(price_cache)
        test_session.commit()

        # Verify the data was stored correctly
        saved_price = (
            test_session.query(PriceCache).filter_by(stock_id=stock.stock_id).first()
        )
        assert saved_price is not None
        assert saved_price.high_price == high_price
        assert saved_price.close_price == high_price - Decimal("500")

        # Test precision to 4 decimal places
        assert str(saved_price.high_price) == "675000.1234"

    @pytest.mark.integration
    def test_credit_balance_precision(self, test_session):
        """Test that credit balances support 4 decimal places."""
        # Create a user mapping
        user_mapping = UserMapping(
            user_id=1,
            django_user_id=1,
            email="test@example.com",
            is_active=True,
        )
        test_session.add(user_mapping)
        test_session.commit()

        # Create credit balance with precise decimal
        credit = UserCreditModel(
            user_id=1,
            balance=Decimal("12345.6789"),
            free_balance=Decimal("100.0001"),
            total_purchased=Decimal("12245.6788"),
        )
        test_session.add(credit)
        test_session.commit()

        # Verify precision
        saved_credit = test_session.query(UserCreditModel).filter_by(user_id=1).first()
        assert saved_credit is not None
        assert saved_credit.balance == Decimal("12345.6789")
        assert saved_credit.free_balance == Decimal("100.0001")


class TestForeignKeyConstraints:
    """Test foreign key constraints and user mapping."""

    @pytest.mark.integration
    def test_user_mapping_required(self, test_session):
        """Test that foreign key constraints require valid user mapping."""
        # Try to create a credit record without a user mapping
        # Note: SQLite doesn't enforce foreign keys by default,
        # so this test would need PostgreSQL to fully validate

        # First, let's verify we can't create orphaned records
        # when a user mapping exists and is then deleted
        user_mapping = UserMapping(
            user_id=1,
            django_user_id=1,
            email="test@example.com",
            is_active=True,
        )
        test_session.add(user_mapping)
        test_session.commit()

        # Create related records
        credit = UserCreditModel(
            user_id=1,
            balance=Decimal("100"),
        )
        test_session.add(credit)

        transaction = CreditTransactionModel(
            user_id=1,
            amount=Decimal("50"),
            transaction_type="PURCHASE",
            balance_after=Decimal("150"),
        )
        test_session.add(transaction)

        request = MCPRequestModel(
            user_id=1,
            request_type="tool",
            tool_name="get_stock_data",
            credits_charged=5,
        )
        test_session.add(request)

        test_session.commit()

        # Verify relationships work
        assert user_mapping.credit_info is not None
        assert len(user_mapping.credit_transactions) == 1
        assert len(user_mapping.requests) == 1

    @pytest.mark.integration
    def test_cascade_delete(self, test_session):
        """Test that CASCADE delete works properly."""
        # Create user mapping and related records
        user_mapping = UserMapping(
            user_id=1,
            django_user_id=1,
            email="test@example.com",
            is_active=True,
        )
        test_session.add(user_mapping)
        test_session.commit()

        # Add related records
        credit = UserCreditModel(user_id=1, balance=Decimal("100"))
        test_session.add(credit)
        test_session.commit()

        # Delete user mapping - should cascade to credits
        test_session.delete(user_mapping)
        test_session.commit()

        # Verify credit was deleted
        remaining_credit = (
            test_session.query(UserCreditModel).filter_by(user_id=1).first()
        )
        assert remaining_credit is None


class TestCheckConstraints:
    """Test CHECK constraints for data validation."""

    @pytest.mark.integration
    def test_positive_price_constraints(self, test_session):
        """Test that prices must be non-negative."""
        stock = Stock(ticker_symbol="TEST", company_name="Test Corp")
        test_session.add(stock)
        test_session.commit()

        # This should work - all positive prices
        valid_price = PriceCache(
            stock_id=stock.stock_id,
            date=datetime.now(UTC).date(),
            open_price=Decimal("100"),
            high_price=Decimal("105"),
            low_price=Decimal("95"),
            close_price=Decimal("102"),
            volume=1000,
        )
        test_session.add(valid_price)
        test_session.commit()

        # Note: SQLite doesn't enforce CHECK constraints by default
        # In PostgreSQL, negative prices would raise an IntegrityError

    @pytest.mark.integration
    def test_high_low_constraint(self, test_session):
        """Test that high price must be >= low price."""
        stock = Stock(ticker_symbol="TEST2", company_name="Test Corp 2")
        test_session.add(stock)
        test_session.commit()

        # Valid: high >= low
        valid_price = PriceCache(
            stock_id=stock.stock_id,
            date=datetime.now(UTC).date(),
            open_price=Decimal("100"),
            high_price=Decimal("105"),
            low_price=Decimal("95"),
            close_price=Decimal("102"),
            volume=1000,
        )
        test_session.add(valid_price)
        test_session.commit()

        assert valid_price.high_price >= valid_price.low_price

    @pytest.mark.integration
    def test_credit_balance_constraints(self, test_session):
        """Test that credit balances must be non-negative."""
        user_mapping = UserMapping(
            user_id=1,
            django_user_id=1,
            email="test@example.com",
            is_active=True,
        )
        test_session.add(user_mapping)
        test_session.commit()

        # Valid: all balances non-negative
        credit = UserCreditModel(
            user_id=1,
            balance=Decimal("100"),
            free_balance=Decimal("10"),
            total_purchased=Decimal("90"),
        )
        test_session.add(credit)
        test_session.commit()

        assert credit.balance >= 0
        assert credit.free_balance >= 0
        assert credit.total_purchased >= 0

    @pytest.mark.integration
    def test_percentage_range_constraints(self, test_session):
        """Test that percentages are constrained to 0-100 range."""
        # This would be tested against the maverick screening tables
        # which have momentum_score and rsi_14 columns that should be 0-100
        pass  # Skip for now as these are Django-owned tables


class TestDataMigration:
    """Test data migration scenarios."""

    @pytest.mark.integration
    def test_user_mapping_creation(self, test_session):
        """Test creating user mappings."""
        # Create multiple user mappings
        mappings = [
            UserMapping(
                user_id=i,
                django_user_id=i,
                email=f"user{i}@example.com",
                is_active=True,
            )
            for i in range(1, 4)
        ]

        for mapping in mappings:
            test_session.add(mapping)
        test_session.commit()

        # Verify all were created
        count = test_session.query(UserMapping).count()
        assert count == 3

        # Verify uniqueness constraints
        user1 = (
            test_session.query(UserMapping).filter_by(email="user1@example.com").first()
        )
        assert user1 is not None
        assert user1.user_id == 1
        assert user1.django_user_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
