"""
Integration tests for database operations.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

from tests.integration.base import DatabaseIntegrationTest


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseOperations(DatabaseIntegrationTest):
    """Test database operations with real PostgreSQL instance."""

    async def test_stock_crud_operations(self):
        """Test CRUD operations for stock data."""
        from maverick_mcp.data.models import Stock

        # Create
        stock = Stock(
            ticker_symbol="DBTEST",
            company_name="Database Test Stock",
            sector="Technology",
            industry="Software",
        )
        self.db.add(stock)
        self.db.commit()
        self.db.refresh(stock)

        assert stock.stock_id is not None
        assert stock.created_at is not None

        # Read
        retrieved = self.db.query(Stock).filter_by(ticker_symbol="DBTEST").first()
        assert retrieved is not None
        assert retrieved.company_name == "Database Test Stock"
        assert retrieved.sector == "Technology"

        # Update
        retrieved.sector = "Financial"
        self.db.commit()

        updated = self.db.query(Stock).filter_by(ticker_symbol="DBTEST").first()
        assert updated.sector == "Financial"
        assert updated.updated_at > updated.created_at

        # Delete
        self.db.delete(updated)
        self.db.commit()

        deleted = self.db.query(Stock).filter_by(ticker_symbol="DBTEST").first()
        assert deleted is None

    async def test_price_cache_with_constraints(self):
        """Test price cache with database constraints."""
        from maverick_mcp.data.models import PriceCache

        # Create stock
        stock = self.create_test_stock("CONSTRAINT", "Constraint Test")

        # Test unique constraint on (stock_id, date)
        date = datetime.now().date()
        price1 = PriceCache(
            stock_id=stock.stock_id,
            date=date,
            open_price=100.0,
            high_price=105.0,
            low_price=99.0,
            close_price=103.0,
            volume=1000000,
        )
        self.db.add(price1)
        self.db.commit()

        # Try to add duplicate (should fail)
        price2 = PriceCache(
            stock_id=stock.stock_id,
            date=date,
            open_price=101.0,
            high_price=106.0,
            low_price=100.0,
            close_price=104.0,
            volume=2000000,
        )
        self.db.add(price2)

        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            self.db.commit()

        self.db.rollback()

        # Test check constraints (high >= low, etc.)
        invalid_price = PriceCache(
            stock_id=stock.stock_id,
            date=date + timedelta(days=1),
            open_price=100.0,
            high_price=99.0,  # Invalid: high < low
            low_price=105.0,
            close_price=102.0,
            volume=1000000,
        )
        self.db.add(invalid_price)

        from sqlalchemy.exc import DBAPIError, IntegrityError

        with pytest.raises((IntegrityError, DBAPIError)):
            self.db.commit()

        self.db.rollback()

    async def test_transaction_rollback(self):
        """Test database transaction rollback."""
        from maverick_mcp.auth.models import UserCreditModel
        from maverick_mcp.data.models import Stock

        # Start transaction
        initial_count = self.db.query(Stock).count()

        try:
            # Add stock
            stock = Stock(ticker_symbol="ROLLBACK", company_name="Rollback Test")
            self.db.add(stock)

            # Add user credits (this will fail due to missing user)
            credits = UserCreditModel(
                user_id=99999,  # Non-existent user
                balance=1000,
            )
            self.db.add(credits)

            # This should fail
            self.db.commit()
        except Exception:
            self.db.rollback()

        # Verify nothing was saved
        final_count = self.db.query(Stock).count()
        assert final_count == initial_count

        rollback_stock = (
            self.db.query(Stock).filter_by(ticker_symbol="ROLLBACK").first()
        )
        assert rollback_stock is None

    async def test_cascade_delete(self):
        """Test cascade delete operations."""
        from maverick_mcp.auth.models import RefreshToken, User
        from maverick_mcp.data.models import PriceCache

        # Create stock with price data
        stock = self.create_test_stock("CASCADE", "Cascade Test")
        self.create_test_price_data(stock.stock_id, days=10)

        # Verify price data exists
        price_count = (
            self.db.query(PriceCache).filter_by(stock_id=stock.stock_id).count()
        )
        assert price_count == 10

        # Delete stock (should cascade to price data)
        self.db.delete(stock)
        self.db.commit()

        # Verify price data was deleted
        price_count = (
            self.db.query(PriceCache).filter_by(stock_id=stock.stock_id).count()
        )
        assert price_count == 0

        # Test user cascade
        user = User(
            email="cascade@test.com",
            name="cascadeuser",
        )
        user.set_password("password")
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        # Add refresh token
        token = RefreshToken(
            user_id=user.id,
            jti="test-jti-token",
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        self.db.add(token)
        self.db.commit()

        # Delete user (should cascade to tokens)
        self.db.delete(user)
        self.db.commit()

        # Verify token was deleted
        token_exists = (
            self.db.query(RefreshToken).filter_by(jti="test-jti-token").first()
        )
        assert token_exists is None

    async def test_database_indexes(self):
        """Test that database indexes are working properly."""
        from maverick_mcp.data.models import PriceCache, Stock

        # Create test data
        for i in range(100):
            stock = Stock(
                ticker_symbol=f"IDX{i:03d}",
                company_name=f"Index Test {i}",
                sector="Technology" if i % 2 == 0 else "Finance",
            )
            self.db.add(stock)
        self.db.commit()

        # Test index on symbol (should be fast)
        result = self.db.execute(
            text(
                "EXPLAIN ANALYZE SELECT * FROM stocks_stock WHERE ticker_symbol = :ticker_symbol"
            ),
            {"ticker_symbol": "IDX050"},
        )

        # Check that index scan is used (not seq scan)
        plan = result.fetchall()
        plan_text = str(plan)
        assert "Index Scan" in plan_text or "index" in plan_text.lower()

        # Test composite index on price cache
        stock = self.db.query(Stock).first()
        for i in range(100):
            price = PriceCache(
                stock_id=stock.stock_id,
                date=datetime.now() - timedelta(days=i),
                open_price=100 + i,
                high_price=105 + i,
                low_price=95 + i,
                close_price=102 + i,
                volume=1000000,
            )
            self.db.add(price)
        self.db.commit()

        # Query using composite index
        result = self.db.execute(
            text(
                "EXPLAIN ANALYZE SELECT * FROM stocks_pricecache "
                "WHERE stock_id = :stock_id AND date >= :date"
            ),
            {"stock_id": stock.stock_id, "date": datetime.now() - timedelta(days=30)},
        )

        plan = result.fetchall()
        plan_text = str(plan)
        assert "Index Scan" in plan_text or "index" in plan_text.lower()

    async def test_concurrent_credit_transactions(self):
        """Test concurrent credit transactions to ensure data integrity."""
        import asyncio

        from maverick_mcp.auth.models import User, UserCreditModel

        # Create user with credits
        user = User(email="concurrent@test.com", name="concurrentuser")
        user.set_password("password")
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        credits = UserCreditModel(
            user_id=user.id,
            balance=1000,
            free_balance=0,
        )
        self.db.add(credits)
        self.db.commit()

        # Simulate concurrent credit deductions
        async def deduct_credits(amount: int):
            # Each operation should use its own session
            from maverick_mcp.data.models import SessionLocal

            db = SessionLocal()
            try:
                # Use row-level locking
                user_credits = (
                    db.query(UserCreditModel)
                    .filter_by(user_id=user.id)
                    .with_for_update()
                    .first()
                )

                if user_credits and user_credits.balance >= amount:
                    user_credits.balance -= amount
                    db.commit()
                    return True
                return False
            finally:
                db.close()

        # Run concurrent deductions
        tasks = [deduct_credits(100) for _ in range(15)]  # 15 * 100 = 1500 > 1000
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify final balance
        final_credits = (
            self.db.query(UserCreditModel).filter_by(user_id=user.id).first()
        )

        # Balance should never go negative
        assert final_credits.balance >= 0
        assert final_credits.balance <= 100  # At most 100 left

        # Count successful transactions
        successful = sum(1 for r in results if r is True)
        assert successful == 10  # Exactly 10 transactions of 100 each

    async def test_database_migrations_compatibility(self):
        """Test that database schema is compatible with migrations."""
        # Check critical tables exist
        tables = [
            "stocks_stock",
            "stocks_pricecache",
            "mcp_temp_users",
            "mcp_user_credits",
            "mcp_credit_transactions",
            "mcp_api_keys",
            "mcp_refresh_tokens",
        ]

        for table in tables:
            result = self.db.execute(
                text(
                    "SELECT EXISTS ("
                    "SELECT FROM information_schema.tables "
                    "WHERE table_name = :table"
                    ")"
                ),
                {"table": table},
            )
            exists = result.scalar()
            assert exists, f"Table {table} does not exist"

        # Check critical columns
        critical_columns = {
            "stocks_stock": ["stock_id", "ticker_symbol", "company_name", "created_at"],
            "mcp_temp_users": ["id", "email", "name", "password_hash"],
            "mcp_user_credits": ["user_id", "balance", "free_balance"],
        }

        for table, columns in critical_columns.items():
            for column in columns:
                result = self.db.execute(
                    text(
                        "SELECT EXISTS ("
                        "SELECT FROM information_schema.columns "
                        "WHERE table_name = :table AND column_name = :column"
                        ")"
                    ),
                    {"table": table, "column": column},
                )
                exists = result.scalar()
                assert exists, f"Column {column} in table {table} does not exist"
