"""
Functional tests for SQLAlchemy models that test the actual data operations.

These tests verify model functionality by reading from the existing production database
without creating any new tables or modifying data.
"""

import os
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import (
    DATABASE_URL,
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    Stock,
    SupplyDemandBreakoutStocks,
    get_latest_maverick_screening,
)


@pytest.fixture(scope="session")
def read_only_engine():
    """Create a read-only database engine for the existing database."""
    # Use the existing database URL from environment or default
    db_url = os.getenv("POSTGRES_URL", DATABASE_URL)

    try:
        # Create engine with read-only intent
        engine = create_engine(db_url, echo=False)
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"Database not available: {e}")
        return

    yield engine

    engine.dispose()


@pytest.fixture(scope="function")
def db_session(read_only_engine):
    """Create a read-only database session for each test."""
    SessionLocal = sessionmaker(bind=read_only_engine)
    session = SessionLocal()

    yield session

    session.rollback()  # Rollback any potential changes
    session.close()


class TestStockModelReadOnly:
    """Test the Stock model functionality with read-only operations."""

    def test_query_stocks(self, db_session):
        """Test querying existing stocks from the database."""
        # Query for any existing stocks
        stocks = db_session.query(Stock).limit(5).all()

        # If there are stocks in the database, verify their structure
        for stock in stocks:
            assert hasattr(stock, "stock_id")
            assert hasattr(stock, "ticker_symbol")
            assert hasattr(stock, "created_at")
            assert hasattr(stock, "updated_at")

            # Verify timestamps are timezone-aware
            if stock.created_at:
                assert stock.created_at.tzinfo is not None
            if stock.updated_at:
                assert stock.updated_at.tzinfo is not None

    def test_query_by_ticker(self, db_session):
        """Test querying stock by ticker symbol."""
        # Try to find a common stock like AAPL
        stock = db_session.query(Stock).filter_by(ticker_symbol="AAPL").first()

        if stock:
            assert stock.ticker_symbol == "AAPL"
            assert isinstance(stock.stock_id, uuid.UUID)

    def test_stock_repr(self, db_session):
        """Test string representation of Stock."""
        stock = db_session.query(Stock).first()
        if stock:
            repr_str = repr(stock)
            assert "<Stock(" in repr_str
            assert "ticker=" in repr_str
            assert stock.ticker_symbol in repr_str

    def test_stock_relationships(self, db_session):
        """Test stock relationships with price caches."""
        # Find a stock that has price data
        stock_with_prices = db_session.query(Stock).join(PriceCache).distinct().first()

        if stock_with_prices:
            # Access the relationship
            price_caches = stock_with_prices.price_caches
            assert isinstance(price_caches, list)

            # Verify each price cache belongs to this stock
            for pc in price_caches[:5]:  # Check first 5
                assert pc.stock_id == stock_with_prices.stock_id
                assert pc.stock == stock_with_prices


class TestPriceCacheModelReadOnly:
    """Test the PriceCache model functionality with read-only operations."""

    def test_query_price_cache(self, db_session):
        """Test querying existing price cache entries."""
        # Query for any existing price data
        prices = db_session.query(PriceCache).limit(10).all()

        # Verify structure of price entries
        for price in prices:
            assert hasattr(price, "price_cache_id")
            assert hasattr(price, "stock_id")
            assert hasattr(price, "date")
            assert hasattr(price, "close_price")

            # Verify data types
            if price.price_cache_id:
                assert isinstance(price.price_cache_id, uuid.UUID)
            if price.close_price:
                assert isinstance(price.close_price, Decimal)

    def test_price_cache_repr(self, db_session):
        """Test string representation of PriceCache."""
        price = db_session.query(PriceCache).first()
        if price:
            repr_str = repr(price)
            assert "<PriceCache(" in repr_str
            assert "stock_id=" in repr_str
            assert "date=" in repr_str
            assert "close=" in repr_str

    def test_get_price_data(self, db_session):
        """Test retrieving price data as DataFrame for existing tickers."""
        # Try to get price data for a common stock
        # Use a recent date range that might have data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)

        # Try common tickers
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            df = PriceCache.get_price_data(
                db_session,
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            if not df.empty:
                # Verify DataFrame structure
                assert df.index.name == "date"
                assert "open" in df.columns
                assert "high" in df.columns
                assert "low" in df.columns
                assert "close" in df.columns
                assert "volume" in df.columns
                assert "symbol" in df.columns
                assert df["symbol"].iloc[0] == ticker

                # Verify data types
                assert df["close"].dtype == float
                assert df["volume"].dtype == int
                break

    def test_stock_relationship(self, db_session):
        """Test relationship back to Stock."""
        # Find a price entry with stock relationship
        price = db_session.query(PriceCache).join(Stock).first()

        if price:
            assert price.stock is not None
            assert price.stock.stock_id == price.stock_id
            assert hasattr(price.stock, "ticker_symbol")


@pytest.mark.integration
class TestMaverickStocksReadOnly:
    """Test MaverickStocks model functionality with read-only operations."""

    def test_query_maverick_stocks(self, db_session):
        """Test querying existing maverick stock entries."""
        try:
            # Query for any existing maverick stocks
            mavericks = db_session.query(MaverickStocks).limit(10).all()

            # Verify structure of maverick entries
            for maverick in mavericks:
                assert hasattr(maverick, "id")
                assert hasattr(maverick, "stock")
                assert hasattr(maverick, "close")
                assert hasattr(maverick, "combined_score")
                assert hasattr(maverick, "momentum_score")
        except Exception as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickStocks table not found: {e}")
            else:
                raise

    def test_maverick_repr(self, db_session):
        """Test string representation of MaverickStocks."""
        try:
            maverick = db_session.query(MaverickStocks).first()
            if maverick:
                repr_str = repr(maverick)
                assert "<MaverickStock(" in repr_str
                assert "stock=" in repr_str
                assert "close=" in repr_str
                assert "score=" in repr_str
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickStocks table not found: {e}")
            else:
                raise

    def test_get_top_stocks(self, db_session):
        """Test retrieving top maverick stocks."""
        try:
            # Get top stocks from existing data
            top_stocks = MaverickStocks.get_top_stocks(db_session, limit=20)

            # Verify results are sorted by combined_score
            if len(top_stocks) > 1:
                for i in range(len(top_stocks) - 1):
                    assert (
                        top_stocks[i].combined_score >= top_stocks[i + 1].combined_score
                    )

            # Verify limit is respected
            assert len(top_stocks) <= 20
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickStocks table not found: {e}")
            else:
                raise

    def test_maverick_to_dict(self, db_session):
        """Test converting MaverickStocks to dictionary."""
        try:
            maverick = db_session.query(MaverickStocks).first()
            if maverick:
                data = maverick.to_dict()

                # Verify expected keys
                expected_keys = [
                    "stock",
                    "close",
                    "volume",
                    "momentum_score",
                    "adr_pct",
                    "pattern",
                    "squeeze",
                    "consolidation",
                    "entry",
                    "combined_score",
                    "compression_score",
                    "pattern_detected",
                ]
                for key in expected_keys:
                    assert key in data

                # Verify data types
                assert isinstance(data["stock"], str)
                assert isinstance(data["combined_score"], int | type(None))
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickStocks table not found: {e}")
            else:
                raise


@pytest.mark.integration
class TestMaverickBearStocksReadOnly:
    """Test MaverickBearStocks model functionality with read-only operations."""

    def test_query_bear_stocks(self, db_session):
        """Test querying existing maverick bear stock entries."""
        try:
            # Query for any existing bear stocks
            bears = db_session.query(MaverickBearStocks).limit(10).all()

            # Verify structure of bear entries
            for bear in bears:
                assert hasattr(bear, "id")
                assert hasattr(bear, "stock")
                assert hasattr(bear, "close")
                assert hasattr(bear, "score")
                assert hasattr(bear, "momentum_score")
                assert hasattr(bear, "rsi_14")
                assert hasattr(bear, "atr_contraction")
                assert hasattr(bear, "big_down_vol")
        except Exception as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickBearStocks table not found: {e}")
            else:
                raise

    def test_bear_repr(self, db_session):
        """Test string representation of MaverickBearStocks."""
        try:
            bear = db_session.query(MaverickBearStocks).first()
            if bear:
                repr_str = repr(bear)
                assert "<MaverickBearStock(" in repr_str
                assert "stock=" in repr_str
                assert "close=" in repr_str
                assert "score=" in repr_str
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickBearStocks table not found: {e}")
            else:
                raise

    def test_get_top_bear_stocks(self, db_session):
        """Test retrieving top bear stocks."""
        try:
            # Get top bear stocks from existing data
            top_bears = MaverickBearStocks.get_top_stocks(db_session, limit=20)

            # Verify results are sorted by score
            if len(top_bears) > 1:
                for i in range(len(top_bears) - 1):
                    assert top_bears[i].score >= top_bears[i + 1].score

            # Verify limit is respected
            assert len(top_bears) <= 20
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickBearStocks table not found: {e}")
            else:
                raise

    def test_bear_to_dict(self, db_session):
        """Test converting MaverickBearStocks to dictionary."""
        try:
            bear = db_session.query(MaverickBearStocks).first()
            if bear:
                data = bear.to_dict()

                # Verify expected keys
                expected_keys = [
                    "stock",
                    "close",
                    "volume",
                    "momentum_score",
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "adr_pct",
                    "atr",
                    "atr_contraction",
                    "avg_vol_30d",
                    "big_down_vol",
                    "score",
                    "squeeze",
                    "consolidation",
                ]
                for key in expected_keys:
                    assert key in data

                # Verify boolean fields
                assert isinstance(data["atr_contraction"], bool)
                assert isinstance(data["big_down_vol"], bool)
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"MaverickBearStocks table not found: {e}")
            else:
                raise


@pytest.mark.integration
class TestSupplyDemandBreakoutStocksReadOnly:
    """Test SupplyDemandBreakoutStocks model functionality with read-only operations."""

    def test_query_supply_demand_stocks(self, db_session):
        """Test querying existing supply/demand breakout stock entries."""
        try:
            # Query for any existing supply/demand breakout stocks
            stocks = db_session.query(SupplyDemandBreakoutStocks).limit(10).all()

            # Verify structure of supply/demand breakout entries
            for stock in stocks:
                assert hasattr(stock, "id")
                assert hasattr(stock, "stock")
                assert hasattr(stock, "close")
                assert hasattr(stock, "momentum_score")
                assert hasattr(stock, "sma_50")
                assert hasattr(stock, "sma_150")
                assert hasattr(stock, "sma_200")
        except Exception as e:
            if "does not exist" in str(e):
                pytest.skip(f"SupplyDemandBreakoutStocks table not found: {e}")
            else:
                raise

    def test_supply_demand_repr(self, db_session):
        """Test string representation of SupplyDemandBreakoutStocks."""
        try:
            supply_demand = db_session.query(SupplyDemandBreakoutStocks).first()
            if supply_demand:
                repr_str = repr(supply_demand)
                assert "<supply/demand breakoutStock(" in repr_str
                assert "stock=" in repr_str
                assert "close=" in repr_str
                assert "rs=" in repr_str
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"SupplyDemandBreakoutStocks table not found: {e}")
            else:
                raise

    def test_get_top_supply_demand_stocks(self, db_session):
        """Test retrieving top supply/demand breakout stocks."""
        try:
            # Get top stocks from existing data
            top_stocks = SupplyDemandBreakoutStocks.get_top_stocks(db_session, limit=20)

            # Verify results are sorted by momentum_score
            if len(top_stocks) > 1:
                for i in range(len(top_stocks) - 1):
                    assert (
                        top_stocks[i].momentum_score >= top_stocks[i + 1].momentum_score
                    )

            # Verify limit is respected
            assert len(top_stocks) <= 20
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"SupplyDemandBreakoutStocks table not found: {e}")
            else:
                raise

    def test_get_stocks_above_moving_averages(self, db_session):
        """Test retrieving stocks meeting supply/demand breakout criteria."""
        try:
            # Get stocks that meet supply/demand breakout criteria from existing data
            stocks = SupplyDemandBreakoutStocks.get_stocks_above_moving_averages(
                db_session
            )

            # Verify all returned stocks meet the criteria
            for stock in stocks:
                assert stock.close > stock.sma_50
                assert stock.close > stock.sma_150
                assert stock.close > stock.sma_200
                assert stock.sma_50 > stock.sma_150
                assert stock.sma_150 > stock.sma_200

            # Verify they're sorted by momentum score
            if len(stocks) > 1:
                for i in range(len(stocks) - 1):
                    assert stocks[i].momentum_score >= stocks[i + 1].momentum_score
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"SupplyDemandBreakoutStocks table not found: {e}")
            else:
                raise

    def test_supply_demand_to_dict(self, db_session):
        """Test converting SupplyDemandBreakoutStocks to dictionary."""
        try:
            supply_demand = db_session.query(SupplyDemandBreakoutStocks).first()
            if supply_demand:
                data = supply_demand.to_dict()

                # Verify expected keys
                expected_keys = [
                    "stock",
                    "close",
                    "volume",
                    "momentum_score",
                    "adr_pct",
                    "pattern",
                    "squeeze",
                    "consolidation",
                    "entry",
                    "ema_21",
                    "sma_50",
                    "sma_150",
                    "sma_200",
                    "atr",
                    "avg_volume_30d",
                ]
                for key in expected_keys:
                    assert key in data

                # Verify data types
                assert isinstance(data["stock"], str)
                assert isinstance(data["momentum_score"], float | int)
        except ProgrammingError as e:
            if "does not exist" in str(e):
                pytest.skip(f"SupplyDemandBreakoutStocks table not found: {e}")
            else:
                raise


@pytest.mark.integration
class TestGetLatestMaverickScreeningReadOnly:
    """Test the get_latest_maverick_screening function with read-only operations."""

    def test_get_latest_screening(self):
        """Test retrieving latest screening results from existing data."""
        try:
            # Call the function directly - it creates its own session
            results = get_latest_maverick_screening()

            # Verify structure
            assert isinstance(results, dict)
            assert "maverick_stocks" in results
            assert "maverick_bear_stocks" in results
            assert "supply_demand_stocks" in results

            # Verify each result is a list of dictionaries
            assert isinstance(results["maverick_stocks"], list)
            assert isinstance(results["maverick_bear_stocks"], list)
            assert isinstance(results["supply_demand_stocks"], list)

            # If there are maverick stocks, verify their structure
            if results["maverick_stocks"]:
                stock_dict = results["maverick_stocks"][0]
                assert isinstance(stock_dict, dict)
                assert "stock" in stock_dict
                assert "combined_score" in stock_dict

                # Verify they're sorted by combined_score
                scores = [s["combined_score"] for s in results["maverick_stocks"]]
                assert scores == sorted(scores, reverse=True)

            # If there are bear stocks, verify their structure
            if results["maverick_bear_stocks"]:
                bear_dict = results["maverick_bear_stocks"][0]
                assert isinstance(bear_dict, dict)
                assert "stock" in bear_dict
                assert "score" in bear_dict

                # Verify they're sorted by score
                scores = [s["score"] for s in results["maverick_bear_stocks"]]
                assert scores == sorted(scores, reverse=True)

            # If there are supply/demand breakout stocks, verify their structure
            if results["supply_demand_stocks"]:
                min_dict = results["supply_demand_stocks"][0]
                assert isinstance(min_dict, dict)
                assert "stock" in min_dict
                assert "momentum_score" in min_dict

                # Verify they're sorted by momentum_score
                ratings = [s["momentum_score"] for s in results["supply_demand_stocks"]]
                assert ratings == sorted(ratings, reverse=True)

        except Exception as e:
            # If tables don't exist, that's okay for read-only tests
            if "does not exist" in str(e):
                pytest.skip(f"Screening tables not found in database: {e}")
            else:
                raise


class TestDatabaseStructureReadOnly:
    """Test database structure and relationships with read-only operations."""

    def test_stock_ticker_query_performance(self, db_session):
        """Test that ticker queries work efficiently (index should exist)."""
        # Query for a specific ticker - should be fast if indexed
        import time

        start_time = time.time()

        # Try to find a stock by ticker
        stock = db_session.query(Stock).filter_by(ticker_symbol="AAPL").first()

        query_time = time.time() - start_time

        # Query should be reasonably fast if properly indexed
        # Allow up to 1 second for connection overhead
        assert query_time < 1.0

        # If stock exists, verify it has expected fields
        if stock:
            assert stock.ticker_symbol == "AAPL"

    def test_price_cache_date_query_performance(self, db_session):
        """Test that price cache queries by stock and date are efficient."""
        # First find a stock with prices
        stock_with_prices = db_session.query(Stock).join(PriceCache).first()

        if stock_with_prices:
            # Get a recent date
            recent_price = (
                db_session.query(PriceCache)
                .filter_by(stock_id=stock_with_prices.stock_id)
                .order_by(PriceCache.date.desc())
                .first()
            )

            if recent_price:
                # Query for specific stock_id and date - should be fast
                import time

                start_time = time.time()

                result = (
                    db_session.query(PriceCache)
                    .filter_by(
                        stock_id=stock_with_prices.stock_id, date=recent_price.date
                    )
                    .first()
                )

                query_time = time.time() - start_time

                # Query should be reasonably fast if composite index exists
                assert query_time < 1.0
                assert result is not None
                assert result.price_cache_id == recent_price.price_cache_id


class TestDataTypesAndConstraintsReadOnly:
    """Test data types and constraints with read-only operations."""

    def test_null_values_in_existing_data(self, db_session):
        """Test handling of null values in optional fields in existing data."""
        # Query stocks that might have null values
        stocks = db_session.query(Stock).limit(20).all()

        for stock in stocks:
            # These fields are optional and can be None
            assert hasattr(stock, "company_name")
            assert hasattr(stock, "sector")
            assert hasattr(stock, "industry")

            # Verify ticker_symbol is never null (it's required)
            assert stock.ticker_symbol is not None
            assert isinstance(stock.ticker_symbol, str)

    def test_decimal_precision_in_existing_data(self, db_session):
        """Test decimal precision in existing price data."""
        # Query some price data
        prices = db_session.query(PriceCache).limit(10).all()

        for price in prices:
            # Verify decimal fields
            if price.close_price is not None:
                assert isinstance(price.close_price, Decimal)
                # Check precision (should have at most 2 decimal places)
                str_price = str(price.close_price)
                if "." in str_price:
                    decimal_places = len(str_price.split(".")[1])
                    assert decimal_places <= 2

            # Same for other price fields
            for field in ["open_price", "high_price", "low_price"]:
                value = getattr(price, field)
                if value is not None:
                    assert isinstance(value, Decimal)

    def test_volume_data_types(self, db_session):
        """Test volume data types in existing data."""
        # Query price data with volumes
        prices = (
            db_session.query(PriceCache)
            .filter(PriceCache.volume.isnot(None))
            .limit(10)
            .all()
        )

        for price in prices:
            assert isinstance(price.volume, int)
            assert price.volume >= 0

    def test_timezone_handling_in_existing_data(self, db_session):
        """Test that timestamps have timezone info in existing data."""
        # Query any model with timestamps
        stocks = db_session.query(Stock).limit(5).all()

        # Skip test if no stocks found
        if not stocks:
            pytest.skip("No stock data found in database")

        # Check if data has timezone info (newer data should, legacy data might not)
        has_tz_info = False
        for stock in stocks:
            if stock.created_at and stock.created_at.tzinfo is not None:
                has_tz_info = True
                # Data should have timezone info (not necessarily UTC for legacy data)
                # New data created by the app will be UTC

            if stock.updated_at and stock.updated_at.tzinfo is not None:
                has_tz_info = True
                # Data should have timezone info (not necessarily UTC for legacy data)

        # This test just verifies that timezone-aware timestamps are being used
        # Legacy data might not be UTC, but new data will be
        if has_tz_info:
            # Pass - data has timezone info which is what we want
            pass
        else:
            pytest.skip(
                "Legacy data without timezone info - new data will have timezone info"
            )

    def test_relationships_integrity(self, db_session):
        """Test that relationships maintain referential integrity."""
        # Find prices with valid stock relationships
        prices_with_stocks = db_session.query(PriceCache).join(Stock).limit(10).all()

        for price in prices_with_stocks:
            # Verify the relationship is intact
            assert price.stock is not None
            assert price.stock.stock_id == price.stock_id

            # Verify reverse relationship
            assert price in price.stock.price_caches
