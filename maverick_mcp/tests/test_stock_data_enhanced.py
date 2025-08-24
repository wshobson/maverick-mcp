"""
Comprehensive tests for the enhanced stock data provider with SQLAlchemy integration.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from maverick_mcp.data.models import (
    Base,
    MaverickBearStocks,
    MaverickStocks,
    PriceCache,
    Stock,
    SupplyDemandBreakoutStocks,
)
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider


@pytest.fixture(scope="module")
def test_db():
    """Create a test database for the tests."""
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    yield engine

    engine.dispose()


@pytest.fixture
def db_session(test_db):
    """Create a database session for each test."""
    # Clear all data before each test
    Base.metadata.drop_all(bind=test_db)
    Base.metadata.create_all(bind=test_db)

    SessionLocal = sessionmaker(bind=test_db)
    session = SessionLocal()

    yield session

    session.rollback()
    session.close()


@pytest.fixture
def provider():
    """Create an instance of the enhanced provider."""
    return EnhancedStockDataProvider()


@pytest.fixture
def sample_stock(db_session):
    """Create a sample stock in the database."""
    stock = Stock(
        ticker_symbol="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        exchange="NASDAQ",
        currency="USD",
    )
    db_session.add(stock)
    db_session.commit()
    return stock


@pytest.fixture
def sample_price_data(db_session, sample_stock):
    """Create sample price data in the database."""
    prices = []
    base_date = datetime(2024, 1, 1).date()

    for i in range(5):
        price = PriceCache(
            stock_id=sample_stock.stock_id,
            date=base_date + timedelta(days=i),
            open_price=Decimal(f"{150 + i}.00"),
            high_price=Decimal(f"{155 + i}.00"),
            low_price=Decimal(f"{149 + i}.00"),
            close_price=Decimal(f"{152 + i}.00"),
            volume=1000000 + i * 10000,
        )
        prices.append(price)

    db_session.add_all(prices)
    db_session.commit()
    return prices


@pytest.fixture
def mock_yfinance():
    """Mock yfinance responses."""
    with patch("maverick_mcp.providers.stock_data.yf") as mock_yf:
        # Mock ticker
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        # Mock history data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
                "High": [155.0, 156.0, 157.0, 158.0, 159.0],
                "Low": [149.0, 150.0, 151.0, 152.0, 153.0],
                "Close": [152.0, 153.0, 154.0, 155.0, 156.0],
                "Volume": [1000000, 1010000, 1020000, 1030000, 1040000],
            },
            index=dates,
        )
        mock_ticker.history.return_value = mock_df

        # Mock info
        mock_ticker.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NASDAQ",
            "currency": "USD",
            "country": "United States",
            "previousClose": 151.0,
            "quoteType": "EQUITY",
        }

        # Mock other attributes
        mock_ticker.news = []
        mock_ticker.earnings = pd.DataFrame()
        mock_ticker.earnings_dates = pd.DataFrame()
        mock_ticker.earnings_trend = {}
        mock_ticker.recommendations = pd.DataFrame()

        yield mock_yf


class TestEnhancedStockDataProvider:
    """Test the enhanced stock data provider."""

    def test_singleton_pattern(self):
        """Test that provider follows singleton pattern."""
        provider1 = EnhancedStockDataProvider()
        provider2 = EnhancedStockDataProvider()
        assert provider1 is provider2

    def test_get_db_session(self, provider, monkeypatch):
        """Test database session retrieval."""
        mock_session = MagicMock()
        mock_get_db = MagicMock(return_value=iter([mock_session]))

        monkeypatch.setattr("maverick_mcp.providers.stock_data.get_db", mock_get_db)

        session = provider._get_db_session()
        assert session == mock_session

    def test_get_or_create_stock_existing(self, provider, db_session, sample_stock):
        """Test getting an existing stock."""
        stock = provider._get_or_create_stock(db_session, "AAPL")
        assert stock.stock_id == sample_stock.stock_id
        assert stock.ticker_symbol == "AAPL"

    def test_get_or_create_stock_new(self, provider, db_session, mock_yfinance):
        """Test creating a new stock."""
        stock = provider._get_or_create_stock(db_session, "GOOGL")
        assert stock.ticker_symbol == "GOOGL"
        assert stock.company_name == "Apple Inc."  # From mock
        assert stock.sector == "Technology"

        # Verify it was saved
        found = db_session.query(Stock).filter_by(ticker_symbol="GOOGL").first()
        assert found is not None

    def test_get_cached_price_data(
        self, provider, db_session, sample_stock, sample_price_data, monkeypatch
    ):
        """Test retrieving cached price data."""

        # Mock the get_db function to return our test session
        def mock_get_db():
            yield db_session

        monkeypatch.setattr("maverick_mcp.providers.stock_data.get_db", mock_get_db)

        df = provider._get_cached_price_data(
            db_session, "AAPL", "2024-01-01", "2024-01-05"
        )

        assert not df.empty
        assert len(df) == 5
        assert df.index[0] == pd.Timestamp("2024-01-01")
        assert df["Close"].iloc[0] == 152.0

    def test_get_cached_price_data_partial_range(
        self, provider, db_session, sample_stock, sample_price_data
    ):
        """Test retrieving cached data for partial range."""
        df = provider._get_cached_price_data(
            db_session, "AAPL", "2024-01-02", "2024-01-04"
        )

        assert not df.empty
        assert len(df) == 3
        assert df.index[0] == pd.Timestamp("2024-01-02")
        assert df.index[-1] == pd.Timestamp("2024-01-04")

    def test_get_cached_price_data_no_data(self, provider, db_session):
        """Test retrieving cached data when none exists."""
        df = provider._get_cached_price_data(
            db_session, "TSLA", "2024-01-01", "2024-01-05"
        )

        assert df is None

    def test_cache_price_data(self, provider, db_session, sample_stock):
        """Test caching price data."""
        # Create test DataFrame
        dates = pd.date_range("2024-02-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Open": [160.0, 161.0, 162.0],
                "High": [165.0, 166.0, 167.0],
                "Low": [159.0, 160.0, 161.0],
                "Close": [162.0, 163.0, 164.0],
                "Volume": [2000000, 2100000, 2200000],
            },
            index=dates,
        )

        provider._cache_price_data(db_session, "AAPL", df)

        # Verify data was cached
        prices = (
            db_session.query(PriceCache)
            .filter(
                PriceCache.stock_id == sample_stock.stock_id,
                PriceCache.date >= dates[0].date(),
            )
            .all()
        )

        assert len(prices) == 3
        assert prices[0].close_price == Decimal("162.00")

    def test_get_stock_data_with_cache(
        self, provider, db_session, sample_stock, sample_price_data, monkeypatch
    ):
        """Test getting stock data with cache hit."""

        # Mock the get_db function to return our test session
        def mock_get_db():
            yield db_session

        monkeypatch.setattr("maverick_mcp.providers.stock_data.get_db", mock_get_db)

        df = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-05", use_cache=True)

        assert not df.empty
        assert len(df) == 5
        assert df["Close"].iloc[0] == 152.0

    def test_get_stock_data_without_cache(self, provider, mock_yfinance):
        """Test getting stock data without cache."""
        df = provider.get_stock_data(
            "AAPL", "2024-01-01", "2024-01-05", use_cache=False
        )

        assert not df.empty
        assert len(df) == 5
        assert df["Close"].iloc[0] == 152.0

    def test_get_stock_data_cache_miss(
        self, provider, db_session, mock_yfinance, monkeypatch
    ):
        """Test getting stock data with cache miss."""
        # Mock the session getter
        monkeypatch.setattr(provider, "_get_db_session", lambda: db_session)

        df = provider.get_stock_data("TSLA", "2024-01-01", "2024-01-05", use_cache=True)

        assert not df.empty
        assert len(df) == 5
        # Data should come from mock yfinance
        assert df["Close"].iloc[0] == 152.0

    def test_get_stock_data_non_daily_interval(self, provider, mock_yfinance):
        """Test that non-daily intervals bypass cache."""
        df = provider.get_stock_data("AAPL", interval="1wk", period="1mo")

        assert not df.empty
        # Should call yfinance directly
        mock_yfinance.Ticker.return_value.history.assert_called_with(
            period="1mo", interval="1wk"
        )


class TestMaverickRecommendations:
    """Test maverick screening recommendation methods."""

    @pytest.fixture
    def sample_maverick_stocks(self, db_session):
        """Create sample maverick stocks."""
        stocks = []
        for i in range(3):
            stock = MaverickStocks(
                id=i + 1,  # Add explicit ID for SQLite
                stock=f"STOCK{i}",
                close=100.0 + i * 10,
                volume=1000000,
                momentum_score=95.0 - i * 5,
                adr_pct=3.0 + i * 0.5,
                pat="Cup&Handle" if i == 0 else "Base",
                sqz="active" if i < 2 else "neutral",
                consolidation="yes" if i == 0 else "no",
                entry=f"{102.0 + i * 10}",
                combined_score=95 - i * 5,
                compression_score=90 - i * 3,
                pattern_detected=1,
            )
            stocks.append(stock)

        db_session.add_all(stocks)
        db_session.commit()
        return stocks

    def test_get_maverick_recommendations(
        self, provider, db_session, sample_maverick_stocks, monkeypatch
    ):
        """Test getting maverick recommendations."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: db_session)

        recommendations = provider.get_maverick_recommendations(limit=2)

        assert len(recommendations) == 2
        assert recommendations[0]["stock"] == "STOCK0"
        assert recommendations[0]["combined_score"] == 95
        assert recommendations[0]["recommendation_type"] == "maverick_bullish"
        assert "reason" in recommendations[0]
        assert "Exceptional combined score" in recommendations[0]["reason"]

    def test_get_maverick_recommendations_with_min_score(
        self, provider, db_session, sample_maverick_stocks, monkeypatch
    ):
        """Test getting maverick recommendations with minimum score filter."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: db_session)

        recommendations = provider.get_maverick_recommendations(limit=10, min_score=90)

        assert len(recommendations) == 2  # Only STOCK0 and STOCK1 have score >= 90
        assert all(rec["combined_score"] >= 90 for rec in recommendations)

    @pytest.fixture
    def sample_bear_stocks(self, db_session):
        """Create sample bear stocks."""
        stocks = []
        for i in range(3):
            stock = MaverickBearStocks(
                id=i + 1,  # Add explicit ID for SQLite
                stock=f"BEAR{i}",
                close=50.0 - i * 5,
                volume=500000,
                momentum_score=30.0 - i * 5,
                rsi_14=28.0 - i * 3,
                macd=-0.5 - i * 0.1,
                adr_pct=4.0 + i * 0.5,
                atr_contraction=i < 2,
                big_down_vol=i == 0,
                score=90 - i * 5,
                sqz="red" if i < 2 else "neutral",
            )
            stocks.append(stock)

        db_session.add_all(stocks)
        db_session.commit()
        return stocks

    def test_get_maverick_bear_recommendations(
        self, provider, db_session, sample_bear_stocks, monkeypatch
    ):
        """Test getting bear recommendations."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: db_session)

        recommendations = provider.get_maverick_bear_recommendations(limit=2)

        assert len(recommendations) == 2
        assert recommendations[0]["stock"] == "BEAR0"
        assert recommendations[0]["score"] == 90
        assert recommendations[0]["recommendation_type"] == "maverick_bearish"
        assert "reason" in recommendations[0]
        assert "Exceptional bear score" in recommendations[0]["reason"]

    @pytest.fixture
    def sample_trending_stocks(self, db_session):
        """Create sample trending stocks."""
        stocks = []
        for i in range(3):
            stock = SupplyDemandBreakoutStocks(
                id=i + 1,  # Add explicit ID for SQLite
                stock=f"MNRV{i}",
                close=200.0 + i * 10,
                volume=2000000,
                ema_21=195.0 + i * 9,
                sma_50=190.0 + i * 8,
                sma_150=185.0 + i * 7,
                sma_200=180.0 + i * 6,
                momentum_score=92.0 - i * 2,
                adr_pct=2.8 + i * 0.2,
                pat="Base" if i == 0 else "Flag",
                sqz="neutral",
                consolidation="yes" if i < 2 else "no",
                entry=f"{202.0 + i * 10}",
            )
            stocks.append(stock)

        db_session.add_all(stocks)
        db_session.commit()
        return stocks

    def test_get_trending_recommendations(
        self, provider, db_session, sample_trending_stocks, monkeypatch
    ):
        """Test getting trending recommendations."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: db_session)

        recommendations = provider.get_trending_recommendations(limit=2)

        assert len(recommendations) == 2
        assert recommendations[0]["stock"] == "MNRV0"
        assert recommendations[0]["momentum_score"] == 92.0
        assert recommendations[0]["recommendation_type"] == "trending_stage2"
        assert "reason" in recommendations[0]
        assert "Uptrend" in recommendations[0]["reason"]

    def test_get_all_screening_recommendations(self, provider, monkeypatch):
        """Test getting all screening recommendations."""
        mock_results = {
            "maverick_stocks": [
                {"stock": "AAPL", "combined_score": 95, "momentum_score": 90}
            ],
            "maverick_bear_stocks": [
                {"stock": "BEAR", "score": 88, "momentum_score": 25}
            ],
            "trending_stocks": [{"stock": "MSFT", "momentum_score": 91}],
        }

        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.get_latest_maverick_screening",
            lambda: mock_results,
        )

        results = provider.get_all_screening_recommendations()

        assert "maverick_stocks" in results
        assert "maverick_bear_stocks" in results
        assert "trending_stocks" in results

        # Check that reasons were added
        assert (
            results["maverick_stocks"][0]["recommendation_type"] == "maverick_bullish"
        )
        assert "reason" in results["maverick_stocks"][0]

        assert (
            results["maverick_bear_stocks"][0]["recommendation_type"]
            == "maverick_bearish"
        )
        assert "reason" in results["maverick_bear_stocks"][0]

        assert results["trending_stocks"][0]["recommendation_type"] == "trending_stage2"
        assert "reason" in results["trending_stocks"][0]


class TestBackwardCompatibility:
    """Test backward compatibility with original StockDataProvider."""

    def test_get_stock_info(self, provider, mock_yfinance):
        """Test get_stock_info method."""
        info = provider.get_stock_info("AAPL")
        assert info["longName"] == "Apple Inc."
        assert info["sector"] == "Technology"

    def test_get_realtime_data(self, provider, mock_yfinance):
        """Test get_realtime_data method."""
        data = provider.get_realtime_data("AAPL")

        assert data is not None
        assert data["symbol"] == "AAPL"
        assert data["price"] == 156.0  # Last close from mock
        assert data["change"] == 5.0  # 156 - 151 (previousClose)
        assert data["change_percent"] == pytest.approx(3.31, rel=0.01)

    def test_get_all_realtime_data(self, provider, mock_yfinance):
        """Test get_all_realtime_data method."""
        results = provider.get_all_realtime_data(["AAPL", "GOOGL"])

        assert len(results) == 2
        assert "AAPL" in results
        assert "GOOGL" in results
        assert results["AAPL"]["price"] == 156.0

    def test_is_market_open(self, provider, monkeypatch):
        """Test is_market_open method."""
        import pytz

        # Mock a weekday at 10 AM Eastern
        mock_now = datetime(2024, 1, 2, 10, 0, 0)  # Tuesday
        mock_now = pytz.timezone("US/Eastern").localize(mock_now)

        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.datetime",
            MagicMock(now=MagicMock(return_value=mock_now)),
        )

        assert provider.is_market_open() is True

        # Mock a weekend
        mock_now = datetime(2024, 1, 6, 10, 0, 0)  # Saturday
        mock_now = pytz.timezone("US/Eastern").localize(mock_now)

        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.datetime",
            MagicMock(now=MagicMock(return_value=mock_now)),
        )

        assert provider.is_market_open() is False

    def test_get_news(self, provider, mock_yfinance):
        """Test get_news method."""
        mock_news = [
            {
                "title": "Apple News 1",
                "publisher": "Reuters",
                "link": "https://example.com/1",
                "providerPublishTime": 1704134400,  # 2024-01-01 timestamp
                "type": "STORY",
            }
        ]
        mock_yfinance.Ticker.return_value.news = mock_news

        df = provider.get_news("AAPL", limit=5)

        assert not df.empty
        assert len(df) == 1
        assert df["title"].iloc[0] == "Apple News 1"
        assert isinstance(df["providerPublishTime"].iloc[0], pd.Timestamp)

    def test_get_earnings(self, provider, mock_yfinance):
        """Test get_earnings method."""
        result = provider.get_earnings("AAPL")

        assert "earnings" in result
        assert "earnings_dates" in result
        assert "earnings_trend" in result

    def test_get_recommendations(self, provider, mock_yfinance):
        """Test get_recommendations method."""
        df = provider.get_recommendations("AAPL")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["firm", "toGrade", "fromGrade", "action"]

    def test_is_etf(self, provider, mock_yfinance):
        """Test is_etf method."""
        # Test regular stock
        assert provider.is_etf("AAPL") is False

        # Test ETF
        mock_yfinance.Ticker.return_value.info["quoteType"] = "ETF"
        assert provider.is_etf("SPY") is True

        # Test by symbol pattern
        assert provider.is_etf("QQQ") is True


class TestErrorHandling:
    """Test error handling in the enhanced provider."""

    def test_get_stock_data_error_handling(self, provider, mock_yfinance, monkeypatch):
        """Test error handling in get_stock_data."""
        # Mock an exception for all yfinance calls
        mock_yfinance.Ticker.return_value.history.side_effect = Exception("API Error")

        # Also mock the database session to ensure no cache is used
        def mock_get_db_session():
            raise Exception("Database error")

        monkeypatch.setattr(provider, "_get_db_session", mock_get_db_session)

        # Now the provider should return empty DataFrame since both cache and yfinance fail
        df = provider.get_stock_data(
            "AAPL", "2024-01-01", "2024-01-05", use_cache=False
        )

        assert df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_get_cached_price_data_error_handling(
        self, provider, db_session, monkeypatch
    ):
        """Test error handling in _get_cached_price_data."""

        # Mock a database error
        def mock_get_price_data(*args, **kwargs):
            raise Exception("Database error")

        monkeypatch.setattr(PriceCache, "get_price_data", mock_get_price_data)

        result = provider._get_cached_price_data(
            db_session, "AAPL", "2024-01-01", "2024-01-05"
        )
        assert result is None

    def test_cache_price_data_error_handling(self, provider, db_session, monkeypatch):
        """Test error handling in _cache_price_data."""

        # Mock a database error
        def mock_bulk_insert(*args, **kwargs):
            raise Exception("Insert error")

        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.bulk_insert_price_data", mock_bulk_insert
        )

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [155.0, 156.0, 157.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [152.0, 153.0, 154.0],
                "Volume": [1000000, 1010000, 1020000],
            },
            index=dates,
        )

        # Should not raise exception
        provider._cache_price_data(db_session, "AAPL", df)

    def test_get_maverick_recommendations_error_handling(self, provider, monkeypatch):
        """Test error handling in get_maverick_recommendations."""
        # Mock a database session that throws when used
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database query error")
        mock_session.close = MagicMock()

        monkeypatch.setattr(provider, "_get_db_session", lambda: mock_session)

        recommendations = provider.get_maverick_recommendations()
        assert recommendations == []
