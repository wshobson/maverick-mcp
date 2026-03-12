"""
Comprehensive tests for the enhanced stock data provider with SQLAlchemy integration.

Updated to match the current model schema where:
- MaverickStocks/Bear/SupplyDemand use stock_id (FK) + relationship instead of 'stock' string column
- EnhancedStockDataProvider._get_db_session() returns (session, should_close) tuple
- Data access uses YFinancePool connection pool instead of direct yfinance calls
- Column names changed: close -> close_price, pat -> pattern_type, sqz -> squeeze_status, etc.
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
    """Create an instance of the enhanced provider with mocked dependencies."""
    with (
        patch("maverick_mcp.providers.stock_data.get_yfinance_pool") as mock_get_pool,
        patch.object(EnhancedStockDataProvider, "_test_db_connection"),
    ):
        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        p = EnhancedStockDataProvider()
        p._mock_pool = mock_pool  # Store for tests to configure
        return p


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
    """Mock yfinance responses via YFinancePool and direct yf module."""
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

    mock_info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "exchange": "NASDAQ",
        "currency": "USD",
        "country": "United States",
        "previousClose": 151.0,
        "quoteType": "EQUITY",
    }

    # Mock the YFinancePool that the provider actually uses
    mock_pool = MagicMock()
    mock_pool.get_history.return_value = mock_df
    mock_pool.get_info.return_value = mock_info
    mock_pool.batch_download.return_value = mock_df

    # Also mock the direct yf module for _get_or_create_stock and other paths
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df
    mock_ticker.info = mock_info
    mock_ticker.news = []
    mock_ticker.earnings = pd.DataFrame()
    mock_ticker.earnings_dates = pd.DataFrame()
    mock_ticker.earnings_trend = {}
    mock_ticker.recommendations = pd.DataFrame()

    with (
        patch(
            "maverick_mcp.providers.stock_data.get_yfinance_pool",
            return_value=mock_pool,
        ),
        patch("maverick_mcp.providers.stock_data.yf") as mock_yf,
    ):
        mock_yf.Ticker.return_value = mock_ticker
        # Attach pool reference for tests that need it
        mock_yf._pool = mock_pool
        mock_yf._mock_pool = mock_pool
        yield mock_yf


class TestEnhancedStockDataProvider:
    """Test the enhanced stock data provider."""

    def test_provider_instantiation(self):
        """Test that provider can be instantiated."""
        provider1 = EnhancedStockDataProvider()
        provider2 = EnhancedStockDataProvider()
        # Both should be valid instances (singleton behavior is optional)
        assert isinstance(provider1, EnhancedStockDataProvider)
        assert isinstance(provider2, EnhancedStockDataProvider)

    def test_get_db_session(self, provider, monkeypatch):
        """Test database session retrieval returns (session, should_close) tuple."""
        mock_session = MagicMock()

        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.SessionLocal",
            lambda: mock_session,
        )
        # Clear any injected session so factory path is used
        provider._db_session = None

        session, should_close = provider._get_db_session()
        assert session == mock_session
        assert should_close is True

    def test_get_or_create_stock_existing(self, provider, db_session, sample_stock):
        """Test getting an existing stock."""
        stock = provider._get_or_create_stock(db_session, "AAPL")
        assert stock.stock_id == sample_stock.stock_id
        assert stock.ticker_symbol == "AAPL"

    def test_get_or_create_stock_new(self, provider, db_session, mock_yfinance):
        """Test creating a new stock."""
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
        stock = provider._get_or_create_stock(db_session, "GOOGL")
        assert stock.ticker_symbol == "GOOGL"
        # The mock returns "Apple Inc." for all tickers since we use a single mock
        assert stock.company_name == "Apple Inc."
        assert stock.sector == "Technology"

        # Verify it was saved
        found = db_session.query(Stock).filter_by(ticker_symbol="GOOGL").first()
        assert found is not None

    def test_get_cached_price_data(
        self, provider, db_session, sample_stock, sample_price_data, monkeypatch
    ):
        """Test retrieving cached price data."""

        # Mock SessionLocal to return our test session
        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.SessionLocal", lambda: db_session
        )

        df = provider._get_cached_price_data(
            db_session, "AAPL", "2024-01-01", "2024-01-05"
        )

        assert df is not None
        assert not df.empty
        assert len(df) == 5

    def test_get_cached_price_data_partial_range(
        self, provider, db_session, sample_stock, sample_price_data
    ):
        """Test retrieving cached data for partial range."""
        df = provider._get_cached_price_data(
            db_session, "AAPL", "2024-01-02", "2024-01-04"
        )

        assert df is not None
        assert not df.empty
        assert len(df) == 3

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

        # Mock SessionLocal to return our test session
        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.SessionLocal", lambda: db_session
        )

        df = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-05", use_cache=True)

        assert not df.empty
        assert len(df) == 5

    def test_get_stock_data_without_cache(self, provider, mock_yfinance):
        """Test getting stock data without cache."""
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
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
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
        # Mock the session getter - must return (session, should_close) tuple
        monkeypatch.setattr(provider, "_get_db_session", lambda: (db_session, False))

        df = provider.get_stock_data("TSLA", "2024-01-01", "2024-01-05", use_cache=True)

        assert not df.empty
        assert len(df) == 5
        # Data should come from mock pool
        assert df["Close"].iloc[0] == 152.0

    def test_get_stock_data_non_daily_interval(self, provider, mock_yfinance):
        """Test that non-daily intervals bypass cache."""
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
        df = provider.get_stock_data("AAPL", interval="1wk", period="1mo")

        assert not df.empty
        # Should call yfinance pool directly (provider uses _yf_pool.get_history)
        mock_yfinance._mock_pool.get_history.assert_called_with(
            symbol="AAPL", start=None, end=None, period="1mo", interval="1wk"
        )


class TestMaverickRecommendations:
    """Test maverick screening recommendation methods."""

    @pytest.fixture
    def sample_maverick_stocks(self, db_session):
        """Create sample maverick stocks with corresponding Stock entries."""
        # Create Stock entries first for FK references
        stock_entries = []
        for i in range(3):
            s = Stock(
                ticker_symbol=f"STOCK{i}",
                company_name=f"Stock {i} Inc.",
                sector="Technology",
            )
            stock_entries.append(s)
        db_session.add_all(stock_entries)
        db_session.commit()

        stocks = []
        for i in range(3):
            stock = MaverickStocks(
                id=i + 1,
                stock_id=stock_entries[i].stock_id,
                close_price=100.0 + i * 10,
                volume=1000000,
                momentum_score=95.0 - i * 5,
                adr_pct=3.0 + i * 0.5,
                pattern_type="Cup&Handle" if i == 0 else "Base",
                squeeze_status="active" if i < 2 else "neutral",
                consolidation_status="yes" if i == 0 else "no",
                entry_signal=f"{102.0 + i * 10}",
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
        monkeypatch.setattr(provider, "_get_db_session", lambda: (db_session, False))

        recommendations = provider.get_maverick_recommendations(limit=2)

        assert len(recommendations) == 2
        assert recommendations[0]["ticker"] == "STOCK0"
        assert recommendations[0]["combined_score"] == 95
        assert recommendations[0]["recommendation_type"] == "maverick_bullish"
        assert "reason" in recommendations[0]
        assert "Exceptional combined score" in recommendations[0]["reason"]

    def test_get_maverick_recommendations_with_min_score(
        self, provider, db_session, sample_maverick_stocks, monkeypatch
    ):
        """Test getting maverick recommendations with minimum score filter."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: (db_session, False))

        recommendations = provider.get_maverick_recommendations(limit=10, min_score=90)

        assert len(recommendations) == 2  # Only STOCK0 and STOCK1 have score >= 90
        assert all(rec["combined_score"] >= 90 for rec in recommendations)

    @pytest.fixture
    def sample_bear_stocks(self, db_session):
        """Create sample bear stocks with corresponding Stock entries."""
        # Create Stock entries first for FK references
        bear_stock_entries = []
        for i in range(3):
            s = Stock(
                ticker_symbol=f"BEAR{i}",
                company_name=f"Bear {i} Inc.",
                sector="Technology",
            )
            bear_stock_entries.append(s)
        db_session.add_all(bear_stock_entries)
        db_session.commit()

        stocks = []
        for i in range(3):
            stock = MaverickBearStocks(
                id=i + 1,
                stock_id=bear_stock_entries[i].stock_id,
                close_price=50.0 - i * 5,
                volume=500000,
                momentum_score=Decimal(f"{30 - i * 5}.0"),
                rsi_14=Decimal(f"{28 - i * 3}.0"),
                macd=Decimal(f"{-0.5 - i * 0.1}"),
                adr_pct=Decimal(f"{4 + i * 0.5}"),
                atr_contraction=i < 2,
                big_down_vol=i == 0,
                score=90 - i * 5,
                squeeze_status="red" if i < 2 else "neutral",
            )
            stocks.append(stock)

        db_session.add_all(stocks)
        db_session.commit()
        return stocks

    def test_get_maverick_bear_recommendations(
        self, provider, db_session, sample_bear_stocks, monkeypatch
    ):
        """Test getting bear recommendations."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: (db_session, False))

        recommendations = provider.get_maverick_bear_recommendations(limit=2)

        assert len(recommendations) == 2
        assert recommendations[0]["ticker"] == "BEAR0"
        assert recommendations[0]["score"] == 90
        assert recommendations[0]["recommendation_type"] == "maverick_bearish"
        assert "reason" in recommendations[0]
        assert "Exceptional bear score" in recommendations[0]["reason"]

    @pytest.fixture
    def sample_trending_stocks(self, db_session):
        """Create sample trending stocks with corresponding Stock entries."""
        # Create Stock entries first for FK references
        trend_stock_entries = []
        for i in range(3):
            s = Stock(
                ticker_symbol=f"MNRV{i}",
                company_name=f"Minerva {i} Inc.",
                sector="Technology",
            )
            trend_stock_entries.append(s)
        db_session.add_all(trend_stock_entries)
        db_session.commit()

        stocks = []
        for i in range(3):
            stock = SupplyDemandBreakoutStocks(
                id=i + 1,
                stock_id=trend_stock_entries[i].stock_id,
                close_price=200.0 + i * 10,
                volume=2000000,
                ema_21=195.0 + i * 9,
                sma_50=190.0 + i * 8,
                sma_150=185.0 + i * 7,
                sma_200=180.0 + i * 6,
                momentum_score=92.0 - i * 2,
                adr_pct=2.8 + i * 0.2,
                pattern_type="Base" if i == 0 else "Flag",
                squeeze_status="neutral",
                consolidation_status="yes" if i < 2 else "no",
                entry_signal=f"{202.0 + i * 10}",
            )
            stocks.append(stock)

        db_session.add_all(stocks)
        db_session.commit()
        return stocks

    def test_get_trending_recommendations(
        self, provider, db_session, sample_trending_stocks, monkeypatch
    ):
        """Test getting supply/demand breakout recommendations."""
        monkeypatch.setattr(provider, "_get_db_session", lambda: (db_session, False))

        recommendations = provider.get_supply_demand_breakout_recommendations(limit=2)

        assert len(recommendations) == 2
        assert recommendations[0]["ticker"] == "MNRV0"
        assert recommendations[0]["momentum_score"] == 92.0
        assert recommendations[0]["recommendation_type"] == "supply_demand_breakout"
        assert "reason" in recommendations[0]
        assert "Supply/demand breakout" in recommendations[0]["reason"]

    def test_get_all_screening_recommendations(self, provider, monkeypatch):
        """Test getting all screening recommendations."""
        mock_results = {
            "maverick_stocks": [
                {"ticker": "AAPL", "combined_score": 95, "momentum_score": 90}
            ],
            "maverick_bear_stocks": [
                {"ticker": "BEAR", "score": 88, "momentum_score": 25}
            ],
            "supply_demand_breakouts": [{"ticker": "MSFT", "momentum_score": 91}],
        }

        monkeypatch.setattr(
            "maverick_mcp.providers.stock_data.get_latest_maverick_screening",
            lambda: mock_results,
        )

        results = provider.get_all_screening_recommendations()

        assert "maverick_stocks" in results
        assert "maverick_bear_stocks" in results
        assert "supply_demand_breakouts" in results

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

        assert (
            results["supply_demand_breakouts"][0]["recommendation_type"]
            == "supply_demand_breakout"
        )
        assert "reason" in results["supply_demand_breakouts"][0]


class TestBackwardCompatibility:
    """Test backward compatibility with original StockDataProvider."""

    def test_get_stock_info(self, provider, mock_yfinance):
        """Test get_stock_info method."""
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
        info = provider.get_stock_info("AAPL")
        assert info["longName"] == "Apple Inc."
        assert info["sector"] == "Technology"

    def test_get_realtime_data(self, provider, mock_yfinance):
        """Test get_realtime_data method."""
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
        data = provider.get_realtime_data("AAPL")

        assert data is not None
        assert data["symbol"] == "AAPL"
        # Price comes from YFinancePool.get_history mock (last Close value = 156.0)
        assert data["price"] == 156.0
        assert data["change"] == 5.0  # 156 - 151 (previousClose)
        assert data["change_percent"] == pytest.approx(3.31, rel=0.01)

    def test_get_all_realtime_data(self, provider, mock_yfinance):
        """Test get_all_realtime_data method."""
        # Patch the provider's pool to use the mocked pool
        provider._yf_pool = mock_yfinance._mock_pool
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

    @patch("yfinance.Ticker")
    def test_get_news(self, mock_ticker_class, provider):
        """Test get_news method."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.news = [
            {
                "title": "Apple News 1",
                "publisher": "Reuters",
                "link": "https://example.com/1",
                "providerPublishTime": 1704134400,  # 2024-01-01 timestamp
                "type": "STORY",
            }
        ]

        df = provider.get_news("AAPL", limit=5)

        assert not df.empty
        assert len(df) == 1
        assert df["title"].iloc[0] == "Apple News 1"
        assert isinstance(df["providerPublishTime"].iloc[0], pd.Timestamp)

    @patch("yfinance.Ticker")
    def test_get_earnings(self, mock_ticker_class, provider):
        """Test get_earnings method."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.earnings = pd.DataFrame()
        mock_ticker.earnings_dates = pd.DataFrame()
        mock_ticker.earnings_trend = {}

        result = provider.get_earnings("AAPL")

        assert "earnings" in result
        assert "earnings_dates" in result
        assert "earnings_trend" in result

    @patch("yfinance.Ticker")
    def test_get_recommendations(self, mock_ticker_class, provider):
        """Test get_recommendations method."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.recommendations = pd.DataFrame(
            columns=["firm", "toGrade", "fromGrade", "action"]
        )

        df = provider.get_recommendations("AAPL")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["firm", "toGrade", "fromGrade", "action"]

    @patch("yfinance.Ticker")
    def test_is_etf(self, mock_ticker_class, provider):
        """Test is_etf method."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Test regular stock
        mock_ticker.info = {"quoteType": "EQUITY", "longName": "Apple Inc."}
        assert provider.is_etf("AAPL") is False

        # Test ETF
        mock_ticker.info = {"quoteType": "ETF"}
        assert provider.is_etf("SPY") is True

        # Test by symbol pattern
        assert provider.is_etf("QQQ") is True


class TestErrorHandling:
    """Test error handling in the enhanced provider."""

    def test_get_stock_data_error_handling(self, provider, mock_yfinance, monkeypatch):
        """Test error handling in get_stock_data."""
        # Patch the provider's pool to use the mocked pool, then make it raise
        provider._yf_pool = mock_yfinance._mock_pool
        mock_yfinance._mock_pool.get_history.return_value = pd.DataFrame()

        # Mock the database session to ensure smart cache also fails
        def mock_get_db_session():
            raise Exception("Database error")

        monkeypatch.setattr(provider, "_get_db_session", mock_get_db_session)

        # Now the provider should return empty DataFrame since yfinance returns empty
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
            "maverick_mcp.providers.stock_data.bulk_insert_price_data",
            mock_bulk_insert,
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

        monkeypatch.setattr(provider, "_get_db_session", lambda: (mock_session, True))

        recommendations = provider.get_maverick_recommendations()
        assert recommendations == []
