"""
SQLAlchemy models for MaverickMCP.

This module defines database models for financial data storage and analysis,
including PriceCache and Maverick screening models.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    inspect,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from maverick_mcp.config.settings import get_settings
from maverick_mcp.database.base import Base

# Set up logging
logger = logging.getLogger("maverick_mcp.data.models")
settings = get_settings()


# Helper function to get the right integer type for autoincrement primary keys
def get_primary_key_type():
    """Get the appropriate primary key type based on database backend."""
    # SQLite works better with INTEGER for autoincrement, PostgreSQL can use BIGINT
    if "sqlite" in DATABASE_URL:
        return Integer
    else:
        return BigInteger


# Database connection setup
# Try multiple possible environment variable names
# Use SQLite in-memory for GitHub Actions or test environments
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    DATABASE_URL = "sqlite:///:memory:"
else:
    DATABASE_URL = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or "sqlite:///maverick_mcp.db"  # Default to SQLite
    )

# Database configuration from settings
DB_POOL_SIZE = settings.db.pool_size
DB_MAX_OVERFLOW = settings.db.pool_max_overflow
DB_POOL_TIMEOUT = settings.db.pool_timeout
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
DB_POOL_PRE_PING = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"
DB_USE_POOLING = os.getenv("DB_USE_POOLING", "true").lower() == "true"

# Log the connection string (without password) for debugging
if DATABASE_URL:
    # Mask password in URL for logging
    masked_url = DATABASE_URL
    if "@" in DATABASE_URL and "://" in DATABASE_URL:
        parts = DATABASE_URL.split("://", 1)
        if len(parts) == 2 and "@" in parts[1]:
            user_pass, host_db = parts[1].split("@", 1)
            if ":" in user_pass:
                user, _ = user_pass.split(":", 1)
                masked_url = f"{parts[0]}://{user}:****@{host_db}"
    logger.info(f"Using database URL: {masked_url}")
    logger.info(f"Connection pooling: {'ENABLED' if DB_USE_POOLING else 'DISABLED'}")
    if DB_USE_POOLING:
        logger.info(
            f"Pool config: size={DB_POOL_SIZE}, max_overflow={DB_MAX_OVERFLOW}, "
            f"timeout={DB_POOL_TIMEOUT}s, recycle={DB_POOL_RECYCLE}s"
        )

# Create engine with configurable connection pooling
if DB_USE_POOLING:
    # Prepare connection arguments based on database type
    if "postgresql" in DATABASE_URL:
        # PostgreSQL-specific connection args
        sync_connect_args = {
            "connect_timeout": 10,
            "application_name": "maverick_mcp",
            "options": f"-c statement_timeout={settings.db.statement_timeout}",
        }
    elif "sqlite" in DATABASE_URL:
        # SQLite-specific args - no SSL parameters
        sync_connect_args = {"check_same_thread": False}
    else:
        # Default - no connection args
        sync_connect_args = {}

    # Use QueuePool for production environments
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_recycle=DB_POOL_RECYCLE,
        pool_pre_ping=DB_POOL_PRE_PING,
        echo=DB_ECHO,
        connect_args=sync_connect_args,
    )
else:
    # Prepare minimal connection arguments for NullPool
    if "sqlite" in DATABASE_URL:
        sync_connect_args = {"check_same_thread": False}
    else:
        sync_connect_args = {}

    # Use NullPool for serverless/development environments
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        echo=DB_ECHO,
        connect_args=sync_connect_args,
    )

# Create session factory
_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_schema_lock = threading.Lock()
_schema_initialized = False


def ensure_database_schema(force: bool = False) -> bool:
    """Ensure the database schema exists for the configured engine.

    Args:
        force: When ``True`` the schema will be (re)created even if it appears
            to exist already.

    Returns:
        ``True`` if the schema creation routine executed, ``False`` otherwise.
    """

    global _schema_initialized

    # Fast path: skip inspection once the schema has been verified unless the
    # caller explicitly requests a forced refresh.
    if not force and _schema_initialized:
        return False

    with _schema_lock:
        if not force and _schema_initialized:
            return False

        try:
            inspector = inspect(engine)
            existing_tables = set(inspector.get_table_names())
        except SQLAlchemyError as exc:  # pragma: no cover - safety net
            logger.warning(
                "Unable to inspect database schema; attempting to create tables anyway",
                exc_info=exc,
            )
            existing_tables = set()

        defined_tables = set(Base.metadata.tables.keys())
        missing_tables = defined_tables - existing_tables

        should_create = force or bool(missing_tables)
        if should_create:
            if missing_tables:
                logger.info(
                    "Creating missing database tables: %s",
                    ", ".join(sorted(missing_tables)),
                )
            else:
                logger.info("Ensuring database schema is up to date")

            Base.metadata.create_all(bind=engine)
            _schema_initialized = True
            return True

        _schema_initialized = True
        return False


class _SessionFactoryWrapper:
    """Session factory that ensures the schema exists before creating sessions."""

    def __init__(self, factory: sessionmaker):
        self._factory = factory

    def __call__(self, *args, **kwargs):
        ensure_database_schema()
        return self._factory(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._factory, name)


SessionLocal = _SessionFactoryWrapper(_session_factory)

# Create async engine - cached globally for reuse
_async_engine = None
_async_session_factory = None


def _get_async_engine():
    """Get or create the async engine singleton."""
    global _async_engine
    if _async_engine is None:
        # Convert sync URL to async URL
        if DATABASE_URL.startswith("sqlite://"):
            async_url = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
        else:
            async_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

        # Create async engine - don't specify poolclass for async engines
        # SQLAlchemy will use the appropriate async pool automatically
        if DB_USE_POOLING:
            # Prepare connection arguments based on database type
            if "postgresql" in async_url:
                # PostgreSQL-specific connection args
                async_connect_args = {
                    "server_settings": {
                        "application_name": "maverick_mcp_async",
                        "statement_timeout": str(settings.db.statement_timeout),
                    }
                }
            elif "sqlite" in async_url:
                # SQLite-specific args - no SSL parameters
                async_connect_args = {"check_same_thread": False}
            else:
                # Default - no connection args
                async_connect_args = {}

            _async_engine = create_async_engine(
                async_url,
                # Don't specify poolclass - let SQLAlchemy choose the async pool
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_timeout=DB_POOL_TIMEOUT,
                pool_recycle=DB_POOL_RECYCLE,
                pool_pre_ping=DB_POOL_PRE_PING,
                echo=DB_ECHO,
                connect_args=async_connect_args,
            )
        else:
            # Prepare minimal connection arguments for NullPool
            if "sqlite" in async_url:
                async_connect_args = {"check_same_thread": False}
            else:
                async_connect_args = {}

            _async_engine = create_async_engine(
                async_url,
                poolclass=NullPool,
                echo=DB_ECHO,
                connect_args=async_connect_args,
            )
        logger.info("Created async database engine")
    return _async_engine


def _get_async_session_factory():
    """Get or create the async session factory singleton."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = _get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info("Created async session factory")
    return _async_session_factory


def get_db():
    """Get database session."""
    ensure_database_schema()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Async database support - imports moved to top of file


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session using the cached engine."""
    # Get the cached session factory
    async_session_factory = _get_async_session_factory()

    # Create and yield a session
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def close_async_db_connections():
    """Close the async database engine and cleanup connections."""
    global _async_engine, _async_session_factory
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("Closed async database engine")


def init_db():
    """Initialize database by creating all tables."""

    ensure_database_schema(force=True)


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class Stock(Base, TimestampMixin):
    """Stock model for storing basic stock information."""

    __tablename__ = "mcp_stocks"

    stock_id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    ticker_symbol = Column(String(10), unique=True, nullable=False, index=True)
    company_name = Column(String(255))
    description = Column(Text)
    sector = Column(String(100))
    industry = Column(String(100))
    exchange = Column(String(50))
    country = Column(String(50))
    currency = Column(String(3))
    isin = Column(String(12))

    # Additional stock metadata
    market_cap = Column(BigInteger)
    shares_outstanding = Column(BigInteger)
    is_etf = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True, index=True)

    # Relationships
    price_caches = relationship(
        "PriceCache",
        back_populates="stock",
        cascade="all, delete-orphan",
        lazy="selectin",  # Eager load price caches to prevent N+1 queries
    )
    maverick_stocks = relationship(
        "MaverickStocks", back_populates="stock", cascade="all, delete-orphan"
    )
    maverick_bear_stocks = relationship(
        "MaverickBearStocks", back_populates="stock", cascade="all, delete-orphan"
    )
    supply_demand_stocks = relationship(
        "SupplyDemandBreakoutStocks",
        back_populates="stock",
        cascade="all, delete-orphan",
    )
    technical_cache = relationship(
        "TechnicalCache", back_populates="stock", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Stock(ticker={self.ticker_symbol}, name={self.company_name})>"

    @classmethod
    def get_or_create(cls, session: Session, ticker_symbol: str, **kwargs) -> Stock:
        """Get existing stock or create new one."""
        stock = (
            session.query(cls).filter_by(ticker_symbol=ticker_symbol.upper()).first()
        )
        if not stock:
            stock = cls(ticker_symbol=ticker_symbol.upper(), **kwargs)
            session.add(stock)
            session.commit()
        return stock


class PriceCache(Base, TimestampMixin):
    """Cache for historical stock price data."""

    __tablename__ = "mcp_price_cache"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="mcp_price_cache_stock_date_unique"),
        Index("mcp_price_cache_stock_id_date_idx", "stock_id", "date"),
        Index("mcp_price_cache_ticker_date_idx", "stock_id", "date"),
    )

    price_cache_id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    stock_id = Column(Uuid, ForeignKey("mcp_stocks.stock_id"), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Numeric(12, 4))
    high_price = Column(Numeric(12, 4))
    low_price = Column(Numeric(12, 4))
    close_price = Column(Numeric(12, 4))
    volume = Column(BigInteger)

    # Relationships
    stock = relationship(
        "Stock", back_populates="price_caches", lazy="joined"
    )  # Eager load stock info

    def __repr__(self):
        return f"<PriceCache(stock_id={self.stock_id}, date={self.date}, close={self.close_price})>"

    @classmethod
    def get_price_data(
        cls,
        session: Session,
        ticker_symbol: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a pandas DataFrame of price data for the specified symbol and date range.

        Args:
            session: Database session
            ticker_symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        if not end_date:
            end_date = datetime.now(UTC).strftime("%Y-%m-%d")

        # Query with join to get ticker symbol
        query = (
            session.query(
                cls.date,
                cls.open_price.label("open"),
                cls.high_price.label("high"),
                cls.low_price.label("low"),
                cls.close_price.label("close"),
                cls.volume,
            )
            .join(Stock)
            .filter(
                Stock.ticker_symbol == ticker_symbol.upper(),
                cls.date >= pd.to_datetime(start_date).date(),
                cls.date <= pd.to_datetime(end_date).date(),
            )
            .order_by(cls.date)
        )

        # Convert to DataFrame
        df = pd.DataFrame(query.all())

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # Convert decimal types to float
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)

            df["volume"] = df["volume"].astype(int)
            df["symbol"] = ticker_symbol.upper()

        return df


class MaverickStocks(Base, TimestampMixin):
    """Maverick stocks screening results - self-contained model."""

    __tablename__ = "mcp_maverick_stocks"
    __table_args__ = (
        Index("mcp_maverick_stocks_combined_score_idx", "combined_score"),
        Index(
            "mcp_maverick_stocks_momentum_score_idx", "momentum_score"
        ),  # formerly rs_rating_idx
        Index("mcp_maverick_stocks_date_analyzed_idx", "date_analyzed"),
        Index("mcp_maverick_stocks_stock_date_idx", "stock_id", "date_analyzed"),
    )

    id = Column(get_primary_key_type(), primary_key=True, autoincrement=True)
    stock_id = Column(
        Uuid,
        ForeignKey("mcp_stocks.stock_id"),
        nullable=False,
        index=True,
    )
    date_analyzed = Column(
        Date, nullable=False, default=lambda: datetime.now(UTC).date()
    )
    # OHLCV Data
    open_price = Column(Numeric(12, 4), default=0)
    high_price = Column(Numeric(12, 4), default=0)
    low_price = Column(Numeric(12, 4), default=0)
    close_price = Column(Numeric(12, 4), default=0)
    volume = Column(BigInteger, default=0)

    # Technical Indicators
    ema_21 = Column(Numeric(12, 4), default=0)
    sma_50 = Column(Numeric(12, 4), default=0)
    sma_150 = Column(Numeric(12, 4), default=0)
    sma_200 = Column(Numeric(12, 4), default=0)
    momentum_score = Column(Numeric(5, 2), default=0)  # formerly rs_rating
    avg_vol_30d = Column(Numeric(15, 2), default=0)
    adr_pct = Column(Numeric(5, 2), default=0)
    atr = Column(Numeric(12, 4), default=0)

    # Pattern Analysis
    pattern_type = Column(String(50))  # 'pat' field
    squeeze_status = Column(String(50))  # 'sqz' field
    consolidation_status = Column(String(50))  # formerly vcp_status, 'vcp' field
    entry_signal = Column(String(50))  # 'entry' field

    # Scoring
    compression_score = Column(Integer, default=0)
    pattern_detected = Column(Integer, default=0)
    combined_score = Column(Integer, default=0)

    # Relationships
    stock = relationship("Stock", back_populates="maverick_stocks")

    def __repr__(self):
        return f"<MaverickStock(stock_id={self.stock_id}, close={self.close_price}, score={self.combined_score})>"

    @classmethod
    def get_top_stocks(
        cls, session: Session, limit: int = 20
    ) -> Sequence[MaverickStocks]:
        """Get top maverick stocks by combined score."""
        return (
            session.query(cls)
            .join(Stock)
            .order_by(cls.combined_score.desc())
            .limit(limit)
            .all()
        )

    @classmethod
    def get_latest_analysis(
        cls, session: Session, days_back: int = 1
    ) -> Sequence[MaverickStocks]:
        """Get latest maverick analysis within specified days."""
        cutoff_date = datetime.now(UTC).date() - timedelta(days=days_back)
        return (
            session.query(cls)
            .join(Stock)
            .filter(cls.date_analyzed >= cutoff_date)
            .order_by(cls.combined_score.desc())
            .all()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stock_id": str(self.stock_id),
            "ticker": self.stock.ticker_symbol if self.stock else None,
            "date_analyzed": self.date_analyzed.isoformat()
            if self.date_analyzed
            else None,
            "close": float(self.close_price) if self.close_price else 0,
            "volume": self.volume,
            "momentum_score": float(self.momentum_score)
            if self.momentum_score
            else 0,  # formerly rs_rating
            "adr_pct": float(self.adr_pct) if self.adr_pct else 0,
            "pattern": self.pattern_type,
            "squeeze": self.squeeze_status,
            "consolidation": self.consolidation_status,  # formerly vcp
            "entry": self.entry_signal,
            "combined_score": self.combined_score,
            "compression_score": self.compression_score,
            "pattern_detected": self.pattern_detected,
            "ema_21": float(self.ema_21) if self.ema_21 else 0,
            "sma_50": float(self.sma_50) if self.sma_50 else 0,
            "sma_150": float(self.sma_150) if self.sma_150 else 0,
            "sma_200": float(self.sma_200) if self.sma_200 else 0,
            "atr": float(self.atr) if self.atr else 0,
            "avg_vol_30d": float(self.avg_vol_30d) if self.avg_vol_30d else 0,
        }


class MaverickBearStocks(Base, TimestampMixin):
    """Maverick bear stocks screening results - self-contained model."""

    __tablename__ = "mcp_maverick_bear_stocks"
    __table_args__ = (
        Index("mcp_maverick_bear_stocks_score_idx", "score"),
        Index(
            "mcp_maverick_bear_stocks_momentum_score_idx", "momentum_score"
        ),  # formerly rs_rating_idx
        Index("mcp_maverick_bear_stocks_date_analyzed_idx", "date_analyzed"),
        Index("mcp_maverick_bear_stocks_stock_date_idx", "stock_id", "date_analyzed"),
    )

    id = Column(get_primary_key_type(), primary_key=True, autoincrement=True)
    stock_id = Column(
        Uuid,
        ForeignKey("mcp_stocks.stock_id"),
        nullable=False,
        index=True,
    )
    date_analyzed = Column(
        Date, nullable=False, default=lambda: datetime.now(UTC).date()
    )

    # OHLCV Data
    open_price = Column(Numeric(12, 4), default=0)
    high_price = Column(Numeric(12, 4), default=0)
    low_price = Column(Numeric(12, 4), default=0)
    close_price = Column(Numeric(12, 4), default=0)
    volume = Column(BigInteger, default=0)

    # Technical Indicators
    momentum_score = Column(Numeric(5, 2), default=0)  # formerly rs_rating
    ema_21 = Column(Numeric(12, 4), default=0)
    sma_50 = Column(Numeric(12, 4), default=0)
    sma_200 = Column(Numeric(12, 4), default=0)
    rsi_14 = Column(Numeric(5, 2), default=0)

    # MACD Indicators
    macd = Column(Numeric(12, 6), default=0)
    macd_signal = Column(Numeric(12, 6), default=0)
    macd_histogram = Column(Numeric(12, 6), default=0)

    # Additional Bear Market Indicators
    dist_days_20 = Column(Integer, default=0)  # Days from 20 SMA
    adr_pct = Column(Numeric(5, 2), default=0)
    atr_contraction = Column(Boolean, default=False)
    atr = Column(Numeric(12, 4), default=0)
    avg_vol_30d = Column(Numeric(15, 2), default=0)
    big_down_vol = Column(Boolean, default=False)

    # Pattern Analysis
    squeeze_status = Column(String(50))  # 'sqz' field
    consolidation_status = Column(String(50))  # formerly vcp_status, 'vcp' field

    # Scoring
    score = Column(Integer, default=0)

    # Relationships
    stock = relationship("Stock", back_populates="maverick_bear_stocks")

    def __repr__(self):
        return f"<MaverickBearStock(stock_id={self.stock_id}, close={self.close_price}, score={self.score})>"

    @classmethod
    def get_top_stocks(
        cls, session: Session, limit: int = 20
    ) -> Sequence[MaverickBearStocks]:
        """Get top maverick bear stocks by score."""
        return (
            session.query(cls).join(Stock).order_by(cls.score.desc()).limit(limit).all()
        )

    @classmethod
    def get_latest_analysis(
        cls, session: Session, days_back: int = 1
    ) -> Sequence[MaverickBearStocks]:
        """Get latest bear analysis within specified days."""
        cutoff_date = datetime.now(UTC).date() - timedelta(days=days_back)
        return (
            session.query(cls)
            .join(Stock)
            .filter(cls.date_analyzed >= cutoff_date)
            .order_by(cls.score.desc())
            .all()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stock_id": str(self.stock_id),
            "ticker": self.stock.ticker_symbol if self.stock else None,
            "date_analyzed": self.date_analyzed.isoformat()
            if self.date_analyzed
            else None,
            "close": float(self.close_price) if self.close_price else 0,
            "volume": self.volume,
            "momentum_score": float(self.momentum_score)
            if self.momentum_score
            else 0,  # formerly rs_rating
            "rsi_14": float(self.rsi_14) if self.rsi_14 else 0,
            "macd": float(self.macd) if self.macd else 0,
            "macd_signal": float(self.macd_signal) if self.macd_signal else 0,
            "macd_histogram": float(self.macd_histogram) if self.macd_histogram else 0,
            "adr_pct": float(self.adr_pct) if self.adr_pct else 0,
            "atr": float(self.atr) if self.atr else 0,
            "atr_contraction": self.atr_contraction,
            "avg_vol_30d": float(self.avg_vol_30d) if self.avg_vol_30d else 0,
            "big_down_vol": self.big_down_vol,
            "score": self.score,
            "squeeze": self.squeeze_status,
            "consolidation": self.consolidation_status,  # formerly vcp
            "ema_21": float(self.ema_21) if self.ema_21 else 0,
            "sma_50": float(self.sma_50) if self.sma_50 else 0,
            "sma_200": float(self.sma_200) if self.sma_200 else 0,
            "dist_days_20": self.dist_days_20,
        }


class SupplyDemandBreakoutStocks(Base, TimestampMixin):
    """Supply/demand breakout stocks screening results - self-contained model.

    This model identifies stocks experiencing accumulation breakouts with strong relative strength,
    indicating a potential shift from supply to demand dominance in the market structure.
    """

    __tablename__ = "mcp_supply_demand_breakouts"
    __table_args__ = (
        Index(
            "mcp_supply_demand_breakouts_momentum_score_idx", "momentum_score"
        ),  # formerly rs_rating_idx
        Index("mcp_supply_demand_breakouts_date_analyzed_idx", "date_analyzed"),
        Index(
            "mcp_supply_demand_breakouts_stock_date_idx", "stock_id", "date_analyzed"
        ),
        Index(
            "mcp_supply_demand_breakouts_ma_filter_idx",
            "close_price",
            "sma_50",
            "sma_150",
            "sma_200",
        ),
    )

    id = Column(get_primary_key_type(), primary_key=True, autoincrement=True)
    stock_id = Column(
        Uuid,
        ForeignKey("mcp_stocks.stock_id"),
        nullable=False,
        index=True,
    )
    date_analyzed = Column(
        Date, nullable=False, default=lambda: datetime.now(UTC).date()
    )

    # OHLCV Data
    open_price = Column(Numeric(12, 4), default=0)
    high_price = Column(Numeric(12, 4), default=0)
    low_price = Column(Numeric(12, 4), default=0)
    close_price = Column(Numeric(12, 4), default=0)
    volume = Column(BigInteger, default=0)

    # Technical Indicators
    ema_21 = Column(Numeric(12, 4), default=0)
    sma_50 = Column(Numeric(12, 4), default=0)
    sma_150 = Column(Numeric(12, 4), default=0)
    sma_200 = Column(Numeric(12, 4), default=0)
    momentum_score = Column(Numeric(5, 2), default=0)  # formerly rs_rating
    avg_volume_30d = Column(Numeric(15, 2), default=0)
    adr_pct = Column(Numeric(5, 2), default=0)
    atr = Column(Numeric(12, 4), default=0)

    # Pattern Analysis
    pattern_type = Column(String(50))  # 'pat' field
    squeeze_status = Column(String(50))  # 'sqz' field
    consolidation_status = Column(String(50))  # formerly vcp_status, 'vcp' field
    entry_signal = Column(String(50))  # 'entry' field

    # Supply/Demand Analysis
    accumulation_rating = Column(Numeric(5, 2), default=0)
    distribution_rating = Column(Numeric(5, 2), default=0)
    breakout_strength = Column(Numeric(5, 2), default=0)

    # Relationships
    stock = relationship("Stock", back_populates="supply_demand_stocks")

    def __repr__(self):
        return f"<SupplyDemandBreakoutStock(stock_id={self.stock_id}, close={self.close_price}, momentum={self.momentum_score})>"  # formerly rs

    @classmethod
    def get_top_stocks(
        cls, session: Session, limit: int = 20
    ) -> Sequence[SupplyDemandBreakoutStocks]:
        """Get top supply/demand breakout stocks by momentum score."""  # formerly relative strength rating
        return (
            session.query(cls)
            .join(Stock)
            .order_by(cls.momentum_score.desc())  # formerly rs_rating
            .limit(limit)
            .all()
        )

    @classmethod
    def get_stocks_above_moving_averages(
        cls, session: Session
    ) -> Sequence[SupplyDemandBreakoutStocks]:
        """Get stocks in demand expansion phase - trading above all major moving averages.

        This identifies stocks with:
        - Price above 50, 150, and 200-day moving averages (demand zone)
        - Upward trending moving averages (accumulation structure)
        - Indicates institutional accumulation and supply absorption
        """
        return (
            session.query(cls)
            .join(Stock)
            .filter(
                cls.close_price > cls.sma_50,
                cls.close_price > cls.sma_150,
                cls.close_price > cls.sma_200,
                cls.sma_50 > cls.sma_150,
                cls.sma_150 > cls.sma_200,
            )
            .order_by(cls.momentum_score.desc())  # formerly rs_rating
            .all()
        )

    @classmethod
    def get_latest_analysis(
        cls, session: Session, days_back: int = 1
    ) -> Sequence[SupplyDemandBreakoutStocks]:
        """Get latest supply/demand analysis within specified days."""
        cutoff_date = datetime.now(UTC).date() - timedelta(days=days_back)
        return (
            session.query(cls)
            .join(Stock)
            .filter(cls.date_analyzed >= cutoff_date)
            .order_by(cls.momentum_score.desc())  # formerly rs_rating
            .all()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stock_id": str(self.stock_id),
            "ticker": self.stock.ticker_symbol if self.stock else None,
            "date_analyzed": self.date_analyzed.isoformat()
            if self.date_analyzed
            else None,
            "close": float(self.close_price) if self.close_price else 0,
            "volume": self.volume,
            "momentum_score": float(self.momentum_score)
            if self.momentum_score
            else 0,  # formerly rs_rating
            "adr_pct": float(self.adr_pct) if self.adr_pct else 0,
            "pattern": self.pattern_type,
            "squeeze": self.squeeze_status,
            "consolidation": self.consolidation_status,  # formerly vcp
            "entry": self.entry_signal,
            "ema_21": float(self.ema_21) if self.ema_21 else 0,
            "sma_50": float(self.sma_50) if self.sma_50 else 0,
            "sma_150": float(self.sma_150) if self.sma_150 else 0,
            "sma_200": float(self.sma_200) if self.sma_200 else 0,
            "atr": float(self.atr) if self.atr else 0,
            "avg_volume_30d": float(self.avg_volume_30d) if self.avg_volume_30d else 0,
            "accumulation_rating": float(self.accumulation_rating)
            if self.accumulation_rating
            else 0,
            "distribution_rating": float(self.distribution_rating)
            if self.distribution_rating
            else 0,
            "breakout_strength": float(self.breakout_strength)
            if self.breakout_strength
            else 0,
        }


class TechnicalCache(Base, TimestampMixin):
    """Cache for calculated technical indicators."""

    __tablename__ = "mcp_technical_cache"
    __table_args__ = (
        UniqueConstraint(
            "stock_id",
            "date",
            "indicator_type",
            name="mcp_technical_cache_stock_date_indicator_unique",
        ),
        Index("mcp_technical_cache_stock_date_idx", "stock_id", "date"),
        Index("mcp_technical_cache_indicator_idx", "indicator_type"),
        Index("mcp_technical_cache_date_idx", "date"),
    )

    id = Column(get_primary_key_type(), primary_key=True, autoincrement=True)
    stock_id = Column(Uuid, ForeignKey("mcp_stocks.stock_id"), nullable=False)
    date = Column(Date, nullable=False)
    indicator_type = Column(
        String(50), nullable=False
    )  # 'SMA_20', 'EMA_21', 'RSI_14', etc.

    # Flexible indicator values
    value = Column(Numeric(20, 8))  # Primary indicator value
    value_2 = Column(Numeric(20, 8))  # Secondary value (e.g., MACD signal)
    value_3 = Column(Numeric(20, 8))  # Tertiary value (e.g., MACD histogram)

    # Text values for complex indicators
    meta_data = Column(Text)  # JSON string for additional metadata

    # Calculation parameters
    period = Column(Integer)  # Period used (20 for SMA_20, etc.)
    parameters = Column(Text)  # JSON string for additional parameters

    # Relationships
    stock = relationship("Stock", back_populates="technical_cache")

    def __repr__(self):
        return (
            f"<TechnicalCache(stock_id={self.stock_id}, date={self.date}, "
            f"indicator={self.indicator_type}, value={self.value})>"
        )

    @classmethod
    def get_indicator(
        cls,
        session: Session,
        ticker_symbol: str,
        indicator_type: str,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Get technical indicator data for a symbol and date range.

        Args:
            session: Database session
            ticker_symbol: Stock ticker symbol
            indicator_type: Type of indicator (e.g., 'SMA_20', 'RSI_14')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)

        Returns:
            DataFrame with indicator data indexed by date
        """
        if not end_date:
            end_date = datetime.now(UTC).strftime("%Y-%m-%d")

        query = (
            session.query(
                cls.date,
                cls.value,
                cls.value_2,
                cls.value_3,
                cls.meta_data,
                cls.parameters,
            )
            .join(Stock)
            .filter(
                Stock.ticker_symbol == ticker_symbol.upper(),
                cls.indicator_type == indicator_type,
                cls.date >= pd.to_datetime(start_date).date(),
                cls.date <= pd.to_datetime(end_date).date(),
            )
            .order_by(cls.date)
        )

        df = pd.DataFrame(query.all())

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # Convert decimal types to float
            for col in ["value", "value_2", "value_3"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            df["symbol"] = ticker_symbol.upper()
            df["indicator_type"] = indicator_type

        return df

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stock_id": str(self.stock_id),
            "date": self.date.isoformat() if self.date else None,
            "indicator_type": self.indicator_type,
            "value": float(self.value) if self.value else None,
            "value_2": float(self.value_2) if self.value_2 else None,
            "value_3": float(self.value_3) if self.value_3 else None,
            "period": self.period,
            "meta_data": self.meta_data,
            "parameters": self.parameters,
        }


# Backtesting Models


class BacktestResult(Base, TimestampMixin):
    """Main backtest results table with comprehensive metrics."""

    __tablename__ = "mcp_backtest_results"
    __table_args__ = (
        Index("mcp_backtest_results_symbol_idx", "symbol"),
        Index("mcp_backtest_results_strategy_idx", "strategy_type"),
        Index("mcp_backtest_results_date_idx", "backtest_date"),
        Index("mcp_backtest_results_sharpe_idx", "sharpe_ratio"),
        Index("mcp_backtest_results_total_return_idx", "total_return"),
        Index("mcp_backtest_results_symbol_strategy_idx", "symbol", "strategy_type"),
    )

    backtest_id = Column(Uuid, primary_key=True, default=uuid.uuid4)

    # Basic backtest metadata
    symbol = Column(String(10), nullable=False, index=True)
    strategy_type = Column(String(50), nullable=False)
    backtest_date = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Date range and setup
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Numeric(15, 2), default=10000.0)

    # Trading costs and parameters
    fees = Column(Numeric(6, 4), default=0.001)  # 0.1% default
    slippage = Column(Numeric(6, 4), default=0.001)  # 0.1% default

    # Strategy parameters (stored as JSON for flexibility)
    parameters = Column(JSON)

    # Key Performance Metrics
    total_return = Column(Numeric(10, 4))  # Total return percentage
    annualized_return = Column(Numeric(10, 4))  # Annualized return percentage
    sharpe_ratio = Column(Numeric(8, 4))
    sortino_ratio = Column(Numeric(8, 4))
    calmar_ratio = Column(Numeric(8, 4))

    # Risk Metrics
    max_drawdown = Column(Numeric(8, 4))  # Maximum drawdown percentage
    max_drawdown_duration = Column(Integer)  # Days
    volatility = Column(Numeric(8, 4))  # Annualized volatility
    downside_volatility = Column(Numeric(8, 4))  # Downside deviation

    # Trade Statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 4))  # Win rate percentage

    # P&L Statistics
    profit_factor = Column(Numeric(8, 4))  # Gross profit / Gross loss
    average_win = Column(Numeric(12, 4))
    average_loss = Column(Numeric(12, 4))
    largest_win = Column(Numeric(12, 4))
    largest_loss = Column(Numeric(12, 4))

    # Portfolio Value Metrics
    final_portfolio_value = Column(Numeric(15, 2))
    peak_portfolio_value = Column(Numeric(15, 2))

    # Additional Analysis
    beta = Column(Numeric(8, 4))  # Market beta
    alpha = Column(Numeric(8, 4))  # Alpha vs market

    # Time series data (stored as JSON for efficient queries)
    equity_curve = Column(JSON)  # Daily portfolio values
    drawdown_series = Column(JSON)  # Daily drawdown values

    # Execution metadata
    execution_time_seconds = Column(Numeric(8, 3))  # How long the backtest took
    data_points = Column(Integer)  # Number of data points used

    # Status and notes
    status = Column(String(20), default="completed")  # completed, failed, in_progress
    error_message = Column(Text)  # Error details if status = failed
    notes = Column(Text)  # User notes

    # Relationships
    trades = relationship(
        "BacktestTrade",
        back_populates="backtest_result",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    optimization_results = relationship(
        "OptimizationResult",
        back_populates="backtest_result",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return (
            f"<BacktestResult(id={self.backtest_id}, symbol={self.symbol}, "
            f"strategy={self.strategy_type}, return={self.total_return})>"
        )

    @classmethod
    def get_by_symbol_and_strategy(
        cls, session: Session, symbol: str, strategy_type: str, limit: int = 10
    ) -> Sequence[BacktestResult]:
        """Get recent backtests for a specific symbol and strategy."""
        return (
            session.query(cls)
            .filter(cls.symbol == symbol.upper(), cls.strategy_type == strategy_type)
            .order_by(cls.backtest_date.desc())
            .limit(limit)
            .all()
        )

    @classmethod
    def get_best_performing(
        cls, session: Session, metric: str = "sharpe_ratio", limit: int = 20
    ) -> Sequence[BacktestResult]:
        """Get best performing backtests by specified metric."""
        metric_column = getattr(cls, metric, cls.sharpe_ratio)
        return (
            session.query(cls)
            .filter(cls.status == "completed")
            .order_by(metric_column.desc())
            .limit(limit)
            .all()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "backtest_id": str(self.backtest_id),
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "backtest_date": self.backtest_date.isoformat()
            if self.backtest_date
            else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_capital": float(self.initial_capital)
            if self.initial_capital
            else 0,
            "total_return": float(self.total_return) if self.total_return else 0,
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else 0,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown else 0,
            "win_rate": float(self.win_rate) if self.win_rate else 0,
            "total_trades": self.total_trades,
            "parameters": self.parameters,
            "status": self.status,
        }


class BacktestTrade(Base, TimestampMixin):
    """Individual trade records from backtests."""

    __tablename__ = "mcp_backtest_trades"
    __table_args__ = (
        Index("mcp_backtest_trades_backtest_idx", "backtest_id"),
        Index("mcp_backtest_trades_entry_date_idx", "entry_date"),
        Index("mcp_backtest_trades_exit_date_idx", "exit_date"),
        Index("mcp_backtest_trades_pnl_idx", "pnl"),
        Index("mcp_backtest_trades_backtest_entry_idx", "backtest_id", "entry_date"),
    )

    trade_id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    backtest_id = Column(
        Uuid, ForeignKey("mcp_backtest_results.backtest_id"), nullable=False
    )

    # Trade identification
    trade_number = Column(
        Integer, nullable=False
    )  # Sequential trade number in backtest

    # Entry details
    entry_date = Column(Date, nullable=False)
    entry_price = Column(Numeric(12, 4), nullable=False)
    entry_time = Column(DateTime(timezone=True))  # For intraday backtests

    # Exit details
    exit_date = Column(Date)
    exit_price = Column(Numeric(12, 4))
    exit_time = Column(DateTime(timezone=True))

    # Position details
    position_size = Column(Numeric(15, 2))  # Number of shares/units
    direction = Column(String(5), nullable=False)  # 'long' or 'short'

    # P&L and performance
    pnl = Column(Numeric(12, 4))  # Profit/Loss in currency
    pnl_percent = Column(Numeric(8, 4))  # P&L as percentage

    # Risk metrics for this trade
    mae = Column(Numeric(8, 4))  # Maximum Adverse Excursion
    mfe = Column(Numeric(8, 4))  # Maximum Favorable Excursion

    # Trade duration
    duration_days = Column(Integer)
    duration_hours = Column(Numeric(8, 2))  # For intraday precision

    # Exit reason and fees
    exit_reason = Column(String(50))  # stop_loss, take_profit, signal, time_exit
    fees_paid = Column(Numeric(10, 4), default=0)
    slippage_cost = Column(Numeric(10, 4), default=0)

    # Relationships
    backtest_result = relationship(
        "BacktestResult", back_populates="trades", lazy="joined"
    )

    def __repr__(self):
        return (
            f"<BacktestTrade(id={self.trade_id}, backtest_id={self.backtest_id}, "
            f"pnl={self.pnl}, duration={self.duration_days}d)>"
        )

    @classmethod
    def get_trades_for_backtest(
        cls, session: Session, backtest_id: str
    ) -> Sequence[BacktestTrade]:
        """Get all trades for a specific backtest."""
        return (
            session.query(cls)
            .filter(cls.backtest_id == backtest_id)
            .order_by(cls.entry_date, cls.trade_number)
            .all()
        )

    @classmethod
    def get_winning_trades(
        cls, session: Session, backtest_id: str
    ) -> Sequence[BacktestTrade]:
        """Get winning trades for a backtest."""
        return (
            session.query(cls)
            .filter(cls.backtest_id == backtest_id, cls.pnl > 0)
            .order_by(cls.pnl.desc())
            .all()
        )

    @classmethod
    def get_losing_trades(
        cls, session: Session, backtest_id: str
    ) -> Sequence[BacktestTrade]:
        """Get losing trades for a backtest."""
        return (
            session.query(cls)
            .filter(cls.backtest_id == backtest_id, cls.pnl < 0)
            .order_by(cls.pnl)
            .all()
        )


class OptimizationResult(Base, TimestampMixin):
    """Parameter optimization results for strategies."""

    __tablename__ = "mcp_optimization_results"
    __table_args__ = (
        Index("mcp_optimization_results_backtest_idx", "backtest_id"),
        Index("mcp_optimization_results_param_set_idx", "parameter_set"),
        Index("mcp_optimization_results_objective_idx", "objective_value"),
    )

    optimization_id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    backtest_id = Column(
        Uuid, ForeignKey("mcp_backtest_results.backtest_id"), nullable=False
    )

    # Optimization metadata
    optimization_date = Column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    parameter_set = Column(Integer, nullable=False)  # Set number in optimization run

    # Parameters tested (JSON for flexibility)
    parameters = Column(JSON, nullable=False)

    # Optimization objective and results
    objective_function = Column(
        String(50)
    )  # sharpe_ratio, total_return, profit_factor, etc.
    objective_value = Column(Numeric(12, 6))  # Value of objective function

    # Key metrics for this parameter set
    total_return = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(8, 4))
    max_drawdown = Column(Numeric(8, 4))
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(8, 4))
    total_trades = Column(Integer)

    # Ranking within optimization
    rank = Column(Integer)  # 1 = best, 2 = second best, etc.

    # Statistical significance
    is_statistically_significant = Column(Boolean, default=False)
    p_value = Column(Numeric(8, 6))  # Statistical significance test result

    # Relationships
    backtest_result = relationship(
        "BacktestResult", back_populates="optimization_results", lazy="joined"
    )

    def __repr__(self):
        return (
            f"<OptimizationResult(id={self.optimization_id}, "
            f"objective={self.objective_value}, rank={self.rank})>"
        )

    @classmethod
    def get_best_parameters(
        cls, session: Session, backtest_id: str, limit: int = 5
    ) -> Sequence[OptimizationResult]:
        """Get top performing parameter sets for a backtest."""
        return (
            session.query(cls)
            .filter(cls.backtest_id == backtest_id)
            .order_by(cls.rank)
            .limit(limit)
            .all()
        )


class WalkForwardTest(Base, TimestampMixin):
    """Walk-forward validation test results."""

    __tablename__ = "mcp_walk_forward_tests"
    __table_args__ = (
        Index("mcp_walk_forward_tests_parent_idx", "parent_backtest_id"),
        Index("mcp_walk_forward_tests_period_idx", "test_period_start"),
        Index("mcp_walk_forward_tests_performance_idx", "out_of_sample_return"),
    )

    walk_forward_id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    parent_backtest_id = Column(
        Uuid, ForeignKey("mcp_backtest_results.backtest_id"), nullable=False
    )

    # Test configuration
    test_date = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    window_size_months = Column(Integer, nullable=False)  # Training window size
    step_size_months = Column(Integer, nullable=False)  # Step size for walking forward

    # Time periods
    training_start = Column(Date, nullable=False)
    training_end = Column(Date, nullable=False)
    test_period_start = Column(Date, nullable=False)
    test_period_end = Column(Date, nullable=False)

    # Optimization results from training period
    optimal_parameters = Column(JSON)  # Best parameters from training
    training_performance = Column(Numeric(10, 4))  # Training period return

    # Out-of-sample test results
    out_of_sample_return = Column(Numeric(10, 4))
    out_of_sample_sharpe = Column(Numeric(8, 4))
    out_of_sample_drawdown = Column(Numeric(8, 4))
    out_of_sample_trades = Column(Integer)

    # Performance vs training expectations
    performance_ratio = Column(Numeric(8, 4))  # Out-sample return / Training return
    degradation_factor = Column(Numeric(8, 4))  # How much performance degraded

    # Statistical validation
    is_profitable = Column(Boolean)
    is_statistically_significant = Column(Boolean, default=False)

    # Relationships
    parent_backtest = relationship(
        "BacktestResult", foreign_keys=[parent_backtest_id], lazy="joined"
    )

    def __repr__(self):
        return (
            f"<WalkForwardTest(id={self.walk_forward_id}, "
            f"return={self.out_of_sample_return}, ratio={self.performance_ratio})>"
        )

    @classmethod
    def get_walk_forward_results(
        cls, session: Session, parent_backtest_id: str
    ) -> Sequence[WalkForwardTest]:
        """Get all walk-forward test results for a backtest."""
        return (
            session.query(cls)
            .filter(cls.parent_backtest_id == parent_backtest_id)
            .order_by(cls.test_period_start)
            .all()
        )


class BacktestPortfolio(Base, TimestampMixin):
    """Portfolio-level backtests with multiple symbols."""

    __tablename__ = "mcp_backtest_portfolios"
    __table_args__ = (
        Index("mcp_backtest_portfolios_name_idx", "portfolio_name"),
        Index("mcp_backtest_portfolios_date_idx", "backtest_date"),
        Index("mcp_backtest_portfolios_return_idx", "total_return"),
    )

    portfolio_backtest_id = Column(Uuid, primary_key=True, default=uuid.uuid4)

    # Portfolio identification
    portfolio_name = Column(String(100), nullable=False)
    description = Column(Text)

    # Test metadata
    backtest_date = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    # Portfolio composition
    symbols = Column(JSON, nullable=False)  # List of symbols
    weights = Column(JSON)  # Portfolio weights (if not equal weight)
    rebalance_frequency = Column(String(20))  # daily, weekly, monthly, quarterly

    # Portfolio parameters
    initial_capital = Column(Numeric(15, 2), default=100000.0)
    max_positions = Column(Integer)  # Maximum concurrent positions
    position_sizing_method = Column(
        String(50)
    )  # equal_weight, volatility_weighted, etc.

    # Risk management
    portfolio_stop_loss = Column(Numeric(6, 4))  # Portfolio-level stop loss
    max_sector_allocation = Column(Numeric(5, 4))  # Maximum allocation per sector
    correlation_threshold = Column(
        Numeric(5, 4)
    )  # Maximum correlation between holdings

    # Performance metrics (portfolio level)
    total_return = Column(Numeric(10, 4))
    annualized_return = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(8, 4))
    sortino_ratio = Column(Numeric(8, 4))
    max_drawdown = Column(Numeric(8, 4))
    volatility = Column(Numeric(8, 4))

    # Portfolio-specific metrics
    diversification_ratio = Column(Numeric(8, 4))  # Portfolio vol / Weighted avg vol
    concentration_index = Column(Numeric(8, 4))  # Herfindahl index
    turnover_rate = Column(Numeric(8, 4))  # Portfolio turnover

    # Individual component backtests (JSON references)
    component_backtest_ids = Column(JSON)  # List of individual backtest IDs

    # Time series data
    portfolio_equity_curve = Column(JSON)
    portfolio_weights_history = Column(JSON)  # Historical weights over time

    # Status
    status = Column(String(20), default="completed")
    notes = Column(Text)

    def __repr__(self):
        return (
            f"<BacktestPortfolio(id={self.portfolio_backtest_id}, "
            f"name={self.portfolio_name}, return={self.total_return})>"
        )

    @classmethod
    def get_portfolio_backtests(
        cls, session: Session, portfolio_name: str | None = None, limit: int = 10
    ) -> Sequence[BacktestPortfolio]:
        """Get portfolio backtests, optionally filtered by name."""
        query = session.query(cls).order_by(cls.backtest_date.desc())
        if portfolio_name:
            query = query.filter(cls.portfolio_name == portfolio_name)
        return query.limit(limit).all()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "portfolio_backtest_id": str(self.portfolio_backtest_id),
            "portfolio_name": self.portfolio_name,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "total_return": float(self.total_return) if self.total_return else 0,
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else 0,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown else 0,
            "status": self.status,
        }


# Helper functions for working with the models
def bulk_insert_price_data(
    session: Session, ticker_symbol: str, df: pd.DataFrame
) -> int:
    """
    Bulk insert price data from a DataFrame.

    Args:
        session: Database session
        ticker_symbol: Stock ticker symbol
        df: DataFrame with OHLCV data (must have date index)

    Returns:
        Number of records inserted (or would be inserted)
    """
    if df.empty:
        return 0

    # Get or create stock
    stock = Stock.get_or_create(session, ticker_symbol)

    # First, check how many records already exist
    existing_dates = set()
    if hasattr(df.index[0], "date"):
        dates_to_check = [d.date() for d in df.index]
    else:
        dates_to_check = list(df.index)

    existing_query = session.query(PriceCache.date).filter(
        PriceCache.stock_id == stock.stock_id, PriceCache.date.in_(dates_to_check)
    )
    existing_dates = {row[0] for row in existing_query.all()}

    # Prepare data for bulk insert
    records = []
    new_count = 0
    for date_idx, row in df.iterrows():
        # Handle different index types - datetime index vs date index
        if hasattr(date_idx, "date") and callable(date_idx.date):
            date_val = date_idx.date()  # type: ignore[attr-defined]
        elif hasattr(date_idx, "to_pydatetime") and callable(date_idx.to_pydatetime):
            date_val = date_idx.to_pydatetime().date()  # type: ignore[attr-defined]
        else:
            # Assume it's already a date-like object
            date_val = date_idx

        # Skip if already exists
        if date_val in existing_dates:
            continue

        new_count += 1

        # Handle both lowercase and capitalized column names from yfinance
        open_val = row.get("open", row.get("Open", 0))
        high_val = row.get("high", row.get("High", 0))
        low_val = row.get("low", row.get("Low", 0))
        close_val = row.get("close", row.get("Close", 0))
        volume_val = row.get("volume", row.get("Volume", 0))

        # Handle None values
        if volume_val is None:
            volume_val = 0

        records.append(
            {
                "stock_id": stock.stock_id,
                "date": date_val,
                "open_price": Decimal(str(open_val)),
                "high_price": Decimal(str(high_val)),
                "low_price": Decimal(str(low_val)),
                "close_price": Decimal(str(close_val)),
                "volume": int(volume_val),
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
        )

    # Only insert if there are new records
    if records:
        # Use database-specific upsert logic
        if "postgresql" in DATABASE_URL:
            from sqlalchemy.dialects.postgresql import insert

            stmt = insert(PriceCache).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=["stock_id", "date"])
        else:
            # For SQLite, use INSERT OR IGNORE
            from sqlalchemy import insert

            stmt = insert(PriceCache).values(records)
            # SQLite doesn't support on_conflict_do_nothing, use INSERT OR IGNORE
            stmt = stmt.prefix_with("OR IGNORE")

        result = session.execute(stmt)
        session.commit()

        # Log if rowcount differs from expected
        if result.rowcount != new_count:
            logger.warning(
                f"Expected to insert {new_count} records but rowcount was {result.rowcount}"
            )

        return result.rowcount
    else:
        logger.debug(
            f"All {len(df)} records already exist in cache for {ticker_symbol}"
        )
        return 0


def get_latest_maverick_screening(days_back: int = 1) -> dict:
    """Get latest screening results from all maverick tables."""
    with SessionLocal() as session:
        results = {
            "maverick_stocks": [
                stock.to_dict()
                for stock in MaverickStocks.get_latest_analysis(
                    session, days_back=days_back
                )
            ],
            "maverick_bear_stocks": [
                stock.to_dict()
                for stock in MaverickBearStocks.get_latest_analysis(
                    session, days_back=days_back
                )
            ],
            "supply_demand_breakouts": [
                stock.to_dict()
                for stock in SupplyDemandBreakoutStocks.get_latest_analysis(
                    session, days_back=days_back
                )
            ],
        }

    return results


def bulk_insert_screening_data(
    session: Session,
    model_class,
    screening_data: list[dict],
    date_analyzed: date | None = None,
) -> int:
    """
    Bulk insert screening data for any screening model.

    Args:
        session: Database session
        model_class: The screening model class (MaverickStocks, etc.)
        screening_data: List of screening result dictionaries
        date_analyzed: Date of analysis (default: today)

    Returns:
        Number of records inserted
    """
    if not screening_data:
        return 0

    if date_analyzed is None:
        date_analyzed = datetime.now(UTC).date()

    # Remove existing data for this date
    session.query(model_class).filter(
        model_class.date_analyzed == date_analyzed
    ).delete()

    inserted_count = 0
    for data in screening_data:
        # Get or create stock
        ticker = data.get("ticker") or data.get("symbol")
        if not ticker:
            continue

        stock = Stock.get_or_create(session, ticker)

        # Create screening record
        record_data = {
            "stock_id": stock.stock_id,
            "date_analyzed": date_analyzed,
        }

        # Map common fields
        field_mapping = {
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "close": "close_price",
            "pat": "pattern_type",
            "sqz": "squeeze_status",
            "vcp": "consolidation_status",
            "entry": "entry_signal",
        }

        for key, value in data.items():
            if key in ["ticker", "symbol"]:
                continue
            mapped_key = field_mapping.get(key, key)
            if hasattr(model_class, mapped_key):
                record_data[mapped_key] = value

        record = model_class(**record_data)
        session.add(record)
        inserted_count += 1

    session.commit()
    return inserted_count


# Auth models removed for personal use - no multi-user functionality needed

# Initialize tables when module is imported
if __name__ == "__main__":
    logger.info("Creating database tables...")
    init_db()
    logger.info("Database tables created successfully!")
