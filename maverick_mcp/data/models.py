"""
SQLAlchemy models for MaverickMCP.

This module defines database models for financial data storage and analysis,
including PriceCache and Maverick screening models.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import (
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
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from maverick_mcp.config.settings import get_settings
from maverick_mcp.database.base import Base

# Set up logging
logger = logging.getLogger("maverick_mcp.data.models")
settings = get_settings()

# Database connection setup
# Try multiple possible environment variable names
# Use SQLite in-memory for GitHub Actions or test environments
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    DATABASE_URL = "sqlite:///:memory:"
else:
    DATABASE_URL = (
        os.getenv("POSTGRES_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql://localhost/maverick_mcp"
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
        # Connection arguments for better stability
        connect_args={
            "connect_timeout": 10,
            "application_name": "maverick_mcp",
            "options": f"-c statement_timeout={settings.db.statement_timeout}",
        }
        if "postgresql" in DATABASE_URL
        else {},
    )
else:
    # Use NullPool for serverless/development environments
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        echo=DB_ECHO,
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
            _async_engine = create_async_engine(
                async_url,
                # Don't specify poolclass - let SQLAlchemy choose the async pool
                pool_size=DB_POOL_SIZE,
                max_overflow=DB_MAX_OVERFLOW,
                pool_timeout=DB_POOL_TIMEOUT,
                pool_recycle=DB_POOL_RECYCLE,
                pool_pre_ping=DB_POOL_PRE_PING,
                echo=DB_ECHO,
                # Connection arguments for better stability
                connect_args={
                    "server_settings": {
                        "application_name": "maverick_mcp_async",
                        "statement_timeout": str(settings.db.statement_timeout),
                    }
                }
                if "postgresql" in async_url
                else {},
            )
        else:
            _async_engine = create_async_engine(
                async_url,
                poolclass=NullPool,
                echo=DB_ECHO,
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
    """Initialize database by creating all tables.

    Note: This creates all tables for the self-contained MaverickMCP database.
    Use Alembic migrations for production deployments.
    """
    Base.metadata.create_all(bind=engine)


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

    stock_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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

    price_cache_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(
        UUID(as_uuid=True), ForeignKey("mcp_stocks.stock_id"), nullable=False
    )
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        UUID(as_uuid=True),
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        UUID(as_uuid=True),
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        UUID(as_uuid=True),
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        UUID(as_uuid=True), ForeignKey("mcp_stocks.stock_id"), nullable=False
    )
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
    for date, row in df.iterrows():
        # Handle different index types - datetime index vs date index
        if hasattr(date, "date") and callable(date.date):
            date_val = date.date()  # type: ignore[attr-defined]
        elif hasattr(date, "to_pydatetime") and callable(date.to_pydatetime):
            date_val = date.to_pydatetime().date()  # type: ignore[attr-defined]
        else:
            # Assume it's already a date-like object
            date_val = date

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
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(PriceCache).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=["stock_id", "date"])

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
    date_analyzed: datetime.date | None = None,
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


# Import auth models to ensure they're registered with the Base
try:
    from maverick_mcp.auth.models_enhanced import User as AuthUser  # noqa: F401
    from maverick_mcp.queue.models import AsyncJob  # noqa: F401
except ImportError:
    # Models may not be available in all contexts
    pass

# Initialize tables when module is imported
if __name__ == "__main__":
    logger.info("Creating database tables...")
    init_db()
    logger.info("Database tables created successfully!")
