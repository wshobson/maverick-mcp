"""SQLAlchemy models for the trade journal."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from maverick_mcp.data.models import TimestampMixin
from maverick_mcp.database.base import Base


class JournalEntry(Base, TimestampMixin):
    """A single trade record — open or closed."""

    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # "long" | "short"
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    shares = Column(Float, nullable=False)
    entry_date = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    exit_date = Column(DateTime(timezone=True), nullable=True)
    rationale = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    pnl = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    status = Column(String(10), nullable=False, default="open")  # "open" | "closed"


class StrategyPerformance(Base):
    """Aggregated performance metrics for a strategy tag."""

    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_tag = Column(String(100), nullable=False, index=True, unique=True)
    period = Column(String(20), nullable=False, default="all_time")
    win_count = Column(Integer, nullable=False, default=0)
    loss_count = Column(Integer, nullable=False, default=0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    avg_win = Column(Float, nullable=False, default=0.0)
    avg_loss = Column(Float, nullable=False, default=0.0)
    expectancy = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
