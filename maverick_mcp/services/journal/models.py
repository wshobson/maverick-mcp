"""SQLAlchemy models for the trade journal."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from maverick_mcp.data.models import TimestampMixin
from maverick_mcp.database.base import Base


class JournalEntry(Base, TimestampMixin):
    """A single trade record — open or closed."""

    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # "long" | "short"
    entry_price: Mapped[float] = mapped_column(nullable=False)
    exit_price: Mapped[float | None] = mapped_column(nullable=True)
    shares: Mapped[float] = mapped_column(nullable=False)
    entry_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    exit_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    pnl: Mapped[float | None] = mapped_column(nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # "open" | "closed"
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="open")


class StrategyPerformance(Base):
    """Aggregated performance metrics for a strategy tag."""

    __tablename__ = "strategy_performance"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_tag: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, unique=True
    )
    period: Mapped[str] = mapped_column(String(20), nullable=False, default="all_time")
    win_count: Mapped[int] = mapped_column(nullable=False, default=0)
    loss_count: Mapped[int] = mapped_column(nullable=False, default=0)
    total_pnl: Mapped[float] = mapped_column(nullable=False, default=0.0)
    avg_win: Mapped[float] = mapped_column(nullable=False, default=0.0)
    avg_loss: Mapped[float] = mapped_column(nullable=False, default=0.0)
    expectancy: Mapped[float] = mapped_column(nullable=False, default=0.0)
    profit_factor: Mapped[float] = mapped_column(nullable=False, default=0.0)
