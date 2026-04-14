"""SQLAlchemy models for the watchlist intelligence domain."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column, Date, DateTime, Integer, String, Text

from maverick_mcp.data.models import TimestampMixin
from maverick_mcp.database.base import Base


class Watchlist(Base, TimestampMixin):
    """A named collection of ticker symbols to monitor."""

    __tablename__ = "watchlists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)


class WatchlistItem(Base, TimestampMixin):
    """A single ticker symbol within a watchlist."""

    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    watchlist_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    added_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    notes = Column(Text, nullable=True)


class CatalystEvent(Base, TimestampMixin):
    """An upcoming event that may act as a catalyst for a ticker."""

    __tablename__ = "catalyst_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    # earnings, ex_div, fda, other
    event_type = Column(String(50), nullable=False)
    event_date = Column(Date, nullable=False)
    description = Column(Text, nullable=True)
    impact_assessment = Column(Text, nullable=True)
