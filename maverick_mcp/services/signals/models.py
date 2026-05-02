"""SQLAlchemy models for the signal engine."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String

from maverick_mcp.data.models import TimestampMixin
from maverick_mcp.database.base import Base


class Signal(Base, TimestampMixin):
    """Persistent alert signal with condition and evaluation state."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(255), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    condition = Column(JSON, nullable=False)
    interval_seconds = Column(Integer, default=300)
    active = Column(Boolean, default=True, nullable=False)
    previous_state = Column(JSON, nullable=True)


class SignalEvent(Base, TimestampMixin):
    """Record of each time a signal was triggered."""

    __tablename__ = "signal_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, nullable=False, index=True)
    triggered_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    price_at_trigger = Column(Float, nullable=True)
    condition_snapshot = Column(JSON, nullable=True)


class RegimeEvent(Base, TimestampMixin):
    """Record of a market regime detection."""

    __tablename__ = "regime_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regime = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    drivers = Column(JSON, nullable=True)
    previous_regime = Column(String(20), nullable=True)
    detected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
