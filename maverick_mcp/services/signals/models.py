"""SQLAlchemy models for the signal engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from maverick_mcp.data.models import TimestampMixin
from maverick_mcp.database.base import Base


class Signal(Base, TimestampMixin):
    """Persistent alert signal with condition and evaluation state."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    condition: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    # Nullable preserved from pre-Mapped Column(Integer, default=300); rows
    # always get a default at insert time but the column is nominally nullable.
    interval_seconds: Mapped[int | None] = mapped_column(default=300)
    active: Mapped[bool] = mapped_column(default=True, nullable=False)
    previous_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SignalEvent(Base, TimestampMixin):
    """Record of each time a signal was triggered."""

    __tablename__ = "signal_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(nullable=False, index=True)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    price_at_trigger: Mapped[float | None] = mapped_column(nullable=True)
    condition_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )


class RegimeEvent(Base, TimestampMixin):
    """Record of a market regime detection."""

    __tablename__ = "regime_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    regime: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    drivers: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    previous_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
