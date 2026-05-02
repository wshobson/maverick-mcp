"""SQLAlchemy models for the screening pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from maverick_mcp.database.base import Base


class ScreeningRun(Base):
    """Snapshot of screening results at a point in time."""

    __tablename__ = "screening_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    screen_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    result_count: Mapped[int] = mapped_column(nullable=False, default=0)
    results: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)


class ScreeningChange(Base):
    """Record of a symbol entering or exiting a screen between runs."""

    __tablename__ = "screening_changes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # "entry" | "exit" | "rank_change"
    change_type: Mapped[str] = mapped_column(String(20), nullable=False)
    screen_name: Mapped[str] = mapped_column(String(255), nullable=False)
    previous_rank: Mapped[int | None] = mapped_column(nullable=True)
    new_rank: Mapped[int | None] = mapped_column(nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class ScheduledJob(Base):
    """Configuration record for a scheduled screening job."""

    __tablename__ = "scheduled_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    job_type: Mapped[str] = mapped_column(String(100), nullable=False)
    schedule_config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    active: Mapped[bool] = mapped_column(nullable=False, default=True)
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
