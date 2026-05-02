"""SQLAlchemy models for the screening pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String

from maverick_mcp.database.base import Base


class ScreeningRun(Base):
    """Snapshot of screening results at a point in time."""

    __tablename__ = "screening_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    screen_name = Column(String(255), nullable=False, index=True)
    run_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    result_count = Column(Integer, nullable=False, default=0)
    results = Column(JSON, nullable=False, default=list)


class ScreeningChange(Base):
    """Record of a symbol entering or exiting a screen between runs."""

    __tablename__ = "screening_changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    change_type = Column(String(20), nullable=False)  # "entry" | "exit" | "rank_change"
    screen_name = Column(String(255), nullable=False)
    previous_rank = Column(Integer, nullable=True)
    new_rank = Column(Integer, nullable=True)
    detected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class ScheduledJob(Base):
    """Configuration record for a scheduled screening job."""

    __tablename__ = "scheduled_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String(255), nullable=False, unique=True)
    job_type = Column(String(100), nullable=False)
    schedule_config = Column(JSON, nullable=False, default=dict)
    active = Column(Boolean, nullable=False, default=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
