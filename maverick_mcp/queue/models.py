"""
Database models for async job tracking.

This module contains SQLAlchemy models for tracking asynchronous jobs,
their progress, and results.
"""

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func

from maverick_mcp.database.base import Base


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    """Job priority enumeration."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AsyncJob(Base):
    """
    Model for tracking asynchronous jobs.

    This table stores metadata about submitted jobs including their status,
    user association, and execution details.
    """

    __tablename__ = "async_jobs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)

    # Job identification
    celery_task_id = Column(String(255), unique=True, index=True, nullable=False)
    job_type = Column(String(100), nullable=False, index=True)
    job_name = Column(String(255), nullable=False)

    # User association (optional for unauthenticated jobs)
    user_id = Column(BigInteger, ForeignKey("mcp_users.id"), nullable=True, index=True)

    # Job configuration
    parameters = Column(JSON, nullable=False, default=dict)
    priority = Column(String(20), nullable=False, default=JobPriority.NORMAL.value)

    # Status tracking
    status = Column(
        String(20), nullable=False, default=JobStatus.PENDING.value, index=True
    )
    progress_percent = Column(Float, default=0.0)
    status_message = Column(Text, nullable=True)

    # Credit tracking
    credits_reserved = Column(Integer, default=0)
    credits_consumed = Column(Integer, default=0)

    # Timing
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Metadata
    worker_id = Column(String(255), nullable=True)
    queue_name = Column(String(100), nullable=True)
    estimated_duration = Column(Integer, nullable=True)  # seconds

    # Relationships
    progress_updates = relationship(
        "JobProgress", back_populates="job", cascade="all, delete-orphan"
    )
    result = relationship(
        "JobResult", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<AsyncJob(id={self.id}, type={self.job_type}, status={self.status})>"

    @classmethod
    def create(
        cls,
        celery_task_id: str,
        job_type: str,
        job_name: str,
        parameters: dict[str, Any],
        user_id: str | None = None,
        priority: JobPriority = JobPriority.NORMAL,
        credits_reserved: int = 0,
        estimated_duration: int | None = None,
    ) -> "AsyncJob":
        """Create a new async job."""
        return cls(
            celery_task_id=celery_task_id,
            job_type=job_type,
            job_name=job_name,
            parameters=parameters,
            user_id=user_id,
            priority=priority.value,
            credits_reserved=credits_reserved,
            estimated_duration=estimated_duration,
        )

    def update_status(
        self,
        status: JobStatus,
        progress_percent: float | None = None,
        status_message: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job status and related fields."""
        self.status = status.value
        if progress_percent is not None:
            self.progress_percent = progress_percent
        if status_message is not None:
            self.status_message = status_message
        if error_message is not None:
            self.error_message = error_message

        if status == JobStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now(UTC)
        elif status in (JobStatus.SUCCESS, JobStatus.FAILURE, JobStatus.CANCELLED):
            self.completed_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "id": str(self.id),
            "celery_task_id": self.celery_task_id,
            "job_type": self.job_type,
            "job_name": self.job_name,
            "user_id": str(self.user_id) if self.user_id else None,
            "parameters": self.parameters,
            "priority": self.priority,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "status_message": self.status_message,
            "credits_reserved": self.credits_reserved,
            "credits_consumed": self.credits_consumed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self._calculate_duration(),
        }

    def _calculate_duration(self) -> int | None:
        """Calculate actual job duration in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None


class JobProgress(Base):
    """
    Model for tracking job progress updates.

    This table stores detailed progress information for long-running jobs.
    """

    __tablename__ = "job_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("async_jobs.id"), nullable=False, index=True
    )

    # Progress details
    progress_percent = Column(Float, nullable=False)
    stage_name = Column(String(255), nullable=True)
    stage_description = Column(Text, nullable=True)
    items_processed = Column(Integer, nullable=True)
    total_items = Column(Integer, nullable=True)

    # Timing
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Additional data
    extra_data = Column("metadata", JSON, nullable=True)

    # Relationships
    job = relationship("AsyncJob", back_populates="progress_updates")

    def __repr__(self) -> str:
        return f"<JobProgress(job_id={self.job_id}, progress={self.progress_percent}%)>"


class JobResult(Base):
    """
    Model for storing job results.

    This table stores the final output and metadata from completed jobs.
    """

    __tablename__ = "job_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("async_jobs.id"), nullable=False, index=True
    )

    # Result data
    result_data = Column(JSON, nullable=True)
    result_summary = Column(Text, nullable=True)

    # File attachments (for large results)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    file_format = Column(String(50), nullable=True)

    # Metadata
    execution_stats = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    job = relationship("AsyncJob", back_populates="result")

    def __repr__(self) -> str:
        return f"<JobResult(job_id={self.job_id}, created_at={self.created_at})>"

    def is_expired(self) -> bool:
        """Check if result has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at.replace(tzinfo=UTC)


# Helper functions for job management
def get_user_jobs(
    session: Session,
    user_id: str,
    status_filter: JobStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[AsyncJob]:
    """Get jobs for a specific user."""
    query = session.query(AsyncJob).filter(AsyncJob.user_id == user_id)

    if status_filter:
        query = query.filter(AsyncJob.status == status_filter.value)

    return query.order_by(AsyncJob.created_at.desc()).offset(offset).limit(limit).all()


def get_job_by_id(
    session: Session, job_id: str, user_id: str | None = None
) -> AsyncJob | None:
    """Get a job by ID, optionally filtered by user."""
    query = session.query(AsyncJob).filter(AsyncJob.id == job_id)

    if user_id:
        query = query.filter(AsyncJob.user_id == user_id)

    return query.first()


def get_active_jobs(session: Session) -> list[AsyncJob]:
    """Get all active (running or pending) jobs."""
    return (
        session.query(AsyncJob)
        .filter(
            AsyncJob.status.in_(
                [
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                    JobStatus.RETRYING.value,
                ]
            )
        )
        .all()
    )


def cleanup_expired_jobs(session: Session, days_old: int = 7) -> int:
    """Clean up old completed jobs and their results."""
    cutoff_date = datetime.now(UTC) - timedelta(days=days_old)

    # Delete old completed jobs
    deleted_count = (
        session.query(AsyncJob)
        .filter(
            AsyncJob.status.in_(
                [
                    JobStatus.SUCCESS.value,
                    JobStatus.FAILURE.value,
                    JobStatus.CANCELLED.value,
                ]
            ),
            AsyncJob.completed_at < cutoff_date,
        )
        .delete(synchronize_session=False)
    )

    session.commit()
    return deleted_count
