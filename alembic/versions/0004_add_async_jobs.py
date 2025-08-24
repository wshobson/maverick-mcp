"""Add async jobs tables

Revision ID: 0004
Revises: 0003
Create Date: 2025-07-20 12:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "abf9b9afb134"
branch_labels = None
depends_on = None


def upgrade():
    # Create async_jobs table
    op.create_table(
        "async_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=False),
        sa.Column("job_type", sa.String(length=100), nullable=False),
        sa.Column("job_name", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("progress_percent", sa.Float(), nullable=True),
        sa.Column("status_message", sa.Text(), nullable=True),
        sa.Column("credits_reserved", sa.Integer(), nullable=True),
        sa.Column("credits_consumed", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=True),
        sa.Column("max_retries", sa.Integer(), nullable=True),
        sa.Column("worker_id", sa.String(length=255), nullable=True),
        sa.Column("queue_name", sa.String(length=100), nullable=True),
        sa.Column("estimated_duration", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["mcp_users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_async_jobs_celery_task_id"),
        "async_jobs",
        ["celery_task_id"],
        unique=True,
    )
    op.create_index(op.f("ix_async_jobs_id"), "async_jobs", ["id"])
    op.create_index(op.f("ix_async_jobs_job_type"), "async_jobs", ["job_type"])
    op.create_index(op.f("ix_async_jobs_status"), "async_jobs", ["status"])
    op.create_index(op.f("ix_async_jobs_user_id"), "async_jobs", ["user_id"])

    # Create job_progress table
    op.create_table(
        "job_progress",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("progress_percent", sa.Float(), nullable=False),
        sa.Column("stage_name", sa.String(length=255), nullable=True),
        sa.Column("stage_description", sa.Text(), nullable=True),
        sa.Column("items_processed", sa.Integer(), nullable=True),
        sa.Column("total_items", sa.Integer(), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["async_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_progress_job_id"), "job_progress", ["job_id"])

    # Create job_results table
    op.create_table(
        "job_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("result_data", sa.JSON(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("file_format", sa.String(length=50), nullable=True),
        sa.Column("execution_stats", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["async_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_results_job_id"), "job_results", ["job_id"])


def downgrade():
    # Drop indexes first
    op.drop_index(op.f("ix_job_results_job_id"), table_name="job_results")
    op.drop_index(op.f("ix_job_progress_job_id"), table_name="job_progress")
    op.drop_index(op.f("ix_async_jobs_user_id"), table_name="async_jobs")
    op.drop_index(op.f("ix_async_jobs_status"), table_name="async_jobs")
    op.drop_index(op.f("ix_async_jobs_job_type"), table_name="async_jobs")
    op.drop_index(op.f("ix_async_jobs_id"), table_name="async_jobs")
    op.drop_index(op.f("ix_async_jobs_celery_task_id"), table_name="async_jobs")

    # Drop tables
    op.drop_table("job_results")
    op.drop_table("job_progress")
    op.drop_table("async_jobs")
