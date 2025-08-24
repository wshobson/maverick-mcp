"""Add temporary user table for JWT authentication

Revision ID: 005_add_user_table
Revises: f976356b6f07
Create Date: 2025-01-06

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "005_add_user_table"
down_revision = "f976356b6f07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add temporary user table for authentication."""
    # Create temporary users table (until Django integration)
    op.create_table(
        "mcp_temp_users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True, server_default="user"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(
        op.f("ix_mcp_temp_users_email"), "mcp_temp_users", ["email"], unique=True
    )

    # Add indexes to existing tables if not already present
    try:
        op.create_index(
            "ix_mcp_refresh_tokens_user_id", "mcp_refresh_tokens", ["user_id"]
        )
        op.create_index(
            "ix_mcp_auth_audit_log_user_id", "mcp_auth_audit_log", ["user_id"]
        )
        op.create_index(
            "ix_mcp_auth_audit_log_event_type", "mcp_auth_audit_log", ["event_type"]
        )
    except Exception:
        pass  # Indexes might already exist


def downgrade() -> None:
    """Remove temporary user table."""
    op.drop_index(op.f("ix_mcp_temp_users_email"), table_name="mcp_temp_users")
    op.drop_table("mcp_temp_users")
