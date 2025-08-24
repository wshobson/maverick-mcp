"""add_refresh_tokens_and_audit_tables

Revision ID: f976356b6f07
Revises: 004_add_billing_tables
Create Date: 2025-06-05 12:04:43.186140

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "f976356b6f07"
down_revision = "004_add_billing_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create refresh tokens table
    op.create_table(
        "mcp_refresh_tokens",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("jti", sa.String(length=32), nullable=False),
        sa.Column("device_info", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users_customuser.id"], name="fk_mcp_refresh_tokens_user_id"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for refresh tokens
    op.create_index(
        "idx_mcp_refresh_tokens_jti", "mcp_refresh_tokens", ["jti"], unique=True
    )
    op.create_index("idx_mcp_refresh_tokens_user_id", "mcp_refresh_tokens", ["user_id"])
    op.create_index(
        "idx_mcp_refresh_tokens_expires_at", "mcp_refresh_tokens", ["expires_at"]
    )

    # Create auth audit log table
    op.create_table(
        "mcp_auth_audit_log",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users_customuser.id"], name="fk_mcp_auth_audit_log_user_id"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for audit log
    op.create_index("idx_mcp_auth_audit_log_user_id", "mcp_auth_audit_log", ["user_id"])
    op.create_index(
        "idx_mcp_auth_audit_log_event_type", "mcp_auth_audit_log", ["event_type"]
    )
    op.create_index(
        "idx_mcp_auth_audit_log_created_at", "mcp_auth_audit_log", ["created_at"]
    )


def downgrade() -> None:
    # Drop indexes and tables
    op.drop_index("idx_mcp_auth_audit_log_created_at", table_name="mcp_auth_audit_log")
    op.drop_index("idx_mcp_auth_audit_log_event_type", table_name="mcp_auth_audit_log")
    op.drop_index("idx_mcp_auth_audit_log_user_id", table_name="mcp_auth_audit_log")
    op.drop_table("mcp_auth_audit_log")

    op.drop_index("idx_mcp_refresh_tokens_expires_at", table_name="mcp_refresh_tokens")
    op.drop_index("idx_mcp_refresh_tokens_user_id", table_name="mcp_refresh_tokens")
    op.drop_index("idx_mcp_refresh_tokens_jti", table_name="mcp_refresh_tokens")
    op.drop_table("mcp_refresh_tokens")
