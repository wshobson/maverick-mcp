"""Add performance indexes for Maverick-MCP

Revision ID: 003_add_performance_indexes
Revises: 002_add_authentication_tables
Create Date: 2025-06-03 12:00:00

This migration adds performance indexes to improve query speed
for MCP-specific tables only. Django-owned tables are not modified.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "003_add_performance_indexes"
down_revision = "002_add_authentication_tables"
branch_labels = None
depends_on = None


def upgrade():
    """Add performance indexes for MCP tables only."""

    # API key usage performance indexes
    op.create_index(
        "idx_mcp_api_key_usage_api_key_id",
        "mcp_api_key_usage",
        ["api_key_id"],
        postgresql_using="btree",
    )

    print("Performance indexes for MCP tables created successfully!")


def downgrade():
    """Remove performance indexes from MCP tables."""

    # Drop API key usage index
    op.drop_index("idx_mcp_api_key_usage_api_key_id", "mcp_api_key_usage")

    print("Performance indexes removed from MCP tables.")
