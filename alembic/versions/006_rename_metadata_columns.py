"""rename metadata columns

Revision ID: 006_rename_metadata_columns
Revises: f976356b6f07
Create Date: 2025-06-05

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "006_rename_metadata_columns"
down_revision = "f976356b6f07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename metadata columns to avoid SQLAlchemy reserved word conflict
    op.alter_column("mcp_auth_audit_log", "metadata", new_column_name="event_metadata")
    op.alter_column(
        "mcp_user_subscriptions", "metadata", new_column_name="subscription_metadata"
    )


def downgrade() -> None:
    # Revert column names
    op.alter_column("mcp_auth_audit_log", "event_metadata", new_column_name="metadata")
    op.alter_column(
        "mcp_user_subscriptions", "subscription_metadata", new_column_name="metadata"
    )
