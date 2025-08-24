"""Fix financial data precision only

Revision ID: e0c75b0bdadb
Revises: 008_performance_optimization_indexes
Create Date: 2025-06-25 17:16:30.392029

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "e0c75b0bdadb"
down_revision = "add_stripe_webhook_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Fix Financial Data Precision (CRITICAL)"""
    # Change all price fields from Numeric(10,2) to Numeric(12,4)
    # This allows for stocks like BRK.A ($500,000+) and better decimal precision

    # Update stocks_pricecache table
    op.alter_column(
        "stocks_pricecache",
        "open_price",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )
    op.alter_column(
        "stocks_pricecache",
        "high_price",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )
    op.alter_column(
        "stocks_pricecache",
        "low_price",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )
    op.alter_column(
        "stocks_pricecache",
        "close_price",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )

    # Update credit-related tables
    op.alter_column(
        "mcp_user_credits",
        "balance",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )
    op.alter_column(
        "mcp_user_credits",
        "free_balance",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )
    op.alter_column(
        "mcp_user_credits",
        "total_purchased",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )

    op.alter_column(
        "mcp_credit_transactions",
        "amount",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )
    op.alter_column(
        "mcp_credit_transactions",
        "balance_after",
        type_=sa.Numeric(12, 4),
        existing_type=sa.Numeric(10, 2),
    )


def downgrade() -> None:
    """Revert precision changes"""
    # Revert numeric precision changes
    op.alter_column(
        "mcp_credit_transactions",
        "balance_after",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
    op.alter_column(
        "mcp_credit_transactions",
        "amount",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )

    op.alter_column(
        "mcp_user_credits",
        "total_purchased",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
    op.alter_column(
        "mcp_user_credits",
        "free_balance",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
    op.alter_column(
        "mcp_user_credits",
        "balance",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )

    op.alter_column(
        "stocks_pricecache",
        "close_price",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
    op.alter_column(
        "stocks_pricecache",
        "low_price",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
    op.alter_column(
        "stocks_pricecache",
        "high_price",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
    op.alter_column(
        "stocks_pricecache",
        "open_price",
        type_=sa.Numeric(10, 2),
        existing_type=sa.Numeric(12, 4),
    )
