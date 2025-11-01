"""Add portfolio management models

Revision ID: 014_add_portfolio_models
Revises: 013_add_backtest_persistence_models
Create Date: 2025-11-01 12:00:00.000000

This migration adds portfolio management models for tracking user investment holdings:
1. UserPortfolio - Portfolio metadata with user identification
2. PortfolioPosition - Individual position records with cost basis tracking

Features:
- Average cost basis tracking for educational simplicity
- High-precision Decimal types for financial accuracy (Numeric(12,4) for prices, Numeric(20,8) for shares)
- Support for fractional shares
- Single-user design with user_id="default"
- Cascade delete for data integrity
- Comprehensive indexes for common query patterns
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "014_add_portfolio_models"
down_revision = "013_add_backtest_persistence_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create portfolio management tables."""

    # Create portfolios table
    op.create_table(
        "mcp_portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(100),
            nullable=False,
            server_default="default",
        ),
        sa.Column(
            "name",
            sa.String(200),
            nullable=False,
            server_default="My Portfolio",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create indexes on portfolios
    op.create_index("idx_portfolio_user", "mcp_portfolios", ["user_id"])
    op.create_unique_constraint(
        "uq_user_portfolio_name", "mcp_portfolios", ["user_id", "name"]
    )

    # Create positions table
    op.create_table(
        "mcp_portfolio_positions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("shares", sa.Numeric(20, 8), nullable=False),
        sa.Column("average_cost_basis", sa.Numeric(12, 4), nullable=False),
        sa.Column("total_cost", sa.Numeric(20, 4), nullable=False),
        sa.Column("purchase_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"], ["mcp_portfolios.id"], ondelete="CASCADE"
        ),
    )

    # Create indexes on positions
    op.create_index(
        "idx_position_portfolio", "mcp_portfolio_positions", ["portfolio_id"]
    )
    op.create_index("idx_position_ticker", "mcp_portfolio_positions", ["ticker"])
    op.create_index(
        "idx_position_portfolio_ticker",
        "mcp_portfolio_positions",
        ["portfolio_id", "ticker"],
    )
    op.create_unique_constraint(
        "uq_portfolio_position_ticker",
        "mcp_portfolio_positions",
        ["portfolio_id", "ticker"],
    )


def downgrade() -> None:
    """Drop portfolio management tables."""
    # Drop positions table first (due to foreign key)
    op.drop_table("mcp_portfolio_positions")

    # Drop portfolios table
    op.drop_table("mcp_portfolios")
