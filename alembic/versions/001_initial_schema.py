"""Initial schema - MCP-specific tables only

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-06 12:00:00.000000

Note: This migration creates MCP-specific tables with mcp_ prefix.
Django-owned tables (stocks_stock, stocks_pricecache, maverick_stocks,
maverick_bear_stocks, supply_demand_breakouts) are not managed by Alembic.
"""

# revision identifiers, used by Alembic.
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This migration is now empty as all screening tables
    # (maverick_stocks, maverick_bear_stocks, supply_demand_breakouts)
    # are Django-owned and should not be created by MCP
    pass


def downgrade() -> None:
    # No tables to drop as screening tables are Django-owned
    pass
