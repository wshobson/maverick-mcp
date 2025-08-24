"""Add essential performance indexes

Revision ID: f0696e2cac15
Revises: 007_enhance_audit_logging
Create Date: 2025-06-25 17:28:38.473307

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "f0696e2cac15"
down_revision = "007_enhance_audit_logging"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add essential performance indexes for existing tables only."""

    print("Creating essential performance indexes...")

    # 1. Stock data performance indexes (for large stocks_pricecache table)
    try:
        op.create_index(
            "idx_stocks_pricecache_stock_date",
            "stocks_pricecache",
            ["stock_id", "date"],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created stock price cache index")
    except Exception as e:
        print(f"Warning: Could not create stock price cache index: {e}")

    # 2. Stock lookup optimization
    try:
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_stocks_stock_ticker_lower "
            "ON stocks_stock (LOWER(ticker_symbol))"
        )
        print("✓ Created case-insensitive ticker lookup index")
    except Exception as e:
        print(f"Warning: Could not create ticker lookup index: {e}")

    # 3. MCP API keys performance (critical for authentication)
    try:
        op.create_index(
            "idx_mcp_api_keys_active_lookup",
            "mcp_api_keys",
            ["is_active", "expires_at"],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created API keys performance index")
    except Exception as e:
        print(f"Warning: Could not create API keys index: {e}")

    # 4. User credits performance (critical for credit system)
    try:
        op.create_index(
            "idx_mcp_user_credits_user_lookup",
            "mcp_user_credits",
            ["user_id"],
            postgresql_using="hash",
            if_not_exists=True,
        )
        print("✓ Created user credits lookup index")
    except Exception as e:
        print(f"Warning: Could not create user credits index: {e}")

    # 5. Credit transactions performance
    try:
        op.create_index(
            "idx_mcp_credit_transactions_user_time",
            "mcp_credit_transactions",
            ["user_id", sa.text("created_at DESC")],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created credit transactions performance index")
    except Exception as e:
        print(f"Warning: Could not create credit transactions index: {e}")

    # 6. Requests tracking performance
    try:
        op.create_index(
            "idx_mcp_requests_user_time",
            "mcp_requests",
            ["user_id", sa.text("created_at DESC")],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created requests tracking index")
    except Exception as e:
        print(f"Warning: Could not create requests index: {e}")

    # 7. Auth audit log performance
    try:
        op.create_index(
            "idx_mcp_auth_audit_log_user_time",
            "mcp_auth_audit_log",
            ["user_id", sa.text("created_at DESC")],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created auth audit log index")
    except Exception as e:
        print(f"Warning: Could not create auth audit index: {e}")

    # 8. Screening tables performance (if they exist)
    try:
        op.create_index(
            "idx_maverick_stocks_combined_score",
            "maverick_stocks",
            [sa.text('"COMBINED_SCORE" DESC')],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created maverick stocks performance index")
    except Exception as e:
        print(f"Warning: Could not create maverick stocks index: {e}")

    try:
        op.create_index(
            "idx_maverick_bear_stocks_score",
            "maverick_bear_stocks",
            [sa.text('"SCORE" DESC')],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created maverick bear stocks performance index")
    except Exception as e:
        print(f"Warning: Could not create maverick bear stocks index: {e}")

    try:
        op.create_index(
            "idx_supply_demand_breakouts_rs_rating",
            "supply_demand_breakouts",
            [sa.text('"RS_RATING" DESC')],
            postgresql_using="btree",
            if_not_exists=True,
        )
        print("✓ Created supply/demand breakouts performance index")
    except Exception as e:
        print(f"Warning: Could not create supply/demand breakouts index: {e}")

    print("Essential performance indexes creation completed!")


def downgrade() -> None:
    """Remove essential performance indexes."""

    print("Removing essential performance indexes...")

    # Remove indexes (order doesn't matter for drops)
    indexes_to_drop = [
        ("idx_stocks_pricecache_stock_date", "stocks_pricecache"),
        ("idx_mcp_api_keys_active_lookup", "mcp_api_keys"),
        ("idx_mcp_user_credits_user_lookup", "mcp_user_credits"),
        ("idx_mcp_credit_transactions_user_time", "mcp_credit_transactions"),
        ("idx_mcp_requests_user_time", "mcp_requests"),
        ("idx_mcp_auth_audit_log_user_time", "mcp_auth_audit_log"),
        ("idx_maverick_stocks_combined_score", "maverick_stocks"),
        ("idx_maverick_bear_stocks_score", "maverick_bear_stocks"),
        ("idx_supply_demand_breakouts_rs_rating", "supply_demand_breakouts"),
    ]

    for index_name, table_name in indexes_to_drop:
        try:
            op.drop_index(index_name, table_name, if_exists=True)
            print(f"✓ Dropped {index_name}")
        except Exception as e:
            print(f"Warning: Could not drop {index_name}: {e}")

    # Drop special indexes
    try:
        op.execute("DROP INDEX IF EXISTS idx_stocks_stock_ticker_lower")
        print("✓ Dropped ticker lookup index")
    except Exception as e:
        print(f"Warning: Could not drop ticker lookup index: {e}")

    print("Essential performance indexes removal completed!")
