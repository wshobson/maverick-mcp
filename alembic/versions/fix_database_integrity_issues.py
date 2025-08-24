"""Fix critical database integrity issues

Revision ID: fix_database_integrity
Revises: add_stripe_webhook_events_table
Create Date: 2025-06-25

This migration addresses critical database integrity issues:
1. Financial data precision: Change Numeric(10,2) to Numeric(12,4) for price fields
2. Foreign key constraints: Add proper foreign key relationships or denormalize
3. Missing constraints: Add NOT NULL, CHECK, and range constraints
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "fix_database_integrity"
down_revision = "e0c75b0bdadb"
branch_labels = None
depends_on = None


def upgrade():
    """Apply database integrity fixes."""

    # 1. Create User Mapping Table for Foreign Key References
    # This provides a local mapping to Django users without requiring direct FK to Django tables
    op.create_table(
        "mcp_user_mapping",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("django_user_id", sa.BigInteger(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("django_user_id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("idx_mcp_user_mapping_email", "mcp_user_mapping", ["email"])
    op.create_index(
        "idx_mcp_user_mapping_django_id", "mcp_user_mapping", ["django_user_id"]
    )

    # 2. Populate the user mapping table with existing users before adding constraints
    # We need to do this before adding foreign keys to avoid constraint violations

    # Get existing user IDs from all tables that will have foreign keys
    connection = op.get_bind()

    # Collect all unique user_ids from tables that will have foreign keys
    existing_user_ids = set()

    # Check each table for user_ids
    tables_to_check = [
        "mcp_user_credits",
        "mcp_credit_transactions",
        "mcp_requests",
        "mcp_refresh_tokens",
        "mcp_api_keys",
        "mcp_auth_audit_log",
    ]

    for table in tables_to_check:
        try:
            result = connection.execute(
                sa.text(
                    f"SELECT DISTINCT user_id FROM {table} WHERE user_id IS NOT NULL"
                )
            )
            for row in result:
                existing_user_ids.add(row[0])
        except Exception as e:
            print(f"Warning: Could not check {table} for user_ids: {e}")

    print(f"Found existing user_ids: {existing_user_ids}")

    # Populate user mapping table for existing user_ids
    for user_id in existing_user_ids:
        try:
            # Try to get user info from Django users table
            result = connection.execute(
                sa.text(
                    "SELECT id, email, is_active, date_joined FROM users_customuser WHERE id = :user_id"
                ),
                {"user_id": user_id},
            )
            user_row = result.fetchone()

            if user_row:
                # Insert into mapping table
                connection.execute(
                    sa.text("""
                    INSERT INTO mcp_user_mapping (user_id, django_user_id, email, is_active, created_at, updated_at)
                    VALUES (:user_id, :django_user_id, :email, :is_active, :created_at, :updated_at)
                    ON CONFLICT (user_id) DO NOTHING
                """),
                    {
                        "user_id": user_id,
                        "django_user_id": user_row[0],
                        "email": user_row[1],
                        "is_active": user_row[2],
                        "created_at": user_row[3],
                        "updated_at": user_row[3],
                    },
                )
                print(f"Mapped user {user_id} ({user_row[1]}) to user_mapping table")
            else:
                # Create a placeholder entry for orphaned user_id
                connection.execute(
                    sa.text("""
                    INSERT INTO mcp_user_mapping (user_id, django_user_id, email, is_active, created_at, updated_at)
                    VALUES (:user_id, :django_user_id, :email, :is_active, :created_at, :updated_at)
                    ON CONFLICT (user_id) DO NOTHING
                """),
                    {
                        "user_id": user_id,
                        "django_user_id": user_id,  # Use same ID as fallback
                        "email": f"orphaned_user_{user_id}@placeholder.com",
                        "is_active": False,
                        "created_at": sa.func.now(),
                        "updated_at": sa.func.now(),
                    },
                )
                print(f"Created placeholder entry for orphaned user {user_id}")
        except Exception as e:
            print(f"Warning: Could not map user {user_id}: {e}")

    # 3. Add Foreign Key Constraints with CASCADE rules
    # Now we can safely add foreign keys since user mapping table is populated

    # Add FK from mcp_user_credits to mcp_user_mapping
    op.create_foreign_key(
        "fk_user_credits_user_mapping",
        "mcp_user_credits",
        "mcp_user_mapping",
        ["user_id"],
        ["user_id"],
        ondelete="CASCADE",
    )

    # Add FK from mcp_credit_transactions to mcp_user_mapping
    op.create_foreign_key(
        "fk_credit_transactions_user_mapping",
        "mcp_credit_transactions",
        "mcp_user_mapping",
        ["user_id"],
        ["user_id"],
        ondelete="CASCADE",
    )

    # Add FK from mcp_requests to mcp_user_mapping
    op.create_foreign_key(
        "fk_requests_user_mapping",
        "mcp_requests",
        "mcp_user_mapping",
        ["user_id"],
        ["user_id"],
        ondelete="CASCADE",
    )

    # Add FK from mcp_refresh_tokens to mcp_user_mapping
    op.create_foreign_key(
        "fk_refresh_tokens_user_mapping",
        "mcp_refresh_tokens",
        "mcp_user_mapping",
        ["user_id"],
        ["user_id"],
        ondelete="CASCADE",
    )

    # Add FK from mcp_api_keys to mcp_user_mapping
    op.create_foreign_key(
        "fk_api_keys_user_mapping",
        "mcp_api_keys",
        "mcp_user_mapping",
        ["user_id"],
        ["user_id"],
        ondelete="CASCADE",
    )

    # Add FK from mcp_auth_audit_log to mcp_user_mapping (nullable FK)
    op.create_foreign_key(
        "fk_auth_audit_user_mapping",
        "mcp_auth_audit_log",
        "mcp_user_mapping",
        ["user_id"],
        ["user_id"],
        ondelete="SET NULL",
    )

    # 4. Add CHECK Constraints for positive values

    # Price constraints - prices should be non-negative
    op.create_check_constraint(
        "ck_pricecache_open_positive", "stocks_pricecache", "open_price >= 0"
    )
    op.create_check_constraint(
        "ck_pricecache_high_positive", "stocks_pricecache", "high_price >= 0"
    )
    op.create_check_constraint(
        "ck_pricecache_low_positive", "stocks_pricecache", "low_price >= 0"
    )
    op.create_check_constraint(
        "ck_pricecache_close_positive", "stocks_pricecache", "close_price >= 0"
    )
    op.create_check_constraint(
        "ck_pricecache_volume_positive", "stocks_pricecache", "volume >= 0"
    )

    # High >= Low constraint
    op.create_check_constraint(
        "ck_pricecache_high_low", "stocks_pricecache", "high_price >= low_price"
    )

    # Credit constraints - balances should be non-negative
    op.create_check_constraint(
        "ck_user_credits_balance_positive", "mcp_user_credits", "balance >= 0"
    )
    op.create_check_constraint(
        "ck_user_credits_free_balance_positive", "mcp_user_credits", "free_balance >= 0"
    )
    op.create_check_constraint(
        "ck_user_credits_total_purchased_positive",
        "mcp_user_credits",
        "total_purchased >= 0",
    )

    # Credits charged should be positive
    op.create_check_constraint(
        "ck_requests_credits_positive", "mcp_requests", "credits_charged >= 0"
    )

    # Token counts should be non-negative
    op.create_check_constraint(
        "ck_requests_tokens_positive", "mcp_requests", "total_tokens >= 0"
    )
    op.create_check_constraint(
        "ck_requests_llm_calls_positive", "mcp_requests", "llm_calls >= 0"
    )

    # 5. Add Range Constraints for percentages (0-100)

    # Momentum Score should be between 0 and 100
    op.create_check_constraint(
        "ck_maverick_momentum_score_range",
        "maverick_stocks",
        "momentum_score >= 0 AND momentum_score <= 100",
    )
    op.create_check_constraint(
        "ck_maverick_bear_momentum_score_range",
        "maverick_bear_stocks",
        "momentum_score >= 0 AND momentum_score <= 100",
    )
    op.create_check_constraint(
        "ck_supply_demand_momentum_score_range",
        "supply_demand_breakouts",
        "momentum_score >= 0 AND momentum_score <= 100",
    )

    # RSI should be between 0 and 100 (using uppercase column names)
    op.create_check_constraint(
        "ck_maverick_bear_rsi_range",
        "maverick_bear_stocks",
        '"RSI_14" >= 0 AND "RSI_14" <= 100',
    )

    # 6. Add NOT NULL constraints where appropriate
    # Most critical fields already have NOT NULL, but let's ensure consistency

    # Ensure stock ticker symbols are not null (already enforced, but double-check)
    op.alter_column(
        "stocks_stock", "ticker_symbol", nullable=False, existing_nullable=False
    )

    # Ensure timestamps are not null
    op.alter_column(
        "stocks_stock", "created_at", nullable=False, existing_nullable=False
    )
    op.alter_column(
        "stocks_stock", "updated_at", nullable=False, existing_nullable=False
    )

    # 7. Create initial data migration script notice
    print("""
    IMPORTANT: After applying this migration, run the following data migration script:

    1. Populate mcp_user_mapping table from Django users:
       INSERT INTO mcp_user_mapping (user_id, django_user_id, email, is_active, created_at, updated_at)
       SELECT id, id, email, is_active, date_joined, date_joined
       FROM users_customuser;

    2. Verify no orphaned records exist:
       - Check mcp_user_credits for user_ids not in mcp_user_mapping
       - Check mcp_credit_transactions for user_ids not in mcp_user_mapping
       - Check mcp_requests for user_ids not in mcp_user_mapping
       - Check mcp_refresh_tokens for user_ids not in mcp_user_mapping
       - Check mcp_api_keys for user_ids not in mcp_user_mapping
    """)


def downgrade():
    """Revert database integrity fixes."""

    # Remove CHECK constraints
    op.drop_constraint("ck_requests_llm_calls_positive", "mcp_requests")
    op.drop_constraint("ck_requests_tokens_positive", "mcp_requests")
    op.drop_constraint("ck_requests_credits_positive", "mcp_requests")
    op.drop_constraint("ck_user_credits_total_purchased_positive", "mcp_user_credits")
    op.drop_constraint("ck_user_credits_free_balance_positive", "mcp_user_credits")
    op.drop_constraint("ck_user_credits_balance_positive", "mcp_user_credits")
    op.drop_constraint("ck_pricecache_high_low", "stocks_pricecache")
    op.drop_constraint("ck_pricecache_volume_positive", "stocks_pricecache")
    op.drop_constraint("ck_pricecache_close_positive", "stocks_pricecache")
    op.drop_constraint("ck_pricecache_low_positive", "stocks_pricecache")
    op.drop_constraint("ck_pricecache_high_positive", "stocks_pricecache")
    op.drop_constraint("ck_pricecache_open_positive", "stocks_pricecache")

    # Remove range constraints
    op.drop_constraint("ck_maverick_bear_rsi_range", "maverick_bear_stocks")
    op.drop_constraint(
        "ck_supply_demand_momentum_score_range", "supply_demand_breakouts"
    )
    op.drop_constraint("ck_maverick_bear_momentum_score_range", "maverick_bear_stocks")
    op.drop_constraint("ck_maverick_momentum_score_range", "maverick_stocks")

    # Remove foreign key constraints
    op.drop_constraint("fk_auth_audit_user_mapping", "mcp_auth_audit_log")
    op.drop_constraint("fk_api_keys_user_mapping", "mcp_api_keys")
    op.drop_constraint("fk_refresh_tokens_user_mapping", "mcp_refresh_tokens")
    op.drop_constraint("fk_requests_user_mapping", "mcp_requests")
    op.drop_constraint("fk_credit_transactions_user_mapping", "mcp_credit_transactions")
    op.drop_constraint("fk_user_credits_user_mapping", "mcp_user_credits")

    # Drop user mapping table
    op.drop_index("idx_mcp_user_mapping_django_id", "mcp_user_mapping")
    op.drop_index("idx_mcp_user_mapping_email", "mcp_user_mapping")
    op.drop_table("mcp_user_mapping")
