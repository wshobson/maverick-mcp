"""Add comprehensive performance optimization indexes

Revision ID: 008_performance_optimization_indexes
Revises: 007_enhance_audit_logging
Create Date: 2025-06-25 12:00:00

This migration adds comprehensive performance indexes for:
- Stock data queries with date ranges
- Screening table optimizations
- Rate limiting and authentication tables
- Cache key lookup optimizations
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "008_performance_optimization_indexes"
down_revision = "007_enhance_audit_logging"
branch_labels = None
depends_on = None


def upgrade():
    """Add comprehensive performance optimization indexes."""

    # Stock data performance indexes
    print("Creating stock data performance indexes...")

    # Composite index for price cache queries (stock_id, date)
    # This is the most common query pattern for historical data
    op.create_index(
        "idx_stocks_pricecache_stock_date_range",
        "stocks_pricecache",
        ["stock_id", "date"],
        postgresql_using="btree",
    )

    # Index for volume-based queries (high volume screening)
    op.create_index(
        "idx_stocks_pricecache_volume_desc",
        "stocks_pricecache",
        [sa.text("volume DESC")],
        postgresql_using="btree",
    )

    # Index for price-based queries (close price for technical analysis)
    op.create_index(
        "idx_stocks_pricecache_close_price",
        "stocks_pricecache",
        ["close_price"],
        postgresql_using="btree",
    )

    # Stock lookup optimizations
    print("Creating stock lookup optimization indexes...")

    # Case-insensitive ticker lookup (for user input handling)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_stocks_stock_ticker_lower "
        "ON stocks_stock (LOWER(ticker_symbol))"
    )

    # Sector and industry filtering
    op.create_index(
        "idx_stocks_stock_sector",
        "stocks_stock",
        ["sector"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_stocks_stock_industry",
        "stocks_stock",
        ["industry"],
        postgresql_using="btree",
    )

    # Exchange filtering for market-specific queries
    op.create_index(
        "idx_stocks_stock_exchange",
        "stocks_stock",
        ["exchange"],
        postgresql_using="btree",
    )

    # Screening table optimizations
    print("Creating screening performance indexes...")

    # Maverick bullish screening indexes
    op.create_index(
        "idx_stocks_maverickstocks_score_desc",
        "stocks_maverickstocks",
        [sa.text("score DESC")],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_stocks_maverickstocks_rank_asc",
        "stocks_maverickstocks",
        ["rank"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_stocks_maverickstocks_date_analyzed",
        "stocks_maverickstocks",
        [sa.text("date_analyzed DESC")],
        postgresql_using="btree",
    )

    # Composite index for score and date filtering
    op.create_index(
        "idx_stocks_maverickstocks_score_date",
        "stocks_maverickstocks",
        [sa.text("score DESC"), sa.text("date_analyzed DESC")],
        postgresql_using="btree",
    )

    # Maverick bearish screening indexes
    op.create_index(
        "idx_stocks_maverickbearstocks_score_desc",
        "stocks_maverickbearstocks",
        [sa.text("score DESC")],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_stocks_maverickbearstocks_date_analyzed",
        "stocks_maverickbearstocks",
        [sa.text("date_analyzed DESC")],
        postgresql_using="btree",
    )

    # Supply/Demand (Trending) screening indexes
    op.create_index(
        "idx_stocks_supply_demand_breakouts_momentum_score_desc",
        "stocks_supply_demand_breakouts",
        [sa.text("momentum_score DESC")],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_stocks_supply_demand_breakouts_date_analyzed",
        "stocks_supply_demand_breakouts",
        [sa.text("date_analyzed DESC")],
        postgresql_using="btree",
    )

    # Composite index for momentum score and date
    op.create_index(
        "idx_stocks_supply_demand_breakouts_momentum_date",
        "stocks_supply_demand_breakouts",
        [sa.text("momentum_score DESC"), sa.text("date_analyzed DESC")],
        postgresql_using="btree",
    )

    # Authentication and rate limiting optimizations
    print("Creating authentication performance indexes...")

    # API key lookups (most frequent auth operation)
    op.create_index(
        "idx_mcp_api_keys_key_hash",
        "mcp_api_keys",
        ["key_hash"],
        postgresql_using="hash",  # Hash index for exact equality
    )

    # Active API keys filter
    op.create_index(
        "idx_mcp_api_keys_active_expires",
        "mcp_api_keys",
        ["is_active", "expires_at"],
        postgresql_using="btree",
    )

    # User API keys lookup
    op.create_index(
        "idx_mcp_api_keys_user_id_active",
        "mcp_api_keys",
        ["user_id", "is_active"],
        postgresql_using="btree",
    )

    # Refresh token lookups
    op.create_index(
        "idx_mcp_refresh_tokens_token_hash",
        "mcp_refresh_tokens",
        ["token_hash"],
        postgresql_using="hash",
    )

    op.create_index(
        "idx_mcp_refresh_tokens_user_active",
        "mcp_refresh_tokens",
        ["user_id", "is_active"],
        postgresql_using="btree",
    )

    # Request tracking for analytics
    op.create_index(
        "idx_mcp_requests_user_timestamp",
        "mcp_requests",
        ["user_id", sa.text("timestamp DESC")],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_mcp_requests_tool_name",
        "mcp_requests",
        ["tool_name"],
        postgresql_using="btree",
    )

    # Request success rate analysis
    op.create_index(
        "idx_mcp_requests_success_timestamp",
        "mcp_requests",
        ["success", sa.text("timestamp DESC")],
        postgresql_using="btree",
    )

    # Audit logging optimizations
    print("Creating audit logging performance indexes...")

    # User activity tracking
    op.create_index(
        "idx_mcp_audit_logs_user_timestamp",
        "mcp_audit_logs",
        ["user_id", sa.text("timestamp DESC")],
        postgresql_using="btree",
    )

    # Action type filtering
    op.create_index(
        "idx_mcp_audit_logs_action",
        "mcp_audit_logs",
        ["action"],
        postgresql_using="btree",
    )

    # IP address tracking for security
    op.create_index(
        "idx_mcp_audit_logs_ip_timestamp",
        "mcp_audit_logs",
        ["ip_address", sa.text("timestamp DESC")],
        postgresql_using="btree",
    )

    # Partial indexes for common queries
    print("Creating partial indexes for optimal performance...")

    # Active users only (most queries filter for active users)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_users_active_email "
        "ON mcp_users (email) WHERE is_active = true"
    )

    # Recent price data (last 30 days) - most common query pattern
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_stocks_pricecache_recent "
        "ON stocks_pricecache (stock_id, date DESC) "
        "WHERE date >= CURRENT_DATE - INTERVAL '30 days'"
    )

    # High-volume stocks (for active trading analysis)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_stocks_pricecache_high_volume "
        "ON stocks_pricecache (stock_id, date DESC, volume DESC) "
        "WHERE volume > 1000000"
    )

    print("Performance optimization indexes created successfully!")


def downgrade():
    """Remove performance optimization indexes."""

    print("Removing performance optimization indexes...")

    # Stock data indexes
    op.drop_index("idx_stocks_pricecache_stock_date_range", "stocks_pricecache")
    op.drop_index("idx_stocks_pricecache_volume_desc", "stocks_pricecache")
    op.drop_index("idx_stocks_pricecache_close_price", "stocks_pricecache")

    # Stock lookup indexes
    op.execute("DROP INDEX IF EXISTS idx_stocks_stock_ticker_lower")
    op.drop_index("idx_stocks_stock_sector", "stocks_stock")
    op.drop_index("idx_stocks_stock_industry", "stocks_stock")
    op.drop_index("idx_stocks_stock_exchange", "stocks_stock")

    # Screening indexes
    op.drop_index("idx_stocks_maverickstocks_score_desc", "stocks_maverickstocks")
    op.drop_index("idx_stocks_maverickstocks_rank_asc", "stocks_maverickstocks")
    op.drop_index("idx_stocks_maverickstocks_date_analyzed", "stocks_maverickstocks")
    op.drop_index("idx_stocks_maverickstocks_score_date", "stocks_maverickstocks")

    op.drop_index(
        "idx_stocks_maverickbearstocks_score_desc", "stocks_maverickbearstocks"
    )
    op.drop_index(
        "idx_stocks_maverickbearstocks_date_analyzed", "stocks_maverickbearstocks"
    )

    op.drop_index(
        "idx_stocks_supply_demand_breakouts_momentum_score_desc",
        "stocks_supply_demand_breakouts",
    )
    op.drop_index(
        "idx_stocks_supply_demand_breakouts_date_analyzed",
        "stocks_supply_demand_breakouts",
    )
    op.drop_index(
        "idx_stocks_supply_demand_breakouts_momentum_date",
        "stocks_supply_demand_breakouts",
    )

    # Authentication indexes
    op.drop_index("idx_mcp_api_keys_key_hash", "mcp_api_keys")
    op.drop_index("idx_mcp_api_keys_active_expires", "mcp_api_keys")
    op.drop_index("idx_mcp_api_keys_user_id_active", "mcp_api_keys")

    op.drop_index("idx_mcp_refresh_tokens_token_hash", "mcp_refresh_tokens")
    op.drop_index("idx_mcp_refresh_tokens_user_active", "mcp_refresh_tokens")

    op.drop_index("idx_mcp_requests_user_timestamp", "mcp_requests")
    op.drop_index("idx_mcp_requests_tool_name", "mcp_requests")
    op.drop_index("idx_mcp_requests_success_timestamp", "mcp_requests")

    # Audit logging indexes
    op.drop_index("idx_mcp_audit_logs_user_timestamp", "mcp_audit_logs")
    op.drop_index("idx_mcp_audit_logs_action", "mcp_audit_logs")
    op.drop_index("idx_mcp_audit_logs_ip_timestamp", "mcp_audit_logs")

    # Partial indexes
    op.execute("DROP INDEX IF EXISTS idx_mcp_users_active_email")
    op.execute("DROP INDEX IF EXISTS idx_stocks_pricecache_recent")
    op.execute("DROP INDEX IF EXISTS idx_stocks_pricecache_high_volume")
    print("Performance optimization indexes removed.")
