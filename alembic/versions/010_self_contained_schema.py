"""Create self-contained schema with mcp_ prefixed tables

Revision ID: 010_self_contained_schema
Revises: 009_rename_to_supply_demand
Create Date: 2025-01-31

This migration creates a complete self-contained schema for maverick-mcp
with all tables prefixed with 'mcp_' to avoid conflicts with external systems.

Tables created:
- mcp_stocks: Master stock information
- mcp_price_cache: Historical price data
- mcp_maverick_stocks: Maverick screening results
- mcp_maverick_bear_stocks: Bear market screening results
- mcp_supply_demand_breakouts: Supply/demand analysis
- mcp_technical_cache: Technical indicator cache
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers
revision = "010_self_contained_schema"
down_revision = "009_rename_to_supply_demand"
branch_labels = None
depends_on = None


def upgrade():
    """Create self-contained schema with all mcp_ prefixed tables."""

    # Check if we're using PostgreSQL or SQLite
    op.get_bind()

    print("🚀 Creating self-contained maverick-mcp schema...")

    # 1. Create mcp_stocks table (master stock data)
    print("📊 Creating mcp_stocks table...")
    op.create_table(
        "mcp_stocks",
        sa.Column("stock_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "ticker_symbol", sa.String(10), nullable=False, unique=True, index=True
        ),
        sa.Column("company_name", sa.String(255)),
        sa.Column("description", sa.Text()),
        sa.Column("sector", sa.String(100)),
        sa.Column("industry", sa.String(100)),
        sa.Column("exchange", sa.String(50)),
        sa.Column("country", sa.String(50)),
        sa.Column("currency", sa.String(3)),
        sa.Column("isin", sa.String(12)),
        sa.Column("market_cap", sa.BigInteger()),
        sa.Column("shares_outstanding", sa.BigInteger()),
        sa.Column("is_etf", sa.Boolean(), default=False),
        sa.Column("is_active", sa.Boolean(), default=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes for mcp_stocks
    op.create_index("mcp_stocks_ticker_idx", "mcp_stocks", ["ticker_symbol"])
    op.create_index("mcp_stocks_sector_idx", "mcp_stocks", ["sector"])
    op.create_index("mcp_stocks_exchange_idx", "mcp_stocks", ["exchange"])

    # 2. Create mcp_price_cache table
    print("💰 Creating mcp_price_cache table...")
    op.create_table(
        "mcp_price_cache",
        sa.Column("price_cache_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "stock_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("mcp_stocks.stock_id"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("open_price", sa.Numeric(12, 4)),
        sa.Column("high_price", sa.Numeric(12, 4)),
        sa.Column("low_price", sa.Numeric(12, 4)),
        sa.Column("close_price", sa.Numeric(12, 4)),
        sa.Column("volume", sa.BigInteger()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create unique constraint and indexes for price cache
    op.create_unique_constraint(
        "mcp_price_cache_stock_date_unique", "mcp_price_cache", ["stock_id", "date"]
    )
    op.create_index(
        "mcp_price_cache_stock_id_date_idx", "mcp_price_cache", ["stock_id", "date"]
    )
    op.create_index("mcp_price_cache_date_idx", "mcp_price_cache", ["date"])

    # 3. Create mcp_maverick_stocks table
    print("🎯 Creating mcp_maverick_stocks table...")
    op.create_table(
        "mcp_maverick_stocks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "stock_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("mcp_stocks.stock_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("date_analyzed", sa.Date(), nullable=False),
        # OHLCV Data
        sa.Column("open_price", sa.Numeric(12, 4), default=0),
        sa.Column("high_price", sa.Numeric(12, 4), default=0),
        sa.Column("low_price", sa.Numeric(12, 4), default=0),
        sa.Column("close_price", sa.Numeric(12, 4), default=0),
        sa.Column("volume", sa.BigInteger(), default=0),
        # Technical Indicators
        sa.Column("ema_21", sa.Numeric(12, 4), default=0),
        sa.Column("sma_50", sa.Numeric(12, 4), default=0),
        sa.Column("sma_150", sa.Numeric(12, 4), default=0),
        sa.Column("sma_200", sa.Numeric(12, 4), default=0),
        sa.Column("rs_rating", sa.Numeric(5, 2), default=0),
        sa.Column("avg_vol_30d", sa.Numeric(15, 2), default=0),
        sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
        sa.Column("atr", sa.Numeric(12, 4), default=0),
        # Pattern Analysis
        sa.Column("pattern_type", sa.String(50)),
        sa.Column("squeeze_status", sa.String(50)),
        sa.Column("vcp_status", sa.String(50)),
        sa.Column("entry_signal", sa.String(50)),
        # Scoring
        sa.Column("compression_score", sa.Integer(), default=0),
        sa.Column("pattern_detected", sa.Integer(), default=0),
        sa.Column("combined_score", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes for maverick stocks
    op.create_index(
        "mcp_maverick_stocks_combined_score_idx",
        "mcp_maverick_stocks",
        ["combined_score"],
    )
    op.create_index(
        "mcp_maverick_stocks_rs_rating_idx", "mcp_maverick_stocks", ["rs_rating"]
    )
    op.create_index(
        "mcp_maverick_stocks_date_analyzed_idx",
        "mcp_maverick_stocks",
        ["date_analyzed"],
    )
    op.create_index(
        "mcp_maverick_stocks_stock_date_idx",
        "mcp_maverick_stocks",
        ["stock_id", "date_analyzed"],
    )

    # 4. Create mcp_maverick_bear_stocks table
    print("🐻 Creating mcp_maverick_bear_stocks table...")
    op.create_table(
        "mcp_maverick_bear_stocks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "stock_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("mcp_stocks.stock_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("date_analyzed", sa.Date(), nullable=False),
        # OHLCV Data
        sa.Column("open_price", sa.Numeric(12, 4), default=0),
        sa.Column("high_price", sa.Numeric(12, 4), default=0),
        sa.Column("low_price", sa.Numeric(12, 4), default=0),
        sa.Column("close_price", sa.Numeric(12, 4), default=0),
        sa.Column("volume", sa.BigInteger(), default=0),
        # Technical Indicators
        sa.Column("rs_rating", sa.Numeric(5, 2), default=0),
        sa.Column("ema_21", sa.Numeric(12, 4), default=0),
        sa.Column("sma_50", sa.Numeric(12, 4), default=0),
        sa.Column("sma_200", sa.Numeric(12, 4), default=0),
        sa.Column("rsi_14", sa.Numeric(5, 2), default=0),
        # MACD Indicators
        sa.Column("macd", sa.Numeric(12, 6), default=0),
        sa.Column("macd_signal", sa.Numeric(12, 6), default=0),
        sa.Column("macd_histogram", sa.Numeric(12, 6), default=0),
        # Bear Market Indicators
        sa.Column("dist_days_20", sa.Integer(), default=0),
        sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
        sa.Column("atr_contraction", sa.Boolean(), default=False),
        sa.Column("atr", sa.Numeric(12, 4), default=0),
        sa.Column("avg_vol_30d", sa.Numeric(15, 2), default=0),
        sa.Column("big_down_vol", sa.Boolean(), default=False),
        # Pattern Analysis
        sa.Column("squeeze_status", sa.String(50)),
        sa.Column("vcp_status", sa.String(50)),
        # Scoring
        sa.Column("score", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes for bear stocks
    op.create_index(
        "mcp_maverick_bear_stocks_score_idx", "mcp_maverick_bear_stocks", ["score"]
    )
    op.create_index(
        "mcp_maverick_bear_stocks_rs_rating_idx",
        "mcp_maverick_bear_stocks",
        ["rs_rating"],
    )
    op.create_index(
        "mcp_maverick_bear_stocks_date_analyzed_idx",
        "mcp_maverick_bear_stocks",
        ["date_analyzed"],
    )
    op.create_index(
        "mcp_maverick_bear_stocks_stock_date_idx",
        "mcp_maverick_bear_stocks",
        ["stock_id", "date_analyzed"],
    )

    # 5. Create mcp_supply_demand_breakouts table
    print("📈 Creating mcp_supply_demand_breakouts table...")
    op.create_table(
        "mcp_supply_demand_breakouts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "stock_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("mcp_stocks.stock_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("date_analyzed", sa.Date(), nullable=False),
        # OHLCV Data
        sa.Column("open_price", sa.Numeric(12, 4), default=0),
        sa.Column("high_price", sa.Numeric(12, 4), default=0),
        sa.Column("low_price", sa.Numeric(12, 4), default=0),
        sa.Column("close_price", sa.Numeric(12, 4), default=0),
        sa.Column("volume", sa.BigInteger(), default=0),
        # Technical Indicators
        sa.Column("ema_21", sa.Numeric(12, 4), default=0),
        sa.Column("sma_50", sa.Numeric(12, 4), default=0),
        sa.Column("sma_150", sa.Numeric(12, 4), default=0),
        sa.Column("sma_200", sa.Numeric(12, 4), default=0),
        sa.Column("rs_rating", sa.Numeric(5, 2), default=0),
        sa.Column("avg_volume_30d", sa.Numeric(15, 2), default=0),
        sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
        sa.Column("atr", sa.Numeric(12, 4), default=0),
        # Pattern Analysis
        sa.Column("pattern_type", sa.String(50)),
        sa.Column("squeeze_status", sa.String(50)),
        sa.Column("vcp_status", sa.String(50)),
        sa.Column("entry_signal", sa.String(50)),
        # Supply/Demand Analysis
        sa.Column("accumulation_rating", sa.Numeric(5, 2), default=0),
        sa.Column("distribution_rating", sa.Numeric(5, 2), default=0),
        sa.Column("breakout_strength", sa.Numeric(5, 2), default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes for supply/demand breakouts
    op.create_index(
        "mcp_supply_demand_breakouts_rs_rating_idx",
        "mcp_supply_demand_breakouts",
        ["rs_rating"],
    )
    op.create_index(
        "mcp_supply_demand_breakouts_date_analyzed_idx",
        "mcp_supply_demand_breakouts",
        ["date_analyzed"],
    )
    op.create_index(
        "mcp_supply_demand_breakouts_stock_date_idx",
        "mcp_supply_demand_breakouts",
        ["stock_id", "date_analyzed"],
    )
    op.create_index(
        "mcp_supply_demand_breakouts_ma_filter_idx",
        "mcp_supply_demand_breakouts",
        ["close_price", "sma_50", "sma_150", "sma_200"],
    )

    # 6. Create mcp_technical_cache table
    print("🔧 Creating mcp_technical_cache table...")
    op.create_table(
        "mcp_technical_cache",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "stock_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("mcp_stocks.stock_id"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("indicator_type", sa.String(50), nullable=False),
        # Flexible indicator values
        sa.Column("value", sa.Numeric(20, 8)),
        sa.Column("value_2", sa.Numeric(20, 8)),
        sa.Column("value_3", sa.Numeric(20, 8)),
        # Metadata and parameters
        sa.Column("metadata", sa.Text()),
        sa.Column("period", sa.Integer()),
        sa.Column("parameters", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create unique constraint and indexes for technical cache
    op.create_unique_constraint(
        "mcp_technical_cache_stock_date_indicator_unique",
        "mcp_technical_cache",
        ["stock_id", "date", "indicator_type"],
    )
    op.create_index(
        "mcp_technical_cache_stock_date_idx",
        "mcp_technical_cache",
        ["stock_id", "date"],
    )
    op.create_index(
        "mcp_technical_cache_indicator_idx", "mcp_technical_cache", ["indicator_type"]
    )
    op.create_index("mcp_technical_cache_date_idx", "mcp_technical_cache", ["date"])

    print("✅ Self-contained schema created successfully!")
    print("📋 Tables created:")
    print("   - mcp_stocks (master stock data)")
    print("   - mcp_price_cache (historical prices)")
    print("   - mcp_maverick_stocks (maverick screening)")
    print("   - mcp_maverick_bear_stocks (bear screening)")
    print("   - mcp_supply_demand_breakouts (supply/demand analysis)")
    print("   - mcp_technical_cache (technical indicators)")
    print("🎯 Maverick-MCP is now completely self-contained!")


def downgrade():
    """Drop all self-contained tables."""
    print("⚠️  Dropping self-contained maverick-mcp schema...")

    # Drop tables in reverse order due to foreign key constraints
    tables = [
        "mcp_technical_cache",
        "mcp_supply_demand_breakouts",
        "mcp_maverick_bear_stocks",
        "mcp_maverick_stocks",
        "mcp_price_cache",
        "mcp_stocks",
    ]

    for table in tables:
        print(f"🗑️  Dropping {table}...")
        op.drop_table(table)

    print("✅ Self-contained schema removed!")
