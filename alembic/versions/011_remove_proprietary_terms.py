"""Remove proprietary terminology from columns

Revision ID: 011_remove_proprietary_terms
Revises: 010_self_contained_schema
Create Date: 2025-01-10

This migration removes proprietary terminology from database columns:
- rs_rating → momentum_score (more descriptive of what it measures)
- vcp_status → consolidation_status (generic pattern description)

Updates all related indexes and handles both PostgreSQL and SQLite databases.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers
revision = "011_remove_proprietary_terms"
down_revision = "010_self_contained_schema"
branch_labels = None
depends_on = None


def upgrade():
    """Remove proprietary terminology from columns."""

    # Check if we're using PostgreSQL or SQLite
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        print("🗃️  PostgreSQL: Renaming columns and indexes...")

        # 1. Rename columns in mcp_maverick_stocks
        print("   📊 Updating mcp_maverick_stocks...")
        op.alter_column(
            "mcp_maverick_stocks", "rs_rating", new_column_name="momentum_score"
        )
        op.alter_column(
            "mcp_maverick_stocks", "vcp_status", new_column_name="consolidation_status"
        )

        # 2. Rename columns in mcp_maverick_bear_stocks
        print("   🐻 Updating mcp_maverick_bear_stocks...")
        op.alter_column(
            "mcp_maverick_bear_stocks", "rs_rating", new_column_name="momentum_score"
        )
        op.alter_column(
            "mcp_maverick_bear_stocks",
            "vcp_status",
            new_column_name="consolidation_status",
        )

        # 3. Rename columns in mcp_supply_demand_breakouts
        print("   📈 Updating mcp_supply_demand_breakouts...")
        op.alter_column(
            "mcp_supply_demand_breakouts", "rs_rating", new_column_name="momentum_score"
        )
        op.alter_column(
            "mcp_supply_demand_breakouts",
            "vcp_status",
            new_column_name="consolidation_status",
        )

        # 4. Rename indexes to use new column names
        print("   🔍 Updating indexes...")
        op.execute(
            "ALTER INDEX IF EXISTS mcp_maverick_stocks_rs_rating_idx RENAME TO mcp_maverick_stocks_momentum_score_idx"
        )
        op.execute(
            "ALTER INDEX IF EXISTS mcp_maverick_bear_stocks_rs_rating_idx RENAME TO mcp_maverick_bear_stocks_momentum_score_idx"
        )
        op.execute(
            "ALTER INDEX IF EXISTS mcp_supply_demand_breakouts_rs_rating_idx RENAME TO mcp_supply_demand_breakouts_momentum_score_idx"
        )

        # 5. Update any legacy indexes that might still exist
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_supply_demand_breakouts_rs_rating_desc RENAME TO idx_stocks_supply_demand_breakouts_momentum_score_desc"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_supply_demand_breakouts_rs_rating RENAME TO idx_supply_demand_breakouts_momentum_score"
        )

    elif dialect_name == "sqlite":
        print("🗃️  SQLite: Recreating tables with new column names...")

        # SQLite doesn't support column renaming well, need to recreate tables

        # 1. Recreate mcp_maverick_stocks table
        print("   📊 Recreating mcp_maverick_stocks...")
        op.rename_table("mcp_maverick_stocks", "mcp_maverick_stocks_old")

        op.create_table(
            "mcp_maverick_stocks",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
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
            sa.Column("momentum_score", sa.Numeric(5, 2), default=0),  # was rs_rating
            sa.Column("avg_vol_30d", sa.Numeric(15, 2), default=0),
            sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
            sa.Column("atr", sa.Numeric(12, 4), default=0),
            # Pattern Analysis
            sa.Column("pattern_type", sa.String(50)),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("consolidation_status", sa.String(50)),  # was vcp_status
            sa.Column("entry_signal", sa.String(50)),
            # Scoring
            sa.Column("compression_score", sa.Integer(), default=0),
            sa.Column("pattern_detected", sa.Integer(), default=0),
            sa.Column("combined_score", sa.Integer(), default=0),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

        # Copy data with column mapping
        op.execute("""
            INSERT INTO mcp_maverick_stocks
            SELECT
                id, stock_id, date_analyzed, open_price, high_price, low_price, close_price, volume,
                ema_21, sma_50, sma_150, sma_200, rs_rating, avg_vol_30d, adr_pct, atr,
                pattern_type, squeeze_status, vcp_status, entry_signal,
                compression_score, pattern_detected, combined_score, created_at, updated_at
            FROM mcp_maverick_stocks_old
        """)

        op.drop_table("mcp_maverick_stocks_old")

        # Create indexes for maverick stocks
        op.create_index(
            "mcp_maverick_stocks_combined_score_idx",
            "mcp_maverick_stocks",
            ["combined_score"],
        )
        op.create_index(
            "mcp_maverick_stocks_momentum_score_idx",
            "mcp_maverick_stocks",
            ["momentum_score"],
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

        # 2. Recreate mcp_maverick_bear_stocks table
        print("   🐻 Recreating mcp_maverick_bear_stocks...")
        op.rename_table("mcp_maverick_bear_stocks", "mcp_maverick_bear_stocks_old")

        op.create_table(
            "mcp_maverick_bear_stocks",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
            ),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            # OHLCV Data
            sa.Column("open_price", sa.Numeric(12, 4), default=0),
            sa.Column("high_price", sa.Numeric(12, 4), default=0),
            sa.Column("low_price", sa.Numeric(12, 4), default=0),
            sa.Column("close_price", sa.Numeric(12, 4), default=0),
            sa.Column("volume", sa.BigInteger(), default=0),
            # Technical Indicators
            sa.Column("momentum_score", sa.Numeric(5, 2), default=0),  # was rs_rating
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
            sa.Column("consolidation_status", sa.String(50)),  # was vcp_status
            # Scoring
            sa.Column("score", sa.Integer(), default=0),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

        # Copy data with column mapping
        op.execute("""
            INSERT INTO mcp_maverick_bear_stocks
            SELECT
                id, stock_id, date_analyzed, open_price, high_price, low_price, close_price, volume,
                rs_rating, ema_21, sma_50, sma_200, rsi_14,
                macd, macd_signal, macd_histogram, dist_days_20, adr_pct, atr_contraction, atr, avg_vol_30d, big_down_vol,
                squeeze_status, vcp_status, score, created_at, updated_at
            FROM mcp_maverick_bear_stocks_old
        """)

        op.drop_table("mcp_maverick_bear_stocks_old")

        # Create indexes for bear stocks
        op.create_index(
            "mcp_maverick_bear_stocks_score_idx", "mcp_maverick_bear_stocks", ["score"]
        )
        op.create_index(
            "mcp_maverick_bear_stocks_momentum_score_idx",
            "mcp_maverick_bear_stocks",
            ["momentum_score"],
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

        # 3. Recreate mcp_supply_demand_breakouts table
        print("   📈 Recreating mcp_supply_demand_breakouts...")
        op.rename_table(
            "mcp_supply_demand_breakouts", "mcp_supply_demand_breakouts_old"
        )

        op.create_table(
            "mcp_supply_demand_breakouts",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
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
            sa.Column("momentum_score", sa.Numeric(5, 2), default=0),  # was rs_rating
            sa.Column("avg_volume_30d", sa.Numeric(15, 2), default=0),
            sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
            sa.Column("atr", sa.Numeric(12, 4), default=0),
            # Pattern Analysis
            sa.Column("pattern_type", sa.String(50)),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("consolidation_status", sa.String(50)),  # was vcp_status
            sa.Column("entry_signal", sa.String(50)),
            # Supply/Demand Analysis
            sa.Column("accumulation_rating", sa.Numeric(5, 2), default=0),
            sa.Column("distribution_rating", sa.Numeric(5, 2), default=0),
            sa.Column("breakout_strength", sa.Numeric(5, 2), default=0),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

        # Copy data with column mapping
        op.execute("""
            INSERT INTO mcp_supply_demand_breakouts
            SELECT
                id, stock_id, date_analyzed, open_price, high_price, low_price, close_price, volume,
                ema_21, sma_50, sma_150, sma_200, rs_rating, avg_volume_30d, adr_pct, atr,
                pattern_type, squeeze_status, vcp_status, entry_signal,
                accumulation_rating, distribution_rating, breakout_strength, created_at, updated_at
            FROM mcp_supply_demand_breakouts_old
        """)

        op.drop_table("mcp_supply_demand_breakouts_old")

        # Create indexes for supply/demand breakouts
        op.create_index(
            "mcp_supply_demand_breakouts_momentum_score_idx",
            "mcp_supply_demand_breakouts",
            ["momentum_score"],
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

    # Log successful migration
    print("✅ Successfully removed proprietary terminology from database columns:")
    print("   - rs_rating → momentum_score (more descriptive)")
    print("   - vcp_status → consolidation_status (generic pattern description)")
    print("   - All related indexes have been updated")


def downgrade():
    """Revert column names back to original proprietary terms."""

    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        print("🗃️  PostgreSQL: Reverting column names...")

        # 1. Revert indexes first
        print("   🔍 Reverting indexes...")
        op.execute(
            "ALTER INDEX IF EXISTS mcp_maverick_stocks_momentum_score_idx RENAME TO mcp_maverick_stocks_rs_rating_idx"
        )
        op.execute(
            "ALTER INDEX IF EXISTS mcp_maverick_bear_stocks_momentum_score_idx RENAME TO mcp_maverick_bear_stocks_rs_rating_idx"
        )
        op.execute(
            "ALTER INDEX IF EXISTS mcp_supply_demand_breakouts_momentum_score_idx RENAME TO mcp_supply_demand_breakouts_rs_rating_idx"
        )

        # Revert any legacy indexes
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_supply_demand_breakouts_momentum_score_desc RENAME TO idx_stocks_supply_demand_breakouts_rs_rating_desc"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_supply_demand_breakouts_momentum_score RENAME TO idx_supply_demand_breakouts_rs_rating"
        )

        # 2. Revert columns in mcp_maverick_stocks
        print("   📊 Reverting mcp_maverick_stocks...")
        op.alter_column(
            "mcp_maverick_stocks", "momentum_score", new_column_name="rs_rating"
        )
        op.alter_column(
            "mcp_maverick_stocks", "consolidation_status", new_column_name="vcp_status"
        )

        # 3. Revert columns in mcp_maverick_bear_stocks
        print("   🐻 Reverting mcp_maverick_bear_stocks...")
        op.alter_column(
            "mcp_maverick_bear_stocks", "momentum_score", new_column_name="rs_rating"
        )
        op.alter_column(
            "mcp_maverick_bear_stocks",
            "consolidation_status",
            new_column_name="vcp_status",
        )

        # 4. Revert columns in mcp_supply_demand_breakouts
        print("   📈 Reverting mcp_supply_demand_breakouts...")
        op.alter_column(
            "mcp_supply_demand_breakouts", "momentum_score", new_column_name="rs_rating"
        )
        op.alter_column(
            "mcp_supply_demand_breakouts",
            "consolidation_status",
            new_column_name="vcp_status",
        )

    elif dialect_name == "sqlite":
        print("🗃️  SQLite: Recreating tables with original column names...")

        # SQLite: Recreate tables with original names

        # 1. Recreate mcp_maverick_stocks table with original columns
        print("   📊 Recreating mcp_maverick_stocks...")
        op.rename_table("mcp_maverick_stocks", "mcp_maverick_stocks_new")

        op.create_table(
            "mcp_maverick_stocks",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
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
            sa.Column("rs_rating", sa.Numeric(5, 2), default=0),  # restored
            sa.Column("avg_vol_30d", sa.Numeric(15, 2), default=0),
            sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
            sa.Column("atr", sa.Numeric(12, 4), default=0),
            # Pattern Analysis
            sa.Column("pattern_type", sa.String(50)),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("vcp_status", sa.String(50)),  # restored
            sa.Column("entry_signal", sa.String(50)),
            # Scoring
            sa.Column("compression_score", sa.Integer(), default=0),
            sa.Column("pattern_detected", sa.Integer(), default=0),
            sa.Column("combined_score", sa.Integer(), default=0),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

        # Copy data back with column mapping
        op.execute("""
            INSERT INTO mcp_maverick_stocks
            SELECT
                id, stock_id, date_analyzed, open_price, high_price, low_price, close_price, volume,
                ema_21, sma_50, sma_150, sma_200, momentum_score, avg_vol_30d, adr_pct, atr,
                pattern_type, squeeze_status, consolidation_status, entry_signal,
                compression_score, pattern_detected, combined_score, created_at, updated_at
            FROM mcp_maverick_stocks_new
        """)

        op.drop_table("mcp_maverick_stocks_new")

        # Create original indexes
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

        # 2. Recreate mcp_maverick_bear_stocks with original columns
        print("   🐻 Recreating mcp_maverick_bear_stocks...")
        op.rename_table("mcp_maverick_bear_stocks", "mcp_maverick_bear_stocks_new")

        op.create_table(
            "mcp_maverick_bear_stocks",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
            ),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            # OHLCV Data
            sa.Column("open_price", sa.Numeric(12, 4), default=0),
            sa.Column("high_price", sa.Numeric(12, 4), default=0),
            sa.Column("low_price", sa.Numeric(12, 4), default=0),
            sa.Column("close_price", sa.Numeric(12, 4), default=0),
            sa.Column("volume", sa.BigInteger(), default=0),
            # Technical Indicators
            sa.Column("rs_rating", sa.Numeric(5, 2), default=0),  # restored
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
            sa.Column("vcp_status", sa.String(50)),  # restored
            # Scoring
            sa.Column("score", sa.Integer(), default=0),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

        # Copy data back
        op.execute("""
            INSERT INTO mcp_maverick_bear_stocks
            SELECT
                id, stock_id, date_analyzed, open_price, high_price, low_price, close_price, volume,
                momentum_score, ema_21, sma_50, sma_200, rsi_14,
                macd, macd_signal, macd_histogram, dist_days_20, adr_pct, atr_contraction, atr, avg_vol_30d, big_down_vol,
                squeeze_status, consolidation_status, score, created_at, updated_at
            FROM mcp_maverick_bear_stocks_new
        """)

        op.drop_table("mcp_maverick_bear_stocks_new")

        # Create original indexes
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

        # 3. Recreate mcp_supply_demand_breakouts with original columns
        print("   📈 Recreating mcp_supply_demand_breakouts...")
        op.rename_table(
            "mcp_supply_demand_breakouts", "mcp_supply_demand_breakouts_new"
        )

        op.create_table(
            "mcp_supply_demand_breakouts",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
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
            sa.Column("rs_rating", sa.Numeric(5, 2), default=0),  # restored
            sa.Column("avg_volume_30d", sa.Numeric(15, 2), default=0),
            sa.Column("adr_pct", sa.Numeric(5, 2), default=0),
            sa.Column("atr", sa.Numeric(12, 4), default=0),
            # Pattern Analysis
            sa.Column("pattern_type", sa.String(50)),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("vcp_status", sa.String(50)),  # restored
            sa.Column("entry_signal", sa.String(50)),
            # Supply/Demand Analysis
            sa.Column("accumulation_rating", sa.Numeric(5, 2), default=0),
            sa.Column("distribution_rating", sa.Numeric(5, 2), default=0),
            sa.Column("breakout_strength", sa.Numeric(5, 2), default=0),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

        # Copy data back
        op.execute("""
            INSERT INTO mcp_supply_demand_breakouts
            SELECT
                id, stock_id, date_analyzed, open_price, high_price, low_price, close_price, volume,
                ema_21, sma_50, sma_150, sma_200, momentum_score, avg_volume_30d, adr_pct, atr,
                pattern_type, squeeze_status, consolidation_status, entry_signal,
                accumulation_rating, distribution_rating, breakout_strength, created_at, updated_at
            FROM mcp_supply_demand_breakouts_new
        """)

        op.drop_table("mcp_supply_demand_breakouts_new")

        # Create original indexes
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

    print("✅ Successfully reverted column names back to original proprietary terms")
