"""Rename tables to Supply/Demand terminology

Revision ID: 009_rename_to_supply_demand
Revises: 008_performance_optimization_indexes
Create Date: 2025-01-27

This migration renames all database objects to use
supply/demand market structure terminology, removing trademarked references.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "009_rename_to_supply_demand"
down_revision = "008_performance_optimization_indexes"
branch_labels = None
depends_on = None


def upgrade():
    """Rename tables and indexes to supply/demand terminology."""

    # Check if we're using PostgreSQL or SQLite
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        # PostgreSQL supports proper RENAME operations

        # 1. Rename the main table
        op.rename_table("stocks_minervinistocks", "stocks_supply_demand_breakouts")

        # 2. Rename indexes
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_minervinistocks_rs_rating_desc RENAME TO idx_stocks_supply_demand_breakouts_rs_rating_desc"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_minervinistocks_date_analyzed RENAME TO idx_stocks_supply_demand_breakouts_date_analyzed"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_minervinistocks_rs_date RENAME TO idx_stocks_supply_demand_breakouts_rs_date"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_minervini_stocks_rs_rating RENAME TO idx_supply_demand_breakouts_rs_rating"
        )

        # 3. Update any foreign key constraints if they exist
        # Note: Adjust these based on your actual foreign key relationships
        op.execute("""
            DO $$
            BEGIN
                -- Check if constraint exists before trying to rename
                IF EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE constraint_name = 'fk_minervinistocks_symbol'
                ) THEN
                    ALTER TABLE stocks_supply_demand_breakouts
                    RENAME CONSTRAINT fk_minervinistocks_symbol TO fk_supply_demand_breakouts_symbol;
                END IF;
            END $$;
        """)

    elif dialect_name == "sqlite":
        # SQLite doesn't support RENAME operations well, need to recreate

        # 1. Create new table with same structure
        op.create_table(
            "stocks_supply_demand_breakouts",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(10), nullable=False),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            sa.Column("rs_rating", sa.Integer(), nullable=True),
            sa.Column("price", sa.Float(), nullable=True),
            sa.Column("volume", sa.BigInteger(), nullable=True),
            sa.Column("meets_criteria", sa.Boolean(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "symbol", "date_analyzed", name="uq_supply_demand_breakouts_symbol_date"
            ),
        )

        # 2. Copy data from old table to new
        op.execute("""
            INSERT INTO stocks_supply_demand_breakouts
            SELECT * FROM stocks_minervinistocks
        """)

        # 3. Drop old table
        op.drop_table("stocks_minervinistocks")

        # 4. Create indexes on new table
        op.create_index(
            "idx_stocks_supply_demand_breakouts_rs_rating_desc",
            "stocks_supply_demand_breakouts",
            ["rs_rating"],
            postgresql_using="btree",
            postgresql_ops={"rs_rating": "DESC"},
        )
        op.create_index(
            "idx_stocks_supply_demand_breakouts_date_analyzed",
            "stocks_supply_demand_breakouts",
            ["date_analyzed"],
        )
        op.create_index(
            "idx_stocks_supply_demand_breakouts_rs_date",
            "stocks_supply_demand_breakouts",
            ["symbol", "date_analyzed"],
        )

    # Log successful migration
    print("✅ Successfully renamed tables to Supply/Demand Breakout terminology")
    print("   - stocks_minervinistocks → stocks_supply_demand_breakouts")
    print("   - All related indexes have been renamed")


def downgrade():
    """Revert table names back to original terminology."""

    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        # Rename table back
        op.rename_table("stocks_supply_demand_breakouts", "stocks_minervinistocks")

        # Rename indexes back
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_supply_demand_breakouts_rs_rating_desc RENAME TO idx_stocks_minervinistocks_rs_rating_desc"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_supply_demand_breakouts_date_analyzed RENAME TO idx_stocks_minervinistocks_date_analyzed"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_stocks_supply_demand_breakouts_rs_date RENAME TO idx_stocks_minervinistocks_rs_date"
        )
        op.execute(
            "ALTER INDEX IF EXISTS idx_supply_demand_breakouts_rs_rating RENAME TO idx_minervini_stocks_rs_rating"
        )

        # Rename constraints back
        op.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.table_constraints
                    WHERE constraint_name = 'fk_supply_demand_breakouts_symbol'
                ) THEN
                    ALTER TABLE stocks_minervinistocks
                    RENAME CONSTRAINT fk_supply_demand_breakouts_symbol TO fk_minervinistocks_symbol;
                END IF;
            END $$;
        """)

    elif dialect_name == "sqlite":
        # Create old table structure
        op.create_table(
            "stocks_minervinistocks",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("symbol", sa.String(10), nullable=False),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            sa.Column("rs_rating", sa.Integer(), nullable=True),
            sa.Column("price", sa.Float(), nullable=True),
            sa.Column("volume", sa.BigInteger(), nullable=True),
            sa.Column("meets_criteria", sa.Boolean(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "symbol", "date_analyzed", name="uq_minervinistocks_symbol_date"
            ),
        )

        # Copy data back
        op.execute("""
            INSERT INTO stocks_minervinistocks
            SELECT * FROM stocks_supply_demand_breakouts
        """)

        # Drop new table
        op.drop_table("stocks_supply_demand_breakouts")

        # Recreate old indexes
        op.create_index(
            "idx_stocks_minervinistocks_rs_rating_desc",
            "stocks_minervinistocks",
            ["rs_rating"],
        )
        op.create_index(
            "idx_stocks_minervinistocks_date_analyzed",
            "stocks_minervinistocks",
            ["date_analyzed"],
        )
        op.create_index(
            "idx_stocks_minervinistocks_rs_date",
            "stocks_minervinistocks",
            ["symbol", "date_analyzed"],
        )
