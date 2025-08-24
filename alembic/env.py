"""
Alembic environment configuration for Maverick-MCP.

This file configures Alembic to work with the existing Django database,
managing only tables with the mcp_ prefix.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models
from maverick_mcp.data.models import Base as DataBase

# Use data models metadata (auth removed for personal version)
combined_metadata = DataBase.metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get database URL from environment or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("POSTGRES_URL", "postgresql://localhost/local_production_snapshot"),
)

# Override sqlalchemy.url in alembic.ini
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# add your model's MetaData object here
# for 'autogenerate' support
# Use the combined metadata from both Base objects
target_metadata = combined_metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def include_object(object, name, type_, reflected, compare_to):
    """
    Include only MCP-prefixed tables and stock-related tables.

    This ensures Alembic only manages tables that belong to Maverick-MCP,
    not Django tables.
    """
    if type_ == "table":
        # Include MCP tables and stock tables
        return (
            name.startswith("mcp_")
            or name.startswith("stocks_")
            or name
            in ["maverick_stocks", "maverick_bear_stocks", "supply_demand_breakouts"]
        )
    elif type_ in [
        "index",
        "unique_constraint",
        "foreign_key_constraint",
        "check_constraint",
    ]:
        # Include indexes and constraints for our tables
        if hasattr(object, "table") and object.table is not None:
            table_name = object.table.name
            return (
                table_name.startswith("mcp_")
                or table_name.startswith("stocks_")
                or table_name
                in [
                    "maverick_stocks",
                    "maverick_bear_stocks",
                    "supply_demand_breakouts",
                ]
            )
        # For reflected objects, check the table name in the name
        return any(
            name.startswith(prefix)
            for prefix in [
                "idx_mcp_",
                "uq_mcp_",
                "fk_mcp_",
                "ck_mcp_",
                "idx_stocks_",
                "uq_stocks_",
                "fk_stocks_",
                "ck_stocks_",
                "ck_pricecache_",
                "ck_maverick_",
                "ck_supply_demand_",
            ]
        )
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
