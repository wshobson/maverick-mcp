#!/usr/bin/env python3
"""
Database migration script for MaverickMCP.

This script initializes the SQLite database with all necessary tables
and ensures the schema is properly set up for the application.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# noqa: E402 - imports must come after sys.path modification
from sqlalchemy import create_engine, text  # noqa: E402

from maverick_mcp.data.models import Base  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.migrate")


def get_database_url() -> str:
    """Get the database URL from environment or settings."""
    # Use environment variable if set, otherwise default to SQLite
    database_url = os.getenv("DATABASE_URL") or "sqlite:///maverick_mcp.db"
    logger.info(f"Using database URL: {database_url}")
    return database_url


def create_database_if_not_exists(database_url: str) -> None:
    """Create database file if it doesn't exist (for SQLite)."""
    if database_url.startswith("sqlite:///"):
        # Extract the file path from the URL
        db_path = database_url.replace("sqlite:///", "")
        if db_path != ":memory:" and not db_path.startswith("./"):
            # Handle absolute paths
            db_file = Path(db_path)
        else:
            # Handle relative paths
            db_file = Path(db_path.lstrip("./"))

        # Create directory if it doesn't exist
        db_file.parent.mkdir(parents=True, exist_ok=True)

        if not db_file.exists():
            logger.info(f"Creating SQLite database file: {db_file}")
            # Create empty file
            db_file.touch()
        else:
            logger.info(f"SQLite database already exists: {db_file}")


def test_database_connection(database_url: str) -> bool:
    """Test database connection."""
    try:
        logger.info("Testing database connection...")
        engine = create_engine(database_url, echo=False)

        with engine.connect() as conn:
            if database_url.startswith("sqlite"):
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.scalar()
                logger.info(f"Connected to SQLite version: {version}")
            elif database_url.startswith("postgresql"):
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"Connected to PostgreSQL: {version[:50]}...")
            else:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection successful")

        engine.dispose()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def create_tables(database_url: str) -> bool:
    """Create all tables using SQLAlchemy."""
    try:
        logger.info("Creating database tables...")
        engine = create_engine(database_url, echo=False)

        # Create all tables
        Base.metadata.create_all(bind=engine)

        # Verify tables were created
        with engine.connect() as conn:
            if database_url.startswith("sqlite"):
                result = conn.execute(
                    text("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name LIKE 'mcp_%'
                    ORDER BY name
                """)
                )
            else:
                result = conn.execute(
                    text("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema='public' AND table_name LIKE 'mcp_%'
                    ORDER BY table_name
                """)
                )

            tables = [row[0] for row in result.fetchall()]
            logger.info(f"Created {len(tables)} tables: {', '.join(tables)}")

            # Verify expected tables exist
            expected_tables = {
                "mcp_stocks",
                "mcp_price_cache",
                "mcp_maverick_stocks",
                "mcp_maverick_bear_stocks",
                "mcp_supply_demand_breakouts",
                "mcp_technical_cache",
            }

            missing_tables = expected_tables - set(tables)
            if missing_tables:
                logger.warning(f"Missing expected tables: {missing_tables}")
            else:
                logger.info("All expected tables created successfully")

        engine.dispose()
        return True

    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        return False


def verify_schema(database_url: str) -> bool:
    """Verify the database schema is correct."""
    try:
        logger.info("Verifying database schema...")
        engine = create_engine(database_url, echo=False)

        with engine.connect() as conn:
            # Check that we can query each main table
            test_queries = [
                ("mcp_stocks", "SELECT COUNT(*) FROM mcp_stocks"),
                ("mcp_price_cache", "SELECT COUNT(*) FROM mcp_price_cache"),
                ("mcp_maverick_stocks", "SELECT COUNT(*) FROM mcp_maverick_stocks"),
                (
                    "mcp_maverick_bear_stocks",
                    "SELECT COUNT(*) FROM mcp_maverick_bear_stocks",
                ),
                (
                    "mcp_supply_demand_breakouts",
                    "SELECT COUNT(*) FROM mcp_supply_demand_breakouts",
                ),
                ("mcp_technical_cache", "SELECT COUNT(*) FROM mcp_technical_cache"),
            ]

            for table_name, query in test_queries:
                try:
                    result = conn.execute(text(query))
                    count = result.scalar()
                    logger.info(f"✓ {table_name}: {count} records")
                except Exception as e:
                    logger.error(f"✗ {table_name}: {e}")
                    return False

        engine.dispose()
        logger.info("Schema verification completed successfully")
        return True

    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return False


def main():
    """Main migration function."""
    logger.info("Starting MaverickMCP database migration...")

    # Get database URL
    database_url = get_database_url()

    # Create database file if needed (SQLite)
    create_database_if_not_exists(database_url)

    # Test connection
    if not test_database_connection(database_url):
        logger.error("Database connection failed. Exiting.")
        return False

    # Create tables
    if not create_tables(database_url):
        logger.error("Table creation failed. Exiting.")
        return False

    # Verify schema
    if not verify_schema(database_url):
        logger.error("Schema verification failed. Exiting.")
        return False

    logger.info("✅ Database migration completed successfully!")
    logger.info(f"Database ready at: {database_url}")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
