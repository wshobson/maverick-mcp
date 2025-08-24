#!/usr/bin/env python3
"""
Self-contained setup script for Maverick-MCP.

This script sets up a completely self-contained Maverick-MCP installation
with its own database schema, sample data, and validation.

Usage:
    python scripts/setup_self_contained.py --full-setup
    python scripts/setup_self_contained.py --quick-setup
    python scripts/setup_self_contained.py --migrate-only
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from maverick_mcp.config.database_self_contained import (
    get_self_contained_db_config,
    init_self_contained_database,
    run_self_contained_migrations,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("self_contained_setup")


def check_requirements() -> bool:
    """Check if all requirements are met for setup."""
    logger.info("🔍 Checking requirements...")

    # Check environment variables
    required_env = []
    optional_env = {
        "TIINGO_API_TOKEN": "Required for loading market data from Tiingo API",
        "MCP_DATABASE_URL": "Custom database URL (defaults to maverick_mcp database)",
        "POSTGRES_URL": "Alternative database URL",
        "DATABASE_URL": "Fallback database URL",
    }

    missing_required = []
    for env_var in required_env:
        if not os.getenv(env_var):
            missing_required.append(env_var)

    if missing_required:
        logger.error(f"❌ Missing required environment variables: {missing_required}")
        return False

    # Check optional environment variables
    missing_optional = []
    for env_var, description in optional_env.items():
        if not os.getenv(env_var):
            missing_optional.append(f"{env_var}: {description}")

    if missing_optional:
        logger.info("ℹ️  Optional environment variables not set:")
        for var in missing_optional:
            logger.info(f"   - {var}")

    logger.info("✅ Requirements check passed")
    return True


def run_migrations() -> bool:
    """Run database migrations."""
    logger.info("🔄 Running database migrations...")

    try:
        run_self_contained_migrations()
        logger.info("✅ Database migrations completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


def validate_schema() -> bool:
    """Validate the database schema."""
    logger.info("🔍 Validating database schema...")

    try:
        db_config = get_self_contained_db_config()
        if db_config.validate_schema():
            logger.info("✅ Schema validation passed")
            return True
        else:
            logger.error("❌ Schema validation failed")
            return False
    except Exception as e:
        logger.error(f"❌ Schema validation error: {e}")
        return False


def load_sample_data(quick: bool = False) -> bool:
    """Load sample market data."""
    logger.info("📊 Loading sample market data...")

    try:
        # Import here to avoid circular imports
        from load_market_data import TiingoDataLoader

        # Check if Tiingo API token is available
        if not os.getenv("TIINGO_API_TOKEN"):
            logger.warning("⚠️  TIINGO_API_TOKEN not set, skipping market data loading")
            logger.info(
                "   You can load market data later using: python scripts/load_market_data.py"
            )
            return True

        # Determine symbols to load
        if quick:
            symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "JPM",
                "V",
                "PG",
            ]
        else:
            # Load more comprehensive set
            from load_market_data import get_sp500_symbols

            symbols = get_sp500_symbols()

        async def load_data():
            async with TiingoDataLoader() as loader:
                loaded_count = await loader.load_stock_data(symbols)
                return loaded_count

        loaded_count = asyncio.run(load_data())
        logger.info(f"✅ Loaded market data for {loaded_count} stocks")
        return True

    except ImportError as e:
        logger.error(f"❌ Cannot import market data loader: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Market data loading failed: {e}")
        return False


def run_sample_screening(quick: bool = False) -> bool:
    """Run sample stock screening."""
    logger.info("🎯 Running sample stock screening...")

    try:
        # Import here to avoid circular imports
        from datetime import datetime

        from run_stock_screening import StockScreener

        from maverick_mcp.config.database_self_contained import (
            SelfContainedDatabaseSession,
        )
        from maverick_mcp.data.models import MaverickStocks, bulk_insert_screening_data

        async def run_screening():
            screener = StockScreener()
            today = datetime.now().date()

            with SelfContainedDatabaseSession() as session:
                if quick:
                    # Just run Maverick screening
                    results = await screener.run_maverick_screening(session)
                    if results:
                        count = bulk_insert_screening_data(
                            session, MaverickStocks, results, today
                        )
                        return count
                else:
                    # Run all screenings
                    total_count = 0

                    # Maverick screening
                    maverick_results = await screener.run_maverick_screening(session)
                    if maverick_results:
                        count = bulk_insert_screening_data(
                            session, MaverickStocks, maverick_results, today
                        )
                        total_count += count

                    return total_count

            return 0

        count = asyncio.run(run_screening())
        logger.info(f"✅ Completed screening, found {count} candidates")
        return True

    except ImportError as e:
        logger.error(f"❌ Cannot import screening modules: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Sample screening failed: {e}")
        return False


def display_setup_summary() -> None:
    """Display setup summary and next steps."""
    logger.info("📋 Setup Summary:")

    try:
        db_config = get_self_contained_db_config()
        stats = db_config.get_database_stats()

        print("\n📊 Database Statistics:")
        print(f"   Database URL: {stats.get('database_url', 'Unknown')}")
        print(f"   Total Records: {stats.get('total_records', 0)}")

        for table, count in stats.get("tables", {}).items():
            print(f"   {table}: {count}")

    except Exception as e:
        logger.error(f"❌ Could not get database stats: {e}")

    print("\n🎉 Self-contained Maverick-MCP setup completed!")
    print("\n📚 Next Steps:")
    print("   1. Start the MCP server: python start_mcp_server.py")
    print("   2. Load more market data: python scripts/load_market_data.py --sp500")
    print("   3. Run screening: python scripts/run_stock_screening.py --all")
    print("   4. Access the web dashboard: http://localhost:3001")

    print("\n💡 Available Scripts:")
    print("   - scripts/load_market_data.py: Load stock and price data")
    print("   - scripts/run_stock_screening.py: Run screening algorithms")
    print("   - scripts/setup_self_contained.py: This setup script")

    print("\n🔧 Environment Variables:")
    print("   - TIINGO_API_TOKEN: Set to load market data")
    print("   - MCP_DATABASE_URL: Override database URL")
    print("   - DB_POOL_SIZE: Database connection pool size (default: 20)")


async def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup self-contained Maverick-MCP")
    parser.add_argument(
        "--full-setup",
        action="store_true",
        help="Run complete setup with comprehensive data loading",
    )
    parser.add_argument(
        "--quick-setup",
        action="store_true",
        help="Run quick setup with minimal sample data",
    )
    parser.add_argument(
        "--migrate-only", action="store_true", help="Only run database migrations"
    )
    parser.add_argument("--database-url", type=str, help="Override database URL")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip loading market data and screening",
    )

    args = parser.parse_args()

    if not any([args.full_setup, args.quick_setup, args.migrate_only]):
        parser.print_help()
        sys.exit(1)

    print("🚀 Starting Maverick-MCP Self-Contained Setup...")
    print("=" * 60)

    # Step 1: Check requirements
    if not check_requirements():
        sys.exit(1)

    # Step 2: Initialize database
    try:
        logger.info("🗄️  Initializing self-contained database...")
        init_self_contained_database(
            database_url=args.database_url, create_tables=True, validate_schema=True
        )
        logger.info("✅ Database initialization completed")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    # Step 3: Run migrations
    if not run_migrations():
        sys.exit(1)

    # Step 4: Validate schema
    if not validate_schema():
        sys.exit(1)

    # Stop here if migrate-only
    if args.migrate_only:
        logger.info("✅ Migration-only setup completed successfully")
        return

    # Step 5: Load sample data (unless skipped)
    if not args.skip_data:
        quick = args.quick_setup

        if not load_sample_data(quick=quick):
            logger.warning("⚠️  Market data loading failed, but continuing setup")

        # Step 6: Run sample screening
        if not run_sample_screening(quick=quick):
            logger.warning("⚠️  Sample screening failed, but continuing setup")

    # Step 7: Display summary
    display_setup_summary()

    print("\n" + "=" * 60)
    print("🎉 Self-contained Maverick-MCP setup completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
