#!/usr/bin/env python3
"""
Example usage of the Tiingo data loader.

This script demonstrates common usage patterns for loading market data
from Tiingo API into the Maverick-MCP database.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maverick_mcp.data.models import SessionLocal
from scripts.load_tiingo_data import ProgressTracker, TiingoDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_sample_stocks():
    """Load a small sample of stocks for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"Loading sample stocks: {', '.join(symbols)}")

    # Create progress tracker
    progress = ProgressTracker("sample_load_progress.json")
    progress.total_symbols = len(symbols)

    async with TiingoDataLoader(
        batch_size=10, max_concurrent=3, progress_tracker=progress
    ) as loader:
        # Load 1 year of data with indicators
        start_date = "2023-01-01"

        successful, failed = await loader.load_batch_symbols(
            symbols, start_date, calculate_indicators=True, store_indicators=True
        )

        print(f"\nCompleted: {successful} successful, {failed} failed")

        # Run screening
        if successful > 0:
            print("Running screening algorithms...")
            with SessionLocal() as session:
                screening_results = loader.run_screening_algorithms(session)

                print("Screening results:")
                for screen_type, count in screening_results.items():
                    print(f"  {screen_type}: {count} stocks")


async def load_sector_stocks():
    """Load stocks from a specific sector."""
    from scripts.tiingo_config import MARKET_SECTORS

    sector = "technology"
    symbols = MARKET_SECTORS[sector][:10]  # Just first 10 for demo

    print(f"Loading {sector} sector stocks: {len(symbols)} symbols")

    progress = ProgressTracker(f"{sector}_load_progress.json")
    progress.total_symbols = len(symbols)

    async with TiingoDataLoader(
        batch_size=5, max_concurrent=2, progress_tracker=progress
    ) as loader:
        # Load 2 years of data
        start_date = "2022-01-01"

        successful, failed = await loader.load_batch_symbols(
            symbols, start_date, calculate_indicators=True, store_indicators=True
        )

        print(f"\nSector loading completed: {successful} successful, {failed} failed")


async def resume_interrupted_load():
    """Demonstrate resuming from a checkpoint."""
    checkpoint_file = "sample_load_progress.json"

    if not os.path.exists(checkpoint_file):
        print(f"No checkpoint file found: {checkpoint_file}")
        return

    print("Resuming from checkpoint...")

    # Load progress
    progress = ProgressTracker(checkpoint_file)
    progress.load_checkpoint()

    # Get remaining symbols (this would normally come from your original symbol list)
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "ADBE"]
    remaining_symbols = [s for s in all_symbols if s not in progress.completed_symbols]

    if not remaining_symbols:
        print("All symbols already completed!")
        return

    print(f"Resuming with {len(remaining_symbols)} remaining symbols")

    async with TiingoDataLoader(
        batch_size=3, max_concurrent=2, progress_tracker=progress
    ) as loader:
        successful, failed = await loader.load_batch_symbols(
            remaining_symbols,
            "2023-01-01",
            calculate_indicators=True,
            store_indicators=True,
        )

        print(f"Resume completed: {successful} successful, {failed} failed")


def print_database_stats():
    """Print current database statistics."""
    from maverick_mcp.data.models import (
        MaverickStocks,
        PriceCache,
        Stock,
        TechnicalCache,
    )

    with SessionLocal() as session:
        stats = {
            "stocks": session.query(Stock).count(),
            "price_records": session.query(PriceCache).count(),
            "technical_indicators": session.query(TechnicalCache).count(),
            "maverick_stocks": session.query(MaverickStocks).count(),
        }

        print("\n📊 Current Database Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value:,}")


async def main():
    """Main demonstration function."""
    print("Tiingo Data Loader Examples")
    print("=" * 40)

    # Check for API token
    if not os.getenv("TIINGO_API_TOKEN"):
        print("❌ TIINGO_API_TOKEN environment variable not set")
        print("Please set your Tiingo API token:")
        print("export TIINGO_API_TOKEN=your_token_here")
        return

    print("✅ Tiingo API token found")

    # Show current database stats
    print_database_stats()

    # Menu of examples
    print("\nSelect an example to run:")
    print("1. Load sample stocks (5 symbols)")
    print("2. Load technology sector stocks (10 symbols)")
    print("3. Resume interrupted load")
    print("4. Show database stats")
    print("0. Exit")

    try:
        choice = input("\nEnter your choice (0-4): ").strip()

        if choice == "1":
            await load_sample_stocks()
        elif choice == "2":
            await load_sector_stocks()
        elif choice == "3":
            await resume_interrupted_load()
        elif choice == "4":
            print_database_stats()
        elif choice == "0":
            print("Goodbye!")
            return
        else:
            print("Invalid choice")
            return

        # Show updated stats
        print_database_stats()

    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
