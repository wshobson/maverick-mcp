#!/usr/bin/env python3
"""Example usage of the Tiingo data loader.

This script demonstrates common usage patterns for loading market data from the
Tiingo API into the Maverick-MCP database. It is a thin demo wrapper around
``scripts.load_tiingo_data.TiingoDataLoader`` and intentionally uses only that
loader's public API — checkpointing is handled by the loader itself via the
``checkpoint_file`` constructor argument, not a separate tracker.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent directory to path so the script is runnable as `python scripts/load_example.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from maverick_mcp.data.models import (  # noqa: E402
    MaverickStocks,
    PriceCache,
    SessionLocal,
    Stock,
    TechnicalCache,
)
from scripts.load_tiingo_data import TiingoDataLoader  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _today_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


async def load_sample_stocks() -> None:
    """Load a small sample of stocks for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"Loading sample stocks: {', '.join(symbols)}")

    loader = TiingoDataLoader(checkpoint_file="sample_load_progress.json")
    await loader.load_symbols(
        symbols=symbols,
        start_date="2023-01-01",
        end_date=_today_iso(),
        calculate_indicators=True,
        run_screening=True,
        max_concurrent=3,
    )
    print("Sample load complete.")


async def load_sector_stocks() -> None:
    """Load stocks from a specific sector."""
    from scripts.tiingo_config import MARKET_SECTORS

    sector = "technology"
    symbols = MARKET_SECTORS[sector][:10]  # Just first 10 for demo

    print(f"Loading {sector} sector stocks: {len(symbols)} symbols")

    loader = TiingoDataLoader(checkpoint_file=f"{sector}_load_progress.json")
    await loader.load_symbols(
        symbols=symbols,
        start_date="2022-01-01",
        end_date=_today_iso(),
        calculate_indicators=True,
        run_screening=True,
        max_concurrent=2,
    )
    print(f"{sector.title()} sector loading complete.")


async def resume_interrupted_load() -> None:
    """Demonstrate resuming from a checkpoint.

    The loader transparently skips symbols already in the checkpoint file's
    ``completed_symbols`` list, so resuming is just re-instantiating with the
    same ``checkpoint_file`` and re-calling ``load_symbols``.
    """
    checkpoint_file = "sample_load_progress.json"

    if not await asyncio.to_thread(os.path.exists, checkpoint_file):
        print(f"No checkpoint file found: {checkpoint_file}")
        return

    print("Resuming from checkpoint...")

    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "ADBE"]

    loader = TiingoDataLoader(checkpoint_file=checkpoint_file)
    completed = set(loader.checkpoint_data.get("completed_symbols", []))
    remaining = [s for s in all_symbols if s not in completed]

    if not remaining:
        print("All symbols already completed!")
        return

    print(f"Resuming with {len(remaining)} remaining symbols")
    await loader.load_symbols(
        symbols=remaining,
        start_date="2023-01-01",
        end_date=_today_iso(),
        calculate_indicators=True,
        run_screening=True,
        max_concurrent=2,
    )
    print("Resume complete.")


def print_database_stats() -> None:
    """Print current database statistics."""
    with SessionLocal() as session:
        stats = {
            "stocks": session.query(Stock).count(),
            "price_records": session.query(PriceCache).count(),
            "technical_indicators": session.query(TechnicalCache).count(),
            "maverick_stocks": session.query(MaverickStocks).count(),
        }

    print("\nCurrent Database Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:,}")


async def main() -> None:
    """Main demonstration function."""
    print("Tiingo Data Loader Examples")
    print("=" * 40)

    if not os.getenv("TIINGO_API_TOKEN"):
        print("TIINGO_API_TOKEN environment variable not set")
        print("Please set your Tiingo API token:")
        print("export TIINGO_API_TOKEN=your_token_here")
        return

    print("Tiingo API token found")
    print_database_stats()

    print("\nSelect an example to run:")
    print("1. Load sample stocks (5 symbols)")
    print("2. Load technology sector stocks (10 symbols)")
    print("3. Resume interrupted load")
    print("4. Show database stats")
    print("0. Exit")

    try:
        raw_choice = await asyncio.to_thread(input, "\nEnter your choice (0-4): ")
        choice = raw_choice.strip()

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

        print_database_stats()

    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
