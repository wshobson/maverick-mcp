#!/usr/bin/env python3
"""
Market data loading script for self-contained Maverick-MCP database.

This script loads stock and price data from Tiingo API into the self-contained
mcp_ prefixed tables, making Maverick-MCP completely independent.

Usage:
    python scripts/load_market_data.py --symbols AAPL,MSFT,GOOGL
    python scripts/load_market_data.py --file symbols.txt
    python scripts/load_market_data.py --sp500  # Load S&P 500 stocks
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from maverick_mcp.config.database_self_contained import (
    SelfContainedDatabaseSession,
    init_self_contained_database,
)
from maverick_mcp.data.models import (
    Stock,
    bulk_insert_price_data,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("market_data_loader")


class TiingoDataLoader:
    """Loads market data from Tiingo API into self-contained database."""

    def __init__(self, api_token: str | None = None):
        """
        Initialize Tiingo data loader.

        Args:
            api_token: Tiingo API token. If None, will use TIINGO_API_TOKEN env var
        """
        self.api_token = api_token or os.getenv("TIINGO_API_TOKEN")
        if not self.api_token:
            raise ValueError("Tiingo API token required. Set TIINGO_API_TOKEN env var.")

        self.base_url = "https://api.tiingo.com/tiingo"
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Token {self.api_token}"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_stock_metadata(self, symbol: str) -> dict | None:
        """
        Get stock metadata from Tiingo.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Stock metadata dict or None if not found
        """
        url = f"{self.base_url}/daily/{symbol}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 404:
                    logger.warning(f"Stock {symbol} not found in Tiingo")
                    return None
                else:
                    logger.error(
                        f"Error fetching metadata for {symbol}: {response.status}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Exception fetching metadata for {symbol}: {e}")
            return None

    async def get_price_data(
        self, symbol: str, start_date: str, end_date: str | None = None
    ) -> pd.DataFrame | None:
        """
        Get historical price data from Tiingo.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        url = f"{self.base_url}/daily/{symbol}/prices"
        params = {"startDate": start_date, "endDate": end_date, "format": "json"}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if not data:
                        return None

                    df = pd.DataFrame(data)

                    # Convert date column and set as index
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df.set_index("date", inplace=True)

                    # Rename columns to match our model
                    column_mapping = {
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume",
                        "adjOpen": "adj_open",
                        "adjHigh": "adj_high",
                        "adjLow": "adj_low",
                        "adjClose": "adj_close",
                        "adjVolume": "adj_volume",
                    }

                    df = df.rename(columns=column_mapping)
                    df["symbol"] = symbol.upper()

                    logger.info(f"Loaded {len(df)} price records for {symbol}")
                    return df

                elif response.status == 404:
                    logger.warning(f"Price data for {symbol} not found")
                    return None
                else:
                    logger.error(
                        f"Error fetching prices for {symbol}: {response.status}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Exception fetching prices for {symbol}: {e}")
            return None

    async def load_stock_data(self, symbols: list[str]) -> int:
        """
        Load stock metadata and price data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Number of stocks successfully loaded
        """
        loaded_count = 0

        with SelfContainedDatabaseSession() as session:
            for symbol in symbols:
                logger.info(f"Loading data for {symbol}...")

                # Get stock metadata
                metadata = await self.get_stock_metadata(symbol)
                if not metadata:
                    continue

                # Create or update stock record
                Stock.get_or_create(
                    session,
                    symbol,
                    company_name=metadata.get("name", ""),
                    description=metadata.get("description", ""),
                    exchange=metadata.get("exchangeCode", ""),
                    currency="USD",  # Tiingo uses USD
                )

                # Load price data (last 2 years)
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
                price_df = await self.get_price_data(symbol, start_date)

                if price_df is not None and not price_df.empty:
                    # Insert price data
                    records_inserted = bulk_insert_price_data(session, symbol, price_df)
                    logger.info(
                        f"Inserted {records_inserted} price records for {symbol}"
                    )

                loaded_count += 1

                # Rate limiting - Tiingo allows 2400 requests/hour
                await asyncio.sleep(1.5)  # ~2400 requests/hour limit

        return loaded_count


def get_sp500_symbols() -> list[str]:
    """Get S&P 500 stock symbols from a predefined list."""
    # Top 100 S&P 500 stocks for initial loading
    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "BRK.B",
        "UNH",
        "JNJ",
        "V",
        "PG",
        "JPM",
        "HD",
        "CVX",
        "MA",
        "PFE",
        "ABBV",
        "BAC",
        "KO",
        "AVGO",
        "PEP",
        "TMO",
        "COST",
        "WMT",
        "DIS",
        "ABT",
        "ACN",
        "NFLX",
        "ADBE",
        "CRM",
        "VZ",
        "DHR",
        "INTC",
        "NKE",
        "T",
        "TXN",
        "BMY",
        "QCOM",
        "PM",
        "UPS",
        "HON",
        "ORCL",
        "WFC",
        "LOW",
        "LIN",
        "AMD",
        "SBUX",
        "IBM",
        "GE",
        "CAT",
        "MDT",
        "BA",
        "AXP",
        "GILD",
        "RTX",
        "GS",
        "BLK",
        "MMM",
        "CVS",
        "ISRG",
        "NOW",
        "AMT",
        "SPGI",
        "PLD",
        "SYK",
        "TJX",
        "MDLZ",
        "ZTS",
        "MO",
        "CB",
        "CI",
        "PYPL",
        "SO",
        "EL",
        "DE",
        "REGN",
        "CCI",
        "USB",
        "BSX",
        "DUK",
        "AON",
        "CSX",
        "CL",
        "ITW",
        "PNC",
        "FCX",
        "SCHW",
        "EMR",
        "NSC",
        "GM",
        "FDX",
        "MU",
        "BDX",
        "TGT",
        "EOG",
        "SLB",
        "ICE",
        "EQIX",
        "APD",
    ]


def load_symbols_from_file(file_path: str) -> list[str]:
    """
    Load stock symbols from a text file.

    Args:
        file_path: Path to file containing stock symbols (one per line)

    Returns:
        List of stock symbols
    """
    symbols = []
    try:
        with open(file_path) as f:
            for line in f:
                symbol = line.strip().upper()
                if symbol and not symbol.startswith("#"):
                    symbols.append(symbol)
        logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
    except FileNotFoundError:
        logger.error(f"Symbol file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading symbol file {file_path}: {e}")
        sys.exit(1)

    return symbols


async def main():
    """Main function to load market data."""
    parser = argparse.ArgumentParser(
        description="Load market data into self-contained database"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--file", type=str, help="Path to file containing stock symbols (one per line)"
    )
    parser.add_argument(
        "--sp500", action="store_true", help="Load top 100 S&P 500 stocks"
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create database tables if they don't exist",
    )
    parser.add_argument("--database-url", type=str, help="Override database URL")

    args = parser.parse_args()

    # Determine symbols to load
    symbols = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.file:
        symbols = load_symbols_from_file(args.file)
    elif args.sp500:
        symbols = get_sp500_symbols()
    else:
        parser.print_help()
        sys.exit(1)

    logger.info(f"Will load data for {len(symbols)} symbols")

    # Initialize self-contained database
    try:
        init_self_contained_database(
            database_url=args.database_url, create_tables=args.create_tables
        )
        logger.info("Self-contained database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    # Load market data
    try:
        async with TiingoDataLoader() as loader:
            loaded_count = await loader.load_stock_data(symbols)
            logger.info(
                f"Successfully loaded data for {loaded_count}/{len(symbols)} stocks"
            )

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    # Display database stats
    from maverick_mcp.config.database_self_contained import get_self_contained_db_config

    db_config = get_self_contained_db_config()
    stats = db_config.get_database_stats()

    print("\n📊 Database Statistics:")
    print(f"   Total Records: {stats['total_records']}")
    for table, count in stats["tables"].items():
        print(f"   {table}: {count}")

    print("\n✅ Market data loading completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
