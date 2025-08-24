#!/usr/bin/env python3
"""
Tiingo Data Loader for MaverickMCP

Loads market data from Tiingo API into the self-contained MaverickMCP database.
Supports batch loading, rate limiting, progress tracking, and technical indicator calculation.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from maverick_mcp.data.models import (
    Stock,
    bulk_insert_price_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - following tiingo-python patterns from api.py
# Base URL without version suffix (will be added per endpoint)
DEFAULT_BASE_URL = os.getenv("TIINGO_BASE_URL", "https://api.tiingo.com")

# API token from environment
TIINGO_API_TOKEN = os.getenv("TIINGO_API_TOKEN")

# Rate limiting configuration - can be overridden by command line
DEFAULT_MAX_CONCURRENT = int(os.getenv("TIINGO_MAX_CONCURRENT", "5"))
DEFAULT_RATE_LIMIT_PER_HOUR = int(os.getenv("TIINGO_RATE_LIMIT", "2400"))
DEFAULT_CHECKPOINT_FILE = os.getenv(
    "TIINGO_CHECKPOINT_FILE", "tiingo_load_progress.json"
)

# Default timeout for requests (from tiingo-python)
DEFAULT_TIMEOUT = int(os.getenv("TIINGO_TIMEOUT", "10"))


class TiingoDataLoader:
    """Handles loading data from Tiingo API into MaverickMCP database.

    Following the design patterns from tiingo-python library.
    """

    def __init__(
        self,
        api_token: str | None = None,
        db_url: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        rate_limit_per_hour: int | None = None,
        checkpoint_file: str | None = None,
    ):
        """Initialize the Tiingo data loader.

        Args:
            api_token: Tiingo API token (defaults to env var)
            db_url: Database URL (defaults to env var)
            base_url: Base URL for Tiingo API (defaults to env var)
            timeout: Request timeout in seconds
            rate_limit_per_hour: Max requests per hour
            checkpoint_file: Path to checkpoint file
        """
        # API configuration (following tiingo-python patterns)
        self.api_token = api_token or TIINGO_API_TOKEN
        if not self.api_token:
            raise ValueError(
                "API token required. Set TIINGO_API_TOKEN env var or pass api_token parameter."
            )

        # Database configuration
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "Database URL required. Set DATABASE_URL env var or pass db_url parameter."
            )

        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # API endpoint configuration
        self.base_url = base_url or DEFAULT_BASE_URL
        self.timeout = timeout or DEFAULT_TIMEOUT

        # Rate limiting
        self.request_count = 0
        self.start_time = datetime.now()
        self.rate_limit_per_hour = rate_limit_per_hour or DEFAULT_RATE_LIMIT_PER_HOUR
        self.rate_limit_delay = (
            3600 / self.rate_limit_per_hour
        )  # seconds between requests

        # Checkpoint configuration
        self.checkpoint_file = checkpoint_file or DEFAULT_CHECKPOINT_FILE
        self.checkpoint_data = self.load_checkpoint()

        # Session configuration (following tiingo-python)
        self._session = None

    def load_checkpoint(self) -> dict:
        """Load checkpoint data if exists."""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {"completed_symbols": [], "last_symbol": None}

    def save_checkpoint(self, symbol: str):
        """Save checkpoint data."""
        self.checkpoint_data["completed_symbols"].append(symbol)
        self.checkpoint_data["last_symbol"] = symbol
        self.checkpoint_data["timestamp"] = datetime.now().isoformat()

        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")

    def _get_headers(self) -> dict[str, str]:
        """Get request headers following tiingo-python patterns."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_token}",
            "User-Agent": "tiingo-python-client/maverick-mcp",
        }

    async def _request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Any | None:
        """Make HTTP request with rate limiting and error handling.

        Following tiingo-python's request patterns from api.py.

        Args:
            session: aiohttp session
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            max_retries: Maximum number of retries

        Returns:
            Response data or None if failed
        """
        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)
        self.request_count += 1

        # Build URL
        url = f"{self.base_url}{endpoint}"
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{param_str}"

        headers = self._get_headers()

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 400:
                        error_msg = await response.text()
                        logger.error(f"Bad request (400): {error_msg}")
                        return None
                    elif response.status == 404:
                        logger.warning(f"Resource not found (404): {endpoint}")
                        return None
                    elif response.status == 429:
                        # Rate limited, exponential backoff
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                        else:
                            wait_time = min(60 * (2**attempt), 300)
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status >= 500:
                        # Server error, retry with backoff
                        if attempt < max_retries - 1:
                            wait_time = min(5 * (2**attempt), 60)
                            logger.warning(
                                f"Server error {response.status}, retry in {wait_time}s..."
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Server error after {max_retries} attempts")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP {response.status}: {error_text}")
                        return None

            except TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = min(10 * (2**attempt), 60)
                    logger.warning(f"Timeout, retry in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Timeout after {max_retries} attempts")
                    return None

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(10 * (2**attempt), 60)
                    logger.warning(f"Error: {e}, retry in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None

        return None

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw DataFrame from Tiingo API.

        Following tiingo-python's DataFrame processing patterns.

        Args:
            df: Raw DataFrame from API

        Returns:
            Processed DataFrame with proper column names and index
        """
        if df.empty:
            return df

        # Handle date column following tiingo-python
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

            # Standardize column names to match expected format
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "adjOpen": "Adj Open",
                "adjHigh": "Adj High",
                "adjLow": "Adj Low",
                "adjClose": "Adj Close",
                "adjVolume": "Adj Volume",
                "divCash": "Dividend",
                "splitFactor": "Split Factor",
            }

            # Only rename columns that exist
            rename_dict = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            if rename_dict:
                df = df.rename(columns=rename_dict)

            # Set date as index
            df = df.set_index("date")

            # Localize to UTC following tiingo-python approach
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            # For database storage, convert to date only (no time component)
            df.index = df.index.date

        return df

    async def get_ticker_metadata(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> dict[str, Any] | None:
        """Get metadata for a specific ticker.

        Following tiingo-python's get_ticker_metadata pattern.
        """
        endpoint = f"/tiingo/daily/{symbol}"
        return await self._request(session, endpoint)

    async def get_available_symbols(
        self,
        session: aiohttp.ClientSession,
        asset_types: list[str] | None = None,
        exchanges: list[str] | None = None,
    ) -> list[str]:
        """Get list of available symbols from Tiingo with optional filtering.

        Following tiingo-python's list_tickers pattern.
        """
        endpoint = "/tiingo/daily/supported_tickers"
        data = await self._request(session, endpoint)

        if data:
            # Default filters if not provided
            asset_types = asset_types or ["Stock"]
            exchanges = exchanges or ["NYSE", "NASDAQ"]

            symbols = []
            for ticker_info in data:
                if (
                    ticker_info.get("exchange") in exchanges
                    and ticker_info.get("assetType") in asset_types
                    and ticker_info.get("priceCurrency") == "USD"
                ):
                    symbols.append(ticker_info["ticker"])
            return symbols
        return []

    async def get_daily_price_history(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "daily",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch historical price data for a symbol.

        Following tiingo-python's get_dataframe pattern from api.py.

        Args:
            session: aiohttp session
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (daily, weekly, monthly, annually)
            columns: Specific columns to return

        Returns:
            DataFrame with price history
        """
        endpoint = f"/tiingo/daily/{symbol}/prices"

        # Build params following tiingo-python
        params = {
            "format": "json",
            "resampleFreq": frequency,
        }

        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        if columns:
            params["columns"] = ",".join(columns)

        data = await self._request(session, endpoint, params)

        if data:
            try:
                df = pd.DataFrame(data)
                if not df.empty:
                    # Process DataFrame following tiingo-python patterns
                    df = self._process_dataframe(df)

                    # Validate data integrity
                    if len(df) == 0:
                        logger.warning(f"Empty dataset returned for {symbol}")
                        return pd.DataFrame()

                    # Check for required columns
                    required_cols = ["Open", "High", "Low", "Close", "Volume"]
                    missing_cols = [
                        col for col in required_cols if col not in df.columns
                    ]
                    if missing_cols:
                        logger.warning(f"Missing columns for {symbol}: {missing_cols}")

                    return df

            except Exception as e:
                logger.error(f"Error processing data for {symbol}: {e}")
                return pd.DataFrame()

        return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        if df.empty or len(df) < 200:
            return df

        try:
            # Moving averages
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["SMA_150"] = df["Close"].rolling(window=150).mean()
            df["SMA_200"] = df["Close"].rolling(window=200).mean()
            df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

            # RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

            # ATR
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df["ATR"] = true_range.rolling(14).mean()

            # ADR (Average Daily Range) as percentage
            df["ADR_PCT"] = (
                ((df["High"] - df["Low"]) / df["Close"] * 100).rolling(20).mean()
            )

            # Volume indicators
            df["Volume_SMA_30"] = df["Volume"].rolling(window=30).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_30"]

            # Momentum Score (simplified)
            returns = df["Close"].pct_change(periods=252)  # 1-year returns
            df["Momentum_Score"] = returns.rank(pct=True) * 100

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return df

    def run_maverick_screening(self, df: pd.DataFrame, symbol: str) -> dict | None:
        """Run Maverick momentum screening algorithm."""
        if df.empty or len(df) < 200:
            return None

        try:
            latest = df.iloc[-1]

            # Maverick criteria
            price_above_ema21 = latest["Close"] > latest.get("EMA_21", 0)
            ema21_above_sma50 = latest.get("EMA_21", 0) > latest.get("SMA_50", 0)
            sma50_above_sma200 = latest.get("SMA_50", 0) > latest.get("SMA_200", 0)
            strong_momentum = latest.get("Momentum_Score", 0) > 70

            # Calculate combined score
            score = 0
            if price_above_ema21:
                score += 25
            if ema21_above_sma50:
                score += 25
            if sma50_above_sma200:
                score += 25
            if strong_momentum:
                score += 25

            if score >= 75:  # Meets criteria
                return {
                    "stock": symbol,
                    "close": float(latest["Close"]),
                    "volume": int(latest["Volume"]),
                    "momentum_score": float(latest.get("Momentum_Score", 0)),
                    "combined_score": score,
                    "adr_pct": float(latest.get("ADR_PCT", 0)),
                    "atr": float(latest.get("ATR", 0)),
                    "ema_21": float(latest.get("EMA_21", 0)),
                    "sma_50": float(latest.get("SMA_50", 0)),
                    "sma_150": float(latest.get("SMA_150", 0)),
                    "sma_200": float(latest.get("SMA_200", 0)),
                }
        except Exception as e:
            logger.error(f"Error in Maverick screening for {symbol}: {e}")

        return None

    def run_bear_screening(self, df: pd.DataFrame, symbol: str) -> dict | None:
        """Run Bear market screening algorithm."""
        if df.empty or len(df) < 200:
            return None

        try:
            latest = df.iloc[-1]

            # Bear criteria
            price_below_ema21 = latest["Close"] < latest.get("EMA_21", float("inf"))
            ema21_below_sma50 = latest.get("EMA_21", float("inf")) < latest.get(
                "SMA_50", float("inf")
            )
            weak_momentum = latest.get("Momentum_Score", 100) < 30
            negative_macd = latest.get("MACD", 0) < 0

            # Calculate bear score
            score = 0
            if price_below_ema21:
                score += 25
            if ema21_below_sma50:
                score += 25
            if weak_momentum:
                score += 25
            if negative_macd:
                score += 25

            if score >= 75:  # Meets bear criteria
                return {
                    "stock": symbol,
                    "close": float(latest["Close"]),
                    "volume": int(latest["Volume"]),
                    "momentum_score": float(latest.get("Momentum_Score", 0)),
                    "score": score,
                    "rsi_14": float(latest.get("RSI", 0)),
                    "macd": float(latest.get("MACD", 0)),
                    "macd_signal": float(latest.get("MACD_Signal", 0)),
                    "macd_histogram": float(latest.get("MACD_Histogram", 0)),
                    "adr_pct": float(latest.get("ADR_PCT", 0)),
                    "atr": float(latest.get("ATR", 0)),
                    "ema_21": float(latest.get("EMA_21", 0)),
                    "sma_50": float(latest.get("SMA_50", 0)),
                    "sma_200": float(latest.get("SMA_200", 0)),
                }
        except Exception as e:
            logger.error(f"Error in Bear screening for {symbol}: {e}")

        return None

    def run_supply_demand_screening(self, df: pd.DataFrame, symbol: str) -> dict | None:
        """Run Supply/Demand breakout screening algorithm."""
        if df.empty or len(df) < 200:
            return None

        try:
            latest = df.iloc[-1]

            # Supply/Demand criteria (accumulation phase)
            close = latest["Close"]
            sma_50 = latest.get("SMA_50", 0)
            sma_150 = latest.get("SMA_150", 0)
            sma_200 = latest.get("SMA_200", 0)

            # Check for proper alignment
            price_above_all = close > sma_50 > sma_150 > sma_200
            strong_momentum = latest.get("Momentum_Score", 0) > 80

            # Volume confirmation
            volume_confirmation = latest.get("Volume_Ratio", 0) > 1.2

            if price_above_all and strong_momentum and volume_confirmation:
                return {
                    "stock": symbol,
                    "close": float(close),
                    "volume": int(latest["Volume"]),
                    "momentum_score": float(latest.get("Momentum_Score", 0)),
                    "adr_pct": float(latest.get("ADR_PCT", 0)),
                    "atr": float(latest.get("ATR", 0)),
                    "ema_21": float(latest.get("EMA_21", 0)),
                    "sma_50": float(sma_50),
                    "sma_150": float(sma_150),
                    "sma_200": float(sma_200),
                    "avg_volume_30d": float(latest.get("Volume_SMA_30", 0)),
                }
        except Exception as e:
            logger.error(f"Error in Supply/Demand screening for {symbol}: {e}")

        return None

    async def process_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_date: str,
        end_date: str,
        calculate_indicators: bool = True,
        run_screening: bool = True,
    ) -> tuple[bool, dict | None]:
        """Process a single symbol - fetch data, calculate indicators, run screening."""
        try:
            # Skip if already processed
            if symbol in self.checkpoint_data.get("completed_symbols", []):
                logger.info(f"Skipping {symbol} - already processed")
                return True, None

            # Fetch historical data using tiingo-python pattern
            df = await self.get_daily_price_history(
                session, symbol, start_date, end_date
            )

            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return False, None

            # Store in database
            with self.SessionLocal() as db_session:
                # Create or get stock record
                Stock.get_or_create(db_session, symbol)

                # Bulk insert price data
                records_inserted = bulk_insert_price_data(db_session, symbol, df)
                logger.info(f"Inserted {records_inserted} records for {symbol}")

            screening_results = {}

            if calculate_indicators:
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df)

                if run_screening:
                    # Run screening algorithms
                    maverick_result = self.run_maverick_screening(df, symbol)
                    if maverick_result:
                        screening_results["maverick"] = maverick_result

                    bear_result = self.run_bear_screening(df, symbol)
                    if bear_result:
                        screening_results["bear"] = bear_result

                    supply_demand_result = self.run_supply_demand_screening(df, symbol)
                    if supply_demand_result:
                        screening_results["supply_demand"] = supply_demand_result

            # Save checkpoint
            self.save_checkpoint(symbol)

            return True, screening_results

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False, None

    async def load_symbols(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        calculate_indicators: bool = True,
        run_screening: bool = True,
        max_concurrent: int = None,
    ):
        """Load data for multiple symbols with concurrent processing."""
        logger.info(
            f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}"
        )

        # Filter out already processed symbols if resuming
        symbols_to_process = [
            s
            for s in symbols
            if s not in self.checkpoint_data.get("completed_symbols", [])
        ]

        if len(symbols_to_process) < len(symbols):
            logger.info(
                f"Resuming: {len(symbols) - len(symbols_to_process)} symbols already processed"
            )

        screening_results = {"maverick": [], "bear": [], "supply_demand": []}

        # Use provided max_concurrent or default
        concurrent_limit = max_concurrent or DEFAULT_MAX_CONCURRENT

        async with aiohttp.ClientSession() as session:
            # Process in batches with semaphore for rate limiting
            semaphore = asyncio.Semaphore(concurrent_limit)

            async def process_with_semaphore(symbol):
                async with semaphore:
                    return await self.process_symbol(
                        session,
                        symbol,
                        start_date,
                        end_date,
                        calculate_indicators,
                        run_screening,
                    )

            # Create tasks with progress bar
            tasks = []
            for symbol in symbols_to_process:
                tasks.append(process_with_semaphore(symbol))

            # Process with progress bar
            with tqdm(total=len(tasks), desc="Processing symbols") as pbar:
                for coro in asyncio.as_completed(tasks):
                    success, results = await coro
                    if results:
                        for screen_type, data in results.items():
                            screening_results[screen_type].append(data)
                    pbar.update(1)

        # Store screening results in database
        if run_screening:
            self.store_screening_results(screening_results)

        logger.info(f"Completed loading {len(symbols_to_process)} symbols")
        logger.info(
            f"Screening results - Maverick: {len(screening_results['maverick'])}, "
            f"Bear: {len(screening_results['bear'])}, "
            f"Supply/Demand: {len(screening_results['supply_demand'])}"
        )

    def store_screening_results(self, results: dict):
        """Store screening results in database."""
        with self.SessionLocal() as db_session:
            # Store Maverick results
            for _data in results["maverick"]:
                # Implementation would create MaverickStocks records
                pass

            # Store Bear results
            for _data in results["bear"]:
                # Implementation would create MaverickBearStocks records
                pass

            # Store Supply/Demand results
            for _data in results["supply_demand"]:
                # Implementation would create SupplyDemandBreakoutStocks records
                pass

            db_session.commit()


def get_test_symbols() -> list[str]:
    """Get a small test set of symbols for development.

    These are just for testing - production use should load from
    external sources or command line arguments.
    """
    # Test symbols from different sectors for comprehensive testing
    return [
        "AAPL",  # Apple - Tech
        "MSFT",  # Microsoft - Tech
        "GOOGL",  # Alphabet - Tech
        "AMZN",  # Amazon - Consumer Discretionary
        "NVDA",  # NVIDIA - Tech
        "META",  # Meta - Communication
        "TSLA",  # Tesla - Consumer Discretionary
        "UNH",  # UnitedHealth - Healthcare
        "JPM",  # JPMorgan Chase - Financials
        "V",  # Visa - Financials
        "WMT",  # Walmart - Consumer Staples
        "JNJ",  # Johnson & Johnson - Healthcare
        "MA",  # Mastercard - Financials
        "HD",  # Home Depot - Consumer Discretionary
        "PG",  # Procter & Gamble - Consumer Staples
        "XOM",  # ExxonMobil - Energy
        "CVX",  # Chevron - Energy
        "KO",  # Coca-Cola - Consumer Staples
        "PEP",  # PepsiCo - Consumer Staples
        "ADBE",  # Adobe - Tech
        "NFLX",  # Netflix - Communication
        "CRM",  # Salesforce - Tech
        "DIS",  # Disney - Communication
        "COST",  # Costco - Consumer Staples
        "MRK",  # Merck - Healthcare
    ]


def get_sp500_symbols() -> list[str]:
    """Get S&P 500 symbols list from external source or file.

    This function should load S&P 500 symbols from:
    1. Environment variable SP500_SYMBOLS_FILE pointing to a file
    2. Download from a public data source
    3. Return empty list with warning if unavailable
    """
    # Try to load from file specified in environment
    symbols_file = os.getenv("SP500_SYMBOLS_FILE")
    if symbols_file and Path(symbols_file).exists():
        try:
            with open(symbols_file) as f:
                symbols = [line.strip() for line in f if line.strip()]
                logger.info(
                    f"Loaded {len(symbols)} S&P 500 symbols from {symbols_file}"
                )
                return symbols
        except Exception as e:
            logger.warning(f"Could not load S&P 500 symbols from {symbols_file}: {e}")

    # Try to fetch from a public source (like Wikipedia or Yahoo Finance)
    try:
        # Using pandas to read S&P 500 list from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]  # First table contains the S&P 500 list
        symbols = sp500_table["Symbol"].tolist()
        logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")

        # Optional: Save to cache file for future use
        cache_file = os.getenv("SP500_CACHE_FILE", "sp500_symbols_cache.txt")
        try:
            with open(cache_file, "w") as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            logger.info(f"Cached S&P 500 symbols to {cache_file}")
        except Exception as e:
            logger.debug(f"Could not cache symbols: {e}")

        return symbols

    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 symbols from web: {e}")

        # Try to load from cache if web fetch failed
        cache_file = os.getenv("SP500_CACHE_FILE", "sp500_symbols_cache.txt")
        if Path(cache_file).exists():
            try:
                with open(cache_file) as f:
                    symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(symbols)} S&P 500 symbols from cache")
                return symbols
            except Exception as e:
                logger.warning(f"Could not load from cache: {e}")

    logger.error("Unable to load S&P 500 symbols. Please specify --file or --symbols")
    return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load market data from Tiingo API")
    parser.add_argument("--symbols", nargs="+", help="List of symbols to load")
    parser.add_argument("--file", help="Load symbols from file (one per line)")
    parser.add_argument(
        "--test", action="store_true", help="Load test set of 25 symbols"
    )
    parser.add_argument("--sp500", action="store_true", help="Load S&P 500 symbols")
    parser.add_argument(
        "--years", type=int, default=2, help="Number of years of history"
    )
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--calculate-indicators",
        action="store_true",
        help="Calculate technical indicators",
    )
    parser.add_argument(
        "--run-screening", action="store_true", help="Run screening algorithms"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Maximum concurrent requests"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--db-url", help="Database URL override")

    args = parser.parse_args()

    # Check for API token
    if not TIINGO_API_TOKEN:
        logger.error("TIINGO_API_TOKEN environment variable not set")
        sys.exit(1)

    # Determine database URL
    db_url = args.db_url or os.getenv("MCP_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("Database URL not configured")
        sys.exit(1)

    # Determine symbols to load
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.file:
        # Load symbols from file
        try:
            with open(args.file) as f:
                symbols = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(symbols)} symbols from {args.file}")
        except Exception as e:
            logger.error(f"Could not read symbols from file: {e}")
            sys.exit(1)
    elif args.test:
        symbols = get_test_symbols()
        logger.info(f"Using test set of {len(symbols)} symbols")
    elif args.sp500:
        symbols = get_sp500_symbols()
        logger.info(f"Using S&P 500 symbols ({len(symbols)} total)")
    else:
        logger.error("No symbols specified. Use --symbols, --file, --test, or --sp500")
        sys.exit(1)

    # Determine date range
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime(
            "%Y-%m-%d"
        )

    # Create loader using tiingo-python style initialization
    loader = TiingoDataLoader(
        api_token=TIINGO_API_TOKEN,
        db_url=db_url,
        rate_limit_per_hour=DEFAULT_RATE_LIMIT_PER_HOUR,
    )

    # Run async loading
    asyncio.run(
        loader.load_symbols(
            symbols,
            start_date,
            end_date,
            calculate_indicators=args.calculate_indicators,
            run_screening=args.run_screening,
            max_concurrent=args.max_concurrent,
        )
    )

    logger.info("Data loading complete!")

    # Clean up checkpoint if completed successfully
    checkpoint_file = DEFAULT_CHECKPOINT_FILE
    if not args.resume and Path(checkpoint_file).exists():
        os.remove(checkpoint_file)
        logger.info("Removed checkpoint file")


if __name__ == "__main__":
    main()
