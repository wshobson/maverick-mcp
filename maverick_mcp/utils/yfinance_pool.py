"""
Optimized yfinance connection pooling and caching.
Provides thread-safe connection pooling and request optimization for yfinance.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class YFinancePool:
    """Thread-safe yfinance connection pool with optimized session management."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure single connection pool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the connection pool once."""
        if self._initialized:
            return

        # Create optimized session with connection pooling
        self.session = self._create_optimized_session()

        # Thread pool for parallel requests
        self.executor = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="yfinance_pool"
        )

        # Request cache (simple TTL cache)
        self._request_cache: dict[str, tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 60  # 1 minute cache for quotes

        self._initialized = True
        logger.info("YFinance connection pool initialized")

    def _create_optimized_session(self) -> Session:
        """Create an optimized requests session with retry logic and connection pooling."""
        session = Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=50,      # Max connections per pool
            max_retries=retry_strategy,
            pool_block=False      # Don't block when pool is full
        )

        # Mount adapter for HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set headers to avoid rate limiting
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        return session

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a ticker object - let yfinance handle session for compatibility."""
        # Check cache first
        cache_key = f"ticker_{symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Create ticker without custom session (yfinance now requires curl_cffi)
        ticker = yf.Ticker(symbol)

        # Cache for short duration
        self._add_to_cache(cache_key, ticker, ttl=300)  # 5 minutes

        return ticker

    def get_history(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical data with connection pooling."""
        # Create cache key
        cache_key = f"history_{symbol}_{start}_{end}_{period}_{interval}"

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached

        # Get ticker with optimized session
        ticker = self.get_ticker(symbol)

        # Fetch data
        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            if start is None:
                start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if end is None:
                end = datetime.now().strftime("%Y-%m-%d")
            df = ticker.history(start=start, end=end, interval=interval)

        # Cache the result (longer TTL for historical data)
        if not df.empty:
            ttl = 3600 if interval == "1d" else 300  # 1 hour for daily, 5 min for intraday
            self._add_to_cache(cache_key, df, ttl=ttl)

        return df

    def get_info(self, symbol: str) -> dict:
        """Get stock info with caching."""
        cache_key = f"info_{symbol}"

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # Get ticker and info
        ticker = self.get_ticker(symbol)
        info = ticker.info

        # Cache for longer duration (info doesn't change often)
        self._add_to_cache(cache_key, info, ttl=3600)  # 1 hour

        return info

    def batch_download(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        interval: str = "1d",
        group_by: str = "ticker",
        threads: bool = True
    ) -> pd.DataFrame:
        """Download data for multiple symbols efficiently."""
        # Use yfinance's batch download without custom session
        if period:
            data = yf.download(
                tickers=symbols,
                period=period,
                interval=interval,
                group_by=group_by,
                threads=threads,
                progress=False
            )
        else:
            if start is None:
                start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if end is None:
                end = datetime.now().strftime("%Y-%m-%d")

            data = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                interval=interval,
                group_by=group_by,
                threads=threads,
                progress=False
            )

        return data

    def _get_from_cache(self, key: str) -> Any | None:
        """Get item from cache if not expired."""
        with self._cache_lock:
            if key in self._request_cache:
                value, expiry = self._request_cache[key]
                if datetime.now().timestamp() < expiry:
                    logger.debug(f"Cache hit for {key}")
                    return value
                else:
                    del self._request_cache[key]
        return None

    def _add_to_cache(self, key: str, value: Any, ttl: int = 60):
        """Add item to cache with TTL."""
        with self._cache_lock:
            expiry = datetime.now().timestamp() + ttl
            self._request_cache[key] = (value, expiry)

            # Clean up old entries if cache is too large
            if len(self._request_cache) > 1000:
                self._cleanup_cache()

    def _cleanup_cache(self):
        """Remove expired entries from cache."""
        current_time = datetime.now().timestamp()
        expired_keys = [
            k for k, (_, expiry) in self._request_cache.items()
            if expiry < current_time
        ]
        for key in expired_keys:
            del self._request_cache[key]

        # If still too large, remove oldest entries
        if len(self._request_cache) > 800:
            sorted_items = sorted(
                self._request_cache.items(),
                key=lambda x: x[1][1]  # Sort by expiry time
            )
            # Keep only the newest 600 entries
            self._request_cache = dict(sorted_items[-600:])

    def close(self):
        """Clean up resources."""
        try:
            self.session.close()
            self.executor.shutdown(wait=False)
            logger.info("YFinance connection pool closed")
        except Exception as e:
            logger.warning(f"Error closing connection pool: {e}")


# Global instance
_yfinance_pool: YFinancePool | None = None


def get_yfinance_pool() -> YFinancePool:
    """Get or create the global yfinance connection pool."""
    global _yfinance_pool
    if _yfinance_pool is None:
        _yfinance_pool = YFinancePool()
    return _yfinance_pool


def cleanup_yfinance_pool():
    """Clean up the global connection pool."""
    global _yfinance_pool
    if _yfinance_pool:
        _yfinance_pool.close()
        _yfinance_pool = None
