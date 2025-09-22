"""
Cache warming utilities for pre-loading commonly used data.
Improves performance by pre-fetching and caching frequently accessed data.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from maverick_mcp.data.cache import (
    CacheManager,
    ensure_timezone_naive,
    generate_cache_key,
    get_cache_stats,
)
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
from maverick_mcp.utils.yfinance_pool import get_yfinance_pool

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Pre-loads frequently accessed data into cache for improved performance."""

    def __init__(
        self,
        data_provider: EnhancedStockDataProvider | None = None,
        cache_manager: CacheManager | None = None,
        max_workers: int = 5
    ):
        """Initialize cache warmer.

        Args:
            data_provider: Stock data provider instance
            cache_manager: Cache manager instance
            max_workers: Maximum number of parallel workers
        """
        self.data_provider = data_provider or EnhancedStockDataProvider()
        self.cache = cache_manager or CacheManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._yf_pool = get_yfinance_pool()

        # Common symbols to warm up
        self.popular_symbols = [
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            "TSLA", "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD",
            "MA", "DIS", "BAC", "XOM", "PFE", "ABBV", "KO", "CVX", "PEP",
            "TMO", "AVGO", "COST", "MRK", "VZ", "ADBE", "CMCSA", "NKE"
        ]

        # Common date ranges
        self.common_periods = [
            ("1d", 1),    # Yesterday
            ("5d", 5),    # Last week
            ("1mo", 30),  # Last month
            ("3mo", 90),  # Last 3 months
            ("1y", 365),  # Last year
        ]

    async def warm_popular_stocks(self, symbols: list[str] | None = None):
        """Pre-load data for popular stocks.

        Args:
            symbols: List of symbols to warm up (uses default popular list if None)
        """
        symbols = symbols or self.popular_symbols
        logger.info(f"Warming cache for {len(symbols)} popular stocks")

        # Warm up in parallel batches
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            await self._warm_batch(batch)

        logger.info("Popular stocks cache warming completed")

    async def _warm_batch(self, symbols: list[str]):
        """Warm cache for a batch of symbols."""
        tasks = []
        for symbol in symbols:
            # Warm different time periods
            for period_name, days in self.common_periods:
                task = asyncio.create_task(
                    self._warm_symbol_period(symbol, period_name, days)
                )
                tasks.append(task)

        # Wait for all tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30
            )
        except TimeoutError:
            logger.warning(f"Timeout warming batch: {symbols}")

    async def _warm_symbol_period(self, symbol: str, period: str, days: int):
        """Warm cache for a specific symbol and period."""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            # Generate versioned cache key
            cache_key = generate_cache_key(
                "backtest_data",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )

            # Check if already cached
            if await self.cache.exists(cache_key):
                logger.debug(f"Cache already warm for {symbol} ({period})")
                return

            # Fetch data using the data provider
            data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.data_provider.get_stock_data,
                symbol,
                start_date,
                end_date,
                None,  # period
                "1d"   # interval
            )

            if data is not None and not data.empty:
                # Normalize column names and ensure timezone-naive
                data.columns = [col.lower() for col in data.columns]
                data = ensure_timezone_naive(data)

                # Cache with adaptive TTL based on data age
                ttl = 86400 if days > 7 else 3600  # 24h for older data, 1h for recent
                await self.cache.set(cache_key, data, ttl=ttl)

                logger.debug(f"Warmed cache for {symbol} ({period}) - {len(data)} rows")

        except Exception as e:
            logger.warning(f"Failed to warm cache for {symbol} ({period}): {e}")

    async def warm_screening_data(self):
        """Pre-load screening recommendations."""
        logger.info("Warming screening data cache")

        try:
            # Warm maverick recommendations
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.data_provider.get_maverick_recommendations,
                20,  # limit
                None  # min_score
            )

            # Warm bear recommendations
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.data_provider.get_maverick_bear_recommendations,
                20,
                None
            )

            # Warm supply/demand breakouts
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.data_provider.get_supply_demand_breakout_recommendations,
                20,
                None
            )

            logger.info("Screening data cache warming completed")

        except Exception as e:
            logger.error(f"Failed to warm screening cache: {e}")

    async def warm_technical_indicators(self, symbols: list[str] | None = None):
        """Pre-calculate and cache technical indicators for symbols.

        Args:
            symbols: List of symbols (uses top 10 popular if None)
        """
        symbols = symbols or self.popular_symbols[:10]
        logger.info(f"Warming technical indicators for {len(symbols)} stocks")

        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._warm_symbol_technicals(symbol))
            tasks.append(task)

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60
            )
        except TimeoutError:
            logger.warning("Timeout warming technical indicators")

        logger.info("Technical indicators cache warming completed")

    async def _warm_symbol_technicals(self, symbol: str):
        """Warm technical indicator cache for a symbol."""
        try:
            # Get recent data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

            # Common technical indicator cache keys
            indicators = [
                ("sma", [20, 50, 200]),
                ("ema", [12, 26]),
                ("rsi", [14]),
                ("macd", [12, 26, 9]),
                ("bb", [20, 2])
            ]

            for indicator, params in indicators:
                for param in params:
                    cache_key = f"technical:{symbol}:{indicator}:{param}:{start_date}:{end_date}"

                    if await self.cache.exists(cache_key):
                        continue

                    # Note: Actual technical calculation would go here
                    # For now, we're just warming the stock data cache
                    logger.debug(f"Would warm {indicator} for {symbol} with param {param}")

        except Exception as e:
            logger.warning(f"Failed to warm technicals for {symbol}: {e}")

    async def run_full_warmup(self, report_stats: bool = True):
        """Run complete cache warming routine."""
        logger.info("Starting full cache warmup")

        # Get initial cache stats
        initial_stats = get_cache_stats() if report_stats else None

        start_time = asyncio.get_event_loop().time()

        # Run all warming tasks
        results = await asyncio.gather(
            self.warm_popular_stocks(),
            self.warm_screening_data(),
            self.warm_technical_indicators(),
            return_exceptions=True
        )

        end_time = asyncio.get_event_loop().time()

        # Report results and performance
        successful_tasks = sum(1 for r in results if not isinstance(r, Exception))
        failed_tasks = len(results) - successful_tasks

        logger.info(
            f"Full cache warmup completed in {end_time - start_time:.2f}s - "
            f"{successful_tasks} successful, {failed_tasks} failed"
        )

        if report_stats and initial_stats:
            final_stats = get_cache_stats()
            new_items = final_stats["sets"] - initial_stats["sets"]
            hit_rate_change = final_stats["hit_rate_percent"] - initial_stats["hit_rate_percent"]

            logger.info(
                f"Cache warmup results: +{new_items} items cached, "
                f"hit rate change: {hit_rate_change:+.1f}%"
            )

    async def schedule_periodic_warmup(self, interval_minutes: int = 30):
        """Schedule periodic cache warming.

        Args:
            interval_minutes: Minutes between warmup runs
        """
        logger.info(f"Starting periodic cache warmup every {interval_minutes} minutes")

        while True:
            try:
                await self.run_full_warmup()
            except Exception as e:
                logger.error(f"Error in periodic warmup: {e}")

            # Wait for next cycle
            await asyncio.sleep(interval_minutes * 60)

    async def benchmark_cache_performance(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Benchmark cache performance for analysis.

        Args:
            symbols: List of symbols to test (uses top 5 if None)

        Returns:
            Dictionary with benchmark results
        """
        symbols = symbols or self.popular_symbols[:5]
        logger.info(f"Benchmarking cache performance with {len(symbols)} symbols")

        # Test data retrieval performance
        import time
        start_time = time.time()
        cache_hits = 0
        cache_misses = 0

        for symbol in symbols:
            for _period_name, days in self.common_periods:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

                cache_key = generate_cache_key(
                    "backtest_data",
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )

                cached_data = await self.cache.get(cache_key)
                if cached_data is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1

        end_time = time.time()

        # Calculate metrics
        total_requests = cache_hits + cache_misses
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        avg_request_time = (end_time - start_time) / total_requests if total_requests > 0 else 0

        # Get current cache stats
        cache_stats = get_cache_stats()

        benchmark_results = {
            "symbols_tested": len(symbols),
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "avg_request_time_ms": round(avg_request_time * 1000, 2),
            "total_time_seconds": round(end_time - start_time, 2),
            "cache_stats": cache_stats,
        }

        logger.info(
            f"Benchmark completed: {hit_rate:.1f}% hit rate, "
            f"{avg_request_time*1000:.1f}ms avg request time"
        )

        return benchmark_results

    def shutdown(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
        logger.info("Cache warmer shutdown")


async def warm_cache_on_startup():
    """Convenience function to warm cache on application startup."""
    warmer = CacheWarmer()
    try:
        # Only warm the most critical data on startup
        await warmer.warm_popular_stocks(
            ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
        )
        await warmer.warm_screening_data()
    finally:
        warmer.shutdown()


if __name__ == "__main__":
    # Example usage
    async def main():
        warmer = CacheWarmer()
        try:
            await warmer.run_full_warmup()
        finally:
            warmer.shutdown()

    asyncio.run(main())
