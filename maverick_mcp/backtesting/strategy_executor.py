"""
Parallel strategy execution engine for high-performance backtesting.
Implements worker pool pattern with concurrency control and thread-safe operations.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector

from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine
from maverick_mcp.data.cache import CacheManager
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Execution context for strategy runs."""

    strategy_id: str
    symbol: str
    strategy_type: str
    parameters: dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    fees: float = 0.001
    slippage: float = 0.001


@dataclass
class ExecutionResult:
    """Result of strategy execution."""

    context: ExecutionContext
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time: float = 0.0


class StrategyExecutor:
    """High-performance parallel strategy executor with connection pooling."""

    def __init__(
        self,
        max_concurrent_strategies: int = 6,
        max_concurrent_api_requests: int = 10,
        connection_pool_size: int = 100,
        request_timeout: int = 30,
        cache_manager: CacheManager | None = None,
    ):
        """
        Initialize parallel strategy executor.

        Args:
            max_concurrent_strategies: Maximum concurrent strategy executions
            max_concurrent_api_requests: Maximum concurrent API requests
            connection_pool_size: HTTP connection pool size
            request_timeout: Request timeout in seconds
            cache_manager: Optional cache manager instance
        """
        self.max_concurrent_strategies = max_concurrent_strategies
        self.max_concurrent_api_requests = max_concurrent_api_requests
        self.connection_pool_size = connection_pool_size
        self.request_timeout = request_timeout

        # Concurrency control
        self._strategy_semaphore = asyncio.BoundedSemaphore(max_concurrent_strategies)
        self._api_semaphore = asyncio.BoundedSemaphore(max_concurrent_api_requests)

        # Thread pool for CPU-intensive VectorBT operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_concurrent_strategies, thread_name_prefix="vectorbt-worker"
        )

        # HTTP session for connection pooling
        self._http_session: aiohttp.ClientSession | None = None

        # Components
        self.cache_manager = cache_manager or CacheManager()
        self.data_provider = EnhancedStockDataProvider()

        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info(
            f"Initialized StrategyExecutor: "
            f"max_strategies={max_concurrent_strategies}, "
            f"max_api_requests={max_concurrent_api_requests}, "
            f"pool_size={connection_pool_size}"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_http_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()

    async def _initialize_http_session(self):
        """Initialize HTTP session with connection pooling."""
        if self._http_session is None:
            connector = TCPConnector(
                limit=self.connection_pool_size,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )

            timeout = ClientTimeout(total=self.request_timeout)

            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "MaverickMCP/1.0",
                    "Accept": "application/json",
                },
            )

            logger.info("HTTP session initialized with connection pooling")

    async def _cleanup(self):
        """Cleanup resources."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        self._thread_pool.shutdown(wait=True)
        logger.info("Resources cleaned up")

    async def execute_strategies_parallel(
        self, contexts: list[ExecutionContext]
    ) -> list[ExecutionResult]:
        """
        Execute multiple strategies in parallel with concurrency control.

        Args:
            contexts: List of execution contexts

        Returns:
            List of execution results
        """
        if not contexts:
            return []

        logger.info(f"Starting parallel execution of {len(contexts)} strategies")
        start_time = time.time()

        # Ensure HTTP session is initialized
        await self._initialize_http_session()

        # Pre-fetch all required data in batches
        await self._prefetch_data_batch(contexts)

        # Execute strategies with concurrency control
        tasks = [
            self._execute_single_strategy_with_semaphore(context)
            for context in contexts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ExecutionResult(
                        context=contexts[i],
                        success=False,
                        error=f"Execution failed: {str(result)}",
                        execution_time=0.0,
                    )
                )
            else:
                processed_results.append(result)

        total_time = time.time() - start_time
        self._update_stats(processed_results, total_time)

        logger.info(
            f"Parallel execution completed in {total_time:.2f}s: "
            f"{sum(1 for r in processed_results if r.success)}/{len(processed_results)} successful"
        )

        return processed_results

    async def _execute_single_strategy_with_semaphore(
        self, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute single strategy with semaphore control."""
        async with self._strategy_semaphore:
            return await self._execute_single_strategy(context)

    async def _execute_single_strategy(
        self, context: ExecutionContext
    ) -> ExecutionResult:
        """
        Execute a single strategy with thread safety.

        Args:
            context: Execution context

        Returns:
            Execution result
        """
        start_time = time.time()

        try:
            # Create isolated VectorBT engine for thread safety
            engine = VectorBTEngine(
                data_provider=self.data_provider, cache_service=self.cache_manager
            )

            # Execute in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._thread_pool, self._run_backtest_sync, engine, context
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                context=context,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Strategy execution failed for {context.strategy_id}: {e}")

            return ExecutionResult(
                context=context,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    def _run_backtest_sync(
        self, engine: VectorBTEngine, context: ExecutionContext
    ) -> dict[str, Any]:
        """
        Run backtest synchronously in thread pool.

        This method runs in a separate thread to avoid blocking the event loop.
        """
        # Use synchronous approach since we're in a thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                engine.run_backtest(
                    symbol=context.symbol,
                    strategy_type=context.strategy_type,
                    parameters=context.parameters,
                    start_date=context.start_date,
                    end_date=context.end_date,
                    initial_capital=context.initial_capital,
                    fees=context.fees,
                    slippage=context.slippage,
                )
            )
            return result
        finally:
            loop.close()

    async def _prefetch_data_batch(self, contexts: list[ExecutionContext]):
        """
        Pre-fetch all required data in batches to improve cache efficiency.

        Args:
            contexts: List of execution contexts
        """
        # Group by symbol and date range for efficient batching
        data_requests = {}
        for context in contexts:
            key = (context.symbol, context.start_date, context.end_date)
            if key not in data_requests:
                data_requests[key] = []
            data_requests[key].append(context.strategy_id)

        logger.info(
            f"Pre-fetching data for {len(data_requests)} unique symbol/date combinations"
        )

        # Batch fetch with concurrency control
        fetch_tasks = [
            self._fetch_data_with_rate_limit(symbol, start_date, end_date)
            for (symbol, start_date, end_date) in data_requests.keys()
        ]

        await asyncio.gather(*fetch_tasks, return_exceptions=True)

    async def _fetch_data_with_rate_limit(
        self, symbol: str, start_date: str, end_date: str
    ):
        """Fetch data with rate limiting."""
        async with self._api_semaphore:
            try:
                # Add small delay to prevent API hammering
                await asyncio.sleep(0.05)

                # Pre-fetch data into cache
                await self.data_provider.get_stock_data_async(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )

                self._stats["cache_misses"] += 1

            except Exception as e:
                logger.warning(f"Failed to pre-fetch data for {symbol}: {e}")

    async def batch_get_stock_data(
        self, symbols: list[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not symbols:
            return {}

        logger.info(f"Batch fetching data for {len(symbols)} symbols")

        # Ensure HTTP session is initialized
        await self._initialize_http_session()

        # Create tasks with rate limiting
        tasks = [
            self._get_single_stock_data_with_retry(
                symbol, start_date, end_date, interval
            )
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        data_dict = {}
        for symbol, result in zip(symbols, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {symbol}: {result}")
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result

        successful_fetches = sum(1 for df in data_dict.values() if not df.empty)
        logger.info(
            f"Batch fetch completed: {successful_fetches}/{len(symbols)} successful"
        )

        return data_dict

    async def _get_single_stock_data_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Get single stock data with exponential backoff retry."""
        async with self._api_semaphore:
            for attempt in range(max_retries):
                try:
                    # Add progressive delay to prevent API rate limiting
                    if attempt > 0:
                        delay = min(2**attempt, 10)  # Exponential backoff, max 10s
                        await asyncio.sleep(delay)

                    # Check cache first
                    data = await self._check_cache_for_data(
                        symbol, start_date, end_date, interval
                    )
                    if data is not None:
                        self._stats["cache_hits"] += 1
                        return data

                    # Fetch from provider
                    data = await self.data_provider.get_stock_data_async(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                    )

                    if data is not None and not data.empty:
                        self._stats["cache_misses"] += 1
                        return data

                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {symbol}: {e}")
                        raise
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {symbol}: {e}"
                        )

            return pd.DataFrame()

    async def _check_cache_for_data(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame | None:
        """Check cache for existing data."""
        try:
            cache_key = f"stock_data_{symbol}_{start_date}_{end_date}_{interval}"
            cached_data = await self.cache_manager.get(cache_key)

            if cached_data is not None:
                if isinstance(cached_data, pd.DataFrame):
                    return cached_data
                else:
                    # Convert from dict format
                    return pd.DataFrame.from_dict(cached_data, orient="index")

        except Exception as e:
            logger.debug(f"Cache check failed for {symbol}: {e}")

        return None

    def _update_stats(self, results: list[ExecutionResult], total_time: float):
        """Update execution statistics."""
        self._stats["total_executions"] += len(results)
        self._stats["successful_executions"] += sum(1 for r in results if r.success)
        self._stats["failed_executions"] += sum(1 for r in results if not r.success)
        self._stats["total_execution_time"] += total_time

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        stats = self._stats.copy()

        if stats["total_executions"] > 0:
            stats["success_rate"] = (
                stats["successful_executions"] / stats["total_executions"]
            )
            stats["avg_execution_time"] = (
                stats["total_execution_time"] / stats["total_executions"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["avg_execution_time"] = 0.0

        if stats["cache_hits"] + stats["cache_misses"] > 0:
            total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0.0

        return stats

    def reset_statistics(self):
        """Reset execution statistics."""
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }


@asynccontextmanager
async def get_strategy_executor(**kwargs):
    """Context manager for strategy executor with automatic cleanup."""
    executor = StrategyExecutor(**kwargs)
    try:
        async with executor:
            yield executor
    finally:
        # Cleanup is handled by __aexit__
        pass


# Utility functions for easy parallel execution


async def execute_strategies_parallel(
    contexts: list[ExecutionContext], max_concurrent: int = 6
) -> list[ExecutionResult]:
    """Convenience function for parallel strategy execution."""
    async with get_strategy_executor(
        max_concurrent_strategies=max_concurrent
    ) as executor:
        return await executor.execute_strategies_parallel(contexts)


async def batch_fetch_stock_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    max_concurrent: int = 10,
) -> dict[str, pd.DataFrame]:
    """Convenience function for batch stock data fetching."""
    async with get_strategy_executor(
        max_concurrent_api_requests=max_concurrent
    ) as executor:
        return await executor.batch_get_stock_data(
            symbols, start_date, end_date, interval
        )
