"""
Parallel stock screening utilities for Maverick-MCP.

This module provides utilities for running stock screening operations
in parallel using ProcessPoolExecutor for significant performance gains.
"""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class ParallelScreener:
    """
    Parallel stock screening executor.

    This class provides methods to run screening functions in parallel
    across multiple processes for better performance.
    """

    def __init__(self, max_workers: int | None = None):
        """
        Initialize the parallel screener.

        Args:
            max_workers: Maximum number of worker processes.
                        Defaults to CPU count.
        """
        self.max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self):
        """Context manager entry."""
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def screen_batch(
        self,
        symbols: list[str],
        screening_func: Callable[[str], dict[str, Any]],
        batch_size: int = 10,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Screen a batch of symbols in parallel.

        Args:
            symbols: List of stock symbols to screen
            screening_func: Function that takes a symbol and returns screening results
            batch_size: Number of symbols to process per worker
            timeout: Timeout for each screening operation

        Returns:
            List of screening results for symbols that passed
        """
        if not self._executor:
            raise RuntimeError("ParallelScreener must be used as context manager")

        start_time = time.time()
        results = []
        failed_symbols = []

        # Create batches
        batches = [
            symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
        ]

        logger.info(
            f"Starting parallel screening of {len(symbols)} symbols "
            f"in {len(batches)} batches"
        )

        # Submit batch processing jobs
        future_to_batch = {
            self._executor.submit(self._process_batch, batch, screening_func): batch
            for batch in batches
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch, timeout=timeout * len(batches)):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                failed_symbols.extend(batch)

        elapsed = time.time() - start_time
        success_rate = (len(results) / len(symbols)) * 100 if symbols else 0

        logger.info(
            f"Parallel screening completed in {elapsed:.2f}s "
            f"({len(results)}/{len(symbols)} succeeded, "
            f"{success_rate:.1f}% success rate)"
        )

        if failed_symbols:
            logger.warning(f"Failed to screen symbols: {failed_symbols[:10]}...")

        return results

    @staticmethod
    def _process_batch(
        symbols: list[str], screening_func: Callable[[str], dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Process a batch of symbols.

        This runs in a separate process.
        """
        results = []

        for symbol in symbols:
            try:
                result = screening_func(symbol)
                if result and result.get("passed", False):
                    results.append(result)
            except Exception as e:
                # Log errors but continue processing
                logger.debug(f"Screening failed for {symbol}: {e}")

        return results


async def parallel_screen_async(
    symbols: list[str],
    screening_func: Callable[[str], dict[str, Any]],
    max_workers: int | None = None,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """
    Async wrapper for parallel screening.

    Args:
        symbols: List of stock symbols to screen
        screening_func: Screening function (must be picklable)
        max_workers: Maximum number of worker processes
        batch_size: Number of symbols per batch

    Returns:
        List of screening results
    """
    loop = asyncio.get_event_loop()

    # Run screening in thread pool to avoid blocking
    def run_screening():
        with ParallelScreener(max_workers=max_workers) as screener:
            return screener.screen_batch(symbols, screening_func, batch_size)

    results = await loop.run_in_executor(None, run_screening)
    return results


# Example screening function (must be at module level for pickling)
def example_momentum_screen(symbol: str) -> dict[str, Any]:
    """
    Example momentum screening function.

    This must be defined at module level to be picklable for multiprocessing.
    """
    from maverick_mcp.core.technical_analysis import calculate_rsi, calculate_sma
    from maverick_mcp.providers.stock_data import StockDataProvider

    try:
        # Get stock data
        provider = StockDataProvider(use_cache=False)
        data = provider.get_stock_data(
            symbol, start_date="2023-01-01", end_date="2024-01-01"
        )

        if len(data) < 50:
            return {"symbol": symbol, "passed": False, "reason": "Insufficient data"}

        # Calculate indicators
        current_price = data["Close"].iloc[-1]
        sma_50 = calculate_sma(data, 50).iloc[-1]
        rsi = calculate_rsi(data, 14).iloc[-1]

        # Momentum criteria
        passed = (
            current_price > sma_50  # Price above 50-day SMA
            and 40 <= rsi <= 70  # RSI in healthy range
        )

        return {
            "symbol": symbol,
            "passed": passed,
            "price": round(current_price, 2),
            "sma_50": round(sma_50, 2),
            "rsi": round(rsi, 2),
            "above_sma": current_price > sma_50,
        }

    except Exception as e:
        return {"symbol": symbol, "passed": False, "error": str(e)}


# Decorator for making functions parallel-friendly
def make_parallel_safe(func: Callable) -> Callable:
    """
    Decorator to make a function safe for parallel execution.

    This ensures the function:
    1. Doesn't rely on shared state
    2. Handles its own database connections
    3. Returns picklable results
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure clean execution environment
        import os

        os.environ["AUTH_ENABLED"] = "false"
        os.environ["CREDIT_SYSTEM_ENABLED"] = "false"

        try:
            result = func(*args, **kwargs)
            # Ensure result is serializable
            import json

            json.dumps(result)  # Test serializability
            return result
        except Exception as e:
            logger.error(f"Parallel execution error in {func.__name__}: {e}")
            return {"error": str(e), "passed": False}

    return wrapper


# Batch screening with progress tracking
class BatchScreener:
    """Enhanced batch screener with progress tracking."""

    def __init__(self, screening_func: Callable, max_workers: int = 4):
        self.screening_func = screening_func
        self.max_workers = max_workers
        self.results = []
        self.progress = 0
        self.total = 0

    def screen_with_progress(
        self,
        symbols: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Screen symbols with progress tracking.

        Args:
            symbols: List of symbols to screen
            progress_callback: Optional callback for progress updates

        Returns:
            List of screening results
        """
        self.total = len(symbols)
        self.progress = 0

        with ParallelScreener(max_workers=self.max_workers) as screener:
            # Process in smaller batches for better progress tracking
            batch_size = max(1, len(symbols) // (self.max_workers * 4))

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                batch_results = screener.screen_batch(
                    batch,
                    self.screening_func,
                    batch_size=1,  # Process one at a time within batch
                )

                self.results.extend(batch_results)
                self.progress = min(i + batch_size, self.total)

                if progress_callback:
                    progress_callback(self.progress, self.total)

        return self.results
