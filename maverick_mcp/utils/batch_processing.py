"""
Batch processing utilities for efficient multi-symbol operations.

This module provides utilities for processing multiple stock symbols
efficiently using concurrent execution and batching strategies.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import pandas as pd
import yfinance as yf

from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class BatchProcessor:
    """
    Utility class for efficient batch processing of stock operations.

    Provides methods for processing multiple symbols concurrently
    with proper error handling and resource management.
    """

    def __init__(self, max_workers: int = 10, batch_size: int = 50):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Maximum number of symbols per batch
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.executor.shutdown(wait=True)

    def process_symbols_concurrent(
        self,
        symbols: list[str],
        processor_func: Callable[[str], T],
        error_default: T | None = None,
    ) -> dict[str, T]:
        """
        Process multiple symbols concurrently using ThreadPoolExecutor.

        Args:
            symbols: List of stock symbols to process
            processor_func: Function to apply to each symbol
            error_default: Default value to return on error

        Returns:
            Dictionary mapping symbols to their processed results
        """
        results = {}

        # Submit all tasks
        future_to_symbol = {
            self.executor.submit(processor_func, symbol): symbol for symbol in symbols
        }

        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results[symbol] = result
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                if error_default is not None:
                    results[symbol] = error_default

        return results

    async def process_symbols_async(
        self,
        symbols: list[str],
        async_processor_func: Callable[[str], Any],
        max_concurrent: int | None = None,
    ) -> dict[str, Any]:
        """
        Process multiple symbols asynchronously with concurrency limit.

        Args:
            symbols: List of stock symbols to process
            async_processor_func: Async function to apply to each symbol
            max_concurrent: Maximum concurrent operations (defaults to max_workers)

        Returns:
            Dictionary mapping symbols to their processed results
        """
        if max_concurrent is None:
            max_concurrent = self.max_workers

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(symbol: str):
            async with semaphore:
                try:
                    return symbol, await async_processor_func(symbol)
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    return symbol, None

        # Process all symbols concurrently
        tasks = [process_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        # Convert to dictionary, filtering out None results
        return {symbol: result for symbol, result in results if result is not None}

    def process_in_batches(
        self,
        symbols: list[str],
        batch_processor_func: Callable[[list[str]], dict[str, T]],
    ) -> dict[str, T]:
        """
        Process symbols in batches for improved efficiency.

        Args:
            symbols: List of stock symbols to process
            batch_processor_func: Function that processes a batch of symbols

        Returns:
            Dictionary mapping symbols to their processed results
        """
        results = {}

        # Process symbols in batches
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i : i + self.batch_size]
            try:
                batch_results = batch_processor_func(batch)
                results.update(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch {i // self.batch_size + 1}: {e}")
                # Process individual symbols as fallback
                for symbol in batch:
                    try:
                        individual_result = batch_processor_func([symbol])
                        results.update(individual_result)
                    except Exception as e2:
                        logger.warning(
                            f"Error processing individual symbol {symbol}: {e2}"
                        )

        return results


class StockDataBatchProcessor:
    """Specialized batch processor for stock data operations."""

    def __init__(self, provider: EnhancedStockDataProvider | None = None):
        """Initialize with optional stock data provider."""
        self.provider = provider or EnhancedStockDataProvider()

    def get_batch_stock_data(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols efficiently using yfinance batch download.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        try:
            # Use yfinance batch download for efficiency
            tickers_str = " ".join(symbols)
            data = yf.download(
                tickers_str,
                start=start_date,
                end=end_date,
                group_by="ticker",
                threads=True,
                progress=False,
            )

            results = {}

            if len(symbols) == 1:
                # Single symbol case
                symbol = symbols[0]
                if not data.empty:
                    # Standardize column names
                    df = data.copy()
                    if "Close" in df.columns:
                        df.columns = df.columns.str.title()
                    results[symbol] = df
            else:
                # Multiple symbols case
                for symbol in symbols:
                    try:
                        if symbol in data.columns.get_level_values(0):
                            symbol_data = data[symbol].copy()
                            # Remove any NaN-only rows
                            symbol_data = symbol_data.dropna(how="all")
                            if not symbol_data.empty:
                                results[symbol] = symbol_data
                    except Exception as e:
                        logger.warning(f"Error extracting data for {symbol}: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in batch stock data download: {e}")
            # Fallback to individual downloads
            return self._fallback_individual_downloads(symbols, start_date, end_date)

    def _fallback_individual_downloads(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """Fallback to individual downloads if batch fails."""
        results = {}

        with BatchProcessor(max_workers=5) as processor:

            def download_single(symbol: str) -> pd.DataFrame:
                try:
                    return self.provider.get_stock_data(symbol, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Error downloading {symbol}: {e}")
                    return pd.DataFrame()

            symbol_results = processor.process_symbols_concurrent(
                symbols, download_single, pd.DataFrame()
            )

            # Filter out empty DataFrames
            results = {
                symbol: df for symbol, df in symbol_results.items() if not df.empty
            }

        return results

    async def get_batch_stock_data_async(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """
        Async version of batch stock data fetching.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get_batch_stock_data, symbols, start_date, end_date
        )

    def get_batch_stock_info(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get stock info for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to their info dictionaries
        """
        with BatchProcessor(max_workers=10) as processor:

            def get_info(symbol: str) -> dict[str, Any]:
                try:
                    ticker = yf.Ticker(symbol)
                    return ticker.info
                except Exception as e:
                    logger.warning(f"Error getting info for {symbol}: {e}")
                    return {}

            return processor.process_symbols_concurrent(symbols, get_info, {})

    def get_batch_technical_analysis(
        self, symbols: list[str], days: int = 365
    ) -> dict[str, pd.DataFrame]:
        """
        Get technical analysis for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols
            days: Number of days of data

        Returns:
            Dictionary mapping symbols to DataFrames with technical indicators
        """
        from maverick_mcp.utils.stock_helpers import get_stock_dataframe

        with BatchProcessor(max_workers=8) as processor:

            def get_analysis(symbol: str) -> pd.DataFrame:
                try:
                    return get_stock_dataframe(symbol, days)
                except Exception as e:
                    logger.warning(
                        f"Error getting technical analysis for {symbol}: {e}"
                    )
                    return pd.DataFrame()

            results = processor.process_symbols_concurrent(
                symbols, get_analysis, pd.DataFrame()
            )

            # Filter out empty DataFrames
            return {symbol: df for symbol, df in results.items() if not df.empty}


# Convenience functions for common batch operations


def batch_download_stock_data(
    symbols: list[str], start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    """
    Convenience function for batch downloading stock data.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping symbols to their DataFrames
    """
    processor = StockDataBatchProcessor()
    return processor.get_batch_stock_data(symbols, start_date, end_date)


async def batch_download_stock_data_async(
    symbols: list[str], start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    """
    Convenience function for async batch downloading stock data.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping symbols to their DataFrames
    """
    processor = StockDataBatchProcessor()
    return await processor.get_batch_stock_data_async(symbols, start_date, end_date)


def batch_get_stock_info(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Convenience function for batch getting stock info.

    Args:
        symbols: List of stock symbols

    Returns:
        Dictionary mapping symbols to their info dictionaries
    """
    processor = StockDataBatchProcessor()
    return processor.get_batch_stock_info(symbols)


def batch_get_technical_analysis(
    symbols: list[str], days: int = 365
) -> dict[str, pd.DataFrame]:
    """
    Convenience function for batch technical analysis.

    Args:
        symbols: List of stock symbols
        days: Number of days of data

    Returns:
        Dictionary mapping symbols to DataFrames with technical indicators
    """
    processor = StockDataBatchProcessor()
    return processor.get_batch_technical_analysis(symbols, days)
