"""
Data chunking utilities for memory-efficient processing of large datasets.
Provides streaming, batching, and generator-based approaches for handling large DataFrames.
"""

import logging
import math
from typing import Any, Callable, Generator, Iterator, Literal
import warnings

import numpy as np
import pandas as pd

from maverick_mcp.utils.memory_profiler import (
    memory_context,
    optimize_dataframe,
    force_garbage_collection,
    get_dataframe_memory_usage,
)

logger = logging.getLogger(__name__)

# Default chunk size configurations
DEFAULT_CHUNK_SIZE_MB = 50.0
MAX_CHUNK_SIZE_MB = 200.0
MIN_ROWS_PER_CHUNK = 100


class DataChunker:
    """Advanced data chunking utility with multiple strategies."""

    def __init__(self,
                 chunk_size_mb: float = DEFAULT_CHUNK_SIZE_MB,
                 min_rows_per_chunk: int = MIN_ROWS_PER_CHUNK,
                 optimize_chunks: bool = True,
                 auto_gc: bool = True):
        """Initialize data chunker.

        Args:
            chunk_size_mb: Target chunk size in megabytes
            min_rows_per_chunk: Minimum rows per chunk
            optimize_chunks: Whether to optimize chunk memory usage
            auto_gc: Whether to automatically run garbage collection
        """
        self.chunk_size_mb = min(chunk_size_mb, MAX_CHUNK_SIZE_MB)
        self.chunk_size_bytes = int(self.chunk_size_mb * 1024 * 1024)
        self.min_rows_per_chunk = min_rows_per_chunk
        self.optimize_chunks = optimize_chunks
        self.auto_gc = auto_gc

        logger.debug(f"DataChunker initialized: {self.chunk_size_mb}MB chunks, "
                    f"min {self.min_rows_per_chunk} rows")

    def estimate_chunk_size(self, df: pd.DataFrame) -> tuple[int, int]:
        """Estimate optimal chunk size for a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Tuple of (rows_per_chunk, estimated_chunks)
        """
        total_memory = df.memory_usage(deep=True).sum()
        memory_per_row = total_memory / len(df) if len(df) > 0 else 0

        if memory_per_row == 0:
            return len(df), 1

        # Calculate rows per chunk based on memory target
        rows_per_chunk = max(
            self.min_rows_per_chunk,
            int(self.chunk_size_bytes / memory_per_row)
        )

        # Ensure we don't exceed the DataFrame size
        rows_per_chunk = min(rows_per_chunk, len(df))

        estimated_chunks = math.ceil(len(df) / rows_per_chunk)

        logger.debug(f"Estimated chunking: {rows_per_chunk} rows/chunk, "
                    f"{estimated_chunks} chunks total")

        return rows_per_chunk, estimated_chunks

    def chunk_by_rows(self, df: pd.DataFrame,
                      rows_per_chunk: int = None) -> Generator[pd.DataFrame, None, None]:
        """Chunk DataFrame by number of rows.

        Args:
            df: DataFrame to chunk
            rows_per_chunk: Rows per chunk (auto-estimated if None)

        Yields:
            DataFrame chunks
        """
        if rows_per_chunk is None:
            rows_per_chunk, _ = self.estimate_chunk_size(df)

        total_chunks = math.ceil(len(df) / rows_per_chunk)
        logger.debug(f"Chunking {len(df)} rows into {total_chunks} chunks "
                    f"of ~{rows_per_chunk} rows each")

        for i, start_idx in enumerate(range(0, len(df), rows_per_chunk)):
            end_idx = min(start_idx + rows_per_chunk, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()

            if self.optimize_chunks:
                chunk = optimize_dataframe(chunk)

            logger.debug(f"Yielding chunk {i + 1}/{total_chunks}: "
                        f"rows {start_idx}-{end_idx-1}")

            yield chunk

            # Cleanup after yielding
            if self.auto_gc:
                del chunk
                if i % 5 == 0:  # GC every 5 chunks
                    force_garbage_collection()

    def chunk_by_memory(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Chunk DataFrame by memory size.

        Args:
            df: DataFrame to chunk

        Yields:
            DataFrame chunks
        """
        total_memory = df.memory_usage(deep=True).sum()

        if total_memory <= self.chunk_size_bytes:
            if self.optimize_chunks:
                df = optimize_dataframe(df)
            yield df
            return

        # Use row-based chunking with memory-based estimation
        yield from self.chunk_by_rows(df)

    def chunk_by_date(self, df: pd.DataFrame,
                     freq: Literal['D', 'W', 'M', 'Q', 'Y'] = 'M',
                     date_column: str = None) -> Generator[pd.DataFrame, None, None]:
        """Chunk DataFrame by date periods.

        Args:
            df: DataFrame to chunk (must have datetime index or date_column)
            freq: Frequency for chunking (D=daily, W=weekly, M=monthly, etc.)
            date_column: Name of date column (uses index if None)

        Yields:
            DataFrame chunks by date periods
        """
        if date_column:
            if date_column not in df.columns:
                raise ValueError(f"Date column '{date_column}' not found")
            date_series = df[date_column]
        elif isinstance(df.index, pd.DatetimeIndex):
            date_series = df.index.to_series()
        else:
            raise ValueError("DataFrame must have datetime index or specify date_column")

        # Group by period
        period_groups = df.groupby(pd.Grouper(key=date_column, freq=freq)
                                  if date_column else pd.Grouper(freq=freq))

        total_periods = len(period_groups)
        logger.debug(f"Chunking by {freq} periods: {total_periods} chunks")

        for i, (period, group) in enumerate(period_groups):
            if len(group) == 0:
                continue

            if self.optimize_chunks:
                group = optimize_dataframe(group)

            logger.debug(f"Yielding period chunk {i + 1}/{total_periods}: "
                        f"{period} ({len(group)} rows)")

            yield group

            if self.auto_gc and i % 3 == 0:  # GC every 3 periods
                force_garbage_collection()

    def process_in_chunks(self,
                         df: pd.DataFrame,
                         processor: Callable[[pd.DataFrame], Any],
                         combiner: Callable[[list], Any] = None,
                         chunk_method: Literal['rows', 'memory', 'date'] = 'memory',
                         **chunk_kwargs) -> Any:
        """Process DataFrame in chunks and combine results.

        Args:
            df: DataFrame to process
            processor: Function to apply to each chunk
            combiner: Function to combine results (default: list)
            chunk_method: Chunking method to use
            **chunk_kwargs: Additional arguments for chunking method

        Returns:
            Combined results
        """
        results = []

        # Select chunking method
        if chunk_method == 'rows':
            chunk_generator = self.chunk_by_rows(df, **chunk_kwargs)
        elif chunk_method == 'memory':
            chunk_generator = self.chunk_by_memory(df)
        elif chunk_method == 'date':
            chunk_generator = self.chunk_by_date(df, **chunk_kwargs)
        else:
            raise ValueError(f"Unknown chunk method: {chunk_method}")

        with memory_context("chunk_processing"):
            for i, chunk in enumerate(chunk_generator):
                try:
                    with memory_context(f"chunk_{i}"):
                        result = processor(chunk)
                        results.append(result)

                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    raise

        # Combine results
        if combiner:
            return combiner(results)
        elif results and isinstance(results[0], pd.DataFrame):
            # Auto-combine DataFrames
            return pd.concat(results, ignore_index=True)
        else:
            return results


class StreamingDataProcessor:
    """Streaming data processor for very large datasets."""

    def __init__(self, chunk_size_mb: float = DEFAULT_CHUNK_SIZE_MB):
        """Initialize streaming processor.

        Args:
            chunk_size_mb: Chunk size in MB
        """
        self.chunk_size_mb = chunk_size_mb
        self.chunker = DataChunker(chunk_size_mb=chunk_size_mb)

    def stream_from_csv(self,
                       filepath: str,
                       processor: Callable[[pd.DataFrame], Any],
                       chunksize: int = None,
                       **read_kwargs) -> Generator[Any, None, None]:
        """Stream process CSV file in chunks.

        Args:
            filepath: Path to CSV file
            processor: Function to process each chunk
            chunksize: Rows per chunk (auto-estimated if None)
            **read_kwargs: Additional arguments for pd.read_csv

        Yields:
            Processed results for each chunk
        """
        # Estimate chunk size if not provided
        if chunksize is None:
            # Read a sample to estimate memory usage
            sample = pd.read_csv(filepath, nrows=1000, **read_kwargs)
            memory_per_row = sample.memory_usage(deep=True).sum() / len(sample)
            chunksize = max(100, int(self.chunker.chunk_size_bytes / memory_per_row))
            del sample
            force_garbage_collection()

        logger.info(f"Streaming CSV with {chunksize} rows per chunk")

        chunk_reader = pd.read_csv(filepath, chunksize=chunksize, **read_kwargs)

        for i, chunk in enumerate(chunk_reader):
            with memory_context(f"csv_chunk_{i}"):
                # Optimize chunk if needed
                if self.chunker.optimize_chunks:
                    chunk = optimize_dataframe(chunk)

                result = processor(chunk)
                yield result

            # Clean up
            del chunk
            if i % 5 == 0:
                force_garbage_collection()

    def stream_from_database(self,
                            query: str,
                            connection,
                            processor: Callable[[pd.DataFrame], Any],
                            chunksize: int = None) -> Generator[Any, None, None]:
        """Stream process database query results in chunks.

        Args:
            query: SQL query
            connection: Database connection
            processor: Function to process each chunk
            chunksize: Rows per chunk

        Yields:
            Processed results for each chunk
        """
        if chunksize is None:
            chunksize = 10000  # Default for database queries

        logger.info(f"Streaming database query with {chunksize} rows per chunk")

        chunk_reader = pd.read_sql(query, connection, chunksize=chunksize)

        for i, chunk in enumerate(chunk_reader):
            with memory_context(f"db_chunk_{i}"):
                if self.chunker.optimize_chunks:
                    chunk = optimize_dataframe(chunk)

                result = processor(chunk)
                yield result

            del chunk
            if i % 3 == 0:
                force_garbage_collection()


def optimize_dataframe_dtypes(df: pd.DataFrame,
                             aggressive: bool = False,
                             categorical_threshold: float = 0.5) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency.

    Args:
        df: DataFrame to optimize
        aggressive: Use aggressive optimizations (may lose precision)
        categorical_threshold: Threshold for categorical conversion

    Returns:
        Optimized DataFrame
    """
    logger.debug(f"Optimizing DataFrame dtypes: {df.shape}")

    initial_memory = df.memory_usage(deep=True).sum()
    df_opt = df.copy()

    for col in df_opt.columns:
        col_type = df_opt[col].dtype

        try:
            if col_type == 'object':
                # Convert string columns to categorical if beneficial
                unique_count = df_opt[col].nunique()
                total_count = len(df_opt[col])

                if unique_count / total_count < categorical_threshold:
                    df_opt[col] = df_opt[col].astype('category')
                    logger.debug(f"Converted {col} to categorical")

            elif 'int' in str(col_type):
                # Downcast integers
                c_min = df_opt[col].min()
                c_max = df_opt[col].max()

                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df_opt[col] = df_opt[col].astype(np.int32)

            elif 'float' in str(col_type) and col_type == 'float64':
                # Downcast float64 to float32 if no precision loss
                if aggressive:
                    # Check if conversion preserves data
                    temp = df_opt[col].astype(np.float32)
                    if np.allclose(df_opt[col].fillna(0), temp.fillna(0),
                                  rtol=1e-6, equal_nan=True):
                        df_opt[col] = temp
                        logger.debug(f"Converted {col} to float32")

        except Exception as e:
            logger.debug(f"Could not optimize column {col}: {e}")
            continue

    final_memory = df_opt.memory_usage(deep=True).sum()
    memory_saved = initial_memory - final_memory

    if memory_saved > 0:
        logger.info(f"DataFrame optimization saved {memory_saved / (1024**2):.2f}MB "
                   f"({memory_saved / initial_memory * 100:.1f}% reduction)")

    return df_opt


def create_memory_efficient_dataframe(data: dict | list,
                                     optimize: bool = True,
                                     categorical_columns: list[str] = None) -> pd.DataFrame:
    """Create a memory-efficient DataFrame from data.

    Args:
        data: Data to create DataFrame from
        optimize: Whether to optimize dtypes
        categorical_columns: Columns to convert to categorical

    Returns:
        Memory-optimized DataFrame
    """
    with memory_context("creating_dataframe"):
        df = pd.DataFrame(data)

        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        if optimize:
            df = optimize_dataframe_dtypes(df)

        return df


def batch_process_large_dataframe(df: pd.DataFrame,
                                 operation: Callable,
                                 batch_size: int = None,
                                 combine_results: bool = True) -> Any:
    """Process large DataFrame in batches to manage memory.

    Args:
        df: Large DataFrame to process
        operation: Function to apply to each batch
        batch_size: Size of each batch (auto-estimated if None)
        combine_results: Whether to combine batch results

    Returns:
        Combined results or list of batch results
    """
    chunker = DataChunker()

    if batch_size:
        chunk_generator = chunker.chunk_by_rows(df, batch_size)
    else:
        chunk_generator = chunker.chunk_by_memory(df)

    results = []

    with memory_context("batch_processing"):
        for i, batch in enumerate(chunk_generator):
            logger.debug(f"Processing batch {i + 1}")

            with memory_context(f"batch_{i}"):
                result = operation(batch)
                results.append(result)

    if combine_results and results:
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(results[0], (int, float)):
            return sum(results)
        elif isinstance(results[0], list):
            return [item for sublist in results for item in sublist]

    return results


class LazyDataFrame:
    """Lazy evaluation wrapper for large DataFrames."""

    def __init__(self, data_source: str | pd.DataFrame, chunk_size_mb: float = 50.0):
        """Initialize lazy DataFrame.

        Args:
            data_source: File path or DataFrame
            chunk_size_mb: Chunk size for processing
        """
        self.data_source = data_source
        self.chunker = DataChunker(chunk_size_mb=chunk_size_mb)
        self._cached_info = None

    def get_info(self) -> dict[str, Any]:
        """Get DataFrame information without loading full data."""
        if self._cached_info:
            return self._cached_info

        if isinstance(self.data_source, str):
            # Read just the header and a sample
            sample = pd.read_csv(self.data_source, nrows=100)
            total_rows = sum(1 for _ in open(self.data_source)) - 1  # Subtract header

            self._cached_info = {
                "columns": sample.columns.tolist(),
                "dtypes": sample.dtypes.to_dict(),
                "estimated_rows": total_rows,
                "sample_memory_mb": sample.memory_usage(deep=True).sum() / (1024**2),
            }
        else:
            self._cached_info = get_dataframe_memory_usage(self.data_source)

        return self._cached_info

    def apply_chunked(self, operation: Callable) -> Any:
        """Apply operation in chunks."""
        if isinstance(self.data_source, str):
            processor = StreamingDataProcessor(self.chunker.chunk_size_mb)
            results = list(processor.stream_from_csv(self.data_source, operation))
        else:
            results = self.chunker.process_in_chunks(self.data_source, operation)

        return results

    def to_optimized_dataframe(self) -> pd.DataFrame:
        """Load and optimize the full DataFrame."""
        if isinstance(self.data_source, str):
            df = pd.read_csv(self.data_source)
        else:
            df = self.data_source.copy()

        return optimize_dataframe_dtypes(df)


# Utility functions for common operations

def chunked_concat(dataframes: list[pd.DataFrame],
                  chunk_size: int = 10) -> pd.DataFrame:
    """Concatenate DataFrames in chunks to manage memory.

    Args:
        dataframes: List of DataFrames to concatenate
        chunk_size: Number of DataFrames to concat at once

    Returns:
        Concatenated DataFrame
    """
    if not dataframes:
        return pd.DataFrame()

    if len(dataframes) <= chunk_size:
        return pd.concat(dataframes, ignore_index=True)

    # Process in chunks
    results = []
    for i in range(0, len(dataframes), chunk_size):
        chunk = dataframes[i:i + chunk_size]
        with memory_context(f"concat_chunk_{i//chunk_size}"):
            result = pd.concat(chunk, ignore_index=True)
            results.append(result)

        # Clean up chunk
        for df in chunk:
            del df
        force_garbage_collection()

    # Final concatenation
    with memory_context("final_concat"):
        final_result = pd.concat(results, ignore_index=True)

    return final_result


def memory_efficient_groupby(df: pd.DataFrame,
                            group_col: str,
                            agg_func: Callable,
                            chunk_size_mb: float = 50.0) -> pd.DataFrame:
    """Perform memory-efficient groupby operations.

    Args:
        df: DataFrame to group
        group_col: Column to group by
        agg_func: Aggregation function
        chunk_size_mb: Chunk size in MB

    Returns:
        Aggregated DataFrame
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found")

    chunker = DataChunker(chunk_size_mb=chunk_size_mb)
    results = []

    def process_chunk(chunk):
        return chunk.groupby(group_col).apply(agg_func).reset_index()

    results = chunker.process_in_chunks(df, process_chunk)

    # Combine and re-aggregate results
    combined = pd.concat(results, ignore_index=True)
    final_result = combined.groupby(group_col).apply(agg_func).reset_index()

    return final_result