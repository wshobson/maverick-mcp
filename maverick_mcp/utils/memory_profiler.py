"""
Memory profiling and management utilities for the backtesting system.
Provides decorators, monitoring, and optimization tools for memory-efficient operations.
"""

import functools
import gc
import logging
import time
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)

# Memory threshold constants (in bytes)
MEMORY_WARNING_THRESHOLD = 1024 * 1024 * 1024  # 1GB
MEMORY_CRITICAL_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2GB
DATAFRAME_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100MB

# Global memory tracking
_memory_stats = {
    "peak_memory": 0,
    "current_memory": 0,
    "allocation_count": 0,
    "gc_count": 0,
    "warning_count": 0,
    "critical_count": 0,
    "dataframe_optimizations": 0,
}


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_memory: int
    vms_memory: int
    available_memory: int
    memory_percent: float
    peak_memory: int
    tracemalloc_current: int
    tracemalloc_peak: int
    function_name: str = ""


class MemoryProfiler:
    """Advanced memory profiler with tracking and optimization features."""

    def __init__(self, enable_tracemalloc: bool = True):
        """Initialize memory profiler.

        Args:
            enable_tracemalloc: Whether to enable detailed memory tracking
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshots: list[MemorySnapshot] = []
        self.process = psutil.Process()

        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()

    def get_memory_info(self) -> dict[str, Any]:
        """Get current memory information."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()

        result = {
            "rss_memory": memory_info.rss,
            "vms_memory": memory_info.vms,
            "available_memory": virtual_memory.available,
            "memory_percent": self.process.memory_percent(),
            "total_memory": virtual_memory.total,
        }

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            result.update({
                "tracemalloc_current": current,
                "tracemalloc_peak": peak,
            })

        return result

    def take_snapshot(self, function_name: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        memory_info = self.get_memory_info()

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_memory=memory_info["rss_memory"],
            vms_memory=memory_info["vms_memory"],
            available_memory=memory_info["available_memory"],
            memory_percent=memory_info["memory_percent"],
            peak_memory=memory_info.get("tracemalloc_peak", 0),
            tracemalloc_current=memory_info.get("tracemalloc_current", 0),
            tracemalloc_peak=memory_info.get("tracemalloc_peak", 0),
            function_name=function_name,
        )

        self.snapshots.append(snapshot)

        # Update global stats
        _memory_stats["current_memory"] = snapshot.rss_memory
        if snapshot.rss_memory > _memory_stats["peak_memory"]:
            _memory_stats["peak_memory"] = snapshot.rss_memory

        # Check thresholds
        self._check_memory_thresholds(snapshot)

        return snapshot

    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check memory thresholds and log warnings."""
        if snapshot.rss_memory > MEMORY_CRITICAL_THRESHOLD:
            _memory_stats["critical_count"] += 1
            logger.critical(
                f"CRITICAL: Memory usage {snapshot.rss_memory / (1024**3):.2f}GB "
                f"exceeds critical threshold in {snapshot.function_name or 'unknown'}"
            )
        elif snapshot.rss_memory > MEMORY_WARNING_THRESHOLD:
            _memory_stats["warning_count"] += 1
            logger.warning(
                f"WARNING: High memory usage {snapshot.rss_memory / (1024**3):.2f}GB "
                f"in {snapshot.function_name or 'unknown'}"
            )

    def get_memory_report(self) -> dict[str, Any]:
        """Generate comprehensive memory report."""
        if not self.snapshots:
            return {"error": "No memory snapshots available"}

        latest = self.snapshots[-1]
        first = self.snapshots[0]

        report = {
            "current_memory_mb": latest.rss_memory / (1024 ** 2),
            "peak_memory_mb": max(s.rss_memory for s in self.snapshots) / (1024 ** 2),
            "memory_growth_mb": (latest.rss_memory - first.rss_memory) / (1024 ** 2),
            "memory_percent": latest.memory_percent,
            "available_memory_gb": latest.available_memory / (1024 ** 3),
            "snapshots_count": len(self.snapshots),
            "warning_count": _memory_stats["warning_count"],
            "critical_count": _memory_stats["critical_count"],
            "gc_count": _memory_stats["gc_count"],
            "dataframe_optimizations": _memory_stats["dataframe_optimizations"],
        }

        if self.enable_tracemalloc:
            report.update({
                "tracemalloc_current_mb": latest.tracemalloc_current / (1024 ** 2),
                "tracemalloc_peak_mb": latest.tracemalloc_peak / (1024 ** 2),
            })

        return report


# Global profiler instance
_global_profiler = MemoryProfiler()


def get_memory_stats() -> dict[str, Any]:
    """Get global memory statistics."""
    return {**_memory_stats, **_global_profiler.get_memory_report()}


def reset_memory_stats() -> None:
    """Reset global memory statistics."""
    global _memory_stats
    _memory_stats = {
        "peak_memory": 0,
        "current_memory": 0,
        "allocation_count": 0,
        "gc_count": 0,
        "warning_count": 0,
        "critical_count": 0,
        "dataframe_optimizations": 0,
    }
    _global_profiler.snapshots.clear()


def profile_memory(func: Callable = None, *,
                   log_results: bool = True,
                   enable_gc: bool = True,
                   threshold_mb: float = 100.0):
    """Decorator to profile memory usage of a function.

    Args:
        func: Function to decorate
        log_results: Whether to log memory usage results
        enable_gc: Whether to trigger garbage collection
        threshold_mb: Memory usage threshold to log warnings (MB)
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            function_name = f.__name__

            # Take initial snapshot
            initial = _global_profiler.take_snapshot(f"start_{function_name}")

            try:
                # Execute function
                result = f(*args, **kwargs)

                # Take final snapshot
                final = _global_profiler.take_snapshot(f"end_{function_name}")

                # Calculate memory usage
                memory_diff_mb = (final.rss_memory - initial.rss_memory) / (1024 ** 2)

                if log_results:
                    if memory_diff_mb > threshold_mb:
                        logger.warning(
                            f"High memory usage in {function_name}: "
                            f"{memory_diff_mb:.2f}MB (threshold: {threshold_mb}MB)"
                        )
                    else:
                        logger.debug(
                            f"Memory usage in {function_name}: {memory_diff_mb:.2f}MB"
                        )

                # Trigger garbage collection if enabled
                if enable_gc and memory_diff_mb > threshold_mb:
                    force_garbage_collection()

                return result

            except Exception as e:
                # Take error snapshot
                _global_profiler.take_snapshot(f"error_{function_name}")
                raise e

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def memory_context(name: str = "operation",
                   cleanup_after: bool = True) -> Iterator[MemoryProfiler]:
    """Context manager for memory profiling operations.

    Args:
        name: Name of the operation
        cleanup_after: Whether to run garbage collection after

    Yields:
        MemoryProfiler instance for manual snapshots
    """
    profiler = MemoryProfiler()
    initial = profiler.take_snapshot(f"start_{name}")

    try:
        yield profiler
    finally:
        final = profiler.take_snapshot(f"end_{name}")

        memory_diff_mb = (final.rss_memory - initial.rss_memory) / (1024 ** 2)
        logger.debug(f"Memory usage in {name}: {memory_diff_mb:.2f}MB")

        if cleanup_after:
            force_garbage_collection()


def optimize_dataframe(df: pd.DataFrame,
                      aggressive: bool = False,
                      categorical_threshold: float = 0.5) -> pd.DataFrame:
    """Optimize DataFrame memory usage.

    Args:
        df: DataFrame to optimize
        aggressive: Whether to use aggressive optimizations
        categorical_threshold: Threshold for converting to categorical

    Returns:
        Optimized DataFrame
    """
    initial_memory = df.memory_usage(deep=True).sum()

    if initial_memory < DATAFRAME_SIZE_THRESHOLD:
        return df  # Skip optimization for small DataFrames

    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype

        if col_type == 'object':
            # Try to convert to categorical if many duplicates
            unique_ratio = df_optimized[col].nunique() / len(df_optimized[col])
            if unique_ratio < categorical_threshold:
                try:
                    df_optimized[col] = df_optimized[col].astype('category')
                except Exception:
                    pass

        elif 'int' in str(col_type):
            # Downcast integers
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)

        elif 'float' in str(col_type):
            # Downcast floats
            if aggressive:
                # Try float32 first
                try:
                    temp = df_optimized[col].astype(np.float32)
                    if np.allclose(df_optimized[col].fillna(0), temp.fillna(0),
                                  rtol=1e-6, equal_nan=True):
                        df_optimized[col] = temp
                except Exception:
                    pass

    final_memory = df_optimized.memory_usage(deep=True).sum()
    memory_saved = initial_memory - final_memory

    if memory_saved > 0:
        _memory_stats["dataframe_optimizations"] += 1
        logger.debug(
            f"DataFrame optimized: {memory_saved / (1024**2):.2f}MB saved "
            f"({memory_saved / initial_memory * 100:.1f}% reduction)"
        )

    return df_optimized


def force_garbage_collection() -> dict[str, int]:
    """Force garbage collection and return statistics."""
    collected = gc.collect()
    _memory_stats["gc_count"] += 1

    stats = {
        "collected": collected,
        "generation_0": len(gc.get_objects(0)),
        "generation_1": len(gc.get_objects(1)),
        "generation_2": len(gc.get_objects(2)),
        "total_objects": len(gc.get_objects()),
    }

    logger.debug(f"Garbage collection: {collected} objects collected")
    return stats


def check_memory_leak(threshold_mb: float = 100.0) -> bool:
    """Check for potential memory leaks.

    Args:
        threshold_mb: Memory growth threshold to consider a leak

    Returns:
        True if potential leak detected
    """
    if len(_global_profiler.snapshots) < 10:
        return False

    # Compare recent snapshots
    recent = _global_profiler.snapshots[-5:]
    older = _global_profiler.snapshots[-10:-5]

    recent_avg = sum(s.rss_memory for s in recent) / len(recent)
    older_avg = sum(s.rss_memory for s in older) / len(older)

    growth_mb = (recent_avg - older_avg) / (1024 ** 2)

    if growth_mb > threshold_mb:
        logger.warning(f"Potential memory leak detected: {growth_mb:.2f}MB growth")
        return True

    return False


class DataFrameChunker:
    """Utility for processing DataFrames in memory-efficient chunks."""

    def __init__(self, chunk_size_mb: float = 50.0):
        """Initialize chunker.

        Args:
            chunk_size_mb: Maximum chunk size in MB
        """
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)

    def chunk_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks based on memory size.

        Args:
            df: DataFrame to chunk

        Yields:
            DataFrame chunks
        """
        total_memory = df.memory_usage(deep=True).sum()

        if total_memory <= self.chunk_size_bytes:
            yield df
            return

        # Calculate approximate rows per chunk
        memory_per_row = total_memory / len(df)
        rows_per_chunk = max(1, int(self.chunk_size_bytes / memory_per_row))

        logger.debug(f"Chunking DataFrame: {len(df)} rows, "
                    f"~{rows_per_chunk} rows per chunk")

        for i in range(0, len(df), rows_per_chunk):
            chunk = df.iloc[i:i + rows_per_chunk]
            yield chunk

    def process_in_chunks(self,
                         df: pd.DataFrame,
                         processor: Callable[[pd.DataFrame], Any],
                         combine_results: Callable = None) -> Any:
        """Process DataFrame in chunks and optionally combine results.

        Args:
            df: DataFrame to process
            processor: Function to apply to each chunk
            combine_results: Function to combine chunk results

        Returns:
            Combined results or list of chunk results
        """
        results = []

        with memory_context("chunk_processing"):
            for i, chunk in enumerate(self.chunk_dataframe(df)):
                logger.debug(f"Processing chunk {i + 1}")

                with memory_context(f"chunk_{i}"):
                    result = processor(chunk)
                    results.append(result)

        if combine_results:
            return combine_results(results)

        return results


def cleanup_dataframes(*dfs: pd.DataFrame) -> None:
    """Clean up DataFrames and force garbage collection.

    Args:
        *dfs: DataFrames to clean up
    """
    for df in dfs:
        if hasattr(df, '_mgr'):
            # Clear internal references
            df._mgr = None
        del df

    force_garbage_collection()


def get_dataframe_memory_usage(df: pd.DataFrame) -> dict[str, Any]:
    """Get detailed memory usage information for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Memory usage statistics
    """
    memory_usage = df.memory_usage(deep=True)

    return {
        "total_memory_mb": memory_usage.sum() / (1024 ** 2),
        "index_memory_mb": memory_usage.iloc[0] / (1024 ** 2),
        "columns_memory_mb": {
            col: memory_usage.loc[col] / (1024 ** 2)
            for col in df.columns
        },
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "memory_per_row_bytes": memory_usage.sum() / len(df) if len(df) > 0 else 0,
    }


@contextmanager
def memory_limit_context(limit_mb: float) -> Iterator[None]:
    """Context manager to monitor memory usage within a limit.

    Args:
        limit_mb: Memory limit in MB

    Raises:
        MemoryError: If memory usage exceeds limit
    """
    initial_memory = psutil.Process().memory_info().rss
    limit_bytes = limit_mb * 1024 * 1024

    try:
        yield
    finally:
        current_memory = psutil.Process().memory_info().rss
        memory_used = current_memory - initial_memory

        if memory_used > limit_bytes:
            logger.error(f"Memory limit exceeded: {memory_used / (1024**2):.2f}MB > {limit_mb}MB")
            # Force cleanup
            force_garbage_collection()


def suggest_memory_optimizations(df: pd.DataFrame) -> list[str]:
    """Suggest memory optimizations for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        List of optimization suggestions
    """
    suggestions = []
    memory_info = get_dataframe_memory_usage(df)

    # Check for object columns that could be categorical
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                memory_savings = (memory_info["columns_memory_mb"][col] *
                                (1 - unique_ratio))
                suggestions.append(
                    f"Convert '{col}' to categorical (potential savings: "
                    f"{memory_savings:.2f}MB, {unique_ratio:.1%} unique values)"
                )

    # Check for float64 that could be float32
    for col in df.columns:
        if df[col].dtype == 'float64':
            try:
                temp = df[col].astype(np.float32)
                if np.allclose(df[col].fillna(0), temp.fillna(0), rtol=1e-6):
                    savings = memory_info["columns_memory_mb"][col] * 0.5
                    suggestions.append(
                        f"Convert '{col}' from float64 to float32 "
                        f"(potential savings: {savings:.2f}MB)"
                    )
            except Exception:
                pass

    # Check for integer downcasting opportunities
    for col in df.columns:
        if 'int' in str(df[col].dtype):
            c_min = df[col].min()
            c_max = df[col].max()
            current_bytes = df[col].memory_usage(deep=True) / len(df)

            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                if current_bytes > 1:
                    savings = (current_bytes - 1) * len(df) / (1024 ** 2)
                    suggestions.append(
                        f"Convert '{col}' to int8 (potential savings: {savings:.2f}MB)"
                    )

    return suggestions


# Initialize memory monitoring with warning suppression for resource warnings
def _suppress_resource_warnings():
    """Suppress ResourceWarnings that can clutter logs during memory profiling."""
    warnings.filterwarnings("ignore", category=ResourceWarning)


# Auto-initialize
_suppress_resource_warnings()
