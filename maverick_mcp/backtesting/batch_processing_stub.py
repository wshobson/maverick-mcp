"""
Lightweight batch processing stub for import compatibility.

This module provides basic batch processing method stubs that can be imported
even when heavy dependencies like VectorBT, NumPy, etc. are not available.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BatchProcessingStub:
    """Lightweight batch processing stub class."""

    async def run_batch_backtest(
        self,
        batch_configs: list[dict[str, Any]],
        max_workers: int = 6,
        chunk_size: int = 10,
        validate_data: bool = True,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """Stub for run_batch_backtest method."""
        raise ImportError("Batch processing requires VectorBT and other dependencies")

    async def batch_optimize_parameters(
        self,
        optimization_configs: list[dict[str, Any]],
        max_workers: int = 4,
        optimization_method: str = "grid_search",
        max_iterations: int = 100,
    ) -> dict[str, Any]:
        """Stub for batch_optimize_parameters method."""
        raise ImportError("Batch processing requires VectorBT and other dependencies")

    async def batch_validate_strategies(
        self,
        validation_configs: list[dict[str, Any]],
        validation_start_date: str,
        validation_end_date: str,
        max_workers: int = 6,
    ) -> dict[str, Any]:
        """Stub for batch_validate_strategies method."""
        raise ImportError("Batch processing requires VectorBT and other dependencies")

    async def get_batch_results(
        self, batch_id: str, include_detailed_results: bool = False
    ) -> dict[str, Any] | None:
        """Stub for get_batch_results method."""
        raise ImportError("Batch processing requires VectorBT and other dependencies")

    # Alias method for backward compatibility
    async def batch_optimize(self, *args, **kwargs):
        """Alias for batch_optimize_parameters for backward compatibility."""
        return await self.batch_optimize_parameters(*args, **kwargs)


class VectorBTEngineStub(BatchProcessingStub):
    """Stub VectorBT engine that provides batch processing methods."""

    def __init__(self, *args, **kwargs):
        """Initialize stub engine."""
        logger.warning(
            "VectorBT dependencies not available - using stub implementation"
        )

    def __getattr__(self, name):
        """Provide stubs for any missing methods."""
        if name.startswith("batch") or name in ["run_backtest", "optimize_strategy"]:

            async def stub_method(*args, **kwargs):
                raise ImportError(
                    f"Method {name} requires VectorBT and other dependencies"
                )

            return stub_method
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
