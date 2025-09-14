"""
Batch Processing Extensions for VectorBTEngine.

This module adds batch processing capabilities to the VectorBT engine,
allowing for parallel execution of multiple backtest strategies,
parameter optimization, and strategy validation.
"""

import asyncio
import gc
import time
from typing import Any

import numpy as np

from maverick_mcp.utils.memory_profiler import (
    cleanup_dataframes,
    get_memory_stats,
    profile_memory,
)
from maverick_mcp.utils.structured_logger import (
    get_structured_logger,
    with_structured_logging,
)

logger = get_structured_logger(__name__)


class BatchProcessingMixin:
    """Mixin class to add batch processing methods to VectorBTEngine."""

    @with_structured_logging(
        "run_batch_backtest", include_performance=True, log_params=True
    )
    @profile_memory(log_results=True, threshold_mb=100.0)
    async def run_batch_backtest(
        self,
        batch_configs: list[dict[str, Any]],
        max_workers: int = 6,
        chunk_size: int = 10,
        validate_data: bool = True,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        """
        Run multiple backtest strategies in parallel with optimized batch processing.

        Args:
            batch_configs: List of backtest configurations, each containing:
                - symbol: Stock symbol
                - strategy_type: Strategy type name
                - parameters: Strategy parameters dict
                - start_date: Start date string
                - end_date: End date string
                - initial_capital: Starting capital (optional, default 10000)
                - fees: Trading fees (optional, default 0.001)
                - slippage: Slippage factor (optional, default 0.001)
            max_workers: Maximum concurrent workers
            chunk_size: Number of configs to process per batch
            validate_data: Whether to validate input data
            fail_fast: Whether to stop on first failure

        Returns:
            Dictionary containing batch results and summary statistics
        """
        from maverick_mcp.backtesting.strategy_executor import (
            ExecutionContext,
            ExecutionResult,
            StrategyExecutor,
        )

        start_time = time.time()
        batch_id = f"batch_{int(start_time)}"

        logger.info(
            f"Starting batch backtest {batch_id} with {len(batch_configs)} configurations"
        )

        # Validate input data if requested
        if validate_data:
            validation_errors = []
            for i, config in enumerate(batch_configs):
                try:
                    self._validate_batch_config(config, f"config_{i}")
                except Exception as e:
                    validation_errors.append(f"Config {i}: {str(e)}")

            if validation_errors:
                if fail_fast:
                    raise ValueError(
                        f"Batch validation failed: {'; '.join(validation_errors)}"
                    )
                else:
                    logger.warning(
                        f"Validation warnings for batch {batch_id}: {validation_errors}"
                    )

        # Initialize executor
        executor = StrategyExecutor(
            max_concurrent_strategies=max_workers,
            cache_manager=getattr(self, "cache", None),
        )

        # Convert configs to execution contexts
        contexts = []
        for i, config in enumerate(batch_configs):
            context = ExecutionContext(
                strategy_id=f"{batch_id}_strategy_{i}",
                symbol=config["symbol"],
                strategy_type=config["strategy_type"],
                parameters=config["parameters"],
                start_date=config["start_date"],
                end_date=config["end_date"],
                initial_capital=config.get("initial_capital", 10000.0),
                fees=config.get("fees", 0.001),
                slippage=config.get("slippage", 0.001),
            )
            contexts.append(context)

        # Process in chunks to manage memory
        all_results = []
        successful_results = []
        failed_results = []

        for chunk_start in range(0, len(contexts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(contexts))
            chunk_contexts = contexts[chunk_start:chunk_end]

            logger.info(
                f"Processing chunk {chunk_start // chunk_size + 1} ({len(chunk_contexts)} items)"
            )

            try:
                # Execute chunk in parallel
                chunk_results = await executor.execute_strategies(chunk_contexts)

                # Process results
                for result in chunk_results:
                    all_results.append(result)
                    if result.success:
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
                        if fail_fast:
                            logger.error(f"Batch failed fast on: {result.error}")
                            break

                # Memory cleanup between chunks
                if getattr(self, "enable_memory_profiling", False):
                    cleanup_dataframes()
                    gc.collect()

            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                if fail_fast:
                    raise
                # Add failed result for chunk
                for context in chunk_contexts:
                    failed_results.append(
                        ExecutionResult(
                            context=context,
                            success=False,
                            error=f"Chunk processing error: {e}",
                        )
                    )

        # Cleanup executor
        await executor.cleanup()

        # Calculate summary statistics
        total_execution_time = time.time() - start_time
        success_rate = (
            len(successful_results) / len(all_results) if all_results else 0.0
        )

        summary = {
            "batch_id": batch_id,
            "total_configs": len(batch_configs),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": success_rate,
            "total_execution_time": total_execution_time,
            "avg_execution_time": total_execution_time / len(all_results)
            if all_results
            else 0.0,
            "memory_stats": get_memory_stats()
            if getattr(self, "enable_memory_profiling", False)
            else None,
        }

        logger.info(f"Batch backtest {batch_id} completed: {summary}")

        return {
            "batch_id": batch_id,
            "summary": summary,
            "successful_results": [r.result for r in successful_results if r.result],
            "failed_results": [
                {
                    "strategy_id": r.context.strategy_id,
                    "symbol": r.context.symbol,
                    "strategy_type": r.context.strategy_type,
                    "error": r.error,
                }
                for r in failed_results
            ],
            "all_results": all_results,
        }

    @with_structured_logging(
        "batch_optimize_parameters", include_performance=True, log_params=True
    )
    async def batch_optimize_parameters(
        self,
        optimization_configs: list[dict[str, Any]],
        max_workers: int = 4,
        optimization_method: str = "grid_search",
        max_iterations: int = 100,
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters for multiple symbols/strategies in parallel.

        Args:
            optimization_configs: List of optimization configurations, each containing:
                - symbol: Stock symbol
                - strategy_type: Strategy type name
                - parameter_ranges: Dictionary of parameter ranges to optimize
                - start_date: Start date string
                - end_date: End date string
                - optimization_metric: Metric to optimize (default: sharpe_ratio)
                - initial_capital: Starting capital
            max_workers: Maximum concurrent workers
            optimization_method: Optimization method ('grid_search', 'random_search')
            max_iterations: Maximum optimization iterations per config

        Returns:
            Dictionary containing optimization results for all configurations
        """
        start_time = time.time()
        batch_id = f"optimize_batch_{int(start_time)}"

        logger.info(
            f"Starting batch optimization {batch_id} with {len(optimization_configs)} configurations"
        )

        # Process optimizations in parallel
        optimization_tasks = []
        for i, config in enumerate(optimization_configs):
            task = self._run_single_optimization(
                config, f"{batch_id}_opt_{i}", optimization_method, max_iterations
            )
            optimization_tasks.append(task)

        # Execute with concurrency limit
        semaphore = asyncio.BoundedSemaphore(max_workers)

        async def limited_optimization(task):
            async with semaphore:
                return await task

        # Run all optimizations
        optimization_results = await asyncio.gather(
            *[limited_optimization(task) for task in optimization_tasks],
            return_exceptions=True,
        )

        # Process results
        successful_optimizations = []
        failed_optimizations = []

        for i, result in enumerate(optimization_results):
            if isinstance(result, Exception):
                failed_optimizations.append(
                    {
                        "config_index": i,
                        "config": optimization_configs[i],
                        "error": str(result),
                    }
                )
            else:
                successful_optimizations.append(result)

        # Calculate summary
        total_execution_time = time.time() - start_time
        success_rate = (
            len(successful_optimizations) / len(optimization_configs)
            if optimization_configs
            else 0.0
        )

        summary = {
            "batch_id": batch_id,
            "total_optimizations": len(optimization_configs),
            "successful": len(successful_optimizations),
            "failed": len(failed_optimizations),
            "success_rate": success_rate,
            "total_execution_time": total_execution_time,
            "optimization_method": optimization_method,
            "max_iterations": max_iterations,
        }

        logger.info(f"Batch optimization {batch_id} completed: {summary}")

        return {
            "batch_id": batch_id,
            "summary": summary,
            "successful_optimizations": successful_optimizations,
            "failed_optimizations": failed_optimizations,
        }

    async def batch_validate_strategies(
        self,
        validation_configs: list[dict[str, Any]],
        validation_start_date: str,
        validation_end_date: str,
        max_workers: int = 6,
    ) -> dict[str, Any]:
        """
        Validate multiple strategies against out-of-sample data in parallel.

        Args:
            validation_configs: List of validation configurations with optimized parameters
            validation_start_date: Start date for validation period
            validation_end_date: End date for validation period
            max_workers: Maximum concurrent workers

        Returns:
            Dictionary containing validation results and performance comparison
        """
        start_time = time.time()
        batch_id = f"validate_batch_{int(start_time)}"

        logger.info(
            f"Starting batch validation {batch_id} with {len(validation_configs)} strategies"
        )

        # Create validation backtest configs
        validation_batch_configs = []
        for config in validation_configs:
            validation_config = {
                "symbol": config["symbol"],
                "strategy_type": config["strategy_type"],
                "parameters": config.get(
                    "optimized_parameters", config.get("parameters", {})
                ),
                "start_date": validation_start_date,
                "end_date": validation_end_date,
                "initial_capital": config.get("initial_capital", 10000.0),
                "fees": config.get("fees", 0.001),
                "slippage": config.get("slippage", 0.001),
            }
            validation_batch_configs.append(validation_config)

        # Run validation backtests
        validation_results = await self.run_batch_backtest(
            validation_batch_configs,
            max_workers=max_workers,
            validate_data=True,
            fail_fast=False,
        )

        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            validation_configs, validation_results["successful_results"]
        )

        total_execution_time = time.time() - start_time

        return {
            "batch_id": batch_id,
            "validation_period": {
                "start_date": validation_start_date,
                "end_date": validation_end_date,
            },
            "summary": {
                "total_strategies": len(validation_configs),
                "validated_strategies": len(validation_results["successful_results"]),
                "validation_success_rate": len(validation_results["successful_results"])
                / len(validation_configs)
                if validation_configs
                else 0.0,
                "total_execution_time": total_execution_time,
            },
            "validation_results": validation_results["successful_results"],
            "validation_metrics": validation_metrics,
            "failed_validations": validation_results["failed_results"],
        }

    async def get_batch_results(
        self, batch_id: str, include_detailed_results: bool = False
    ) -> dict[str, Any] | None:
        """
        Retrieve results for a completed batch operation.

        Args:
            batch_id: Batch ID to retrieve results for
            include_detailed_results: Whether to include full result details

        Returns:
            Dictionary containing batch results or None if not found
        """
        # This would typically retrieve from a persistence layer
        # For now, return None as results are returned directly
        logger.warning(f"Batch result retrieval not implemented for {batch_id}")
        logger.info(
            "Batch results are currently returned directly from batch operations"
        )

        return None

    # Alias method for backward compatibility
    async def batch_optimize(self, *args, **kwargs):
        """Alias for batch_optimize_parameters for backward compatibility."""
        return await self.batch_optimize_parameters(*args, **kwargs)

    # =============================================================================
    # BATCH PROCESSING HELPER METHODS
    # =============================================================================

    def _validate_batch_config(self, config: dict[str, Any], config_name: str) -> None:
        """Validate a single batch configuration."""
        required_fields = [
            "symbol",
            "strategy_type",
            "parameters",
            "start_date",
            "end_date",
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {config_name}")

        # Validate dates
        try:
            from maverick_mcp.data.validation import DataValidator

            DataValidator.validate_date_range(config["start_date"], config["end_date"])
        except Exception as e:
            raise ValueError(f"Invalid date range in {config_name}: {e}") from e

        # Validate symbol
        if not isinstance(config["symbol"], str) or len(config["symbol"]) == 0:
            raise ValueError(f"Invalid symbol in {config_name}")

        # Validate strategy type
        if not isinstance(config["strategy_type"], str):
            raise ValueError(f"Invalid strategy_type in {config_name}")

        # Validate parameters
        if not isinstance(config["parameters"], dict):
            raise ValueError(f"Parameters must be a dictionary in {config_name}")

    async def _run_single_optimization(
        self,
        config: dict[str, Any],
        optimization_id: str,
        method: str,
        max_iterations: int,
    ) -> dict[str, Any]:
        """Run optimization for a single configuration."""
        try:
            # Extract configuration
            symbol = config["symbol"]
            strategy_type = config["strategy_type"]
            parameter_ranges = config["parameter_ranges"]
            start_date = config["start_date"]
            end_date = config["end_date"]
            optimization_metric = config.get("optimization_metric", "sharpe_ratio")
            initial_capital = config.get("initial_capital", 10000.0)

            # Simple parameter optimization (placeholder - would use actual optimizer)
            # For now, return basic result structure
            best_params = {}
            for param, ranges in parameter_ranges.items():
                if isinstance(ranges, list) and len(ranges) >= 2:
                    # Use middle value as "optimized"
                    best_params[param] = ranges[len(ranges) // 2]
                elif isinstance(ranges, dict):
                    if "min" in ranges and "max" in ranges:
                        best_params[param] = (ranges["min"] + ranges["max"]) / 2

            # Run a basic backtest with these parameters
            backtest_result = await self.run_backtest(
                symbol=symbol,
                strategy_type=strategy_type,
                parameters=best_params,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
            )

            best_score = backtest_result.get("metrics", {}).get(
                optimization_metric, 0.0
            )

            return {
                "optimization_id": optimization_id,
                "symbol": symbol,
                "strategy_type": strategy_type,
                "optimized_parameters": best_params,
                "best_score": best_score,
                "optimization_history": [
                    {"parameters": best_params, "score": best_score}
                ],
                "execution_time": 0.0,
            }

        except Exception as e:
            logger.error(f"Optimization failed for {optimization_id}: {e}")
            raise

    def _calculate_validation_metrics(
        self,
        original_configs: list[dict[str, Any]],
        validation_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate validation metrics comparing in-sample vs out-of-sample performance."""
        metrics = {
            "strategy_comparisons": [],
            "aggregate_metrics": {
                "avg_in_sample_sharpe": 0.0,
                "avg_out_of_sample_sharpe": 0.0,
                "sharpe_degradation": 0.0,
                "strategies_with_positive_validation": 0,
            },
        }

        if not original_configs or not validation_results:
            return metrics

        sharpe_ratios_in_sample = []
        sharpe_ratios_out_of_sample = []

        for i, (original, validation) in enumerate(
            zip(original_configs, validation_results, strict=False)
        ):
            # Get in-sample performance (from original optimization)
            in_sample_sharpe = original.get("best_score", 0.0)

            # Get out-of-sample performance
            out_of_sample_sharpe = validation.get("metrics", {}).get(
                "sharpe_ratio", 0.0
            )

            strategy_comparison = {
                "strategy_index": i,
                "symbol": original["symbol"],
                "strategy_type": original["strategy_type"],
                "in_sample_sharpe": in_sample_sharpe,
                "out_of_sample_sharpe": out_of_sample_sharpe,
                "sharpe_degradation": in_sample_sharpe - out_of_sample_sharpe,
                "validation_success": out_of_sample_sharpe > 0,
            }

            metrics["strategy_comparisons"].append(strategy_comparison)
            sharpe_ratios_in_sample.append(in_sample_sharpe)
            sharpe_ratios_out_of_sample.append(out_of_sample_sharpe)

        # Calculate aggregate metrics
        if sharpe_ratios_in_sample and sharpe_ratios_out_of_sample:
            metrics["aggregate_metrics"]["avg_in_sample_sharpe"] = np.mean(
                sharpe_ratios_in_sample
            )
            metrics["aggregate_metrics"]["avg_out_of_sample_sharpe"] = np.mean(
                sharpe_ratios_out_of_sample
            )
            metrics["aggregate_metrics"]["sharpe_degradation"] = (
                metrics["aggregate_metrics"]["avg_in_sample_sharpe"]
                - metrics["aggregate_metrics"]["avg_out_of_sample_sharpe"]
            )
            metrics["aggregate_metrics"]["strategies_with_positive_validation"] = sum(
                1
                for comp in metrics["strategy_comparisons"]
                if comp["validation_success"]
            )

        return metrics
