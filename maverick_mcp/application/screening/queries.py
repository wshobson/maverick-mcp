"""
Screening application queries.

This module contains application service queries that orchestrate
domain services and infrastructure adapters for screening operations.
"""

from datetime import datetime
from typing import Any

from maverick_mcp.domain.screening.entities import (
    ScreeningResultCollection,
)
from maverick_mcp.domain.screening.services import IStockRepository, ScreeningService
from maverick_mcp.domain.screening.value_objects import (
    ScreeningCriteria,
    ScreeningStrategy,
    SortingOptions,
)


class GetScreeningResultsQuery:
    """
    Application query for retrieving screening results.

    This query orchestrates the domain service and infrastructure
    to provide a complete screening operation.
    """

    def __init__(self, stock_repository: IStockRepository):
        """
        Initialize the query with required dependencies.

        Args:
            stock_repository: Repository for accessing stock data
        """
        self._stock_repository = stock_repository
        self._screening_service = ScreeningService()

    async def execute(
        self,
        strategy: ScreeningStrategy,
        limit: int = 20,
        criteria: ScreeningCriteria | None = None,
        sorting: SortingOptions | None = None,
    ) -> ScreeningResultCollection:
        """
        Execute the screening query.

        Args:
            strategy: The screening strategy to use
            limit: Maximum number of results to return
            criteria: Optional filtering criteria
            sorting: Optional sorting configuration

        Returns:
            ScreeningResultCollection with results and metadata
        """
        # Validate and adjust limit
        validated_limit = self._screening_service.validate_screening_limits(limit)

        # Get raw data from repository based on strategy
        raw_data = await self._get_raw_data_for_strategy(
            strategy, validated_limit, criteria
        )

        # Convert raw data to domain entities
        screening_results = []
        for raw_result in raw_data:
            try:
                result = self._screening_service.create_screening_result_from_raw_data(
                    raw_result, datetime.utcnow()
                )
                screening_results.append(result)
            except Exception as e:
                # Log and skip invalid results
                # In a real application, we'd use proper logging
                print(
                    f"Warning: Skipped invalid result for {raw_result.get('stock', 'unknown')}: {e}"
                )
                continue

        # Apply additional filtering if criteria provided
        if criteria and criteria.has_any_filters():
            screening_results = self._screening_service.apply_screening_criteria(
                screening_results, criteria
            )

        # Apply sorting
        if sorting is None:
            sorting = SortingOptions.for_strategy(strategy)

        screening_results = self._screening_service.sort_screening_results(
            screening_results, sorting
        )

        # Limit results after filtering and sorting
        screening_results = screening_results[:validated_limit]

        # Create and return collection
        return self._screening_service.create_screening_collection(
            screening_results,
            strategy,
            len(raw_data),  # Total candidates before filtering
        )

    async def _get_raw_data_for_strategy(
        self,
        strategy: ScreeningStrategy,
        limit: int,
        criteria: ScreeningCriteria | None,
    ) -> list[dict[str, Any]]:
        """
        Get raw data from repository based on strategy.

        This method handles the strategy-specific repository calls
        and basic filtering that can be done at the data layer.
        """
        if strategy == ScreeningStrategy.MAVERICK_BULLISH:
            min_score = None
            if criteria and criteria.min_combined_score:
                min_score = criteria.min_combined_score

            return self._stock_repository.get_maverick_stocks(
                limit=limit * 2,  # Get more to allow for filtering
                min_score=min_score,
            )

        elif strategy == ScreeningStrategy.MAVERICK_BEARISH:
            min_score = None
            if criteria and criteria.min_bear_score:
                min_score = criteria.min_bear_score

            return self._stock_repository.get_maverick_bear_stocks(
                limit=limit * 2,  # Get more to allow for filtering
                min_score=min_score,
            )

        elif strategy == ScreeningStrategy.TRENDING_STAGE2:
            min_momentum_score = None
            if criteria and criteria.min_momentum_score:
                min_momentum_score = criteria.min_momentum_score

            # Check if we need moving average filtering
            filter_ma = criteria and (
                criteria.require_above_sma50
                or criteria.require_above_sma150
                or criteria.require_above_sma200
                or criteria.require_ma_alignment
            )

            return self._stock_repository.get_trending_stocks(
                limit=limit * 2,  # Get more to allow for filtering
                min_momentum_score=min_momentum_score,
                filter_moving_averages=filter_ma,
            )

        else:
            raise ValueError(f"Unsupported screening strategy: {strategy}")


class GetAllScreeningResultsQuery:
    """
    Application query for retrieving results from all screening strategies.

    This query provides a comprehensive view across all available
    screening strategies.
    """

    def __init__(self, stock_repository: IStockRepository):
        """
        Initialize the query with required dependencies.

        Args:
            stock_repository: Repository for accessing stock data
        """
        self._stock_repository = stock_repository
        self._screening_service = ScreeningService()

    async def execute(
        self, limit_per_strategy: int = 10, criteria: ScreeningCriteria | None = None
    ) -> dict[str, ScreeningResultCollection]:
        """
        Execute screening across all strategies.

        Args:
            limit_per_strategy: Number of results per strategy
            criteria: Optional filtering criteria (applied to all strategies)

        Returns:
            Dictionary mapping strategy names to their result collections
        """
        results = {}

        # Execute each strategy
        for strategy in ScreeningStrategy:
            try:
                query = GetScreeningResultsQuery(self._stock_repository)
                collection = await query.execute(
                    strategy=strategy, limit=limit_per_strategy, criteria=criteria
                )
                results[strategy.value] = collection
            except Exception as e:
                # Log and continue with other strategies
                print(f"Warning: Failed to get results for {strategy.value}: {e}")
                # Create empty collection for failed strategy
                results[strategy.value] = (
                    self._screening_service.create_screening_collection([], strategy, 0)
                )

        return results


class GetScreeningStatisticsQuery:
    """
    Application query for retrieving screening statistics and analytics.

    This query provides business intelligence and analytical insights
    across screening results.
    """

    def __init__(self, stock_repository: IStockRepository):
        """
        Initialize the query with required dependencies.

        Args:
            stock_repository: Repository for accessing stock data
        """
        self._stock_repository = stock_repository
        self._screening_service = ScreeningService()

    async def execute(
        self, strategy: ScreeningStrategy | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """
        Execute the statistics query.

        Args:
            strategy: Optional specific strategy to analyze (None for all)
            limit: Maximum results to analyze per strategy

        Returns:
            Comprehensive statistics and analytics
        """
        if strategy:
            # Single strategy analysis
            query = GetScreeningResultsQuery(self._stock_repository)
            collection = await query.execute(strategy, limit)

            return {
                "strategy": strategy.value,
                "statistics": self._screening_service.calculate_screening_statistics(
                    collection
                ),
                "timestamp": datetime.utcnow().isoformat(),
            }

        else:
            # All strategies analysis
            all_query = GetAllScreeningResultsQuery(self._stock_repository)
            all_collections = await all_query.execute(limit)

            combined_stats = {
                "overall_summary": {
                    "strategies_analyzed": len(all_collections),
                    "total_results": sum(
                        len(c.results) for c in all_collections.values()
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "by_strategy": {},
            }

            # Calculate stats for each strategy
            for strategy_name, collection in all_collections.items():
                combined_stats["by_strategy"][strategy_name] = (
                    self._screening_service.calculate_screening_statistics(collection)
                )

            # Calculate cross-strategy insights
            combined_stats["cross_strategy_analysis"] = (
                self._calculate_cross_strategy_insights(all_collections)
            )

            return combined_stats

    def _calculate_cross_strategy_insights(
        self, collections: dict[str, ScreeningResultCollection]
    ) -> dict[str, Any]:
        """
        Calculate insights that span across multiple strategies.

        This provides valuable business intelligence by comparing
        and contrasting results across different screening approaches.
        """
        all_symbols = set()
        strategy_overlaps = {}

        # Collect all symbols and calculate overlaps
        for strategy_name, collection in collections.items():
            symbols = {r.stock_symbol for r in collection.results}
            all_symbols.update(symbols)
            strategy_overlaps[strategy_name] = symbols

        # Find intersections
        bullish_symbols = strategy_overlaps.get(
            ScreeningStrategy.MAVERICK_BULLISH.value, set()
        )
        bearish_symbols = strategy_overlaps.get(
            ScreeningStrategy.MAVERICK_BEARISH.value, set()
        )
        trending_symbols = strategy_overlaps.get(
            ScreeningStrategy.TRENDING_STAGE2.value, set()
        )

        return {
            "total_unique_symbols": len(all_symbols),
            "strategy_overlaps": {
                "bullish_and_trending": len(bullish_symbols & trending_symbols),
                "conflicting_signals": len(bullish_symbols & bearish_symbols),
                "trending_exclusive": len(
                    trending_symbols - bullish_symbols - bearish_symbols
                ),
            },
            "market_sentiment": {
                "bullish_bias": len(bullish_symbols) > len(bearish_symbols),
                "trend_strength": len(trending_symbols) / max(len(all_symbols), 1),
                "conflict_ratio": len(bullish_symbols & bearish_symbols)
                / max(len(all_symbols), 1),
            },
        }
