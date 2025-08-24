"""
Screening domain services.

This module contains pure business logic services that operate on
screening entities and value objects without any external dependencies.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from .entities import ScreeningResult, ScreeningResultCollection
from .value_objects import (
    ScreeningCriteria,
    ScreeningLimits,
    ScreeningStrategy,
    SortingOptions,
)


class IStockRepository(Protocol):
    """Protocol defining the interface for stock data access."""

    def get_maverick_stocks(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """Get Maverick bullish stocks."""
        ...

    def get_maverick_bear_stocks(
        self, limit: int = 20, min_score: int | None = None
    ) -> list[dict[str, Any]]:
        """Get Maverick bearish stocks."""
        ...

    def get_trending_stocks(
        self,
        limit: int = 20,
        min_momentum_score: Decimal | None = None,
        filter_moving_averages: bool = False,
    ) -> list[dict[str, Any]]:
        """Get trending stocks."""
        ...


class ScreeningService:
    """
    Pure domain service for stock screening business logic.

    This service contains no external dependencies and focuses solely
    on the business rules and logic for screening operations.
    """

    def __init__(self):
        """Initialize the screening service."""
        self._default_limits = ScreeningLimits()

    def create_screening_result_from_raw_data(
        self, raw_data: dict[str, Any], screening_date: datetime | None = None
    ) -> ScreeningResult:
        """
        Create a ScreeningResult entity from raw database data.

        This method handles the transformation of raw data into
        a properly validated domain entity.
        """
        if screening_date is None:
            screening_date = datetime.utcnow()

        return ScreeningResult(
            stock_symbol=raw_data.get("stock", ""),
            screening_date=screening_date,
            open_price=Decimal(str(raw_data.get("open", 0))),
            high_price=Decimal(str(raw_data.get("high", 0))),
            low_price=Decimal(str(raw_data.get("low", 0))),
            close_price=Decimal(str(raw_data.get("close", 0))),
            volume=int(raw_data.get("volume", 0)),
            ema_21=Decimal(str(raw_data.get("ema_21", 0))),
            sma_50=Decimal(str(raw_data.get("sma_50", 0))),
            sma_150=Decimal(str(raw_data.get("sma_150", 0))),
            sma_200=Decimal(str(raw_data.get("sma_200", 0))),
            momentum_score=Decimal(str(raw_data.get("momentum_score", 0))),
            avg_volume_30d=Decimal(
                str(raw_data.get("avg_vol_30d", raw_data.get("avg_volume_30d", 0)))
            ),
            adr_percentage=Decimal(str(raw_data.get("adr_pct", 0))),
            atr=Decimal(str(raw_data.get("atr", 0))),
            pattern=raw_data.get("pat"),
            squeeze=raw_data.get("sqz"),
            vcp=raw_data.get("vcp"),
            entry_signal=raw_data.get("entry"),
            combined_score=int(raw_data.get("combined_score", 0)),
            bear_score=int(raw_data.get("score", 0)),  # Bear score uses 'score' field
            compression_score=int(raw_data.get("compression_score", 0)),
            pattern_detected=int(raw_data.get("pattern_detected", 0)),
            # Bearish-specific fields
            rsi_14=Decimal(str(raw_data["rsi_14"]))
            if raw_data.get("rsi_14") is not None
            else None,
            macd=Decimal(str(raw_data["macd"]))
            if raw_data.get("macd") is not None
            else None,
            macd_signal=Decimal(str(raw_data["macd_s"]))
            if raw_data.get("macd_s") is not None
            else None,
            macd_histogram=Decimal(str(raw_data["macd_h"]))
            if raw_data.get("macd_h") is not None
            else None,
            distribution_days_20=raw_data.get("dist_days_20"),
            atr_contraction=raw_data.get("atr_contraction"),
            big_down_volume=raw_data.get("big_down_vol"),
        )

    def apply_screening_criteria(
        self, results: list[ScreeningResult], criteria: ScreeningCriteria
    ) -> list[ScreeningResult]:
        """
        Apply screening criteria to filter results.

        This method implements all the business rules for filtering
        screening results based on the provided criteria.
        """
        if not criteria.has_any_filters():
            return results

        filtered_results = results

        # Momentum Score filters
        if criteria.min_momentum_score is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r.momentum_score >= criteria.min_momentum_score
            ]

        if criteria.max_momentum_score is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r.momentum_score <= criteria.max_momentum_score
            ]

        # Volume filters
        if criteria.min_volume is not None:
            filtered_results = [
                r for r in filtered_results if r.avg_volume_30d >= criteria.min_volume
            ]

        if criteria.max_volume is not None:
            filtered_results = [
                r for r in filtered_results if r.avg_volume_30d <= criteria.max_volume
            ]

        # Price filters
        if criteria.min_price is not None:
            filtered_results = [
                r for r in filtered_results if r.close_price >= criteria.min_price
            ]

        if criteria.max_price is not None:
            filtered_results = [
                r for r in filtered_results if r.close_price <= criteria.max_price
            ]

        # Score filters
        if criteria.min_combined_score is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r.combined_score >= criteria.min_combined_score
            ]

        if criteria.min_bear_score is not None:
            filtered_results = [
                r for r in filtered_results if r.bear_score >= criteria.min_bear_score
            ]

        # ADR filters
        if criteria.min_adr_percentage is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r.adr_percentage >= criteria.min_adr_percentage
            ]

        if criteria.max_adr_percentage is not None:
            filtered_results = [
                r
                for r in filtered_results
                if r.adr_percentage <= criteria.max_adr_percentage
            ]

        # Pattern filters
        if criteria.require_pattern_detected:
            filtered_results = [r for r in filtered_results if r.pattern_detected > 0]

        if criteria.require_squeeze:
            filtered_results = [
                r
                for r in filtered_results
                if r.squeeze is not None and r.squeeze.strip()
            ]

        if criteria.require_vcp:
            filtered_results = [
                r for r in filtered_results if r.vcp is not None and r.vcp.strip()
            ]

        if criteria.require_entry_signal:
            filtered_results = [
                r
                for r in filtered_results
                if r.entry_signal is not None and r.entry_signal.strip()
            ]

        # Moving average filters
        if criteria.require_above_sma50:
            filtered_results = [r for r in filtered_results if r.close_price > r.sma_50]

        if criteria.require_above_sma150:
            filtered_results = [
                r for r in filtered_results if r.close_price > r.sma_150
            ]

        if criteria.require_above_sma200:
            filtered_results = [
                r for r in filtered_results if r.close_price > r.sma_200
            ]

        if criteria.require_ma_alignment:
            filtered_results = [
                r
                for r in filtered_results
                if (r.sma_50 > r.sma_150 and r.sma_150 > r.sma_200)
            ]

        return filtered_results

    def sort_screening_results(
        self, results: list[ScreeningResult], sorting: SortingOptions
    ) -> list[ScreeningResult]:
        """
        Sort screening results according to the specified options.

        This method implements the business rules for ranking and
        ordering screening results.
        """

        def get_sort_value(result: ScreeningResult, field: str) -> Any:
            """Get the value for sorting from a result."""
            if field == "combined_score":
                return result.combined_score
            elif field == "bear_score":
                return result.bear_score
            elif field == "momentum_score":
                return result.momentum_score
            elif field == "close_price":
                return result.close_price
            elif field == "volume":
                return result.volume
            elif field == "avg_volume_30d":
                return result.avg_volume_30d
            elif field == "adr_percentage":
                return result.adr_percentage
            elif field == "quality_score":
                return result.get_quality_score()
            else:
                return 0

        # Sort by primary field
        sorted_results = sorted(
            results,
            key=lambda r: get_sort_value(r, sorting.field),
            reverse=sorting.descending,
        )

        # Apply secondary sort if specified
        if sorting.secondary_field:
            sorted_results = sorted(
                sorted_results,
                key=lambda r: (
                    get_sort_value(r, sorting.field),
                    get_sort_value(r, sorting.secondary_field),
                ),
                reverse=sorting.descending,
            )

        return sorted_results

    def create_screening_collection(
        self,
        results: list[ScreeningResult],
        strategy: ScreeningStrategy,
        total_candidates: int,
    ) -> ScreeningResultCollection:
        """
        Create a ScreeningResultCollection from individual results.

        This method assembles the aggregate root with proper validation.
        """
        return ScreeningResultCollection(
            results=results,
            strategy_used=strategy.value,
            screening_timestamp=datetime.utcnow(),
            total_candidates_analyzed=total_candidates,
        )

    def validate_screening_limits(self, requested_limit: int) -> int:
        """
        Validate and adjust the requested result limit.

        Business rule: Limits must be within acceptable bounds.
        """
        return self._default_limits.validate_limit(requested_limit)

    def calculate_screening_statistics(
        self, collection: ScreeningResultCollection
    ) -> dict[str, Any]:
        """
        Calculate comprehensive statistics for a screening collection.

        This method provides business intelligence metrics for
        screening result analysis.
        """
        base_stats = collection.get_statistics()

        # Add additional business metrics
        results = collection.results
        if not results:
            return base_stats

        # Quality distribution
        quality_scores = [r.get_quality_score() for r in results]
        base_stats.update(
            {
                "quality_distribution": {
                    "high_quality": sum(1 for q in quality_scores if q >= 80),
                    "medium_quality": sum(1 for q in quality_scores if 50 <= q < 80),
                    "low_quality": sum(1 for q in quality_scores if q < 50),
                },
                "avg_quality_score": sum(quality_scores) / len(quality_scores),
            }
        )

        # Risk/reward analysis
        risk_rewards = [r.calculate_risk_reward_ratio() for r in results]
        valid_ratios = [rr for rr in risk_rewards if rr > 0]

        if valid_ratios:
            base_stats.update(
                {
                    "risk_reward_analysis": {
                        "avg_ratio": float(sum(valid_ratios) / len(valid_ratios)),
                        "favorable_setups": sum(1 for rr in valid_ratios if rr >= 2),
                        "conservative_setups": sum(
                            1 for rr in valid_ratios if 1 <= rr < 2
                        ),
                        "risky_setups": sum(1 for rr in valid_ratios if rr < 1),
                    }
                }
            )

        # Strategy-specific metrics
        if collection.strategy_used == ScreeningStrategy.MAVERICK_BULLISH.value:
            base_stats["momentum_analysis"] = self._calculate_momentum_metrics(results)
        elif collection.strategy_used == ScreeningStrategy.MAVERICK_BEARISH.value:
            base_stats["weakness_analysis"] = self._calculate_weakness_metrics(results)
        elif collection.strategy_used == ScreeningStrategy.TRENDING_STAGE2.value:
            base_stats["trend_analysis"] = self._calculate_trend_metrics(results)

        return base_stats

    def _calculate_momentum_metrics(
        self, results: list[ScreeningResult]
    ) -> dict[str, Any]:
        """Calculate momentum-specific metrics for bullish screens."""
        return {
            "high_momentum": sum(1 for r in results if r.combined_score >= 80),
            "pattern_breakouts": sum(1 for r in results if r.pattern_detected > 0),
            "strong_momentum": sum(1 for r in results if r.momentum_score >= 90),
        }

    def _calculate_weakness_metrics(
        self, results: list[ScreeningResult]
    ) -> dict[str, Any]:
        """Calculate weakness-specific metrics for bearish screens."""
        return {
            "severe_weakness": sum(1 for r in results if r.bear_score >= 80),
            "distribution_signals": sum(
                1
                for r in results
                if r.distribution_days_20 is not None and r.distribution_days_20 >= 5
            ),
            "breakdown_candidates": sum(
                1 for r in results if r.close_price < r.sma_200
            ),
        }

    def _calculate_trend_metrics(
        self, results: list[ScreeningResult]
    ) -> dict[str, Any]:
        """Calculate trend-specific metrics for trending screens."""
        return {
            "strong_trends": sum(1 for r in results if r.is_trending_stage2()),
            "perfect_alignment": sum(
                1 for r in results if (r.sma_50 > r.sma_150 and r.sma_150 > r.sma_200)
            ),
            "elite_momentum": sum(1 for r in results if r.momentum_score >= 95),
        }
