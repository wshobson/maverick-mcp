"""
Screening domain value objects.

This module contains immutable value objects that represent
core concepts in the screening domain.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class ScreeningStrategy(Enum):
    """
    Enumeration of available screening strategies.

    Each strategy represents a different approach to identifying
    investment opportunities in the stock market.
    """

    MAVERICK_BULLISH = "maverick_bullish"
    MAVERICK_BEARISH = "maverick_bearish"
    TRENDING_STAGE2 = "trending_stage2"

    def get_description(self) -> str:
        """Get human-readable description of the strategy."""
        descriptions = {
            self.MAVERICK_BULLISH: "High momentum stocks with bullish technical setups",
            self.MAVERICK_BEARISH: "Weak stocks with bearish technical setups",
            self.TRENDING_STAGE2: "Uptrend stocks meeting trending criteria",
        }
        return descriptions[self]

    def get_primary_sort_field(self) -> str:
        """Get the primary field used for sorting results."""
        sort_fields = {
            self.MAVERICK_BULLISH: "combined_score",
            self.MAVERICK_BEARISH: "bear_score",
            self.TRENDING_STAGE2: "momentum_score",
        }
        return sort_fields[self]

    def get_minimum_score_threshold(self) -> int:
        """Get the minimum score threshold for meaningful results."""
        thresholds = {
            self.MAVERICK_BULLISH: 50,
            self.MAVERICK_BEARISH: 30,
            self.TRENDING_STAGE2: 70,
        }
        return thresholds[self]


@dataclass(frozen=True)
class ScreeningCriteria:
    """
    Immutable value object representing screening filter criteria.

    This encapsulates all the parameters that can be used to filter
    and refine screening results.
    """

    # Basic filters
    min_momentum_score: Decimal | None = None
    max_momentum_score: Decimal | None = None
    min_volume: int | None = None
    max_volume: int | None = None
    min_price: Decimal | None = None
    max_price: Decimal | None = None

    # Technical filters
    min_combined_score: int | None = None
    min_bear_score: int | None = None
    min_adr_percentage: Decimal | None = None
    max_adr_percentage: Decimal | None = None

    # Pattern filters
    require_pattern_detected: bool = False
    require_squeeze: bool = False
    require_consolidation: bool = False
    require_entry_signal: bool = False

    # Moving average filters
    require_above_sma50: bool = False
    require_above_sma150: bool = False
    require_above_sma200: bool = False
    require_ma_alignment: bool = False  # 50 > 150 > 200

    # Sector/Industry filters
    allowed_sectors: list[str] | None = None
    excluded_sectors: list[str] | None = None

    def __post_init__(self):
        """Validate criteria constraints."""
        self._validate_rating_ranges()
        self._validate_volume_ranges()
        self._validate_price_ranges()
        self._validate_score_ranges()

    def _validate_rating_ranges(self) -> None:
        """Validate momentum score range constraints."""
        if self.min_momentum_score is not None:
            if not (0 <= self.min_momentum_score <= 100):
                raise ValueError("Minimum momentum score must be between 0 and 100")

        if self.max_momentum_score is not None:
            if not (0 <= self.max_momentum_score <= 100):
                raise ValueError("Maximum momentum score must be between 0 and 100")

        if (
            self.min_momentum_score is not None
            and self.max_momentum_score is not None
            and self.min_momentum_score > self.max_momentum_score
        ):
            raise ValueError(
                "Minimum momentum score cannot exceed maximum momentum score"
            )

    def _validate_volume_ranges(self) -> None:
        """Validate volume range constraints."""
        if self.min_volume is not None and self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")

        if self.max_volume is not None and self.max_volume < 0:
            raise ValueError("Maximum volume cannot be negative")

        if (
            self.min_volume is not None
            and self.max_volume is not None
            and self.min_volume > self.max_volume
        ):
            raise ValueError("Minimum volume cannot exceed maximum volume")

    def _validate_price_ranges(self) -> None:
        """Validate price range constraints."""
        if self.min_price is not None and self.min_price <= 0:
            raise ValueError("Minimum price must be positive")

        if self.max_price is not None and self.max_price <= 0:
            raise ValueError("Maximum price must be positive")

        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("Minimum price cannot exceed maximum price")

    def _validate_score_ranges(self) -> None:
        """Validate score range constraints."""
        if self.min_combined_score is not None and self.min_combined_score < 0:
            raise ValueError("Minimum combined score cannot be negative")

        if self.min_bear_score is not None and self.min_bear_score < 0:
            raise ValueError("Minimum bear score cannot be negative")

    def has_any_filters(self) -> bool:
        """Check if any filters are applied."""
        return any(
            [
                self.min_momentum_score is not None,
                self.max_momentum_score is not None,
                self.min_volume is not None,
                self.max_volume is not None,
                self.min_price is not None,
                self.max_price is not None,
                self.min_combined_score is not None,
                self.min_bear_score is not None,
                self.min_adr_percentage is not None,
                self.max_adr_percentage is not None,
                self.require_pattern_detected,
                self.require_squeeze,
                self.require_consolidation,
                self.require_entry_signal,
                self.require_above_sma50,
                self.require_above_sma150,
                self.require_above_sma200,
                self.require_ma_alignment,
                self.allowed_sectors is not None,
                self.excluded_sectors is not None,
            ]
        )

    def get_filter_description(self) -> str:
        """Get human-readable description of active filters."""
        filters = []

        if self.min_momentum_score is not None:
            filters.append(f"Momentum Score >= {self.min_momentum_score}")

        if self.max_momentum_score is not None:
            filters.append(f"Momentum Score <= {self.max_momentum_score}")

        if self.min_volume is not None:
            filters.append(f"Volume >= {self.min_volume:,}")

        if self.min_price is not None:
            filters.append(f"Price >= ${self.min_price}")

        if self.max_price is not None:
            filters.append(f"Price <= ${self.max_price}")

        if self.require_above_sma50:
            filters.append("Above SMA 50")

        if self.require_pattern_detected:
            filters.append("Pattern Detected")

        if not filters:
            return "No filters applied"

        return "; ".join(filters)


@dataclass(frozen=True)
class ScreeningLimits:
    """
    Value object representing limits and constraints for screening operations.

    This encapsulates business rules around result limits, timeouts,
    and resource constraints.
    """

    max_results: int = 100
    default_results: int = 20
    min_results: int = 1
    max_timeout_seconds: int = 30

    def __post_init__(self):
        """Validate limit constraints."""
        if self.min_results <= 0:
            raise ValueError("Minimum results must be positive")

        if self.default_results < self.min_results:
            raise ValueError("Default results cannot be less than minimum")

        if self.max_results < self.default_results:
            raise ValueError("Maximum results cannot be less than default")

        if self.max_timeout_seconds <= 0:
            raise ValueError("Maximum timeout must be positive")

    def validate_limit(self, requested_limit: int) -> int:
        """
        Validate and adjust requested result limit.

        Returns the adjusted limit within valid bounds.
        """
        if requested_limit < self.min_results:
            return self.min_results

        if requested_limit > self.max_results:
            return self.max_results

        return requested_limit


@dataclass(frozen=True)
class SortingOptions:
    """
    Value object representing sorting options for screening results.

    This encapsulates the various ways results can be ordered.
    """

    field: str
    descending: bool = True
    secondary_field: str | None = None
    secondary_descending: bool = True

    # Valid sortable fields
    VALID_FIELDS = {
        "combined_score",
        "bear_score",
        "momentum_score",
        "close_price",
        "volume",
        "avg_volume_30d",
        "adr_percentage",
        "quality_score",
    }

    def __post_init__(self):
        """Validate sorting configuration."""
        if self.field not in self.VALID_FIELDS:
            raise ValueError(
                f"Invalid sort field: {self.field}. Must be one of {self.VALID_FIELDS}"
            )

        if (
            self.secondary_field is not None
            and self.secondary_field not in self.VALID_FIELDS
        ):
            raise ValueError(f"Invalid secondary sort field: {self.secondary_field}")

    @classmethod
    def for_strategy(cls, strategy: ScreeningStrategy) -> "SortingOptions":
        """Create default sorting options for a screening strategy."""
        primary_field = strategy.get_primary_sort_field()

        # Add appropriate secondary sort field
        secondary_field = (
            "momentum_score" if primary_field != "momentum_score" else "close_price"
        )

        return cls(
            field=primary_field,
            descending=True,
            secondary_field=secondary_field,
            secondary_descending=True,
        )
