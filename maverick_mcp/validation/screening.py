"""
Validation models for stock screening tools.

This module provides Pydantic models for validating inputs
to all screening-related tools.
"""

from typing import Literal

from pydantic import Field, field_validator

from .base import (
    PaginationMixin,
    PositiveFloat,
    PositiveInt,
    StrictBaseModel,
)


class MaverickScreeningRequest(StrictBaseModel, PaginationMixin):
    """Validation for get_maverick_stocks tool."""

    limit: PositiveInt = Field(
        default=20, le=100, description="Maximum number of stocks to return"
    )

    model_config = {"json_schema_extra": {"examples": [{"limit": 20}, {"limit": 50}]}}


class SupplyDemandBreakoutRequest(StrictBaseModel, PaginationMixin):
    """Validation for get_supply_demand_breakouts tool."""

    limit: PositiveInt = Field(
        default=20, le=100, description="Maximum number of stocks to return"
    )
    filter_moving_averages: bool = Field(
        default=False,
        description="If True, only return stocks in demand expansion phase (above all moving averages)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"limit": 20, "filter_moving_averages": False},
                {"limit": 15, "filter_moving_averages": True},
            ]
        }
    }


class CustomScreeningRequest(StrictBaseModel, PaginationMixin):
    """Validation for get_screening_by_criteria tool."""

    min_momentum_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Minimum momentum score (0-100)",
    )
    min_volume: PositiveInt | None = Field(
        default=None, description="Minimum average daily volume"
    )
    max_price: PositiveFloat | None = Field(
        default=None, description="Maximum stock price"
    )
    sector: str | None = Field(
        default=None,
        max_length=100,
        description="Specific sector to filter (e.g., 'Technology')",
    )
    limit: PositiveInt = Field(
        default=20, le=100, description="Maximum number of results"
    )

    @field_validator("sector")
    @classmethod
    def normalize_sector(cls, v: str | None) -> str | None:
        """Normalize sector name."""
        if v is not None:
            # Title case for consistency
            return v.strip().title()
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"min_momentum_score": 85.0, "min_volume": 1000000, "limit": 20},
                {
                    "max_price": 50.0,
                    "sector": "Technology",
                    "min_momentum_score": 80.0,
                    "limit": 30,
                },
            ]
        }
    }


class ScreeningType(StrictBaseModel):
    """Enum for screening types."""

    screening_type: Literal[
        "maverick_bullish", "maverick_bearish", "supply_demand_breakout", "all"
    ] = Field(default="all", description="Type of screening to retrieve")
