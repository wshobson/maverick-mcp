"""
Base validation models and common validators for Maverick-MCP.

This module provides base classes and common validation functions
used across all validation models.
"""

import re
from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from maverick_mcp.config.settings import get_settings

settings = get_settings()

# Custom type annotations
TickerSymbol = Annotated[
    str,
    Field(
        min_length=settings.validation.min_symbol_length,
        max_length=settings.validation.max_symbol_length,
        pattern=r"^[A-Z0-9\-\.]{1,10}$",
        description="Stock ticker symbol (e.g., AAPL, BRK.B, SPY)",
    ),
]

DateString = Annotated[
    str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in YYYY-MM-DD format")
]

PositiveInt = Annotated[int, Field(gt=0, description="Positive integer value")]

PositiveFloat = Annotated[float, Field(gt=0.0, description="Positive float value")]

Percentage = Annotated[
    float, Field(ge=0.0, le=100.0, description="Percentage value (0-100)")
]


class StrictBaseModel(BaseModel):
    """
    Base model with strict validation settings.

    - Forbids extra fields
    - Validates on assignment
    - Uses strict mode for type coercion
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=True,
        str_strip_whitespace=True,
        json_schema_extra={"examples": []},
    )


class TickerValidator:
    """Common ticker validation methods."""

    @staticmethod
    def validate_ticker(value: str) -> str:
        """Validate and normalize ticker symbol."""
        # Convert to uppercase
        ticker = value.upper().strip()

        # Check pattern
        pattern = f"^[A-Z0-9\\-\\.]{{1,{settings.validation.max_symbol_length}}}$"
        if not re.match(pattern, ticker):
            raise ValueError(
                f"Invalid ticker symbol: {value}. "
                f"Must be {settings.validation.min_symbol_length}-{settings.validation.max_symbol_length} characters, alphanumeric with optional . or -"
            )

        return ticker

    @staticmethod
    def validate_ticker_list(values: list[str]) -> list[str]:
        """Validate and normalize a list of tickers."""
        if not values:
            raise ValueError("At least one ticker symbol is required")

        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []

        for ticker in values:
            normalized = TickerValidator.validate_ticker(ticker)
            if normalized not in seen:
                seen.add(normalized)
                unique_tickers.append(normalized)

        return unique_tickers


class DateValidator:
    """Common date validation methods."""

    @staticmethod
    def validate_date_string(value: str) -> str:
        """Validate date string format."""
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {value}. Must be YYYY-MM-DD")
        return value

    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
        """Validate that end_date is after start_date."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if end < start:
            raise ValueError(
                f"End date ({end_date}) must be after start date ({start_date})"
            )

        # Check dates aren't too far in the future
        today = datetime.now(UTC).date()
        if end.date() > today:
            raise ValueError(f"End date ({end_date}) cannot be in the future")

        return start_date, end_date


class PaginationMixin(BaseModel):
    """Mixin for pagination parameters."""

    limit: PositiveInt = Field(
        default=20, le=100, description="Maximum number of results to return"
    )
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


class DateRangeMixin(BaseModel):
    """Mixin for date range parameters."""

    start_date: DateString | None = Field(
        default=None, description="Start date in YYYY-MM-DD format"
    )
    end_date: DateString | None = Field(
        default=None, description="End date in YYYY-MM-DD format"
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: str | None, info) -> str | None:
        """Ensure end_date is after start_date if both are provided."""
        if v is None:
            return v

        start = info.data.get("start_date")
        if start is not None:
            DateValidator.validate_date_range(start, v)

        return v


class BaseRequest(BaseModel):
    """Base class for all API request models."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


class BaseResponse(BaseModel):
    """Base class for all API response models."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
    )
