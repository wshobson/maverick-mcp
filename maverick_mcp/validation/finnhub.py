"""
Validation models for Finnhub alternative data tools.

This module provides Pydantic models for validating inputs
to all Finnhub API tools (company news, earnings, analyst
recommendations, institutional ownership, etc.).
"""

from typing import Literal

from pydantic import Field, field_validator

from .base import (
    StrictBaseModel,
    TickerSymbol,
    TickerValidator,
)


class CompanyNewsRequest(StrictBaseModel):
    """Validation for finnhub_company_news tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    from_date: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format (defaults to 7 days ago)",
    )
    to_date: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format (defaults to today)",
    )
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum number of articles to return"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ticker": "AAPL"},
                {
                    "ticker": "TSLA",
                    "from_date": "2026-03-01",
                    "to_date": "2026-03-13",
                    "limit": 10,
                },
            ]
        }
    }


class EarningsCalendarRequest(StrictBaseModel):
    """Validation for finnhub_earnings_calendar tool."""

    from_date: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format (defaults to today)",
    )
    to_date: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format (defaults to 14 days from now)",
    )
    ticker: str | None = Field(
        default=None,
        description="Optional ticker symbol to filter results",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str | None) -> str | None:
        """Normalize ticker to uppercase if provided."""
        if v is None:
            return v
        return TickerValidator.validate_ticker(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {},
                {"ticker": "AAPL"},
                {"from_date": "2026-03-13", "to_date": "2026-03-27"},
            ]
        }
    }


class EarningsSurprisesRequest(StrictBaseModel):
    """Validation for finnhub_earnings_surprises tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    limit: int = Field(
        default=4, ge=1, le=20, description="Number of quarters to return"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class RecommendationsRequest(StrictBaseModel):
    """Validation for finnhub_analyst_recommendations tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class OwnershipRequest(StrictBaseModel):
    """Validation for finnhub_institutional_ownership tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    limit: int = Field(
        default=20, ge=1, le=50, description="Maximum number of holders to return"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class PeersRequest(StrictBaseModel):
    """Validation for finnhub_company_peers tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class EconomicCalendarRequest(StrictBaseModel):
    """Validation for finnhub_economic_calendar tool."""

    from_date: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format (defaults to today)",
    )
    to_date: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format (defaults to 7 days from now)",
    )


class MarketNewsRequest(StrictBaseModel):
    """Validation for finnhub_market_news tool."""

    category: Literal["general", "forex", "crypto", "merger"] = Field(
        default="general", description="News category"
    )
    min_id: int = Field(
        default=0, ge=0, description="Minimum article ID for pagination"
    )
