"""
Validation models for data-related tools.

This module provides Pydantic models for validating inputs
to all data fetching and caching tools.
"""

from pydantic import Field, field_validator, model_validator

from .base import (
    DateRangeMixin,
    DateString,
    DateValidator,
    StrictBaseModel,
    TickerSymbol,
    TickerValidator,
)


class FetchStockDataRequest(StrictBaseModel, DateRangeMixin):
    """Validation for fetch_stock_data tool."""

    ticker: TickerSymbol = Field(
        ...,
        description="Stock ticker symbol (e.g., AAPL, MSFT)",
        json_schema_extra={"examples": ["AAPL", "MSFT", "GOOGL"]},
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                {"ticker": "MSFT"},
            ]
        }
    }


class StockDataBatchRequest(StrictBaseModel, DateRangeMixin):
    """Validation for fetch_stock_data_batch tool."""

    tickers: list[TickerSymbol] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of ticker symbols (max 50)",
        json_schema_extra={"examples": [["AAPL", "MSFT", "GOOGL"]]},
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        """Validate and normalize ticker list."""
        return TickerValidator.validate_ticker_list(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tickers": ["AAPL", "MSFT", "GOOGL"], "start_date": "2024-01-01"},
                {
                    "tickers": ["SPY", "QQQ", "IWM"],
                    "start_date": "2024-06-01",
                    "end_date": "2024-12-31",
                },
            ]
        }
    }


class GetStockInfoRequest(StrictBaseModel):
    """Validation for get_stock_info tool."""

    ticker: TickerSymbol = Field(
        ..., description="Stock ticker symbol", json_schema_extra={"examples": ["AAPL"]}
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class GetNewsRequest(StrictBaseModel):
    """Validation for get_news_sentiment tool."""

    ticker: TickerSymbol = Field(
        ..., description="Stock ticker symbol", json_schema_extra={"examples": ["AAPL"]}
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class GetChartLinksRequest(StrictBaseModel):
    """Validation for get_chart_links tool."""

    ticker: TickerSymbol = Field(
        ..., description="Stock ticker symbol", json_schema_extra={"examples": ["AAPL"]}
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class CachedPriceDataRequest(StrictBaseModel):
    """Validation for get_cached_price_data tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    start_date: DateString = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: DateString | None = Field(
        default=None, description="End date in YYYY-MM-DD format (defaults to today)"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str | None) -> str | None:
        """Validate date format."""
        if v is not None:
            DateValidator.validate_date_string(v)
        return v

    @model_validator(mode="after")
    def validate_date_range(self):
        """Ensure end_date is after start_date."""
        if self.end_date is not None:
            DateValidator.validate_date_range(self.start_date, self.end_date)
        return self


class ClearCacheRequest(StrictBaseModel):
    """Validation for clear_cache tool."""

    ticker: TickerSymbol | None = Field(
        default=None, description="Specific ticker to clear (None to clear all)"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str | None) -> str | None:
        """Normalize ticker to uppercase if provided."""
        if v is not None:
            return TickerValidator.validate_ticker(v)
        return v
