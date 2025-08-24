"""
Validation models for technical analysis tools.

This module provides Pydantic models for validating inputs
to all technical analysis tools.
"""

from pydantic import Field, field_validator

from .base import (
    PositiveInt,
    StrictBaseModel,
    TickerSymbol,
    TickerValidator,
)


class RSIAnalysisRequest(StrictBaseModel):
    """Validation for get_rsi_analysis tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    period: PositiveInt = Field(
        default=14, le=100, description="RSI period (typically 14)"
    )
    days: PositiveInt = Field(
        default=365,
        le=3650,  # Max 10 years
        description="Number of days of historical data",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ticker": "AAPL", "period": 14, "days": 365},
                {"ticker": "MSFT", "period": 21, "days": 90},
            ]
        }
    }


class MACDAnalysisRequest(StrictBaseModel):
    """Validation for get_macd_analysis tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    fast_period: PositiveInt = Field(default=12, le=50, description="Fast EMA period")
    slow_period: PositiveInt = Field(default=26, le=100, description="Slow EMA period")
    signal_period: PositiveInt = Field(
        default=9, le=50, description="Signal line period"
    )
    days: PositiveInt = Field(
        default=365, le=3650, description="Number of days of historical data"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    @field_validator("slow_period")
    @classmethod
    def validate_slow_greater_than_fast(cls, v: int, info) -> int:
        """Ensure slow period is greater than fast period."""
        fast = info.data.get("fast_period", 12)
        if v <= fast:
            raise ValueError(
                f"Slow period ({v}) must be greater than fast period ({fast})"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                },
                {
                    "ticker": "GOOGL",
                    "fast_period": 10,
                    "slow_period": 20,
                    "signal_period": 5,
                    "days": 180,
                },
            ]
        }
    }


class SupportResistanceRequest(StrictBaseModel):
    """Validation for get_support_resistance tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    days: PositiveInt = Field(
        default=365, le=3650, description="Number of days of historical data"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class TechnicalAnalysisRequest(StrictBaseModel):
    """Validation for get_full_technical_analysis tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    days: PositiveInt = Field(
        default=365, le=3650, description="Number of days of historical data"
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ticker": "AAPL", "days": 365},
                {"ticker": "TSLA", "days": 90},
            ]
        }
    }


class StockChartRequest(StrictBaseModel):
    """Validation for get_stock_chart_analysis tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)
