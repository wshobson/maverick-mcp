"""
Validation models for portfolio analysis tools.

This module provides Pydantic models for validating inputs
to all portfolio-related tools.
"""

from pydantic import Field, field_validator

from .base import (
    Percentage,
    PositiveInt,
    StrictBaseModel,
    TickerSymbol,
    TickerValidator,
)


class RiskAnalysisRequest(StrictBaseModel):
    """Validation for risk_adjusted_analysis tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    risk_level: Percentage = Field(
        default=50.0,
        description="Risk tolerance from 0 (conservative) to 100 (aggressive)",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ticker": "AAPL", "risk_level": 50.0},
                {"ticker": "TSLA", "risk_level": 75.0},
                {"ticker": "JNJ", "risk_level": 25.0},
            ]
        }
    }


class PortfolioComparisonRequest(StrictBaseModel):
    """Validation for compare_tickers tool."""

    tickers: list[TickerSymbol] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="List of ticker symbols to compare (2-20 tickers)",
    )
    days: PositiveInt = Field(
        default=90,
        le=1825,  # Max 5 years
        description="Number of days of historical data for comparison",
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        """Validate and normalize ticker list."""
        tickers = TickerValidator.validate_ticker_list(v)
        if len(tickers) < 2:
            raise ValueError("At least 2 unique tickers are required for comparison")
        return tickers

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tickers": ["AAPL", "MSFT", "GOOGL"], "days": 90},
                {"tickers": ["SPY", "QQQ", "IWM", "DIA"], "days": 180},
            ]
        }
    }


class CorrelationAnalysisRequest(StrictBaseModel):
    """Validation for portfolio_correlation_analysis tool."""

    tickers: list[TickerSymbol] = Field(
        ...,
        min_length=2,
        max_length=30,
        description="List of ticker symbols for correlation analysis",
    )
    days: PositiveInt = Field(
        default=252,  # 1 trading year
        ge=30,  # Need at least 30 days for meaningful correlation
        le=2520,  # Max 10 years
        description="Number of days for correlation calculation",
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        """Validate and normalize ticker list."""
        tickers = TickerValidator.validate_ticker_list(v)
        if len(tickers) < 2:
            raise ValueError(
                "At least 2 unique tickers are required for correlation analysis"
            )
        return tickers

    @field_validator("days")
    @classmethod
    def validate_days_for_correlation(cls, v: int) -> int:
        """Ensure enough days for meaningful correlation."""
        if v < 30:
            raise ValueError(
                "At least 30 days of data required for meaningful correlation analysis"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"], "days": 252},
                {
                    "tickers": ["SPY", "TLT", "GLD", "DBC", "VNQ"],
                    "days": 504,  # 2 years
                },
            ]
        }
    }
