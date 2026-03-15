"""
Validation models for options analysis tools.

This module provides Pydantic models for validating inputs
to all options chain and Greeks analysis tools.
"""

from typing import Literal

from pydantic import Field, field_validator

from .base import (
    StrictBaseModel,
    TickerSymbol,
    TickerValidator,
)


class OptionsChainRequest(StrictBaseModel):
    """Validation for options_get_chain tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    expiration: str | None = Field(
        default=None,
        description="Expiration date in YYYY-MM-DD format (nearest if omitted)",
    )
    min_volume: int = Field(
        default=10, ge=0, description="Minimum contract volume filter"
    )
    min_open_interest: int = Field(
        default=100, ge=0, description="Minimum open interest filter"
    )
    max_bid_ask_spread_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Maximum bid-ask spread as percentage of mid price",
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
                    "ticker": "MSFT",
                    "expiration": "2026-04-17",
                    "min_volume": 50,
                    "min_open_interest": 500,
                },
            ]
        }
    }


class GreeksRequest(StrictBaseModel):
    """Validation for options_calculate_greeks tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    strike: float = Field(..., gt=0, description="Strike price")
    expiration: str = Field(..., description="Expiration date in YYYY-MM-DD format")
    option_type: Literal["call", "put"] = Field(
        default="call", description="Option type: call or put"
    )
    risk_free_rate: float = Field(
        default=0.0425, ge=0.0, le=1.0, description="Risk-free rate (decimal)"
    )
    dividend_yield: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Dividend yield (decimal, auto-fetched if omitted)",
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
                    "strike": 200.0,
                    "expiration": "2026-04-17",
                    "option_type": "call",
                },
            ]
        }
    }


class OptionPriceRequest(StrictBaseModel):
    """Validation for options_price_option tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    strike: float = Field(..., gt=0, description="Strike price")
    expiration: str = Field(..., description="Expiration date in YYYY-MM-DD format")
    option_type: Literal["call", "put"] = Field(
        default="call", description="Option type: call or put"
    )
    model: Literal["bsm", "baw"] = Field(
        default="baw",
        description="Pricing model: bsm (Black-Scholes) or baw (Barone-Adesi Whaley)",
    )
    risk_free_rate: float = Field(
        default=0.0425, ge=0.0, le=1.0, description="Risk-free rate (decimal)"
    )
    dividend_yield: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Dividend yield (decimal, auto-fetched if omitted)",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class StrategyLeg(StrictBaseModel):
    """A single leg of a multi-leg options strategy."""

    strike: float = Field(..., gt=0, description="Strike price")
    option_type: Literal["call", "put"] = Field(
        ..., description="Option type: call or put"
    )
    action: Literal["buy", "sell"] = Field(..., description="Buy or sell this leg")
    quantity: int = Field(default=1, gt=0, description="Number of contracts")
    premium: float = Field(
        default=0.0, ge=0.0, description="Premium per share for this leg"
    )


class StrategyAnalysisRequest(StrictBaseModel):
    """Validation for options_analyze_strategy tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    strategy_type: Literal[
        "covered_call",
        "protective_put",
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
        "straddle",
        "strangle",
    ] = Field(default="covered_call", description="Strategy type")
    expiration: str | None = Field(
        default=None,
        description="Expiration date in YYYY-MM-DD format (nearest if omitted)",
    )
    legs: list[StrategyLeg] | None = Field(
        default=None,
        description="Custom legs (auto-built from strategy_type if omitted)",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class UnusualActivityRequest(StrictBaseModel):
    """Validation for options_unusual_activity tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    volume_oi_threshold: float = Field(
        default=2.0,
        gt=0.0,
        description="Volume/OI ratio threshold for flagging unusual activity",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)


class HedgePortfolioRequest(StrictBaseModel):
    """Validation for options_hedge_portfolio tool."""

    ticker: str | None = Field(
        default=None,
        description="Specific ticker to hedge (all positions if omitted)",
    )
    risk_level: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Risk tolerance 0-100 (lower = more conservative hedging)",
    )
    user_id: str = Field(default="default", description="User identifier")
    portfolio_name: str = Field(default="My Portfolio", description="Portfolio name")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str | None) -> str | None:
        """Normalize ticker to uppercase if provided."""
        if v is None:
            return v
        return TickerValidator.validate_ticker(v)


class IVAnalysisRequest(StrictBaseModel):
    """Validation for options_iv_analysis tool."""

    ticker: TickerSymbol = Field(..., description="Stock ticker symbol")
    expiration: str | None = Field(
        default=None,
        description="Specific expiration for skew analysis (all if omitted)",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return TickerValidator.validate_ticker(v)
