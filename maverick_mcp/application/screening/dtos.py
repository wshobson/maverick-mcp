"""
Screening application DTOs (Data Transfer Objects).

This module contains DTOs for request/response communication
between the API layer and application layer.
"""

from typing import Any

from pydantic import BaseModel, Field, validator

from maverick_mcp.domain.screening.value_objects import ScreeningStrategy


class ScreeningRequestDTO(BaseModel):
    """
    DTO for screening requests from the API layer.

    This DTO validates and structures incoming screening requests.
    """

    strategy: str = Field(
        description="Screening strategy to use", example="maverick_bullish"
    )
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum number of results to return"
    )

    # Filtering criteria
    min_momentum_score: float | None = Field(
        default=None, ge=0, le=100, description="Minimum momentum score"
    )
    max_momentum_score: float | None = Field(
        default=None, ge=0, le=100, description="Maximum momentum score"
    )
    min_volume: int | None = Field(
        default=None, ge=0, description="Minimum average daily volume"
    )
    max_volume: int | None = Field(
        default=None, ge=0, description="Maximum average daily volume"
    )
    min_price: float | None = Field(
        default=None, gt=0, description="Minimum stock price"
    )
    max_price: float | None = Field(
        default=None, gt=0, description="Maximum stock price"
    )
    min_combined_score: int | None = Field(
        default=None, ge=0, description="Minimum combined score for bullish screening"
    )
    min_bear_score: int | None = Field(
        default=None, ge=0, description="Minimum bear score for bearish screening"
    )
    min_adr_percentage: float | None = Field(
        default=None, ge=0, description="Minimum average daily range percentage"
    )
    max_adr_percentage: float | None = Field(
        default=None, ge=0, description="Maximum average daily range percentage"
    )

    # Pattern filters
    require_pattern_detected: bool = Field(
        default=False, description="Require pattern detection signal"
    )
    require_squeeze: bool = Field(default=False, description="Require squeeze signal")
    require_consolidation: bool = Field(
        default=False, description="Require consolidation pattern"
    )
    require_entry_signal: bool = Field(
        default=False, description="Require entry signal"
    )

    # Moving average filters
    require_above_sma50: bool = Field(
        default=False, description="Require price above SMA 50"
    )
    require_above_sma150: bool = Field(
        default=False, description="Require price above SMA 150"
    )
    require_above_sma200: bool = Field(
        default=False, description="Require price above SMA 200"
    )
    require_ma_alignment: bool = Field(
        default=False,
        description="Require proper moving average alignment (50>150>200)",
    )

    # Sorting options
    sort_field: str | None = Field(
        default=None, description="Field to sort by (strategy default if not specified)"
    )
    sort_descending: bool = Field(default=True, description="Sort in descending order")

    @validator("strategy")
    def validate_strategy(cls, v):
        """Validate that strategy is a known screening strategy."""
        valid_strategies = [s.value for s in ScreeningStrategy]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        return v

    @validator("max_momentum_score")
    def validate_momentum_score_range(cls, v, values):
        """Validate that max_momentum_score >= min_momentum_score if both specified."""
        if (
            v is not None
            and "min_momentum_score" in values
            and values["min_momentum_score"] is not None
        ):
            if v < values["min_momentum_score"]:
                raise ValueError(
                    "max_momentum_score cannot be less than min_momentum_score"
                )
        return v

    @validator("max_volume")
    def validate_volume_range(cls, v, values):
        """Validate that max_volume >= min_volume if both specified."""
        if (
            v is not None
            and "min_volume" in values
            and values["min_volume"] is not None
        ):
            if v < values["min_volume"]:
                raise ValueError("max_volume cannot be less than min_volume")
        return v

    @validator("max_price")
    def validate_price_range(cls, v, values):
        """Validate that max_price >= min_price if both specified."""
        if v is not None and "min_price" in values and values["min_price"] is not None:
            if v < values["min_price"]:
                raise ValueError("max_price cannot be less than min_price")
        return v

    @validator("sort_field")
    def validate_sort_field(cls, v):
        """Validate sort field if specified."""
        if v is not None:
            valid_fields = {
                "combined_score",
                "bear_score",
                "momentum_score",
                "close_price",
                "volume",
                "avg_volume_30d",
                "adr_percentage",
                "quality_score",
            }
            if v not in valid_fields:
                raise ValueError(f"Invalid sort field. Must be one of: {valid_fields}")
        return v


class ScreeningResultDTO(BaseModel):
    """
    DTO for individual screening results.

    This DTO represents a single stock screening result for API responses.
    """

    stock_symbol: str = Field(description="Stock ticker symbol")
    screening_date: str = Field(description="Date when screening was performed")
    close_price: float = Field(description="Current closing price")
    volume: int = Field(description="Current volume")
    momentum_score: float = Field(description="Momentum score (0-100)")
    adr_percentage: float = Field(description="Average daily range percentage")

    # Technical indicators
    ema_21: float = Field(description="21-period exponential moving average")
    sma_50: float = Field(description="50-period simple moving average")
    sma_150: float = Field(description="150-period simple moving average")
    sma_200: float = Field(description="200-period simple moving average")
    avg_volume_30d: float = Field(description="30-day average volume")
    atr: float = Field(description="Average True Range")

    # Pattern signals
    pattern: str | None = Field(default=None, description="Detected pattern")
    squeeze: str | None = Field(default=None, description="Squeeze signal")
    consolidation: str | None = Field(
        default=None, description="Consolidation pattern signal"
    )
    entry_signal: str | None = Field(default=None, description="Entry signal")

    # Scores
    combined_score: int = Field(description="Combined bullish score")
    bear_score: int = Field(description="Bearish score")
    quality_score: int = Field(description="Overall quality score")

    # Business rule indicators
    is_bullish: bool = Field(description="Meets bullish setup criteria")
    is_bearish: bool = Field(description="Meets bearish setup criteria")
    is_trending: bool = Field(description="Meets trending criteria")
    risk_reward_ratio: float = Field(description="Calculated risk/reward ratio")

    # Bearish-specific fields (optional)
    rsi_14: float | None = Field(default=None, description="14-period RSI")
    macd: float | None = Field(default=None, description="MACD line")
    macd_signal: float | None = Field(default=None, description="MACD signal line")
    macd_histogram: float | None = Field(default=None, description="MACD histogram")
    distribution_days_20: int | None = Field(
        default=None, description="Distribution days in last 20 days"
    )
    atr_contraction: bool | None = Field(
        default=None, description="ATR contraction detected"
    )
    big_down_volume: bool | None = Field(
        default=None, description="Big down volume detected"
    )


class ScreeningCollectionDTO(BaseModel):
    """
    DTO for screening result collections.

    This DTO represents the complete response for a screening operation.
    """

    strategy_used: str = Field(description="Screening strategy that was used")
    screening_timestamp: str = Field(description="When the screening was performed")
    total_candidates_analyzed: int = Field(
        description="Total number of candidates analyzed"
    )
    results_returned: int = Field(description="Number of results returned")
    results: list[ScreeningResultDTO] = Field(
        description="Individual screening results"
    )

    # Statistics and metadata
    statistics: dict[str, Any] = Field(description="Collection statistics")
    applied_filters: dict[str, Any] = Field(description="Filters that were applied")
    sorting_applied: dict[str, Any] = Field(description="Sorting configuration used")

    # Status information
    status: str = Field(default="success", description="Operation status")
    execution_time_ms: float | None = Field(
        default=None, description="Execution time in milliseconds"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Any warnings during processing"
    )


class AllScreeningResultsDTO(BaseModel):
    """
    DTO for comprehensive screening results across all strategies.

    This DTO represents results from all available screening strategies.
    """

    screening_timestamp: str = Field(description="When the screening was performed")
    strategies_executed: list[str] = Field(
        description="List of strategies that were executed"
    )

    # Results by strategy
    maverick_bullish: ScreeningCollectionDTO | None = Field(
        default=None, description="Maverick bullish screening results"
    )
    maverick_bearish: ScreeningCollectionDTO | None = Field(
        default=None, description="Maverick bearish screening results"
    )
    trending: ScreeningCollectionDTO | None = Field(
        default=None, description="Trending screening results"
    )

    # Cross-strategy analysis
    cross_strategy_analysis: dict[str, Any] = Field(
        description="Analysis across multiple strategies"
    )

    # Overall statistics
    overall_summary: dict[str, Any] = Field(
        description="Summary statistics across all strategies"
    )

    # Status information
    status: str = Field(default="success", description="Operation status")
    execution_time_ms: float | None = Field(
        default=None, description="Total execution time in milliseconds"
    )
    errors: list[str] = Field(
        default_factory=list, description="Any errors during processing"
    )


class ScreeningStatisticsDTO(BaseModel):
    """
    DTO for screening statistics and analytics.

    This DTO provides comprehensive analytics and business intelligence
    for screening operations.
    """

    strategy: str | None = Field(
        default=None, description="Strategy analyzed (None for all)"
    )
    timestamp: str = Field(description="When the analysis was performed")

    # Single strategy statistics
    statistics: dict[str, Any] | None = Field(
        default=None, description="Statistics for single strategy analysis"
    )

    # Multi-strategy statistics
    overall_summary: dict[str, Any] | None = Field(
        default=None, description="Summary across all strategies"
    )
    by_strategy: dict[str, dict[str, Any]] | None = Field(
        default=None, description="Statistics broken down by strategy"
    )
    cross_strategy_analysis: dict[str, Any] | None = Field(
        default=None, description="Cross-strategy insights and analysis"
    )

    # Metadata
    analysis_scope: str = Field(description="Scope of the analysis (single/all)")
    results_analyzed: int = Field(description="Total number of results analyzed")


class ErrorResponseDTO(BaseModel):
    """
    DTO for error responses.

    This DTO provides standardized error information for API responses.
    """

    status: str = Field(default="error", description="Response status")
    error_code: str = Field(description="Machine-readable error code")
    error_message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    timestamp: str = Field(description="When the error occurred")
    request_id: str | None = Field(
        default=None, description="Request identifier for tracking"
    )
