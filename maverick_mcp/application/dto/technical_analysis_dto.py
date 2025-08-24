"""
Data Transfer Objects for technical analysis.

These DTOs are used to transfer data between the application layer
and the API layer, providing a stable contract for API responses.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class RSIAnalysisDTO(BaseModel):
    """RSI analysis response DTO."""

    current_value: float = Field(..., description="Current RSI value")
    period: int = Field(..., description="RSI calculation period")
    signal: str = Field(..., description="Trading signal")
    is_overbought: bool = Field(..., description="Whether RSI indicates overbought")
    is_oversold: bool = Field(..., description="Whether RSI indicates oversold")
    interpretation: str = Field(..., description="Human-readable interpretation")


class MACDAnalysisDTO(BaseModel):
    """MACD analysis response DTO."""

    macd_line: float = Field(..., description="MACD line value")
    signal_line: float = Field(..., description="Signal line value")
    histogram: float = Field(..., description="MACD histogram value")
    signal: str = Field(..., description="Trading signal")
    is_bullish_crossover: bool = Field(..., description="Bullish crossover detected")
    is_bearish_crossover: bool = Field(..., description="Bearish crossover detected")
    interpretation: str = Field(..., description="Human-readable interpretation")


class BollingerBandsDTO(BaseModel):
    """Bollinger Bands analysis response DTO."""

    upper_band: float = Field(..., description="Upper band value")
    middle_band: float = Field(..., description="Middle band (SMA) value")
    lower_band: float = Field(..., description="Lower band value")
    current_price: float = Field(..., description="Current stock price")
    bandwidth: float = Field(..., description="Band width (volatility indicator)")
    percent_b: float = Field(..., description="Position within bands (0-1)")
    signal: str = Field(..., description="Trading signal")
    interpretation: str = Field(..., description="Human-readable interpretation")


class StochasticDTO(BaseModel):
    """Stochastic oscillator response DTO."""

    k_value: float = Field(..., description="%K value")
    d_value: float = Field(..., description="%D value")
    signal: str = Field(..., description="Trading signal")
    is_overbought: bool = Field(..., description="Whether indicating overbought")
    is_oversold: bool = Field(..., description="Whether indicating oversold")
    interpretation: str = Field(..., description="Human-readable interpretation")


class PriceLevelDTO(BaseModel):
    """Price level (support/resistance) DTO."""

    price: float = Field(..., description="Price level")
    strength: int = Field(..., ge=1, le=5, description="Level strength (1-5)")
    touches: int = Field(..., description="Number of times tested")
    distance_from_current: float = Field(
        ..., description="Distance from current price (%)"
    )


class VolumeAnalysisDTO(BaseModel):
    """Volume analysis response DTO."""

    current_volume: int = Field(..., description="Current trading volume")
    average_volume: float = Field(..., description="Average volume")
    relative_volume: float = Field(..., description="Volume relative to average")
    volume_trend: str = Field(..., description="Volume trend direction")
    unusual_activity: bool = Field(..., description="Unusual volume detected")
    interpretation: str = Field(..., description="Human-readable interpretation")


class TrendAnalysisDTO(BaseModel):
    """Trend analysis response DTO."""

    direction: str = Field(..., description="Trend direction")
    strength: float = Field(..., ge=0, le=100, description="Trend strength (0-100)")
    interpretation: str = Field(..., description="Human-readable interpretation")


class TechnicalAnalysisRequestDTO(BaseModel):
    """Request DTO for technical analysis."""

    symbol: str = Field(..., description="Stock ticker symbol")
    days: int = Field(
        default=365, ge=30, le=1825, description="Days of historical data"
    )
    indicators: list[str] | None = Field(
        default=None, description="Specific indicators to calculate (default: all)"
    )


class CompleteTechnicalAnalysisDTO(BaseModel):
    """Complete technical analysis response DTO."""

    symbol: str = Field(..., description="Stock ticker symbol")
    analysis_date: datetime = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current stock price")

    # Trend
    trend: TrendAnalysisDTO = Field(..., description="Trend analysis")

    # Indicators
    rsi: RSIAnalysisDTO | None = Field(None, description="RSI analysis")
    macd: MACDAnalysisDTO | None = Field(None, description="MACD analysis")
    bollinger_bands: BollingerBandsDTO | None = Field(
        None, description="Bollinger Bands"
    )
    stochastic: StochasticDTO | None = Field(None, description="Stochastic oscillator")

    # Levels
    support_levels: list[PriceLevelDTO] = Field(
        default_factory=list, description="Support levels"
    )
    resistance_levels: list[PriceLevelDTO] = Field(
        default_factory=list, description="Resistance levels"
    )

    # Volume
    volume_analysis: VolumeAnalysisDTO | None = Field(
        None, description="Volume analysis"
    )

    # Overall analysis
    composite_signal: str = Field(..., description="Overall trading signal")
    confidence_score: float = Field(
        ..., ge=0, le=100, description="Analysis confidence (0-100)"
    )
    risk_reward_ratio: float | None = Field(None, description="Risk/reward ratio")

    # Summary
    summary: str = Field(..., description="Executive summary of analysis")
    key_levels: dict[str, float] = Field(
        ..., description="Key price levels for trading"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
