"""
Stock analysis entity.

This entity represents a complete technical analysis of a stock.
"""

from dataclasses import dataclass
from datetime import datetime

from maverick_mcp.domain.value_objects.technical_indicators import (
    BollingerBands,
    MACDIndicator,
    PriceLevel,
    RSIIndicator,
    Signal,
    StochasticOscillator,
    TrendDirection,
    VolumeProfile,
)


@dataclass
class StockAnalysis:
    """
    Entity representing a comprehensive technical analysis of a stock.

    This is a domain entity that aggregates various technical indicators
    and analysis results for a specific stock at a point in time.
    """

    # Basic information
    symbol: str
    analysis_date: datetime
    current_price: float

    # Trend analysis
    trend_direction: TrendDirection
    trend_strength: float  # 0-100

    # Technical indicators
    rsi: RSIIndicator | None = None
    macd: MACDIndicator | None = None
    bollinger_bands: BollingerBands | None = None
    stochastic: StochasticOscillator | None = None

    # Price levels
    support_levels: list[PriceLevel] | None = None
    resistance_levels: list[PriceLevel] | None = None

    # Volume analysis
    volume_profile: VolumeProfile | None = None

    # Composite analysis
    composite_signal: Signal = Signal.NEUTRAL
    confidence_score: float = 0.0  # 0-100

    # Analysis metadata
    analysis_period_days: int = 365
    indicators_used: list[str] | None = None

    def __post_init__(self):
        """Initialize default values."""
        if self.support_levels is None:
            self.support_levels = []
        if self.resistance_levels is None:
            self.resistance_levels = []
        if self.indicators_used is None:
            self.indicators_used = []

    @property
    def has_bullish_setup(self) -> bool:
        """Check if the analysis indicates a bullish setup."""
        bullish_signals = [
            Signal.BUY,
            Signal.STRONG_BUY,
        ]
        return self.composite_signal in bullish_signals

    @property
    def has_bearish_setup(self) -> bool:
        """Check if the analysis indicates a bearish setup."""
        bearish_signals = [
            Signal.SELL,
            Signal.STRONG_SELL,
        ]
        return self.composite_signal in bearish_signals

    @property
    def nearest_support(self) -> PriceLevel | None:
        """Get the nearest support level below current price."""
        if not self.support_levels:
            return None
        below_price = [s for s in self.support_levels if s.price < self.current_price]
        if below_price:
            return max(below_price, key=lambda x: x.price)
        return None

    @property
    def nearest_resistance(self) -> PriceLevel | None:
        """Get the nearest resistance level above current price."""
        if not self.resistance_levels:
            return None
        above_price = [
            r for r in self.resistance_levels if r.price > self.current_price
        ]
        if above_price:
            return min(above_price, key=lambda x: x.price)
        return None

    @property
    def risk_reward_ratio(self) -> float | None:
        """Calculate risk/reward ratio based on nearest support/resistance."""
        support = self.nearest_support
        resistance = self.nearest_resistance

        if not support or not resistance:
            return None

        risk = self.current_price - support.price
        reward = resistance.price - self.current_price

        if risk <= 0:
            return None

        return reward / risk

    def get_indicator_summary(self) -> dict[str, str]:
        """Get a summary of all indicator signals."""
        summary = {}

        if self.rsi:
            summary["RSI"] = f"{self.rsi.value:.1f} ({self.rsi.signal.value})"

        if self.macd:
            summary["MACD"] = self.macd.signal.value

        if self.bollinger_bands:
            summary["Bollinger"] = self.bollinger_bands.signal.value

        if self.stochastic:
            summary["Stochastic"] = (
                f"{self.stochastic.k_value:.1f} ({self.stochastic.signal.value})"
            )

        if self.volume_profile:
            summary["Volume"] = f"{self.volume_profile.relative_volume:.1f}x average"

        return summary

    def get_key_levels(self) -> dict[str, float]:
        """Get key price levels for trading decisions."""
        levels = {
            "current_price": self.current_price,
        }

        if self.nearest_support:
            levels["nearest_support"] = self.nearest_support.price

        if self.nearest_resistance:
            levels["nearest_resistance"] = self.nearest_resistance.price

        if self.bollinger_bands:
            levels["bollinger_upper"] = self.bollinger_bands.upper_band
            levels["bollinger_lower"] = self.bollinger_bands.lower_band

        return levels

    def to_dict(self) -> dict:
        """Convert the analysis to a dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "analysis_date": self.analysis_date.isoformat(),
            "current_price": self.current_price,
            "trend": {
                "direction": self.trend_direction.value,
                "strength": self.trend_strength,
            },
            "indicators": self.get_indicator_summary(),
            "levels": self.get_key_levels(),
            "signal": self.composite_signal.value,
            "confidence": self.confidence_score,
            "risk_reward_ratio": self.risk_reward_ratio,
        }
