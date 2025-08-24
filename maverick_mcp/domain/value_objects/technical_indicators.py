"""
Value objects for technical indicators.

These are immutable objects representing technical analysis concepts
in the domain layer. They contain no infrastructure dependencies.
"""

from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal types."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TrendDirection(Enum):
    """Market trend directions."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass(frozen=True)
class RSIIndicator:
    """Relative Strength Index value object."""

    value: float
    period: int = 14

    def __post_init__(self):
        if not 0 <= self.value <= 100:
            raise ValueError("RSI must be between 0 and 100")
        if self.period <= 0:
            raise ValueError("Period must be positive")

    @property
    def is_overbought(self) -> bool:
        """Check if RSI indicates overbought condition."""
        return self.value >= 70

    @property
    def is_oversold(self) -> bool:
        """Check if RSI indicates oversold condition."""
        return self.value <= 30

    @property
    def signal(self) -> Signal:
        """Get trading signal based on RSI value."""
        if self.value >= 80:
            return Signal.STRONG_SELL
        elif self.value >= 70:
            return Signal.SELL
        elif self.value <= 20:
            return Signal.STRONG_BUY
        elif self.value <= 30:
            return Signal.BUY
        else:
            return Signal.NEUTRAL


@dataclass(frozen=True)
class MACDIndicator:
    """MACD (Moving Average Convergence Divergence) value object."""

    macd_line: float
    signal_line: float
    histogram: float
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    @property
    def is_bullish_crossover(self) -> bool:
        """Check if MACD crossed above signal line."""
        return self.macd_line > self.signal_line and self.histogram > 0

    @property
    def is_bearish_crossover(self) -> bool:
        """Check if MACD crossed below signal line."""
        return self.macd_line < self.signal_line and self.histogram < 0

    @property
    def signal(self) -> Signal:
        """Get trading signal based on MACD."""
        if self.is_bullish_crossover and self.histogram > 0.5:
            return Signal.STRONG_BUY
        elif self.is_bullish_crossover:
            return Signal.BUY
        elif self.is_bearish_crossover and self.histogram < -0.5:
            return Signal.STRONG_SELL
        elif self.is_bearish_crossover:
            return Signal.SELL
        else:
            return Signal.NEUTRAL


@dataclass(frozen=True)
class BollingerBands:
    """Bollinger Bands value object."""

    upper_band: float
    middle_band: float
    lower_band: float
    current_price: float
    period: int = 20
    std_dev: int = 2

    @property
    def bandwidth(self) -> float:
        """Calculate bandwidth (volatility indicator)."""
        return (self.upper_band - self.lower_band) / self.middle_band

    @property
    def percent_b(self) -> float:
        """Calculate %B (position within bands)."""
        denominator = self.upper_band - self.lower_band
        if denominator == 0:
            return 0.5  # Return middle if bands are flat
        return (self.current_price - self.lower_band) / denominator

    @property
    def is_squeeze(self) -> bool:
        """Check if bands are in a squeeze (low volatility)."""
        return self.bandwidth < 0.1

    @property
    def signal(self) -> Signal:
        """Get trading signal based on Bollinger Bands."""
        if self.current_price > self.upper_band:
            return Signal.SELL
        elif self.current_price < self.lower_band:
            return Signal.BUY
        elif self.percent_b > 0.8:
            return Signal.SELL
        elif self.percent_b < 0.2:
            return Signal.BUY
        else:
            return Signal.NEUTRAL


@dataclass(frozen=True)
class StochasticOscillator:
    """Stochastic Oscillator value object."""

    k_value: float
    d_value: float
    period: int = 14

    def __post_init__(self):
        if not 0 <= self.k_value <= 100:
            raise ValueError("%K must be between 0 and 100")
        if not 0 <= self.d_value <= 100:
            raise ValueError("%D must be between 0 and 100")

    @property
    def is_overbought(self) -> bool:
        """Check if stochastic indicates overbought."""
        return self.k_value >= 80

    @property
    def is_oversold(self) -> bool:
        """Check if stochastic indicates oversold."""
        return self.k_value <= 20

    @property
    def signal(self) -> Signal:
        """Get trading signal based on stochastic."""
        if self.k_value > self.d_value and self.k_value < 20:
            return Signal.BUY
        elif self.k_value < self.d_value and self.k_value > 80:
            return Signal.SELL
        elif self.is_oversold:
            return Signal.BUY
        elif self.is_overbought:
            return Signal.SELL
        else:
            return Signal.NEUTRAL


@dataclass(frozen=True)
class PriceLevel:
    """Support or resistance price level."""

    price: float
    strength: int  # 1-5, with 5 being strongest
    touches: int  # Number of times price touched this level

    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if not 1 <= self.strength <= 5:
            raise ValueError("Strength must be between 1 and 5")
        if self.touches < 0:
            raise ValueError("Touches must be non-negative")


@dataclass(frozen=True)
class VolumeProfile:
    """Volume analysis value object."""

    current_volume: int
    average_volume: float
    volume_trend: TrendDirection
    unusual_activity: bool

    @property
    def relative_volume(self) -> float:
        """Calculate volume relative to average."""
        return (
            self.current_volume / self.average_volume if self.average_volume > 0 else 0
        )

    @property
    def is_high_volume(self) -> bool:
        """Check if volume is significantly above average."""
        return self.relative_volume > 1.5

    @property
    def is_low_volume(self) -> bool:
        """Check if volume is significantly below average."""
        return self.relative_volume < 0.5
