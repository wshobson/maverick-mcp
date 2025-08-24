"""
Technical Analysis Constants and Configuration

This module centralizes all technical analysis parameters and thresholds
to follow Open/Closed Principle and eliminate magic numbers.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TechnicalAnalysisConfig:
    """Configuration class for technical analysis parameters."""

    # RSI Configuration
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0

    # Moving Average Configuration
    SMA_SHORT_PERIOD: int = 50
    SMA_LONG_PERIOD: int = 200
    EMA_PERIOD: int = 21
    EMA_FAST_PERIOD: int = 12
    EMA_SLOW_PERIOD: int = 26

    # MACD Configuration
    MACD_FAST_PERIOD: int = 12
    MACD_SLOW_PERIOD: int = 26
    MACD_SIGNAL_PERIOD: int = 9

    # Bollinger Bands Configuration
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0

    # Stochastic Oscillator Configuration
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3
    STOCH_OVERBOUGHT: float = 80.0
    STOCH_OVERSOLD: float = 20.0

    # Volume Analysis Configuration
    HIGH_VOLUME_THRESHOLD: float = 1.5  # 1.5x average volume
    LOW_VOLUME_THRESHOLD: float = 0.7  # 0.7x average volume
    VOLUME_SMA_PERIOD: int = 20

    # Chart Pattern Configuration
    PATTERN_SIMILARITY_THRESHOLD: float = 0.05
    PATTERN_MIN_SEPARATION: int = 5

    # Support and Resistance Configuration
    SUPPORT_RESISTANCE_LOOKBACK: int = 20
    SUPPORT_RESISTANCE_TOLERANCE: float = 0.02  # 2% tolerance

    # ATR Configuration
    ATR_PERIOD: int = 14

    # CCI Configuration
    CCI_PERIOD: int = 20
    CCI_OVERBOUGHT: float = 100.0
    CCI_OVERSOLD: float = -100.0

    # Williams %R Configuration
    WILLIAMS_R_PERIOD: int = 14
    WILLIAMS_R_OVERBOUGHT: float = -20.0
    WILLIAMS_R_OVERSOLD: float = -80.0


# Global configuration instance
TECHNICAL_CONFIG: Final[TechnicalAnalysisConfig] = TechnicalAnalysisConfig()


# Screening Strategy Configuration
@dataclass(frozen=True)
class ScreeningConfig:
    """Configuration for stock screening strategies."""

    # Maverick Bullish Strategy
    MIN_VOLUME: int = 1_000_000
    MIN_PRICE: float = 5.0
    MAX_PRICE: float = 500.0
    MIN_MARKET_CAP: float = 100_000_000  # $100M

    # RSI Requirements
    RSI_MIN_BULLISH: float = 30.0
    RSI_MAX_BULLISH: float = 70.0

    # Volume Requirements
    VOLUME_SPIKE_THRESHOLD: float = 1.5  # 1.5x average volume

    # Moving Average Requirements
    MA_CROSSOVER_PERIOD: int = 5  # Days to check for crossover

    # Bear Strategy Thresholds
    RSI_MAX_BEARISH: float = 30.0
    PRICE_DECLINE_THRESHOLD: float = -0.10  # 10% decline

    # Trending Breakout Strategy
    BREAKOUT_VOLUME_MULTIPLIER: float = 2.0
    BREAKOUT_PRICE_THRESHOLD: float = 0.05  # 5% price increase

    # General Filtering
    EXCLUDE_PENNY_STOCKS: bool = True
    EXCLUDE_ETFS: bool = False
    MAX_RESULTS_PER_STRATEGY: int = 50


# Global screening configuration instance
SCREENING_CONFIG: Final[ScreeningConfig] = ScreeningConfig()
