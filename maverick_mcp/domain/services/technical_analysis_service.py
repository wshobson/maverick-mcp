"""
Technical analysis domain service.

This service contains pure business logic for technical analysis calculations.
It has no dependencies on infrastructure, databases, or external APIs.
"""

import numpy as np
import pandas as pd

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


class TechnicalAnalysisService:
    """
    Domain service for technical analysis calculations.

    This service contains pure business logic and mathematical calculations
    for technical indicators. It operates on price data and returns
    domain value objects.
    """

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> RSIIndicator:
        """
        Calculate the Relative Strength Index.

        Args:
            prices: Series of closing prices
            period: RSI period (default: 14)

        Returns:
            RSIIndicator value object
        """
        if len(prices) < period:
            raise ValueError(f"Need at least {period} prices to calculate RSI")

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        # Handle edge case where there are no losses
        rs = avg_gain / avg_loss if avg_loss.iloc[-1] != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))

        # Get the latest RSI value
        current_rsi = float(rsi.iloc[-1])

        return RSIIndicator(value=current_rsi, period=period)

    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> MACDIndicator:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of closing prices
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            MACDIndicator value object
        """
        if len(prices) < slow_period:
            raise ValueError(f"Need at least {slow_period} prices to calculate MACD")

        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Get current values
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])

        return MACDIndicator(
            macd_line=current_macd,
            signal_line=current_signal,
            histogram=current_histogram,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> BollingerBands:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of closing prices
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2)

        Returns:
            BollingerBands value object
        """
        if len(prices) < period:
            raise ValueError(
                f"Need at least {period} prices to calculate Bollinger Bands"
            )

        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()

        # Calculate standard deviation
        std = prices.rolling(window=period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        # Get current values
        current_price = float(prices.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_middle = float(middle_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])

        return BollingerBands(
            upper_band=current_upper,
            middle_band=current_middle,
            lower_band=current_lower,
            current_price=current_price,
            period=period,
            std_dev=std_dev,
        )

    def calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> StochasticOscillator:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Look-back period (default: 14)

        Returns:
            StochasticOscillator value object
        """
        if len(close) < period:
            raise ValueError(f"Need at least {period} prices to calculate Stochastic")

        # Calculate %K
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))

        # Calculate %D (3-period SMA of %K)
        d_percent = k_percent.rolling(window=3).mean()

        # Get current values
        current_k = float(k_percent.iloc[-1])
        current_d = float(d_percent.iloc[-1])

        return StochasticOscillator(k_value=current_k, d_value=current_d, period=period)

    def identify_trend(self, prices: pd.Series, period: int = 50) -> TrendDirection:
        """
        Identify the current price trend.

        Args:
            prices: Series of closing prices
            period: Period for trend calculation (default: 50)

        Returns:
            TrendDirection enum value
        """
        if len(prices) < period:
            return TrendDirection.SIDEWAYS

        # Calculate moving averages
        sma_short = prices.rolling(window=period // 2).mean()
        sma_long = prices.rolling(window=period).mean()

        # Calculate trend strength
        current_price = prices.iloc[-1]
        short_ma = sma_short.iloc[-1]
        long_ma = sma_long.iloc[-1]

        # Calculate percentage differences
        price_vs_short = (current_price - short_ma) / short_ma * 100
        short_vs_long = (short_ma - long_ma) / long_ma * 100

        # Determine trend
        if price_vs_short > 5 and short_vs_long > 3:
            return TrendDirection.STRONG_UPTREND
        elif price_vs_short > 2 and short_vs_long > 1:
            return TrendDirection.UPTREND
        elif price_vs_short < -5 and short_vs_long < -3:
            return TrendDirection.STRONG_DOWNTREND
        elif price_vs_short < -2 and short_vs_long < -1:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS

    def analyze_volume(self, volume: pd.Series, period: int = 20) -> VolumeProfile:
        """
        Analyze volume patterns.

        Args:
            volume: Series of volume data
            period: Period for average calculation (default: 20)

        Returns:
            VolumeProfile value object
        """
        if len(volume) < period:
            raise ValueError(f"Need at least {period} volume data points")

        # Calculate average volume
        avg_volume = float(volume.rolling(window=period).mean().iloc[-1])
        current_volume = int(volume.iloc[-1])

        # Determine volume trend
        recent_avg = float(volume.tail(5).mean())
        older_avg = float(volume.iloc[-period:-5].mean())

        if recent_avg > older_avg * 1.2:
            volume_trend = TrendDirection.UPTREND
        elif recent_avg < older_avg * 0.8:
            volume_trend = TrendDirection.DOWNTREND
        else:
            volume_trend = TrendDirection.SIDEWAYS

        # Check for unusual activity
        unusual_activity = current_volume > avg_volume * 2

        return VolumeProfile(
            current_volume=current_volume,
            average_volume=avg_volume,
            volume_trend=volume_trend,
            unusual_activity=unusual_activity,
        )

    def calculate_composite_signal(
        self,
        rsi: RSIIndicator | None = None,
        macd: MACDIndicator | None = None,
        bollinger: BollingerBands | None = None,
        stochastic: StochasticOscillator | None = None,
    ) -> Signal:
        """
        Calculate a composite trading signal from multiple indicators.

        Args:
            rsi: RSI indicator
            macd: MACD indicator
            bollinger: Bollinger Bands indicator
            stochastic: Stochastic indicator

        Returns:
            Composite Signal
        """
        signals = []
        weights = []

        # Collect signals and weights
        if rsi:
            signals.append(rsi.signal)
            weights.append(2.0)  # RSI has higher weight

        if macd:
            signals.append(macd.signal)
            weights.append(1.5)  # MACD has medium weight

        if bollinger:
            signals.append(bollinger.signal)
            weights.append(1.0)

        if stochastic:
            signals.append(stochastic.signal)
            weights.append(1.0)

        if not signals:
            return Signal.NEUTRAL

        # Convert signals to numeric scores
        signal_scores = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.NEUTRAL: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2,
        }

        # Calculate weighted average
        total_score = sum(
            signal_scores[signal] * weight
            for signal, weight in zip(signals, weights, strict=False)
        )
        total_weight = sum(weights)
        avg_score = total_score / total_weight

        # Map back to signal
        if avg_score >= 1.5:
            return Signal.STRONG_BUY
        elif avg_score >= 0.5:
            return Signal.BUY
        elif avg_score <= -1.5:
            return Signal.STRONG_SELL
        elif avg_score <= -0.5:
            return Signal.SELL
        else:
            return Signal.NEUTRAL

    def find_support_levels(self, df: pd.DataFrame) -> list[PriceLevel]:
        """
        Find support levels in the price data.

        Args:
            df: DataFrame with OHLC price data

        Returns:
            List of support PriceLevel objects
        """
        lows = df["low"].rolling(window=20).min()
        unique_levels = lows.dropna().unique()

        support_levels = []
        current_price = df["close"].iloc[-1]

        # Filter for levels below current price first, then sort and take closest 5
        below_current = [
            level
            for level in unique_levels
            if level > 0 and level < current_price * 0.98
        ]

        for level in sorted(below_current, reverse=True)[
            :5
        ]:  # Top 5 levels below current
            # Safe division with level > 0 check above
            touches = len(df[abs(df["low"] - level) / level < 0.01])
            strength = min(5, touches)
            support_levels.append(
                PriceLevel(price=float(level), strength=strength, touches=touches)
            )

        return support_levels

    def find_resistance_levels(self, df: pd.DataFrame) -> list[PriceLevel]:
        """
        Find resistance levels in the price data.

        Args:
            df: DataFrame with OHLC price data

        Returns:
            List of resistance PriceLevel objects
        """
        highs = df["high"].rolling(window=20).max()
        unique_levels = highs.dropna().unique()

        resistance_levels = []
        current_price = df["close"].iloc[-1]

        # Filter for levels above current price first, then sort and take closest 5
        above_current = [
            level
            for level in unique_levels
            if level > 0 and level > current_price * 1.02
        ]

        for level in sorted(above_current)[:5]:  # Bottom 5 levels above current
            # Safe division with level > 0 check above
            touches = len(df[abs(df["high"] - level) / level < 0.01])
            strength = min(5, touches)
            resistance_levels.append(
                PriceLevel(price=float(level), strength=strength, touches=touches)
            )

        return resistance_levels
