"""
Market Analyzer Agent for intelligent market regime detection.

This agent analyzes market conditions to determine the current market regime
(trending, ranging, volatile, etc.) and provides context for strategy selection.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from maverick_mcp.data.cache import CacheManager
from maverick_mcp.providers.stock_data import EnhancedStockDataProvider
from maverick_mcp.workflows.state import BacktestingWorkflowState

logger = logging.getLogger(__name__)


class MarketAnalyzerAgent:
    """Intelligent market regime analyzer for backtesting workflows."""

    def __init__(
        self,
        data_provider: EnhancedStockDataProvider | None = None,
        cache_manager: CacheManager | None = None,
    ):
        """Initialize market analyzer agent.

        Args:
            data_provider: Stock data provider instance
            cache_manager: Cache manager for performance optimization
        """
        self.data_provider = data_provider or EnhancedStockDataProvider()
        self.cache = cache_manager or CacheManager()

        # Market regime detection thresholds
        self.TREND_THRESHOLD = 0.15  # 15% for strong trend
        self.VOLATILITY_THRESHOLD = 0.02  # 2% daily volatility threshold
        self.VOLUME_THRESHOLD = 1.5  # 1.5x average volume for high volume

        # Analysis periods for different regimes
        self.SHORT_PERIOD = 20  # Short-term trend analysis
        self.MEDIUM_PERIOD = 50  # Medium-term trend analysis
        self.LONG_PERIOD = 200  # Long-term trend analysis

        logger.info("MarketAnalyzerAgent initialized")

    async def analyze_market_regime(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Analyze market regime and update state.

        Args:
            state: Current workflow state

        Returns:
            Updated state with market regime analysis
        """
        start_time = datetime.now()

        try:
            logger.info(f"Analyzing market regime for {state['symbol']}")

            # Get historical data for analysis
            extended_start = self._calculate_extended_start_date(state["start_date"])
            price_data = await self._get_price_data(
                state["symbol"], extended_start, state["end_date"]
            )

            if price_data is None or len(price_data) < self.LONG_PERIOD:
                raise ValueError(
                    f"Insufficient data for market regime analysis: {state['symbol']}"
                )

            # Perform comprehensive market analysis
            regime_analysis = self._perform_regime_analysis(price_data)

            # Update state with analysis results
            state["market_regime"] = regime_analysis["regime"]
            state["regime_confidence"] = regime_analysis["confidence"]
            state["regime_indicators"] = regime_analysis["indicators"]
            state["volatility_percentile"] = regime_analysis["volatility_percentile"]
            state["trend_strength"] = regime_analysis["trend_strength"]
            state["market_conditions"] = regime_analysis["market_conditions"]
            state["volume_profile"] = regime_analysis["volume_profile"]
            state["support_resistance_levels"] = regime_analysis["support_resistance"]

            # Record execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            state["regime_analysis_time_ms"] = execution_time

            # Update workflow status
            state["workflow_status"] = "selecting_strategies"
            state["current_step"] = "market_analysis_completed"
            state["steps_completed"].append("market_regime_analysis")

            logger.info(
                f"Market regime analysis completed for {state['symbol']}: "
                f"{state['market_regime']} (confidence: {state['regime_confidence']:.2f})"
            )

            return state

        except Exception as e:
            error_info = {
                "step": "market_regime_analysis",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbol": state["symbol"],
            }
            state["errors_encountered"].append(error_info)

            # Set fallback regime
            state["market_regime"] = "unknown"
            state["regime_confidence"] = 0.0
            state["fallback_strategies_used"].append("regime_detection_fallback")

            logger.error(f"Market regime analysis failed for {state['symbol']}: {e}")
            return state

    def _calculate_extended_start_date(self, start_date: str) -> str:
        """Calculate extended start date to ensure sufficient data for analysis."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        # Add extra buffer for technical indicators
        extended_start = start_dt - timedelta(days=self.LONG_PERIOD + 50)
        return extended_start.strftime("%Y-%m-%d")

    async def _get_price_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """Get price data with caching."""
        cache_key = f"market_analysis:{symbol}:{start_date}:{end_date}"

        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)

        try:
            # Fetch from provider
            data = self.data_provider.get_stock_data(
                symbol=symbol, start_date=start_date, end_date=end_date, interval="1d"
            )

            if data is not None and not data.empty:
                # Cache for 30 minutes
                await self.cache.set(cache_key, data.to_dict(), ttl=1800)
                return data

            return None

        except Exception as e:
            logger.error(f"Failed to fetch price data for {symbol}: {e}")
            return None

    def _perform_regime_analysis(self, data: pd.DataFrame) -> dict[str, Any]:
        """Perform comprehensive market regime analysis."""
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]

        # Calculate technical indicators
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Trend analysis
        trend_analysis = self._analyze_trend(close)

        # Volatility analysis
        volatility_analysis = self._analyze_volatility(close)

        # Volume analysis
        volume_analysis = self._analyze_volume(volume, close)

        # Support/resistance analysis
        support_resistance = self._identify_support_resistance(high, low, close)

        # Market structure analysis
        market_structure = self._analyze_market_structure(high, low, close)

        # Determine overall regime
        regime_info = self._classify_regime(
            trend_analysis, volatility_analysis, volume_analysis, market_structure
        )

        return {
            "regime": regime_info["regime"],
            "confidence": regime_info["confidence"],
            "indicators": {
                "trend_slope": trend_analysis["slope"],
                "trend_r2": trend_analysis["r_squared"],
                "volatility_20d": volatility_analysis["volatility_20d"],
                "volume_ratio": volume_analysis["volume_ratio"],
                "rsi_14": trend_analysis["rsi"],
                "adx": trend_analysis["adx"],
            },
            "volatility_percentile": volatility_analysis["percentile"],
            "trend_strength": trend_analysis["strength"],
            "market_conditions": {
                "trend_direction": trend_analysis["direction"],
                "trend_consistency": trend_analysis["consistency"],
                "volatility_regime": volatility_analysis["regime"],
                "volume_regime": volume_analysis["regime"],
                "market_structure": market_structure["structure_type"],
            },
            "volume_profile": volume_analysis["profile"],
            "support_resistance": support_resistance,
        }

    def _analyze_trend(self, close: pd.Series) -> dict[str, Any]:
        """Analyze trend characteristics."""
        # Calculate moving averages
        ma_20 = ta.sma(close, length=self.SHORT_PERIOD)
        ma_50 = ta.sma(close, length=self.MEDIUM_PERIOD)
        ma_200 = ta.sma(close, length=self.LONG_PERIOD)

        # Calculate trend slope using linear regression
        recent_data = close.tail(self.MEDIUM_PERIOD).reset_index(drop=True)
        x = np.arange(len(recent_data))

        if len(recent_data) > 1:
            slope, intercept = np.polyfit(x, recent_data, 1)
            y_pred = slope * x + intercept
            r_squared = 1 - (
                np.sum((recent_data - y_pred) ** 2)
                / np.sum((recent_data - np.mean(recent_data)) ** 2)
            )
        else:
            slope = 0
            r_squared = 0

        # Normalize slope by price for comparability
        normalized_slope = slope / close.iloc[-1] if close.iloc[-1] != 0 else 0

        # Calculate RSI and ADX for trend strength
        rsi = ta.rsi(close, length=14).iloc[-1] if len(close) >= 14 else 50
        adx_result = ta.adx(
            close.to_frame().rename(columns={"close": "high"}),
            close.to_frame().rename(columns={"close": "low"}),
            close,
            length=14,
        )
        adx = (
            adx_result.iloc[-1, 0]
            if adx_result is not None and len(adx_result) > 0
            else 25
        )

        # Determine trend direction and strength
        if normalized_slope > 0.001:  # 0.1% daily trend
            direction = "bullish"
            strength = min(abs(normalized_slope) * 1000, 1.0)  # Cap at 1.0
        elif normalized_slope < -0.001:
            direction = "bearish"
            strength = min(abs(normalized_slope) * 1000, 1.0)
        else:
            direction = "sideways"
            strength = 0.2  # Low strength for sideways

        # Calculate trend consistency
        ma_alignment = 0
        if len(ma_20) > 0 and len(ma_50) > 0 and len(ma_200) > 0:
            current_price = close.iloc[-1]
            if ma_20.iloc[-1] > ma_50.iloc[-1] > ma_200.iloc[-1] > current_price * 0.95:
                ma_alignment = 1.0  # Bullish alignment
            elif (
                ma_20.iloc[-1] < ma_50.iloc[-1] < ma_200.iloc[-1] < current_price * 1.05
            ):
                ma_alignment = -1.0  # Bearish alignment
            else:
                ma_alignment = 0.0  # Mixed alignment

        consistency = (abs(ma_alignment) + r_squared) / 2

        return {
            "slope": normalized_slope,
            "r_squared": r_squared,
            "direction": direction,
            "strength": strength,
            "consistency": consistency,
            "rsi": rsi,
            "adx": adx,
        }

    def _analyze_volatility(self, close: pd.Series) -> dict[str, Any]:
        """Analyze volatility characteristics."""
        # Calculate various volatility measures
        returns = close.pct_change().dropna()

        volatility_5d = (
            returns.tail(5).std() * math.sqrt(252) if len(returns) >= 5 else 0
        )
        volatility_20d = (
            returns.tail(20).std() * math.sqrt(252) if len(returns) >= 20 else 0
        )
        volatility_60d = (
            returns.tail(60).std() * math.sqrt(252) if len(returns) >= 60 else 0
        )

        # Calculate historical volatility percentile
        historical_vol = returns.rolling(20).std() * math.sqrt(252)
        if len(historical_vol.dropna()) > 0:
            current_vol = historical_vol.iloc[-1]
            percentile = (historical_vol < current_vol).sum() / len(
                historical_vol.dropna()
            )
        else:
            percentile = 0.5

        # Classify volatility regime
        if volatility_20d > 0.4:  # > 40% annualized
            regime = "high_volatility"
        elif volatility_20d > 0.2:  # 20-40% annualized
            regime = "medium_volatility"
        else:
            regime = "low_volatility"

        return {
            "volatility_5d": volatility_5d,
            "volatility_20d": volatility_20d,
            "volatility_60d": volatility_60d,
            "percentile": percentile,
            "regime": regime,
        }

    def _analyze_volume(self, volume: pd.Series, close: pd.Series) -> dict[str, Any]:
        """Analyze volume characteristics."""
        # Calculate volume moving averages
        volume_ma_20 = volume.rolling(20).mean()

        # Current volume ratio vs average
        current_volume = volume.iloc[-1] if len(volume) > 0 else 0
        avg_volume = volume_ma_20.iloc[-1] if len(volume_ma_20.dropna()) > 0 else 1
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Volume trend
        recent_volume = volume.tail(10)
        volume_trend = (
            "increasing"
            if recent_volume.iloc[-1] > recent_volume.mean()
            else "decreasing"
        )

        # Price-volume relationship
        price_change = close.pct_change().tail(10)
        volume_change = volume.pct_change().tail(10)

        correlation = price_change.corr(volume_change) if len(price_change) >= 2 else 0

        # Volume regime classification
        if volume_ratio > 2.0:
            regime = "high_volume"
        elif volume_ratio > 1.5:
            regime = "elevated_volume"
        elif volume_ratio < 0.5:
            regime = "low_volume"
        else:
            regime = "normal_volume"

        return {
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "price_volume_correlation": correlation,
            "regime": regime,
            "profile": {
                "current_vs_20d": volume_ratio,
                "trend_direction": volume_trend,
                "price_correlation": correlation,
            },
        }

    def _identify_support_resistance(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> list[float]:
        """Identify key support and resistance levels."""
        levels = []

        try:
            # Recent price range
            recent_data = close.tail(50) if len(close) >= 50 else close
            price_range = recent_data.max() - recent_data.min()

            # Identify local peaks and troughs
            try:
                from scipy.signal import find_peaks

                # Find resistance levels (peaks)
                peaks, _ = find_peaks(
                    high.values, distance=5, prominence=price_range * 0.02
                )
                resistance_levels = high.iloc[peaks].tolist()

                # Find support levels (troughs)
                troughs, _ = find_peaks(
                    -low.values, distance=5, prominence=price_range * 0.02
                )
                support_levels = low.iloc[troughs].tolist()
            except ImportError:
                logger.warning("scipy not available, using simple peak detection")
                # Fallback to simple method
                resistance_levels = [recent_data.max()]
                support_levels = [recent_data.min()]

            # Combine and filter levels
            all_levels = resistance_levels + support_levels

            # Remove levels too close to each other
            filtered_levels = []
            for level in sorted(all_levels):
                if not any(
                    abs(level - existing) < price_range * 0.01
                    for existing in filtered_levels
                ):
                    filtered_levels.append(level)

            # Keep only most significant levels
            levels = sorted(filtered_levels)[-10:]  # Top 10 levels

        except Exception as e:
            logger.warning(f"Failed to calculate support/resistance levels: {e}")
            # Fallback to simple levels
            current_price = close.iloc[-1]
            levels = [
                current_price * 0.95,  # 5% below
                current_price * 1.05,  # 5% above
            ]

        return levels

    def _analyze_market_structure(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict[str, Any]:
        """Analyze market structure patterns."""
        try:
            # Calculate recent highs and lows
            lookback = min(20, len(close))
            recent_highs = high.tail(lookback)
            recent_lows = low.tail(lookback)

            # Identify higher highs, higher lows, etc.
            higher_highs = (recent_highs.rolling(3).max() == recent_highs).sum()
            higher_lows = (recent_lows.rolling(3).min() == recent_lows).sum()

            # Classify structure
            if higher_highs > lookback * 0.3 and higher_lows > lookback * 0.3:
                structure_type = "uptrend_structure"
            elif higher_highs < lookback * 0.1 and higher_lows < lookback * 0.1:
                structure_type = "downtrend_structure"
            else:
                structure_type = "ranging_structure"

            return {
                "structure_type": structure_type,
                "higher_highs": higher_highs,
                "higher_lows": higher_lows,
            }

        except Exception as e:
            logger.warning(f"Failed to analyze market structure: {e}")
            return {
                "structure_type": "unknown_structure",
                "higher_highs": 0,
                "higher_lows": 0,
            }

    def _classify_regime(
        self,
        trend_analysis: dict,
        volatility_analysis: dict,
        volume_analysis: dict,
        market_structure: dict,
    ) -> dict[str, Any]:
        """Classify overall market regime based on component analyses."""

        # Initialize scoring system
        regime_scores = {
            "trending": 0.0,
            "ranging": 0.0,
            "volatile": 0.0,
            "low_volume": 0.0,
        }

        # Trend scoring
        if trend_analysis["strength"] > 0.6 and trend_analysis["consistency"] > 0.6:
            regime_scores["trending"] += 0.4

        if trend_analysis["adx"] > 25:  # Strong trend
            regime_scores["trending"] += 0.2

        # Ranging scoring
        if (
            trend_analysis["strength"] < 0.3
            and trend_analysis["direction"] == "sideways"
        ):
            regime_scores["ranging"] += 0.4

        if market_structure["structure_type"] == "ranging_structure":
            regime_scores["ranging"] += 0.2

        # Volatility scoring
        if volatility_analysis["regime"] == "high_volatility":
            regime_scores["volatile"] += 0.3

        if volatility_analysis["percentile"] > 0.8:  # High volatility percentile
            regime_scores["volatile"] += 0.2

        # Volume scoring
        if volume_analysis["regime"] == "low_volume":
            regime_scores["low_volume"] += 0.3

        # Determine primary regime
        primary_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime_name = primary_regime[0]

        # Combine regimes for complex cases
        if regime_scores["volatile"] > 0.3 and regime_scores["trending"] > 0.3:
            regime_name = "volatile_trending"
        elif regime_scores["low_volume"] > 0.2 and regime_scores["ranging"] > 0.3:
            regime_name = "low_volume_ranging"

        # Calculate confidence based on score spread
        sorted_scores = sorted(regime_scores.values(), reverse=True)
        confidence = (
            sorted_scores[0] - sorted_scores[1]
            if len(sorted_scores) > 1
            else sorted_scores[0]
        )
        confidence = min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95

        return {
            "regime": regime_name,
            "confidence": confidence,
            "scores": regime_scores,
        }
