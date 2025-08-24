"""
Application query for getting technical analysis.

This query orchestrates the domain services and repositories
to provide technical analysis functionality.
"""

from datetime import UTC, datetime, timedelta
from typing import Protocol

import pandas as pd

from maverick_mcp.application.dto.technical_analysis_dto import (
    BollingerBandsDTO,
    CompleteTechnicalAnalysisDTO,
    MACDAnalysisDTO,
    PriceLevelDTO,
    RSIAnalysisDTO,
    StochasticDTO,
    TrendAnalysisDTO,
    VolumeAnalysisDTO,
)
from maverick_mcp.domain.entities.stock_analysis import StockAnalysis
from maverick_mcp.domain.services.technical_analysis_service import (
    TechnicalAnalysisService,
)
from maverick_mcp.domain.value_objects.technical_indicators import (
    Signal,
)


class StockDataRepository(Protocol):
    """Protocol for stock data repository."""

    def get_price_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get historical price data."""
        ...


class GetTechnicalAnalysisQuery:
    """
    Application query for retrieving technical analysis.

    This query coordinates between the domain layer and infrastructure
    to provide technical analysis without exposing domain complexity.
    """

    def __init__(
        self,
        stock_repository: StockDataRepository,
        technical_service: TechnicalAnalysisService,
    ):
        """
        Initialize the query handler.

        Args:
            stock_repository: Repository for fetching stock data
            technical_service: Domain service for technical calculations
        """
        self.stock_repository = stock_repository
        self.technical_service = technical_service

    async def execute(
        self,
        symbol: str,
        days: int = 365,
        indicators: list[str] | None = None,
        rsi_period: int = 14,
    ) -> CompleteTechnicalAnalysisDTO:
        """
        Execute the technical analysis query.

        Args:
            symbol: Stock ticker symbol
            days: Number of days of historical data
            indicators: Specific indicators to calculate (None = all)
            rsi_period: Period for RSI calculation (default: 14)

        Returns:
            Complete technical analysis DTO
        """
        # Calculate date range
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        # Fetch stock data from repository
        df = self.stock_repository.get_price_data(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Create domain entity
        analysis = StockAnalysis(
            symbol=symbol,
            analysis_date=datetime.now(UTC),
            current_price=float(df["close"].iloc[-1]),
            trend_direction=self.technical_service.identify_trend(
                pd.Series(df["close"])
            ),
            trend_strength=self._calculate_trend_strength(df),
            analysis_period_days=days,
            indicators_used=[],  # Initialize indicators_used
        )

        # Calculate requested indicators
        # Since we initialized indicators_used as [], it's safe to use
        assert analysis.indicators_used is not None

        if not indicators or "rsi" in indicators:
            analysis.rsi = self.technical_service.calculate_rsi(
                pd.Series(df["close"]), period=rsi_period
            )
            analysis.indicators_used.append("RSI")

        if not indicators or "macd" in indicators:
            analysis.macd = self.technical_service.calculate_macd(
                pd.Series(df["close"])
            )
            analysis.indicators_used.append("MACD")

        if not indicators or "bollinger" in indicators:
            analysis.bollinger_bands = self.technical_service.calculate_bollinger_bands(
                pd.Series(df["close"])
            )
            analysis.indicators_used.append("Bollinger Bands")

        if not indicators or "stochastic" in indicators:
            analysis.stochastic = self.technical_service.calculate_stochastic(
                pd.Series(df["high"]), pd.Series(df["low"]), pd.Series(df["close"])
            )
            analysis.indicators_used.append("Stochastic")

        # Analyze volume
        if "volume" in df.columns:
            analysis.volume_profile = self.technical_service.analyze_volume(
                pd.Series(df["volume"])
            )

        # Calculate support and resistance levels
        analysis.support_levels = self.technical_service.find_support_levels(df)
        analysis.resistance_levels = self.technical_service.find_resistance_levels(df)

        # Calculate composite signal
        analysis.composite_signal = self.technical_service.calculate_composite_signal(
            analysis.rsi,
            analysis.macd,
            analysis.bollinger_bands,
            analysis.stochastic,
        )

        # Calculate confidence score
        analysis.confidence_score = self._calculate_confidence_score(analysis)

        # Convert to DTO
        return self._map_to_dto(analysis)

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength as a percentage."""
        # Simple implementation using price change
        if len(df) < 20:
            return 0.0

        price_change = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df[
            "close"
        ].iloc[-20]
        return float(min(abs(price_change) * 100, 100.0))

    def _calculate_confidence_score(self, analysis: StockAnalysis) -> float:
        """Calculate confidence score based on indicator agreement."""
        signals = []

        if analysis.rsi:
            signals.append(analysis.rsi.signal)
        if analysis.macd:
            signals.append(analysis.macd.signal)
        if analysis.bollinger_bands:
            signals.append(analysis.bollinger_bands.signal)
        if analysis.stochastic:
            signals.append(analysis.stochastic.signal)

        if not signals:
            return 0.0

        # Count agreeing signals
        signal_counts: dict[Signal, int] = {}
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        max_agreement = max(signal_counts.values())
        confidence = (max_agreement / len(signals)) * 100

        # Boost confidence if volume confirms
        if analysis.volume_profile and analysis.volume_profile.unusual_activity:
            confidence = min(100, confidence + 10)

        return float(confidence)

    def _map_to_dto(self, analysis: StockAnalysis) -> CompleteTechnicalAnalysisDTO:
        """Map domain entity to DTO."""
        dto = CompleteTechnicalAnalysisDTO(
            symbol=analysis.symbol,
            analysis_date=analysis.analysis_date,
            current_price=analysis.current_price,
            trend=TrendAnalysisDTO(
                direction=analysis.trend_direction.value,
                strength=analysis.trend_strength,
                interpretation=self._interpret_trend(analysis),
            ),
            composite_signal=analysis.composite_signal.value,
            confidence_score=analysis.confidence_score,
            risk_reward_ratio=analysis.risk_reward_ratio,
            summary=self._generate_summary(analysis),
            key_levels=analysis.get_key_levels(),
            rsi=None,
            macd=None,
            bollinger_bands=None,
            stochastic=None,
            volume_analysis=None,
        )

        # Map indicators if present
        if analysis.rsi:
            dto.rsi = RSIAnalysisDTO(
                current_value=analysis.rsi.value,
                period=analysis.rsi.period,
                signal=analysis.rsi.signal.value,
                is_overbought=analysis.rsi.is_overbought,
                is_oversold=analysis.rsi.is_oversold,
                interpretation=self._interpret_rsi(analysis.rsi),
            )

        if analysis.macd:
            dto.macd = MACDAnalysisDTO(
                macd_line=analysis.macd.macd_line,
                signal_line=analysis.macd.signal_line,
                histogram=analysis.macd.histogram,
                signal=analysis.macd.signal.value,
                is_bullish_crossover=analysis.macd.is_bullish_crossover,
                is_bearish_crossover=analysis.macd.is_bearish_crossover,
                interpretation=self._interpret_macd(analysis.macd),
            )

        if analysis.bollinger_bands:
            dto.bollinger_bands = BollingerBandsDTO(
                upper_band=analysis.bollinger_bands.upper_band,
                middle_band=analysis.bollinger_bands.middle_band,
                lower_band=analysis.bollinger_bands.lower_band,
                current_price=analysis.bollinger_bands.current_price,
                bandwidth=analysis.bollinger_bands.bandwidth,
                percent_b=analysis.bollinger_bands.percent_b,
                signal=analysis.bollinger_bands.signal.value,
                interpretation=self._interpret_bollinger(analysis.bollinger_bands),
            )

        if analysis.stochastic:
            dto.stochastic = StochasticDTO(
                k_value=analysis.stochastic.k_value,
                d_value=analysis.stochastic.d_value,
                signal=analysis.stochastic.signal.value,
                is_overbought=analysis.stochastic.is_overbought,
                is_oversold=analysis.stochastic.is_oversold,
                interpretation=self._interpret_stochastic(analysis.stochastic),
            )

        # Map levels
        dto.support_levels = [
            PriceLevelDTO(
                price=level.price,
                strength=level.strength,
                touches=level.touches,
                distance_from_current=(
                    (analysis.current_price - level.price)
                    / analysis.current_price
                    * 100
                ),
            )
            for level in (analysis.support_levels or [])
        ]

        dto.resistance_levels = [
            PriceLevelDTO(
                price=level.price,
                strength=level.strength,
                touches=level.touches,
                distance_from_current=(
                    (level.price - analysis.current_price)
                    / analysis.current_price
                    * 100
                ),
            )
            for level in (analysis.resistance_levels or [])
        ]

        # Map volume if present
        if analysis.volume_profile:
            dto.volume_analysis = VolumeAnalysisDTO(
                current_volume=analysis.volume_profile.current_volume,
                average_volume=analysis.volume_profile.average_volume,
                relative_volume=analysis.volume_profile.relative_volume,
                volume_trend=analysis.volume_profile.volume_trend.value,
                unusual_activity=analysis.volume_profile.unusual_activity,
                interpretation=self._interpret_volume(analysis.volume_profile),
            )

        return dto

    def _generate_summary(self, analysis: StockAnalysis) -> str:
        """Generate executive summary of the analysis."""
        signal_text = {
            Signal.STRONG_BUY: "strong buy signal",
            Signal.BUY: "buy signal",
            Signal.NEUTRAL: "neutral stance",
            Signal.SELL: "sell signal",
            Signal.STRONG_SELL: "strong sell signal",
        }

        summary_parts = [
            f"{analysis.symbol} shows a {signal_text[analysis.composite_signal]}",
            f"with {analysis.confidence_score:.0f}% confidence.",
            f"The stock is in a {analysis.trend_direction.value.replace('_', ' ')}.",
        ]

        if analysis.risk_reward_ratio:
            summary_parts.append(
                f"Risk/reward ratio is {analysis.risk_reward_ratio:.2f}."
            )

        return " ".join(summary_parts)

    def _interpret_trend(self, analysis: StockAnalysis) -> str:
        """Generate trend interpretation."""
        return (
            f"The stock is showing a {analysis.trend_direction.value.replace('_', ' ')} "
            f"with {analysis.trend_strength:.0f}% strength."
        )

    def _interpret_rsi(self, rsi) -> str:
        """Generate RSI interpretation."""
        if rsi.is_overbought:
            return f"RSI at {rsi.value:.1f} indicates overbought conditions."
        elif rsi.is_oversold:
            return f"RSI at {rsi.value:.1f} indicates oversold conditions."
        else:
            return f"RSI at {rsi.value:.1f} is in neutral territory."

    def _interpret_macd(self, macd) -> str:
        """Generate MACD interpretation."""
        if macd.is_bullish_crossover:
            return "MACD shows bullish crossover - potential buy signal."
        elif macd.is_bearish_crossover:
            return "MACD shows bearish crossover - potential sell signal."
        else:
            return "MACD is neutral, no clear signal."

    def _interpret_bollinger(self, bb) -> str:
        """Generate Bollinger Bands interpretation."""
        if bb.is_squeeze:
            return "Bollinger Bands are squeezing - expect volatility breakout."
        elif bb.percent_b > 1:
            return "Price above upper band - potential overbought."
        elif bb.percent_b < 0:
            return "Price below lower band - potential oversold."
        else:
            return f"Price at {bb.percent_b:.1%} of bands range."

    def _interpret_stochastic(self, stoch) -> str:
        """Generate Stochastic interpretation."""
        if stoch.is_overbought:
            return f"Stochastic at {stoch.k_value:.1f} indicates overbought."
        elif stoch.is_oversold:
            return f"Stochastic at {stoch.k_value:.1f} indicates oversold."
        else:
            return f"Stochastic at {stoch.k_value:.1f} is neutral."

    def _interpret_volume(self, volume) -> str:
        """Generate volume interpretation."""
        if volume.unusual_activity:
            return f"Unusual volume at {volume.relative_volume:.1f}x average!"
        elif volume.is_high_volume:
            return f"High volume at {volume.relative_volume:.1f}x average."
        elif volume.is_low_volume:
            return f"Low volume at {volume.relative_volume:.1f}x average."
        else:
            return "Normal trading volume."
