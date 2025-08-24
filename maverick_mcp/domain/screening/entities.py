"""
Screening domain entities.

This module contains the core business entities for stock screening,
with embedded business rules and validation logic.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass
class ScreeningResult:
    """
    Domain entity representing a stock screening result.

    This entity encapsulates all business rules related to screening results,
    including validation, scoring, and ranking logic.
    """

    # Core identification
    stock_symbol: str
    screening_date: datetime

    # Price data
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int

    # Technical indicators
    ema_21: Decimal
    sma_50: Decimal
    sma_150: Decimal
    sma_200: Decimal
    momentum_score: Decimal
    avg_volume_30d: Decimal
    adr_percentage: Decimal
    atr: Decimal

    # Pattern analysis
    pattern: str | None = None
    squeeze: str | None = None
    consolidation: str | None = None
    entry_signal: str | None = None

    # Screening-specific scores
    combined_score: int = 0
    bear_score: int = 0
    compression_score: int = 0
    pattern_detected: int = 0

    # Additional bearish indicators
    rsi_14: Decimal | None = None
    macd: Decimal | None = None
    macd_signal: Decimal | None = None
    macd_histogram: Decimal | None = None
    distribution_days_20: int | None = None
    atr_contraction: bool | None = None
    big_down_volume: bool | None = None

    def __post_init__(self):
        """Validate business rules after initialization."""
        self._validate_stock_symbol()
        self._validate_price_data()
        self._validate_technical_indicators()

    def _validate_stock_symbol(self) -> None:
        """Validate stock symbol format."""
        if not self.stock_symbol or not isinstance(self.stock_symbol, str):
            raise ValueError("Stock symbol must be a non-empty string")

        if len(self.stock_symbol) > 10:
            raise ValueError("Stock symbol cannot exceed 10 characters")

    def _validate_price_data(self) -> None:
        """Validate price data consistency."""
        if self.close_price <= 0:
            raise ValueError("Close price must be positive")

        if self.volume < 0:
            raise ValueError("Volume cannot be negative")

        if self.high_price < self.low_price:
            raise ValueError("High price cannot be less than low price")

        if not (self.low_price <= self.close_price <= self.high_price):
            raise ValueError("Close price must be between low and high prices")

    def _validate_technical_indicators(self) -> None:
        """Validate technical indicator ranges."""
        if not (0 <= self.momentum_score <= 100):
            raise ValueError("Momentum score must be between 0 and 100")

        if self.adr_percentage < 0:
            raise ValueError("ADR percentage cannot be negative")

        if self.avg_volume_30d < 0:
            raise ValueError("Average volume cannot be negative")

    def is_bullish_setup(self) -> bool:
        """
        Determine if this is a bullish screening setup.

        Business rule: A stock is considered bullish if it meets
        momentum and trend criteria.
        """
        return (
            self.close_price > self.sma_50
            and self.close_price > self.sma_150
            and self.momentum_score >= 70
            and self.combined_score >= 50
        )

    def is_bearish_setup(self) -> bool:
        """
        Determine if this is a bearish screening setup.

        Business rule: A stock is considered bearish if it shows
        weakness and distribution characteristics.
        """
        return (
            self.close_price < self.sma_50
            and self.momentum_score <= 30
            and self.bear_score >= 50
        )

    def is_trending_stage2(self) -> bool:
        """
        Determine if this meets trending criteria.

        Business rule: Trending requires proper moving average alignment
        and strong relative strength.
        """
        return (
            self.close_price > self.sma_50
            and self.close_price > self.sma_150
            and self.close_price > self.sma_200
            and self.sma_50 > self.sma_150
            and self.sma_150 > self.sma_200
            and self.momentum_score >= 80
        )

    def meets_volume_criteria(self, min_volume: int) -> bool:
        """Check if stock meets minimum volume requirements."""
        return self.avg_volume_30d >= min_volume

    def meets_price_criteria(self, min_price: Decimal, max_price: Decimal) -> bool:
        """Check if stock meets price range criteria."""
        return min_price <= self.close_price <= max_price

    def calculate_risk_reward_ratio(
        self, stop_loss_percentage: Decimal = Decimal("0.08")
    ) -> Decimal:
        """
        Calculate risk/reward ratio based on current price and stop loss.

        Business rule: Risk is calculated as the distance to stop loss,
        reward is calculated as the potential upside to resistance levels.
        """
        stop_loss_price = self.close_price * (1 - stop_loss_percentage)
        risk = self.close_price - stop_loss_price

        # Simple reward calculation based on ADR
        potential_reward = self.close_price * (self.adr_percentage / 100)

        if risk <= 0:
            return Decimal("0")

        return potential_reward / risk

    def get_quality_score(self) -> int:
        """
        Calculate overall quality score based on multiple factors.

        Business rule: Quality score combines technical strength,
        volume characteristics, and pattern recognition.
        """
        score = 0

        # Momentum Score contribution (0-40 points)
        score += int(self.momentum_score * 0.4)

        # Volume quality (0-20 points)
        if self.avg_volume_30d >= 1_000_000:
            score += 20
        elif self.avg_volume_30d >= 500_000:
            score += 15
        elif self.avg_volume_30d >= 100_000:
            score += 10

        # Pattern recognition (0-20 points)
        if self.pattern_detected > 0:
            score += 20

        # Price action (0-20 points)
        if self.close_price > self.sma_50:
            score += 10
        if self.close_price > self.sma_200:
            score += 10

        return min(score, 100)  # Cap at 100

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "stock_symbol": self.stock_symbol,
            "screening_date": self.screening_date.isoformat(),
            "close_price": float(self.close_price),
            "volume": self.volume,
            "momentum_score": float(self.momentum_score),
            "adr_percentage": float(self.adr_percentage),
            "pattern": self.pattern,
            "squeeze": self.squeeze,
            "vcp": self.vcp,
            "entry_signal": self.entry_signal,
            "combined_score": self.combined_score,
            "bear_score": self.bear_score,
            "quality_score": self.get_quality_score(),
            "is_bullish": self.is_bullish_setup(),
            "is_bearish": self.is_bearish_setup(),
            "is_trending_stage2": self.is_trending_stage2(),
            "risk_reward_ratio": float(self.calculate_risk_reward_ratio()),
        }


@dataclass
class ScreeningResultCollection:
    """
    Domain entity representing a collection of screening results.

    This aggregate root manages business rules that apply across
    multiple screening results, such as ranking and filtering.
    """

    results: list[ScreeningResult]
    strategy_used: str
    screening_timestamp: datetime
    total_candidates_analyzed: int

    def __post_init__(self):
        """Validate collection business rules."""
        if self.total_candidates_analyzed < len(self.results):
            raise ValueError("Total candidates cannot be less than results count")

    def get_top_ranked(self, limit: int) -> list[ScreeningResult]:
        """
        Get top-ranked results based on screening strategy.

        Business rule: Ranking depends on the screening strategy used.
        """
        if self.strategy_used == "maverick_bullish":
            return sorted(self.results, key=lambda r: r.combined_score, reverse=True)[
                :limit
            ]
        elif self.strategy_used == "maverick_bearish":
            return sorted(self.results, key=lambda r: r.bear_score, reverse=True)[
                :limit
            ]
        elif self.strategy_used == "trending_stage2":
            return sorted(self.results, key=lambda r: r.momentum_score, reverse=True)[
                :limit
            ]
        else:
            # Default to quality score
            return sorted(
                self.results, key=lambda r: r.get_quality_score(), reverse=True
            )[:limit]

    def filter_by_criteria(
        self,
        min_momentum_score: Decimal | None = None,
        min_volume: int | None = None,
        max_price: Decimal | None = None,
        min_price: Decimal | None = None,
    ) -> list[ScreeningResult]:
        """
        Filter results by business criteria.

        Business rule: All filters must be satisfied simultaneously.
        """
        filtered_results = self.results

        if min_momentum_score is not None:
            filtered_results = [
                r for r in filtered_results if r.momentum_score >= min_momentum_score
            ]

        if min_volume is not None:
            filtered_results = [
                r for r in filtered_results if r.avg_volume_30d >= min_volume
            ]

        if max_price is not None:
            filtered_results = [
                r for r in filtered_results if r.close_price <= max_price
            ]

        if min_price is not None:
            filtered_results = [
                r for r in filtered_results if r.close_price >= min_price
            ]

        return filtered_results

    def get_statistics(self) -> dict[str, Any]:
        """Get collection statistics for analysis."""
        if not self.results:
            return {
                "total_results": 0,
                "avg_momentum_score": 0,
                "avg_volume": 0,
                "avg_price": 0,
                "bullish_setups": 0,
                "bearish_setups": 0,
                "trending_stage2": 0,
            }

        return {
            "total_results": len(self.results),
            "avg_momentum_score": float(
                sum(r.momentum_score for r in self.results) / len(self.results)
            ),
            "avg_volume": int(
                sum(r.avg_volume_30d for r in self.results) / len(self.results)
            ),
            "avg_price": float(
                sum(r.close_price for r in self.results) / len(self.results)
            ),
            "bullish_setups": sum(1 for r in self.results if r.is_bullish_setup()),
            "bearish_setups": sum(1 for r in self.results if r.is_bearish_setup()),
            "trending_stage2": sum(1 for r in self.results if r.is_trending_stage2()),
        }
