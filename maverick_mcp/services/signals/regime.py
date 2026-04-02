"""Market regime detector using composite multi-factor scoring.

The RegimeDetector classifies the current market environment into one of
four regimes — ``bull``, ``bear``, ``choppy``, or ``transitional`` — by
combining four weighted factors: trend, volatility, momentum, and breadth.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

try:
    import pandas_ta as ta  # noqa: F401
except ImportError:
    ta = None  # pandas_ta requires numba (Python <3.14)


def _require_ta():
    """Raise ImportError if pandas_ta is not available."""
    if ta is None:
        raise ImportError(
            "pandas_ta is required for this feature. "
            "Install it with: pip install pandas_ta (requires Python <3.14)"
        )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_WEIGHT_TREND = 0.35
_WEIGHT_VOLATILITY = 0.25
_WEIGHT_MOMENTUM = 0.25
_WEIGHT_BREADTH = 0.15

# Confidence threshold below which regime is labelled "transitional"
_TRANSITIONAL_THRESHOLD = 0.45


class RegimeDetector:
    """Classify market regime from price series and macro signals.

    Attributes:
        short_window: Period for the short-term SMA (default 20).
        long_window: Period for the long-term SMA (default 50).
        momentum_window: Look-back for rate-of-change (default 10).
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        momentum_window: int = 10,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_window = momentum_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(
        self,
        market_prices: pd.Series,
        vix_level: float,
        breadth_ratio: float | None = None,
    ) -> dict[str, Any]:
        """Classify market regime.

        Args:
            market_prices: Series of market close prices (most recent last).
            vix_level: Current VIX level.
            breadth_ratio: Optional ratio of advancing to total issues (0–1).
                           If None, a neutral breadth score is assumed.

        Returns:
            Dict with keys:
            - ``regime`` (str): one of bull / bear / choppy / transitional
            - ``confidence`` (float): 0–1
            - ``drivers`` (dict): per-factor scores
            - ``votes`` (dict): raw vote per factor (bull=+1, bear=-1, neutral=0)
        """
        trend_score, trend_vote = self._score_trend(market_prices)
        vol_score, vol_vote = self._score_volatility(vix_level)
        mom_score, mom_vote = self._score_momentum(market_prices)
        breadth_score, breadth_vote = self._score_breadth(breadth_ratio)

        composite = (
            _WEIGHT_TREND * trend_score
            + _WEIGHT_VOLATILITY * vol_score
            + _WEIGHT_MOMENTUM * mom_score
            + _WEIGHT_BREADTH * breadth_score
        )
        # composite is in [-1, +1]; confidence is the normalised absolute value
        confidence = min(abs(composite), 1.0)

        if confidence < _TRANSITIONAL_THRESHOLD:
            regime = "transitional"
        elif composite > 0:
            # Positive composite → bullish or choppy
            # If trend score is weak but composite is positive → choppy
            if trend_score < 0.15 and vol_score < 0.1:
                regime = "choppy"
            else:
                regime = "bull"
        else:
            regime = "bear"

        # Special case: very low volatility signal with no clear direction → choppy
        if regime != "bear" and trend_score < 0.05 and vol_score < 0.05:
            regime = "choppy"

        return {
            "regime": regime,
            "confidence": round(confidence, 4),
            "drivers": {
                "trend": round(trend_score, 4),
                "volatility": round(vol_score, 4),
                "momentum": round(mom_score, 4),
                "breadth": round(breadth_score, 4),
            },
            "votes": {
                "trend": trend_vote,
                "volatility": vol_vote,
                "momentum": mom_vote,
                "breadth": breadth_vote,
            },
        }

    # ------------------------------------------------------------------
    # Private scoring helpers  (return (score, vote))
    # score is in [-1, +1] where +1 = strongly bullish, -1 = strongly bearish
    # vote is a label string for human readability
    # ------------------------------------------------------------------

    def _score_trend(self, prices: pd.Series) -> tuple[float, str]:
        """Score based on price vs short/long SMA and slope."""
        if len(prices) < self.long_window + 1:
            return 0.0, "neutral"

        short_sma = float(prices.iloc[-self.short_window :].mean())
        long_sma = float(prices.iloc[-self.long_window :].mean())
        current = float(prices.iloc[-1])

        # Distance of current price relative to long SMA (normalised)
        pct_vs_long = (current - long_sma) / long_sma if long_sma != 0 else 0.0
        sma_cross = 1.0 if short_sma > long_sma else -1.0

        # Slope: % change over last short_window bars
        start_price = float(prices.iloc[-(self.short_window + 1)])
        slope = (current - start_price) / start_price if start_price != 0 else 0.0

        # Combine: SMA relationship weighted heavier, plus slope confirmation
        raw = 0.5 * sma_cross + 0.3 * _clip(pct_vs_long * 10) + 0.2 * _clip(slope * 20)
        return _clip(raw), "bull" if raw > 0 else "bear"

    def _score_volatility(self, vix: float) -> tuple[float, str]:
        """Score based on VIX level.  High VIX → bearish, low VIX → bullish."""
        if vix < 16:
            return 0.8, "bull"
        if vix < 22:
            return 0.0, "neutral"
        if vix < 30:
            return -0.6, "bear"
        return -1.0, "bear"  # strong bear

    def _score_momentum(self, prices: pd.Series) -> tuple[float, str]:
        """Rate of change over the momentum window."""
        if len(prices) < self.momentum_window + 1:
            return 0.0, "neutral"
        start = float(prices.iloc[-(self.momentum_window + 1)])
        end = float(prices.iloc[-1])
        roc = (end - start) / start if start != 0 else 0.0
        # Normalise: +/-10% move maps to +/-1.0
        score = _clip(roc * 10)
        return score, "bull" if score > 0 else ("bear" if score < 0 else "neutral")

    def _score_breadth(self, breadth_ratio: float | None) -> tuple[float, str]:
        """Score from advance/decline breadth ratio (0–1 scale)."""
        if breadth_ratio is None:
            return 0.0, "neutral"  # default when not provided
        # > 0.6 → bullish, < 0.4 → bearish, 0.4-0.6 → neutral
        if breadth_ratio > 0.6:
            score = _clip((breadth_ratio - 0.5) * 5)
            return score, "bull"
        if breadth_ratio < 0.4:
            score = _clip((breadth_ratio - 0.5) * 5)
            return score, "bear"
        return 0.0, "neutral"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, float(value)))
