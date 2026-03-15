"""Market regime detection gate for screener filtering.

Thin wrapper around MarketRegimeDetector that provides cached regime
detection for SPY, used by screeners to filter signals based on market state.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# In-memory cache for regime detection results
_regime_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours


def get_current_regime(symbol: str = "SPY", method: str = "hmm") -> dict[str, Any]:
    """Detect the current market regime for a symbol.

    Uses MarketRegimeDetector with caching to avoid repeated computation.

    Args:
        symbol: Symbol to analyze (default: SPY for broad market)
        method: Detection method ('hmm', 'kmeans', 'threshold')

    Returns:
        Dict with keys: label ('bull'|'bear'|'sideways'), regime_id (int),
        confidence (float 0-1), method (str), cached (bool)
    """
    cache_key = f"{symbol}:{method}"

    # Check cache
    if cache_key in _regime_cache:
        cached = _regime_cache[cache_key]
        if time.time() - cached["timestamp"] < _CACHE_TTL_SECONDS:
            return {**cached["result"], "cached": True}

    try:
        import asyncio
        from datetime import datetime, timedelta

        from maverick_mcp.backtesting.strategies.ml.regime_aware import (
            MarketRegimeDetector,
        )
        from maverick_mcp.backtesting.vectorbt_engine import VectorBTEngine

        engine = VectorBTEngine()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Run async data fetch synchronously if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                data = loop.run_in_executor(
                    pool,
                    lambda: asyncio.run(
                        engine.get_historical_data(symbol, start_date, end_date)
                    ),
                )
                # This won't work in async context directly — use sync fallback
                from maverick_mcp.providers.stock_data import StockDataProvider

                provider = StockDataProvider()
                data = provider.get_stock_data(symbol, start_date, end_date)
        else:
            from maverick_mcp.providers.stock_data import StockDataProvider

            provider = StockDataProvider()
            data = provider.get_stock_data(symbol, start_date, end_date)

        if data.empty or len(data) < 100:
            return {
                "label": "sideways",
                "regime_id": 1,
                "confidence": 0.0,
                "method": method,
                "cached": False,
                "error": "Insufficient data for regime detection",
            }

        detector = MarketRegimeDetector(method=method, n_regimes=3, lookback_period=50)
        detector.fit_regimes(data)

        # Get current regime
        regimes = detector.predict_regimes(data)
        current_regime_id = int(regimes[-1]) if len(regimes) > 0 else 1

        # Map regime IDs to labels
        # Convention from backtesting.py: 0=Bear, 1=Sideways, 2=Bull
        regime_labels = {0: "bear", 1: "sideways", 2: "bull"}
        label = regime_labels.get(current_regime_id, "sideways")

        # Calculate confidence from recent regime stability
        recent_window = min(20, len(regimes))
        recent_regimes = regimes[-recent_window:]
        confidence = sum(1 for r in recent_regimes if r == current_regime_id) / len(
            recent_regimes
        )

        result = {
            "label": label,
            "regime_id": current_regime_id,
            "confidence": round(confidence, 3),
            "method": method,
            "cached": False,
        }

        # Cache the result
        _regime_cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
        }

        return result

    except Exception as e:
        logger.warning(f"Regime detection failed for {symbol}: {e}")
        return {
            "label": "sideways",
            "regime_id": 1,
            "confidence": 0.0,
            "method": method,
            "cached": False,
            "error": str(e),
        }


def apply_regime_filter(
    stocks: list[dict], regime: dict[str, Any], screening_type: str
) -> tuple[list[dict], dict[str, Any]]:
    """Filter screener results based on current market regime.

    Args:
        stocks: List of stock dicts from screener
        regime: Result from get_current_regime()
        screening_type: 'maverick_bullish' or 'supply_demand_breakout'

    Returns:
        Tuple of (filtered_stocks, regime_context_dict)
    """
    label = regime.get("label", "sideways")
    confidence = regime.get("confidence", 0.0)

    regime_context = {
        "regime": label,
        "confidence": confidence,
        "filter_applied": True,
    }

    if label == "bull":
        # Bull market: return all results
        regime_context["action"] = "no_filter"
        regime_context["message"] = "Bull regime detected — all signals returned"
        return stocks, regime_context

    elif label == "bear":
        if screening_type == "maverick_bullish":
            # Bear market: suppress bullish signals
            regime_context["action"] = "suppressed"
            regime_context["message"] = (
                "Bear regime detected — bullish signals suppressed. "
                "Consider bearish screener instead."
            )
            return [], regime_context
        else:
            # Supply/demand breakouts in bear: only strongest signals
            filtered = sorted(
                stocks,
                key=lambda s: s.get("momentum_score", 0),
                reverse=True,
            )[:5]
            regime_context["action"] = "filtered_top5"
            regime_context["message"] = (
                "Bear regime detected — only top 5 highest momentum breakouts shown"
            )
            return filtered, regime_context

    else:
        # Sideways: reduce results, add caution
        limit = max(len(stocks) // 2, 5)
        filtered = stocks[:limit]
        regime_context["action"] = "reduced"
        regime_context["message"] = (
            f"Sideways regime detected — results reduced to top {limit}"
        )
        return filtered, regime_context
