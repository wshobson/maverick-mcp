"""Tests for maverick_mcp/core/regime_gate.py — regime detection & filtering."""

import time
from unittest.mock import MagicMock, patch

import pandas as pd

import maverick_mcp.core.regime_gate as regime_gate_module
from maverick_mcp.core.regime_gate import apply_regime_filter, get_current_regime

# ---------------------------------------------------------------------------
# TestApplyRegimeFilter — pure logic, no mocks needed
# ---------------------------------------------------------------------------


class TestApplyRegimeFilter:
    def test_bull_returns_all_stocks(self):
        stocks = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        regime = {"label": "bull", "confidence": 0.9}
        filtered, ctx = apply_regime_filter(stocks, regime, "maverick_bullish")
        assert filtered == stocks
        assert ctx["action"] == "no_filter"

    def test_bear_maverick_bullish_suppressed(self):
        stocks = [{"ticker": "X"}, {"ticker": "Y"}]
        regime = {"label": "bear", "confidence": 0.85}
        filtered, ctx = apply_regime_filter(stocks, regime, "maverick_bullish")
        assert filtered == []
        assert ctx["action"] == "suppressed"

    def test_bear_supply_demand_returns_top5(self):
        stocks = [{"ticker": f"S{i}", "momentum_score": i} for i in range(10)]
        regime = {"label": "bear", "confidence": 0.8}
        filtered, ctx = apply_regime_filter(stocks, regime, "supply_demand_breakout")
        assert len(filtered) == 5
        # Should be sorted by momentum_score descending
        scores = [s["momentum_score"] for s in filtered]
        assert scores == [9, 8, 7, 6, 5]
        assert ctx["action"] == "filtered_top5"

    def test_sideways_reduces_to_half(self):
        stocks = [{"ticker": f"T{i}"} for i in range(20)]
        regime = {"label": "sideways", "confidence": 0.6}
        filtered, ctx = apply_regime_filter(stocks, regime, "maverick_bullish")
        assert len(filtered) == 10  # 20 // 2
        assert ctx["action"] == "reduced"

    def test_empty_stocks_list(self):
        regime = {"label": "bull", "confidence": 0.9}
        filtered, ctx = apply_regime_filter([], regime, "maverick_bullish")
        assert filtered == []


# ---------------------------------------------------------------------------
# TestGetCurrentRegime — requires mocking providers
# ---------------------------------------------------------------------------


class TestGetCurrentRegime:
    def setup_method(self):
        """Clear the module-level cache before every test."""
        regime_gate_module._regime_cache.clear()

    @patch(
        "maverick_mcp.backtesting.strategies.ml.regime_aware.MarketRegimeDetector",
    )
    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_cache_miss_fetches_and_caches(self, mock_prov_cls, mock_det_cls):
        """First call fetches data, detects regime, stores in cache."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider
        # Return a DataFrame with > 100 rows
        provider.get_stock_data.return_value = pd.DataFrame({"close": list(range(200))})

        detector = MagicMock()
        mock_det_cls.return_value = detector
        detector.predict_regimes.return_value = [2] * 20  # bull

        result = get_current_regime("SPY", "hmm")

        assert result["label"] == "bull"
        assert result["cached"] is False
        assert "SPY:hmm" in regime_gate_module._regime_cache

    def test_cache_hit_skips_provider(self):
        """Pre-populated fresh cache returns without calling provider."""
        regime_gate_module._regime_cache["SPY:hmm"] = {
            "result": {
                "label": "bear",
                "regime_id": 0,
                "confidence": 0.75,
                "method": "hmm",
            },
            "timestamp": time.time(),  # fresh
        }

        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider"
        ) as mock_prov_cls:
            result = get_current_regime("SPY", "hmm")
            mock_prov_cls.assert_not_called()

        assert result["label"] == "bear"
        assert result["cached"] is True

    @patch(
        "maverick_mcp.backtesting.strategies.ml.regime_aware.MarketRegimeDetector",
    )
    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_insufficient_data_returns_sideways(self, mock_prov_cls, mock_det_cls):
        """When provider returns fewer than 100 rows, sideways default."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider
        provider.get_stock_data.return_value = pd.DataFrame({"close": list(range(50))})

        result = get_current_regime("SPY", "hmm")
        assert result["label"] == "sideways"
        assert "error" in result

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_exception_returns_sideways_fallback(self, mock_prov_cls):
        """Provider exception yields sideways fallback with error key."""
        mock_prov_cls.return_value.get_stock_data.side_effect = RuntimeError("API down")
        result = get_current_regime("SPY", "hmm")
        assert result["label"] == "sideways"
        assert result["confidence"] == 0.0
        assert "error" in result
