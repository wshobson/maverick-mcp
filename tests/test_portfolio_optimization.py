"""Tests for maverick_mcp/core/portfolio_optimization.py — HRP optimization."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from maverick_mcp.core.portfolio_optimization import optimize_hrp


class TestOptimizeHrp:
    def test_fewer_than_2_symbols(self):
        """Single symbol returns error dict."""
        result = optimize_hrp(["AAPL"])
        assert result["status"] == "error"
        assert "2 symbols" in result["error"]

    @patch("builtins.__import__")
    def test_riskfolio_not_installed(self, mock_import):
        """ImportError for riskfolio is caught and reported."""
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def selective_import(name, *args, **kwargs):
            if name == "riskfolio":
                raise ImportError("No module named 'riskfolio'")
            return original_import(name, *args, **kwargs)

        mock_import.side_effect = selective_import

        result = optimize_hrp(["AAPL", "MSFT"])
        assert result["status"] == "error"
        assert "riskfolio" in result["error"].lower()

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_success(self, mock_prov_cls):
        """Full successful HRP path with mocked riskfolio."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider

        np.random.seed(42)
        n = 300
        dates = pd.bdate_range("2024-01-01", periods=n)

        def make_df(symbol, *_a, **_kw):
            prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
            return pd.DataFrame({"close": prices}, index=dates)

        provider.get_stock_data.side_effect = make_df

        # Mock riskfolio inside the function scope
        mock_rp = MagicMock()
        weights_df = pd.DataFrame({"weights": [0.6, 0.4]}, index=["AAPL", "MSFT"])
        mock_port = MagicMock()
        mock_port.optimization.return_value = weights_df
        mock_rp.HCPortfolio.return_value = mock_port

        with patch.dict("sys.modules", {"riskfolio": mock_rp}):
            result = optimize_hrp(["AAPL", "MSFT"], days=252)

        assert result["status"] == "success"
        assert "weights" in result
        assert "metrics" in result
        assert "equal_weight_comparison" in result
        assert set(result["weights"].keys()) == {"AAPL", "MSFT"}

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_insufficient_data(self, mock_prov_cls):
        """Fewer than 30 return rows triggers error."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider

        # Return only 20 rows → after pct_change + dropna → 19 rows < 30
        provider.get_stock_data.side_effect = lambda sym, *a, **kw: pd.DataFrame(
            {"close": list(range(1, 21))}
        )

        mock_rp = MagicMock()
        with patch.dict("sys.modules", {"riskfolio": mock_rp}):
            result = optimize_hrp(["AAPL", "MSFT"])

        assert result["status"] == "error"
        assert "Insufficient" in result["error"] or "30" in result["error"]

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_all_symbols_fail(self, mock_prov_cls):
        """All fetches raise → error about <2 valid symbols."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider
        provider.get_stock_data.side_effect = Exception("timeout")

        mock_rp = MagicMock()
        with patch.dict("sys.modules", {"riskfolio": mock_rp}):
            result = optimize_hrp(["AAPL", "MSFT", "GOOG"])

        assert result["status"] == "error"
        assert "2+" in result["error"] or "valid" in result["error"].lower()
        assert "failed_symbols" in result

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_partial_success(self, mock_prov_cls):
        """One symbol fails, rest succeed — failed_symbols list populated."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider

        np.random.seed(42)
        n = 300
        dates = pd.bdate_range("2024-01-01", periods=n)

        def get_data(symbol, *_a, **_kw):
            if symbol == "BAD":
                raise Exception("not found")
            prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
            return pd.DataFrame({"close": prices}, index=dates)

        provider.get_stock_data.side_effect = get_data

        mock_rp = MagicMock()
        weights_df = pd.DataFrame({"weights": [0.5, 0.5]}, index=["AAPL", "MSFT"])
        mock_port = MagicMock()
        mock_port.optimization.return_value = weights_df
        mock_rp.HCPortfolio.return_value = mock_port

        with patch.dict("sys.modules", {"riskfolio": mock_rp}):
            result = optimize_hrp(["AAPL", "BAD", "MSFT"], days=252)

        assert result["status"] == "success"
        assert "BAD" in result["failed_symbols"]

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_weights_sum_approximately_one(self, mock_prov_cls):
        """Sum of HRP weights should be close to 1.0."""
        provider = MagicMock()
        mock_prov_cls.return_value = provider

        np.random.seed(7)
        n = 300
        dates = pd.bdate_range("2024-01-01", periods=n)

        provider.get_stock_data.side_effect = lambda sym, *a, **kw: pd.DataFrame(
            {"close": 100 + np.cumsum(np.random.randn(n) * 0.5)}, index=dates
        )

        mock_rp = MagicMock()
        weights_df = pd.DataFrame(
            {"weights": [0.35, 0.35, 0.30]}, index=["A", "B", "C"]
        )
        mock_port = MagicMock()
        mock_port.optimization.return_value = weights_df
        mock_rp.HCPortfolio.return_value = mock_port

        with patch.dict("sys.modules", {"riskfolio": mock_rp}):
            result = optimize_hrp(["A", "B", "C"], days=252)

        assert result["status"] == "success"
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0, abs=0.01)
