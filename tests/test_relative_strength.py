"""Tests for maverick_mcp/core/relative_strength.py — EARS scoring."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from maverick_mcp.core.relative_strength import (
    _get_return,
    calculate_ears_score,
    enrich_stocks_with_ears,
)

# ---------------------------------------------------------------------------
# TestGetReturn
# ---------------------------------------------------------------------------


class TestGetReturn:
    def test_normal_case(self):
        """Return over lookback window with known prices."""
        df = pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0, 140.0]})
        # lookback=3 → start at index -3 (120), end at index -1 (140)
        result = _get_return(df, 3)
        assert result is not None
        assert result == pytest.approx((140 - 120) / 120)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"close": []})
        assert _get_return(df, 5) is None

    def test_insufficient_lookback(self):
        """DataFrame shorter than lookback returns None."""
        df = pd.DataFrame({"close": [100.0, 110.0]})
        assert _get_return(df, 5) is None

    def test_zero_start_price(self):
        """Zero start price avoids division-by-zero, returns None."""
        df = pd.DataFrame({"close": [50.0, 0.0, 120.0]})
        # lookback=2 → start at index -2 which is 0.0
        assert _get_return(df, 2) is None

    def test_uppercase_close_column(self):
        """Handles 'Close' (capital C) column name."""
        df = pd.DataFrame({"Close": [200.0, 250.0, 300.0]})
        result = _get_return(df, 2)
        assert result == pytest.approx((300 - 250) / 250)


# ---------------------------------------------------------------------------
# TestCalculateEarsScore
# ---------------------------------------------------------------------------


class TestCalculateEarsScore:
    def test_with_both_returns(self):
        """Weighted blend of RS vs SPY (60 %) and RS vs sector (40 %)."""
        # ticker +20 %, SPY +10 %, sector +5 %
        score = calculate_ears_score(0.20, 0.10, 0.05)
        rs_spy = (0.20 / 0.10) * 50 + 50  # 150
        rs_sec = (0.20 / 0.05) * 50 + 50  # 250
        expected = max(0, min(100, rs_spy * 0.6 + rs_sec * 0.4))
        assert score == pytest.approx(expected, abs=0.01)

    def test_spy_only_no_sector(self):
        """When sector_return is None only SPY component is used."""
        score = calculate_ears_score(0.10, 0.10, None)
        # rs_vs_spy = (0.10/0.10)*50 + 50 = 100
        assert score == pytest.approx(100.0)

    def test_zero_spy_return(self):
        """SPY return == 0 gives rs_vs_spy = 50."""
        score = calculate_ears_score(0.05, 0.0, None)
        assert score == 50.0

    def test_score_clamped_to_0_100(self):
        """Extreme values are clamped to [0, 100]."""
        # ticker hugely outperforms → raw score > 100
        score_high = calculate_ears_score(1.0, 0.01, None)
        assert score_high == 100.0

        # ticker hugely underperforms → raw score < 0
        score_low = calculate_ears_score(-1.0, 0.01, None)
        assert score_low == 0.0


# ---------------------------------------------------------------------------
# TestEnrichStocksWithEars
# ---------------------------------------------------------------------------


class TestEnrichStocksWithEars:
    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_success_with_sector_etf(self, mock_provider_cls):
        """Stocks are enriched with ears_score when data is available."""
        provider = MagicMock()
        mock_provider_cls.return_value = provider

        # Build simple DataFrames with enough rows (lookback default=63)
        n = 130  # > 63*2 days fetched
        spy_prices = [100.0] * (n - 1) + [110.0]
        ticker_prices = [100.0] * (n - 1) + [130.0]
        sector_prices = [100.0] * (n - 1) + [115.0]

        def get_data(symbol, *_a, **_kw):
            mapping = {
                "SPY": spy_prices,
                "AAPL": ticker_prices,
                "XLK": sector_prices,
            }
            return pd.DataFrame({"close": mapping[symbol]})

        provider.get_stock_data.side_effect = get_data

        stocks = [{"ticker": "AAPL", "sector": "Technology"}]
        result = enrich_stocks_with_ears(stocks, days=63)

        assert len(result) == 1
        assert "ears_score" in result[0]
        assert isinstance(result[0]["ears_score"], float)
        assert 0 <= result[0]["ears_score"] <= 100

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_spy_unavailable_returns_stocks_unchanged(self, mock_provider_cls):
        """When SPY fetch raises, stocks are returned without ears_score."""
        provider = MagicMock()
        mock_provider_cls.return_value = provider
        provider.get_stock_data.side_effect = Exception("network error")

        stocks = [{"ticker": "MSFT"}]
        result = enrich_stocks_with_ears(stocks)

        assert result == stocks
        assert "ears_score" not in result[0]

    def test_empty_list(self):
        """Empty input returns empty without calling provider."""
        result = enrich_stocks_with_ears([])
        assert result == []
