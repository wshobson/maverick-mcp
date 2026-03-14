"""Tests for maverick_mcp/api/routers/data.py."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd

from maverick_mcp.api.routers.data import (
    _dataframe_to_split_dict,
    clear_cache,
    fetch_stock_data,
    fetch_stock_data_batch,
    get_cached_price_data,
    get_chart_links,
    get_fundamental_analysis,
    get_news_sentiment,
    get_stock_info,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_cm(mock_session=None):
    """Return a context-manager that yields *mock_session*."""
    if mock_session is None:
        mock_session = MagicMock()

    @contextmanager
    def _cm():
        yield mock_session

    return _cm


def _sample_df():
    """Small OHLCV-like DataFrame with a DatetimeIndex."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {"open": [100.0, 101.0], "close": [102.0, 103.0]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# _dataframe_to_split_dict
# ---------------------------------------------------------------------------


class TestDataframeToSplitDict:
    def test_empty_dataframe_returns_empty_dict(self):
        result = _dataframe_to_split_dict(pd.DataFrame())
        assert result == {}

    def test_converts_dates_to_iso_strings(self):
        df = _sample_df()
        result = _dataframe_to_split_dict(df)
        assert "index" in result
        assert "columns" in result
        assert "data" in result
        # ISO strings should contain "2024-01-02"
        assert "2024-01-02" in result["index"][0]
        assert "2024-01-03" in result["index"][1]

    def test_non_datetime_index_converted_to_str(self):
        df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        result = _dataframe_to_split_dict(df)
        assert result["index"] == ["x", "y"]


# ---------------------------------------------------------------------------
# fetch_stock_data
# ---------------------------------------------------------------------------


class TestFetchStockData:
    @patch("maverick_mcp.api.routers.data.StockAnalysisService")
    @patch("maverick_mcp.api.routers.data.CacheManagementService")
    @patch("maverick_mcp.api.routers.data.StockDataFetchingService")
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_success(
        self, mock_get_db, mock_fetch_svc, mock_cache_svc, mock_analysis_svc
    ):
        mock_get_db.side_effect = _make_session_cm()
        mock_service_instance = MagicMock()
        mock_service_instance.get_stock_data.return_value = _sample_df()
        mock_analysis_svc.return_value = mock_service_instance

        result = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31")

        assert result["ticker"] == "AAPL"
        assert result["record_count"] == 2
        assert result["interval"] == "1d"
        assert "index" in result

    @patch("maverick_mcp.api.routers.data.StockDataFetchingService")
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_error_returns_error_dict(self, mock_get_db, mock_fetch_svc):
        mock_get_db.side_effect = RuntimeError("db down")

        result = fetch_stock_data("AAPL")

        assert "error" in result
        assert result["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# fetch_stock_data_batch
# ---------------------------------------------------------------------------


class TestFetchStockDataBatch:
    @patch("maverick_mcp.api.routers.data.StockAnalysisService")
    @patch("maverick_mcp.api.routers.data.CacheManagementService")
    @patch("maverick_mcp.api.routers.data.StockDataFetchingService")
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_success_and_error_mixed(
        self, mock_get_db, mock_fetch_svc, mock_cache_svc, mock_analysis_svc
    ):
        mock_get_db.side_effect = _make_session_cm()
        mock_service_instance = MagicMock()
        # First call succeeds, second raises
        mock_service_instance.get_stock_data.side_effect = [
            _sample_df(),
            RuntimeError("not found"),
        ]
        mock_analysis_svc.return_value = mock_service_instance

        result = fetch_stock_data_batch(["AAPL", "BAD"])

        assert result["success_count"] == 1
        assert result["error_count"] == 1
        assert result["results"]["AAPL"]["status"] == "success"
        assert result["results"]["BAD"]["status"] == "error"


# ---------------------------------------------------------------------------
# get_stock_info
# ---------------------------------------------------------------------------


class TestGetStockInfo:
    @patch("maverick_mcp.api.routers.data.StockDataProvider")
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_success(self, mock_get_db, mock_provider_cls):
        mock_get_db.side_effect = _make_session_cm()
        mock_provider_cls.return_value.get_stock_info.return_value = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currentPrice": 195.0,
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.5,
        }

        result = get_stock_info("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["company"]["name"] == "Apple Inc."
        assert result["company"]["sector"] == "Technology"
        assert result["market_data"]["current_price"] == 195.0
        assert result["valuation"]["pe_ratio"] == 30.5

    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_error(self, mock_get_db):
        mock_get_db.side_effect = RuntimeError("db error")

        result = get_stock_info("AAPL")
        assert "error" in result
        assert result["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# get_fundamental_analysis
# ---------------------------------------------------------------------------


class TestGetFundamentalAnalysis:
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_success(self, mock_get_db):
        mock_get_db.side_effect = _make_session_cm()

        with (
            patch(
                "maverick_mcp.api.routers.data.StockDataProvider"
            ) as mock_provider_cls,
            patch(
                "maverick_mcp.core.fundamental_analysis.compute_fundamental_score"
            ) as mock_score,
            patch(
                "maverick_mcp.core.fundamental_analysis.get_earnings_analysis"
            ) as mock_earn,
            patch(
                "maverick_mcp.core.fundamental_analysis.get_valuation_assessment"
            ) as mock_val,
            patch(
                "maverick_mcp.core.fundamental_analysis.get_financial_health"
            ) as mock_health,
        ):
            mock_provider_cls.return_value.get_stock_info.return_value = {
                "longName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
            }
            mock_score.return_value = {"fundamental_score": 82, "grade": "B+"}
            mock_earn.return_value = {"eps": 6.5}
            mock_val.return_value = {"pe_fair_value": 28}
            mock_health.return_value = {"current_ratio": 1.5}

            result = get_fundamental_analysis("AAPL")

            assert result["ticker"] == "AAPL"
            assert result["fundamental_score"] == 82
            assert result["grade"] == "B+"

    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_error(self, mock_get_db):
        mock_get_db.side_effect = RuntimeError("import boom")

        result = get_fundamental_analysis("AAPL")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_news_sentiment
# ---------------------------------------------------------------------------


class TestGetNewsSentiment:
    @patch("maverick_mcp.api.routers.data.settings")
    def test_no_api_key_returns_fallback(self, mock_settings):
        mock_settings.external_data.api_key = None

        result = get_news_sentiment("AAPL")

        assert result["status"] == "fallback_mode"
        assert result["sentiment"] == "neutral"
        assert result["ticker"] == "AAPL"

    @patch("maverick_mcp.api.routers.data.requests.get")
    @patch("maverick_mcp.api.routers.data.settings")
    def test_success(self, mock_settings, mock_get):
        mock_settings.external_data.api_key = "test-key"
        mock_settings.external_data.base_url = "https://api.example.com"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ticker": "AAPL", "sentiment": "bullish"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = get_news_sentiment("AAPL")

        assert result["sentiment"] == "bullish"
        mock_get.assert_called_once()

    @patch("maverick_mcp.api.routers.data.requests.get")
    @patch("maverick_mcp.api.routers.data.settings")
    def test_404_returns_not_found(self, mock_settings, mock_get):
        mock_settings.external_data.api_key = "test-key"
        mock_settings.external_data.base_url = "https://api.example.com"
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = get_news_sentiment("XYZ")

        assert result["status"] == "not_found"

    @patch("maverick_mcp.api.routers.data.requests.get")
    @patch("maverick_mcp.api.routers.data.settings")
    def test_timeout(self, mock_settings, mock_get):
        mock_settings.external_data.api_key = "test-key"
        mock_settings.external_data.base_url = "https://api.example.com"
        import requests.exceptions

        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        result = get_news_sentiment("AAPL")

        assert result["status"] == "timeout"


# ---------------------------------------------------------------------------
# get_cached_price_data
# ---------------------------------------------------------------------------


class TestGetCachedPriceData:
    @patch("maverick_mcp.api.routers.data.PriceCache")
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_no_data(self, mock_get_db, mock_price_cache):
        mock_get_db.side_effect = _make_session_cm()
        mock_price_cache.get_price_data.return_value = pd.DataFrame()

        result = get_cached_price_data("AAPL", "2024-01-01")

        assert result["status"] == "success"
        assert result["data"] == []

    @patch("maverick_mcp.api.routers.data.PriceCache")
    @patch("maverick_mcp.api.routers.data.get_db_session_read_only")
    def test_with_data(self, mock_get_db, mock_price_cache):
        mock_get_db.side_effect = _make_session_cm()
        df = _sample_df()
        mock_price_cache.get_price_data.return_value = df

        result = get_cached_price_data("AAPL", "2024-01-01", "2024-01-31")

        assert result["status"] == "success"
        assert result["count"] == 2
        assert result["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# get_chart_links
# ---------------------------------------------------------------------------


class TestGetChartLinks:
    def test_links_contain_ticker(self):
        result = get_chart_links("MSFT")

        assert result["ticker"] == "MSFT"
        charts = result["charts"]
        assert "MSFT" in charts["trading_view"]
        assert "MSFT" in charts["finviz"]
        assert "MSFT" in charts["yahoo_finance"]
        assert "MSFT" in charts["stock_charts"]
        assert "MSFT" in charts["seeking_alpha"]
        assert "MSFT" in charts["marketwatch"]

    def test_returns_six_chart_providers(self):
        result = get_chart_links("AAPL")
        assert len(result["charts"]) == 6


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    @patch("maverick_mcp.data.cache.clear_cache", return_value=5)
    def test_clear_specific_ticker(self, mock_cache_clear):
        result = clear_cache("AAPL")

        assert result["status"] == "success"
        assert result["entries_cleared"] == 5
        mock_cache_clear.assert_called_once_with("stock:AAPL:*")

    @patch("maverick_mcp.data.cache.clear_cache", return_value=42)
    def test_clear_all(self, mock_cache_clear):
        result = clear_cache(None)

        assert result["status"] == "success"
        assert result["entries_cleared"] == 42
        mock_cache_clear.assert_called_once_with()

    @patch(
        "maverick_mcp.data.cache.clear_cache", side_effect=RuntimeError("redis down")
    )
    def test_error(self, mock_cache_clear):
        result = clear_cache("AAPL")

        assert result["status"] == "error"
        assert "error" in result
