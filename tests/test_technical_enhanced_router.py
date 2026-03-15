"""Tests for the enhanced technical analysis router."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from maverick_mcp.api.routers.technical_enhanced import (
    get_full_technical_analysis_enhanced,
    get_stock_chart_analysis_enhanced,
)


def _make_request(ticker="AAPL", days=365):
    """Create a mock TechnicalAnalysisRequest."""
    req = MagicMock()
    req.ticker = ticker
    req.days = days
    return req


def _make_df(rows=100):
    """Create a sample OHLCV DataFrame."""
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(rows).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.random(rows),
            "high": close + rng.random(rows),
            "low": close - rng.random(rows),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, rows),
        },
        index=dates,
    )


class TestGetFullTechnicalAnalysisEnhanced:
    @pytest.mark.asyncio
    async def test_success(self):
        df = _make_df()
        request = _make_request()

        with (
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_stock_dataframe_async",
                new_callable=AsyncMock,
                return_value=df,
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_access_token",
                side_effect=Exception("no auth"),
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.analyze_rsi",
                return_value={"value": 55},
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.analyze_macd",
                return_value={"signal": "bullish"},
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.analyze_stochastic",
                return_value={"k": 60},
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.analyze_trend",
                return_value="uptrend",
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.analyze_bollinger_bands",
                return_value={"upper": 110},
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.analyze_volume",
                return_value={"avg_vol": 5_000_000},
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.identify_chart_patterns",
                return_value=[],
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.identify_support_levels",
                return_value=[95.0, 90.0],
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.identify_resistance_levels",
                return_value=[105.0, 110.0],
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.generate_outlook",
                return_value="bullish",
            ),
        ):
            result = await get_full_technical_analysis_enhanced(request)
            assert result["status"] == "completed"
            assert result["ticker"] == "AAPL"
            assert "indicators" in result
            assert "levels" in result
            assert result["levels"]["support"] == [90.0, 95.0]

    @pytest.mark.asyncio
    async def test_timeout(self):
        request = _make_request()

        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(30)
            return _make_df()

        with (
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_stock_dataframe_async",
                side_effect=slow_fetch,
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_access_token",
                side_effect=Exception("no auth"),
            ),
        ):
            result = await get_full_technical_analysis_enhanced(request)
            assert result["status"] == "failed"
            # The inner 8s data-fetch timeout fires first as TechnicalAnalysisError,
            # caught by the outer handler, so error_type may be either
            assert result["error_type"] in ("timeout", "TechnicalAnalysisError")
            assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_data(self):
        request = _make_request()
        empty_df = pd.DataFrame()

        with (
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_stock_dataframe_async",
                new_callable=AsyncMock,
                return_value=empty_df,
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_access_token",
                side_effect=Exception("no auth"),
            ),
        ):
            result = await get_full_technical_analysis_enhanced(request)
            assert result["status"] == "failed"
            assert "No data" in result["error"] or "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_general_exception(self):
        request = _make_request()

        with (
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_stock_dataframe_async",
                new_callable=AsyncMock,
                side_effect=ValueError("bad ticker"),
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_access_token",
                side_effect=Exception("no auth"),
            ),
        ):
            result = await get_full_technical_analysis_enhanced(request)
            assert result["status"] == "failed"
            assert result["error_type"] in ("ValueError", "TechnicalAnalysisError")


class TestGetStockChartAnalysisEnhanced:
    @pytest.mark.asyncio
    async def test_success(self):
        df = _make_df()
        # Small base64 string to pass the 200KB check
        small_data_uri = "data:image/png;base64," + "A" * 1000

        with (
            patch(
                "maverick_mcp.api.routers.technical_enhanced.get_stock_dataframe_async",
                new_callable=AsyncMock,
                return_value=df,
            ),
            patch(
                "maverick_mcp.api.routers.technical_enhanced._generate_chart_with_logging",
                new_callable=AsyncMock,
                return_value={
                    "ticker": "AAPL",
                    "chart_data": small_data_uri,
                    "chart_format": "png",
                    "chart_size": {"height": 400, "width": 600},
                    "data_points": 100,
                    "status": "completed",
                    "timestamp": "2024-01-01",
                },
            ),
        ):
            result = await get_stock_chart_analysis_enhanced("AAPL")
            assert result["status"] == "completed"
            assert result["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_timeout(self):
        async def slow_chart(*args, **kwargs):
            await asyncio.sleep(20)

        with patch(
            "maverick_mcp.api.routers.technical_enhanced._generate_chart_with_logging",
            side_effect=slow_chart,
        ):
            result = await get_stock_chart_analysis_enhanced("AAPL")
            assert result["status"] == "failed"
            assert result["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_general_exception(self):
        with patch(
            "maverick_mcp.api.routers.technical_enhanced._generate_chart_with_logging",
            new_callable=AsyncMock,
            side_effect=RuntimeError("chart broke"),
        ):
            result = await get_stock_chart_analysis_enhanced("AAPL")
            assert result["status"] == "failed"
            assert "chart broke" in result["error"]
