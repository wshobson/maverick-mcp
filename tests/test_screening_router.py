"""Tests for maverick_mcp/api/routers/screening.py."""

from unittest.mock import MagicMock, patch


def _make_stock_mock(ticker="AAPL", momentum_score=85, combined_score=90):
    """Create a mock stock object with to_dict()."""
    mock = MagicMock()
    mock.to_dict.return_value = {
        "ticker": ticker,
        "momentum_score": momentum_score,
        "combined_score": combined_score,
        "close_price": 150.0,
        "avg_vol_30d": 50_000_000,
    }
    return mock


class TestGetMaverickStocks:
    """Tests for get_maverick_stocks."""

    @patch("maverick_mcp.core.relative_strength.enrich_stocks_with_ears")
    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickStocks")
    def test_basic_success_no_regime_filter(
        self, mock_maverick_model, mock_session_local, mock_ears
    ):
        """Basic success with regime_filter=False."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        stocks = [_make_stock_mock("AAPL"), _make_stock_mock("MSFT", 80, 85)]
        mock_maverick_model.get_top_stocks.return_value = stocks
        stock_dicts = [s.to_dict() for s in stocks]
        mock_ears.return_value = stock_dicts

        from maverick_mcp.api.routers.screening import get_maverick_stocks

        result = get_maverick_stocks(limit=20, regime_filter=False)

        assert result["status"] == "success"
        assert result["count"] == 2
        assert result["screening_type"] == "maverick_bullish"
        assert len(result["stocks"]) == 2

    @patch("maverick_mcp.core.relative_strength.enrich_stocks_with_ears")
    @patch("maverick_mcp.core.regime_gate.apply_regime_filter")
    @patch("maverick_mcp.core.regime_gate.get_current_regime")
    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickStocks")
    def test_with_regime_filter(
        self,
        mock_maverick_model,
        mock_session_local,
        mock_get_regime,
        mock_apply_filter,
        mock_ears,
    ):
        """With regime_filter=True, regime gate is applied."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        stocks = [_make_stock_mock("AAPL")]
        mock_maverick_model.get_top_stocks.return_value = stocks

        stock_dicts = [s.to_dict() for s in stocks]
        mock_get_regime.return_value = "bull"
        filtered = [stock_dicts[0]]
        regime_context = {"regime": "bull", "confidence": 0.8}
        mock_apply_filter.return_value = (filtered, regime_context)
        mock_ears.return_value = filtered

        from maverick_mcp.api.routers.screening import get_maverick_stocks

        result = get_maverick_stocks(limit=20, regime_filter=True)

        assert result["status"] == "success"
        assert "current_regime" in result
        mock_get_regime.assert_called_once()

    @patch("maverick_mcp.core.relative_strength.enrich_stocks_with_ears")
    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickStocks")
    def test_ears_enrichment(
        self,
        mock_maverick_model,
        mock_session_local,
        mock_ears,
    ):
        """EARS enrichment is applied to stocks."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        stocks = [_make_stock_mock("AAPL")]
        mock_maverick_model.get_top_stocks.return_value = stocks

        enriched = [{"ticker": "AAPL", "momentum_score": 85, "ears_score": 92}]
        mock_ears.return_value = enriched

        from maverick_mcp.api.routers.screening import get_maverick_stocks

        result = get_maverick_stocks(limit=10, regime_filter=False)

        assert result["status"] == "success"
        mock_ears.assert_called_once()

    @patch("maverick_mcp.data.models.SessionLocal")
    def test_db_error(self, mock_session_local):
        """DB error returns error response."""
        mock_session_local.side_effect = RuntimeError("DB connection failed")

        from maverick_mcp.api.routers.screening import get_maverick_stocks

        result = get_maverick_stocks(limit=20, regime_filter=False)

        assert result["status"] == "error"
        assert "DB connection failed" in result["error"]


class TestGetMaverickBearStocks:
    """Tests for get_maverick_bear_stocks."""

    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickBearStocks")
    def test_success(self, mock_bear_model, mock_session_local):
        """Basic success returns bear stocks."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        stocks = [_make_stock_mock("XYZ", 20, 15)]
        mock_bear_model.get_top_stocks.return_value = stocks

        from maverick_mcp.api.routers.screening import get_maverick_bear_stocks

        result = get_maverick_bear_stocks(limit=10)

        assert result["status"] == "success"
        assert result["screening_type"] == "maverick_bearish"
        assert result["count"] == 1

    @patch("maverick_mcp.data.models.SessionLocal")
    def test_db_error(self, mock_session_local):
        """DB error returns error response."""
        mock_session_local.side_effect = RuntimeError("Connection refused")

        from maverick_mcp.api.routers.screening import get_maverick_bear_stocks

        result = get_maverick_bear_stocks(limit=20)

        assert result["status"] == "error"
        assert "Connection refused" in result["error"]


class TestGetSupplyDemandBreakouts:
    """Tests for get_supply_demand_breakouts."""

    @patch("maverick_mcp.core.relative_strength.enrich_stocks_with_ears")
    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.SupplyDemandBreakoutStocks")
    def test_without_filter_moving_averages(
        self, mock_sd_model, mock_session_local, mock_ears
    ):
        """Without filter_moving_averages uses get_top_stocks."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        stocks = [_make_stock_mock("TSLA", 75, 80)]
        mock_sd_model.get_top_stocks.return_value = stocks
        mock_ears.return_value = [s.to_dict() for s in stocks]

        from maverick_mcp.api.routers.screening import get_supply_demand_breakouts

        result = get_supply_demand_breakouts(
            limit=20, filter_moving_averages=False, regime_filter=False
        )

        assert result["status"] == "success"
        assert result["screening_type"] == "supply_demand_breakout"
        mock_sd_model.get_top_stocks.assert_called_once_with(mock_session, limit=20)

    @patch("maverick_mcp.core.relative_strength.enrich_stocks_with_ears")
    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.SupplyDemandBreakoutStocks")
    def test_with_filter_moving_averages(
        self, mock_sd_model, mock_session_local, mock_ears
    ):
        """With filter_moving_averages uses get_stocks_above_moving_averages."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        stocks = [_make_stock_mock("NVDA", 90, 95)]
        mock_sd_model.get_stocks_above_moving_averages.return_value = stocks
        mock_ears.return_value = [s.to_dict() for s in stocks]

        from maverick_mcp.api.routers.screening import get_supply_demand_breakouts

        result = get_supply_demand_breakouts(
            limit=20, filter_moving_averages=True, regime_filter=False
        )

        assert result["status"] == "success"
        mock_sd_model.get_stocks_above_moving_averages.assert_called_once_with(
            mock_session
        )


class TestGetAllScreeningRecommendations:
    """Tests for get_all_screening_recommendations."""

    @patch("maverick_mcp.providers.stock_data.StockDataProvider")
    def test_success(self, mock_provider_cls):
        """Calls provider.get_all_screening_recommendations."""
        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_provider.get_all_screening_recommendations.return_value = {
            "status": "success",
            "maverick_stocks": [{"ticker": "AAPL"}],
            "maverick_bear_stocks": [{"ticker": "XYZ"}],
            "supply_demand_breakouts": [{"ticker": "TSLA"}],
        }

        from maverick_mcp.api.routers.screening import (
            get_all_screening_recommendations,
        )

        result = get_all_screening_recommendations()

        assert result["status"] == "success"
        assert "maverick_stocks" in result


class TestGetScreeningByCriteria:
    """Tests for get_screening_by_criteria."""

    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickStocks")
    def test_with_numeric_params(self, mock_model, mock_session_local):
        """Numeric params are used as filters."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        mock_stock = _make_stock_mock("AAPL", 90, 95)

        # Make the model attributes return objects that support >= comparisons
        mock_model.momentum_score = MagicMock()
        mock_model.momentum_score.__ge__ = MagicMock(return_value="filter_expr")
        mock_model.avg_vol_30d = MagicMock()
        mock_model.avg_vol_30d.__ge__ = MagicMock(return_value="filter_expr")
        mock_model.close_price = MagicMock()
        mock_model.close_price.__le__ = MagicMock(return_value="filter_expr")
        mock_model.combined_score = MagicMock()

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_stock]

        from maverick_mcp.api.routers.screening import get_screening_by_criteria

        result = get_screening_by_criteria(
            min_momentum_score=80.0,
            min_volume=1_000_000,
            max_price=200.0,
            limit=10,
        )

        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["criteria"]["min_momentum_score"] == 80.0

    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickStocks")
    def test_with_string_params_coercion(self, mock_model, mock_session_local):
        """String params are coerced to numeric types."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        # Make the model attributes support comparisons
        mock_model.momentum_score = MagicMock()
        mock_model.momentum_score.__ge__ = MagicMock(return_value="filter_expr")
        mock_model.avg_vol_30d = MagicMock()
        mock_model.avg_vol_30d.__ge__ = MagicMock(return_value="filter_expr")
        mock_model.close_price = MagicMock()
        mock_model.close_price.__le__ = MagicMock(return_value="filter_expr")
        mock_model.combined_score = MagicMock()

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        from maverick_mcp.api.routers.screening import get_screening_by_criteria

        # Pass strings instead of numbers — function should coerce them
        result = get_screening_by_criteria(
            min_momentum_score="75",
            min_volume="500000",
            max_price="300",
            limit="15",
        )

        assert result["status"] == "success"
        assert result["criteria"]["min_momentum_score"] == 75.0
        assert result["criteria"]["min_volume"] == 500000
        assert result["criteria"]["max_price"] == 300.0

    @patch("maverick_mcp.data.models.SessionLocal")
    @patch("maverick_mcp.data.models.MaverickStocks")
    def test_empty_results(self, mock_model, mock_session_local):
        """Empty results return count=0."""
        mock_session = MagicMock()
        mock_session_local.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = MagicMock(return_value=False)

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        from maverick_mcp.api.routers.screening import get_screening_by_criteria

        result = get_screening_by_criteria()

        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["stocks"] == []
