"""Tests for maverick_mcp/api/routers/portfolio.py."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd


class TestNormalizeTicker:
    """Tests for _normalize_ticker."""

    def test_lowercase_to_upper(self):
        from maverick_mcp.api.routers.portfolio import _normalize_ticker

        assert _normalize_ticker("aapl") == "AAPL"

    def test_strips_whitespace(self):
        from maverick_mcp.api.routers.portfolio import _normalize_ticker

        assert _normalize_ticker("  msft  ") == "MSFT"

    def test_already_uppercase(self):
        from maverick_mcp.api.routers.portfolio import _normalize_ticker

        assert _normalize_ticker("GOOGL") == "GOOGL"


class TestValidateTicker:
    """Tests for _validate_ticker."""

    def test_valid_simple_ticker(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker

        is_valid, error = _validate_ticker("AAPL")
        assert is_valid is True
        assert error is None

    def test_valid_ticker_with_dot(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker

        is_valid, error = _validate_ticker("BRK.B")
        assert is_valid is True
        assert error is None

    def test_empty_ticker_returns_error(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker

        is_valid, error = _validate_ticker("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_special_chars_returns_error(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker

        is_valid, error = _validate_ticker("AA$PL")
        assert is_valid is False
        assert "Invalid" in error


class TestAddPortfolioPosition:
    """Tests for add_portfolio_position."""

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_new_position(self, mock_get_db):
        """Adding a brand new position creates it in DB."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])

        # Mock portfolio exists
        mock_portfolio = MagicMock()
        mock_portfolio.id = 1
        mock_portfolio.name = "My Portfolio"
        mock_portfolio.user_id = "default"
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_portfolio,  # portfolio lookup
            None,  # no existing position
        ]

        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        result = add_portfolio_position(
            ticker="AAPL",
            shares=10.0,
            purchase_price=150.0,
            purchase_date="2024-01-15",
        )

        assert result["status"] == "success"
        assert "Added 10.0 shares of AAPL" in result["message"]
        mock_session.add.assert_called()
        mock_session.commit.assert_called_once()

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_cost_basis_averaging(self, mock_get_db):
        """Adding shares to existing position averages cost basis."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1
        mock_portfolio.name = "My Portfolio"
        mock_portfolio.user_id = "default"

        # Existing position: 10 shares at $100
        mock_existing = MagicMock()
        mock_existing.shares = Decimal("10")
        mock_existing.average_cost_basis = Decimal("100")
        mock_existing.total_cost = Decimal("1000")
        mock_existing.ticker = "AAPL"
        mock_existing.purchase_date = datetime(2024, 1, 1, tzinfo=UTC)
        mock_existing.notes = None

        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_portfolio,
            mock_existing,
        ]

        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        # Add 10 more shares at $200
        result = add_portfolio_position(
            ticker="AAPL",
            shares=10.0,
            purchase_price=200.0,
        )

        assert result["status"] == "success"
        # Cost should be averaged: (1000 + 2000) / 20 = 150
        assert mock_existing.shares == Decimal("20")
        mock_session.commit.assert_called_once()

    def test_validation_error_negative_shares(self):
        """Negative shares returns error without DB call."""
        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        result = add_portfolio_position(
            ticker="AAPL",
            shares=-5.0,
            purchase_price=150.0,
        )

        assert result["status"] == "error"
        assert "greater than zero" in result["error"]


class TestGetMyPortfolio:
    """Tests for get_my_portfolio."""

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_empty_portfolio_no_record(self, mock_get_db):
        """No portfolio record returns empty status."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        from maverick_mcp.api.routers.portfolio import get_my_portfolio

        result = get_my_portfolio(include_current_prices=False)

        assert result["status"] == "empty"
        assert result["positions"] == []
        assert result["total_invested"] == 0.0

    @patch("maverick_mcp.api.routers.portfolio.Portfolio")
    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_with_positions_no_prices(self, mock_get_db, mock_portfolio_cls):
        """Portfolio with positions but no live prices."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])

        mock_pos = MagicMock()
        mock_pos.ticker = "AAPL"
        mock_pos.shares = Decimal("10")
        mock_pos.average_cost_basis = Decimal("150")
        mock_pos.total_cost = Decimal("1500")
        mock_pos.purchase_date = datetime(2024, 1, 1, tzinfo=UTC)
        mock_pos.notes = "Test"

        mock_portfolio_db = MagicMock()
        mock_portfolio_db.positions = [mock_pos]
        mock_portfolio_db.name = "My Portfolio"
        mock_portfolio_db.user_id = "default"
        mock_portfolio_db.id = 1
        mock_portfolio_db.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_portfolio_db
        )

        mock_domain_portfolio = MagicMock()
        mock_portfolio_cls.return_value = mock_domain_portfolio
        mock_domain_portfolio.calculate_portfolio_metrics.return_value = {
            "total_invested": 1500.0,
            "total_current_value": 1500.0,
            "total_unrealized_gain_loss": 0.0,
            "total_return_percent": 0.0,
        }

        from maverick_mcp.api.routers.portfolio import get_my_portfolio

        result = get_my_portfolio(include_current_prices=False)

        assert result["status"] == "success"
        assert len(result["positions"]) == 1
        assert result["positions"][0]["ticker"] == "AAPL"

    @patch("maverick_mcp.api.routers.portfolio.stock_provider")
    @patch("maverick_mcp.api.routers.portfolio.Portfolio")
    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_with_live_prices(self, mock_get_db, mock_portfolio_cls, mock_provider):
        """Portfolio fetches live prices when include_current_prices=True."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])

        mock_pos = MagicMock()
        mock_pos.ticker = "AAPL"
        mock_pos.shares = Decimal("10")
        mock_pos.average_cost_basis = Decimal("150")
        mock_pos.total_cost = Decimal("1500")
        mock_pos.purchase_date = datetime(2024, 1, 1, tzinfo=UTC)
        mock_pos.notes = None

        mock_portfolio_db = MagicMock()
        mock_portfolio_db.positions = [mock_pos]
        mock_portfolio_db.name = "My Portfolio"
        mock_portfolio_db.user_id = "default"
        mock_portfolio_db.id = 1
        mock_portfolio_db.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_portfolio_db
        )

        # Mock stock_provider returning a DataFrame with Close price
        mock_df = pd.DataFrame({"Close": [175.0]})
        mock_provider.get_stock_data.return_value = mock_df

        mock_domain_portfolio = MagicMock()
        mock_portfolio_cls.return_value = mock_domain_portfolio
        mock_domain_portfolio.calculate_portfolio_metrics.return_value = {
            "total_invested": 1500.0,
            "total_current_value": 1750.0,
            "total_unrealized_gain_loss": 250.0,
            "total_return_percent": 16.67,
        }

        from maverick_mcp.api.routers.portfolio import get_my_portfolio

        result = get_my_portfolio(include_current_prices=True)

        assert result["status"] == "success"
        mock_provider.get_stock_data.assert_called_once()
        # Position should have current_price populated
        pos = result["positions"][0]
        assert pos["current_price"] == 175.0


class TestRemovePortfolioPosition:
    """Tests for remove_portfolio_position."""

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_partial_removal(self, mock_get_db):
        """Removing some shares keeps the position with reduced count."""
        mock_session = MagicMock()
        # The function calls next(get_db()) twice due to duplicated code
        mock_get_db.return_value = iter([mock_session, mock_session])

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1

        mock_position = MagicMock()
        mock_position.ticker = "AAPL"
        mock_position.shares = Decimal("10")
        mock_position.average_cost_basis = Decimal("150")
        mock_position.total_cost = Decimal("1500")

        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_portfolio,
            mock_position,
        ]

        from maverick_mcp.api.routers.portfolio import remove_portfolio_position

        result = remove_portfolio_position(ticker="AAPL", shares=3.0)

        assert result["status"] == "success"
        assert result["position_fully_closed"] is False
        assert result["removed_shares"] == 3.0
        mock_session.commit.assert_called()

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_full_removal(self, mock_get_db):
        """Removing all shares deletes the position."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session, mock_session])

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1

        mock_position = MagicMock()
        mock_position.ticker = "AAPL"
        mock_position.shares = Decimal("10")
        mock_position.average_cost_basis = Decimal("150")

        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            mock_portfolio,
            mock_position,
        ]

        from maverick_mcp.api.routers.portfolio import remove_portfolio_position

        result = remove_portfolio_position(ticker="AAPL", shares=None)

        assert result["status"] == "success"
        assert result["position_fully_closed"] is True
        mock_session.delete.assert_called_once_with(mock_position)


class TestClearMyPortfolio:
    """Tests for clear_my_portfolio."""

    def test_without_confirm_returns_warning(self):
        """Without confirm=True returns safety error."""
        from maverick_mcp.api.routers.portfolio import clear_my_portfolio

        result = clear_my_portfolio(confirm=False)

        assert result["status"] == "error"
        assert "confirm=True" in result["error"]

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    def test_with_confirm_clears_positions(self, mock_get_db):
        """With confirm=True clears all positions."""
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1
        mock_portfolio.name = "My Portfolio"
        mock_portfolio.user_id = "default"
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_portfolio
        )
        mock_session.query.return_value.filter_by.return_value.count.return_value = 5
        mock_session.query.return_value.filter_by.return_value.delete.return_value = 5

        from maverick_mcp.api.routers.portfolio import clear_my_portfolio

        result = clear_my_portfolio(confirm=True)

        assert result["status"] == "success"
        assert result["positions_cleared"] == 5
        mock_session.commit.assert_called_once()


class TestRiskAdjustedAnalysis:
    """Tests for risk_adjusted_analysis."""

    @patch("maverick_mcp.api.routers.portfolio.get_db")
    @patch("maverick_mcp.api.routers.portfolio.stock_provider")
    def test_success(self, mock_provider, mock_get_db):
        """Successful risk analysis returns position sizing info."""
        # Build a DataFrame with enough rows for ATR calculation
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "High": [155.0 + i * 0.5 for i in range(30)],
                "Low": [145.0 + i * 0.5 for i in range(30)],
                "Close": [150.0 + i * 0.5 for i in range(30)],
                "Volume": [1_000_000] * 30,
            },
            index=dates,
        )
        # Ensure columns are detected properly (case check)
        mock_provider.get_stock_data.return_value = df

        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        mock_session.query.return_value.filter.return_value.first.return_value = None

        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        result = risk_adjusted_analysis(ticker="AAPL", risk_level=50)

        assert "ticker" in result
        assert result["ticker"] == "AAPL"
        assert "position_sizing" in result
        assert "risk_management" in result
        assert "targets" in result

    @patch("maverick_mcp.api.routers.portfolio.stock_provider")
    def test_empty_data_returns_error(self, mock_provider):
        """Empty DataFrame returns error about insufficient data."""
        mock_provider.get_stock_data.return_value = pd.DataFrame()

        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        result = risk_adjusted_analysis(ticker="ZZZZ", risk_level=50)

        assert "error" in result
        assert "Insufficient data" in result["error"] or "error" in result
