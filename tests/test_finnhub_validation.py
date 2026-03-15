"""Tests for Finnhub validation models."""

import pytest
from pydantic import ValidationError

from maverick_mcp.validation.finnhub import (
    CompanyNewsRequest,
    EarningsCalendarRequest,
    EarningsSurprisesRequest,
    EconomicCalendarRequest,
    MarketNewsRequest,
    OwnershipRequest,
    PeersRequest,
    RecommendationsRequest,
)


class TestCompanyNewsRequest:
    """Tests for CompanyNewsRequest validation."""

    def test_valid_minimal(self):
        req = CompanyNewsRequest(ticker="AAPL")
        assert req.ticker == "AAPL"
        assert req.from_date is None
        assert req.to_date is None
        assert req.limit == 20

    def test_valid_full(self):
        req = CompanyNewsRequest(
            ticker="TSLA",
            from_date="2026-03-01",
            to_date="2026-03-13",
            limit=10,
        )
        assert req.ticker == "TSLA"
        assert req.from_date == "2026-03-01"
        assert req.limit == 10

    def test_ticker_must_be_uppercase(self):
        """Strict mode requires uppercase tickers (pattern enforced before validator)."""
        with pytest.raises(ValidationError):
            CompanyNewsRequest(ticker="aapl")

    def test_limit_too_high(self):
        with pytest.raises(ValidationError):
            CompanyNewsRequest(ticker="AAPL", limit=101)

    def test_limit_zero(self):
        with pytest.raises(ValidationError):
            CompanyNewsRequest(ticker="AAPL", limit=0)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            CompanyNewsRequest(ticker="AAPL", extra_field="bad")


class TestEarningsCalendarRequest:
    """Tests for EarningsCalendarRequest validation."""

    def test_valid_empty(self):
        req = EarningsCalendarRequest()
        assert req.from_date is None
        assert req.to_date is None
        assert req.ticker is None

    def test_valid_with_ticker(self):
        req = EarningsCalendarRequest(ticker="MSFT")
        assert req.ticker == "MSFT"

    def test_ticker_normalised(self):
        req = EarningsCalendarRequest(ticker="msft")
        assert req.ticker == "MSFT"

    def test_none_ticker_allowed(self):
        req = EarningsCalendarRequest(ticker=None)
        assert req.ticker is None


class TestEarningsSurprisesRequest:
    """Tests for EarningsSurprisesRequest validation."""

    def test_valid_defaults(self):
        req = EarningsSurprisesRequest(ticker="AAPL")
        assert req.limit == 4

    def test_limit_bounds(self):
        req = EarningsSurprisesRequest(ticker="AAPL", limit=20)
        assert req.limit == 20

        with pytest.raises(ValidationError):
            EarningsSurprisesRequest(ticker="AAPL", limit=21)


class TestRecommendationsRequest:
    """Tests for RecommendationsRequest validation."""

    def test_valid(self):
        req = RecommendationsRequest(ticker="GOOGL")
        assert req.ticker == "GOOGL"

    def test_invalid_ticker(self):
        with pytest.raises(ValidationError):
            RecommendationsRequest(ticker="")


class TestOwnershipRequest:
    """Tests for OwnershipRequest validation."""

    def test_defaults(self):
        req = OwnershipRequest(ticker="AMZN")
        assert req.limit == 20

    def test_limit_max(self):
        req = OwnershipRequest(ticker="AMZN", limit=50)
        assert req.limit == 50

        with pytest.raises(ValidationError):
            OwnershipRequest(ticker="AMZN", limit=51)


class TestPeersRequest:
    """Tests for PeersRequest validation."""

    def test_valid(self):
        req = PeersRequest(ticker="META")
        assert req.ticker == "META"


class TestEconomicCalendarRequest:
    """Tests for EconomicCalendarRequest validation."""

    def test_valid_empty(self):
        req = EconomicCalendarRequest()
        assert req.from_date is None
        assert req.to_date is None

    def test_valid_dates(self):
        req = EconomicCalendarRequest(from_date="2026-03-10", to_date="2026-03-17")
        assert req.from_date == "2026-03-10"


class TestMarketNewsRequest:
    """Tests for MarketNewsRequest validation."""

    def test_defaults(self):
        req = MarketNewsRequest()
        assert req.category == "general"
        assert req.min_id == 0

    def test_valid_category(self):
        for cat in ("general", "forex", "crypto", "merger"):
            req = MarketNewsRequest(category=cat)
            assert req.category == cat

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            MarketNewsRequest(category="sports")

    def test_negative_min_id(self):
        with pytest.raises(ValidationError):
            MarketNewsRequest(min_id=-1)
