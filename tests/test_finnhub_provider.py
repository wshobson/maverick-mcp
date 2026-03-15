"""Tests for FinnhubDataProvider."""

from unittest.mock import MagicMock, patch

import pytest

from maverick_mcp.providers.finnhub_data import FinnhubDataProvider

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def mock_settings():
    """Patch settings to provide a dummy API key and defaults."""
    with patch("maverick_mcp.providers.finnhub_data.settings") as s:
        s.finnhub.api_key = "test-key"
        s.finnhub.cache_ttl_seconds = 300
        s.finnhub.rate_limit_per_minute = 60
        yield s


@pytest.fixture()
def mock_client():
    """Create a MagicMock that stands in for finnhub.Client."""
    return MagicMock()


@pytest.fixture()
def provider(mock_settings, mock_client):
    """Return a FinnhubDataProvider with a mocked client."""
    with patch("maverick_mcp.providers.finnhub_data.finnhub") as finnhub_mod:
        finnhub_mod.Client.return_value = mock_client
        prov = FinnhubDataProvider()
    # Swap in the mock client (in case __init__ stored a real one)
    prov._client = mock_client
    return prov


@pytest.fixture()
def provider_no_key():
    """Return a FinnhubDataProvider without an API key."""
    with patch("maverick_mcp.providers.finnhub_data.settings") as s:
        s.finnhub.api_key = None
        s.finnhub.cache_ttl_seconds = 300
        s.finnhub.rate_limit_per_minute = 60
        prov = FinnhubDataProvider()
    return prov


# --------------------------------------------------------------------------- #
# Init tests
# --------------------------------------------------------------------------- #


class TestProviderInit:
    def test_init_with_key(self, provider):
        assert provider._client is not None

    def test_init_without_key(self, provider_no_key):
        assert provider_no_key._client is None

    def test_rate_limiter_initialised(self, provider):
        assert provider._max_tokens == 60.0
        assert provider._tokens > 0


# --------------------------------------------------------------------------- #
# Company news
# --------------------------------------------------------------------------- #


class TestCompanyNews:
    def test_returns_articles(self, provider, mock_client):
        mock_client.company_news.return_value = [
            {"headline": "AAPL up", "source": "Reuters", "url": "http://r.com"}
        ]
        result = provider.get_company_news("AAPL")
        assert len(result) == 1
        assert result[0]["headline"] == "AAPL up"
        mock_client.company_news.assert_called_once()

    def test_api_error_returns_empty(self, provider, mock_client):
        mock_client.company_news.side_effect = Exception("timeout")
        result = provider.get_company_news("AAPL")
        assert result == []


# --------------------------------------------------------------------------- #
# Earnings calendar
# --------------------------------------------------------------------------- #


class TestEarningsCalendar:
    def test_returns_calendar(self, provider, mock_client):
        mock_client.earnings_calendar.return_value = {
            "earningsCalendar": [
                {"symbol": "AAPL", "date": "2026-04-01", "epsEstimate": 1.50}
            ]
        }
        result = provider.get_earnings_calendar()
        assert "earningsCalendar" in result
        assert len(result["earningsCalendar"]) == 1

    def test_filtered_by_ticker(self, provider, mock_client):
        mock_client.earnings_calendar.return_value = {"earningsCalendar": []}
        provider.get_earnings_calendar(ticker="AAPL")
        call_kwargs = mock_client.earnings_calendar.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"


# --------------------------------------------------------------------------- #
# Earnings surprises
# --------------------------------------------------------------------------- #


class TestEarningsSurprises:
    def test_returns_surprises(self, provider, mock_client):
        mock_client.company_earnings.return_value = [
            {"actual": 1.55, "estimate": 1.50, "period": "2026-01-01", "surprise": 0.05}
        ]
        result = provider.get_earnings_surprises("AAPL")
        assert len(result) == 1
        assert result[0]["surprise"] == 0.05

    def test_error_returns_empty(self, provider, mock_client):
        mock_client.company_earnings.side_effect = RuntimeError("err")
        result = provider.get_earnings_surprises("AAPL")
        assert result == []


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


class TestRecommendations:
    def test_returns_trends(self, provider, mock_client):
        mock_client.recommendation_trends.return_value = [
            {"buy": 10, "hold": 5, "sell": 2, "strongBuy": 3, "strongSell": 0}
        ]
        result = provider.get_recommendation_trends("AAPL")
        assert len(result) == 1
        assert result[0]["buy"] == 10

    def test_error_returns_empty(self, provider, mock_client):
        mock_client.recommendation_trends.side_effect = Exception("fail")
        assert provider.get_recommendation_trends("AAPL") == []


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #


class TestOwnership:
    def test_returns_holders(self, provider, mock_client):
        mock_client.ownership.return_value = {
            "ownership": [{"name": "Vanguard", "share": 1000000}]
        }
        result = provider.get_institutional_ownership("AAPL")
        assert "ownership" in result
        assert result["ownership"][0]["name"] == "Vanguard"


# --------------------------------------------------------------------------- #
# Peers
# --------------------------------------------------------------------------- #


class TestPeers:
    def test_returns_peers(self, provider, mock_client):
        mock_client.company_peers.return_value = ["MSFT", "GOOGL", "META"]
        result = provider.get_company_peers("AAPL")
        assert "MSFT" in result


# --------------------------------------------------------------------------- #
# Economic calendar
# --------------------------------------------------------------------------- #


class TestEconomicCalendar:
    def test_returns_events(self, provider, mock_client):
        mock_client.economic_calendar.return_value = {
            "economicCalendar": [{"event": "CPI", "country": "US"}]
        }
        result = provider.get_economic_calendar()
        assert len(result["economicCalendar"]) == 1


# --------------------------------------------------------------------------- #
# Market news
# --------------------------------------------------------------------------- #


class TestMarketNews:
    def test_returns_news(self, provider, mock_client):
        mock_client.general_news.return_value = [
            {"headline": "Markets rally", "source": "CNBC"}
        ]
        result = provider.get_market_news()
        assert len(result) == 1


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


class TestCaching:
    def test_cache_hit(self, provider, mock_client):
        mock_client.company_peers.return_value = ["MSFT"]
        provider.get_company_peers("AAPL")
        provider.get_company_peers("AAPL")  # should hit cache
        assert mock_client.company_peers.call_count == 1

    def test_cache_miss_different_key(self, provider, mock_client):
        mock_client.company_peers.return_value = ["MSFT"]
        provider.get_company_peers("AAPL")
        provider.get_company_peers("TSLA")
        assert mock_client.company_peers.call_count == 2

    def test_cache_expiry(self, provider, mock_client):
        mock_client.recommendation_trends.return_value = [{"buy": 1}]
        provider.get_recommendation_trends("AAPL")

        # Expire the cache entry manually
        for key in list(provider._cache):
            ts, val = provider._cache[key]
            provider._cache[key] = (ts - 999, val)

        provider.get_recommendation_trends("AAPL")
        assert mock_client.recommendation_trends.call_count == 2


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #


class TestGracefulDegradation:
    def test_no_api_key_returns_empty(self, provider_no_key):
        assert provider_no_key.get_company_news("AAPL") == []
        assert provider_no_key.get_earnings_calendar() == {"earningsCalendar": []}
        assert provider_no_key.get_recommendation_trends("AAPL") == []
        assert provider_no_key.get_institutional_ownership("AAPL") == {"ownership": []}
        assert provider_no_key.get_company_peers("AAPL") == []
        assert provider_no_key.get_economic_calendar() == {"economicCalendar": []}
        assert provider_no_key.get_market_news() == []
        assert provider_no_key.get_quote("AAPL") == {}

    def test_api_error_returns_empty(self, provider, mock_client):
        mock_client.quote.side_effect = ConnectionError("no network")
        assert provider.get_quote("AAPL") == {}
