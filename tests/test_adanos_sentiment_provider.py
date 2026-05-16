from maverick_mcp.api.routers.data import get_adanos_market_sentiment
from maverick_mcp.providers.adanos_sentiment import AdanosSentimentProvider


class MockResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_provider_reports_not_configured_without_api_key(monkeypatch):
    monkeypatch.delenv("ADANOS_API_KEY", raising=False)

    result = AdanosSentimentProvider().get_sentiment(ticker="AAPL")

    assert result["status"] == "not_configured"
    assert result["provider"] == "adanos"


def test_provider_fetches_selected_sources(monkeypatch):
    calls = []

    def fake_get(url, headers, params, timeout):
        calls.append(
            {"url": url, "headers": headers, "params": params, "timeout": timeout}
        )
        return MockResponse(payload={"sentiment_score": 0.42})

    monkeypatch.setattr("requests.get", fake_get)

    provider = AdanosSentimentProvider(
        api_key="test-key",
        base_url="https://api.example.test/",
        timeout_seconds=3,
    )
    result = provider.get_sentiment(ticker="msft", days=14, sources=["reddit", "news"])

    assert result["status"] == "success"
    assert result["ticker"] == "MSFT"
    assert list(result["sources"]) == ["reddit", "news"]
    assert calls == [
        {
            "url": "https://api.example.test/reddit/stocks/v1/stock/MSFT",
            "headers": {"X-API-Key": "test-key"},
            "params": {"days": 14},
            "timeout": 3,
        },
        {
            "url": "https://api.example.test/stock/MSFT",
            "headers": {"X-API-Key": "test-key"},
            "params": {"days": 14},
            "timeout": 3,
        },
    ]


def test_provider_fetches_market_sentiment(monkeypatch):
    requested_urls = []

    def fake_get(url, headers, params, timeout):
        requested_urls.append(url)
        return MockResponse(payload={"buzz_score": 55})

    monkeypatch.setattr("requests.get", fake_get)

    provider = AdanosSentimentProvider(api_key="test-key")
    result = provider.get_sentiment(sources=["polymarket"], days=3)

    assert result["ticker"] is None
    assert result["sources"]["polymarket"]["buzz_score"] == 55
    assert requested_urls == [
        "https://api.adanos.org/polymarket/stocks/v1/market-sentiment"
    ]


def test_router_returns_invalid_request_for_unknown_source(monkeypatch):
    monkeypatch.setenv("ADANOS_API_KEY", "test-key")

    result = get_adanos_market_sentiment(
        ticker="AAPL",
        sources=["reddit", "unknown"],
    )

    assert result["status"] == "invalid_request"
    assert "unknown" in result["error"]


def test_provider_does_not_leak_environment_key(monkeypatch):
    def fake_get(url, headers, params, timeout):
        return MockResponse(status_code=401)

    monkeypatch.setenv("ADANOS_API_KEY", "secret-test-key")
    monkeypatch.setattr("requests.get", fake_get)

    provider = AdanosSentimentProvider()
    result = provider.get_sentiment(ticker="AAPL")

    assert "secret-test-key" not in str(result)
