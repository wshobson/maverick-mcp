"""Tests for the enhanced news sentiment router."""

from unittest.mock import MagicMock, patch

from maverick_mcp.api.routers.news_sentiment_enhanced import (
    _basic_news_analysis,
    _extract_sentiment_from_research,
    _extract_themes_from_articles,
    _provide_basic_sentiment_fallback,
)


class TestExtractSentimentFromResearch:
    def test_bullish(self):
        data = {"overall_sentiment": {"direction": "bullish"}}
        assert _extract_sentiment_from_research(data) == "bullish"

    def test_bearish(self):
        data = {"overall_sentiment": "negative trend"}
        assert _extract_sentiment_from_research(data) == "bearish"

    def test_neutral_default(self):
        data = {"overall_sentiment": "mixed signals", "sentiment_confidence": 0.5}
        assert _extract_sentiment_from_research(data) == "neutral"

    def test_high_confidence_bullish(self):
        data = {"overall_sentiment": "mixed", "sentiment_confidence": 0.75}
        assert _extract_sentiment_from_research(data) == "bullish"

    def test_low_confidence_bearish(self):
        data = {"overall_sentiment": "mixed", "sentiment_confidence": 0.3}
        assert _extract_sentiment_from_research(data) == "bearish"


class TestExtractThemesFromArticles:
    def test_with_sources(self):
        articles = [
            {"source": "Reuters"},
            {"source": "Bloomberg"},
            {"source": "CNBC"},
            {"source": "Reuters"},  # duplicate
        ]
        themes = _extract_themes_from_articles(articles)
        assert len(themes) == 3
        assert "Reuters" in themes

    def test_empty_articles(self):
        themes = _extract_themes_from_articles([])
        assert themes == ["Market activity", "Company news", "Industry trends"]

    def test_no_sources(self):
        articles = [{"title": "headline"}]
        themes = _extract_themes_from_articles(articles)
        assert themes == ["Market activity", "Company news", "Industry trends"]


class TestBasicNewsAnalysis:
    def test_positive_articles(self):
        articles = [
            {"title": "Stock gains strong momentum", "description": "Prices rise"},
            {"title": "Beating expectations", "description": "Strong buy signal"},
        ]
        with patch(
            "maverick_mcp.api.routers.news_sentiment_enhanced.FinBERTAnalyzer"
        ) as mock_cls:
            mock_cls.get_instance.return_value = None
            result = _basic_news_analysis(articles)
            assert result["overall_sentiment"] == "bullish"
            assert result["breakdown"]["positive"] == 2

    def test_negative_articles(self):
        articles = [
            {"title": "Stock falls sharply", "description": "Bear market weakness"},
            {"title": "Sell downgrade loss", "description": "Negative outlook"},
            {"title": "Company misses targets", "description": "Below expectations"},
        ]
        with patch(
            "maverick_mcp.api.routers.news_sentiment_enhanced.FinBERTAnalyzer"
        ) as mock_cls:
            mock_cls.get_instance.return_value = None
            result = _basic_news_analysis(articles)
            assert result["overall_sentiment"] == "bearish"
            assert result["breakdown"]["negative"] == 3

    def test_empty_articles(self):
        with patch(
            "maverick_mcp.api.routers.news_sentiment_enhanced.FinBERTAnalyzer"
        ) as mock_cls:
            mock_cls.get_instance.return_value = None
            result = _basic_news_analysis([])
            assert result["overall_sentiment"] == "neutral"
            assert result["confidence"] == 0.0

    def test_finbert_used_when_available(self):
        articles = [{"title": "Test", "description": "Article"}]
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_sentiment.return_value = {
            "overall_sentiment": "bullish",
            "confidence": 0.9,
            "breakdown": {"positive": 1, "negative": 0, "neutral": 0},
        }
        with patch(
            "maverick_mcp.api.routers.news_sentiment_enhanced.FinBERTAnalyzer"
        ) as mock_cls:
            mock_cls.get_instance.return_value = mock_analyzer
            result = _basic_news_analysis(articles)
            assert result["overall_sentiment"] == "bullish"
            mock_analyzer.analyze_sentiment.assert_called_once()


class TestProvideBasicSentimentFallback:
    def test_basic(self):
        result = _provide_basic_sentiment_fallback("AAPL", "req-123")
        assert result["ticker"] == "AAPL"
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["status"] == "all_methods_failed"
        assert result["request_id"] == "req-123"

    def test_with_error_detail(self):
        result = _provide_basic_sentiment_fallback("AAPL", "req-123", "API timeout")
        assert result["error_detail"] == "API timeout"

    def test_long_error_truncated(self):
        long_error = "x" * 500
        result = _provide_basic_sentiment_fallback("AAPL", "req-123", long_error)
        assert len(result["error_detail"]) == 200
