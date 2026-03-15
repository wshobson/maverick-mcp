"""Tests for the FinBERT-based sentiment analyzer."""

from unittest.mock import MagicMock, patch

import pytest

from maverick_mcp.core.sentiment_analyzer import _LABEL_MAP, FinBERTAnalyzer


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the FinBERTAnalyzer singleton before each test."""
    FinBERTAnalyzer.reset()
    yield
    FinBERTAnalyzer.reset()


# --- Singleton / graceful degradation ---


class TestFinBERTAvailability:
    def test_returns_none_when_transformers_missing(self):
        with patch.dict("sys.modules", {"transformers": None}):
            # Force re-check by resetting
            FinBERTAnalyzer.reset()
            instance = FinBERTAnalyzer.get_instance()
            assert instance is None

    def test_returns_instance_when_transformers_available(self):
        with patch(
            "maverick_mcp.core.sentiment_analyzer.FinBERTAnalyzer._available", None
        ):
            mock_transformers = MagicMock()
            with patch.dict("sys.modules", {"transformers": mock_transformers}):
                FinBERTAnalyzer.reset()
                instance = FinBERTAnalyzer.get_instance()
                assert instance is not None
                assert isinstance(instance, FinBERTAnalyzer)

    def test_singleton_returns_same_instance(self):
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            FinBERTAnalyzer.reset()
            a = FinBERTAnalyzer.get_instance()
            b = FinBERTAnalyzer.get_instance()
            assert a is b

    def test_reset_clears_state(self):
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            FinBERTAnalyzer.reset()
            _ = FinBERTAnalyzer.get_instance()
            FinBERTAnalyzer.reset()
            assert FinBERTAnalyzer._instance is None
            assert FinBERTAnalyzer._available is None


# --- Label mapping ---


class TestLabelMapping:
    def test_positive_maps_to_bullish(self):
        assert _LABEL_MAP["positive"] == "bullish"

    def test_negative_maps_to_bearish(self):
        assert _LABEL_MAP["negative"] == "bearish"

    def test_neutral_maps_to_neutral(self):
        assert _LABEL_MAP["neutral"] == "neutral"


# --- classify_texts ---


class TestClassifyTexts:
    def _make_analyzer_with_mock_pipeline(self, mock_results):
        """Create an analyzer with a mocked pipeline."""
        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = MagicMock(return_value=mock_results)
        return analyzer

    def test_empty_input_returns_empty(self):
        analyzer = FinBERTAnalyzer()
        assert analyzer.classify_texts([]) == []

    def test_passes_texts_to_pipeline(self):
        mock_pipeline = MagicMock(return_value=[{"label": "positive", "score": 0.95}])
        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = mock_pipeline

        result = analyzer.classify_texts(["AAPL beats earnings"])
        mock_pipeline.assert_called_once_with(
            ["AAPL beats earnings"], truncation=True, max_length=512
        )
        assert result == [{"label": "positive", "score": 0.95}]

    def test_truncates_long_texts(self):
        mock_pipeline = MagicMock(return_value=[{"label": "neutral", "score": 0.5}])
        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = mock_pipeline

        long_text = "x" * 1000
        analyzer.classify_texts([long_text])

        # Verify the text was truncated to 512 chars
        call_args = mock_pipeline.call_args[0][0]
        assert len(call_args[0]) == 512


# --- analyze_sentiment ---


class TestAnalyzeSentiment:
    def _make_analyzer(self, pipeline_results):
        """Create analyzer with mock pipeline returning given results."""
        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = MagicMock(return_value=pipeline_results)
        return analyzer

    def test_empty_texts_returns_neutral(self):
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze_sentiment([])
        assert result["overall_sentiment"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["breakdown"] == {"positive": 0, "negative": 0, "neutral": 0}
        assert result["per_text"] == []

    def test_all_positive(self):
        pipeline_results = [
            {"label": "positive", "score": 0.95},
            {"label": "positive", "score": 0.88},
            {"label": "positive", "score": 0.92},
        ]
        analyzer = self._make_analyzer(pipeline_results)
        texts = ["Good earnings", "Strong growth", "Beat expectations"]

        result = analyzer.analyze_sentiment(texts)
        assert result["overall_sentiment"] == "bullish"
        assert result["confidence"] == 1.0
        assert result["breakdown"]["positive"] == 3
        assert result["breakdown"]["negative"] == 0

    def test_all_negative(self):
        pipeline_results = [
            {"label": "negative", "score": 0.90},
            {"label": "negative", "score": 0.85},
        ]
        analyzer = self._make_analyzer(pipeline_results)
        texts = ["Revenue miss", "Downgrade issued"]

        result = analyzer.analyze_sentiment(texts)
        assert result["overall_sentiment"] == "bearish"
        assert result["confidence"] == 1.0
        assert result["breakdown"]["negative"] == 2

    def test_mixed_sentiment(self):
        pipeline_results = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.8},
            {"label": "neutral", "score": 0.7},
        ]
        analyzer = self._make_analyzer(pipeline_results)
        texts = ["Good news", "Bad news", "Normal day"]

        result = analyzer.analyze_sentiment(texts)
        # All equal counts — dominant determined by highest weighted score
        assert result["overall_sentiment"] in ("bullish", "bearish", "neutral")
        assert 0.0 < result["confidence"] <= 1.0
        assert sum(result["breakdown"].values()) == 3

    def test_per_text_results_included(self):
        pipeline_results = [
            {"label": "positive", "score": 0.95},
        ]
        analyzer = self._make_analyzer(pipeline_results)

        result = analyzer.analyze_sentiment(["AAPL beats Q4 estimates"])
        assert len(result["per_text"]) == 1
        assert result["per_text"][0]["label"] == "bullish"
        assert result["per_text"][0]["score"] == 0.95

    def test_per_text_truncates_long_text(self):
        pipeline_results = [
            {"label": "neutral", "score": 0.5},
        ]
        analyzer = self._make_analyzer(pipeline_results)
        long_text = "A" * 200

        result = analyzer.analyze_sentiment([long_text])
        # Per-text display truncated to 100 chars
        assert len(result["per_text"][0]["text"]) == 100

    def test_confidence_proportional_to_consensus(self):
        # 3 out of 4 are positive → confidence = 0.75
        pipeline_results = [
            {"label": "positive", "score": 0.9},
            {"label": "positive", "score": 0.85},
            {"label": "positive", "score": 0.88},
            {"label": "negative", "score": 0.7},
        ]
        analyzer = self._make_analyzer(pipeline_results)
        texts = ["Good", "Great", "Better", "Worse"]

        result = analyzer.analyze_sentiment(texts)
        assert result["overall_sentiment"] == "bullish"
        assert result["confidence"] == 0.75
