"""On-device financial sentiment analysis using FinBERT.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text.
It classifies text as positive, negative, or neutral with confidence scores.

The model loads lazily on first call (~2s for model download + init) and is
cached as a module-level singleton. Subsequent calls run in <100ms.

Requires optional ``ml`` dependencies::

    uv sync --extra ml   # installs transformers + torch

When transformers/torch are not installed, ``FinBERTAnalyzer.get_instance()``
returns ``None`` and callers should fall back to LLM or keyword-based analysis.

DISCLAIMER: Sentiment analysis results are for educational purposes only
and do not constitute investment advice.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("maverick_mcp.sentiment_analyzer")

# FinBERT label → MaverickMCP sentiment mapping
_LABEL_MAP: dict[str, str] = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
}


class FinBERTAnalyzer:
    """Singleton wrapper around HuggingFace's FinBERT pipeline.

    Usage::

        analyzer = FinBERTAnalyzer.get_instance()
        if analyzer is not None:
            result = analyzer.analyze_sentiment(["AAPL beats earnings"])
    """

    _instance: FinBERTAnalyzer | None = None
    _available: bool | None = None  # tri-state: None = unchecked

    def __init__(self) -> None:
        self._pipeline: Any = None  # transformers.Pipeline, lazy-loaded

    @classmethod
    def get_instance(cls) -> FinBERTAnalyzer | None:
        """Return singleton instance, or ``None`` if deps aren't installed."""
        if cls._available is False:
            return None
        if cls._instance is not None:
            return cls._instance
        try:
            import transformers  # noqa: F401

            cls._instance = cls()
            cls._available = True
            return cls._instance
        except ImportError:
            cls._available = False
            logger.info(
                "transformers not installed — FinBERT unavailable. "
                "Install with: uv sync --extra ml"
            )
            return None

    @classmethod
    def reset(cls) -> None:
        """Reset singleton state (for testing)."""
        cls._instance = None
        cls._available = None

    def _ensure_pipeline(self) -> None:
        """Lazy-load the FinBERT pipeline on first inference call."""
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline

        logger.info("Loading FinBERT model (first call — this takes ~2s)...")
        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU — safe default for desktop
        )
        logger.info("FinBERT model loaded successfully")

    def classify_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of texts into positive/negative/neutral.

        Args:
            texts: List of financial text strings to classify.

        Returns:
            List of dicts, each with ``label`` (str) and ``score`` (float).
        """
        if not texts:
            return []
        self._ensure_pipeline()
        # FinBERT max input is 512 tokens; truncate long texts
        truncated = [t[:512] for t in texts]
        results: list[dict[str, Any]] = self._pipeline(
            truncated, truncation=True, max_length=512
        )
        return results

    def analyze_sentiment(self, texts: list[str]) -> dict[str, Any]:
        """Analyze texts and return aggregated sentiment summary.

        Args:
            texts: List of financial text strings (e.g. headlines).

        Returns:
            Dict with keys:
            - overall_sentiment: "bullish", "bearish", or "neutral"
            - confidence: float 0.0-1.0
            - breakdown: {"positive": int, "negative": int, "neutral": int}
            - per_text: list of per-text results
        """
        if not texts:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "breakdown": {"positive": 0, "negative": 0, "neutral": 0},
                "per_text": [],
            }

        raw_results = self.classify_texts(texts)

        # Count labels
        counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        weighted_scores: dict[str, float] = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }
        per_text: list[dict[str, Any]] = []

        for text, result in zip(texts, raw_results, strict=True):
            label = result["label"].lower()
            score = float(result["score"])
            counts[label] = counts.get(label, 0) + 1
            weighted_scores[label] = weighted_scores.get(label, 0.0) + score
            per_text.append(
                {
                    "text": text[:100],  # truncate for readability
                    "label": _LABEL_MAP.get(label, "neutral"),
                    "score": round(score, 4),
                }
            )

        # Determine overall sentiment by highest count, break ties by score
        total = len(texts)
        dominant_label = max(counts, key=lambda k: (counts[k], weighted_scores[k]))
        overall_sentiment = _LABEL_MAP.get(dominant_label, "neutral")

        # Confidence = proportion of dominant label
        confidence = counts[dominant_label] / total if total > 0 else 0.0

        return {
            "overall_sentiment": overall_sentiment,
            "confidence": round(confidence, 4),
            "breakdown": counts,
            "per_text": per_text,
        }
