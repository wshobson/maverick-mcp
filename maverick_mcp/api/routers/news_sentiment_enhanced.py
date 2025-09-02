"""
Enhanced news sentiment analysis using Tiingo News API or LLM-based analysis.

This module provides reliable news sentiment analysis by:
1. Using Tiingo's get_news method (if available)
2. Falling back to LLM-based sentiment analysis using existing research tools
3. Never relying on undefined EXTERNAL_DATA_API_KEY
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from tiingo import TiingoClient

from maverick_mcp.api.middleware.mcp_logging import get_tool_logger
from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_tiingo_client() -> TiingoClient | None:
    """Get or create Tiingo client if API key is available."""
    api_key = os.getenv("TIINGO_API_KEY")
    if api_key:
        try:
            config = {"session": True, "api_key": api_key}
            return TiingoClient(config)
        except Exception as e:
            logger.warning(f"Failed to initialize Tiingo client: {e}")
    return None


def get_llm():
    """Get LLM for sentiment analysis (optimized for speed)."""
    from maverick_mcp.providers.llm_factory import get_llm as get_llm_factory
    from maverick_mcp.providers.openrouter_provider import TaskType

    # Use sentiment analysis task type with fast preference
    return get_llm_factory(
        task_type=TaskType.SENTIMENT_ANALYSIS, prefer_fast=True, prefer_cheap=True
    )


async def get_news_sentiment_enhanced(
    ticker: str, timeframe: str = "7d", limit: int = 10
) -> dict[str, Any]:
    """
    Enhanced news sentiment analysis using Tiingo News API or LLM analysis.

    This tool provides reliable sentiment analysis by:
    1. First attempting to use Tiingo's news API (if available)
    2. Analyzing news sentiment using LLM if news is retrieved
    3. Falling back to research-based sentiment if Tiingo unavailable
    4. Providing guaranteed responses with appropriate fallbacks

    Args:
        ticker: Stock ticker symbol
        timeframe: Time frame for news (1d, 7d, 30d, etc.)
        limit: Maximum number of news articles to analyze

    Returns:
        Dictionary containing news sentiment analysis with confidence scores
    """
    tool_logger = get_tool_logger("data_get_news_sentiment_enhanced")
    request_id = str(uuid.uuid4())

    try:
        # Step 1: Try Tiingo News API
        tool_logger.step("tiingo_check", f"Checking Tiingo News API for {ticker}")

        tiingo_client = get_tiingo_client()
        if tiingo_client:
            try:
                # Calculate date range from timeframe
                end_date = datetime.now()
                days = int(timeframe.rstrip("d")) if timeframe.endswith("d") else 7
                start_date = end_date - timedelta(days=days)

                tool_logger.step(
                    "tiingo_fetch", f"Fetching news from Tiingo for {ticker}"
                )

                # Fetch news using Tiingo's get_news method
                news_articles = await asyncio.wait_for(
                    asyncio.to_thread(
                        tiingo_client.get_news,
                        tickers=[ticker],
                        startDate=start_date.strftime("%Y-%m-%d"),
                        endDate=end_date.strftime("%Y-%m-%d"),
                        limit=limit,
                        sortBy="publishedDate",
                        onlyWithTickers=True,
                    ),
                    timeout=10.0,
                )

                if news_articles:
                    tool_logger.step(
                        "llm_analysis",
                        f"Analyzing {len(news_articles)} articles with LLM",
                    )

                    # Analyze sentiment using LLM
                    sentiment_result = await _analyze_news_sentiment_with_llm(
                        news_articles, ticker, tool_logger
                    )

                    tool_logger.complete(
                        f"Tiingo news sentiment analysis completed for {ticker}"
                    )

                    return {
                        "ticker": ticker,
                        "sentiment": sentiment_result["overall_sentiment"],
                        "confidence": sentiment_result["confidence"],
                        "source": "tiingo_news_with_llm_analysis",
                        "status": "success",
                        "analysis": {
                            "articles_analyzed": len(news_articles),
                            "sentiment_breakdown": sentiment_result["breakdown"],
                            "key_themes": sentiment_result["themes"],
                            "recent_headlines": sentiment_result["headlines"][:3],
                        },
                        "timeframe": timeframe,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat(),
                    }

            except TimeoutError:
                tool_logger.step(
                    "tiingo_timeout", "Tiingo API timed out, using fallback"
                )
            except Exception as e:
                # Check if it's a permissions issue (free tier doesn't include news)
                if (
                    "403" in str(e)
                    or "permission" in str(e).lower()
                    or "unauthorized" in str(e).lower()
                ):
                    tool_logger.step(
                        "tiingo_no_permission",
                        "Tiingo news not available (requires paid plan)",
                    )
                else:
                    tool_logger.step("tiingo_error", f"Tiingo error: {str(e)}")

        # Step 2: Fallback to research-based sentiment
        tool_logger.step("research_fallback", "Using research-based sentiment analysis")

        from maverick_mcp.api.routers.research import analyze_market_sentiment

        # Use research tools to gather sentiment
        result = await asyncio.wait_for(
            analyze_market_sentiment(
                topic=f"{ticker} stock news sentiment recent {timeframe}",
                timeframe="1w" if days <= 7 else "1m",
                persona="moderate",
            ),
            timeout=15.0,
        )

        if result.get("success", False):
            sentiment_data = result.get("sentiment_analysis", {})
            return {
                "ticker": ticker,
                "sentiment": _extract_sentiment_from_research(sentiment_data),
                "confidence": sentiment_data.get("sentiment_confidence", 0.5),
                "source": "research_based_sentiment",
                "status": "fallback_success",
                "analysis": {
                    "overall_sentiment": sentiment_data.get("overall_sentiment", {}),
                    "key_themes": sentiment_data.get("sentiment_themes", [])[:3],
                    "market_insights": sentiment_data.get("market_insights", [])[:2],
                },
                "timeframe": timeframe,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Using research-based sentiment (Tiingo news unavailable on free tier)",
            }

        # Step 3: Basic fallback
        return _provide_basic_sentiment_fallback(ticker, request_id)

    except Exception as e:
        tool_logger.error("sentiment_error", e, f"Sentiment analysis failed: {str(e)}")
        return _provide_basic_sentiment_fallback(ticker, request_id, str(e))


async def _analyze_news_sentiment_with_llm(
    news_articles: list, ticker: str, tool_logger
) -> dict[str, Any]:
    """Analyze news articles sentiment using LLM."""

    llm = get_llm()
    if not llm:
        # No LLM available, do basic analysis
        return _basic_news_analysis(news_articles)

    try:
        # Prepare news summary for LLM
        news_summary = []
        for article in news_articles[:10]:  # Limit to 10 most recent
            news_summary.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", "")[:200]
                    if article.get("description")
                    else "",
                    "publishedDate": article.get("publishedDate", ""),
                    "source": article.get("source", ""),
                }
            )

        # Create sentiment analysis prompt
        prompt = f"""Analyze the sentiment of these recent news articles about {ticker} stock.

News Articles:
{chr(10).join([f"- {a['title']} ({a['source']}, {a['publishedDate'][:10] if a['publishedDate'] else 'Unknown date'})" for a in news_summary[:5]])}

Provide a JSON response with:
1. overall_sentiment: "bullish", "bearish", or "neutral"
2. confidence: 0.0 to 1.0
3. breakdown: dict with counts of positive, negative, neutral articles
4. themes: list of 3 key themes from the news
5. headlines: list of 3 most important headlines

Response format:
{{"overall_sentiment": "...", "confidence": 0.X, "breakdown": {{"positive": X, "negative": Y, "neutral": Z}}, "themes": ["...", "...", "..."], "headlines": ["...", "...", "..."]}}"""

        # Get LLM analysis
        response = await asyncio.to_thread(lambda: llm.invoke(prompt).content)

        # Parse JSON response
        import json

        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            elif "{" in response:
                # Find JSON object in response
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            result = json.loads(json_str)

            # Ensure all required fields
            return {
                "overall_sentiment": result.get("overall_sentiment", "neutral"),
                "confidence": float(result.get("confidence", 0.5)),
                "breakdown": result.get(
                    "breakdown",
                    {"positive": 0, "negative": 0, "neutral": len(news_articles)},
                ),
                "themes": result.get(
                    "themes",
                    ["Market movement", "Company performance", "Industry trends"],
                ),
                "headlines": [a.get("title", "") for a in news_summary[:3]],
            }

        except (json.JSONDecodeError, ValueError) as e:
            tool_logger.step("llm_parse_error", f"Failed to parse LLM response: {e}")
            return _basic_news_analysis(news_articles)

    except Exception as e:
        tool_logger.step("llm_error", f"LLM analysis failed: {e}")
        return _basic_news_analysis(news_articles)


def _basic_news_analysis(news_articles: list) -> dict[str, Any]:
    """Basic sentiment analysis without LLM."""

    # Simple keyword-based sentiment
    positive_keywords = [
        "gain",
        "rise",
        "up",
        "beat",
        "exceed",
        "strong",
        "bull",
        "buy",
        "upgrade",
        "positive",
    ]
    negative_keywords = [
        "loss",
        "fall",
        "down",
        "miss",
        "below",
        "weak",
        "bear",
        "sell",
        "downgrade",
        "negative",
    ]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for article in news_articles:
        title = (
            article.get("title", "") + " " + article.get("description", "")
        ).lower()

        pos_score = sum(1 for keyword in positive_keywords if keyword in title)
        neg_score = sum(1 for keyword in negative_keywords if keyword in title)

        if pos_score > neg_score:
            positive_count += 1
        elif neg_score > pos_score:
            negative_count += 1
        else:
            neutral_count += 1

    total = len(news_articles)
    if total == 0:
        return {
            "overall_sentiment": "neutral",
            "confidence": 0.0,
            "breakdown": {"positive": 0, "negative": 0, "neutral": 0},
            "themes": [],
            "headlines": [],
        }

    # Determine overall sentiment
    if positive_count > negative_count * 1.5:
        overall = "bullish"
    elif negative_count > positive_count * 1.5:
        overall = "bearish"
    else:
        overall = "neutral"

    # Calculate confidence based on consensus
    max_count = max(positive_count, negative_count, neutral_count)
    confidence = max_count / total if total > 0 else 0.0

    return {
        "overall_sentiment": overall,
        "confidence": confidence,
        "breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
        },
        "themes": ["Recent news", "Market activity", "Company updates"],
        "headlines": [a.get("title", "") for a in news_articles[:3]],
    }


def _extract_sentiment_from_research(sentiment_data: dict) -> str:
    """Extract simple sentiment direction from research data."""

    overall = sentiment_data.get("overall_sentiment", {})

    # Check for sentiment keywords
    if isinstance(overall, dict):
        sentiment_str = str(overall).lower()
    else:
        sentiment_str = str(overall).lower()

    if "bullish" in sentiment_str or "positive" in sentiment_str:
        return "bullish"
    elif "bearish" in sentiment_str or "negative" in sentiment_str:
        return "bearish"

    # Check confidence for direction
    confidence = sentiment_data.get("sentiment_confidence", 0.5)
    if confidence > 0.6:
        return "bullish"
    elif confidence < 0.4:
        return "bearish"

    return "neutral"


def _provide_basic_sentiment_fallback(
    ticker: str, request_id: str, error_detail: str = None
) -> dict[str, Any]:
    """Provide basic fallback when all methods fail."""

    response = {
        "ticker": ticker,
        "sentiment": "neutral",
        "confidence": 0.0,
        "source": "fallback",
        "status": "all_methods_failed",
        "message": "Unable to fetch news sentiment - returning neutral baseline",
        "analysis": {
            "note": "Consider using a paid Tiingo plan for news access or check API keys"
        },
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
    }

    if error_detail:
        response["error_detail"] = error_detail[:200]  # Limit error message length

    return response
