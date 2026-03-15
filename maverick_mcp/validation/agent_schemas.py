"""Pydantic schemas for structured agent output.

These models define the JSON structure returned by agent tools when
structured_output=True is set, making agent output composable with other tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MarketSignal(BaseModel):
    """A single actionable market signal from agent analysis."""

    ticker: str = Field(description="Stock ticker symbol")
    direction: Literal["long", "short", "neutral"] = Field(
        description="Recommended position direction"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence score 0-1")
    rationale: str = Field(description="Brief reasoning for this signal")


class StructuredMarketAnalysis(BaseModel):
    """Structured output from market analysis agent."""

    signals: list[MarketSignal] = Field(
        default_factory=list, description="Actionable trading signals"
    )
    risk_level: Literal["low", "moderate", "high", "extreme"] = Field(
        default="moderate", description="Overall market risk level"
    )
    regime: Literal["bull", "bear", "sideways"] = Field(
        default="sideways", description="Detected market regime"
    )
    top_picks: list[str] = Field(
        default_factory=list, description="Top ticker picks from analysis"
    )
    summary: str = Field(default="", description="Human-readable analysis summary")


STRUCTURED_OUTPUT_PROMPT = """
You MUST respond with valid JSON matching this exact schema:
{
  "signals": [{"ticker": "AAPL", "direction": "long|short|neutral", "confidence": 0.0-1.0, "rationale": "..."}],
  "risk_level": "low|moderate|high|extreme",
  "regime": "bull|bear|sideways",
  "top_picks": ["AAPL", "MSFT"],
  "summary": "Brief analysis summary"
}

Rules:
- signals: List of specific actionable signals with tickers
- risk_level: Overall market risk assessment
- regime: Current market regime detection
- top_picks: Top 3-5 ticker symbols from your analysis
- summary: 1-2 sentence summary

Respond ONLY with the JSON object, no markdown or extra text.
"""
