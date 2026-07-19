"""Screening payload types. Bottom layer: imports nothing from this domain."""

from typing import Literal

from pydantic import BaseModel

ScreenName = Literal["bullish", "bearish", "supply_demand"]


class ScreeningResult(BaseModel):
    symbol: str
    screen: ScreenName
    date_analyzed: str
    close: float
    combined_score: int
    momentum_score: float | None
    indicators: dict[str, float | None]
    flags: dict[str, bool]
    reason: str


class AllScreeningResults(BaseModel):
    bullish: list[ScreeningResult]
    bearish: list[ScreeningResult]
    supply_demand: list[ScreeningResult]
    date_analyzed: str | None = None


class ScreenRun(BaseModel):
    screen: ScreenName
    symbols_screened: int
    symbols_qualified: int
    symbols_failed: int
    date_analyzed: str
    duration_seconds: float


class ScreeningCriteria(BaseModel):
    min_momentum_score: float | None = None
    min_volume: int | None = None
    max_price: float | None = None
    min_combined_score: int | None = None
