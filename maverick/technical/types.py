"""Technical payload types. Bottom layer: imports nothing from this domain."""

from typing import Any

from pydantic import BaseModel


class RSIAnalysis(BaseModel):
    current: float | None
    period: int
    signal: str
    description: str


class MACDAnalysis(BaseModel):
    macd: float | None
    signal_line: float | None
    histogram: float | None
    indicator_signal: str
    crossover: str
    description: str


class StochasticAnalysis(BaseModel):
    k: float | None
    d: float | None
    signal: str
    crossover: str
    description: str


class BollingerAnalysis(BaseModel):
    upper: float | None
    middle: float | None
    lower: float | None
    current_price: float | None
    position: str
    volatility: str
    description: str


class VolumeAnalysis(BaseModel):
    current: float | None
    average: float | None
    ratio: float | None
    description: str
    signal: str


class TrendAnalysis(BaseModel):
    score: int
    direction: str
    adx: float | None


class LevelsResult(BaseModel):
    support: list[float]
    resistance: list[float]


class FullTechnicalAnalysis(BaseModel):
    ticker: str
    current_price: float
    trend: TrendAnalysis
    outlook: str
    rsi: RSIAnalysis
    macd: MACDAnalysis
    stochastic: StochasticAnalysis
    bollinger: BollingerAnalysis
    volume: VolumeAnalysis
    levels: LevelsResult
    analysis_metadata: dict[str, Any]
