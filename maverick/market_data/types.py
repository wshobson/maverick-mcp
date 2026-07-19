"""Market data payload types. Bottom layer: imports nothing from this domain."""

from pydantic import BaseModel

PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class Quote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str


class Mover(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int


class IndexQuote(BaseModel):
    name: str
    symbol: str
    price: float
    change: float
    change_percent: float


SectorPerformance = dict[str, float]


class Volatility(BaseModel):
    vix: float | None
    vix_change_percent: float | None
    fear_level: str


class MarketOverview(BaseModel):
    indices: dict[str, IndexQuote]
    sectors: SectorPerformance
    top_gainers: list[Mover]
    top_losers: list[Mover]
    volatility: Volatility
    last_updated: str


class CompanyInfo(BaseModel):
    name: str | None
    sector: str | None
    industry: str | None
    website: str | None
    description: str | None


class MarketNumbers(BaseModel):
    current_price: float | None
    market_cap: float | None
    enterprise_value: float | None
    shares_outstanding: float | None
    float_shares: float | None


class TradingStats(BaseModel):
    avg_volume: float | None
    avg_volume_10d: float | None
    beta: float | None
    week_52_high: float | None
    week_52_low: float | None


class Fundamentals(BaseModel):
    symbol: str
    company: CompanyInfo
    market_data: MarketNumbers
    valuation: dict[str, float | None]
    financials: dict[str, float | None]
    trading: TradingStats


def fear_level_from_vix(vix: float | None) -> str:
    """Map a VIX level to a fear-level label."""
    if vix is None:
        return "unknown"
    if vix < 20:
        return "low"
    if vix < 30:
        return "elevated"
    return "high"
