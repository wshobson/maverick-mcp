"""Portfolio payload types. Bottom layer: imports nothing from this domain."""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class PositionPayload(BaseModel):
    ticker: str
    shares: Decimal = Field(gt=0)
    average_cost_basis: Decimal = Field(gt=0)
    total_cost: Decimal = Field(gt=0)
    purchase_date: str
    notes: str | None = None


class PositionWithPrice(PositionPayload):
    current_price: float | None
    current_value: float | None
    unrealized_pnl: float | None
    unrealized_pnl_percent: float | None


class PortfolioMetrics(BaseModel):
    total_invested: Decimal
    total_value: float | None
    total_pnl: float | None
    total_pnl_percent: float | None
    position_count: int


class PortfolioSnapshot(BaseModel):
    user_id: str
    name: str
    positions: list[PositionWithPrice]
    metrics: PortfolioMetrics
    as_of: str


class RemoveResult(BaseModel):
    ticker: str
    shares_removed: Decimal
    position_fully_closed: bool


class ComparisonResult(BaseModel):
    """Advisory-float comparison payload. Per-ticker metrics (performance,
    technical, volume, rankings) are nested dicts, not sub-models."""

    comparison: dict[str, dict[str, Any]]
    best_performer: str
    strongest_trend: str
    portfolio_context: dict[str, Any] | None = None


class CorrelationResult(BaseModel):
    """Advisory-float correlation payload."""

    matrix: dict[str, dict[str, float]]
    high_correlation_pairs: list[dict[str, Any]]
    hedges: list[dict[str, Any]]
    average_correlation: float
    diversification_score: float


class RiskAnalysis(BaseModel):
    """Advisory-float risk-sizing payload for a single ticker."""

    ticker: str
    position_sizing: dict[str, Any]
    stop_loss: dict[str, Any]
    entry_strategy: dict[str, Any]
    targets: dict[str, Any]
    existing_position: dict[str, Any] | None = None
