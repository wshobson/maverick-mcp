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
    period_days: int
    as_of: str
    portfolio_context: dict[str, Any] | None = None


class CorrelationResult(BaseModel):
    """Advisory-float correlation payload."""

    matrix: dict[str, dict[str, float]]
    high_correlation_pairs: list[dict[str, Any]]
    hedges: list[dict[str, Any]]
    average_correlation: float
    diversification_score: float
    recommendation: str
    period_days: int
    data_points: int
    portfolio_context: dict[str, Any] | None = None


class RiskAnalysis(BaseModel):
    """Advisory-float risk-sizing payload for a single ticker."""

    ticker: str
    current_price: float
    atr: float
    risk_level: float
    position_sizing: dict[str, Any]
    stop_loss: dict[str, Any]
    entry_strategy: dict[str, Any]
    targets: dict[str, Any]
    analysis: dict[str, Any] | None = None
    existing_position: dict[str, Any] | None = None


class PositionExposure(BaseModel):
    """A position's risk-computation inputs: `risk.py`'s pure functions take
    lists of these, never `PositionPayload`/ledger types directly. `sector`
    defaults to "Unknown" because positions do not carry sector data (Phase
    4 decision), matching the legacy risk-dashboard router's hardcoded
    placeholder."""

    symbol: str
    shares: float
    cost_basis: float
    current_price: float
    sector: str = "Unknown"


class RiskDashboard(BaseModel):
    """Advisory-float portfolio risk snapshot: total value, sector
    concentration, parametric VaR (95/99), and total unrealized P&L."""

    total_value: float
    sector_concentration: dict[str, float]
    max_sector_pct: float
    portfolio_var_95: float
    portfolio_var_99: float
    total_pnl: float
    position_count: int


class PositionRiskImpact(BaseModel):
    """The prospective position's own contribution to the projected dashboard."""

    ticker: str
    shares: float
    price: float
    position_value: float
    pct_of_projected_portfolio: float


class PositionRiskCheck(BaseModel):
    """Pre-trade risk check: current vs. projected dashboard after merging
    in a prospective new (or added-to) position."""

    current: RiskDashboard
    projected: RiskDashboard
    new_position: PositionRiskImpact


class RegimeAdjustedSizing(BaseModel):
    """Position size scaled by the detected market regime's risk multiplier."""

    shares: int
    position_value: float
    risk_amount: float
    regime_multiplier: float
    adjusted_risk_pct: float
    regime: str


class RiskAlert(BaseModel):
    """A single risk-threshold breach (concentration/sizing/drawdown), not persisted."""

    alert_type: str
    severity: str
    message: str
    details: dict[str, Any]


class RiskAlertsResult(BaseModel):
    """Service-tier carrier for the alerts tool. `position_count` is not part
    of the legacy tool's JSON payload -- it lets the tool distinguish "no
    positions" (legacy's `status: empty` branch) from "positions present,
    zero alerts" without a second portfolio read."""

    alert_count: int
    alerts: list[RiskAlert]
    position_count: int
