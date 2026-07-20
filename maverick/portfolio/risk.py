"""Portfolio risk-dashboard pure computation: exposure aggregation,
parametric VaR, pre-trade position risk checks, regime-adjusted sizing,
market regime classification, and alert thresholds.

Sibling of `analysis.py` at the same layer, but strictly pure: unlike
`analysis.py` (which is injected `MarketDataService` and does I/O), every
function here takes already-resolved primitives (`PositionExposure` lists,
an already-fetched price `Series`) and returns typed results synchronously.
The service tier owns all I/O -- reading positions (via its own reads),
fetching current prices and SPY history (via `MarketDataService`) -- and
calls the functions below with the results.

Ported faithfully from the legacy `maverick_mcp.services.risk.service.
RiskService` (dashboard / position-risk-check / regime-adjusted-sizing /
alerts) and `maverick_mcp.services.signals.regime.RegimeDetector`
(market-regime classification, consumed by the sizing tool). All outputs
are advisory floats, matching legacy and the existing `analysis.py`
outputs -- positions feeding this module have already round-tripped
through the Decimal ledger upstream in the service tier.

Sector: portfolio positions do not carry sector data (Phase 4 decision:
"sector info not stored on position"), so every `PositionExposure.sector`
is "Unknown" in practice, exactly as the legacy router hardcoded it.
"""

import math
from typing import Any

import pandas as pd

from maverick.portfolio.config import PortfolioSettings
from maverick.portfolio.types import (
    PositionExposure,
    PositionRiskCheck,
    PositionRiskImpact,
    RegimeAdjustedSizing,
    RiskAlert,
    RiskDashboard,
)

# ---------------------------------------------------------------------------
# Regime classification rubric: RegimeDetector's internal per-factor weights
# and windows. Never configurable in the legacy service (only `vix_level`
# and `breadth_ratio` were caller-supplied), so these stay module constants
# here too -- mirroring `analysis.py`'s `_ATR_PERIOD`.
# ---------------------------------------------------------------------------

_WEIGHT_TREND = 0.35
_WEIGHT_VOLATILITY = 0.25
_WEIGHT_MOMENTUM = 0.25
_WEIGHT_BREADTH = 0.15
_TRANSITIONAL_THRESHOLD = 0.45
_SHORT_WINDOW = 20
_LONG_WINDOW = 50
_MOMENTUM_WINDOW = 10


# ---------------------------------------------------------------------------
# Dashboard / exposure aggregation
# ---------------------------------------------------------------------------


def _estimate_portfolio_std(
    positions: list[PositionExposure],
    total_value: float,
    daily_vol_per_position: float,
) -> float:
    """Simplified parametric-VaR std: flat per-position daily vol assumed,
    zero cross-position correlation assumed."""
    if total_value <= 0 or not positions:
        return daily_vol_per_position

    variance = 0.0
    for pos in positions:
        weight = pos.shares * pos.current_price / total_value
        variance += (weight * daily_vol_per_position) ** 2
    return math.sqrt(variance)


def compute_dashboard(
    positions: list[PositionExposure], settings: PortfolioSettings
) -> RiskDashboard:
    """Total value, sector concentration, parametric VaR (95/99), total P&L."""
    if not positions:
        return RiskDashboard(
            total_value=0.0,
            sector_concentration={},
            max_sector_pct=0.0,
            portfolio_var_95=0.0,
            portfolio_var_99=0.0,
            total_pnl=0.0,
            position_count=0,
        )

    total_value = sum(p.shares * p.current_price for p in positions)
    total_pnl = sum((p.current_price - p.cost_basis) * p.shares for p in positions)

    sector_values: dict[str, float] = {}
    for pos in positions:
        sector_values[pos.sector] = (
            sector_values.get(pos.sector, 0.0) + pos.shares * pos.current_price
        )

    sector_concentration = (
        {sector: value / total_value for sector, value in sector_values.items()}
        if total_value > 0
        else {}
    )
    max_sector_pct = max(sector_concentration.values(), default=0.0)

    portfolio_std = _estimate_portfolio_std(
        positions, total_value, settings.risk_daily_vol_per_position
    )
    portfolio_var_95 = settings.risk_var_z_95 * portfolio_std * total_value
    portfolio_var_99 = settings.risk_var_z_99 * portfolio_std * total_value

    return RiskDashboard(
        total_value=round(total_value, 2),
        sector_concentration={k: round(v, 4) for k, v in sector_concentration.items()},
        max_sector_pct=round(max_sector_pct, 4),
        portfolio_var_95=round(portfolio_var_95, 2),
        portfolio_var_99=round(portfolio_var_99, 2),
        total_pnl=round(total_pnl, 2),
        position_count=len(positions),
    )


# ---------------------------------------------------------------------------
# Pre-trade position risk check
# ---------------------------------------------------------------------------


def check_position_risk(
    positions: list[PositionExposure],
    new_symbol: str,
    new_shares: float,
    new_price: float,
    settings: PortfolioSettings,
) -> PositionRiskCheck:
    """Current vs. projected dashboard after merging in a prospective new
    (or added-to, if `new_symbol` is already held) position."""
    current = compute_dashboard(positions, settings)

    normalized_symbol = new_symbol.upper()
    merged: list[PositionExposure] = []
    found = False
    for pos in positions:
        if pos.symbol.upper() == normalized_symbol:
            total_shares = pos.shares + new_shares
            avg_cost = (
                (pos.shares * pos.cost_basis + new_shares * new_price) / total_shares
                if total_shares > 0
                else new_price
            )
            merged.append(
                pos.model_copy(
                    update={
                        "shares": total_shares,
                        "cost_basis": avg_cost,
                        "current_price": new_price,
                    }
                )
            )
            found = True
        else:
            merged.append(pos)
    if not found:
        merged.append(
            PositionExposure(
                symbol=normalized_symbol,
                shares=new_shares,
                cost_basis=new_price,
                current_price=new_price,
                sector="Unknown",
            )
        )

    projected = compute_dashboard(merged, settings)
    position_value = new_shares * new_price

    return PositionRiskCheck(
        current=current,
        projected=projected,
        new_position=PositionRiskImpact(
            ticker=normalized_symbol,
            shares=new_shares,
            price=new_price,
            position_value=round(position_value, 2),
            pct_of_projected_portfolio=(
                round(position_value / projected.total_value, 4)
                if projected.total_value > 0
                else 0.0
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Regime-adjusted position sizing
# ---------------------------------------------------------------------------


def regime_adjusted_size(
    account_size: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float,
    regime: str,
    settings: PortfolioSettings,
) -> RegimeAdjustedSizing:
    """Position size scaled by `regime`'s risk multiplier (bull = full risk,
    choppy/transitional = 75%, bear = 50%, per legacy defaults)."""
    multiplier = settings.risk_regime_multipliers.get(regime.lower(), 1.0)
    adjusted_risk_pct = risk_pct * multiplier
    risk_amount = account_size * (adjusted_risk_pct / 100.0)

    risk_per_share = abs(entry_price - stop_loss)
    shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    position_value = shares * entry_price

    return RegimeAdjustedSizing(
        shares=shares,
        position_value=round(position_value, 2),
        risk_amount=round(risk_amount, 2),
        regime_multiplier=multiplier,
        adjusted_risk_pct=round(adjusted_risk_pct, 4),
        regime=regime.lower(),
    )


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def generate_alerts(
    positions: list[PositionExposure], settings: PortfolioSettings
) -> list[RiskAlert]:
    """Sector concentration, oversized-position, and drawdown threshold
    breaches. Not persisted (matches legacy: alerts were computed on demand,
    never written to the DB despite the legacy module defining a table)."""
    dashboard = compute_dashboard(positions, settings)
    alerts: list[RiskAlert] = []
    total_value = dashboard.total_value

    for sector, pct in dashboard.sector_concentration.items():
        if pct > settings.risk_sector_critical_pct:
            alerts.append(
                RiskAlert(
                    alert_type="concentration",
                    severity="critical",
                    message=(
                        f"Sector '{sector}' represents {pct:.1%} of portfolio "
                        f"(threshold: {settings.risk_sector_critical_pct:.0%})"
                    ),
                    details={"sector": sector, "pct": pct},
                )
            )
        elif pct > settings.risk_sector_warn_pct:
            alerts.append(
                RiskAlert(
                    alert_type="concentration",
                    severity="warning",
                    message=(
                        f"Sector '{sector}' represents {pct:.1%} of portfolio "
                        f"(threshold: {settings.risk_sector_warn_pct:.0%})"
                    ),
                    details={"sector": sector, "pct": pct},
                )
            )

    if total_value > 0:
        for pos in positions:
            pos_value = pos.shares * pos.current_price
            pos_pct = pos_value / total_value
            if pos_pct > settings.risk_position_warn_pct:
                alerts.append(
                    RiskAlert(
                        alert_type="sizing",
                        severity="warning",
                        message=(
                            f"Position '{pos.symbol}' is {pos_pct:.1%} of portfolio "
                            f"(threshold: {settings.risk_position_warn_pct:.0%})"
                        ),
                        details={"ticker": pos.symbol, "pct": pos_pct},
                    )
                )

    total_cost = sum(p.cost_basis * p.shares for p in positions)
    if total_cost > 0:
        loss_pct = (total_cost - total_value) / total_cost
        if loss_pct > settings.risk_portfolio_loss_warn_pct:
            alerts.append(
                RiskAlert(
                    alert_type="drawdown",
                    severity="warning",
                    message=(
                        f"Portfolio is down {loss_pct:.1%} from cost basis "
                        f"(threshold: {settings.risk_portfolio_loss_warn_pct:.0%})"
                    ),
                    details={"loss_pct": loss_pct, "total_cost": total_cost},
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# Market regime classification (ported from RegimeDetector.classify)
# ---------------------------------------------------------------------------


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _score_trend(prices: pd.Series) -> tuple[float, str]:
    """Score based on price vs. short/long SMA and slope."""
    if len(prices) < _LONG_WINDOW + 1:
        return 0.0, "neutral"

    short_sma = float(prices.iloc[-_SHORT_WINDOW:].mean())
    long_sma = float(prices.iloc[-_LONG_WINDOW:].mean())
    current = float(prices.iloc[-1])

    pct_vs_long = (current - long_sma) / long_sma if long_sma != 0 else 0.0
    sma_cross = 1.0 if short_sma > long_sma else -1.0

    start_price = float(prices.iloc[-(_SHORT_WINDOW + 1)])
    slope = (current - start_price) / start_price if start_price != 0 else 0.0

    raw = 0.5 * sma_cross + 0.3 * _clip(pct_vs_long * 10) + 0.2 * _clip(slope * 20)
    return _clip(raw), "bull" if raw > 0 else "bear"


def _score_volatility(vix: float) -> tuple[float, str]:
    """Score based on VIX level. High VIX -> bearish, low VIX -> bullish."""
    if vix < 16:
        return 0.8, "bull"
    if vix < 22:
        return 0.0, "neutral"
    if vix < 30:
        return -0.6, "bear"
    return -1.0, "bear"


def _score_momentum(prices: pd.Series) -> tuple[float, str]:
    """Rate of change over the momentum window."""
    if len(prices) < _MOMENTUM_WINDOW + 1:
        return 0.0, "neutral"
    start = float(prices.iloc[-(_MOMENTUM_WINDOW + 1)])
    end = float(prices.iloc[-1])
    roc = (end - start) / start if start != 0 else 0.0
    score = _clip(roc * 10)
    return score, "bull" if score > 0 else ("bear" if score < 0 else "neutral")


def _score_breadth(breadth_ratio: float | None) -> tuple[float, str]:
    """Score from advance/decline breadth ratio (0-1 scale)."""
    if breadth_ratio is None:
        return 0.0, "neutral"
    if breadth_ratio > 0.6:
        return _clip((breadth_ratio - 0.5) * 5), "bull"
    if breadth_ratio < 0.4:
        return _clip((breadth_ratio - 0.5) * 5), "bear"
    return 0.0, "neutral"


def classify_regime(
    prices: pd.Series, vix_level: float, breadth_ratio: float | None = None
) -> dict[str, Any]:
    """Classify market regime from a price series (most recent last).

    Returns a dict with keys `regime` (bull/bear/choppy/transitional),
    `confidence` (0-1), `drivers` (per-factor scores), and `votes`
    (per-factor bull/bear/neutral labels) -- identical shape to the legacy
    `RegimeDetector.classify`.
    """
    trend_score, trend_vote = _score_trend(prices)
    vol_score, vol_vote = _score_volatility(vix_level)
    mom_score, mom_vote = _score_momentum(prices)
    breadth_score, breadth_vote = _score_breadth(breadth_ratio)

    composite = (
        _WEIGHT_TREND * trend_score
        + _WEIGHT_VOLATILITY * vol_score
        + _WEIGHT_MOMENTUM * mom_score
        + _WEIGHT_BREADTH * breadth_score
    )
    confidence = min(abs(composite), 1.0)

    if confidence < _TRANSITIONAL_THRESHOLD:
        regime = "transitional"
    elif composite > 0:
        regime = "choppy" if trend_score < 0.15 and vol_score < 0.1 else "bull"
    else:
        regime = "bear"

    if regime != "bear" and trend_score < 0.05 and vol_score < 0.05:
        regime = "choppy"

    return {
        "regime": regime,
        "confidence": round(confidence, 4),
        "drivers": {
            "trend": round(trend_score, 4),
            "volatility": round(vol_score, 4),
            "momentum": round(mom_score, 4),
            "breadth": round(breadth_score, 4),
        },
        "votes": {
            "trend": trend_vote,
            "volatility": vol_vote,
            "momentum": mom_vote,
            "breadth": breadth_vote,
        },
    }
