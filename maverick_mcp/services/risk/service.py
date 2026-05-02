"""Risk dashboard service — portfolio risk analytics and alerting."""

from __future__ import annotations

import logging
import math
from typing import Any

from sqlalchemy.orm import Session

from maverick_mcp.services.risk.models import RiskAlert

logger = logging.getLogger(__name__)

# Parametric VaR z-scores
_Z_95 = 1.645
_Z_99 = 2.326

# Alert thresholds
_SECTOR_WARN_PCT = 0.30
_SECTOR_CRITICAL_PCT = 0.50
_POSITION_WARN_PCT = 0.20
_PORTFOLIO_LOSS_WARN_PCT = 0.10

# Regime risk multipliers
_REGIME_MULTIPLIERS: dict[str, float] = {
    "bull": 1.0,
    "choppy": 0.75,
    "transitional": 0.75,
    "bear": 0.5,
}


class RiskService:
    """Business logic for portfolio risk analysis and alerting.

    Args:
        db_session: A SQLAlchemy synchronous session.
    """

    def __init__(self, db_session: Session) -> None:
        self._db = db_session

    # ------------------------------------------------------------------
    # Dashboard computation
    # ------------------------------------------------------------------

    def compute_dashboard(
        self, portfolio_positions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute risk metrics for a portfolio.

        Args:
            portfolio_positions: List of position dicts, each containing:
                - symbol: ticker symbol
                - shares: number of shares
                - cost_basis: average cost per share
                - current_price: current market price
                - sector: sector classification (optional)

        Returns:
            Dict with total_value, sector_concentration, max_sector_pct,
            portfolio_var_95, portfolio_var_99, total_pnl.
        """
        if not portfolio_positions:
            return {
                "total_value": 0.0,
                "sector_concentration": {},
                "max_sector_pct": 0.0,
                "portfolio_var_95": 0.0,
                "portfolio_var_99": 0.0,
                "total_pnl": 0.0,
                "position_count": 0,
            }

        total_value = sum(
            float(p.get("shares", 0)) * float(p.get("current_price", 0))
            for p in portfolio_positions
        )

        total_pnl = sum(
            (float(p.get("current_price", 0)) - float(p.get("cost_basis", 0)))
            * float(p.get("shares", 0))
            for p in portfolio_positions
        )

        # Sector concentration
        sector_values: dict[str, float] = {}
        for pos in portfolio_positions:
            sector = pos.get("sector") or "Unknown"
            val = float(pos.get("shares", 0)) * float(pos.get("current_price", 0))
            sector_values[sector] = sector_values.get(sector, 0.0) + val

        sector_concentration: dict[str, float] = {}
        if total_value > 0:
            for sector, val in sector_values.items():
                sector_concentration[sector] = val / total_value

        max_sector_pct = max(sector_concentration.values(), default=0.0)

        # Simplified parametric VaR — assumes equal-weight portfolio std of 2% daily
        portfolio_std = _estimate_portfolio_std(portfolio_positions, total_value)
        portfolio_var_95 = _Z_95 * portfolio_std * math.sqrt(1) * total_value
        portfolio_var_99 = _Z_99 * portfolio_std * math.sqrt(1) * total_value

        return {
            "total_value": round(total_value, 2),
            "sector_concentration": {
                k: round(v, 4) for k, v in sector_concentration.items()
            },
            "max_sector_pct": round(max_sector_pct, 4),
            "portfolio_var_95": round(portfolio_var_95, 2),
            "portfolio_var_99": round(portfolio_var_99, 2),
            "total_pnl": round(total_pnl, 2),
            "position_count": len(portfolio_positions),
        }

    # ------------------------------------------------------------------
    # Pre-trade risk check
    # ------------------------------------------------------------------

    def check_position_risk(
        self,
        portfolio_positions: list[dict[str, Any]],
        new_ticker: str,
        new_shares: int,
        new_price: float,
    ) -> dict[str, Any]:
        """Compute portfolio risk including a prospective new position.

        Args:
            portfolio_positions: Current portfolio positions (same schema as compute_dashboard).
            new_ticker: Ticker for the proposed new position.
            new_shares: Number of shares to add.
            new_price: Entry price per share.

        Returns:
            Dict with current and projected risk metrics plus position impact.
        """
        current = self.compute_dashboard(portfolio_positions)

        new_position: dict[str, Any] = {
            "symbol": new_ticker.upper(),
            "shares": new_shares,
            "cost_basis": new_price,
            "current_price": new_price,
            "sector": "Unknown",
        }

        # Merge — if ticker already exists, add shares
        merged: list[dict[str, Any]] = []
        ticker_found = False
        for pos in portfolio_positions:
            if pos.get("symbol", "").upper() == new_ticker.upper():
                merged_pos = dict(pos)
                existing_shares = float(pos.get("shares", 0))
                existing_cb = float(pos.get("cost_basis", 0))
                total_shares = existing_shares + new_shares
                avg_cb = (
                    (existing_shares * existing_cb + new_shares * new_price)
                    / total_shares
                    if total_shares > 0
                    else new_price
                )
                merged_pos["shares"] = total_shares
                merged_pos["cost_basis"] = avg_cb
                merged_pos["current_price"] = new_price
                merged.append(merged_pos)
                ticker_found = True
            else:
                merged.append(dict(pos))

        if not ticker_found:
            merged.append(new_position)

        projected = self.compute_dashboard(merged)
        position_value = new_shares * new_price

        return {
            "current": current,
            "projected": projected,
            "new_position": {
                "ticker": new_ticker.upper(),
                "shares": new_shares,
                "price": new_price,
                "position_value": round(position_value, 2),
                "pct_of_projected_portfolio": round(
                    position_value / projected["total_value"], 4
                )
                if projected["total_value"] > 0
                else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Regime-adjusted position sizing
    # ------------------------------------------------------------------

    def get_regime_adjusted_size(
        self,
        account_size: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float,
        regime: str,
    ) -> dict[str, Any]:
        """Calculate position size adjusted by market regime.

        Args:
            account_size: Total account value in dollars.
            entry_price: Intended entry price per share.
            stop_loss: Stop-loss price per share.
            risk_pct: Base risk percentage of account (e.g. 2.0 for 2%).
            regime: Market regime string (bull / choppy / transitional / bear).

        Returns:
            Dict with shares, position_value, risk_amount, regime_multiplier.
        """
        multiplier = _REGIME_MULTIPLIERS.get(regime.lower(), 1.0)
        adjusted_risk_pct = risk_pct * multiplier
        risk_amount = account_size * (adjusted_risk_pct / 100.0)

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            shares = 0
        else:
            shares = int(risk_amount / risk_per_share)

        position_value = shares * entry_price

        return {
            "shares": shares,
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "regime_multiplier": multiplier,
            "adjusted_risk_pct": round(adjusted_risk_pct, 4),
            "regime": regime.lower(),
        }

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def generate_alerts(
        self,
        portfolio_positions: list[dict[str, Any]],
        portfolio_name: str = "default",
    ) -> list[RiskAlert]:
        """Check portfolio for risk threshold violations and generate alerts.

        Args:
            portfolio_positions: Portfolio positions (same schema as compute_dashboard).
            portfolio_name: Portfolio identifier for alert tagging.

        Returns:
            List of :class:`RiskAlert` instances (not persisted).
        """
        dashboard = self.compute_dashboard(portfolio_positions)
        alerts: list[RiskAlert] = []

        total_value = dashboard["total_value"]

        # Sector concentration checks
        for sector, pct in dashboard["sector_concentration"].items():
            if pct > _SECTOR_CRITICAL_PCT:
                alerts.append(
                    RiskAlert(
                        portfolio_name=portfolio_name,
                        alert_type="concentration",
                        severity="critical",
                        message=(
                            f"Sector '{sector}' represents {pct:.1%} of portfolio "
                            f"(threshold: {_SECTOR_CRITICAL_PCT:.0%})"
                        ),
                        details={"sector": sector, "pct": pct},
                    )
                )
            elif pct > _SECTOR_WARN_PCT:
                alerts.append(
                    RiskAlert(
                        portfolio_name=portfolio_name,
                        alert_type="concentration",
                        severity="warning",
                        message=(
                            f"Sector '{sector}' represents {pct:.1%} of portfolio "
                            f"(threshold: {_SECTOR_WARN_PCT:.0%})"
                        ),
                        details={"sector": sector, "pct": pct},
                    )
                )

        # Single position size check
        if total_value > 0:
            for pos in portfolio_positions:
                pos_value = float(pos.get("shares", 0)) * float(
                    pos.get("current_price", 0)
                )
                pos_pct = pos_value / total_value
                ticker = pos.get("symbol", pos.get("ticker", "?"))
                if pos_pct > _POSITION_WARN_PCT:
                    alerts.append(
                        RiskAlert(
                            portfolio_name=portfolio_name,
                            alert_type="sizing",
                            severity="warning",
                            message=(
                                f"Position '{ticker}' is {pos_pct:.1%} of portfolio "
                                f"(threshold: {_POSITION_WARN_PCT:.0%})"
                            ),
                            details={"ticker": ticker, "pct": pos_pct},
                        )
                    )

        # Portfolio drawdown / loss check
        total_cost = sum(
            float(p.get("cost_basis", 0)) * float(p.get("shares", 0))
            for p in portfolio_positions
        )
        if total_cost > 0:
            loss_pct = (total_cost - total_value) / total_cost
            if loss_pct > _PORTFOLIO_LOSS_WARN_PCT:
                alerts.append(
                    RiskAlert(
                        portfolio_name=portfolio_name,
                        alert_type="drawdown",
                        severity="warning",
                        message=(
                            f"Portfolio is down {loss_pct:.1%} from cost basis "
                            f"(threshold: {_PORTFOLIO_LOSS_WARN_PCT:.0%})"
                        ),
                        details={"loss_pct": loss_pct, "total_cost": total_cost},
                    )
                )

        return alerts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_portfolio_std(
    positions: list[dict[str, Any]], total_value: float
) -> float:
    """Estimate a simplified portfolio daily standard deviation.

    Uses a flat 2% daily volatility assumption per position and assumes
    zero cross-position correlation for simplicity. This yields a
    diversification-adjusted portfolio std.
    """
    if total_value <= 0 or not positions:
        return 0.02  # default 2%

    daily_vol_per_position = 0.02  # 2% assumed daily vol per position
    variance = 0.0
    for pos in positions:
        weight = (
            float(pos.get("shares", 0))
            * float(pos.get("current_price", 0))
            / total_value
        )
        variance += (weight * daily_vol_per_position) ** 2

    return math.sqrt(variance)
