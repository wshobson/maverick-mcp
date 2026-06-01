"""Live Schwab portfolio analysis helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .client import SchwabClient
from .mapper import normalize_positions
from .sync import summarize_accounts, sync_schwab_portfolio


def _as_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def refresh_and_analyze_portfolio(
    client: SchwabClient,
    *,
    portfolio_name: str = "Schwab",
    user_id: str = "default",
    top_n: int = 10,
) -> dict[str, Any]:
    """Refresh local Schwab snapshot and return a live concentration/P&L view."""
    accounts = client.accounts(fields="positions") or []
    positions = normalize_positions(accounts)
    summaries = summarize_accounts(accounts)
    sync_result = sync_schwab_portfolio(
        client,
        portfolio_name=portfolio_name,
        user_id=user_id,
        positions=positions,
    )

    market_value = sum(
        (p.market_value for p in positions if p.market_value is not None),
        start=Decimal("0"),
    )
    total_cost = sum((p.total_cost for p in positions), start=Decimal("0"))
    unrealized_pnl = market_value - total_cost if market_value else Decimal("0")
    unrealized_pnl_pct = (
        (unrealized_pnl / total_cost * Decimal("100")) if total_cost > 0 else Decimal("0")
    )
    cash_balance = sum(
        (s.cash_balance for s in summaries if s.cash_balance is not None),
        start=Decimal("0"),
    )
    total_liquidation_value = sum(
        (s.liquidation_value for s in summaries if s.liquidation_value is not None),
        start=Decimal("0"),
    )

    position_rows: list[dict[str, Any]] = []
    for position in positions:
        value = position.market_value or Decimal("0")
        allocation = value / market_value if market_value > 0 else Decimal("0")
        position_rows.append(
            {
                "ticker": position.ticker,
                "shares": float(position.shares),
                "average_price": float(position.average_price),
                "market_value": _as_float(position.market_value),
                "total_cost": float(position.total_cost),
                "unrealized_pnl": float(value - position.total_cost),
                "allocation_pct": float(allocation * Decimal("100")),
                "asset_type": position.asset_type,
            }
        )

    position_rows.sort(key=lambda row: row["market_value"] or 0.0, reverse=True)
    top_positions = position_rows[: max(1, top_n)]
    concentration_warnings = [
        {
            "ticker": row["ticker"],
            "allocation_pct": round(row["allocation_pct"], 2),
            "message": "Position is above 20% of invested Schwab market value.",
        }
        for row in position_rows
        if row["allocation_pct"] >= 20
    ]

    return {
        "status": "ok",
        "portfolio_name": portfolio_name,
        "user_id": user_id,
        "sync": sync_result,
        "summary": {
            "account_count": len(summaries),
            "position_count": len(positions),
            "cash_balance": float(cash_balance),
            "total_liquidation_value": float(total_liquidation_value),
            "positions_market_value": float(market_value),
            "total_cost_basis": float(total_cost),
            "unrealized_pnl": float(unrealized_pnl),
            "unrealized_pnl_pct": float(unrealized_pnl_pct),
            "cash_pct_of_liquidation_value": float(
                cash_balance / total_liquidation_value * Decimal("100")
            )
            if total_liquidation_value > 0
            else 0.0,
        },
        "top_positions": top_positions,
        "concentration_warnings": concentration_warnings,
        "notes": [
            "Read-only analysis; no orders were placed.",
            "Allocation percentages are based on Schwab position market value.",
            "Educational/informational use only; not financial advice.",
        ],
    }
