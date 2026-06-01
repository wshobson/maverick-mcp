"""Schwab-to-Maverick portfolio sync helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from maverick_mcp.data.models import PortfolioPosition, SessionLocal, UserPortfolio

from .client import SchwabClient
from .mapper import SchwabPosition, normalize_positions


@dataclass(frozen=True)
class SchwabAccountSummary:
    """Scrubbed Schwab account summary."""

    account_type: str | None
    positions_count: int
    liquidation_value: Decimal | None
    cash_balance: Decimal | None


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def summarize_accounts(accounts: list[dict[str, Any]]) -> list[SchwabAccountSummary]:
    """Return scrubbed account-level summaries from Schwab account payloads."""
    summaries: list[SchwabAccountSummary] = []
    for account in accounts:
        securities_account = account.get("securitiesAccount") or {}
        balances = securities_account.get("currentBalances") or {}
        summaries.append(
            SchwabAccountSummary(
                account_type=securities_account.get("type"),
                positions_count=len(securities_account.get("positions") or []),
                liquidation_value=_decimal(balances.get("liquidationValue")),
                cash_balance=_decimal(
                    balances.get("cashBalance")
                    or balances.get("cashAvailableForTrading")
                    or balances.get("availableFunds")
                ),
            )
        )
    return summaries


def fetch_schwab_positions(client: SchwabClient) -> list[SchwabPosition]:
    """Fetch and normalize live Schwab positions."""
    return normalize_positions(client.accounts(fields="positions") or [])


def sync_schwab_portfolio(
    client: SchwabClient,
    *,
    portfolio_name: str = "Schwab",
    user_id: str = "default",
) -> dict[str, Any]:
    """Snapshot live Schwab positions into Maverick's local portfolio storage."""
    positions = fetch_schwab_positions(client)
    synced_at = datetime.now(UTC)

    with SessionLocal() as session:
        portfolio = (
            session.query(UserPortfolio)
            .filter_by(user_id=user_id, name=portfolio_name)
            .first()
        )
        if not portfolio:
            portfolio = UserPortfolio(user_id=user_id, name=portfolio_name)
            session.add(portfolio)
            session.flush()

        deleted = (
            session.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id)
            .delete()
        )

        for position in positions:
            note = (
                "Synced from Schwab"
                if position.asset_type is None
                else f"Synced from Schwab ({position.asset_type})"
            )
            session.add(
                PortfolioPosition(
                    portfolio_id=portfolio.id,
                    ticker=position.ticker,
                    shares=position.shares,
                    average_cost_basis=position.average_price,
                    total_cost=position.total_cost.quantize(Decimal("0.0001")),
                    purchase_date=synced_at,
                    notes=note,
                )
            )

        session.commit()

    return {
        "status": "ok",
        "portfolio_name": portfolio_name,
        "user_id": user_id,
        "positions_synced": len(positions),
        "positions_replaced": deleted,
        "as_of": synced_at.isoformat(),
        "tickers": [p.ticker for p in positions],
    }
