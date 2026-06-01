"""Normalize Schwab account payloads for Maverick portfolio sync."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class SchwabPosition:
    """Normalized equity position from Schwab."""

    ticker: str
    shares: Decimal
    average_price: Decimal
    market_value: Decimal | None
    asset_type: str | None

    @property
    def total_cost(self) -> Decimal:
        """Return shares times average price."""
        return self.shares * self.average_price


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def normalize_positions(accounts: list[dict[str, Any]]) -> list[SchwabPosition]:
    """Convert Schwab account payloads into syncable positions."""
    normalized: list[SchwabPosition] = []

    for account in accounts:
        securities_account = account.get("securitiesAccount") or {}
        for position in securities_account.get("positions") or []:
            instrument = position.get("instrument") or {}
            ticker = str(instrument.get("symbol") or "").strip().upper()
            if not ticker:
                continue

            long_quantity = _decimal(position.get("longQuantity")) or Decimal("0")
            short_quantity = _decimal(position.get("shortQuantity")) or Decimal("0")
            shares = long_quantity if long_quantity > 0 else short_quantity
            average_price = _decimal(position.get("averagePrice"))

            if shares <= 0 or average_price is None or average_price <= 0:
                continue

            normalized.append(
                SchwabPosition(
                    ticker=ticker,
                    shares=shares,
                    average_price=average_price,
                    market_value=_decimal(position.get("marketValue")),
                    asset_type=instrument.get("assetType"),
                )
            )

    return normalized
