"""Pure Decimal position math. Third layer (sibling of data): imports config and types.

`purchase_date` on `PositionPayload` is an ISO 8601 string, not a `datetime`.
It is parsed with `datetime.fromisoformat` only transiently, to compare two
dates for earliest-date-wins; the winning date is returned as its original
string (never reformatted), so callers' formatting is preserved verbatim.
"""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from maverick.portfolio.types import PortfolioMetrics, PositionPayload, RemoveResult

_BASIS_QUANT = Decimal("0.0001")
_MONEY_QUANT = Decimal("0.01")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def add_shares(
    position: PositionPayload | None,
    ticker: str,
    shares: Decimal,
    price: Decimal,
    purchase_date: str,
    notes: str | None = None,
) -> PositionPayload:
    """Add shares to `position` (or create one if `position` is None).

    Average-cost formula: new average = (stored total_cost + shares * price)
    / new total shares, quantized to 0.0001 with ROUND_HALF_UP. total_cost
    itself is never quantized or recomputed from shares * basis. On merge
    into an existing position, `notes` is ignored (legacy behavior: notes
    are only captured for brand-new positions) and the earlier of the two
    purchase dates wins.
    """
    if shares <= 0:
        raise ValueError(f"Shares to add must be positive, got {shares}")
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")

    ticker = ticker.upper()

    if position is None:
        return PositionPayload(
            ticker=ticker,
            shares=shares,
            average_cost_basis=price,
            total_cost=shares * price,
            purchase_date=purchase_date,
            notes=notes,
        )

    new_total_shares = position.shares + shares
    new_total_cost = position.total_cost + (shares * price)
    new_avg_cost = (new_total_cost / new_total_shares).quantize(
        _BASIS_QUANT, rounding=ROUND_HALF_UP
    )
    earliest_date = (
        purchase_date
        if _parse_date(purchase_date) < _parse_date(position.purchase_date)
        else position.purchase_date
    )

    return PositionPayload(
        ticker=position.ticker,
        shares=new_total_shares,
        average_cost_basis=new_avg_cost,
        total_cost=new_total_cost,
        purchase_date=earliest_date,
        notes=position.notes,
    )


def remove_shares(
    position: PositionPayload, shares: Decimal | None
) -> tuple[PositionPayload | None, RemoveResult]:
    """Remove shares from `position`.

    `shares=None` or `shares >= position.shares` fully closes the position
    (returns None plus a RemoveResult reporting the actually-held shares as
    removed). Otherwise the position survives with the same average cost
    basis (average cost does not change on partial sales) and
    total_cost = remaining shares * basis.
    """
    if shares is not None and shares <= 0:
        raise ValueError(f"Shares to remove must be positive, got {shares}")

    if shares is None or shares >= position.shares:
        return None, RemoveResult(
            ticker=position.ticker,
            shares_removed=position.shares,
            position_fully_closed=True,
        )

    new_shares = position.shares - shares
    updated = PositionPayload(
        ticker=position.ticker,
        shares=new_shares,
        average_cost_basis=position.average_cost_basis,
        total_cost=new_shares * position.average_cost_basis,
        purchase_date=position.purchase_date,
        notes=position.notes,
    )
    return updated, RemoveResult(
        ticker=position.ticker,
        shares_removed=shares,
        position_fully_closed=False,
    )


def position_value(
    position: PositionPayload, current_price: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Return (current_value, unrealized_pnl, unrealized_pnl_percent).

    All three are quantized to 0.01 with ROUND_HALF_UP. `total_cost == 0`
    is safe: pnl_percent is 0.00 instead of dividing by zero.
    """
    value = (position.shares * current_price).quantize(
        _MONEY_QUANT, rounding=ROUND_HALF_UP
    )
    pnl = (value - position.total_cost).quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)

    if position.total_cost > 0:
        pnl_percent = (pnl / position.total_cost * 100).quantize(
            _MONEY_QUANT, rounding=ROUND_HALF_UP
        )
    else:
        pnl_percent = Decimal("0.00")

    return value, pnl, pnl_percent


def portfolio_metrics(
    positions: list[PositionPayload], prices: dict[str, Decimal]
) -> PortfolioMetrics:
    """Aggregate metrics across `positions` using `prices` (keyed by ticker).

    A ticker missing from `prices` falls back to that position's own
    average_cost_basis (matching legacy behavior: an unpriced position
    contributes zero P&L rather than being dropped or erroring).
    """
    total_value = Decimal("0")
    total_cost = Decimal("0")

    for position in positions:
        current_price = prices.get(position.ticker, position.average_cost_basis)
        value, _pnl, _pnl_percent = position_value(position, current_price)
        total_value += value
        total_cost += position.total_cost

    total_pnl = total_value - total_cost
    total_pnl_percent = (
        (total_pnl / total_cost * 100).quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)
        if total_cost > 0
        else Decimal("0.00")
    )

    return PortfolioMetrics(
        total_invested=total_cost,
        total_value=float(total_value),
        total_pnl=float(total_pnl),
        total_pnl_percent=float(total_pnl_percent),
        position_count=len(positions),
    )
