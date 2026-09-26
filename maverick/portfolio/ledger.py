"""Pure Decimal position math. Third layer (sibling of data): imports config and types.

`purchase_date` on `PositionPayload` is an ISO 8601 string, not a `datetime`.
It is parsed with `datetime.fromisoformat` only transiently, to compare two
dates for earliest-date-wins; the winning date is returned as its original
string (never reformatted), so callers' formatting is preserved verbatim.

Every position `add_shares` and `remove_shares` return is rounded
ROUND_HALF_UP to the storage scales in `types.py`: `shares` to
`SHARES_SCALE` places, `average_cost_basis` and `total_cost` to `COST_SCALE`.
Inputs arrive with any number of places (the tools build them with
`Decimal(str(float))`), and `data.py`'s columns declare the same scales, so
what the ledger returns is exactly what the database stores. On SQLite, which
stores floats, that holds up to 15 significant digits: shares below
10,000,000 and costs below 100,000,000,000.
"""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from maverick.portfolio.types import (
    COST_SCALE,
    SHARES_SCALE,
    PortfolioMetrics,
    PositionPayload,
    PositionWithPrice,
    RemoveResult,
)

_SHARES_QUANT = Decimal(1).scaleb(-SHARES_SCALE)
_COST_QUANT = Decimal(1).scaleb(-COST_SCALE)
_MONEY_QUANT = Decimal("0.01")


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _round_shares(value: Decimal) -> Decimal:
    return value.quantize(_SHARES_QUANT, rounding=ROUND_HALF_UP)


def _round_cost(value: Decimal) -> Decimal:
    return value.quantize(_COST_QUANT, rounding=ROUND_HALF_UP)


def _require_storable(
    shares: Decimal, basis: Decimal, total_cost: Decimal, added: Decimal, price: Decimal
) -> None:
    """Reject a purchase whose rounded position has a zero field.
    `PositionPayload` requires all three to be positive."""
    for label, value, places in (
        ("total cost", total_cost, COST_SCALE),
        ("share count", shares, SHARES_SCALE),
        ("average cost basis", basis, COST_SCALE),
    ):
        if value <= 0:
            raise ValueError(
                f"Position too small: {label} rounds to 0 at {places} decimal "
                f"places (shares={added:f}, price={price:f})."
            )


def add_shares(
    position: PositionPayload | None,
    ticker: str,
    shares: Decimal,
    price: Decimal,
    purchase_date: str,
    notes: str | None = None,
    sector: str | None = None,
) -> PositionPayload:
    """Add shares to `position` (or create one if `position` is None).

    Average-cost formula: new average = (stored total_cost + shares * price)
    / new total shares. total_cost is the stored total_cost plus
    shares * price, never recomputed from shares * basis. All three are
    rounded to the storage scales (see the module docstring), and a result
    with any of them at 0 raises "Position too small". On merge
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
        new_shares = _round_shares(shares)
        basis = _round_cost(price)
        total_cost = _round_cost(shares * price)
        _require_storable(new_shares, basis, total_cost, shares, price)
        return PositionPayload(
            ticker=ticker,
            shares=new_shares,
            average_cost_basis=basis,
            total_cost=total_cost,
            purchase_date=purchase_date,
            notes=notes,
            sector=sector,
        )

    unrounded_shares = position.shares + shares
    unrounded_cost = position.total_cost + (shares * price)
    new_shares = _round_shares(unrounded_shares)
    basis = _round_cost(unrounded_cost / unrounded_shares)
    total_cost = _round_cost(unrounded_cost)
    _require_storable(new_shares, basis, total_cost, shares, price)
    earliest_date = (
        purchase_date
        if _parse_date(purchase_date) < _parse_date(position.purchase_date)
        else position.purchase_date
    )

    return PositionPayload(
        ticker=position.ticker,
        shares=new_shares,
        average_cost_basis=basis,
        total_cost=total_cost,
        purchase_date=earliest_date,
        notes=position.notes,
        sector=position.sector or sector,
    )


def remove_shares(
    position: PositionPayload, shares: Decimal | None
) -> tuple[PositionPayload | None, RemoveResult]:
    """Remove shares from `position`.

    The sale amount is rounded to the share scale before it is subtracted,
    the same way `add_shares` rounds a purchase. `shares=None`, or a sale
    whose remaining shares or remaining total cost rounds to 0, fully closes
    the position (returns None plus a RemoveResult reporting the
    actually-held shares as removed). Otherwise the position survives with
    the same average cost basis (average cost does not change on partial
    sales) and total_cost = remaining shares * basis, rounded to the storage
    scales like `add_shares`.
    """
    if shares is not None and shares <= 0:
        raise ValueError(f"Shares to remove must be positive, got {shares}")

    if shares is not None:
        sold = _round_shares(shares)
        remaining = position.shares - sold
        basis = _round_cost(position.average_cost_basis)
        total_cost = _round_cost(remaining * basis)
        if remaining > 0 and total_cost > 0:
            updated = PositionPayload(
                ticker=position.ticker,
                shares=remaining,
                average_cost_basis=basis,
                total_cost=total_cost,
                purchase_date=position.purchase_date,
                notes=position.notes,
                sector=position.sector,
            )
            return updated, RemoveResult(
                ticker=position.ticker,
                shares_removed=sold,
                position_fully_closed=False,
            )

    return None, RemoveResult(
        ticker=position.ticker,
        shares_removed=position.shares,
        position_fully_closed=True,
    )


def find_position(
    positions: list[PositionPayload], ticker: str
) -> PositionPayload | None:
    """Return the position for ``ticker`` in ``positions``, or None."""
    return next((position for position in positions if position.ticker == ticker), None)


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


def portfolio_snapshot_values(
    positions: list[PositionPayload], prices: dict[str, Decimal]
) -> tuple[list[PositionWithPrice], PortfolioMetrics, dict[str, float]]:
    """Build priced positions, aggregate metrics, and sector exposure.

    Failed quotes leave per-position price fields empty while metrics and
    sector exposure fall back to average cost basis.
    """
    positions_with_price: list[PositionWithPrice] = []
    sector_values: dict[str, Decimal] = {}
    total_value = Decimal("0")
    total_cost = Decimal("0")
    sector_total_value = Decimal("0")

    for position in positions:
        price = prices.get(position.ticker)
        exposure_price = price if price is not None else position.average_cost_basis
        exposure_value = position.shares * exposure_price
        value, pnl, pnl_percent = position_value(position, exposure_price)
        sector = position.sector or "Unknown"
        sector_values[sector] = sector_values.get(sector, Decimal("0")) + exposure_value
        total_value += value
        total_cost += position.total_cost
        sector_total_value += exposure_value

        if price is None:
            positions_with_price.append(
                PositionWithPrice(
                    **position.model_dump(),
                    current_price=None,
                    current_value=None,
                    unrealized_pnl=None,
                    unrealized_pnl_percent=None,
                )
            )
            continue
        positions_with_price.append(
            PositionWithPrice(
                **position.model_dump(),
                current_price=float(price),
                current_value=float(value),
                unrealized_pnl=float(pnl),
                unrealized_pnl_percent=float(pnl_percent),
            )
        )

    sector_exposure = (
        {
            sector: round(float(value / sector_total_value), 4)
            for sector, value in sector_values.items()
        }
        if sector_total_value > 0
        else {}
    )
    total_pnl = total_value - total_cost
    total_pnl_percent = (
        (total_pnl / total_cost * 100).quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)
        if total_cost > 0
        else Decimal("0.00")
    )
    metrics = PortfolioMetrics(
        total_invested=total_cost,
        total_value=float(total_value),
        total_pnl=float(total_pnl),
        total_pnl_percent=float(total_pnl_percent),
        position_count=len(positions),
    )
    return positions_with_price, metrics, sector_exposure
