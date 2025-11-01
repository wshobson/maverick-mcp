"""
Portfolio domain entities for MaverickMCP.

This module implements pure business logic for portfolio management following
Domain-Driven Design (DDD) principles. These entities are framework-independent
and contain the core portfolio logic including cost basis averaging and P&L calculations.

Cost Basis Method: Average Cost
- Simplest for educational purposes
- Total cost / total shares
- Does not change on partial sales
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional


@dataclass
class Position:
    """
    Value object representing a single portfolio position.

    A position tracks shares held in a specific ticker with cost basis information.
    Uses immutable operations - modifications return new Position instances.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL")
        shares: Number of shares owned (supports fractional shares)
        average_cost_basis: Average cost per share
        total_cost: Total capital invested (shares × average_cost_basis)
        purchase_date: Earliest purchase date for this position
        notes: Optional user notes about the position
    """

    ticker: str
    shares: Decimal
    average_cost_basis: Decimal
    total_cost: Decimal
    purchase_date: datetime
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate position invariants after initialization."""
        if self.shares <= 0:
            raise ValueError(f"Shares must be positive, got {self.shares}")
        if self.average_cost_basis <= 0:
            raise ValueError(
                f"Average cost basis must be positive, got {self.average_cost_basis}"
            )
        if self.total_cost <= 0:
            raise ValueError(f"Total cost must be positive, got {self.total_cost}")

        # Normalize ticker to uppercase
        object.__setattr__(self, "ticker", self.ticker.upper())

    def add_shares(self, shares: Decimal, price: Decimal, date: datetime) -> "Position":
        """
        Add shares to position with automatic cost basis averaging.

        This creates a new Position instance with updated shares and averaged cost basis.
        The average cost method is used: (total_cost + new_cost) / total_shares

        Args:
            shares: Number of shares to add (must be > 0)
            price: Purchase price per share (must be > 0)
            date: Purchase date

        Returns:
            New Position instance with averaged cost basis

        Raises:
            ValueError: If shares or price is not positive

        Example:
            >>> pos = Position("AAPL", Decimal("10"), Decimal("150"), Decimal("1500"), datetime.now())
            >>> pos = pos.add_shares(Decimal("10"), Decimal("170"), datetime.now())
            >>> pos.shares
            Decimal('20')
            >>> pos.average_cost_basis
            Decimal('160.00')
        """
        if shares <= 0:
            raise ValueError(f"Shares to add must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        new_total_shares = self.shares + shares
        new_total_cost = self.total_cost + (shares * price)
        new_avg_cost = (new_total_cost / new_total_shares).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        return Position(
            ticker=self.ticker,
            shares=new_total_shares,
            average_cost_basis=new_avg_cost,
            total_cost=new_total_cost,
            purchase_date=min(self.purchase_date, date),
            notes=self.notes,
        )

    def remove_shares(self, shares: Decimal) -> Optional["Position"]:
        """
        Remove shares from position.

        Returns None if the removal would close the position entirely (sold_shares >= held_shares).
        For partial sales, average cost basis remains unchanged.

        Args:
            shares: Number of shares to remove (must be > 0)

        Returns:
            New Position instance with reduced shares, or None if position closed

        Raises:
            ValueError: If shares is not positive

        Example:
            >>> pos = Position("AAPL", Decimal("20"), Decimal("160"), Decimal("3200"), datetime.now())
            >>> pos = pos.remove_shares(Decimal("10"))
            >>> pos.shares
            Decimal('10')
            >>> pos.average_cost_basis  # Unchanged
            Decimal('160.00')
        """
        if shares <= 0:
            raise ValueError(f"Shares to remove must be positive, got {shares}")

        if shares >= self.shares:
            # Full position close
            return None

        new_shares = self.shares - shares
        new_total_cost = new_shares * self.average_cost_basis

        return Position(
            ticker=self.ticker,
            shares=new_shares,
            average_cost_basis=self.average_cost_basis,
            total_cost=new_total_cost,
            purchase_date=self.purchase_date,
            notes=self.notes,
        )

    def calculate_current_value(self, current_price: Decimal) -> dict[str, Decimal]:
        """
        Calculate live position value and P&L metrics.

        Args:
            current_price: Current market price per share

        Returns:
            Dictionary containing:
                - current_value: Current market value (shares × price)
                - unrealized_pnl: Unrealized profit/loss (current_value - total_cost)
                - pnl_percentage: P&L as percentage of total cost

        Example:
            >>> pos = Position("AAPL", Decimal("20"), Decimal("160"), Decimal("3200"), datetime.now())
            >>> metrics = pos.calculate_current_value(Decimal("175.50"))
            >>> metrics["current_value"]
            Decimal('3510.00')
            >>> metrics["unrealized_pnl"]
            Decimal('310.00')
            >>> metrics["pnl_percentage"]
            Decimal('9.6875')
        """
        current_value = (self.shares * current_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        unrealized_pnl = (current_value - self.total_cost).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if self.total_cost > 0:
            pnl_percentage = (unrealized_pnl / self.total_cost * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            pnl_percentage = Decimal("0.00")

        return {
            "current_value": current_value,
            "unrealized_pnl": unrealized_pnl,
            "pnl_percentage": pnl_percentage,
        }

    def to_dict(self) -> dict:
        """
        Convert position to dictionary for serialization.

        Returns:
            Dictionary representation with float values for JSON compatibility
        """
        return {
            "ticker": self.ticker,
            "shares": float(self.shares),
            "average_cost_basis": float(self.average_cost_basis),
            "total_cost": float(self.total_cost),
            "purchase_date": self.purchase_date.isoformat(),
            "notes": self.notes,
        }


@dataclass
class Portfolio:
    """
    Aggregate root for user portfolio.

    Manages a collection of positions with operations for adding, removing, and analyzing
    holdings. Enforces business rules and maintains consistency.

    Attributes:
        portfolio_id: Unique identifier (UUID as string)
        user_id: User identifier (default: "default" for single-user system)
        name: Portfolio display name
        positions: List of Position value objects
        created_at: Portfolio creation timestamp
        updated_at: Last modification timestamp
    """

    portfolio_id: str
    user_id: str
    name: str
    positions: list[Position] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_position(
        self,
        ticker: str,
        shares: Decimal,
        price: Decimal,
        date: datetime,
        notes: str | None = None,
    ) -> None:
        """
        Add or update position with automatic cost basis averaging.

        If the ticker already exists, shares are added and cost basis is averaged.
        Otherwise, a new position is created.

        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to add
            price: Purchase price per share
            date: Purchase date
            notes: Optional notes (only used for new positions)

        Example:
            >>> portfolio = Portfolio("id", "default", "My Portfolio")
            >>> portfolio.add_position("AAPL", Decimal("10"), Decimal("150"), datetime.now())
            >>> portfolio.add_position("AAPL", Decimal("10"), Decimal("170"), datetime.now())
            >>> portfolio.get_position("AAPL").shares
            Decimal('20')
        """
        ticker = ticker.upper()

        # Find existing position
        for i, pos in enumerate(self.positions):
            if pos.ticker == ticker:
                self.positions[i] = pos.add_shares(shares, price, date)
                self.updated_at = datetime.now(UTC)
                return

        # Create new position
        new_position = Position(
            ticker=ticker,
            shares=shares,
            average_cost_basis=price,
            total_cost=shares * price,
            purchase_date=date,
            notes=notes,
        )
        self.positions.append(new_position)
        self.updated_at = datetime.now(UTC)

    def remove_position(self, ticker: str, shares: Decimal | None = None) -> bool:
        """
        Remove position or reduce shares.

        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to remove (None = remove entire position)

        Returns:
            True if position was found and removed/reduced, False otherwise

        Example:
            >>> portfolio.remove_position("AAPL", Decimal("10"))  # Partial
            True
            >>> portfolio.remove_position("AAPL")  # Full removal
            True
        """
        ticker = ticker.upper()

        for i, pos in enumerate(self.positions):
            if pos.ticker == ticker:
                if shares is None or shares >= pos.shares:
                    # Full position removal
                    self.positions.pop(i)
                else:
                    # Partial removal
                    updated_pos = pos.remove_shares(shares)
                    if updated_pos:
                        self.positions[i] = updated_pos
                    else:
                        self.positions.pop(i)

                self.updated_at = datetime.now(UTC)
                return True

        return False

    def get_position(self, ticker: str) -> Position | None:
        """
        Get position by ticker symbol.

        Args:
            ticker: Stock ticker symbol (case-insensitive)

        Returns:
            Position if found, None otherwise
        """
        ticker = ticker.upper()
        return next((pos for pos in self.positions if pos.ticker == ticker), None)

    def get_total_invested(self) -> Decimal:
        """
        Calculate total capital invested across all positions.

        Returns:
            Sum of all position total costs
        """
        return sum((pos.total_cost for pos in self.positions), Decimal("0"))

    def calculate_portfolio_metrics(self, current_prices: dict[str, Decimal]) -> dict:
        """
        Calculate comprehensive portfolio metrics with live prices.

        Args:
            current_prices: Dictionary mapping ticker symbols to current prices

        Returns:
            Dictionary containing:
                - total_value: Current market value of all positions
                - total_invested: Total capital invested
                - total_pnl: Total unrealized profit/loss
                - total_pnl_percentage: Total P&L as percentage
                - position_count: Number of positions
                - positions: List of position details with current metrics

        Example:
            >>> prices = {"AAPL": Decimal("175.50"), "MSFT": Decimal("380.00")}
            >>> metrics = portfolio.calculate_portfolio_metrics(prices)
            >>> metrics["total_value"]
            15250.50
        """
        total_value = Decimal("0")
        total_cost = Decimal("0")
        position_details = []

        for pos in self.positions:
            # Use current price if available, otherwise fall back to cost basis
            current_price = current_prices.get(pos.ticker, pos.average_cost_basis)
            metrics = pos.calculate_current_value(current_price)

            total_value += metrics["current_value"]
            total_cost += pos.total_cost

            position_details.append(
                {
                    "ticker": pos.ticker,
                    "shares": float(pos.shares),
                    "cost_basis": float(pos.average_cost_basis),
                    "current_price": float(current_price),
                    "current_value": float(metrics["current_value"]),
                    "unrealized_pnl": float(metrics["unrealized_pnl"]),
                    "pnl_percentage": float(metrics["pnl_percentage"]),
                    "purchase_date": pos.purchase_date.isoformat(),
                    "notes": pos.notes,
                }
            )

        total_pnl = total_value - total_cost
        total_pnl_pct = (
            (total_pnl / total_cost * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if total_cost > 0
            else Decimal("0.00")
        )

        return {
            "total_value": float(total_value),
            "total_invested": float(total_cost),
            "total_pnl": float(total_pnl),
            "total_pnl_percentage": float(total_pnl_pct),
            "position_count": len(self.positions),
            "positions": position_details,
        }

    def clear_all_positions(self) -> None:
        """
        Remove all positions from the portfolio.

        ⚠️ WARNING: This operation cannot be undone.
        """
        self.positions.clear()
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict:
        """
        Convert portfolio to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "portfolio_id": self.portfolio_id,
            "user_id": self.user_id,
            "name": self.name,
            "positions": [pos.to_dict() for pos in self.positions],
            "position_count": len(self.positions),
            "total_invested": float(self.get_total_invested()),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
