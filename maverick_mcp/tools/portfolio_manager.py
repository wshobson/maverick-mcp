"""
Portfolio manager for financial portfolio analysis and management.
This module provides a portfolio management interface for tracking and analyzing investment portfolios.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.portfolio_manager")

# Load environment variables
load_dotenv()


class PortfolioManager:
    """
    Portfolio manager for tracking and analyzing investment portfolios.
    """

    def __init__(
        self,
        portfolio_name: str,
        risk_profile: str = "moderate",
        portfolio_file: str | None = None,
    ):
        """
        Initialize the portfolio manager

        Args:
            portfolio_name: Name of the portfolio
            risk_profile: Risk profile of the portfolio ('conservative', 'moderate', 'aggressive')
            portfolio_file: Path to a JSON file containing portfolio data
        """
        self.portfolio_name = portfolio_name
        self.risk_profile = risk_profile
        self.portfolio_file = portfolio_file

        # Load portfolio from file if provided
        self.portfolio = []
        if portfolio_file and os.path.exists(portfolio_file):
            with open(portfolio_file) as f:
                data = json.load(f)
                self.portfolio = data.get("holdings", [])
                self.risk_profile = data.get("risk_profile", risk_profile)
                self.portfolio_name = data.get("name", portfolio_name)

        self.transaction_history: list[dict[str, Any]] = []

    async def add_to_portfolio(self, symbol: str, shares: float, price: float):
        """
        Add a stock to the portfolio

        Args:
            symbol: Stock ticker symbol
            shares: Number of shares to add
            price: Purchase price per share
        """
        # Check if stock already exists in portfolio
        for holding in self.portfolio:
            if holding["symbol"] == symbol:
                # Update existing holding
                old_shares = holding["shares"]
                old_price = holding["avg_price"]
                total_cost = (old_shares * old_price) + (shares * price)
                total_shares = old_shares + shares
                holding["shares"] = total_shares
                holding["avg_price"] = total_cost / total_shares
                holding["last_update"] = datetime.now(UTC).isoformat()

                # Record transaction
                self.transaction_history.append(
                    {
                        "type": "buy",
                        "symbol": symbol,
                        "shares": shares,
                        "price": price,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

                return

        # Add new holding
        self.portfolio.append(
            {
                "symbol": symbol,
                "shares": shares,
                "avg_price": price,
                "purchase_date": datetime.now(UTC).isoformat(),
                "last_update": datetime.now(UTC).isoformat(),
            }
        )

        # Record transaction
        self.transaction_history.append(
            {
                "type": "buy",
                "symbol": symbol,
                "shares": shares,
                "price": price,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    async def remove_from_portfolio(
        self, symbol: str, shares: float | None = None, price: float | None = None
    ):
        """
        Remove a stock from the portfolio

        Args:
            symbol: Stock ticker symbol
            shares: Number of shares to remove (if None, remove all shares)
            price: Selling price per share
        """
        for i, holding in enumerate(self.portfolio):
            if holding["symbol"] == symbol:
                if shares is None or shares >= holding["shares"]:
                    # Remove entire holding
                    removed_holding = self.portfolio.pop(i)

                    # Record transaction
                    self.transaction_history.append(
                        {
                            "type": "sell",
                            "symbol": symbol,
                            "shares": removed_holding["shares"],
                            "price": price,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                else:
                    # Partially remove holding
                    holding["shares"] -= shares
                    holding["last_update"] = datetime.now(UTC).isoformat()

                    # Record transaction
                    self.transaction_history.append(
                        {
                            "type": "sell",
                            "symbol": symbol,
                            "shares": shares,
                            "price": price,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

                return True

        return False

    async def get_portfolio_value(self) -> dict[str, Any]:
        """
        Get the current value of the portfolio

        Returns:
            Dictionary with portfolio value information
        """
        if not self.portfolio:
            return {
                "total_value": 0,
                "holdings": [],
                "timestamp": datetime.now(UTC).isoformat(),
            }

        total_value = 0
        holdings_data = []

        for holding in self.portfolio:
            symbol = holding["symbol"]
            shares = holding["shares"]
            avg_price = holding["avg_price"]
            current_price = avg_price  # In a real implementation, fetch current price from market data API

            # Calculate values
            position_value = shares * current_price
            cost_basis = shares * avg_price
            gain_loss = position_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0

            holdings_data.append(
                {
                    "symbol": symbol,
                    "shares": shares,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "position_value": position_value,
                    "cost_basis": cost_basis,
                    "gain_loss": gain_loss,
                    "gain_loss_pct": gain_loss_pct,
                }
            )

            total_value += position_value

        return {
            "total_value": total_value,
            "holdings": holdings_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_portfolio_analysis(self) -> dict[str, Any]:
        """
        Get a comprehensive analysis of the portfolio

        Returns:
            Dictionary with portfolio analysis information
        """
        if not self.portfolio:
            return {
                "analysis": "Portfolio is empty. No analysis available.",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Get current portfolio value
        portfolio_value = await self.get_portfolio_value()

        # In a real implementation, perform portfolio analysis here
        analysis = "Portfolio analysis not implemented"

        return {
            "portfolio_data": portfolio_value,
            "analysis": analysis,
            "risk_profile": self.risk_profile,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_rebalance_recommendations(self) -> dict[str, Any]:
        """
        Get recommendations for rebalancing the portfolio

        Returns:
            Dictionary with rebalance recommendations
        """
        if not self.portfolio:
            return {
                "recommendations": "Portfolio is empty. No rebalance recommendations available.",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Get current portfolio value
        portfolio_value = await self.get_portfolio_value()

        # In a real implementation, generate rebalancing recommendations here
        recommendations = "Rebalance recommendations not implemented"

        return {
            "portfolio_data": portfolio_value,
            "recommendations": recommendations,
            "risk_profile": self.risk_profile,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def save_portfolio(self, filepath: str | None = None):
        """
        Save the portfolio to a file

        Args:
            filepath: Path to save the portfolio to (if None, use the portfolio file path)
        """
        if not filepath:
            filepath = (
                self.portfolio_file
                or f"{self.portfolio_name.replace(' ', '_').lower()}_portfolio.json"
            )

        data = {
            "name": self.portfolio_name,
            "risk_profile": self.risk_profile,
            "holdings": self.portfolio,
            "transaction_history": self.transaction_history,
            "last_update": datetime.now(UTC).isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Portfolio saved to {filepath}")

        return filepath


async def main():
    """Example usage of the portfolio manager"""
    # Create a sample portfolio
    portfolio = [
        {"symbol": "AAPL", "shares": 10, "avg_price": 170.50},
        {"symbol": "MSFT", "shares": 5, "avg_price": 325.25},
        {"symbol": "GOOGL", "shares": 2, "avg_price": 140.75},
        {"symbol": "AMZN", "shares": 3, "avg_price": 178.30},
        {"symbol": "TSLA", "shares": 8, "avg_price": 185.60},
    ]

    # Create the portfolio manager
    manager = PortfolioManager(
        portfolio_name="Tech Growth Portfolio",
        risk_profile="moderate",
    )

    # Add the sample stocks to the portfolio
    for holding in portfolio:
        await manager.add_to_portfolio(
            symbol=str(holding["symbol"]),
            shares=float(holding["shares"]),  # type: ignore[arg-type]
            price=float(holding["avg_price"]),  # type: ignore[arg-type]
        )

    try:
        # Get portfolio value
        print("Getting portfolio value...")
        portfolio_value = await manager.get_portfolio_value()
        print(f"Total portfolio value: ${portfolio_value['total_value']:.2f}")

        # Get portfolio analysis
        print("\nAnalyzing portfolio...")
        analysis = await manager.get_portfolio_analysis()
        print("\nPortfolio Analysis:")
        print(analysis["analysis"])

        # Get rebalance recommendations
        print("\nGetting rebalance recommendations...")
        rebalance = await manager.get_rebalance_recommendations()
        print("\nRebalance Recommendations:")
        print(rebalance["recommendations"])

        # Save the portfolio
        filepath = manager.save_portfolio()
        print(f"\nPortfolio saved to {filepath}")

    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
