"""
Portfolio analysis router for Maverick-MCP.

This module contains all portfolio-related tools including:
- Portfolio management (add, get, remove, clear positions)
- Risk analysis and comparisons
- Optimization functions
"""

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pandas_ta as ta
from fastmcp import FastMCP
from sqlalchemy.orm import Session

from maverick_mcp.data.models import PortfolioPosition, UserPortfolio, get_db
from maverick_mcp.domain.portfolio import Portfolio
from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.utils.stock_helpers import get_stock_dataframe

logger = logging.getLogger(__name__)

# Create the portfolio router
portfolio_router: FastMCP = FastMCP("Portfolio_Analysis")

# Initialize data provider
stock_provider = StockDataProvider()


def _normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol to uppercase and strip whitespace."""
    return ticker.strip().upper()


def _validate_ticker(ticker: str) -> tuple[bool, str | None]:
    """
    Validate ticker symbol format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ticker or not ticker.strip():
        return False, "Ticker symbol cannot be empty"

    normalized = ticker.strip().upper()

    # Basic validation: 1-5 alphanumeric characters
    if not normalized.isalnum():
        return (
            False,
            f"Invalid ticker symbol '{ticker}': must contain only letters and numbers",
        )

    if len(normalized) > 10:
        return False, f"Invalid ticker symbol '{ticker}': too long (max 10 characters)"

    return True, None


def risk_adjusted_analysis(
    ticker: str,
    risk_level: float | str | None = 50.0,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """
    Perform risk-adjusted stock analysis with position sizing.

    DISCLAIMER: This analysis is for educational purposes only and does not
    constitute investment advice. All investments carry risk of loss. Always
    consult with qualified financial professionals before making investment decisions.

    This tool analyzes a stock with risk parameters tailored to different investment
    styles. It provides:
    - Position sizing recommendations based on ATR
    - Stop loss suggestions
    - Entry points with scaling
    - Risk/reward ratio calculations
    - Confidence score based on technicals

    **Portfolio Integration:** If you already own this stock, the analysis includes:
    - Current position details (shares, cost basis, unrealized P&L)
    - Position sizing relative to existing holdings
    - Recommendations for averaging up/down

    The risk_level parameter (0-100) adjusts the analysis from conservative (low)
    to aggressive (high).

    Args:
        ticker: The ticker symbol to analyze
        risk_level: Risk tolerance from 0 (conservative) to 100 (aggressive)
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")

    Returns:
        Dictionary containing risk-adjusted analysis results with optional position context
    """
    try:
        # Convert risk_level to float if it's a string
        if isinstance(risk_level, str):
            try:
                risk_level = float(risk_level)
            except ValueError:
                risk_level = 50.0

        # Use explicit date range to avoid weekend/holiday issues
        from datetime import UTC, datetime, timedelta

        end_date = (datetime.now(UTC) - timedelta(days=7)).strftime(
            "%Y-%m-%d"
        )  # Last week to be safe
        start_date = (datetime.now(UTC) - timedelta(days=365)).strftime(
            "%Y-%m-%d"
        )  # 1 year ago
        df = stock_provider.get_stock_data(
            ticker, start_date=start_date, end_date=end_date
        )

        # Validate dataframe has required columns (check for both upper and lower case)
        required_cols = ["high", "low", "close"]
        actual_cols_lower = [col.lower() for col in df.columns]
        if df.empty or not all(col in actual_cols_lower for col in required_cols):
            return {
                "error": f"Insufficient data for {ticker}",
                "details": "Unable to retrieve required price data (High, Low, Close) for analysis",
                "ticker": ticker,
                "required_data": ["High", "Low", "Close", "Volume"],
                "available_columns": list(df.columns),
            }

        df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=20)
        atr = df["atr"].iloc[-1]
        current_price = df["Close"].iloc[-1]
        risk_factor = (risk_level or 50.0) / 100  # Convert to 0-1 scale
        account_size = 100000
        analysis = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "atr": round(atr, 2),
            "risk_level": risk_level,
            "position_sizing": {
                "suggested_position_size": round(account_size * 0.01 * risk_factor, 2),
                "max_shares": int((account_size * 0.01 * risk_factor) / current_price),
                "position_value": round(account_size * 0.01 * risk_factor, 2),
                "percent_of_portfolio": round(1 * risk_factor, 2),
            },
            "risk_management": {
                "stop_loss": round(current_price - (atr * (2 - risk_factor)), 2),
                "stop_loss_percent": round(
                    ((atr * (2 - risk_factor)) / current_price) * 100, 2
                ),
                "max_risk_amount": round(account_size * 0.01 * risk_factor, 2),
            },
            "entry_strategy": {
                "immediate_entry": round(current_price, 2),
                "scale_in_levels": [
                    round(current_price, 2),
                    round(current_price - (atr * 0.5), 2),
                    round(current_price - atr, 2),
                ],
            },
            "targets": {
                "price_target": round(current_price + (atr * 3 * risk_factor), 2),
                "profit_potential": round(atr * 3 * risk_factor, 2),
                "risk_reward_ratio": round(3 * risk_factor, 2),
            },
            "analysis": {
                "confidence_score": round(70 * risk_factor, 2),
                "strategy_type": "aggressive"
                if (risk_level or 50.0) > 70
                else "moderate"
                if (risk_level or 50.0) > 30
                else "conservative",
                "time_horizon": "short-term"
                if (risk_level or 50.0) > 70
                else "medium-term"
                if (risk_level or 50.0) > 30
                else "long-term",
            },
        }

        # Check if user already owns this position
        db: Session = next(get_db())
        try:
            portfolio = (
                db.query(UserPortfolio)
                .filter(
                    UserPortfolio.user_id == user_id,
                    UserPortfolio.name == portfolio_name,
                )
                .first()
            )

            if portfolio:
                existing_position = next(
                    (
                        pos
                        for pos in portfolio.positions
                        if pos.ticker.upper() == ticker.upper()
                    ),
                    None,
                )

                if existing_position:
                    # Calculate unrealized P&L
                    unrealized_pnl = (
                        current_price - float(existing_position.average_cost_basis)
                    ) * float(existing_position.shares)
                    unrealized_pnl_pct = (
                        (current_price - float(existing_position.average_cost_basis))
                        / float(existing_position.average_cost_basis)
                    ) * 100

                    analysis["existing_position"] = {
                        "shares_owned": float(existing_position.shares),
                        "average_cost_basis": float(
                            existing_position.average_cost_basis
                        ),
                        "total_invested": float(existing_position.total_cost),
                        "current_value": float(existing_position.shares)
                        * current_price,
                        "unrealized_pnl": round(unrealized_pnl, 2),
                        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                        "position_recommendation": "Consider averaging down"
                        if current_price < float(existing_position.average_cost_basis)
                        else "Consider taking partial profits"
                        if unrealized_pnl_pct > 20
                        else "Hold current position",
                    }
        finally:
            db.close()

        return analysis
    except Exception as e:
        logger.error(f"Error performing risk analysis for {ticker}: {e}")
        return {"error": str(e)}


def compare_tickers(
    tickers: list[str] | None = None,
    days: int = 90,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """
    Compare multiple tickers using technical and fundamental metrics.

    This tool provides side-by-side comparison of stocks including:
    - Price performance
    - Technical indicators (RSI, trend strength)
    - Volume characteristics
    - Momentum strength ratings
    - Risk metrics

    **Portfolio Integration:** If no tickers are provided, automatically compares
    all positions in your portfolio, making it easy to see which holdings are
    performing best.

    Args:
        tickers: List of ticker symbols to compare (minimum 2). If None, uses portfolio holdings.
        days: Number of days of historical data to analyze (default: 90)
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")

    Returns:
        Dictionary containing comparison results with optional portfolio context

    Example:
        >>> compare_tickers()  # Automatically compares all portfolio holdings
        >>> compare_tickers(["AAPL", "MSFT", "GOOGL"])  # Manual comparison
    """
    try:
        # Auto-fill tickers from portfolio if not provided
        if tickers is None or len(tickers) == 0:
            db: Session = next(get_db())
            try:
                # Get portfolio positions
                portfolio = (
                    db.query(UserPortfolio)
                    .filter(
                        UserPortfolio.user_id == user_id,
                        UserPortfolio.name == portfolio_name,
                    )
                    .first()
                )

                if not portfolio or len(portfolio.positions) < 2:
                    return {
                        "error": "No portfolio found or insufficient positions for comparison",
                        "details": "Please provide at least 2 tickers manually or add more positions to your portfolio",
                        "status": "error",
                    }

                tickers = [pos.ticker for pos in portfolio.positions]
                portfolio_context = {
                    "using_portfolio": True,
                    "portfolio_name": portfolio_name,
                    "position_count": len(tickers),
                }
            finally:
                db.close()
        else:
            portfolio_context = {"using_portfolio": False}

        if len(tickers) < 2:
            raise ValueError("At least two tickers are required for comparison")

        from maverick_mcp.core.technical_analysis import analyze_rsi, analyze_trend

        results = {}
        for ticker in tickers:
            df = get_stock_dataframe(ticker, days)

            # Basic analysis for comparison
            current_price = df["close"].iloc[-1]
            rsi = analyze_rsi(df)
            trend = analyze_trend(df)

            # Calculate performance metrics
            start_price = df["close"].iloc[0]
            price_change_pct = ((current_price - start_price) / start_price) * 100

            # Calculate volatility (standard deviation of returns)
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() * (252**0.5) * 100  # Annualized

            # Calculate volume metrics
            volume_change_pct = 0.0
            if len(df) >= 22 and df["volume"].iloc[-22] > 0:
                volume_change_pct = float(
                    (df["volume"].iloc[-1] / df["volume"].iloc[-22] - 1) * 100
                )

            avg_volume = df["volume"].mean()

            results[ticker] = {
                "current_price": float(current_price),
                "performance": {
                    "price_change_pct": round(price_change_pct, 2),
                    "period_high": float(df["high"].max()),
                    "period_low": float(df["low"].min()),
                    "volatility_annual": round(volatility, 2),
                },
                "technical": {
                    "rsi": rsi["current"] if rsi and "current" in rsi else None,
                    "rsi_signal": rsi["signal"]
                    if rsi and "signal" in rsi
                    else "unavailable",
                    "trend_strength": trend,
                    "trend_description": "Strong Uptrend"
                    if trend >= 6
                    else "Uptrend"
                    if trend >= 4
                    else "Neutral"
                    if trend >= 3
                    else "Downtrend",
                },
                "volume": {
                    "current_volume": int(df["volume"].iloc[-1]),
                    "avg_volume": int(avg_volume),
                    "volume_change_pct": volume_change_pct,
                    "volume_trend": "Increasing"
                    if volume_change_pct > 20
                    else "Decreasing"
                    if volume_change_pct < -20
                    else "Stable",
                },
            }

        # Add relative rankings
        tickers_list = list(results.keys())

        # Rank by performance
        def get_performance(ticker: str) -> float:
            ticker_result = results[ticker]
            assert isinstance(ticker_result, dict)
            perf_dict = ticker_result["performance"]
            assert isinstance(perf_dict, dict)
            return float(perf_dict["price_change_pct"])

        def get_trend(ticker: str) -> float:
            ticker_result = results[ticker]
            assert isinstance(ticker_result, dict)
            tech_dict = ticker_result["technical"]
            assert isinstance(tech_dict, dict)
            return float(tech_dict["trend_strength"])

        perf_sorted = sorted(tickers_list, key=get_performance, reverse=True)
        trend_sorted = sorted(tickers_list, key=get_trend, reverse=True)

        for i, ticker in enumerate(perf_sorted):
            results[ticker]["rankings"] = {
                "performance_rank": i + 1,
                "trend_rank": trend_sorted.index(ticker) + 1,
            }

        response = {
            "comparison": results,
            "period_days": days,
            "as_of": datetime.now(UTC).isoformat(),
            "best_performer": perf_sorted[0],
            "strongest_trend": trend_sorted[0],
        }

        # Add portfolio context if applicable
        if portfolio_context["using_portfolio"]:
            response["portfolio_context"] = portfolio_context

        return response
    except Exception as e:
        logger.error(f"Error comparing tickers {tickers}: {str(e)}")
        return {"error": str(e), "status": "error"}


def portfolio_correlation_analysis(
    tickers: list[str] | None = None,
    days: int = 252,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """
    Analyze correlation between multiple securities.

    DISCLAIMER: This correlation analysis is for educational purposes only.
    Past correlations do not guarantee future relationships between securities.
    Always diversify appropriately and consult with financial professionals.

    This tool calculates the correlation matrix for a portfolio of stocks,
    helping to identify:
    - Highly correlated positions (diversification issues)
    - Negative correlations (natural hedges)
    - Overall portfolio correlation metrics

    **Portfolio Integration:** If no tickers are provided, automatically analyzes
    correlation between all positions in your portfolio, helping you understand
    diversification and identify concentration risk.

    Args:
        tickers: List of ticker symbols to analyze. If None, uses portfolio holdings.
        days: Number of days for correlation calculation (default: 252 for 1 year)
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")

    Returns:
        Dictionary containing correlation analysis with optional portfolio context

    Example:
        >>> portfolio_correlation_analysis()  # Automatically analyzes portfolio
        >>> portfolio_correlation_analysis(["AAPL", "MSFT", "GOOGL"])  # Manual analysis
    """
    try:
        # Auto-fill tickers from portfolio if not provided
        if tickers is None or len(tickers) == 0:
            db: Session = next(get_db())
            try:
                # Get portfolio positions
                portfolio = (
                    db.query(UserPortfolio)
                    .filter(
                        UserPortfolio.user_id == user_id,
                        UserPortfolio.name == portfolio_name,
                    )
                    .first()
                )

                if not portfolio or len(portfolio.positions) < 2:
                    return {
                        "error": "No portfolio found or insufficient positions for correlation analysis",
                        "details": "Please provide at least 2 tickers manually or add more positions to your portfolio",
                        "status": "error",
                    }

                tickers = [pos.ticker for pos in portfolio.positions]
                portfolio_context = {
                    "using_portfolio": True,
                    "portfolio_name": portfolio_name,
                    "position_count": len(tickers),
                }
            finally:
                db.close()
        else:
            portfolio_context = {"using_portfolio": False}

        if len(tickers) < 2:
            raise ValueError("At least two tickers required for correlation analysis")

        # Fetch data for all tickers
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        price_data = {}
        failed_tickers = []
        for ticker in tickers:
            try:
                df = stock_provider.get_stock_data(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )
                if not df.empty:
                    price_data[ticker] = df["close"]
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
                failed_tickers.append(ticker)

        # Check if we have enough valid tickers
        if len(price_data) < 2:
            return {
                "error": f"Insufficient valid price data (need 2+ tickers, got {len(price_data)})",
                "details": f"Failed tickers: {', '.join(failed_tickers)}"
                if failed_tickers
                else "No tickers provided sufficient data",
                "status": "error",
            }

        # Create price DataFrame
        prices_df = pd.DataFrame(price_data)

        # Calculate returns
        returns_df = prices_df.pct_change().dropna()

        # Check for sufficient data points
        if len(returns_df) < 30:
            return {
                "error": "Insufficient data points for correlation analysis",
                "details": f"Need at least 30 data points, got {len(returns_df)}. Try increasing the days parameter.",
                "status": "error",
            }

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        # Check for NaN/Inf values
        if (
            correlation_matrix.isnull().any().any()
            or not correlation_matrix.applymap(lambda x: abs(x) <= 1.0).all().all()
        ):
            return {
                "error": "Invalid correlation values detected",
                "details": "Correlation matrix contains NaN or invalid values. This may indicate insufficient price variation.",
                "status": "error",
            }

        # Find highly correlated pairs
        high_correlation_pairs = []
        low_correlation_pairs = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                corr_val = correlation_matrix.iloc[i, j]
                corr = float(corr_val.item() if hasattr(corr_val, "item") else corr_val)
                pair = (tickers[i], tickers[j])

                if corr > 0.7:
                    high_correlation_pairs.append(
                        {
                            "pair": pair,
                            "correlation": round(corr, 3),
                            "interpretation": "High positive correlation",
                        }
                    )
                elif corr < -0.3:
                    low_correlation_pairs.append(
                        {
                            "pair": pair,
                            "correlation": round(corr, 3),
                            "interpretation": "Negative correlation (potential hedge)",
                        }
                    )

        # Calculate average portfolio correlation
        mask = correlation_matrix.values != 1  # Exclude diagonal
        avg_correlation = correlation_matrix.values[mask].mean()

        response = {
            "correlation_matrix": correlation_matrix.round(3).to_dict(),
            "average_portfolio_correlation": round(avg_correlation, 3),
            "high_correlation_pairs": high_correlation_pairs,
            "low_correlation_pairs": low_correlation_pairs,
            "diversification_score": round((1 - avg_correlation) * 100, 1),
            "recommendation": "Well diversified"
            if avg_correlation < 0.3
            else "Moderately diversified"
            if avg_correlation < 0.5
            else "Consider adding uncorrelated assets",
            "period_days": days,
            "data_points": len(returns_df),
        }

        # Add portfolio context if applicable
        if portfolio_context["using_portfolio"]:
            response["portfolio_context"] = portfolio_context

        return response

    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        return {"error": str(e), "status": "error"}


# ============================================================================
# Portfolio Management Tools
# ============================================================================


def add_portfolio_position(
    ticker: str,
    shares: float,
    purchase_price: float,
    purchase_date: str | None = None,
    notes: str | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """
    Add a stock position to your portfolio.

    This tool adds a new position or increases an existing position in your portfolio.
    If the ticker already exists, it will average the cost basis automatically.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        shares: Number of shares (supports fractional shares)
        purchase_price: Price per share at purchase
        purchase_date: Purchase date in YYYY-MM-DD format (defaults to today)
        notes: Optional notes about this position
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")

    Returns:
        Dictionary containing the updated position information

    Example:
        >>> add_portfolio_position("AAPL", 10, 150.50, "2024-01-15", "Long-term hold")
    """
    try:
        # Validate and normalize ticker
        is_valid, error_msg = _validate_ticker(ticker)
        if not is_valid:
            return {"error": error_msg, "status": "error"}

        ticker = _normalize_ticker(ticker)

        # Validate shares
        if shares <= 0:
            return {"error": "Shares must be greater than zero", "status": "error"}
        if shares > 1_000_000_000:  # Sanity check
            return {
                "error": "Shares value too large (max 1 billion shares)",
                "status": "error",
            }

        # Validate purchase price
        if purchase_price <= 0:
            return {
                "error": "Purchase price must be greater than zero",
                "status": "error",
            }
        if purchase_price > 1_000_000:  # Sanity check
            return {
                "error": "Purchase price too large (max $1M per share)",
                "status": "error",
            }

        # Parse purchase date
        if purchase_date:
            try:
                parsed_date = datetime.fromisoformat(
                    purchase_date.replace("Z", "+00:00")
                )
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=UTC)
            except ValueError:
                return {
                    "error": "Invalid date format. Use YYYY-MM-DD",
                    "status": "error",
                }
        else:
            parsed_date = datetime.now(UTC)

        db: Session = next(get_db())
        try:
            # Get or create portfolio
            portfolio_db = (
                db.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )

            if not portfolio_db:
                portfolio_db = UserPortfolio(user_id=user_id, name=portfolio_name)
                db.add(portfolio_db)
                db.flush()

            # Get existing position if any
            existing_position = (
                db.query(PortfolioPosition)
                .filter_by(portfolio_id=portfolio_db.id, ticker=ticker.upper())
                .first()
            )

            total_cost = Decimal(str(shares)) * Decimal(str(purchase_price))

            if existing_position:
                # Update existing position (average cost basis)
                old_total = (
                    existing_position.shares * existing_position.average_cost_basis
                )
                new_total = old_total + total_cost
                new_shares = existing_position.shares + Decimal(str(shares))
                new_avg_cost = new_total / new_shares

                existing_position.shares = new_shares
                existing_position.average_cost_basis = new_avg_cost
                existing_position.total_cost = new_total
                existing_position.purchase_date = parsed_date
                if notes:
                    existing_position.notes = notes

                position_result = existing_position
            else:
                # Create new position
                position_result = PortfolioPosition(
                    portfolio_id=portfolio_db.id,
                    ticker=ticker.upper(),
                    shares=Decimal(str(shares)),
                    average_cost_basis=Decimal(str(purchase_price)),
                    total_cost=total_cost,
                    purchase_date=parsed_date,
                    notes=notes,
                )
                db.add(position_result)

            db.commit()

            return {
                "status": "success",
                "message": f"Added {shares} shares of {ticker.upper()}",
                "position": {
                    "ticker": position_result.ticker,
                    "shares": float(position_result.shares),
                    "average_cost_basis": float(position_result.average_cost_basis),
                    "total_cost": float(position_result.total_cost),
                    "purchase_date": position_result.purchase_date.isoformat(),
                    "notes": position_result.notes,
                },
                "portfolio": {
                    "name": portfolio_db.name,
                    "user_id": portfolio_db.user_id,
                },
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error adding position {ticker}: {str(e)}")
        return {"error": str(e), "status": "error"}


def get_my_portfolio(
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
    include_current_prices: bool = True,
) -> dict[str, Any]:
    """
    Get your complete portfolio with all positions and performance metrics.

    This tool retrieves your entire portfolio including:
    - All stock positions with cost basis
    - Current market values (if prices available)
    - Profit/loss for each position
    - Portfolio-wide performance metrics

    Args:
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")
        include_current_prices: Whether to fetch live prices for P&L (default: True)

    Returns:
        Dictionary containing complete portfolio information with performance metrics

    Example:
        >>> get_my_portfolio()
    """
    try:
        db: Session = next(get_db())
        try:
            # Get portfolio
            portfolio_db = (
                db.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )

            if not portfolio_db:
                return {
                    "status": "empty",
                    "message": f"No portfolio found for user '{user_id}' with name '{portfolio_name}'",
                    "positions": [],
                    "total_invested": 0.0,
                }

            # Get all positions
            positions = (
                db.query(PortfolioPosition)
                .filter_by(portfolio_id=portfolio_db.id)
                .all()
            )

            if not positions:
                return {
                    "status": "empty",
                    "message": "Portfolio is empty",
                    "portfolio": {
                        "name": portfolio_db.name,
                        "user_id": portfolio_db.user_id,
                    },
                    "positions": [],
                    "total_invested": 0.0,
                }

            # Convert to domain model for calculations
            portfolio = Portfolio(name=portfolio_db.name)
            for pos_db in positions:
                portfolio = portfolio.add_position(
                    pos_db.ticker,
                    pos_db.shares,
                    pos_db.average_cost_basis,
                    pos_db.purchase_date,
                )

            # Fetch current prices if requested
            current_prices = {}
            if include_current_prices:
                for pos in positions:
                    try:
                        df = stock_provider.get_stock_data(
                            pos.ticker,
                            start_date=(datetime.now(UTC) - timedelta(days=7)).strftime(
                                "%Y-%m-%d"
                            ),
                            end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                        )
                        if not df.empty:
                            current_prices[pos.ticker] = float(df["Close"].iloc[-1])
                    except Exception as e:
                        logger.warning(
                            f"Could not fetch price for {pos.ticker}: {str(e)}"
                        )

            # Calculate metrics
            metrics = portfolio.calculate_portfolio_metrics(current_prices)

            # Build response
            positions_list = []
            for pos_db in positions:
                position_dict = {
                    "ticker": pos_db.ticker,
                    "shares": float(pos_db.shares),
                    "average_cost_basis": float(pos_db.average_cost_basis),
                    "total_cost": float(pos_db.total_cost),
                    "purchase_date": pos_db.purchase_date.isoformat(),
                    "notes": pos_db.notes,
                }

                # Add current price and P&L if available
                if pos_db.ticker in current_prices:
                    current_price = current_prices[pos_db.ticker]
                    position_dict["current_price"] = current_price
                    position_dict["current_value"] = (
                        float(pos_db.shares) * current_price
                    )
                    position_dict["unrealized_gain_loss"] = position_dict[
                        "current_value"
                    ] - float(pos_db.total_cost)
                    position_dict["unrealized_gain_loss_percent"] = (
                        position_dict["unrealized_gain_loss"] / float(pos_db.total_cost)
                    ) * 100

                positions_list.append(position_dict)

            return {
                "status": "success",
                "portfolio": {
                    "name": portfolio_db.name,
                    "user_id": portfolio_db.user_id,
                    "created_at": portfolio_db.created_at.isoformat(),
                },
                "positions": positions_list,
                "metrics": {
                    "total_invested": metrics["total_invested"],
                    "total_current_value": metrics["total_current_value"],
                    "total_unrealized_gain_loss": metrics["total_unrealized_gain_loss"],
                    "total_return_percent": metrics["total_return_percent"],
                    "number_of_positions": len(positions_list),
                },
                "as_of": datetime.now(UTC).isoformat(),
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        return {"error": str(e), "status": "error"}


def remove_portfolio_position(
    ticker: str,
    shares: float | None = None,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """
    Remove shares from a position in your portfolio.

    This tool removes some or all shares of a stock from your portfolio.
    If no share count is specified, the entire position is removed.

    Args:
        ticker: Stock ticker symbol
        shares: Number of shares to remove (None = remove entire position)
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")

    Returns:
        Dictionary containing the updated or removed position

    Example:
        >>> remove_portfolio_position("AAPL", 5)  # Remove 5 shares
        >>> remove_portfolio_position("MSFT")     # Remove entire position
    """
    try:
        # Validate and normalize ticker
        is_valid, error_msg = _validate_ticker(ticker)
        if not is_valid:
            return {"error": error_msg, "status": "error"}

        ticker = _normalize_ticker(ticker)

        # Validate shares if provided
        if shares is not None and shares <= 0:
            return {
                "error": "Shares to remove must be greater than zero",
                "status": "error",
            }

        db: Session = next(get_db())
        if shares is not None and shares <= 0:
            return {"error": "Shares must be greater than zero", "status": "error"}

        db: Session = next(get_db())
        try:
            # Get portfolio
            portfolio_db = (
                db.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )

            if not portfolio_db:
                return {
                    "error": f"Portfolio '{portfolio_name}' not found for user '{user_id}'",
                    "status": "error",
                }

            # Get position
            position_db = (
                db.query(PortfolioPosition)
                .filter_by(portfolio_id=portfolio_db.id, ticker=ticker.upper())
                .first()
            )

            if not position_db:
                return {
                    "error": f"Position {ticker.upper()} not found in portfolio",
                    "status": "error",
                }

            # Remove entire position or partial shares
            if shares is None or shares >= float(position_db.shares):
                # Remove entire position
                removed_shares = float(position_db.shares)
                db.delete(position_db)
                db.commit()

                return {
                    "status": "success",
                    "message": f"Removed entire position of {removed_shares} shares of {ticker.upper()}",
                    "removed_shares": removed_shares,
                    "position_fully_closed": True,
                }
            else:
                # Remove partial shares
                new_shares = position_db.shares - Decimal(str(shares))
                new_total_cost = new_shares * position_db.average_cost_basis

                position_db.shares = new_shares
                position_db.total_cost = new_total_cost
                db.commit()

                return {
                    "status": "success",
                    "message": f"Removed {shares} shares of {ticker.upper()}",
                    "removed_shares": shares,
                    "position_fully_closed": False,
                    "remaining_position": {
                        "ticker": position_db.ticker,
                        "shares": float(position_db.shares),
                        "average_cost_basis": float(position_db.average_cost_basis),
                        "total_cost": float(position_db.total_cost),
                    },
                }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error removing position {ticker}: {str(e)}")
        return {"error": str(e), "status": "error"}


def clear_my_portfolio(
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
    confirm: bool = False,
) -> dict[str, Any]:
    """
    Clear all positions from your portfolio.

    CAUTION: This removes all positions from the specified portfolio.
    This action cannot be undone.

    Args:
        user_id: User identifier (defaults to "default")
        portfolio_name: Portfolio name (defaults to "My Portfolio")
        confirm: Must be True to confirm deletion (safety check)

    Returns:
        Dictionary containing confirmation of cleared positions

    Example:
        >>> clear_my_portfolio(confirm=True)
    """
    try:
        if not confirm:
            return {
                "error": "Must set confirm=True to clear portfolio",
                "status": "error",
                "message": "This is a safety check to prevent accidental deletion",
            }

        db: Session = next(get_db())
        try:
            # Get portfolio
            portfolio_db = (
                db.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )

            if not portfolio_db:
                return {
                    "error": f"Portfolio '{portfolio_name}' not found for user '{user_id}'",
                    "status": "error",
                }

            # Count positions before deletion
            positions_count = (
                db.query(PortfolioPosition)
                .filter_by(portfolio_id=portfolio_db.id)
                .count()
            )

            if positions_count == 0:
                return {
                    "status": "success",
                    "message": "Portfolio was already empty",
                    "positions_cleared": 0,
                }

            # Delete all positions
            db.query(PortfolioPosition).filter_by(portfolio_id=portfolio_db.id).delete()
            db.commit()

            return {
                "status": "success",
                "message": f"Cleared all positions from portfolio '{portfolio_name}'",
                "positions_cleared": positions_count,
                "portfolio": {
                    "name": portfolio_db.name,
                    "user_id": portfolio_db.user_id,
                },
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error clearing portfolio: {str(e)}")
        return {"error": str(e), "status": "error"}
