"""Hierarchical Risk Parity (HRP) portfolio optimization.

Uses riskfolio-lib to compute HRP allocations from historical returns data.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_hrp(
    symbols: list[str],
    days: int = 252,
    risk_measure: str = "MV",
) -> dict[str, Any]:
    """Compute Hierarchical Risk Parity weights for a basket of symbols.

    Args:
        symbols: List of ticker symbols (minimum 2)
        days: Number of trading days of historical data (default: 252 ~ 1 year)
        risk_measure: Risk measure for HRP ('MV' for variance, 'CVaR', 'CDaR', etc.)

    Returns:
        Dict with weights, expected metrics, and comparison to equal-weight
    """
    if len(symbols) < 2:
        return {
            "error": "At least 2 symbols required for HRP optimization",
            "status": "error",
        }

    try:
        import riskfolio as rp
    except ImportError:
        return {
            "error": "riskfolio-lib is not installed. Run: pip install riskfolio-lib",
            "status": "error",
        }

    from maverick_mcp.providers.stock_data import StockDataProvider

    provider = StockDataProvider()
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
    start_date = (datetime.now(UTC) - timedelta(days=int(days * 1.5))).strftime(
        "%Y-%m-%d"
    )

    # Fetch price data
    price_data = {}
    failed = []
    for symbol in symbols:
        try:
            df = provider.get_stock_data(symbol, start_date, end_date)
            if not df.empty:
                close_col = "Close" if "Close" in df.columns else "close"
                price_data[symbol] = df[close_col]
            else:
                failed.append(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol} for HRP: {e}")
            failed.append(symbol)

    valid_symbols = list(price_data.keys())
    if len(valid_symbols) < 2:
        return {
            "error": f"Need 2+ symbols with valid data, got {len(valid_symbols)}",
            "failed_symbols": failed,
            "status": "error",
        }

    # Build returns DataFrame
    prices_df = pd.DataFrame(price_data)
    returns_df = prices_df.pct_change().dropna()

    if len(returns_df) < 30:
        return {
            "error": f"Insufficient data points ({len(returns_df)}). Need 30+.",
            "status": "error",
        }

    # Truncate to requested number of days
    returns_df = returns_df.iloc[-days:]

    # Run HRP optimization
    try:
        port = rp.HCPortfolio(returns=returns_df)
        weights = port.optimization(
            model="HRP",
            rm=risk_measure,
            linkage="ward",
        )

        if weights is None or weights.empty:
            return {"error": "HRP optimization returned no weights", "status": "error"}

        # Extract weights as dict
        weight_dict = {
            symbol: round(float(w), 4) for symbol, w in weights.iloc[:, 0].items()
        }

        # Calculate portfolio metrics
        mean_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252

        w = np.array([weight_dict.get(s, 0) for s in valid_symbols])
        r = np.array([mean_returns[s] for s in valid_symbols])

        portfolio_return = float(np.dot(w, r))
        portfolio_risk = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))

        # Equal-weight comparison
        ew = np.ones(len(valid_symbols)) / len(valid_symbols)
        ew_return = float(np.dot(ew, r))
        ew_risk = float(np.sqrt(np.dot(ew.T, np.dot(cov_matrix.values, ew))))

        return {
            "status": "success",
            "weights": weight_dict,
            "metrics": {
                "expected_annual_return": round(portfolio_return, 4),
                "expected_annual_risk": round(portfolio_risk, 4),
                "sharpe_estimate": round(portfolio_return / portfolio_risk, 4)
                if portfolio_risk > 0
                else None,
            },
            "equal_weight_comparison": {
                "ew_return": round(ew_return, 4),
                "ew_risk": round(ew_risk, 4),
                "ew_sharpe": round(ew_return / ew_risk, 4) if ew_risk > 0 else None,
                "hrp_improvement": round(
                    (portfolio_return / portfolio_risk - ew_return / ew_risk), 4
                )
                if portfolio_risk > 0 and ew_risk > 0
                else None,
            },
            "risk_measure": risk_measure,
            "data_points": len(returns_df),
            "symbols_used": valid_symbols,
            "failed_symbols": failed,
        }

    except Exception as e:
        logger.error(f"HRP optimization failed: {e}")
        return {"error": f"HRP optimization failed: {str(e)}", "status": "error"}
