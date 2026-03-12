"""Enhanced Adaptive Relative Strength (EARS) scoring.

Calculates relative strength scores for stocks vs SPY and sector ETFs.
Score = weighted RS vs SPY (60%) + RS vs sector ETF (40%), normalized to 0-100.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# GICS sector to ETF mapping
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

# Default SPY weight in EARS score
SPY_WEIGHT = 0.6
SECTOR_WEIGHT = 0.4


def _get_return(data: pd.DataFrame, lookback: int) -> float | None:
    """Calculate return over lookback period from a price DataFrame."""
    if data.empty or len(data) < lookback:
        return None

    # Handle both upper and lower case column names
    close_col = "Close" if "Close" in data.columns else "close"
    if close_col not in data.columns:
        return None

    prices = data[close_col]
    start_price = prices.iloc[-lookback]
    end_price = prices.iloc[-1]

    if start_price == 0:
        return None

    return (end_price - start_price) / start_price


def calculate_ears_score(
    ticker_return: float,
    spy_return: float,
    sector_return: float | None = None,
) -> float:
    """Calculate EARS score from pre-computed returns.

    Args:
        ticker_return: Ticker's return over lookback period
        spy_return: SPY return over same period
        sector_return: Sector ETF return over same period (optional)

    Returns:
        EARS score from 0-100 (50 = market-matching)
    """
    if spy_return == 0:
        rs_vs_spy = 50.0
    else:
        rs_vs_spy = (ticker_return / abs(spy_return)) * 50.0 + 50.0

    if sector_return is not None and sector_return != 0:
        rs_vs_sector = (ticker_return / abs(sector_return)) * 50.0 + 50.0
        score = rs_vs_spy * SPY_WEIGHT + rs_vs_sector * SECTOR_WEIGHT
    else:
        score = rs_vs_spy

    # Clamp to 0-100
    return max(0.0, min(100.0, round(score, 2)))


def enrich_stocks_with_ears(
    stocks: list[dict[str, Any]], days: int = 63
) -> list[dict[str, Any]]:
    """Add EARS scores to a list of stock dicts.

    Fetches SPY and sector ETF data once, then computes EARS for each stock.

    Args:
        stocks: List of stock dicts (must have 'ticker' or 'symbol' key)
        days: Lookback period in trading days (default: 63 ~ 3 months)

    Returns:
        Same list with 'ears_score' added to each dict
    """
    if not stocks:
        return stocks

    from maverick_mcp.providers.stock_data import StockDataProvider

    provider = StockDataProvider()
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
    start_date = (datetime.now(UTC) - timedelta(days=days * 2)).strftime("%Y-%m-%d")

    # Fetch SPY data once
    try:
        spy_data = provider.get_stock_data("SPY", start_date, end_date)
        spy_return = _get_return(spy_data, days)
    except Exception as e:
        logger.warning(f"Failed to fetch SPY data for EARS: {e}")
        spy_return = None

    if spy_return is None:
        logger.warning("Could not compute SPY return for EARS scoring")
        return stocks

    # Cache sector ETF data
    sector_etf_cache: dict[str, float | None] = {}

    for stock in stocks:
        ticker = stock.get("ticker") or stock.get("symbol")
        if not ticker:
            continue

        try:
            # Get ticker return
            ticker_data = provider.get_stock_data(ticker, start_date, end_date)
            ticker_return = _get_return(ticker_data, days)

            if ticker_return is None:
                stock["ears_score"] = None
                continue

            # Get sector ETF return (cached)
            sector = stock.get("sector", "")
            sector_etf = SECTOR_ETF_MAP.get(sector)
            sector_return = None

            if sector_etf:
                if sector_etf not in sector_etf_cache:
                    try:
                        etf_data = provider.get_stock_data(
                            sector_etf, start_date, end_date
                        )
                        sector_etf_cache[sector_etf] = _get_return(etf_data, days)
                    except Exception:
                        sector_etf_cache[sector_etf] = None

                sector_return = sector_etf_cache.get(sector_etf)

            stock["ears_score"] = calculate_ears_score(
                ticker_return, spy_return, sector_return
            )

        except Exception as e:
            logger.warning(f"Failed to compute EARS for {ticker}: {e}")
            stock["ears_score"] = None

    return stocks
