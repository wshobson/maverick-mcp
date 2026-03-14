"""Options analysis router for MaverickMCP.

Provides MCP tools for options chain data, Greeks calculation,
IV analysis, strategy P&L, unusual activity detection, and
portfolio-aware hedging recommendations.

DISCLAIMER: All options analysis tools are for educational purposes only.
Options trading involves significant risk and is not suitable for all investors.
Always consult qualified financial professionals before trading options.
"""

from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from maverick_mcp.core.options_analysis import (
    _days_to_expiry,
    analyze_iv_skew,
    analyze_iv_term_structure,
    analyze_strategy,
    build_common_strategy,
    calculate_all_greeks,
    detect_unusual_activity,
    suggest_hedges,
)
from maverick_mcp.core.options_analysis import (
    price_option as core_price_option,
)
from maverick_mcp.providers.options_data import OptionsDataProvider
from maverick_mcp.utils.error_handling import safe_error_message
from maverick_mcp.utils.logging import PerformanceMonitor, get_logger
from maverick_mcp.utils.mcp_logging import with_logging

logger = get_logger("maverick_mcp.routers.options")

# Thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=4)
atexit.register(_executor.shutdown, wait=False)

# Shared data provider (in-memory cache is per-process)
_provider = OptionsDataProvider()


# ------------------------------------------------------------------ #
# Tool 1: Get Options Chain
# ------------------------------------------------------------------ #


@with_logging("options_get_chain")
async def get_options_chain(
    ticker: str,
    expiration: str | None = None,
    min_volume: int = 10,
    min_open_interest: int = 100,
    max_bid_ask_spread_pct: float = 10.0,
) -> dict[str, Any]:
    """Fetch options chain with liquidity filtering.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        expiration: Expiration date YYYY-MM-DD (nearest if omitted)
        min_volume: Minimum contract volume (default 10)
        min_open_interest: Minimum open interest (default 100)
        max_bid_ask_spread_pct: Maximum bid-ask spread % (default 10)

    Returns:
        Filtered options chain with calls, puts, and underlying price
    """
    try:
        with PerformanceMonitor(f"options_chain_{ticker}"):
            loop = asyncio.get_event_loop()

            # Get available expirations
            expirations = await loop.run_in_executor(
                _executor, _provider.get_available_expirations, ticker
            )

            if not expirations:
                return {
                    "error": f"No options available for {ticker}",
                    "ticker": ticker,
                }

            # Use specified or nearest expiration
            exp = (
                expiration
                if expiration and expiration in expirations
                else expirations[0]
            )

            chain = await loop.run_in_executor(
                _executor,
                _provider.get_option_chain,
                ticker,
                exp,
                min_volume,
                min_open_interest,
                max_bid_ask_spread_pct,
            )

            return {
                "ticker": ticker,
                "expiration": exp,
                "available_expirations": expirations,
                "underlying_price": chain.get("underlying_price", 0),
                "calls_count": len(chain.get("calls", [])),
                "puts_count": len(chain.get("puts", [])),
                "calls": chain.get("calls", []),
                "puts": chain.get("puts", []),
                "filters_applied": {
                    "min_volume": min_volume,
                    "min_open_interest": min_open_interest,
                    "max_bid_ask_spread_pct": max_bid_ask_spread_pct,
                },
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
    except Exception as e:
        logger.exception("Error fetching options chain for %s: %s", ticker, e)
        return {
            "error": safe_error_message(e, context="fetching options chain"),
            "ticker": ticker,
        }


# ------------------------------------------------------------------ #
# Tool 2: Calculate Greeks
# ------------------------------------------------------------------ #


@with_logging("options_calculate_greeks")
async def calculate_option_greeks(
    ticker: str,
    strike: float,
    expiration: str,
    option_type: str = "call",
    risk_free_rate: float = 0.0425,
    dividend_yield: float | None = None,
) -> dict[str, Any]:
    """Calculate full European + American Greeks for an option.

    Args:
        ticker: Stock ticker symbol
        strike: Strike price
        expiration: Expiration date YYYY-MM-DD
        option_type: 'call' or 'put'
        risk_free_rate: Risk-free rate (default 0.0425)
        dividend_yield: Dividend yield (auto-fetched if omitted)

    Returns:
        European (1st-3rd order) and American (BAW) Greeks
    """
    try:
        with PerformanceMonitor(f"greeks_{ticker}_{strike}"):
            loop = asyncio.get_event_loop()

            # Fetch spot price and dividend yield in parallel
            spot_future = loop.run_in_executor(
                _executor, _provider.get_underlying_price, ticker
            )
            if dividend_yield is None:
                div_future = loop.run_in_executor(
                    _executor, _provider.get_dividend_yield, ticker
                )
                spot, div_yield = await asyncio.gather(spot_future, div_future)
            else:
                spot = await spot_future
                div_yield = dividend_yield

            if spot <= 0:
                return {
                    "error": f"Could not fetch price for {ticker}",
                    "ticker": ticker,
                }

            # Try to get IV from the chain
            iv = await _get_iv_from_chain(ticker, expiration, strike, option_type)
            if iv <= 0:
                iv = 0.25  # default 25% if unavailable

            T = _days_to_expiry(expiration)

            result = calculate_all_greeks(
                spot=spot,
                strike=strike,
                time_to_expiry=T,
                risk_free_rate=risk_free_rate,
                volatility=iv,
                dividend_yield=div_yield,
                option_type=option_type,
            )
            result["ticker"] = ticker
            result["expiration"] = expiration
            result["implied_volatility_used"] = round(iv, 4)
            result["timestamp"] = datetime.now(tz=UTC).isoformat()
            return result
    except Exception as e:
        logger.exception("Error calculating Greeks for %s: %s", ticker, e)
        return {
            "error": safe_error_message(e, context="calculating option Greeks"),
            "ticker": ticker,
        }


# ------------------------------------------------------------------ #
# Tool 3: IV Analysis
# ------------------------------------------------------------------ #


@with_logging("options_iv_analysis")
async def get_iv_analysis(
    ticker: str,
    expiration: str | None = None,
) -> dict[str, Any]:
    """Analyze implied volatility skew and term structure.

    Args:
        ticker: Stock ticker symbol
        expiration: Specific expiration for skew (all expirations for term structure)

    Returns:
        IV skew analysis and term structure
    """
    try:
        with PerformanceMonitor(f"iv_analysis_{ticker}"):
            loop = asyncio.get_event_loop()

            expirations = await loop.run_in_executor(
                _executor, _provider.get_available_expirations, ticker
            )

            if not expirations:
                return {"error": f"No options available for {ticker}", "ticker": ticker}

            spot = await loop.run_in_executor(
                _executor, _provider.get_underlying_price, ticker
            )

            # Skew analysis for target expiration
            target_exp = (
                expiration
                if expiration and expiration in expirations
                else expirations[0]
            )
            chain = await loop.run_in_executor(
                _executor, _provider.get_option_chain, ticker, target_exp, 0, 0, 999.0
            )
            skew = analyze_iv_skew(chain, spot)

            # Term structure: fetch chains for first few expirations
            max_exp = min(len(expirations), 6)
            chains_by_exp: dict[str, dict[str, Any]] = {}
            for exp in expirations[:max_exp]:
                if exp == target_exp:
                    chains_by_exp[exp] = chain
                else:
                    c = await loop.run_in_executor(
                        _executor, _provider.get_option_chain, ticker, exp, 0, 0, 999.0
                    )
                    chains_by_exp[exp] = c

            term_structure = analyze_iv_term_structure(chains_by_exp, spot)

            return {
                "ticker": ticker,
                "underlying_price": spot,
                "skew": skew,
                "term_structure": term_structure,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
    except Exception as e:
        logger.exception("Error in IV analysis for %s: %s", ticker, e)
        return {"error": safe_error_message(e, context="IV analysis"), "ticker": ticker}


# ------------------------------------------------------------------ #
# Tool 4: Price Option
# ------------------------------------------------------------------ #


@with_logging("options_price_option")
async def price_option(
    ticker: str,
    strike: float,
    expiration: str,
    option_type: str = "call",
    model: str = "baw",
    risk_free_rate: float = 0.0425,
    dividend_yield: float | None = None,
) -> dict[str, Any]:
    """Price an option using BSM or Barone-Adesi Whaley model.

    Args:
        ticker: Stock ticker symbol
        strike: Strike price
        expiration: Expiration date YYYY-MM-DD
        option_type: 'call' or 'put'
        model: 'bsm' (European) or 'baw' (American, default)
        risk_free_rate: Risk-free rate (default 0.0425)
        dividend_yield: Dividend yield (auto-fetched if omitted)

    Returns:
        Option price, intrinsic/time value, and Greeks
    """
    try:
        with PerformanceMonitor(f"price_option_{ticker}"):
            loop = asyncio.get_event_loop()

            spot = await loop.run_in_executor(
                _executor, _provider.get_underlying_price, ticker
            )
            if dividend_yield is None:
                div_yield = await loop.run_in_executor(
                    _executor, _provider.get_dividend_yield, ticker
                )
            else:
                div_yield = dividend_yield

            if spot <= 0:
                return {
                    "error": f"Could not fetch price for {ticker}",
                    "ticker": ticker,
                }

            iv = await _get_iv_from_chain(ticker, expiration, strike, option_type)
            if iv <= 0:
                iv = 0.25

            T = _days_to_expiry(expiration)

            result = core_price_option(
                spot=spot,
                strike=strike,
                time_to_expiry=T,
                risk_free_rate=risk_free_rate,
                volatility=iv,
                dividend_yield=div_yield,
                option_type=option_type,
                model=model,
            )
            result["ticker"] = ticker
            result["expiration"] = expiration
            result["implied_volatility_used"] = round(iv, 4)
            result["timestamp"] = datetime.now(tz=UTC).isoformat()
            return result
    except Exception as e:
        logger.exception("Error pricing option for %s: %s", ticker, e)
        return {
            "error": safe_error_message(e, context="pricing option"),
            "ticker": ticker,
        }


# ------------------------------------------------------------------ #
# Tool 5: Analyze Strategy
# ------------------------------------------------------------------ #


@with_logging("options_analyze_strategy")
async def analyze_options_strategy(
    ticker: str,
    strategy_type: str = "covered_call",
    expiration: str | None = None,
    legs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze an options strategy's P&L profile.

    Args:
        ticker: Stock ticker symbol
        strategy_type: covered_call, protective_put, bull_call_spread,
            bear_put_spread, iron_condor, straddle, strangle
        expiration: Expiration date (nearest if omitted)
        legs: Custom legs (auto-built from strategy_type if omitted)

    Returns:
        Max profit/loss, breakeven points, and P&L profile
    """
    try:
        with PerformanceMonitor(f"strategy_{ticker}_{strategy_type}"):
            loop = asyncio.get_event_loop()

            spot = await loop.run_in_executor(
                _executor, _provider.get_underlying_price, ticker
            )
            if spot <= 0:
                return {
                    "error": f"Could not fetch price for {ticker}",
                    "ticker": ticker,
                }

            if legs:
                # Use custom legs
                strategy_legs = [
                    {
                        "strike": leg["strike"],
                        "option_type": leg["option_type"],
                        "action": leg["action"],
                        "quantity": leg.get("quantity", 1),
                        "premium": leg.get("premium", 0),
                    }
                    for leg in legs
                ]
            else:
                # Auto-build from chain
                expirations = await loop.run_in_executor(
                    _executor, _provider.get_available_expirations, ticker
                )
                if not expirations:
                    return {"error": f"No options for {ticker}", "ticker": ticker}

                exp = (
                    expiration
                    if expiration and expiration in expirations
                    else expirations[0]
                )
                chain = await loop.run_in_executor(
                    _executor, _provider.get_option_chain, ticker, exp, 5, 50, 15.0
                )
                strategy_legs = build_common_strategy(strategy_type, chain, spot)
                if not strategy_legs:
                    return {
                        "error": f"Could not build {strategy_type} from available contracts",
                        "ticker": ticker,
                    }

            result = analyze_strategy(strategy_legs, spot)
            result["ticker"] = ticker
            result["strategy_type"] = strategy_type
            result["underlying_price"] = spot
            result["timestamp"] = datetime.now(tz=UTC).isoformat()
            return result
    except Exception as e:
        logger.exception("Error analyzing strategy for %s: %s", ticker, e)
        return {
            "error": safe_error_message(e, context="analyzing options strategy"),
            "ticker": ticker,
        }


# ------------------------------------------------------------------ #
# Tool 6: Unusual Activity
# ------------------------------------------------------------------ #


@with_logging("options_unusual_activity")
async def get_unusual_options_activity(
    ticker: str,
    volume_oi_threshold: float = 2.0,
) -> dict[str, Any]:
    """Detect unusual options activity (volume/OI spikes, put/call ratios).

    Args:
        ticker: Stock ticker symbol
        volume_oi_threshold: Volume/OI ratio to flag as unusual (default 2.0)

    Returns:
        Unusual contracts, put/call ratios, and sentiment signal
    """
    try:
        with PerformanceMonitor(f"unusual_activity_{ticker}"):
            loop = asyncio.get_event_loop()

            expirations = await loop.run_in_executor(
                _executor, _provider.get_available_expirations, ticker
            )
            if not expirations:
                return {"error": f"No options for {ticker}", "ticker": ticker}

            # Check nearest expiration with relaxed filters
            chain = await loop.run_in_executor(
                _executor,
                _provider.get_option_chain,
                ticker,
                expirations[0],
                1,
                1,
                100.0,
            )

            result = detect_unusual_activity(chain, volume_oi_threshold)
            result["timestamp"] = datetime.now(tz=UTC).isoformat()
            return result
    except Exception as e:
        logger.exception("Error detecting unusual activity for %s: %s", ticker, e)
        return {
            "error": safe_error_message(
                e, context="detecting unusual options activity"
            ),
            "ticker": ticker,
        }


# ------------------------------------------------------------------ #
# Tool 7: Hedge Portfolio
# ------------------------------------------------------------------ #


@with_logging("options_hedge_portfolio")
async def hedge_portfolio(
    ticker: str | None = None,
    risk_level: float = 50.0,
    user_id: str = "default",
    portfolio_name: str = "My Portfolio",
) -> dict[str, Any]:
    """Suggest hedging strategies for portfolio positions.

    Args:
        ticker: Specific ticker to hedge (all positions if omitted)
        risk_level: Risk tolerance 0-100 (lower = more protection)
        user_id: User identifier (default: 'default')
        portfolio_name: Portfolio name (default: 'My Portfolio')

    Returns:
        Hedging suggestions with protective puts and covered calls
    """
    try:
        with PerformanceMonitor("hedge_portfolio"):
            loop = asyncio.get_event_loop()

            # Get portfolio positions
            positions = await _get_portfolio_positions(user_id, portfolio_name)
            if not positions:
                return {
                    "error": "No portfolio positions found",
                    "user_id": user_id,
                    "portfolio_name": portfolio_name,
                }

            # Filter to specific ticker if requested
            if ticker:
                positions = [p for p in positions if p["ticker"] == ticker.upper()]
                if not positions:
                    return {
                        "error": f"No position found for {ticker}",
                        "ticker": ticker,
                    }

            # Fetch options chains for each position
            chains: dict[str, dict[str, Any]] = {}
            for pos in positions:
                t = pos["ticker"]
                expirations = await loop.run_in_executor(
                    _executor, _provider.get_available_expirations, t
                )
                if expirations:
                    # Use ~30 day expiration for hedging
                    target_exp = expirations[min(1, len(expirations) - 1)]
                    chain = await loop.run_in_executor(
                        _executor,
                        _provider.get_option_chain,
                        t,
                        target_exp,
                        5,
                        50,
                        15.0,
                    )
                    chains[t] = chain

            result = suggest_hedges(positions, chains, risk_level)
            result["timestamp"] = datetime.now(tz=UTC).isoformat()
            return result
    except Exception as e:
        logger.exception("Error generating hedge suggestions: %s", e)
        return {"error": safe_error_message(e, context="generating hedge suggestions")}


# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


async def _get_iv_from_chain(
    ticker: str, expiration: str, strike: float, option_type: str
) -> float:
    """Try to fetch IV for a specific contract from the chain."""
    try:
        loop = asyncio.get_event_loop()
        chain = await loop.run_in_executor(
            _executor, _provider.get_option_chain, ticker, expiration, 0, 0, 999.0
        )
        contracts = chain.get("calls" if option_type == "call" else "puts", [])
        # Find closest strike
        if contracts:
            best = min(contracts, key=lambda c: abs(c["strike"] - strike))
            iv = best.get("impliedVolatility", 0)
            if iv > 0:
                return float(iv)
    except Exception:
        pass
    return 0.0


async def _get_portfolio_positions(
    user_id: str, portfolio_name: str
) -> list[dict[str, Any]]:
    """Fetch portfolio positions from the database."""
    try:
        from maverick_mcp.data.database import get_session
        from maverick_mcp.data.models import PortfolioPosition, UserPortfolio

        with get_session() as session:
            portfolio = (
                session.query(UserPortfolio)
                .filter_by(user_id=user_id, name=portfolio_name)
                .first()
            )
            if not portfolio:
                return []

            db_positions = (
                session.query(PortfolioPosition)
                .filter_by(portfolio_id=portfolio.id)
                .all()
            )

            positions: list[dict[str, Any]] = []
            loop = asyncio.get_event_loop()

            for pos in db_positions:
                price = await loop.run_in_executor(
                    _executor, _provider.get_underlying_price, pos.ticker
                )
                positions.append(
                    {
                        "ticker": pos.ticker,
                        "shares": float(pos.shares),
                        "cost_basis": float(pos.average_cost),
                        "current_price": price,
                    }
                )

            return positions
    except Exception as e:
        logger.error("Error fetching portfolio: %s", e)
        return []
