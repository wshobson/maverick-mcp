"""Options analysis: Greeks, IV, pricing, strategy P&L, unusual activity, hedging.

All functions are pure (no I/O) and return ``dict[str, Any]``.
Data fetching is handled by :mod:`maverick_mcp.providers.options_data`.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
from blackscholes import BlackScholesCall, BlackScholesPut
from OptionsPricerLib import BaroneAdesiWhaley

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_MIN_T = 1 / 365  # 1 day minimum
_MIN_SIGMA = 0.001


def _days_to_expiry(expiration_str: str) -> float:
    """Convert a YYYY-MM-DD string to years-to-expiry (≥ 1 day)."""
    exp = datetime.strptime(expiration_str, "%Y-%m-%d").replace(tzinfo=UTC)
    now = datetime.now(tz=UTC)
    days = max((exp - now).total_seconds() / 86400, 1.0)
    return days / 365.0


def _moneyness(spot: float, strike: float, option_type: str) -> str:
    """Return 'ITM', 'ATM' or 'OTM'."""
    pct = abs(spot - strike) / spot if spot > 0 else 0
    if pct < 0.005:
        return "ATM"
    if option_type == "call":
        return "ITM" if spot > strike else "OTM"
    return "ITM" if spot < strike else "OTM"


def _find_atm_strike(strikes: list[float], spot: float) -> float:
    """Return the strike closest to *spot*."""
    if not strikes:
        return spot
    return min(strikes, key=lambda k: abs(k - spot))


def _safe_params(T: float, sigma: float) -> tuple[float, float]:
    """Clamp T and sigma to safe minimums to avoid numerical blow-ups."""
    return max(T, _MIN_T), max(sigma, _MIN_SIGMA)


# ------------------------------------------------------------------ #
# Section 1 – Greeks Calculation
# ------------------------------------------------------------------ #


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> dict[str, Any]:
    """European Greeks (1st through 3rd order) via the *blackscholes* library."""
    T, sigma = _safe_params(time_to_expiry, volatility)

    cls = BlackScholesCall if option_type == "call" else BlackScholesPut
    opt = cls(S=spot, K=strike, T=T, r=risk_free_rate, sigma=sigma, q=dividend_yield)

    return {
        "price": round(opt.price(), 4),
        # 1st order
        "delta": round(opt.delta(), 6),
        "gamma": round(opt.gamma(), 6),
        "theta": round(opt.theta(), 4),
        "vega": round(opt.vega(), 4),
        "rho": round(opt.rho(), 4),
        # 2nd order
        "charm": round(opt.charm(), 6),
        "vanna": round(opt.vanna(), 6),
        "vomma": round(opt.vomma(), 6),
        # 3rd order
        "speed": round(opt.speed(), 8),
        "color": round(opt.color(), 8),
        "zomma": round(opt.zomma(), 8),
        "ultima": round(opt.ultima(), 6),
        # Meta
        "option_type": option_type,
        "model": "black_scholes",
        "moneyness": _moneyness(spot, strike, option_type),
    }


def calculate_greeks_american(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> dict[str, Any]:
    """American-style Greeks via Barone-Adesi Whaley approximation."""
    T, sigma = _safe_params(time_to_expiry, volatility)
    ot = "calls" if option_type == "call" else "puts"

    try:
        price = BaroneAdesiWhaley.price(
            sigma, spot, strike, T, risk_free_rate, dividend_yield, ot
        )
        delta = BaroneAdesiWhaley.calculate_delta(
            sigma, spot, strike, T, risk_free_rate, dividend_yield, ot
        )
        gamma = BaroneAdesiWhaley.calculate_gamma(
            sigma, spot, strike, T, risk_free_rate, dividend_yield, ot
        )
        theta = BaroneAdesiWhaley.calculate_theta(
            sigma, spot, strike, T, risk_free_rate, dividend_yield, ot
        )
        vega = BaroneAdesiWhaley.calculate_vega(
            sigma, spot, strike, T, risk_free_rate, dividend_yield, ot
        )
        rho = BaroneAdesiWhaley.calculate_rho(
            sigma, spot, strike, T, risk_free_rate, dividend_yield, ot
        )
    except Exception as e:
        # BAW can fail for extreme parameters; fall back to BSM values
        bsm = calculate_greeks(
            spot, strike, T, risk_free_rate, volatility, dividend_yield, option_type
        )
        return {**bsm, "model": "black_scholes_fallback", "baw_error": str(e)}

    return {
        "price": round(price, 4),
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "rho": round(rho, 4),
        "option_type": option_type,
        "model": "barone_adesi_whaley",
        "moneyness": _moneyness(spot, strike, option_type),
    }


def calculate_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> dict[str, Any]:
    """Combined European (full higher-order) + American (BAW) Greeks."""
    european = calculate_greeks(
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
        option_type,
    )
    american = calculate_greeks_american(
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
        option_type,
    )

    early_exercise = round(american["price"] - european["price"], 4)

    return {
        "european": european,
        "american": american,
        "early_exercise_premium": max(early_exercise, 0.0),
        "spot": spot,
        "strike": strike,
        "time_to_expiry_years": round(time_to_expiry, 6),
        "risk_free_rate": risk_free_rate,
        "volatility": round(volatility, 4),
        "dividend_yield": dividend_yield,
        "option_type": option_type,
    }


# ------------------------------------------------------------------ #
# Section 2 – Implied Volatility Analysis
# ------------------------------------------------------------------ #


def analyze_iv_skew(
    chain_data: dict[str, Any],
    spot_price: float,
) -> dict[str, Any]:
    """Analyse IV by strike for a single expiration.

    Returns skew classification: normal, reverse, smile, or flat.
    """
    calls = chain_data.get("calls", [])
    puts = chain_data.get("puts", [])

    call_strikes = [c["strike"] for c in calls if c.get("impliedVolatility", 0) > 0]
    call_ivs = [
        c["impliedVolatility"] for c in calls if c.get("impliedVolatility", 0) > 0
    ]
    put_strikes = [p["strike"] for p in puts if p.get("impliedVolatility", 0) > 0]
    put_ivs = [
        p["impliedVolatility"] for p in puts if p.get("impliedVolatility", 0) > 0
    ]

    # ATM IV
    all_contracts = calls + puts
    atm_strike = _find_atm_strike([c["strike"] for c in all_contracts], spot_price)
    atm_ivs = [
        c["impliedVolatility"]
        for c in all_contracts
        if abs(c["strike"] - atm_strike) < 0.01 and c.get("impliedVolatility", 0) > 0
    ]
    atm_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else 0.0

    # Determine skew type using OTM puts vs OTM calls
    otm_put_ivs = [
        p["impliedVolatility"]
        for p in puts
        if p["strike"] < spot_price and p.get("impliedVolatility", 0) > 0
    ]
    otm_call_ivs = [
        c["impliedVolatility"]
        for c in calls
        if c["strike"] > spot_price and c.get("impliedVolatility", 0) > 0
    ]

    avg_otm_put_iv = sum(otm_put_ivs) / len(otm_put_ivs) if otm_put_ivs else 0
    avg_otm_call_iv = sum(otm_call_ivs) / len(otm_call_ivs) if otm_call_ivs else 0

    if atm_iv == 0:
        skew_type = "insufficient_data"
    elif avg_otm_put_iv > atm_iv * 1.05 and avg_otm_call_iv > atm_iv * 1.05:
        skew_type = "smile"
    elif avg_otm_put_iv > atm_iv * 1.05:
        skew_type = "normal_skew"  # typical equity skew
    elif avg_otm_call_iv > atm_iv * 1.05:
        skew_type = "reverse_skew"
    else:
        skew_type = "flat"

    # Skew slope (linear regression of put IV vs strike)
    skew_slope = 0.0
    if len(put_strikes) >= 3:
        x = np.array(put_strikes)
        y = np.array(put_ivs)
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            skew_slope = float(coeffs[0])

    return {
        "atm_iv": round(atm_iv, 4),
        "atm_strike": atm_strike,
        "skew_type": skew_type,
        "skew_slope": round(skew_slope, 6),
        "avg_otm_put_iv": round(avg_otm_put_iv, 4),
        "avg_otm_call_iv": round(avg_otm_call_iv, 4),
        "otm_put_iv_premium": round(avg_otm_put_iv - atm_iv, 4) if atm_iv > 0 else 0,
        "call_strikes": call_strikes,
        "call_ivs": [round(iv, 4) for iv in call_ivs],
        "put_strikes": put_strikes,
        "put_ivs": [round(iv, 4) for iv in put_ivs],
        "expiration": chain_data.get("expiration", ""),
    }


def analyze_iv_term_structure(
    chains_by_expiry: dict[str, dict[str, Any]],
    spot_price: float,
) -> dict[str, Any]:
    """ATM IV across multiple expirations → contango / backwardation."""
    expirations: list[str] = []
    atm_ivs: list[float] = []

    for exp in sorted(chains_by_expiry.keys()):
        chain = chains_by_expiry[exp]
        skew = analyze_iv_skew(chain, spot_price)
        if skew["atm_iv"] > 0:
            expirations.append(exp)
            atm_ivs.append(skew["atm_iv"])

    # Classify shape
    if len(atm_ivs) < 2:
        shape = "insufficient_data"
    elif all(atm_ivs[i] <= atm_ivs[i + 1] for i in range(len(atm_ivs) - 1)):
        shape = "contango"  # IV rises with maturity
    elif all(atm_ivs[i] >= atm_ivs[i + 1] for i in range(len(atm_ivs) - 1)):
        shape = "backwardation"  # IV falls with maturity
    else:
        shape = "mixed"

    return {
        "expirations": expirations,
        "atm_ivs": [round(iv, 4) for iv in atm_ivs],
        "term_structure_shape": shape,
    }


def calculate_iv_percentile(
    current_iv: float, historical_ivs: list[float]
) -> dict[str, Any]:
    """IV rank and percentile from historical IV values."""
    if not historical_ivs:
        return {
            "iv_rank": 0.0,
            "iv_percentile": 0.0,
            "current_iv": round(current_iv, 4),
            "52w_high_iv": 0.0,
            "52w_low_iv": 0.0,
        }

    high_iv = max(historical_ivs)
    low_iv = min(historical_ivs)
    iv_range = high_iv - low_iv

    iv_rank = ((current_iv - low_iv) / iv_range * 100) if iv_range > 0 else 50.0
    below = sum(1 for iv in historical_ivs if iv < current_iv)
    iv_percentile = below / len(historical_ivs) * 100

    return {
        "iv_rank": round(iv_rank, 2),
        "iv_percentile": round(iv_percentile, 2),
        "current_iv": round(current_iv, 4),
        "52w_high_iv": round(high_iv, 4),
        "52w_low_iv": round(low_iv, 4),
    }


# ------------------------------------------------------------------ #
# Section 3 – Option Pricing
# ------------------------------------------------------------------ #


def price_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    model: str = "baw",
) -> dict[str, Any]:
    """Price an option with BSM or BAW and return Greeks."""
    if model == "bsm":
        greeks = calculate_greeks(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
            option_type,
        )
    else:
        greeks = calculate_greeks_american(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
            option_type,
        )

    price = greeks["price"]
    intrinsic = (
        max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    )
    time_value = max(price - intrinsic, 0.0)

    return {
        "model": model,
        "price": round(price, 4),
        "intrinsic_value": round(intrinsic, 4),
        "time_value": round(time_value, 4),
        "greeks": greeks,
        "inputs": {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": round(time_to_expiry, 6),
            "risk_free_rate": risk_free_rate,
            "volatility": round(volatility, 4),
            "dividend_yield": dividend_yield,
            "option_type": option_type,
        },
    }


# ------------------------------------------------------------------ #
# Section 4 – Strategy Analysis
# ------------------------------------------------------------------ #


def analyze_strategy(
    legs: list[dict[str, Any]],
    spot_price: float,
    risk_free_rate: float = 0.0425,
) -> dict[str, Any]:
    """Compute P&L profile for a multi-leg options strategy.

    Each leg dict must have:
        strike, option_type, action (buy/sell), quantity, premium
    """
    if not legs:
        return {"error": "No strategy legs provided"}

    # Build price range: spot ±30 %
    low = spot_price * 0.70
    high = spot_price * 1.30
    prices = np.linspace(low, high, 100)

    # Net premium (positive = credit received)
    net_premium = 0.0
    for leg in legs:
        sign = -1 if leg["action"] == "buy" else 1
        net_premium += sign * leg["premium"] * leg["quantity"] * 100

    # Compute P&L at expiry for each underlying price
    pnl_values: list[float] = []
    for p in prices:
        total = 0.0
        for leg in legs:
            strike = leg["strike"]
            qty = leg["quantity"]
            premium = leg["premium"]
            if leg["option_type"] == "call":
                intrinsic = max(p - strike, 0.0)
            else:
                intrinsic = max(strike - p, 0.0)

            if leg["action"] == "buy":
                total += (intrinsic - premium) * qty * 100
            else:
                total += (premium - intrinsic) * qty * 100

        pnl_values.append(round(total, 2))

    pnl_arr = np.array(pnl_values)
    max_profit = float(np.max(pnl_arr))
    max_loss = float(np.min(pnl_arr))

    # Detect if max profit / loss is at the boundary → could be unlimited
    max_profit_str: float | str = max_profit
    max_loss_str: float | str = max_loss
    if pnl_arr[-1] == max_profit and pnl_arr[-2] < pnl_arr[-1]:
        max_profit_str = "unlimited"
    if pnl_arr[0] == max_loss and pnl_arr[1] > pnl_arr[0]:
        max_loss_str = "unlimited"

    # Breakeven points (where P&L crosses zero)
    breakevens: list[float] = []
    for i in range(len(pnl_arr) - 1):
        if pnl_arr[i] * pnl_arr[i + 1] < 0:
            # Linear interpolation
            frac = abs(pnl_arr[i]) / (abs(pnl_arr[i]) + abs(pnl_arr[i + 1]))
            be = float(prices[i] + frac * (prices[i + 1] - prices[i]))
            breakevens.append(round(be, 2))

    # Risk/reward ratio
    if isinstance(max_profit_str, str) or isinstance(max_loss_str, str):
        risk_reward = None
    elif max_loss == 0:
        risk_reward = float("inf")
    else:
        risk_reward = round(abs(max_profit / max_loss), 2)

    return {
        "legs": legs,
        "spot_price": spot_price,
        "net_premium": round(net_premium, 2),
        "max_profit": max_profit_str,
        "max_loss": max_loss_str,
        "breakeven_points": breakevens,
        "risk_reward_ratio": risk_reward,
        "pnl_at_expiry": [
            {"price": round(float(prices[i]), 2), "pnl": pnl_values[i]}
            for i in range(0, len(prices), 5)  # every 5th point to keep response small
        ],
    }


def build_common_strategy(
    strategy_type: str,
    chain_data: dict[str, Any],
    spot_price: float,
) -> list[dict[str, Any]]:
    """Auto-select strikes/premiums for common strategies from chain data."""
    calls = sorted(chain_data.get("calls", []), key=lambda c: c["strike"])
    puts = sorted(chain_data.get("puts", []), key=lambda c: c["strike"])

    if not calls and not puts:
        return []

    atm_call_strike = (
        _find_atm_strike([c["strike"] for c in calls], spot_price)
        if calls
        else spot_price
    )
    atm_put_strike = (
        _find_atm_strike([p["strike"] for p in puts], spot_price)
        if puts
        else spot_price
    )

    def _find_contract(contracts: list[dict], target_strike: float) -> dict[str, Any]:
        """Find the contract closest to target_strike."""
        if not contracts:
            return {"strike": target_strike, "mid": 0.0}
        return min(contracts, key=lambda c: abs(c["strike"] - target_strike))

    def _otm_call(pct: float = 1.03) -> dict[str, Any]:
        target = spot_price * pct
        return _find_contract(calls, target)

    def _otm_put(pct: float = 0.97) -> dict[str, Any]:
        target = spot_price * pct
        return _find_contract(puts, target)

    legs: list[dict[str, Any]] = []

    if strategy_type == "covered_call":
        cc = _otm_call(1.05)
        legs = [
            {
                "strike": cc["strike"],
                "option_type": "call",
                "action": "sell",
                "quantity": 1,
                "premium": cc.get("mid", 0),
            },
        ]

    elif strategy_type == "protective_put":
        pp = _otm_put(0.95)
        legs = [
            {
                "strike": pp["strike"],
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": pp.get("mid", 0),
            },
        ]

    elif strategy_type == "bull_call_spread":
        lower = _find_contract(calls, atm_call_strike)
        upper = _otm_call(1.05)
        legs = [
            {
                "strike": lower["strike"],
                "option_type": "call",
                "action": "buy",
                "quantity": 1,
                "premium": lower.get("mid", 0),
            },
            {
                "strike": upper["strike"],
                "option_type": "call",
                "action": "sell",
                "quantity": 1,
                "premium": upper.get("mid", 0),
            },
        ]

    elif strategy_type == "bear_put_spread":
        upper = _find_contract(puts, atm_put_strike)
        lower = _otm_put(0.95)
        legs = [
            {
                "strike": upper["strike"],
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": upper.get("mid", 0),
            },
            {
                "strike": lower["strike"],
                "option_type": "put",
                "action": "sell",
                "quantity": 1,
                "premium": lower.get("mid", 0),
            },
        ]

    elif strategy_type == "iron_condor":
        otm_put_low = _otm_put(0.93)
        otm_put_high = _otm_put(0.97)
        otm_call_low = _otm_call(1.03)
        otm_call_high = _otm_call(1.07)
        legs = [
            {
                "strike": otm_put_low["strike"],
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": otm_put_low.get("mid", 0),
            },
            {
                "strike": otm_put_high["strike"],
                "option_type": "put",
                "action": "sell",
                "quantity": 1,
                "premium": otm_put_high.get("mid", 0),
            },
            {
                "strike": otm_call_low["strike"],
                "option_type": "call",
                "action": "sell",
                "quantity": 1,
                "premium": otm_call_low.get("mid", 0),
            },
            {
                "strike": otm_call_high["strike"],
                "option_type": "call",
                "action": "buy",
                "quantity": 1,
                "premium": otm_call_high.get("mid", 0),
            },
        ]

    elif strategy_type == "straddle":
        atm_c = _find_contract(calls, atm_call_strike)
        atm_p = _find_contract(puts, atm_put_strike)
        legs = [
            {
                "strike": atm_c["strike"],
                "option_type": "call",
                "action": "buy",
                "quantity": 1,
                "premium": atm_c.get("mid", 0),
            },
            {
                "strike": atm_p["strike"],
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": atm_p.get("mid", 0),
            },
        ]

    elif strategy_type == "strangle":
        otm_c = _otm_call(1.05)
        otm_p = _otm_put(0.95)
        legs = [
            {
                "strike": otm_c["strike"],
                "option_type": "call",
                "action": "buy",
                "quantity": 1,
                "premium": otm_c.get("mid", 0),
            },
            {
                "strike": otm_p["strike"],
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": otm_p.get("mid", 0),
            },
        ]

    return legs


# ------------------------------------------------------------------ #
# Section 5 – Unusual Activity Detection
# ------------------------------------------------------------------ #


def detect_unusual_activity(
    chain_data: dict[str, Any],
    volume_oi_threshold: float = 2.0,
) -> dict[str, Any]:
    """Flag contracts with unusual volume relative to open interest."""
    calls = chain_data.get("calls", [])
    puts = chain_data.get("puts", [])

    unusual: list[dict[str, Any]] = []
    total_call_vol = 0
    total_put_vol = 0
    total_call_oi = 0
    total_put_oi = 0

    for c in calls:
        vol = c.get("volume", 0)
        oi = c.get("openInterest", 0)
        total_call_vol += vol
        total_call_oi += oi
        if oi > 0 and vol / oi > volume_oi_threshold:
            unusual.append(
                {
                    "contractSymbol": c.get("contractSymbol", ""),
                    "type": "call",
                    "strike": c["strike"],
                    "volume": vol,
                    "openInterest": oi,
                    "volume_oi_ratio": round(vol / oi, 2),
                    "impliedVolatility": round(c.get("impliedVolatility", 0), 4),
                }
            )

    for p in puts:
        vol = p.get("volume", 0)
        oi = p.get("openInterest", 0)
        total_put_vol += vol
        total_put_oi += oi
        if oi > 0 and vol / oi > volume_oi_threshold:
            unusual.append(
                {
                    "contractSymbol": p.get("contractSymbol", ""),
                    "type": "put",
                    "strike": p["strike"],
                    "volume": vol,
                    "openInterest": oi,
                    "volume_oi_ratio": round(vol / oi, 2),
                    "impliedVolatility": round(p.get("impliedVolatility", 0), 4),
                }
            )

    # Sort by volume/OI ratio descending
    unusual.sort(key=lambda x: x["volume_oi_ratio"], reverse=True)

    # Put/call ratios
    pc_volume = (total_put_vol / total_call_vol) if total_call_vol > 0 else 0
    pc_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0

    # Sentiment signal
    if pc_volume > 1.5:
        sentiment = "bearish"
    elif pc_volume < 0.5:
        sentiment = "bullish"
    else:
        sentiment = "neutral"

    return {
        "unusual_contracts": unusual[:20],  # top 20
        "unusual_count": len(unusual),
        "put_call_volume_ratio": round(pc_volume, 3),
        "put_call_oi_ratio": round(pc_oi, 3),
        "total_call_volume": total_call_vol,
        "total_put_volume": total_put_vol,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "sentiment_signal": sentiment,
        "ticker": chain_data.get("ticker", ""),
        "expiration": chain_data.get("expiration", ""),
    }


# ------------------------------------------------------------------ #
# Section 6 – Portfolio Hedging
# ------------------------------------------------------------------ #


def suggest_hedges(
    positions: list[dict[str, Any]],
    chains: dict[str, dict[str, Any]],
    risk_level: float = 50.0,
) -> dict[str, Any]:
    """Suggest protective puts / covered calls for portfolio positions.

    *positions*: list of dicts with keys ``ticker``, ``shares``, ``current_price``
    *chains*: ``{ticker: chain_data}`` from :class:`OptionsDataProvider`
    *risk_level*: 0–100, lower = more conservative (buys more protection)
    """
    suggestions: list[dict[str, Any]] = []
    total_cost = 0.0
    total_delta_before = 0.0
    total_delta_after = 0.0

    # Protection level: conservative (0) → 5% OTM puts, aggressive (100) → 15% OTM puts
    otm_pct = 0.95 - (risk_level / 100) * 0.10  # 0.95 → 0.85

    for pos in positions:
        ticker = pos.get("ticker", "")
        shares = pos.get("shares", 0)
        price = pos.get("current_price", 0)
        chain = chains.get(ticker)

        if not chain or shares <= 0 or price <= 0:
            continue

        total_delta_before += shares  # stock delta = 1 per share

        puts = sorted(chain.get("puts", []), key=lambda p: p["strike"])
        calls = sorted(chain.get("calls", []), key=lambda c: c["strike"])

        # Protective put suggestion
        if puts:
            target_put_strike = price * otm_pct
            best_put = min(puts, key=lambda p: abs(p["strike"] - target_put_strike))
            contracts_needed = math.ceil(shares / 100)
            put_cost = (
                best_put.get("mid", best_put.get("lastPrice", 0))
                * contracts_needed
                * 100
            )

            suggestions.append(
                {
                    "ticker": ticker,
                    "strategy": "protective_put",
                    "strike": best_put["strike"],
                    "premium_per_share": best_put.get("mid", 0),
                    "contracts": contracts_needed,
                    "total_cost": round(put_cost, 2),
                    "protection_level": round(
                        (1 - best_put["strike"] / price) * 100, 1
                    ),
                    "max_loss_with_hedge": round(
                        (price - best_put["strike"]) * shares + put_cost, 2
                    ),
                }
            )
            total_cost += put_cost
            # Approximate delta reduction from puts
            total_delta_after += (
                shares - contracts_needed * 100 * 0.3
            )  # rough put delta

        # Covered call suggestion (for income on larger positions)
        if calls and shares >= 100:
            target_call_strike = price * 1.05
            best_call = min(calls, key=lambda c: abs(c["strike"] - target_call_strike))
            cc_contracts = shares // 100
            cc_income = best_call.get("mid", 0) * cc_contracts * 100

            suggestions.append(
                {
                    "ticker": ticker,
                    "strategy": "covered_call",
                    "strike": best_call["strike"],
                    "premium_per_share": best_call.get("mid", 0),
                    "contracts": cc_contracts,
                    "total_income": round(cc_income, 2),
                    "upside_cap": round((best_call["strike"] / price - 1) * 100, 1),
                }
            )
            total_cost -= cc_income  # income offsets cost

    return {
        "hedging_suggestions": suggestions,
        "total_hedge_cost": round(total_cost, 2),
        "portfolio_delta_before": round(total_delta_before, 2),
        "portfolio_delta_after": round(total_delta_after, 2),
        "risk_level": risk_level,
        "positions_analyzed": len(positions),
    }
