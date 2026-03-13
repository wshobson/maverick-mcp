"""Unit tests for options analysis core module and data provider."""

import time
from unittest.mock import MagicMock, patch

import pytest

from maverick_mcp.core.options_analysis import (
    _days_to_expiry,
    _find_atm_strike,
    _moneyness,
    analyze_iv_skew,
    analyze_iv_term_structure,
    analyze_strategy,
    build_common_strategy,
    calculate_all_greeks,
    calculate_greeks,
    calculate_greeks_american,
    calculate_iv_percentile,
    detect_unusual_activity,
    price_option,
    suggest_hedges,
)
from maverick_mcp.providers.options_data import OptionsDataProvider


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class TestHelpers:
    def test_days_to_expiry_future_date(self):
        t = _days_to_expiry("2030-01-01")
        assert t > 0

    def test_days_to_expiry_past_date_clamps(self):
        t = _days_to_expiry("2020-01-01")
        assert t >= 1 / 365  # at least 1 day

    def test_moneyness_call_itm(self):
        assert _moneyness(110, 100, "call") == "ITM"

    def test_moneyness_call_otm(self):
        assert _moneyness(90, 100, "call") == "OTM"

    def test_moneyness_put_itm(self):
        assert _moneyness(90, 100, "put") == "ITM"

    def test_moneyness_atm(self):
        assert _moneyness(100, 100, "call") == "ATM"

    def test_find_atm_strike(self):
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        assert _find_atm_strike(strikes, 102.0) == 100.0

    def test_find_atm_strike_empty(self):
        assert _find_atm_strike([], 100.0) == 100.0


# ------------------------------------------------------------------ #
# Greeks Calculation
# ------------------------------------------------------------------ #


class TestGreeksCalculation:
    def test_call_greeks_first_order(self):
        result = calculate_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="call",
        )
        assert 0 < result["delta"] < 1
        assert result["gamma"] > 0
        assert result["theta"] < 0
        assert result["vega"] > 0
        assert result["price"] > 0

    def test_put_greeks_first_order(self):
        result = calculate_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="put",
        )
        assert -1 < result["delta"] < 0
        assert result["gamma"] > 0

    def test_higher_order_greeks_present(self):
        result = calculate_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
        )
        for key in ("charm", "vanna", "vomma", "speed", "color", "zomma", "ultima"):
            assert key in result

    def test_american_price_gte_european(self):
        eu = calculate_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.02,
            option_type="put",
        )
        am = calculate_greeks_american(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.02,
            option_type="put",
        )
        assert am["price"] >= eu["price"] - 0.01  # allow tiny float tolerance

    def test_all_greeks_combined(self):
        result = calculate_all_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
        )
        assert "european" in result
        assert "american" in result
        assert result["early_exercise_premium"] >= 0

    def test_zero_time_no_error(self):
        result = calculate_greeks(
            spot=100,
            strike=100,
            time_to_expiry=0.0,
            risk_free_rate=0.05,
            volatility=0.2,
        )
        assert "price" in result


# ------------------------------------------------------------------ #
# IV Analysis
# ------------------------------------------------------------------ #

_MOCK_CHAIN = {
    "calls": [
        {"strike": 90, "impliedVolatility": 0.30, "volume": 100, "openInterest": 500},
        {"strike": 100, "impliedVolatility": 0.25, "volume": 200, "openInterest": 1000},
        {"strike": 110, "impliedVolatility": 0.28, "volume": 150, "openInterest": 800},
    ],
    "puts": [
        {"strike": 90, "impliedVolatility": 0.35, "volume": 80, "openInterest": 400},
        {"strike": 100, "impliedVolatility": 0.25, "volume": 180, "openInterest": 900},
        {"strike": 110, "impliedVolatility": 0.22, "volume": 50, "openInterest": 300},
    ],
    "underlying_price": 100,
    "expiration": "2026-06-19",
    "ticker": "TEST",
}


class TestIVAnalysis:
    def test_iv_skew_detection(self):
        result = analyze_iv_skew(_MOCK_CHAIN, 100.0)
        assert result["atm_iv"] > 0
        assert result["skew_type"] in (
            "normal_skew",
            "reverse_skew",
            "smile",
            "flat",
            "insufficient_data",
        )

    def test_iv_term_structure(self):
        chains = {
            "2026-04-17": _MOCK_CHAIN,
            "2026-06-19": {**_MOCK_CHAIN, "expiration": "2026-06-19"},
        }
        result = analyze_iv_term_structure(chains, 100.0)
        assert "term_structure_shape" in result
        assert len(result["expirations"]) <= 2

    def test_iv_percentile(self):
        historical = [0.15, 0.20, 0.25, 0.30, 0.35]
        result = calculate_iv_percentile(0.28, historical)
        assert 0 <= result["iv_rank"] <= 100
        assert 0 <= result["iv_percentile"] <= 100
        assert result["52w_high_iv"] == 0.35
        assert result["52w_low_iv"] == 0.15


# ------------------------------------------------------------------ #
# Strategy Analysis
# ------------------------------------------------------------------ #


class TestStrategyAnalysis:
    def test_covered_call_max_profit_capped(self):
        legs = [
            {
                "strike": 105,
                "option_type": "call",
                "action": "sell",
                "quantity": 1,
                "premium": 3.0,
            },
        ]
        result = analyze_strategy(legs, spot_price=100)
        assert result["max_profit"] is not None

    def test_protective_put_max_loss_limited(self):
        legs = [
            {
                "strike": 95,
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": 2.0,
            },
        ]
        result = analyze_strategy(legs, spot_price=100)
        assert isinstance(result["max_loss"], (int, float, str))

    def test_iron_condor_defined_risk(self):
        legs = [
            {
                "strike": 90,
                "option_type": "put",
                "action": "buy",
                "quantity": 1,
                "premium": 0.5,
            },
            {
                "strike": 95,
                "option_type": "put",
                "action": "sell",
                "quantity": 1,
                "premium": 1.5,
            },
            {
                "strike": 105,
                "option_type": "call",
                "action": "sell",
                "quantity": 1,
                "premium": 1.5,
            },
            {
                "strike": 110,
                "option_type": "call",
                "action": "buy",
                "quantity": 1,
                "premium": 0.5,
            },
        ]
        result = analyze_strategy(legs, spot_price=100)
        assert result["max_profit"] is not None
        assert result["max_loss"] is not None

    def test_breakeven_calculation(self):
        legs = [
            {
                "strike": 100,
                "option_type": "call",
                "action": "buy",
                "quantity": 1,
                "premium": 5.0,
            },
            {
                "strike": 110,
                "option_type": "call",
                "action": "sell",
                "quantity": 1,
                "premium": 2.0,
            },
        ]
        result = analyze_strategy(legs, spot_price=100)
        assert len(result["breakeven_points"]) >= 1

    def test_build_common_strategy_covered_call(self):
        legs = build_common_strategy("covered_call", _MOCK_CHAIN, 100.0)
        assert len(legs) == 1
        assert legs[0]["action"] == "sell"
        assert legs[0]["option_type"] == "call"

    def test_build_common_strategy_straddle(self):
        legs = build_common_strategy("straddle", _MOCK_CHAIN, 100.0)
        assert len(legs) == 2

    def test_empty_chain(self):
        legs = build_common_strategy("covered_call", {"calls": [], "puts": []}, 100.0)
        assert legs == []


# ------------------------------------------------------------------ #
# Unusual Activity
# ------------------------------------------------------------------ #


class TestUnusualActivity:
    def test_high_volume_oi_detection(self):
        chain = {
            "calls": [
                {
                    "contractSymbol": "C1",
                    "strike": 100,
                    "volume": 5000,
                    "openInterest": 100,
                    "impliedVolatility": 0.3,
                },
                {
                    "contractSymbol": "C2",
                    "strike": 105,
                    "volume": 50,
                    "openInterest": 500,
                    "impliedVolatility": 0.25,
                },
            ],
            "puts": [],
            "ticker": "TEST",
            "expiration": "2026-04-17",
        }
        result = detect_unusual_activity(chain, volume_oi_threshold=2.0)
        assert result["unusual_count"] >= 1
        assert result["unusual_contracts"][0]["volume_oi_ratio"] > 2.0

    def test_put_call_ratio(self):
        chain = {
            "calls": [
                {
                    "strike": 100,
                    "volume": 1000,
                    "openInterest": 5000,
                    "impliedVolatility": 0.25,
                }
            ],
            "puts": [
                {
                    "strike": 100,
                    "volume": 3000,
                    "openInterest": 5000,
                    "impliedVolatility": 0.30,
                }
            ],
            "ticker": "TEST",
            "expiration": "2026-04-17",
        }
        result = detect_unusual_activity(chain)
        assert result["put_call_volume_ratio"] == 3.0

    def test_sentiment_signal(self):
        chain = {
            "calls": [
                {
                    "strike": 100,
                    "volume": 100,
                    "openInterest": 500,
                    "impliedVolatility": 0.25,
                }
            ],
            "puts": [
                {
                    "strike": 100,
                    "volume": 200,
                    "openInterest": 500,
                    "impliedVolatility": 0.30,
                }
            ],
            "ticker": "TEST",
            "expiration": "2026-04-17",
        }
        result = detect_unusual_activity(chain)
        assert result["sentiment_signal"] in ("bullish", "bearish", "neutral")


# ------------------------------------------------------------------ #
# Data Provider Cache
# ------------------------------------------------------------------ #


class TestOptionsDataProvider:
    def test_cache_hit(self):
        provider = OptionsDataProvider(cache_ttl_seconds=60)
        provider._set_cached("test_key", {"data": 1})
        assert provider._get_cached("test_key") == {"data": 1}

    def test_cache_miss(self):
        provider = OptionsDataProvider(cache_ttl_seconds=60)
        assert provider._get_cached("nonexistent") is None

    def test_cache_expiry(self):
        provider = OptionsDataProvider(cache_ttl_seconds=0)  # immediate expiry
        provider._set_cached("test_key", {"data": 1})
        time.sleep(0.01)
        assert provider._get_cached("test_key") is None

    def test_liquidity_filtering(self):
        raw = {
            "calls": [
                {
                    "strike": 100,
                    "volume": 5,
                    "openInterest": 50,
                    "bidAskSpreadPct": 5.0,
                },
                {
                    "strike": 105,
                    "volume": 100,
                    "openInterest": 500,
                    "bidAskSpreadPct": 2.0,
                },
            ],
            "puts": [],
            "underlying_price": 100,
            "expiration": "2026-04-17",
            "ticker": "TEST",
        }
        filtered = OptionsDataProvider._apply_filters(
            raw, min_volume=10, min_open_interest=100, max_bid_ask_spread_pct=10.0
        )
        assert len(filtered["calls"]) == 1
        assert filtered["calls"][0]["strike"] == 105


# ------------------------------------------------------------------ #
# Portfolio Hedging
# ------------------------------------------------------------------ #


class TestPortfolioHedging:
    def test_protective_put_suggestion(self):
        positions = [{"ticker": "AAPL", "shares": 100, "current_price": 180.0}]
        chains = {
            "AAPL": {
                "puts": [
                    {"strike": 170, "mid": 3.0, "lastPrice": 3.0},
                    {"strike": 165, "mid": 2.0, "lastPrice": 2.0},
                ],
                "calls": [
                    {"strike": 190, "mid": 4.0, "lastPrice": 4.0},
                ],
            }
        }
        result = suggest_hedges(positions, chains, risk_level=50.0)
        assert len(result["hedging_suggestions"]) >= 1
        strategies = [s["strategy"] for s in result["hedging_suggestions"]]
        assert "protective_put" in strategies

    def test_covered_call_suggestion(self):
        positions = [{"ticker": "AAPL", "shares": 200, "current_price": 180.0}]
        chains = {
            "AAPL": {
                "puts": [{"strike": 170, "mid": 3.0}],
                "calls": [{"strike": 190, "mid": 4.0}],
            }
        }
        result = suggest_hedges(positions, chains, risk_level=50.0)
        strategies = [s["strategy"] for s in result["hedging_suggestions"]]
        assert "covered_call" in strategies

    def test_empty_portfolio(self):
        result = suggest_hedges([], {}, risk_level=50.0)
        assert result["hedging_suggestions"] == []
        assert result["positions_analyzed"] == 0


# ------------------------------------------------------------------ #
# Option Pricing
# ------------------------------------------------------------------ #


class TestPricing:
    def test_bsm_pricing(self):
        result = price_option(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            model="bsm",
        )
        assert result["model"] == "bsm"
        assert result["price"] > 0
        assert result["intrinsic_value"] >= 0
        assert result["time_value"] >= 0

    def test_baw_pricing(self):
        result = price_option(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.2,
            model="baw",
        )
        assert result["model"] == "baw"
        assert result["price"] > 0
