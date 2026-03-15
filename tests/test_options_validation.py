"""Unit tests for options validation models."""

import pytest
from pydantic import ValidationError

from maverick_mcp.validation.options import (
    GreeksRequest,
    HedgePortfolioRequest,
    IVAnalysisRequest,
    OptionPriceRequest,
    OptionsChainRequest,
    StrategyAnalysisRequest,
    StrategyLeg,
    UnusualActivityRequest,
)


class TestOptionsChainRequest:
    def test_valid_minimal(self):
        req = OptionsChainRequest(ticker="AAPL")
        assert req.ticker == "AAPL"
        assert req.expiration is None
        assert req.min_volume == 10

    def test_valid_full(self):
        req = OptionsChainRequest(
            ticker="MSFT",
            expiration="2026-04-17",
            min_volume=50,
            min_open_interest=500,
            max_bid_ask_spread_pct=5.0,
        )
        assert req.ticker == "MSFT"

    def test_invalid_ticker_rejected(self):
        with pytest.raises(ValidationError):
            OptionsChainRequest(ticker="")


class TestGreeksRequest:
    def test_valid(self):
        req = GreeksRequest(
            ticker="AAPL", strike=200.0, expiration="2026-04-17", option_type="call"
        )
        assert req.ticker == "AAPL"
        assert req.strike == 200.0
        assert req.risk_free_rate == 0.0425

    def test_invalid_option_type(self):
        with pytest.raises(ValidationError):
            GreeksRequest(
                ticker="AAPL",
                strike=200.0,
                expiration="2026-04-17",
                option_type="strangle",
            )

    def test_negative_strike_rejected(self):
        with pytest.raises(ValidationError):
            GreeksRequest(ticker="AAPL", strike=-10.0, expiration="2026-04-17")


class TestOptionPriceRequest:
    def test_valid_baw(self):
        req = OptionPriceRequest(
            ticker="AAPL", strike=200.0, expiration="2026-04-17", model="baw"
        )
        assert req.model == "baw"

    def test_valid_bsm(self):
        req = OptionPriceRequest(
            ticker="AAPL", strike=200.0, expiration="2026-04-17", model="bsm"
        )
        assert req.model == "bsm"

    def test_invalid_model(self):
        with pytest.raises(ValidationError):
            OptionPriceRequest(
                ticker="AAPL", strike=200.0, expiration="2026-04-17", model="binomial"
            )


class TestStrategyLeg:
    def test_valid_leg(self):
        leg = StrategyLeg(strike=100.0, option_type="call", action="buy")
        assert leg.quantity == 1
        assert leg.premium == 0.0

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            StrategyLeg(strike=100.0, option_type="call", action="hold")


class TestStrategyAnalysisRequest:
    def test_valid_minimal(self):
        req = StrategyAnalysisRequest(ticker="AAPL")
        assert req.strategy_type == "covered_call"
        assert req.legs is None

    def test_valid_with_type(self):
        req = StrategyAnalysisRequest(ticker="AAPL", strategy_type="iron_condor")
        assert req.strategy_type == "iron_condor"

    def test_invalid_strategy_type(self):
        with pytest.raises(ValidationError):
            StrategyAnalysisRequest(ticker="AAPL", strategy_type="butterfly")


class TestUnusualActivityRequest:
    def test_valid(self):
        req = UnusualActivityRequest(ticker="TSLA", volume_oi_threshold=3.0)
        assert req.ticker == "TSLA"
        assert req.volume_oi_threshold == 3.0

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValidationError):
            UnusualActivityRequest(ticker="TSLA", volume_oi_threshold=-1.0)


class TestHedgePortfolioRequest:
    def test_valid_no_ticker(self):
        req = HedgePortfolioRequest()
        assert req.ticker is None
        assert req.risk_level == 50.0

    def test_valid_with_ticker(self):
        req = HedgePortfolioRequest(ticker="aapl", risk_level=30.0)
        assert req.ticker == "AAPL"

    def test_risk_level_bounds(self):
        with pytest.raises(ValidationError):
            HedgePortfolioRequest(risk_level=150.0)


class TestIVAnalysisRequest:
    def test_valid(self):
        req = IVAnalysisRequest(ticker="SPY")
        assert req.ticker == "SPY"
        assert req.expiration is None
