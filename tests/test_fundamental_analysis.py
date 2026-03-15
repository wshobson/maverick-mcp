"""Tests for the fundamental analysis module."""

import pytest

from maverick_mcp.core.fundamental_analysis import (
    _compute_grade,
    _safe_get,
    _score_metric,
    compute_fundamental_score,
    compute_growth_score,
    compute_quality_score,
    compute_value_score,
    get_earnings_analysis,
    get_financial_health,
    get_valuation_assessment,
)

# --- Fixtures ---


@pytest.fixture
def complete_yfinance_info():
    """A realistic yfinance info dict with all fields populated."""
    return {
        "longName": "Test Corp",
        "shortName": "TEST",
        "sector": "Technology",
        "industry": "Software",
        # Valuation
        "trailingPE": 18.5,
        "forwardPE": 16.0,
        "pegRatio": 1.2,
        "priceToBook": 4.5,
        "priceToSalesTrailing12Months": 5.0,
        "enterpriseToEbitda": 12.0,
        # Profitability
        "returnOnEquity": 0.22,
        "returnOnAssets": 0.12,
        "profitMargins": 0.18,
        "operatingMargins": 0.25,
        # Growth
        "earningsGrowth": 0.15,
        "revenueGrowth": 0.12,
        # Balance sheet
        "debtToEquity": 0.8,
        "currentRatio": 2.1,
        "quickRatio": 1.8,
        "totalDebt": 5_000_000_000,
        "totalCash": 10_000_000_000,
        "freeCashflow": 3_000_000_000,
        "operatingCashflow": 5_000_000_000,
        # Earnings
        "trailingEps": 5.50,
        "forwardEps": 6.20,
        # Dividends
        "dividendYield": 0.015,
        "payoutRatio": 0.30,
        # Market
        "currentPrice": 150.00,
        "marketCap": 200_000_000_000,
    }


@pytest.fixture
def minimal_yfinance_info():
    """A yfinance info dict with only basic fields."""
    return {
        "longName": "Minimal Corp",
        "sector": "Healthcare",
    }


@pytest.fixture
def struggling_company_info():
    """A yfinance info dict for a struggling company."""
    return {
        "longName": "Struggling Inc",
        "trailingPE": 55.0,
        "forwardPE": 45.0,
        "pegRatio": 3.5,
        "priceToBook": 15.0,
        "returnOnEquity": -0.05,
        "profitMargins": -0.03,
        "earningsGrowth": -0.20,
        "revenueGrowth": -0.10,
        "debtToEquity": 3.5,
        "currentRatio": 0.6,
        "freeCashflow": -500_000_000,
        "trailingEps": -1.50,
        "forwardEps": -0.80,
    }


# --- Helper function tests ---


class TestSafeGet:
    def test_normal_value(self):
        assert _safe_get({"key": 42.5}, "key") == 42.5

    def test_missing_key(self):
        assert _safe_get({"key": 42}, "missing") is None

    def test_missing_key_with_default(self):
        assert _safe_get({"key": 42}, "missing", 0.0) == 0.0

    def test_none_value(self):
        assert _safe_get({"key": None}, "key") is None

    def test_string_convertible(self):
        assert _safe_get({"key": "3.14"}, "key") == 3.14

    def test_non_numeric_string(self):
        assert _safe_get({"key": "not_a_number"}, "key") is None

    def test_nan_value(self):
        assert _safe_get({"key": float("nan")}, "key") is None


class TestScoreMetric:
    def test_higher_is_better(self):
        thresholds = [(0.25, 95), (0.15, 80), (0.10, 65), (0.0, 50)]
        assert _score_metric(0.30, thresholds, higher_is_better=True) == 95
        assert _score_metric(0.20, thresholds, higher_is_better=True) == 80
        assert _score_metric(0.12, thresholds, higher_is_better=True) == 65

    def test_lower_is_better(self):
        thresholds = [(10, 95), (15, 80), (20, 65)]
        assert _score_metric(8, thresholds, higher_is_better=False) == 95
        assert _score_metric(12, thresholds, higher_is_better=False) == 80

    def test_none_returns_neutral(self):
        assert _score_metric(None, [(10, 95)]) == 50.0


class TestComputeGrade:
    def test_grade_a(self):
        assert _compute_grade(85) == "A"
        assert _compute_grade(80) == "A"

    def test_grade_b(self):
        assert _compute_grade(70) == "B"
        assert _compute_grade(65) == "B"

    def test_grade_c(self):
        assert _compute_grade(55) == "C"
        assert _compute_grade(50) == "C"

    def test_grade_d(self):
        assert _compute_grade(40) == "D"
        assert _compute_grade(35) == "D"

    def test_grade_f(self):
        assert _compute_grade(34) == "F"
        assert _compute_grade(10) == "F"


# --- Growth score tests ---


class TestComputeGrowthScore:
    def test_strong_growth(self, complete_yfinance_info):
        result = compute_growth_score(complete_yfinance_info)
        assert result["growth_score"] > 60
        assert result["earnings_growth"] == 0.15
        assert result["revenue_growth"] == 0.12

    def test_negative_growth(self, struggling_company_info):
        result = compute_growth_score(struggling_company_info)
        assert result["growth_score"] < 40

    def test_missing_data(self, minimal_yfinance_info):
        result = compute_growth_score(minimal_yfinance_info)
        assert result["growth_score"] == 50.0  # Neutral for missing data
        assert result["earnings_growth"] is None


# --- Value score tests ---


class TestComputeValueScore:
    def test_fairly_valued(self, complete_yfinance_info):
        result = compute_value_score(complete_yfinance_info)
        assert 40 < result["value_score"] < 80
        assert result["trailing_pe"] == 18.5

    def test_overvalued(self, struggling_company_info):
        result = compute_value_score(struggling_company_info)
        assert result["value_score"] < 40

    def test_missing_data(self, minimal_yfinance_info):
        result = compute_value_score(minimal_yfinance_info)
        assert result["value_score"] == 50.0


# --- Quality score tests ---


class TestComputeQualityScore:
    def test_high_quality(self, complete_yfinance_info):
        result = compute_quality_score(complete_yfinance_info)
        assert result["quality_score"] > 60
        assert result["return_on_equity"] == 0.22

    def test_low_quality(self, struggling_company_info):
        result = compute_quality_score(struggling_company_info)
        assert result["quality_score"] < 40

    def test_missing_data(self, minimal_yfinance_info):
        result = compute_quality_score(minimal_yfinance_info)
        assert result["quality_score"] == 50.0


# --- Composite score tests ---


class TestComputeFundamentalScore:
    def test_complete_data(self, complete_yfinance_info):
        result = compute_fundamental_score(complete_yfinance_info)
        assert "fundamental_score" in result
        assert "grade" in result
        assert "growth" in result
        assert "value" in result
        assert "quality" in result
        assert 0 <= result["fundamental_score"] <= 100
        assert result["grade"] in ("A", "B", "C", "D", "F")

    def test_good_company_gets_high_score(self, complete_yfinance_info):
        result = compute_fundamental_score(complete_yfinance_info)
        assert result["fundamental_score"] >= 60
        assert result["grade"] in ("A", "B")

    def test_struggling_company_gets_low_score(self, struggling_company_info):
        result = compute_fundamental_score(struggling_company_info)
        assert result["fundamental_score"] < 40
        assert result["grade"] in ("D", "F")

    def test_minimal_data_gets_neutral_score(self, minimal_yfinance_info):
        result = compute_fundamental_score(minimal_yfinance_info)
        assert result["fundamental_score"] == 50.0
        assert result["grade"] == "C"

    def test_empty_dict(self):
        result = compute_fundamental_score({})
        assert result["fundamental_score"] == 50.0
        assert result["grade"] == "C"


# --- Financial health tests ---


class TestGetFinancialHealth:
    def test_healthy_company(self, complete_yfinance_info):
        result = get_financial_health(complete_yfinance_info)
        assert result["assessment"] in ("strong", "adequate")
        assert "positive_fcf" in result["health_flags"]
        assert result["current_ratio"] == 2.1

    def test_struggling_company(self, struggling_company_info):
        result = get_financial_health(struggling_company_info)
        assert result["assessment"] == "weak"
        assert "high_leverage" in result["health_flags"]
        assert "liquidity_risk" in result["health_flags"]

    def test_minimal_data(self, minimal_yfinance_info):
        result = get_financial_health(minimal_yfinance_info)
        assert result["assessment"] == "adequate"
        assert result["health_flags"] == []


# --- Earnings analysis tests ---


class TestGetEarningsAnalysis:
    def test_accelerating_earnings(self, complete_yfinance_info):
        result = get_earnings_analysis(complete_yfinance_info)
        assert result["trend"] == "accelerating"
        assert result["eps_revision_pct"] is not None
        assert result["eps_revision_pct"] > 0

    def test_declining_earnings(self, struggling_company_info):
        result = get_earnings_analysis(struggling_company_info)
        # Forward EPS (-0.80) > trailing EPS (-1.50) * 1.05 → "accelerating"
        # because the loss is narrowing (moving toward profitability)
        assert result["trend"] in ("accelerating", "growing")

    def test_missing_data(self, minimal_yfinance_info):
        result = get_earnings_analysis(minimal_yfinance_info)
        assert result["trend"] == "unknown"
        assert result["eps_revision_pct"] is None


# --- Valuation assessment tests ---


class TestGetValuationAssessment:
    def test_fair_valuation(self, complete_yfinance_info):
        result = get_valuation_assessment(complete_yfinance_info)
        assert result["overall_valuation"] in ("fairly_valued", "undervalued")
        assert "pe_assessment" in result
        assert result["trailing_pe"] == 18.5

    def test_overvalued(self, struggling_company_info):
        result = get_valuation_assessment(struggling_company_info)
        assert result["overall_valuation"] == "overvalued"

    def test_missing_data(self, minimal_yfinance_info):
        result = get_valuation_assessment(minimal_yfinance_info)
        assert result["overall_valuation"] == "insufficient_data"

    def test_benchmarks_present(self, complete_yfinance_info):
        result = get_valuation_assessment(complete_yfinance_info)
        assert "benchmarks" in result
        assert result["benchmarks"]["trailing_pe"] == 20.0
