"""
Fundamental analysis module for computing composite scores from yfinance data.

Provides growth, value, and quality scoring with letter grades
to complement the platform's technical screening capabilities.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _safe_get(info: dict, key: str, default: float | None = None) -> float | None:
    """Safely extract a numeric value from yfinance info dict."""
    val = info.get(key)
    if val is None:
        return default
    try:
        result = float(val)
        if result != result:  # NaN check
            return default
        return result
    except (ValueError, TypeError):
        return default


def _score_metric(
    value: float | None,
    thresholds: list[tuple[float, float]],
    higher_is_better: bool = True,
) -> float:
    """Score a single metric on a 0-100 scale using threshold brackets.

    Args:
        value: The metric value to score.
        thresholds: List of (threshold, score) tuples, ordered from
                    best to worst if higher_is_better, worst to best otherwise.
        higher_is_better: Whether higher values are better.

    Returns:
        Score between 0 and 100.
    """
    if value is None:
        return 50.0  # Neutral score for missing data

    for threshold, score in thresholds:
        if higher_is_better and value >= threshold:
            return score
        elif not higher_is_better and value <= threshold:
            return score

    # Return lowest score if no threshold matched
    return thresholds[-1][1] if thresholds else 50.0


def compute_growth_score(info: dict) -> dict[str, Any]:
    """Compute growth score (0-100) from earnings and revenue growth.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with individual metrics and composite growth_score.
    """
    earnings_growth = _safe_get(info, "earningsGrowth")
    revenue_growth = _safe_get(info, "revenueGrowth")

    # Score earnings growth (higher is better)
    eg_score = _score_metric(
        earnings_growth,
        [
            (0.25, 95),  # 25%+ growth = excellent
            (0.15, 80),  # 15%+ = strong
            (0.10, 70),  # 10%+ = good
            (0.05, 60),  # 5%+ = moderate
            (0.00, 45),  # Flat
            (-0.10, 30),  # -10% = declining
            (-0.25, 15),  # -25%+ = severe decline
        ],
        higher_is_better=True,
    )

    # Score revenue growth (higher is better)
    rg_score = _score_metric(
        revenue_growth,
        [
            (0.20, 95),
            (0.10, 80),
            (0.05, 65),
            (0.00, 45),
            (-0.05, 30),
            (-0.15, 15),
        ],
        higher_is_better=True,
    )

    # Weighted composite: earnings growth matters more
    growth_score = round(0.6 * eg_score + 0.4 * rg_score, 1)

    return {
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "earnings_growth_score": round(eg_score, 1),
        "revenue_growth_score": round(rg_score, 1),
        "growth_score": growth_score,
    }


def compute_value_score(info: dict) -> dict[str, Any]:
    """Compute value score (0-100) from valuation metrics.

    Lower P/E, PEG, and P/B generally indicate better value.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with individual metrics and composite value_score.
    """
    trailing_pe = _safe_get(info, "trailingPE")
    forward_pe = _safe_get(info, "forwardPE")
    peg_ratio = _safe_get(info, "pegRatio")
    price_to_book = _safe_get(info, "priceToBook")

    # Score trailing P/E (lower is better for value)
    pe_score = _score_metric(
        trailing_pe,
        [
            (10, 95),  # Very cheap
            (15, 80),  # Attractive
            (20, 65),  # Fair
            (25, 50),  # Moderate
            (35, 35),  # Expensive
            (50, 20),  # Very expensive
        ],
        higher_is_better=False,
    )

    # Score forward P/E (lower is better)
    fpe_score = _score_metric(
        forward_pe,
        [
            (10, 95),
            (15, 80),
            (20, 65),
            (25, 50),
            (35, 35),
            (50, 20),
        ],
        higher_is_better=False,
    )

    # Score PEG ratio (lower is better, <1 is attractive)
    peg_score = _score_metric(
        peg_ratio,
        [
            (0.5, 95),  # Very attractive
            (1.0, 80),  # Attractive
            (1.5, 65),  # Fair
            (2.0, 50),  # Moderate
            (3.0, 30),  # Expensive
        ],
        higher_is_better=False,
    )

    # Score price-to-book (lower is better, <1 is deep value)
    ptb_score = _score_metric(
        price_to_book,
        [
            (1.0, 95),  # Deep value
            (2.0, 80),
            (3.0, 65),
            (5.0, 50),
            (10.0, 30),
            (20.0, 15),
        ],
        higher_is_better=False,
    )

    # Weighted composite
    value_score = round(
        0.30 * pe_score + 0.25 * fpe_score + 0.25 * peg_score + 0.20 * ptb_score, 1
    )

    return {
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "peg_ratio": peg_ratio,
        "price_to_book": price_to_book,
        "pe_score": round(pe_score, 1),
        "forward_pe_score": round(fpe_score, 1),
        "peg_score": round(peg_score, 1),
        "price_to_book_score": round(ptb_score, 1),
        "value_score": value_score,
    }


def compute_quality_score(info: dict) -> dict[str, Any]:
    """Compute quality score (0-100) from profitability and balance sheet metrics.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with individual metrics and composite quality_score.
    """
    roe = _safe_get(info, "returnOnEquity")
    roa = _safe_get(info, "returnOnAssets")
    profit_margin = _safe_get(info, "profitMargins")
    operating_margin = _safe_get(info, "operatingMargins")
    debt_to_equity = _safe_get(info, "debtToEquity")
    current_ratio = _safe_get(info, "currentRatio")

    # Score ROE (higher is better)
    roe_score = _score_metric(
        roe,
        [
            (0.25, 95),  # 25%+ = excellent
            (0.15, 80),  # 15%+ = strong
            (0.10, 65),
            (0.05, 50),
            (0.00, 30),
            (-0.05, 15),
        ],
        higher_is_better=True,
    )

    # Score profit margin (higher is better)
    pm_score = _score_metric(
        profit_margin,
        [
            (0.20, 95),  # 20%+ = excellent
            (0.10, 80),
            (0.05, 65),
            (0.02, 50),
            (0.00, 30),
            (-0.05, 15),
        ],
        higher_is_better=True,
    )

    # Score debt-to-equity (lower is better)
    dte_score = _score_metric(
        debt_to_equity,
        [
            (0.3, 95),  # Very low debt
            (0.5, 85),
            (1.0, 70),
            (1.5, 55),
            (2.0, 40),
            (3.0, 25),
        ],
        higher_is_better=False,
    )

    # Score current ratio (higher is better, but not too high)
    cr_score = 50.0  # default
    if current_ratio is not None:
        if current_ratio >= 3.0:
            cr_score = 70.0  # High but may indicate inefficiency
        elif current_ratio >= 2.0:
            cr_score = 90.0  # Healthy
        elif current_ratio >= 1.5:
            cr_score = 80.0  # Good
        elif current_ratio >= 1.0:
            cr_score = 60.0  # Adequate
        elif current_ratio >= 0.5:
            cr_score = 35.0  # Tight
        else:
            cr_score = 15.0  # Liquidity risk

    # Weighted composite
    quality_score = round(
        0.30 * roe_score + 0.25 * pm_score + 0.25 * dte_score + 0.20 * cr_score, 1
    )

    return {
        "return_on_equity": roe,
        "return_on_assets": roa,
        "profit_margin": profit_margin,
        "operating_margin": operating_margin,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "roe_score": round(roe_score, 1),
        "profit_margin_score": round(pm_score, 1),
        "debt_to_equity_score": round(dte_score, 1),
        "current_ratio_score": round(cr_score, 1),
        "quality_score": quality_score,
    }


def _compute_grade(score: float) -> str:
    """Convert a 0-100 score to a letter grade."""
    if score >= 80:
        return "A"
    elif score >= 65:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 35:
        return "D"
    else:
        return "F"


def compute_fundamental_score(info: dict) -> dict[str, Any]:
    """Compute composite fundamental score from yfinance info dict.

    Combines growth, value, and quality sub-scores into a single
    fundamental score with a letter grade.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with sub-scores, composite fundamental_score, and grade.
    """
    growth = compute_growth_score(info)
    value = compute_value_score(info)
    quality = compute_quality_score(info)

    # Composite: quality and value weighted slightly higher than growth
    fundamental_score = round(
        0.35 * quality["quality_score"]
        + 0.35 * value["value_score"]
        + 0.30 * growth["growth_score"],
        1,
    )

    grade = _compute_grade(fundamental_score)

    return {
        "fundamental_score": fundamental_score,
        "grade": grade,
        "growth": growth,
        "value": value,
        "quality": quality,
    }


def get_financial_health(info: dict) -> dict[str, Any]:
    """Extract balance sheet health indicators from yfinance info.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with balance sheet metrics and health assessment.
    """
    debt_to_equity = _safe_get(info, "debtToEquity")
    current_ratio = _safe_get(info, "currentRatio")
    quick_ratio = _safe_get(info, "quickRatio")
    free_cashflow = _safe_get(info, "freeCashflow")
    operating_cashflow = _safe_get(info, "operatingCashflow")
    total_debt = _safe_get(info, "totalDebt")
    total_cash = _safe_get(info, "totalCash")

    # Assess overall health
    health_flags = []
    if debt_to_equity is not None:
        if debt_to_equity > 2.0:
            health_flags.append("high_leverage")
        elif debt_to_equity < 0.5:
            health_flags.append("low_debt")

    if current_ratio is not None:
        if current_ratio < 1.0:
            health_flags.append("liquidity_risk")
        elif current_ratio > 2.0:
            health_flags.append("strong_liquidity")

    if free_cashflow is not None:
        if free_cashflow > 0:
            health_flags.append("positive_fcf")
        else:
            health_flags.append("negative_fcf")

    # Simple health assessment
    positive = sum(
        1 for f in health_flags if f in ("low_debt", "strong_liquidity", "positive_fcf")
    )
    negative = sum(
        1
        for f in health_flags
        if f in ("high_leverage", "liquidity_risk", "negative_fcf")
    )

    if positive >= 2 and negative == 0:
        assessment = "strong"
    elif negative >= 2:
        assessment = "weak"
    elif negative >= 1:
        assessment = "moderate"
    else:
        assessment = "adequate"

    return {
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "free_cashflow": free_cashflow,
        "operating_cashflow": operating_cashflow,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "health_flags": health_flags,
        "assessment": assessment,
    }


def get_earnings_analysis(info: dict) -> dict[str, Any]:
    """Extract earnings metrics and trend assessment.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with earnings metrics and trend analysis.
    """
    trailing_eps = _safe_get(info, "trailingEps")
    forward_eps = _safe_get(info, "forwardEps")
    earnings_growth = _safe_get(info, "earningsGrowth")
    revenue_growth = _safe_get(info, "revenueGrowth")

    # Determine earnings trend
    trend = "unknown"
    if trailing_eps is not None and forward_eps is not None:
        if forward_eps > trailing_eps * 1.05:
            trend = "accelerating"
        elif forward_eps > trailing_eps:
            trend = "growing"
        elif forward_eps > trailing_eps * 0.95:
            trend = "stable"
        else:
            trend = "declining"
    elif earnings_growth is not None:
        if earnings_growth > 0.10:
            trend = "accelerating"
        elif earnings_growth > 0:
            trend = "growing"
        elif earnings_growth > -0.05:
            trend = "stable"
        else:
            trend = "declining"

    # EPS estimate revision direction
    eps_revision = None
    if trailing_eps is not None and forward_eps is not None and trailing_eps != 0:
        eps_revision = round((forward_eps - trailing_eps) / abs(trailing_eps) * 100, 1)

    return {
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "eps_revision_pct": eps_revision,
        "trend": trend,
    }


def get_valuation_assessment(info: dict) -> dict[str, Any]:
    """Assess stock valuation relative to market norms.

    Args:
        info: Raw yfinance info dictionary.

    Returns:
        Dict with valuation metrics and assessment.
    """
    trailing_pe = _safe_get(info, "trailingPE")
    forward_pe = _safe_get(info, "forwardPE")
    peg_ratio = _safe_get(info, "pegRatio")
    price_to_book = _safe_get(info, "priceToBook")
    price_to_sales = _safe_get(info, "priceToSalesTrailing12Months")
    ev_to_ebitda = _safe_get(info, "enterpriseToEbitda")

    # S&P 500 historical median benchmarks
    benchmarks = {
        "trailing_pe": 20.0,
        "forward_pe": 18.0,
        "peg_ratio": 1.5,
        "price_to_book": 3.5,
        "ev_to_ebitda": 14.0,
    }

    assessments = {}

    if trailing_pe is not None:
        ratio = trailing_pe / benchmarks["trailing_pe"]
        if ratio < 0.7:
            assessments["pe_assessment"] = "undervalued"
        elif ratio < 1.0:
            assessments["pe_assessment"] = "fairly_valued"
        elif ratio < 1.5:
            assessments["pe_assessment"] = "moderately_overvalued"
        else:
            assessments["pe_assessment"] = "overvalued"

    if peg_ratio is not None:
        if peg_ratio < 1.0:
            assessments["peg_assessment"] = "attractive_growth_value"
        elif peg_ratio < 1.5:
            assessments["peg_assessment"] = "fairly_priced_growth"
        elif peg_ratio < 2.5:
            assessments["peg_assessment"] = "expensive_growth"
        else:
            assessments["peg_assessment"] = "overpriced_growth"

    # Overall valuation
    if trailing_pe is not None and peg_ratio is not None:
        if trailing_pe < 15 and peg_ratio < 1.0:
            overall = "undervalued"
        elif trailing_pe < 25 and peg_ratio < 1.5:
            overall = "fairly_valued"
        elif trailing_pe < 40:
            overall = "moderately_overvalued"
        else:
            overall = "overvalued"
    elif trailing_pe is not None:
        if trailing_pe < 15:
            overall = "undervalued"
        elif trailing_pe < 25:
            overall = "fairly_valued"
        else:
            overall = "overvalued"
    else:
        overall = "insufficient_data"

    return {
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "peg_ratio": peg_ratio,
        "price_to_book": price_to_book,
        "price_to_sales": price_to_sales,
        "ev_to_ebitda": ev_to_ebitda,
        "benchmarks": benchmarks,
        "overall_valuation": overall,
        **assessments,
    }
