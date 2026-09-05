"""Financial-relevance scoring shared by every search provider. Third-layer sibling: imports nothing from the domain.

Moved out of `exa.py` so `SearXNGProvider` and `ExaSearchProvider` score and
sort results identically. The domain lists, keyword list, and weights are
verbatim from the legacy Exa provider. `ExaSearchProvider` keeps thin method
wrappers around these functions so its callers and tests are unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlparse

# Financial-specific domain preferences for better results.
FINANCIAL_DOMAINS = [
    "sec.gov",
    "edgar.sec.gov",
    "investor.gov",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
    "marketwatch.com",
    "yahoo.com/finance",
    "finance.yahoo.com",
    "morningstar.com",
    "fool.com",
    "seekingalpha.com",
    "investopedia.com",
    "barrons.com",
    "cnbc.com",
    "nasdaq.com",
    "nyse.com",
    "finra.org",
    "federalreserve.gov",
    "treasury.gov",
    "bls.gov",
]

AUTHORITATIVE_DOMAINS = [
    "sec.gov",
    "edgar.sec.gov",
    "federalreserve.gov",
    "treasury.gov",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
]

FINANCIAL_KEYWORDS = [
    "earnings",
    "revenue",
    "profit",
    "financial",
    "quarterly",
    "annual",
    "sec filing",
    "10-k",
    "10-q",
    "balance sheet",
    "income statement",
    "cash flow",
    "dividend",
    "market cap",
    "valuation",
    "analyst",
    "forecast",
    "guidance",
    "ebitda",
    "eps",
    "pe ratio",
]

_TOP_TIER_DOMAINS = ["sec.gov", "edgar.sec.gov", "federalreserve.gov"]
_HIGH_QUALITY_DOMAINS = ["bloomberg.com", "reuters.com", "wsj.com", "ft.com"]
_TITLE_TERMS = ["financial", "earnings", "quarterly", "annual", "sec"]

# Scoring weights, verbatim from legacy.
_DOMAIN_SCORE_TOP_TIER = 0.4
_DOMAIN_SCORE_HIGH_QUALITY = 0.3
_DOMAIN_SCORE_OTHER = 0.2
_KEYWORD_SCORE_PER_MATCH = 0.05
_KEYWORD_SCORE_MAX = 0.3
_TITLE_SCORE = 0.1
_RECENCY_SCORE_30D = 0.1
_RECENCY_SCORE_90D = 0.05


def extract_domain(url: str) -> str:
    """Return the lowercased host of `url` without a `www.` prefix, or `""`."""
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_authoritative_source(url: str) -> bool:
    """Whether `url` is from an authoritative financial source."""
    return extract_domain(url) in AUTHORITATIVE_DOMAINS


def financial_relevance(
    *,
    url: str,
    text: str | None,
    title: str | None,
    published_date: str | None,
    financial_domains: list[str] | None = None,
) -> float:
    """Score a search result's financial relevance from 0.0 to 1.0.

    Domain tier, keyword density in `text` (capped), a title bonus, and a
    recency bonus for ISO `published_date` values within 30 or 90 days.
    """
    domains = financial_domains if financial_domains is not None else FINANCIAL_DOMAINS
    score = 0.0

    domain = extract_domain(url)
    if domain in domains:
        if domain in _TOP_TIER_DOMAINS:
            score += _DOMAIN_SCORE_TOP_TIER
        elif domain in _HIGH_QUALITY_DOMAINS:
            score += _DOMAIN_SCORE_HIGH_QUALITY
        else:
            score += _DOMAIN_SCORE_OTHER

    if text:
        text_lower = text.lower()
        keyword_matches = sum(
            1 for keyword in FINANCIAL_KEYWORDS if keyword in text_lower
        )
        score += min(keyword_matches * _KEYWORD_SCORE_PER_MATCH, _KEYWORD_SCORE_MAX)

    if title:
        title_lower = title.lower()
        if any(term in title_lower for term in _TITLE_TERMS):
            score += _TITLE_SCORE

    if published_date:
        try:
            date_str = str(published_date)
            if date_str.endswith("Z"):
                date_str = date_str.replace("Z", "+00:00")
            pub_date = datetime.fromisoformat(date_str)
            days_old = (datetime.now(UTC) - pub_date).days
            if days_old <= 30:
                score += _RECENCY_SCORE_30D
            elif days_old <= 90:
                score += _RECENCY_SCORE_90D
        except (ValueError, AttributeError, TypeError):
            pass

    return min(score, 1.0)
