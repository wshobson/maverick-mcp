"""Tests for `maverick.research.providers.scoring`, the relevance helpers shared by
every search provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from maverick.research.providers.scoring import (
    extract_domain,
    financial_relevance,
    is_authoritative_source,
)


def test_extract_domain_lowercases_and_strips_www():
    assert extract_domain("https://www.Reuters.com/markets/x") == "reuters.com"


def test_extract_domain_of_empty_input_is_empty():
    assert extract_domain("") == ""


def test_authoritative_source_matches_the_domain_list():
    assert is_authoritative_source("https://www.sec.gov/edgar") is True
    assert is_authoritative_source("https://example.com") is False


def test_financial_relevance_tiers_domains():
    def score(url: str) -> float:
        return financial_relevance(url=url, text=None, title=None, published_date=None)

    assert score("https://sec.gov/x") == pytest.approx(0.4)
    assert score("https://reuters.com/x") == pytest.approx(0.3)
    assert score("https://fool.com/x") == pytest.approx(0.2)
    assert score("https://example.com/x") == pytest.approx(0.0)


def test_financial_relevance_caps_keyword_bonus_and_adds_title_bonus():
    text = "earnings revenue profit dividend valuation analyst forecast guidance"
    score = financial_relevance(
        url="https://example.com",
        text=text,
        title="Quarterly earnings",
        published_date=None,
    )
    # Eight keyword hits cap at 0.3; the title term adds 0.1.
    assert score == pytest.approx(0.4)


def test_financial_relevance_recency_bonus():
    recent = (datetime.now(UTC) - timedelta(days=5)).isoformat()
    older = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    assert financial_relevance(
        url="https://example.com", text=None, title=None, published_date=recent
    ) == pytest.approx(0.1)
    assert financial_relevance(
        url="https://example.com", text=None, title=None, published_date=older
    ) == pytest.approx(0.05)


def test_financial_relevance_ignores_unparseable_dates():
    assert (
        financial_relevance(
            url="https://example.com",
            text=None,
            title=None,
            published_date="not a date",
        )
        == 0.0
    )


def test_financial_relevance_honors_a_custom_domain_list():
    score = financial_relevance(
        url="https://example.com/x",
        text=None,
        title=None,
        published_date=None,
        financial_domains=["example.com"],
    )
    assert score == pytest.approx(0.2)
