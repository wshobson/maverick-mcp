"""Tests for maverick.market_data.types."""

import pytest

from maverick.market_data.types import (
    IndexQuote,
    MarketOverview,
    Mover,
    Quote,
    Volatility,
    fear_level_from_vix,
)


def test_quote_roundtrips_through_model_dump():
    q = Quote(
        symbol="AAPL",
        price=190.5,
        change=1.5,
        change_percent=0.79,
        volume=55_000_000,
        timestamp="2026-07-19T14:30:00",
    )
    data = q.model_dump()
    assert data["symbol"] == "AAPL"
    assert Quote(**data) == q


@pytest.mark.parametrize(
    ("vix", "expected"),
    [
        (None, "unknown"),
        (12.0, "low"),
        (19.99, "low"),
        (20.0, "elevated"),
        (29.99, "elevated"),
        (30.0, "high"),
        (55.0, "high"),
    ],
)
def test_fear_level_bands(vix, expected):
    assert fear_level_from_vix(vix) == expected


def test_market_overview_composes():
    overview = MarketOverview(
        indices={
            "^GSPC": IndexQuote(
                name="S&P 500",
                symbol="^GSPC",
                price=6100.0,
                change=12.0,
                change_percent=0.2,
            )
        },
        sectors={"Technology": 0.8},
        top_gainers=[
            Mover(
                symbol="XYZ",
                price=10.0,
                change=2.0,
                change_percent=25.0,
                volume=1_000_000,
            )
        ],
        top_losers=[],
        volatility=Volatility(vix=18.5, vix_change_percent=-2.1, fear_level="low"),
        last_updated="2026-07-19T14:30:00",
    )
    assert overview.indices["^GSPC"].price == 6100.0
    assert overview.volatility.fear_level == "low"
