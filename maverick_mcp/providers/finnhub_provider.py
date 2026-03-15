"""
Finnhub data provider for Maverick-MCP.

Provides real-time quotes, company profiles, and earnings calendar data
from the Finnhub API as an alternative/backup data source.

Requires FINNHUB_API_KEY environment variable.
Free tier: 60 API calls/minute.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger("maverick_mcp.finnhub")

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubProvider:
    """
    Finnhub API data provider.

    Provides:
    - Real-time stock quotes
    - Company profiles
    - Earnings calendar
    - Stock candles (OHLCV)
    - Basic financials
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"X-Finnhub-Token": self.api_key})

    @property
    def is_configured(self) -> bool:
        """Check if the provider has a valid API key."""
        return bool(self.api_key)

    def _get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make a GET request to the Finnhub API."""
        if not self.is_configured:
            raise ValueError(
                "Finnhub API key not configured. Set FINNHUB_API_KEY environment variable."
            )

        url = f"{FINNHUB_BASE_URL}/{endpoint}"
        params = params or {}
        params["token"] = self.api_key

        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get real-time quote for a symbol.

        Returns:
            Dictionary with keys: c (current), d (change), dp (percent change),
            h (high), l (low), o (open), pc (previous close), t (timestamp)
        """
        data = self._get("quote", {"symbol": symbol.upper()})

        if not data or data.get("c", 0) == 0:
            raise ValueError(f"No quote data returned for {symbol}")

        return {
            "symbol": symbol.upper(),
            "current_price": data.get("c", 0),
            "change": data.get("d", 0),
            "change_percent": data.get("dp", 0),
            "high": data.get("h", 0),
            "low": data.get("l", 0),
            "open": data.get("o", 0),
            "previous_close": data.get("pc", 0),
            "timestamp": datetime.fromtimestamp(data.get("t", 0)).isoformat()
            if data.get("t")
            else None,
            "source": "finnhub",
        }

    def get_company_profile(self, symbol: str) -> dict[str, Any]:
        """Get company profile information."""
        data = self._get("stock/profile2", {"symbol": symbol.upper()})

        if not data or not data.get("name"):
            raise ValueError(f"No profile data returned for {symbol}")

        return {
            "symbol": symbol.upper(),
            "name": data.get("name", ""),
            "country": data.get("country", ""),
            "currency": data.get("currency", ""),
            "exchange": data.get("exchange", ""),
            "industry": data.get("finnhubIndustry", ""),
            "ipo_date": data.get("ipo", ""),
            "logo": data.get("logo", ""),
            "market_cap": data.get("marketCapitalization", 0),
            "outstanding_shares": data.get("shareOutstanding", 0),
            "phone": data.get("phone", ""),
            "web_url": data.get("weburl", ""),
            "source": "finnhub",
        }

    def get_stock_candles(
        self,
        symbol: str,
        resolution: str = "D",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV candle data.

        Args:
            symbol: Stock ticker symbol
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=365)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        from_ts = int(start_dt.timestamp())
        to_ts = int(end_dt.timestamp())

        data = self._get(
            "stock/candle",
            {
                "symbol": symbol.upper(),
                "resolution": resolution,
                "from": from_ts,
                "to": to_ts,
            },
        )

        if data.get("s") != "ok" or not data.get("t"):
            logger.warning(f"No candle data returned for {symbol}: {data.get('s')}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(
            {
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"],
            },
            index=pd.DatetimeIndex(
                [datetime.fromtimestamp(t) for t in data["t"]], name="Date"
            ),
        )

        return df

    def get_earnings_calendar(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get upcoming earnings calendar."""
        if from_date is None:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        data = self._get(
            "calendar/earnings", {"from": from_date, "to": to_date}
        )

        earnings = data.get("earningsCalendar", [])
        return [
            {
                "symbol": e.get("symbol", ""),
                "date": e.get("date", ""),
                "hour": e.get("hour", ""),
                "eps_estimate": e.get("epsEstimate"),
                "eps_actual": e.get("epsActual"),
                "revenue_estimate": e.get("revenueEstimate"),
                "revenue_actual": e.get("revenueActual"),
                "source": "finnhub",
            }
            for e in earnings[:50]  # Limit to 50 results
        ]

    def get_basic_financials(self, symbol: str) -> dict[str, Any]:
        """Get basic financial metrics."""
        data = self._get(
            "stock/metric", {"symbol": symbol.upper(), "metric": "all"}
        )

        metrics = data.get("metric", {})
        if not metrics:
            raise ValueError(f"No financial metrics returned for {symbol}")

        return {
            "symbol": symbol.upper(),
            "pe_ratio": metrics.get("peNormalizedAnnual"),
            "pb_ratio": metrics.get("pbAnnual"),
            "ps_ratio": metrics.get("psAnnual"),
            "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
            "beta": metrics.get("beta"),
            "market_cap": metrics.get("marketCapitalization"),
            "52_week_high": metrics.get("52WeekHigh"),
            "52_week_low": metrics.get("52WeekLow"),
            "10_day_avg_volume": metrics.get("10DayAverageTradingVolume"),
            "roe": metrics.get("roeTTM"),
            "roa": metrics.get("roaTTM"),
            "revenue_growth": metrics.get("revenueGrowthQuarterlyYoy"),
            "eps_growth": metrics.get("epsGrowthQuarterlyYoy"),
            "source": "finnhub",
        }

    def health_check(self) -> dict[str, Any]:
        """Check Finnhub API connectivity."""
        if not self.is_configured:
            return {
                "status": "unconfigured",
                "message": "FINNHUB_API_KEY not set",
            }
        try:
            self.get_quote("AAPL")
            return {"status": "healthy", "message": "Finnhub API accessible"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}


# Module-level singleton
_finnhub_provider: FinnhubProvider | None = None


def get_finnhub_provider() -> FinnhubProvider:
    """Get or create the Finnhub provider singleton."""
    global _finnhub_provider
    if _finnhub_provider is None:
        _finnhub_provider = FinnhubProvider()
    return _finnhub_provider
