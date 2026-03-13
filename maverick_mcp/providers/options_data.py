"""Options chain data provider with in-memory TTL caching.

Wraps yfinance options chain API and applies liquidity filtering
(bid-ask spread, volume, open interest) to return high-quality
options data.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import yfinance as yf

logger = logging.getLogger("maverick_mcp.providers.options_data")


class OptionsDataProvider:
    """Provider for options chain data with short-lived in-memory caching."""

    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl_seconds

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_available_expirations(self, ticker: str) -> list[str]:
        """Return available option expiration dates for *ticker*."""
        cache_key = f"expirations:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            t = yf.Ticker(ticker)
            expirations = list(t.options)  # tuple of date strings
            self._set_cached(cache_key, expirations)
            return expirations
        except Exception as e:
            logger.error("Failed to fetch expirations for %s: %s", ticker, e)
            return []

    def get_option_chain(
        self,
        ticker: str,
        expiration: str,
        min_volume: int = 10,
        min_open_interest: int = 100,
        max_bid_ask_spread_pct: float = 10.0,
    ) -> dict[str, Any]:
        """Fetch and filter options chain for *ticker* at *expiration*.

        Applies liquidity filters to remove illiquid contracts.
        """
        cache_key = f"chain:{ticker}:{expiration}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            # Re-apply filters (caller may use different thresholds)
            return self._apply_filters(
                cached, min_volume, min_open_interest, max_bid_ask_spread_pct
            )

        try:
            t = yf.Ticker(ticker)
            chain = t.option_chain(expiration)

            # Get underlying price
            spot = self._get_spot_price(t)

            raw_data = {
                "calls": self._process_contracts(chain.calls, spot),
                "puts": self._process_contracts(chain.puts, spot),
                "underlying_price": spot,
                "expiration": expiration,
                "ticker": ticker,
            }

            self._set_cached(cache_key, raw_data)

            return self._apply_filters(
                raw_data, min_volume, min_open_interest, max_bid_ask_spread_pct
            )
        except Exception as e:
            logger.error(
                "Failed to fetch chain for %s exp %s: %s", ticker, expiration, e
            )
            return {
                "calls": [],
                "puts": [],
                "underlying_price": 0.0,
                "expiration": expiration,
                "ticker": ticker,
                "error": str(e),
            }

    def get_underlying_price(self, ticker: str) -> float:
        """Return current underlying price for *ticker*."""
        cache_key = f"price:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            t = yf.Ticker(ticker)
            price = self._get_spot_price(t)
            self._set_cached(cache_key, price)
            return price
        except Exception as e:
            logger.error("Failed to fetch price for %s: %s", ticker, e)
            return 0.0

    def get_dividend_yield(self, ticker: str) -> float:
        """Return annualised dividend yield for *ticker* (decimal)."""
        cache_key = f"divyield:{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            t = yf.Ticker(ticker)
            info = t.info
            div_yield = float(info.get("dividendYield", 0) or 0)
            self._set_cached(cache_key, div_yield)
            return div_yield
        except Exception as e:
            logger.debug("Failed to fetch dividend yield for %s: %s", ticker, e)
            return 0.0

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_spot_price(ticker_obj: yf.Ticker) -> float:
        """Extract current price from a yfinance Ticker object."""
        try:
            info = ticker_obj.fast_info
            price = float(getattr(info, "last_price", 0) or 0)
            if price > 0:
                return price
        except Exception:
            pass

        # Fallback: use last close from history
        try:
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _process_contracts(df: Any, spot: float) -> list[dict[str, Any]]:
        """Convert a yfinance options DataFrame into a list of dicts with computed fields."""
        if df is None or df.empty:
            return []

        contracts: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            mid = (
                (bid + ask) / 2
                if (bid + ask) > 0
                else float(row.get("lastPrice", 0) or 0)
            )
            spread = ask - bid if ask > bid else 0.0
            spread_pct = (spread / mid * 100) if mid > 0 else 999.0

            strike = float(row.get("strike", 0))

            contracts.append(
                {
                    "contractSymbol": str(row.get("contractSymbol", "")),
                    "strike": strike,
                    "lastPrice": float(row.get("lastPrice", 0) or 0),
                    "bid": bid,
                    "ask": ask,
                    "mid": round(mid, 2),
                    "bidAskSpread": round(spread, 2),
                    "bidAskSpreadPct": round(spread_pct, 2),
                    "volume": int(row.get("volume", 0) or 0),
                    "openInterest": int(row.get("openInterest", 0) or 0),
                    "impliedVolatility": float(row.get("impliedVolatility", 0) or 0),
                    "inTheMoney": bool(row.get("inTheMoney", False)),
                }
            )

        return contracts

    @staticmethod
    def _apply_filters(
        raw_data: dict[str, Any],
        min_volume: int,
        min_open_interest: int,
        max_bid_ask_spread_pct: float,
    ) -> dict[str, Any]:
        """Filter contracts by liquidity thresholds."""

        def _passes(c: dict[str, Any]) -> bool:
            if c["volume"] < min_volume:
                return False
            if c["openInterest"] < min_open_interest:
                return False
            if c["bidAskSpreadPct"] > max_bid_ask_spread_pct:
                return False
            return True

        return {
            **raw_data,
            "calls": [c for c in raw_data["calls"] if _passes(c)],
            "puts": [c for c in raw_data["puts"] if _passes(c)],
        }

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts, _ = self._cache[key]
        return (time.monotonic() - ts) < self._cache_ttl

    def _get_cached(self, key: str) -> Any | None:
        if self._is_cache_valid(key):
            return self._cache[key][1]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)
