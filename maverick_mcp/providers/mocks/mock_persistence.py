"""
Mock data persistence implementation for testing.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session


class MockSession:
    """Mock SQLAlchemy session for testing."""

    def __init__(self):
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def close(self):
        self.closed = True

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class MockDataPersistence:
    """
    Mock implementation of IDataPersistence for testing.
    """

    def __init__(self):
        """Initialize the mock persistence layer."""
        self._price_data: dict[str, pd.DataFrame] = {}
        self._stock_records: dict[str, dict[str, Any]] = {}
        self._screening_results: dict[str, list[dict[str, Any]]] = {}
        self._call_log: list[dict[str, Any]] = []

    async def get_session(self) -> Session:
        """Get a mock database session."""
        self._log_call("get_session", {})
        return MockSession()

    async def get_read_only_session(self) -> Session:
        """Get a mock read-only database session."""
        self._log_call("get_read_only_session", {})
        return MockSession()

    async def save_price_data(
        self, session: Session, symbol: str, data: pd.DataFrame
    ) -> int:
        """Save mock price data."""
        self._log_call("save_price_data", {"symbol": symbol, "data_length": len(data)})

        symbol = symbol.upper()

        # Store the data
        if symbol in self._price_data:
            # Merge with existing data
            existing_data = self._price_data[symbol]
            combined = pd.concat([existing_data, data])
            # Remove duplicates and sort
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            self._price_data[symbol] = combined
        else:
            self._price_data[symbol] = data.copy()

        return len(data)

    async def get_price_data(
        self,
        session: Session,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Retrieve mock price data."""
        self._log_call(
            "get_price_data",
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        symbol = symbol.upper()

        if symbol not in self._price_data:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        data = self._price_data[symbol].copy()

        # Filter by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        return data

    async def get_or_create_stock(self, session: Session, symbol: str) -> Any:
        """Get or create a mock stock record."""
        self._log_call("get_or_create_stock", {"symbol": symbol})

        symbol = symbol.upper()

        if symbol not in self._stock_records:
            self._stock_records[symbol] = {
                "symbol": symbol,
                "company_name": f"{symbol} Inc.",
                "sector": "Technology",
                "industry": "Software",
                "exchange": "NASDAQ",
                "currency": "USD",
                "country": "US",
                "created_at": datetime.now(),
            }

        return self._stock_records[symbol]

    async def save_screening_results(
        self,
        session: Session,
        screening_type: str,
        results: list[dict[str, Any]],
    ) -> int:
        """Save mock screening results."""
        self._log_call(
            "save_screening_results",
            {
                "screening_type": screening_type,
                "results_count": len(results),
            },
        )

        self._screening_results[screening_type] = results.copy()
        return len(results)

    async def get_screening_results(
        self,
        session: Session,
        screening_type: str,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve mock screening results."""
        self._log_call(
            "get_screening_results",
            {
                "screening_type": screening_type,
                "limit": limit,
                "min_score": min_score,
            },
        )

        if screening_type not in self._screening_results:
            return self._generate_default_screening_results(screening_type)

        results = self._screening_results[screening_type].copy()

        # Apply filters
        if min_score is not None:
            if screening_type == "maverick":
                results = [
                    r for r in results if r.get("combined_score", 0) >= min_score
                ]
            elif screening_type == "bearish":
                results = [r for r in results if r.get("score", 0) >= min_score]
            elif screening_type == "trending":
                results = [
                    r for r in results if r.get("momentum_score", 0) >= min_score
                ]

        if limit is not None:
            results = results[:limit]

        return results

    async def get_latest_screening_data(self) -> dict[str, list[dict[str, Any]]]:
        """Get mock latest screening data."""
        self._log_call("get_latest_screening_data", {})

        return {
            "maverick_stocks": await self.get_screening_results(None, "maverick"),
            "maverick_bear_stocks": await self.get_screening_results(None, "bearish"),
            "supply_demand_breakouts": await self.get_screening_results(
                None, "trending"
            ),
        }

    async def check_data_freshness(self, symbol: str, max_age_hours: int = 24) -> bool:
        """Check mock data freshness."""
        self._log_call(
            "check_data_freshness", {"symbol": symbol, "max_age_hours": max_age_hours}
        )

        # For testing, assume data is fresh if it exists
        symbol = symbol.upper()
        return symbol in self._price_data

    async def bulk_save_price_data(
        self, session: Session, symbol: str, data: pd.DataFrame
    ) -> int:
        """Bulk save mock price data."""
        return await self.save_price_data(session, symbol, data)

    async def get_symbols_with_data(
        self, session: Session, limit: int | None = None
    ) -> list[str]:
        """Get mock list of symbols with data."""
        self._log_call("get_symbols_with_data", {"limit": limit})

        symbols = list(self._price_data.keys())

        if limit is not None:
            symbols = symbols[:limit]

        return symbols

    async def cleanup_old_data(self, session: Session, days_to_keep: int = 365) -> int:
        """Mock cleanup of old data."""
        self._log_call("cleanup_old_data", {"days_to_keep": days_to_keep})

        # For testing, return 0 (no cleanup performed)
        return 0

    def _generate_default_screening_results(
        self, screening_type: str
    ) -> list[dict[str, Any]]:
        """Generate default screening results for testing."""
        if screening_type == "maverick":
            return [
                {
                    "symbol": "TEST1",
                    "combined_score": 95,
                    "momentum_score": 92,
                    "pattern": "Cup with Handle",
                    "consolidation": "yes",
                    "squeeze": "firing",
                },
                {
                    "symbol": "TEST2",
                    "combined_score": 88,
                    "momentum_score": 85,
                    "pattern": "Flat Base",
                    "consolidation": "no",
                    "squeeze": "setup",
                },
            ]
        elif screening_type == "bearish":
            return [
                {
                    "symbol": "BEAR1",
                    "score": 92,
                    "momentum_score": 25,
                    "rsi_14": 28,
                    "atr_contraction": True,
                    "big_down_vol": True,
                },
            ]
        elif screening_type == "trending":
            return [
                {
                    "symbol": "TREND1",
                    "momentum_score": 95,
                    "close": 150.25,
                    "sma_50": 145.50,
                    "sma_150": 140.25,
                    "sma_200": 135.75,
                    "pattern": "Breakout",
                },
            ]
        else:
            return []

    # Testing utilities

    def _log_call(self, method: str, args: dict[str, Any]) -> None:
        """Log method calls for testing verification."""
        self._call_log.append(
            {
                "method": method,
                "args": args,
                "timestamp": datetime.now(),
            }
        )

    def get_call_log(self) -> list[dict[str, Any]]:
        """Get the log of method calls."""
        return self._call_log.copy()

    def clear_call_log(self) -> None:
        """Clear the method call log."""
        self._call_log.clear()

    def set_price_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Set price data for testing."""
        self._price_data[symbol.upper()] = data

    def get_stored_price_data(self, symbol: str) -> pd.DataFrame | None:
        """Get stored price data for testing verification."""
        return self._price_data.get(symbol.upper())

    def set_screening_results(
        self, screening_type: str, results: list[dict[str, Any]]
    ) -> None:
        """Set screening results for testing."""
        self._screening_results[screening_type] = results

    def clear_all_data(self) -> None:
        """Clear all stored data."""
        self._price_data.clear()
        self._stock_records.clear()
        self._screening_results.clear()
