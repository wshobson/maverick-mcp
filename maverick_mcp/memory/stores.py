"""
Memory stores for agent conversations and user data.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """Base class for memory storage."""

    def __init__(self, ttl_hours: float = 24.0):
        self.ttl_hours = ttl_hours
        self.store: dict[str, dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl_hours: float | None = None) -> None:
        """Store a value with optional custom TTL."""
        ttl = ttl_hours or self.ttl_hours
        expiry = datetime.now() + timedelta(hours=ttl)

        self.store[key] = {
            "value": value,
            "expiry": expiry.isoformat(),
            "created": datetime.now().isoformat(),
        }

    def get(self, key: str) -> Any | None:
        """Get a value if not expired."""
        if key not in self.store:
            return None

        entry = self.store[key]
        expiry = datetime.fromisoformat(entry["expiry"])

        if datetime.now() > expiry:
            del self.store[key]
            return None

        return entry["value"]

    def delete(self, key: str) -> None:
        """Delete a value."""
        if key in self.store:
            del self.store[key]

    def clear_expired(self) -> int:
        """Clear all expired entries."""
        now = datetime.now()
        expired_keys = []

        for key, entry in self.store.items():
            if now > datetime.fromisoformat(entry["expiry"]):
                expired_keys.append(key)

        for key in expired_keys:
            del self.store[key]

        return len(expired_keys)


class ConversationStore(MemoryStore):
    """Store for conversation-specific data."""

    def save_analysis(
        self, session_id: str, symbol: str, analysis_type: str, data: dict[str, Any]
    ) -> None:
        """Save analysis results for a conversation."""
        key = f"{session_id}:analysis:{symbol}:{analysis_type}"

        analysis_record = {
            "symbol": symbol,
            "type": analysis_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        self.set(key, analysis_record)

    def get_analysis(
        self, session_id: str, symbol: str, analysis_type: str
    ) -> dict[str, Any] | None:
        """Get cached analysis for a symbol."""
        key = f"{session_id}:analysis:{symbol}:{analysis_type}"
        return self.get(key)

    def save_context(self, session_id: str, context_type: str, data: Any) -> None:
        """Save conversation context."""
        key = f"{session_id}:context:{context_type}"
        self.set(key, data)

    def get_context(self, session_id: str, context_type: str) -> Any | None:
        """Get conversation context."""
        key = f"{session_id}:context:{context_type}"
        return self.get(key)

    def list_analyses(self, session_id: str) -> list[dict[str, Any]]:
        """List all analyses for a session."""
        analyses = []
        prefix = f"{session_id}:analysis:"

        for key, entry in self.store.items():
            if key.startswith(prefix):
                analyses.append(entry["value"])

        return analyses


class UserMemoryStore(MemoryStore):
    """Store for user-specific long-term memory."""

    def __init__(self, ttl_hours: float = 168.0):  # 1 week default
        super().__init__(ttl_hours)

    def save_preference(self, user_id: str, preference_type: str, value: Any) -> None:
        """Save user preference."""
        key = f"user:{user_id}:pref:{preference_type}"
        self.set(key, value, ttl_hours=self.ttl_hours * 4)  # Longer TTL for preferences

    def get_preference(self, user_id: str, preference_type: str) -> Any | None:
        """Get user preference."""
        key = f"user:{user_id}:pref:{preference_type}"
        return self.get(key)

    def save_trade_history(self, user_id: str, trade: dict[str, Any]) -> None:
        """Save trade to history."""
        key = f"user:{user_id}:trades"

        trades = self.get(key) or []
        trades.append({**trade, "timestamp": datetime.now().isoformat()})

        # Keep last 100 trades
        trades = trades[-100:]
        self.set(key, trades)

    def get_trade_history(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get user's trade history."""
        key = f"user:{user_id}:trades"
        trades = self.get(key) or []
        return trades[-limit:]

    def save_watchlist(self, user_id: str, symbols: list[str]) -> None:
        """Save user's watchlist."""
        key = f"user:{user_id}:watchlist"
        self.set(key, symbols)

    def get_watchlist(self, user_id: str) -> list[str]:
        """Get user's watchlist."""
        key = f"user:{user_id}:watchlist"
        return self.get(key) or []

    def update_risk_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        """Update user's risk profile."""
        key = f"user:{user_id}:risk_profile"
        self.set(key, profile, ttl_hours=self.ttl_hours * 4)

    def get_risk_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get user's risk profile."""
        key = f"user:{user_id}:risk_profile"
        return self.get(key)
