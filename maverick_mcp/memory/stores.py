"""
Memory stores for agent conversations and user data.

Provides write-through persistence using SQLite so that all data
survives server restarts while keeping the in-memory cache for fast reads.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger(__name__)


def _resolve_db_path(path: str) -> str:
    """Resolve a database path, creating parent directories as needed.

    If the path is relative it is resolved against the project root
    (two levels up from this file: memory/ -> maverick_mcp/ -> project root).
    """
    p = Path(path)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        p = project_root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# Re-export for use by checkpointer.py
resolve_db_path = _resolve_db_path


class MemoryStore:
    """Base class for memory storage with SQLite write-through persistence."""

    def __init__(self, ttl_hours: float = 24.0, db_path: str | None = None):
        self.ttl_hours = ttl_hours
        self.store: dict[str, dict[str, Any]] = {}

        # Resolve the persistent database path
        if db_path is None:
            try:
                settings = get_settings()
                db_path = settings.memory.memory_db_path
            except Exception:
                db_path = "data/memory_store.db"
        self._db_path = _resolve_db_path(db_path)

        # Thread lock for SQLite access (sqlite3 connections are not thread-safe)
        self._lock = threading.Lock()

        # Persistent connection (reused for all reads/writes)
        self._conn = sqlite3.connect(
            self._db_path, timeout=10, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")

        # Initialize the persistent store and load existing data
        self._init_db()
        self._load_from_db()

    # ------------------------------------------------------------------ #
    # SQLite helpers
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the persistence table if it does not exist."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_store (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    created TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def _load_from_db(self) -> None:
        """Load all non-expired entries from SQLite into the in-memory dict."""
        now = datetime.now()
        with self._lock:
            # Bulk-delete expired rows first
            self._conn.execute(
                "DELETE FROM memory_store WHERE expiry < ?", (now.isoformat(),)
            )
            self._conn.commit()

            cursor = self._conn.execute(
                "SELECT key, value, expiry, created FROM memory_store"
            )
            for row in cursor.fetchall():
                key, value_json, expiry_iso, created_iso = row
                try:
                    self.store[key] = {
                        "value": json.loads(value_json),
                        "expiry": expiry_iso,
                        "created": created_iso,
                    }
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Skipping corrupt memory entry for key=%s", key)

        logger.info(
            "Loaded %d entries from persistent memory store at %s",
            len(self.store),
            self._db_path,
        )

    def _persist_entry(self, key: str, entry: dict[str, Any]) -> None:
        """Write a single entry to SQLite (upsert)."""
        value_json = json.dumps(entry["value"])
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_store (key, value, expiry, created)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value   = excluded.value,
                    expiry  = excluded.expiry,
                    created = excluded.created
                """,
                (key, value_json, entry["expiry"], entry["created"]),
            )
            self._conn.commit()

    def _delete_entry(self, key: str) -> None:
        """Delete a single entry from SQLite."""
        with self._lock:
            self._conn.execute("DELETE FROM memory_store WHERE key = ?", (key,))
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------ #

    def set(self, key: str, value: Any, ttl_hours: float | None = None) -> None:
        """Store a value with optional custom TTL."""
        ttl = ttl_hours or self.ttl_hours
        expiry = datetime.now() + timedelta(hours=ttl)

        entry = {
            "value": value,
            "expiry": expiry.isoformat(),
            "created": datetime.now().isoformat(),
        }
        self.store[key] = entry
        self._persist_entry(key, entry)

    def get(self, key: str) -> Any | None:
        """Get a value if not expired."""
        if key not in self.store:
            return None

        entry = self.store[key]
        expiry = datetime.fromisoformat(entry["expiry"])

        if datetime.now() > expiry:
            del self.store[key]
            self._delete_entry(key)
            return None

        return entry["value"]

    def delete(self, key: str) -> None:
        """Delete a value."""
        if key in self.store:
            del self.store[key]
        self._delete_entry(key)

    def clear_expired(self) -> int:
        """Clear all expired entries."""
        now = datetime.now()
        expired_keys = []

        for key, entry in self.store.items():
            if now > datetime.fromisoformat(entry["expiry"]):
                expired_keys.append(key)

        for key in expired_keys:
            del self.store[key]
            self._delete_entry(key)

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

    def __init__(
        self, ttl_hours: float = 168.0, db_path: str | None = None
    ):  # 1 week default
        super().__init__(ttl_hours, db_path=db_path)

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


class SharedAgentContext:
    """Cross-agent shared context for a coordination session.

    Allows agents to share findings during multi-agent execution,
    enabling later agents to build on earlier findings.
    """

    def __init__(self):
        self._contexts: dict[str, dict[str, Any]] = {}  # session_id -> context
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> None:
        """Initialize a new coordination session."""
        with self._lock:
            if session_id in self._contexts:
                return  # Don't clobber existing session context
            self._contexts[session_id] = {
                "findings": [],  # List of agent findings
                "metadata": {},  # Session metadata
                "agent_order": [],  # Which agents have contributed
                "created_at": datetime.now().isoformat(),
            }

    def add_finding(
        self, session_id: str, agent_name: str, finding: dict[str, Any]
    ) -> None:
        """Add an agent's finding to the shared context."""
        with self._lock:
            ctx = self._contexts.get(session_id)
            if ctx is None:
                # Auto-create session if it doesn't exist yet
                self._contexts[session_id] = {
                    "findings": [],
                    "metadata": {},
                    "agent_order": [],
                    "created_at": datetime.now().isoformat(),
                }
                ctx = self._contexts[session_id]
            ctx["findings"].append(
                {
                    "agent": agent_name,
                    "data": finding,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            if agent_name not in ctx["agent_order"]:
                ctx["agent_order"].append(agent_name)

    def get_prior_findings(
        self, session_id: str, exclude_agent: str | None = None
    ) -> list[dict[str, Any]]:
        """Get findings from other agents in this session."""
        with self._lock:
            ctx = self._contexts.get(session_id, {})
            findings = ctx.get("findings", [])
            if exclude_agent:
                return [f for f in findings if f["agent"] != exclude_agent]
            return list(findings)

    def get_summary(self, session_id: str) -> str:
        """Get a text summary of all findings for handoff context."""
        findings = self.get_prior_findings(session_id)
        if not findings:
            return ""
        lines = []
        for f in findings:
            agent = f["agent"]
            data = f["data"]
            # Extract key points from the finding
            summary = data.get("summary", str(data)[:200])
            lines.append(f"[{agent}]: {summary}")
        return "\n".join(lines)

    def cleanup_session(self, session_id: str) -> None:
        """Remove a completed session's context."""
        with self._lock:
            self._contexts.pop(session_id, None)


# Module-level singleton
_shared_context = SharedAgentContext()


def get_shared_context() -> SharedAgentContext:
    """Return the module-level shared agent context singleton."""
    return _shared_context
