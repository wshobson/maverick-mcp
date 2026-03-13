"""
Persistent checkpointer factory for LangGraph agents.

Provides a SQLite-backed checkpointer that survives server restarts,
replacing the default in-memory MemorySaver.
"""

import logging
import sqlite3
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from maverick_mcp.memory.stores import resolve_db_path

logger = logging.getLogger(__name__)

# Module-level cache so all agents share one checkpointer instance.
_checkpointer: BaseCheckpointSaver | None = None
_sqlite_conn: sqlite3.Connection | None = None


def get_persistent_checkpointer(
    db_path: str | None = None,
) -> BaseCheckpointSaver[Any]:
    """Return a shared, persistent SQLite-backed checkpointer.

    The checkpointer is created once and reused across all agents in
    the process.  If ``langgraph-checkpoint-sqlite`` is not installed
    or the database cannot be opened, falls back to the in-memory
    ``MemorySaver`` with a warning.

    Args:
        db_path: Optional explicit path to the checkpoint SQLite file.
                 Defaults to the value of ``MAVERICK_CHECKPOINT_DB_PATH``
                 from settings (``data/checkpoints.db``).

    Returns:
        A ``BaseCheckpointSaver`` instance (``SqliteSaver`` when possible).
    """
    global _checkpointer, _sqlite_conn  # noqa: PLW0603

    if _checkpointer is not None:
        return _checkpointer

    # Resolve the database path
    if db_path is None:
        try:
            from maverick_mcp.config.settings import get_settings

            settings = get_settings()
            db_path = settings.memory.checkpoint_db_path
        except Exception:
            db_path = "data/checkpoints.db"

    resolved_path = resolve_db_path(db_path)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(resolved_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn)
        saver.setup()
        _sqlite_conn = conn
        _checkpointer = saver
        logger.info("Using persistent SqliteSaver at %s", resolved_path)
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed; "
            "falling back to in-memory MemorySaver. "
            "Install it with: uv add langgraph-checkpoint-sqlite"
        )
        _checkpointer = MemorySaver()
    except Exception:
        logger.exception(
            "Failed to open SQLite checkpoint database at %s; "
            "falling back to in-memory MemorySaver",
            resolved_path,
        )
        _checkpointer = MemorySaver()

    return _checkpointer
