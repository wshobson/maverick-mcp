"""
Decision Logger for agent audit trail.

Provides async-safe logging of agent decisions to the database.
All methods are designed to be non-blocking and never crash the main flow --
if a logging write fails, the error is silently captured and logged
to the standard Python logger.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from maverick_mcp.data.models import DecisionLog, SessionLocal

logger = logging.getLogger("maverick_mcp.utils.decision_logger")

# Module-level flag and lock for lazy table creation
_table_ensured = False
_table_lock = threading.Lock()


def _ensure_table_exists() -> None:
    """Create the DecisionLog table if it does not already exist.

    Uses a module-level flag so the check runs at most once per process.
    Errors are swallowed -- table creation failure must never crash the caller.
    """
    global _table_ensured
    if _table_ensured:
        return

    with _table_lock:
        # Double-checked locking
        if _table_ensured:
            return
        try:
            from maverick_mcp.data.models import engine

            DecisionLog.__table__.create(bind=engine, checkfirst=True)
            _table_ensured = True
            logger.debug("DecisionLog table ensured")
        except Exception:
            # Leave _table_ensured False so the next call retries.
            # This handles transient DB failures during startup.
            logger.debug("DecisionLog table check failed; will retry", exc_info=True)


class DecisionLogger:
    """
    Async-safe logger that writes DecisionLog records to the database.

    Every public method catches all exceptions internally so that decision
    logging can never interfere with the primary agent workflow.
    """

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @staticmethod
    def log_decision(
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        query_text: str | None = None,
        query_classification: str | None = None,
        routing_decision: list[str] | None = None,
        models_used: list[str] | None = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        estimated_cost_usd: float = 0.0,
        confidence_score: float = 0.0,
        response_summary: str | None = None,
        duration_ms: int = 0,
        status: str = "success",
        error_category: str | None = None,
    ) -> None:
        """
        Write a single decision record to the database.

        All parameters are keyword-only to improve call-site readability.
        Errors are caught and logged -- this method never raises.
        """
        try:
            _ensure_table_exists()

            # Truncate response summary to prevent oversized text columns
            if response_summary and len(response_summary) > 500:
                response_summary = response_summary[:497] + "..."

            record = DecisionLog(
                timestamp=datetime.now(UTC),
                session_id=session_id,
                request_id=request_id or str(uuid.uuid4())[:8],
                query_text=query_text,
                query_classification=query_classification,
                routing_decision=routing_decision,
                models_used=models_used,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                estimated_cost_usd=Decimal(str(estimated_cost_usd)),
                confidence_score=Decimal(str(confidence_score)),
                response_summary=response_summary,
                duration_ms=duration_ms,
                status=status,
                error_category=error_category,
            )

            with SessionLocal() as db:
                db.add(record)
                db.commit()

            logger.debug(
                "Decision logged: classification=%s, status=%s, duration=%dms",
                query_classification,
                status,
                duration_ms,
            )
        except SQLAlchemyError:
            logger.warning("Failed to write decision log to database", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in decision logger", exc_info=True)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def get_decisions(
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent decision log entries.

        Args:
            session_id: Optional filter by session. When ``None``, returns
                        the most recent entries across all sessions.
            limit: Maximum number of entries to return (default 50).

        Returns:
            List of decision dictionaries, newest first.
        """
        try:
            with SessionLocal() as db:
                if session_id:
                    rows = DecisionLog.get_by_session(db, session_id, limit=limit)
                else:
                    rows = DecisionLog.get_recent(db, limit=limit)
                return [row.to_dict() for row in rows]
        except SQLAlchemyError:
            logger.warning("Failed to read decision log", exc_info=True)
            return []
        except Exception:
            logger.warning("Unexpected error reading decision log", exc_info=True)
            return []

    @staticmethod
    def get_cost_summary(days: int = 7) -> dict[str, Any]:
        """
        Aggregate cost and usage statistics over the last *days* days.

        Returns a dictionary with total cost, token counts, request count,
        average duration, and status breakdown.
        """
        try:
            cutoff = datetime.now(UTC) - timedelta(days=days)

            with SessionLocal() as db:
                rows = (
                    db.query(
                        func.count(DecisionLog.id).label("total_requests"),
                        func.sum(DecisionLog.tokens_input).label("total_tokens_input"),
                        func.sum(DecisionLog.tokens_output).label(
                            "total_tokens_output"
                        ),
                        func.sum(DecisionLog.estimated_cost_usd).label(
                            "total_cost_usd"
                        ),
                        func.avg(DecisionLog.duration_ms).label("avg_duration_ms"),
                    )
                    .filter(DecisionLog.timestamp >= cutoff)
                    .first()
                )

                # Status breakdown
                status_rows = (
                    db.query(
                        DecisionLog.status,
                        func.count(DecisionLog.id).label("count"),
                    )
                    .filter(DecisionLog.timestamp >= cutoff)
                    .group_by(DecisionLog.status)
                    .all()
                )
                status_breakdown = {row.status: row.count for row in status_rows}

                # Classification breakdown
                class_rows = (
                    db.query(
                        DecisionLog.query_classification,
                        func.count(DecisionLog.id).label("count"),
                    )
                    .filter(
                        DecisionLog.timestamp >= cutoff,
                        DecisionLog.query_classification.isnot(None),
                    )
                    .group_by(DecisionLog.query_classification)
                    .all()
                )
                classification_breakdown = {
                    row.query_classification: row.count for row in class_rows
                }

            return {
                "period_days": days,
                "total_requests": rows.total_requests if rows else 0,
                "total_tokens_input": int(rows.total_tokens_input or 0) if rows else 0,
                "total_tokens_output": int(rows.total_tokens_output or 0)
                if rows
                else 0,
                "total_cost_usd": float(rows.total_cost_usd or 0) if rows else 0.0,
                "avg_duration_ms": float(rows.avg_duration_ms or 0) if rows else 0.0,
                "status_breakdown": status_breakdown,
                "classification_breakdown": classification_breakdown,
            }
        except SQLAlchemyError:
            logger.warning("Failed to compute cost summary", exc_info=True)
            return {"error": "Failed to compute cost summary", "period_days": days}
        except Exception:
            logger.warning("Unexpected error computing cost summary", exc_info=True)
            return {"error": "Unexpected error", "period_days": days}


# Module-level singleton for convenient access
decision_logger = DecisionLogger()
