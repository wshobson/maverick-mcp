"""SQLAlchemy models for the risk dashboard."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text

from maverick_mcp.data.models import TimestampMixin
from maverick_mcp.database.base import Base


class RiskAlert(Base, TimestampMixin):
    """Persistent risk alert for a portfolio."""

    __tablename__ = "risk_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_name = Column(String(255), nullable=False, index=True)
    alert_type = Column(
        String(50), nullable=False
    )  # concentration / correlation / drawdown / sizing
    severity = Column(String(20), nullable=False)  # warning / critical
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False)


class RiskSnapshot(Base, TimestampMixin):
    """Point-in-time risk metrics snapshot for a portfolio."""

    __tablename__ = "risk_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_name = Column(String(255), nullable=False, index=True)
    var_95 = Column(Float, nullable=False)
    var_99 = Column(Float, nullable=False)
    max_sector_pct = Column(Float, nullable=False)
    max_correlation = Column(Float, nullable=False)
    beta_weighted_delta = Column(Float, nullable=True)
    regime = Column(String(20), nullable=True)
    snapshot_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
