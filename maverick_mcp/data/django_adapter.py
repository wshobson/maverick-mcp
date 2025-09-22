"""
Django database adapter for Maverick-MCP.

This module provides integration between Maverick-MCP and an existing
Django database, allowing MCP to read Django-owned
tables while maintaining separation of concerns.
"""

import logging
from typing import Any

from sqlalchemy import BigInteger, Boolean, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Create a separate Base for Django table mappings
DjangoBase: Any = declarative_base()


class DjangoUser(DjangoBase):
    """Read-only mapping to Django's users_customuser table."""

    __tablename__ = "users_customuser"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True)
    username = Column(String(150), nullable=False)
    email = Column(String(254), nullable=False)
    first_name = Column(String(150))
    last_name = Column(String(150))
    is_active = Column(Boolean, default=True)
    is_staff = Column(Boolean, default=False)

    def __repr__(self):
        return (
            f"<DjangoUser(id={self.id}, username={self.username}, email={self.email})>"
        )


class DjangoStock(DjangoBase):
    """Read-only mapping to Django's stocks_stock table."""

    __tablename__ = "stocks_stock"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(BigInteger)

    def __repr__(self):
        return f"<DjangoStock(symbol={self.symbol}, name={self.name})>"
class DjangoAdapter:
    """
    Adapter for accessing Django-owned database tables.

    This adapter provides read-only access to Django tables,
    ensuring MCP doesn't modify Django-managed data.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_user_by_email(self, email: str) -> DjangoUser | None:
        """Get Django user by email address."""
        return self.session.query(DjangoUser).filter(DjangoUser.email == email).first()

    def get_user_by_id(self, user_id: int) -> DjangoUser | None:
        """Get Django user by ID."""
        return self.session.query(DjangoUser).filter(DjangoUser.id == user_id).first()

    def get_stock_by_symbol(self, symbol: str) -> DjangoStock | None:
        """Get stock by symbol from Django table."""
        return (
            self.session.query(DjangoStock)
            .filter(DjangoStock.symbol == symbol.upper())
            .first()
        )

    def link_mcp_user_to_django(self, email: str) -> dict | None:
        """
        Link MCP API key to Django user via email.

        Returns user info with placeholder subscription metadata.
        """
        # Find Django user
        django_user = self.get_user_by_email(email)
        if not django_user:
            return None

        return {
            "user_id": django_user.id,
            "username": django_user.username,
            "email": django_user.email,
            "is_active": django_user.is_active,
            "has_subscription": False,
            "subscription_status": None,
            "external_customer_id": None,
        }
