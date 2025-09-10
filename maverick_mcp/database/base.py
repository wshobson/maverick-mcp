"""
Shared database base class for all SQLAlchemy models.

This module provides a common Base class to avoid circular imports
and ensure all models are registered with the same metadata.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models using SQLAlchemy 2.0+ style."""

    pass
