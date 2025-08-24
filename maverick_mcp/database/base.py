"""
Shared database base class for all SQLAlchemy models.

This module provides a common Base class to avoid circular imports
and ensure all models are registered with the same metadata.
"""

from typing import Any

from sqlalchemy.ext.declarative import declarative_base

# Shared base class for all SQLAlchemy models
Base: Any = declarative_base()
