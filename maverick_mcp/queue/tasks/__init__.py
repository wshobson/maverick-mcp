"""
Async task implementations for Maverick-MCP.

This module contains Celery task definitions for various long-running operations.
"""

from .base import BaseTask
from .data_processing import (
    bulk_technical_analysis_task,
    cache_warming_task,
    cleanup_expired_jobs,
    health_check,
)
from .portfolio import (
    multi_ticker_analysis_task,
    portfolio_correlation_task,
    risk_adjusted_analysis_task,
)
from .screening import (
    comprehensive_screening_task,
    custom_screening_task,
    maverick_screening_task,
    trending_screening_task,
)

__all__ = [
    "BaseTask",
    # Screening tasks
    "maverick_screening_task",
    "trending_screening_task",
    "custom_screening_task",
    "comprehensive_screening_task",
    # Portfolio tasks
    "portfolio_correlation_task",
    "multi_ticker_analysis_task",
    "risk_adjusted_analysis_task",
    # Data processing tasks
    "bulk_technical_analysis_task",
    "cache_warming_task",
    "cleanup_expired_jobs",
    "health_check",
]
