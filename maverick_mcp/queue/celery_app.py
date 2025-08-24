"""
Celery application configuration for Maverick-MCP.

This module configures Celery with Redis backend for async job processing.
"""

from celery import Celery

from maverick_mcp.config.settings import settings
from maverick_mcp.utils.logging import get_logger

logger = get_logger("maverick_mcp.queue.celery")

# Redis configuration for Celery
redis_url = (
    f"redis://{settings.redis.host}:{settings.redis.port}/{settings.redis.db + 1}"
)
result_backend_url = (
    f"redis://{settings.redis.host}:{settings.redis.port}/{settings.redis.db + 2}"
)

# Create Celery application
celery_app = Celery(
    "maverick_mcp",
    broker=redis_url,
    backend=result_backend_url,
    include=[
        "maverick_mcp.queue.tasks.screening",
        "maverick_mcp.queue.tasks.portfolio",
        "maverick_mcp.queue.tasks.data_processing",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "maverick_mcp.queue.tasks.screening.*": {"queue": "screening"},
        "maverick_mcp.queue.tasks.portfolio.*": {"queue": "portfolio"},
        "maverick_mcp.queue.tasks.data_processing.*": {"queue": "data_processing"},
    },
    # Task execution settings
    task_time_limit=3600,  # 1 hour max execution time
    task_soft_time_limit=3300,  # 55 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_compression="gzip",
    result_serializer="json",
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    # Retry settings
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-expired-jobs": {
            "task": "maverick_mcp.queue.tasks.data_processing.cleanup_expired_jobs",
            "schedule": 3600.0,  # Run every hour
        },
        "health-check": {
            "task": "maverick_mcp.queue.tasks.data_processing.health_check",
            "schedule": 300.0,  # Run every 5 minutes
        },
    },
    timezone="UTC",
    # Worker settings
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    # Monitoring
    task_send_sent_event=True,
    worker_send_task_events=True,
)

# Configure logging for Celery
if settings.api.debug:
    celery_app.conf.worker_log_level = "DEBUG"
else:
    celery_app.conf.worker_log_level = "INFO"

logger.info(f"Celery app configured with broker: {redis_url}")
logger.info(f"Result backend: {result_backend_url}")
