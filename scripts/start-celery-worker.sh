#!/bin/bash
#
# Start Celery worker for async job processing
#

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Source environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default configuration
WORKER_NAME=${CELERY_WORKER_NAME:-"maverick-worker"}
CONCURRENCY=${CELERY_CONCURRENCY:-4}
LOG_LEVEL=${CELERY_LOG_LEVEL:-"info"}
QUEUES=${CELERY_QUEUES:-"screening,portfolio,data_processing,default"}

echo "Starting Celery worker..."
echo "Worker name: $WORKER_NAME"
echo "Concurrency: $CONCURRENCY"
echo "Queues: $QUEUES"
echo "Log level: $LOG_LEVEL"

# Start Celery worker
exec celery -A maverick_mcp.queue.celery_app worker \
    --hostname="$WORKER_NAME@%h" \
    --concurrency="$CONCURRENCY" \
    --queues="$QUEUES" \
    --loglevel="$LOG_LEVEL" \
    --pool=prefork \
    --without-gossip \
    --without-mingle \
    --without-heartbeat \
    --time-limit=3600 \
    --soft-time-limit=3300 \
    --max-tasks-per-child=1000 \
    --max-memory-per-child=500000 \
    --optimization=fair