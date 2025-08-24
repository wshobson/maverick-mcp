#!/bin/bash
#
# Start Celery beat scheduler for periodic tasks
#

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Source environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default configuration
LOG_LEVEL=${CELERY_LOG_LEVEL:-"info"}
BEAT_SCHEDULE_FILE=${CELERY_BEAT_SCHEDULE:-"celerybeat-schedule"}

echo "Starting Celery beat scheduler..."
echo "Log level: $LOG_LEVEL"
echo "Schedule file: $BEAT_SCHEDULE_FILE"

# Remove existing schedule file if it exists (for clean restart)
if [ -f "$BEAT_SCHEDULE_FILE" ]; then
    echo "Removing existing schedule file: $BEAT_SCHEDULE_FILE"
    rm -f "$BEAT_SCHEDULE_FILE"
fi

# Start Celery beat
exec celery -A maverick_mcp.queue.celery_app beat \
    --loglevel="$LOG_LEVEL" \
    --schedule="$BEAT_SCHEDULE_FILE" \
    --pidfile=celerybeat.pid