#!/bin/bash
#
# Start Flower monitoring for Celery
#

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Source environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default configuration
FLOWER_PORT=${FLOWER_PORT:-5555}
FLOWER_BASIC_AUTH=${FLOWER_BASIC_AUTH:-"admin:maverick123"}

echo "Starting Flower Celery monitoring..."
echo "Port: $FLOWER_PORT"
echo "Access: http://localhost:$FLOWER_PORT"

# Start Flower
exec celery -A maverick_mcp.queue.celery_app flower \
    --port="$FLOWER_PORT" \
    --basic_auth="$FLOWER_BASIC_AUTH" \
    --url_prefix=flower \
    --persistent \
    --db=flower.db