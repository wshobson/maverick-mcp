#!/bin/bash

# Start the MaverickMCP backend server with API endpoints
# This script starts the multi-transport server with both MCP and HTTP API endpoints

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TRANSPORT="http"
WORKERS="${WORKERS:-1}"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) 
            TRANSPORT="dev"
            export AUTH_ENABLED=false
            export CREDIT_SYSTEM_ENABLED=false
            echo "Running in development mode (auth and credits disabled)"
            ;;
        --stdio) 
            TRANSPORT="stdio"
            ;;
        --host) 
            HOST="$2"
            shift
            ;;
        --port) 
            PORT="$2"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift
            ;;
        *) 
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--dev] [--stdio] [--host HOST] [--port PORT] [--workers N]"
            exit 1
            ;;
    esac
    shift
done

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Log the configuration
echo "Starting MaverickMCP Backend Server with API endpoints"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Transport: $TRANSPORT"
echo "Workers: $WORKERS"
echo "Auth Enabled: ${AUTH_ENABLED:-true}"
echo "Credit System Enabled: ${CREDIT_SYSTEM_ENABLED:-true}"

# Run the multi-transport server
if [ "$TRANSPORT" = "stdio" ]; then
    # Run in stdio mode for MCP clients
    uv run python -m maverick_mcp.api.server_multi --transport stdio
elif [ "$TRANSPORT" = "dev" ]; then
    # Development mode with hot reloading
    uv run python -m maverick_mcp.api.server_multi --transport dev --host "$HOST" --port "$PORT"
else
    # Production mode with gunicorn
    if [ "$WORKERS" -gt 1 ]; then
        echo "Starting with gunicorn ($WORKERS workers)..."
        uv run gunicorn maverick_mcp.api.server_multi:app \
            --bind "$HOST:$PORT" \
            --workers "$WORKERS" \
            --worker-class uvicorn.workers.UvicornWorker \
            --timeout 120 \
            --keep-alive 5 \
            --max-requests 1000 \
            --max-requests-jitter 50 \
            --access-logfile - \
            --error-logfile -
    else
        # Single worker mode (uses uvicorn directly)
        uv run python -m maverick_mcp.api.server_multi --transport http --host "$HOST" --port "$PORT"
    fi
fi