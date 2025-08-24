#!/bin/bash

# Start Backend API Server

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting MaverickMCP Backend${NC}"

# Change to project root
cd "$(dirname "$0")/.."

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo -e "${RED}Warning: .env file not found${NC}"
fi

# Check Redis
if ! pgrep -x "redis-server" > /dev/null; then
    echo -e "${YELLOW}Starting Redis...${NC}"
    if command -v brew &> /dev/null; then
        brew services start redis
    else
        redis-server --daemonize yes
    fi
fi

# Check PostgreSQL (optional)
if [ ! -z "$DATABASE_URL" ]; then
    echo -e "${YELLOW}Using PostgreSQL database${NC}"
fi

# Check migration status (but don't run automatically)
if [ -f alembic.ini ] && [ ! -z "$DATABASE_URL" ]; then
    echo -e "${YELLOW}Checking migration status...${NC}"
    CURRENT_REV=$(alembic current 2>/dev/null | grep -v "INFO" | head -1 || echo "none")
    LATEST_REV=$(alembic heads 2>/dev/null | grep -v "INFO" | head -1 || echo "none")
    
    if [ "$CURRENT_REV" != "$LATEST_REV" ]; then
        echo -e "${RED}WARNING: Database migrations are pending!${NC}"
        echo -e "${YELLOW}Current revision: $CURRENT_REV${NC}"
        echo -e "${YELLOW}Latest revision:  $LATEST_REV${NC}"
        echo -e "${YELLOW}Run './scripts/run-migrations.sh upgrade' to apply migrations${NC}"
        
        # In production, fail if migrations are pending
        if [ "$AUTH_ENABLED" == "true" ] || [ "$1" != "--dev" ]; then
            echo -e "${RED}Error: Cannot start in production mode with pending migrations${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Database is up to date${NC}"
    fi
fi

# Start the server
echo -e "${GREEN}Starting API server on http://localhost:8000${NC}"

# Check which server mode to use
if [ "$1" == "--dev" ]; then
    # Development mode with SSE + stdio (auth and credits disabled)
    echo -e "${YELLOW}Starting in development mode (auth/credits disabled, SSE + stdio)${NC}"
    echo -e "${GREEN}SSE endpoint: http://localhost:8000/sse${NC}"
    echo -e "${GREEN}stdio transport: Available for MCP clients${NC}"
    AUTH_ENABLED=false CREDIT_SYSTEM_ENABLED=false uv run python -m maverick_mcp.api.server_multi --transport dev --host 0.0.0.0 --port 8000
elif [ "$1" == "--stdio" ]; then
    # MCP stdio mode for Claude Desktop
    echo -e "${YELLOW}Starting in stdio mode for MCP clients${NC}"
    uv run python -m maverick_mcp.api.server_multi --transport stdio
elif [ "$1" == "--multi" ]; then
    # Multi-transport mode (both SSE and Streamable HTTP)
    echo -e "${YELLOW}Starting in multi-transport mode (SSE + Streamable HTTP)${NC}"
    echo -e "${GREEN}SSE endpoint: http://localhost:8000/sse${NC}"
    echo -e "${GREEN}HTTP endpoint: http://localhost:8000/mcp${NC}"
    AUTH_ENABLED=${AUTH_ENABLED:-true} CREDIT_SYSTEM_ENABLED=${CREDIT_SYSTEM_ENABLED:-true} uv run python -m maverick_mcp.api.server_multi --host 0.0.0.0 --port 8000
else
    # Production mode with authentication and credits (multi-transport by default)
    echo -e "${YELLOW}Starting in production mode (auth/credits enabled, multi-transport with gunicorn)${NC}"
    echo -e "${GREEN}SSE endpoint: http://localhost:8000/sse${NC}"
    echo -e "${GREEN}HTTP endpoint: http://localhost:8000/mcp${NC}"
    AUTH_ENABLED=${AUTH_ENABLED:-true} CREDIT_SYSTEM_ENABLED=${CREDIT_SYSTEM_ENABLED:-true} gunicorn maverick_mcp.api.server_multi:app \
        --bind 0.0.0.0:8000 \
        --workers ${WORKERS:-4} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 120 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --access-logfile - \
        --error-logfile -
fi