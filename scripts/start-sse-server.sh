#!/bin/bash

# Start MaverickMCP SSE Server for production
# This creates an SSE endpoint that MCP clients can connect to

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MaverickMCP SSE Server${NC}"

# Set environment variables for production SSE
export AUTH_ENABLED=${AUTH_ENABLED:-true}
export CREDIT_SYSTEM_ENABLED=${CREDIT_SYSTEM_ENABLED:-true}

# Default to port 8080 for SSE
export PORT=${PORT:-8080}

echo -e "${YELLOW}Configuration:${NC}"
echo "  - Port: $PORT"
echo "  - Authentication: $AUTH_ENABLED"
echo "  - Credit System: $CREDIT_SYSTEM_ENABLED"
echo "  - SSE Endpoint: http://localhost:$PORT/sse"

# Change to project directory
cd "$(dirname "$0")/.."

# Run the server with SSE transport
echo -e "${GREEN}Starting server...${NC}"
exec uv run python -m maverick_mcp.api.server_multi --transport http --port $PORT