#!/bin/bash

# Maverick-MCP Development Script
# This script starts the backend MCP server for personal stock analysis

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Maverick-MCP Development Environment${NC}"

# Move to repo root early so `.env` can influence defaults
cd "$(dirname "$0")/.."
echo -e "${YELLOW}Current directory: $(pwd)${NC}"

# Source .env if it exists
if [ -f .env ]; then
    source .env
fi

# Golden path defaults:
# - Streamable HTTP for local HTTP dev (localhost-only by default)
# - STDIO for Claude Desktop (use `make dev-stdio` or set MAVERICK_TRANSPORT=stdio)
TRANSPORT=${MAVERICK_TRANSPORT:-streamable-http}
HOST=${MAVERICK_HOST:-127.0.0.1}
PORT=${MAVERICK_PORT:-8003}

# Kill any existing processes on the port to avoid conflicts
echo -e "${YELLOW}Checking for existing processes on port ${PORT}...${NC}"
EXISTING_PID=$(lsof -ti:${PORT} 2>/dev/null || true)
if [ ! -z "$EXISTING_PID" ]; then
    echo -e "${YELLOW}Found existing process(es) on port ${PORT}: $EXISTING_PID${NC}"
    echo -e "${YELLOW}Killing existing processes...${NC}"
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 1
else
    echo -e "${GREEN}No existing processes found on port ${PORT}${NC}"
fi

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo -e "${YELLOW}Starting Redis...${NC}"
    if command -v brew &> /dev/null; then
        brew services start redis
    else
        redis-server --daemonize yes
    fi
else
    echo -e "${GREEN}Redis is already running${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    # Kill backend process
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Development environment stopped${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start backend
echo -e "${YELLOW}Starting backend MCP server...${NC}"

# Check if uv is available (more relevant than python since we use uv run)
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv not found! Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

# Validate critical environment variables
echo -e "${YELLOW}Validating environment...${NC}"
if [ -z "$TIINGO_API_KEY" ]; then
    echo -e "${RED}Warning: TIINGO_API_KEY not set - stock data tools may not work${NC}"
fi

if [ -z "$EXA_API_KEY" ] && [ -z "$TAVILY_API_KEY" ]; then
    echo -e "${RED}Warning: Neither EXA_API_KEY nor TAVILY_API_KEY set - research tools may be limited${NC}"
fi

echo -e "${YELLOW}Starting backend with: uv run python -m maverick_mcp.api.server --transport ${TRANSPORT} --host ${HOST} --port ${PORT}${NC}"
echo -e "${YELLOW}Transport: ${TRANSPORT}${NC}"

# Run backend with FastMCP in development mode (show real-time output)
echo -e "${YELLOW}Starting server with real-time output...${NC}"
# PYTHONUNBUFFERED=1 ensures logs flush immediately through the tee pipe,
# so the health-check grep below can detect tool-registration messages.
# PYTHONWARNINGS suppresses websockets deprecation warnings from uvicorn.
PYTHONUNBUFFERED=1 \
PYTHONWARNINGS="ignore::DeprecationWarning:websockets.*,ignore::DeprecationWarning:uvicorn.*" \
uv run python -m maverick_mcp.api.server --transport ${TRANSPORT} --host ${HOST} --port ${PORT} 2>&1 | tee backend.log &
BACKEND_PID=$!
echo -e "${YELLOW}Backend PID: $BACKEND_PID${NC}"

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"

# Wait up to 60 seconds for readiness.
# Primary probe: GET /health/ready returns {"ready": true, "tools": N, ...}
# once the DB is reachable AND tools are registered. The structured endpoint
# survives log-message renames (unlike grepping backend.log).
# Fallback: log-grep + port check, used only for the STDIO transport which
# has no HTTP endpoint.
TOOLS_REGISTERED=false
PORT_OPEN=false
TOOL_COUNT=0
READY_TIMEOUT=60

# STDIO transport has no HTTP endpoint — rely on log signal + process liveness
if [ "$TRANSPORT" = "stdio" ]; then
    for i in $(seq 1 $READY_TIMEOUT); do
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "${RED}Backend process died! Check output above for errors.${NC}"
            exit 1
        fi
        if [ -f backend.log ] && grep -q "Tool registration process completed" backend.log 2>/dev/null; then
            TOOLS_REGISTERED=true
            echo -e "${GREEN}Tool registration complete (STDIO transport)${NC}"
            break
        fi
        echo -e "${YELLOW}Waiting for STDIO backend... ($i/${READY_TIMEOUT})${NC}"
        sleep 1
    done
else
    for i in $(seq 1 $READY_TIMEOUT); do
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "${RED}Backend process died! Check output above for errors.${NC}"
            exit 1
        fi

        # Probe the structured readiness endpoint. Parse JSON via python3
        # (portable across GNU/BSD; avoids sed `\+` dialect issue on macOS).
        READY_JSON=$(curl -fsS --max-time 2 "http://localhost:${PORT}/health/ready" 2>/dev/null || true)
        if [ -n "$READY_JSON" ]; then
            PORT_OPEN=true
            PARSED=$(printf '%s' "$READY_JSON" | python3 -c 'import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(str(d.get("ready", False)).lower(), d.get("tools", 0))
except Exception:
    print("false 0")
' 2>/dev/null || echo "false 0")
            READY_FLAG=$(echo "$PARSED" | awk '{print $1}')
            TOOL_COUNT=$(echo "$PARSED" | awk '{print $2}')
            if [ "$READY_FLAG" = "true" ]; then
                TOOLS_REGISTERED=true
                echo -e "${GREEN}Backend ready: tools=${TOOL_COUNT}${NC}"
                break
            fi
        fi

        echo -e "${YELLOW}Waiting for backend... ($i/${READY_TIMEOUT}) port=${PORT_OPEN} ready=${TOOLS_REGISTERED}${NC}"

        if [ $i -eq $READY_TIMEOUT ]; then
            echo -e "${RED}Backend failed to become ready after ${READY_TIMEOUT} seconds!${NC}"
            echo -e "${RED}Final state: port_open=${PORT_OPEN} tools_registered=${TOOLS_REGISTERED}${NC}"
            echo -e "${RED}Check backend.log for errors.${NC}"
            # Don't exit - let it continue in case tools load later
        fi

        sleep 1
    done
fi

if [ "$TOOLS_REGISTERED" = true ]; then
    echo -e "${GREEN}Backend is ready with tools registered!${NC}"
else
    echo -e "${YELLOW}Backend appears to be running but tool registration status unclear${NC}"
fi

echo -e "${GREEN}Backend started successfully on http://localhost:${PORT}${NC}"

# Show information
echo -e "\n${GREEN}Development environment is running!${NC}"
echo -e "${YELLOW}MCP Server:${NC} http://localhost:${PORT}"
echo -e "${YELLOW}Health Check:${NC} http://localhost:${PORT}/health"

# Show endpoint based on transport type
if [ "$TRANSPORT" = "sse" ]; then
    echo -e "${YELLOW}MCP SSE Endpoint:${NC} http://localhost:${PORT}/sse"
elif [ "$TRANSPORT" = "streamable-http" ]; then
    echo -e "${YELLOW}MCP HTTP Endpoint:${NC} http://localhost:${PORT}/mcp"
    echo -e "${YELLOW}Test with curl:${NC} curl -X POST http://localhost:${PORT}/mcp"
elif [ "$TRANSPORT" = "stdio" ]; then
    echo -e "${YELLOW}MCP Transport:${NC} STDIO (no HTTP endpoint)"
fi

echo -e "${YELLOW}Logs:${NC} tail -f backend.log"

if [ "$TOOLS_REGISTERED" = true ]; then
    echo -e "\n${GREEN}✓ Research tools are registered and ready${NC}"
else
    echo -e "\n${YELLOW}⚠ Tool registration status unclear${NC}"
    echo -e "${YELLOW}Debug: Check backend.log for tool registration messages${NC}"
    echo -e "${YELLOW}Debug: Look for 'Successfully registered' or 'research tools' in logs${NC}"
fi

echo -e "\n${YELLOW}Claude Desktop Configuration:${NC}"
if [ "$TRANSPORT" = "sse" ]; then
    echo -e "${GREEN}SSE Transport (debug/inspector use):${NC}"
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "npx", "args": ["-y", "mcp-remote", "http://localhost:'${PORT}'/sse"]}}}'
elif [ "$TRANSPORT" = "stdio" ]; then
    echo -e "${GREEN}STDIO Transport (recommended):${NC}"
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "uv", "args": ["run", "python", "-m", "maverick_mcp.api.server", "--transport", "stdio"], "cwd": "'$(pwd)'"}}}'
elif [ "$TRANSPORT" = "streamable-http" ]; then
    echo -e "${GREEN}Streamable-HTTP Transport (mcp-remote bridge):${NC}"
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "npx", "args": ["-y", "mcp-remote", "http://localhost:'${PORT}'/mcp"]}}}'
else
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "npx", "args": ["-y", "mcp-remote", "http://localhost:'${PORT}'/mcp"]}}}'
fi

echo -e "\n${YELLOW}Connection Stability Features:${NC}"
if [ "$TRANSPORT" = "sse" ]; then
    echo -e "  • SSE transport (debug/inspector use)"
    echo -e "  • Uses mcp-remote bridge"
elif [ "$TRANSPORT" = "stdio" ]; then
    echo -e "  • Direct STDIO transport (no network layer)"
    echo -e "  • No mcp-remote needed (direct Claude Desktop integration)"
    echo -e "  • No session management issues"
    echo -e "  • No timeout problems"
elif [ "$TRANSPORT" = "streamable-http" ]; then
    echo -e "  • Streamable-HTTP transport (FastMCP 2.0 standard)"
    echo -e "  • Uses mcp-remote bridge for Claude Desktop"
    echo -e "  • Ideal for testing with curl/Postman/REST clients"
    echo -e "  • Good for debugging transport-specific issues"
    echo -e "  • Alternative to SSE for compatibility testing"
else
    echo -e "  • HTTP transport with mcp-remote bridge"
    echo -e "  • Alternative to SSE for compatibility"
    echo -e "  • Single process management"
fi
echo -e "\nPress Ctrl+C to stop the server"

# Wait for process
wait
