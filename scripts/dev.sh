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

# Kill any existing processes on port 8003 to avoid conflicts
echo -e "${YELLOW}Checking for existing processes on port 8003...${NC}"
EXISTING_PID=$(lsof -ti:8003 2>/dev/null || true)
if [ ! -z "$EXISTING_PID" ]; then
    echo -e "${YELLOW}Found existing process(es) on port 8003: $EXISTING_PID${NC}"
    echo -e "${YELLOW}Killing existing processes...${NC}"
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 1
else
    echo -e "${GREEN}No existing processes found on port 8003${NC}"
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
cd "$(dirname "$0")/.."
echo -e "${YELLOW}Current directory: $(pwd)${NC}"

# Source .env if it exists
if [ -f .env ]; then
    source .env
fi

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

# Choose transport based on environment variable or default to SSE for reliability
TRANSPORT=${MAVERICK_TRANSPORT:-sse}
echo -e "${YELLOW}Starting backend with: uv run python -m maverick_mcp.api.server --transport ${TRANSPORT} --host 0.0.0.0 --port 8003${NC}"
echo -e "${YELLOW}Transport: ${TRANSPORT} (recommended for Claude Desktop stability)${NC}"

# Run backend with FastMCP in development mode (show real-time output)
echo -e "${YELLOW}Starting server with real-time output...${NC}"
# Set PYTHONWARNINGS to suppress websockets deprecation warnings from uvicorn
PYTHONWARNINGS="ignore::DeprecationWarning:websockets.*,ignore::DeprecationWarning:uvicorn.*" \
uv run python -m maverick_mcp.api.server --transport ${TRANSPORT} --host 0.0.0.0 --port 8003 2>&1 | tee backend.log &
BACKEND_PID=$!
echo -e "${YELLOW}Backend PID: $BACKEND_PID${NC}"

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"

# Wait up to 45 seconds for the backend to start and tools to register
TOOLS_REGISTERED=false
for i in {1..45}; do
    # Check if backend process is still running first
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}Backend process died! Check output above for errors.${NC}"
        exit 1
    fi
    
    # Check if port is open
    if nc -z localhost 8003 2>/dev/null || curl -s http://localhost:8003/health >/dev/null 2>&1; then
        if [ "$TOOLS_REGISTERED" = false ]; then
            echo -e "${GREEN}Backend port is open, checking for tool registration...${NC}"
            
            # Check backend.log for tool registration messages
            if grep -q "Research tools registered successfully" backend.log 2>/dev/null || 
               grep -q "Tool registration process completed" backend.log 2>/dev/null || 
               grep -q "Tools registered successfully" backend.log 2>/dev/null; then
                echo -e "${GREEN}Research tools successfully registered!${NC}"
                TOOLS_REGISTERED=true
                break
            else
                echo -e "${YELLOW}Backend running but tools not yet registered... ($i/45)${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Still waiting for backend to start... ($i/45)${NC}"
    fi
    
    if [ $i -eq 45 ]; then
        echo -e "${RED}Backend failed to fully initialize after 45 seconds!${NC}"
        echo -e "${RED}Server may be running but tools not registered. Check output above.${NC}"
        # Don't exit - let it continue in case tools load later
    fi
    
    sleep 1
done

if [ "$TOOLS_REGISTERED" = true ]; then
    echo -e "${GREEN}Backend is ready with tools registered!${NC}"
else
    echo -e "${YELLOW}Backend appears to be running but tool registration status unclear${NC}"
fi

echo -e "${GREEN}Backend started successfully on http://localhost:8003${NC}"

# Show information
echo -e "\n${GREEN}Development environment is running!${NC}"
echo -e "${YELLOW}MCP Server:${NC} http://localhost:8003"
echo -e "${YELLOW}Health Check:${NC} http://localhost:8003/health"
echo -e "${YELLOW}MCP SSE Endpoint:${NC} http://localhost:8003/sse"
echo -e "${YELLOW}Logs:${NC} tail -f backend.log"

if [ "$TOOLS_REGISTERED" = true ]; then
    echo -e "\n${GREEN}✓ Research tools are registered and ready${NC}"
else
    echo -e "\n${YELLOW}⚠ Tool registration status unclear${NC}"
    echo -e "${YELLOW}Debug: Check backend.log for tool registration messages${NC}"
    echo -e "${YELLOW}Debug: Look for 'Successfully registered' or 'research tools' in logs${NC}"
fi

echo -e "\n${YELLOW}Claude Desktop Configuration (SSE - tested and stable):${NC}"
if [ "$TRANSPORT" = "sse" ]; then
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "npx", "args": ["-y", "mcp-remote", "http://localhost:8003/sse"]}}}'
elif [ "$TRANSPORT" = "stdio" ]; then
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "uv", "args": ["run", "python", "-m", "maverick_mcp.api.server", "--transport", "stdio"], "cwd": "'$(pwd)'"}}}'
else
    echo -e '{"mcpServers": {"maverick-mcp": {"command": "npx", "args": ["-y", "mcp-remote", "http://localhost:8003/mcp"]}}}'
fi

echo -e "\n${YELLOW}Connection Stability Features:${NC}"
if [ "$TRANSPORT" = "sse" ]; then
    echo -e "  • SSE transport (tested and stable for Claude Desktop)"
    echo -e "  • Uses mcp-remote bridge for reliable connection"
    echo -e "  • Prevents tools from disappearing"
    echo -e "  • Persistent connection with session management"
    echo -e "  • Adaptive timeout system for research tools"
elif [ "$TRANSPORT" = "stdio" ]; then
    echo -e "  • Direct STDIO transport (no network layer)"
    echo -e "  • No mcp-remote needed (direct Claude Desktop integration)"
    echo -e "  • No session management issues"
    echo -e "  • No timeout problems"
else
    echo -e "  • HTTP transport with mcp-remote bridge"
    echo -e "  • Alternative to SSE for compatibility"
    echo -e "  • Single process management"
fi
echo -e "\nPress Ctrl+C to stop the server"

# Wait for process
wait