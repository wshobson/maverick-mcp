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

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Python not found!${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting backend with: uv run python -m maverick_mcp.api.server --transport sse --host 0.0.0.0 --port 8000${NC}"

# Run backend with FastMCP in development mode
uv run python -m maverick_mcp.api.server --transport sse --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${YELLOW}Backend PID: $BACKEND_PID${NC}"

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"

# Wait up to 30 seconds for the backend to start
for i in {1..30}; do
    # Try both nc and curl for better compatibility
    if nc -z localhost 8000 2>/dev/null || curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}Backend is ready!${NC}"
        break
    fi
    
    # Check if backend process is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}Backend process died! Check backend.log for errors.${NC}"
        if [ -f backend.log ]; then
            echo -e "${RED}Last 20 lines of backend.log:${NC}"
            tail -20 backend.log
        fi
        exit 1
    fi
    
    if [ $i -eq 30 ]; then
        echo -e "${RED}Backend failed to start on port 8000 after 30 seconds!${NC}"
        if [ -f backend.log ]; then
            echo -e "${RED}Last 20 lines of backend.log:${NC}"
            tail -20 backend.log
        fi
        exit 1
    fi
    
    echo -e "${YELLOW}Still waiting... ($i/30)${NC}"
    sleep 1
done

echo -e "${GREEN}Backend started successfully on http://localhost:8000${NC}"

# Show information
echo -e "\n${GREEN}Development environment is running!${NC}"
echo -e "${YELLOW}MCP Server:${NC} http://localhost:8000"
echo -e "${YELLOW}Health Check:${NC} http://localhost:8000/health"
echo -e "${YELLOW}MCP SSE Endpoint:${NC} http://localhost:8000/sse"
echo -e "\nPress Ctrl+C to stop the server"

# Wait for process
wait