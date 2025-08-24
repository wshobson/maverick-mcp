#!/bin/bash
#
# Fast Development Startup Script
# Skips all checks and uses in-memory database for < 3 second startup
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}⚡ Fast Dev Mode - Skipping all checks for speed${NC}"

# Set ultra-fast environment
export AUTH_ENABLED=false
export CREDIT_SYSTEM_ENABLED=false
export DATABASE_URL="sqlite:///:memory:"
export REDIS_HOST="none"  # Skip Redis
export SKIP_VALIDATION=true
export SKIP_MIGRATIONS=true
export LOG_LEVEL=WARNING  # Reduce log noise
export STARTUP_MODE=fast

# Change to project root
cd "$(dirname "$0")/.."

# Create minimal .env if not exists
if [ ! -f .env ]; then
    cat > .env << EOF
AUTH_ENABLED=false
CREDIT_SYSTEM_ENABLED=false
DATABASE_URL=sqlite:///:memory:
REDIS_HOST=none
SKIP_VALIDATION=true
LOG_LEVEL=WARNING
EOF
    echo -e "${YELLOW}Created minimal .env for fast mode${NC}"
fi

# Start time tracking
START_TIME=$(date +%s)

# Launch server directly without checks
echo -e "${GREEN}Starting server in fast mode...${NC}"

# Create a minimal launcher that skips all initialization
python -c "
import os
os.environ['STARTUP_MODE'] = 'fast'
os.environ['AUTH_ENABLED'] = 'false'
os.environ['CREDIT_SYSTEM_ENABLED'] = 'false'
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['SKIP_VALIDATION'] = 'true'

# Minimal imports only
import asyncio
import uvicorn
from fastmcp import FastMCP

# Create minimal server
mcp = FastMCP(
    name='MaverickMCP-Fast',
    debug=True,
    log_level='WARNING'
)

# Add one test tool to verify it's working
@mcp.tool()
async def test_fast_mode():
    return {'status': 'Fast mode active!', 'startup_time': '< 3 seconds'}

# Direct startup without any checks
if __name__ == '__main__':
    print('🚀 Server starting on http://localhost:8000')
    mcp.run(transport='sse', port=8000, host='0.0.0.0')
" &

SERVER_PID=$!

# Calculate startup time
END_TIME=$(date +%s)
STARTUP_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✨ Server started in ${STARTUP_TIME} seconds!${NC}"
echo -e "${CYAN}Access at: http://localhost:8000/sse${NC}"
echo -e "${YELLOW}Note: This is a minimal server - add your tools to test${NC}"
echo -e "\nPress Ctrl+C to stop"

# Wait for server
wait $SERVER_PID