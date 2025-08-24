#!/bin/bash

# Frontend has been removed from Maverick-MCP
# This system is now a personal-use stock analysis MCP server only

set -e

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}NOTICE: Frontend Removed${NC}"
echo -e "${YELLOW}The Maverick-MCP frontend has been permanently removed.${NC}"
echo -e "${YELLOW}This system is now a personal-use stock analysis MCP server only.${NC}"
echo ""
echo -e "To start the backend MCP server, use:"
echo -e "${YELLOW}  ./scripts/dev.sh${NC}"
echo -e "${YELLOW}  # or${NC}"
echo -e "${YELLOW}  ./scripts/start-backend.sh${NC}"
echo ""
echo -e "MCP Server will be available at: http://localhost:8000"
echo -e "MCP SSE endpoint: http://localhost:8000/sse"
echo ""
exit 1