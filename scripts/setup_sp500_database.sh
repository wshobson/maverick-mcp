#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 MaverickMCP S&P 500 Database Setup${NC}"
echo "======================================"

# Check environment
echo -e "${YELLOW}📋 Environment Check:${NC}"

# Check for required environment variables
if [[ -z "${TIINGO_API_KEY}" ]]; then
    echo -e "   TIINGO_API_KEY: ${YELLOW}⚠️  Not set (optional for yfinance)${NC}"
else
    echo -e "   TIINGO_API_KEY: ${GREEN}✅ Set${NC}"
fi

# Show database URL
DATABASE_URL=${DATABASE_URL:-"sqlite:///maverick.db"}
echo "   DATABASE_URL: $DATABASE_URL"

# Clear existing database for fresh S&P 500 start
if [[ "$DATABASE_URL" == "sqlite:///"* ]]; then
    DB_FILE=$(echo $DATABASE_URL | sed 's/sqlite:\/\/\///g')
    if [[ -f "$DB_FILE" ]]; then
        echo -e "${YELLOW}🗑️  Removing existing database for fresh S&P 500 setup...${NC}"
        rm "$DB_FILE"
    fi
fi

# Run database migration
echo -e "${BLUE}1️⃣ Running database migration...${NC}"
echo "--------------------------------"
if uv run python scripts/migrate_db.py; then
    echo -e "${GREEN}✅ Migration completed successfully${NC}"
else
    echo -e "${RED}❌ Migration failed${NC}"
    exit 1
fi

# Run S&P 500 seeding
echo -e "${BLUE}2️⃣ Running S&P 500 database seeding...${NC}"
echo "-------------------------------------"
if uv run python scripts/seed_sp500.py; then
    echo -e "${GREEN}✅ S&P 500 seeding completed successfully${NC}"
else
    echo -e "${RED}❌ S&P 500 seeding failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 S&P 500 database setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run the MCP server: ${YELLOW}make dev${NC}"
echo "2. Connect with Claude Desktop using mcp-remote"
echo "3. Test with: ${YELLOW}'Show me top S&P 500 momentum stocks'${NC}"
echo ""
echo -e "${BLUE}Available S&P 500 screening tools:${NC}"
echo "- get_maverick_recommendations (bullish momentum stocks)"
echo "- get_maverick_bear_recommendations (bearish setups)"
echo "- get_trending_breakout_recommendations (supply/demand breakouts)"