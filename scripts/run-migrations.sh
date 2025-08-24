#!/bin/bash

# MaverickMCP Database Migration Script
# This script manages database migrations separately from server startup

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Change to project root
cd "$(dirname "$0")/.."

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

# Function to display usage
usage() {
    echo -e "${BLUE}MaverickMCP Database Migration Tool${NC}"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  status      Show current migration status"
    echo "  upgrade     Apply all pending migrations"
    echo "  downgrade   Downgrade to a specific revision"
    echo "  history     Show migration history"
    echo "  validate    Validate migration files"
    echo "  backup      Create database backup before migration"
    echo ""
    echo "Options:"
    echo "  -r, --revision <rev>   Target revision for downgrade"
    echo "  -n, --dry-run         Show what would be done without applying"
    echo "  -f, --force           Skip confirmation prompts"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status                    # Check migration status"
    echo "  $0 upgrade                   # Apply all migrations"
    echo "  $0 upgrade --dry-run         # Preview migrations"
    echo "  $0 downgrade -r 001          # Downgrade to revision 001"
    echo "  $0 backup                    # Create backup"
}

# Function to check database connection
check_database() {
    echo -e "${YELLOW}Checking database connection...${NC}"
    
    if [ -z "$DATABASE_URL" ]; then
        echo -e "${RED}Error: DATABASE_URL not set${NC}"
        exit 1
    fi
    
    # Extract database name from URL
    DB_NAME=$(echo $DATABASE_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')
    
    # Test connection with Python
    uv run python -c "
import sys
from sqlalchemy import create_engine, text
try:
    engine = create_engine('$DATABASE_URL')
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" || exit 1
    
    echo -e "${GREEN}✓ Connected to database: $DB_NAME${NC}"
}

# Function to validate alembic configuration
validate_alembic() {
    if [ ! -f alembic.ini ]; then
        echo -e "${RED}Error: alembic.ini not found${NC}"
        exit 1
    fi
    
    if [ ! -d alembic/versions ]; then
        echo -e "${RED}Error: alembic/versions directory not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Alembic configuration validated${NC}"
}

# Function to show migration status
show_status() {
    echo -e "${BLUE}Current Migration Status${NC}"
    echo "========================"
    
    # Show current revision
    echo -e "\n${YELLOW}Current database revision:${NC}"
    alembic current 2>/dev/null || echo "No migrations applied"
    
    # Show pending migrations
    echo -e "\n${YELLOW}Pending migrations:${NC}"
    alembic heads 2>/dev/null || echo "No pending migrations"
    
    # Count migration files
    MIGRATION_COUNT=$(find alembic/versions -name "*.py" | grep -v __pycache__ | wc -l)
    echo -e "\n${YELLOW}Total migration files:${NC} $MIGRATION_COUNT"
}

# Function to show migration history
show_history() {
    echo -e "${BLUE}Migration History${NC}"
    echo "================="
    alembic history --verbose
}

# Function to validate migrations
validate_migrations() {
    echo -e "${BLUE}Validating Migrations${NC}"
    echo "===================="
    
    # Check for duplicate revisions
    echo -e "${YELLOW}Checking for duplicate revisions...${NC}"
    DUPLICATES=$(find alembic/versions -name "*.py" -exec grep -H "^revision = " {} \; | 
                 grep -v __pycache__ | 
                 awk -F: '{print $2}' | 
                 sort | uniq -d)
    
    if [ -n "$DUPLICATES" ]; then
        echo -e "${RED}Error: Duplicate revisions found:${NC}"
        echo "$DUPLICATES"
        exit 1
    else
        echo -e "${GREEN}✓ No duplicate revisions${NC}"
    fi
    
    # Check for broken dependencies
    echo -e "${YELLOW}Checking migration dependencies...${NC}"
    uv run python -c "
from alembic.config import Config
from alembic.script import ScriptDirectory
config = Config('alembic.ini')
script_dir = ScriptDirectory.from_config(config)
try:
    script_dir.walk_revisions()
    print('✓ All migration dependencies valid')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" || exit 1
}

# Function to create database backup
create_backup() {
    echo -e "${BLUE}Creating Database Backup${NC}"
    echo "======================"
    
    # Extract connection details from DATABASE_URL
    DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_NAME=$(echo $DATABASE_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')
    
    BACKUP_FILE="backups/db_backup_$(date +%Y%m%d_%H%M%S).sql"
    mkdir -p backups
    
    echo -e "${YELLOW}Creating backup: $BACKUP_FILE${NC}"
    
    # Use pg_dump if PostgreSQL
    if [[ $DATABASE_URL == *"postgresql"* ]]; then
        pg_dump $DATABASE_URL > $BACKUP_FILE
    else
        echo -e "${RED}Backup not implemented for this database type${NC}"
        exit 1
    fi
    
    if [ -f $BACKUP_FILE ]; then
        SIZE=$(du -h $BACKUP_FILE | cut -f1)
        echo -e "${GREEN}✓ Backup created: $BACKUP_FILE ($SIZE)${NC}"
    else
        echo -e "${RED}Error: Backup failed${NC}"
        exit 1
    fi
}

# Function to apply migrations
apply_migrations() {
    local DRY_RUN=$1
    local FORCE=$2
    
    echo -e "${BLUE}Applying Migrations${NC}"
    echo "=================="
    
    # Show pending migrations
    echo -e "${YELLOW}Checking for pending migrations...${NC}"
    PENDING=$(alembic upgrade head --sql 2>/dev/null | grep -c "UPDATE alembic_version" || echo "0")
    
    if [ "$PENDING" -eq "0" ]; then
        echo -e "${GREEN}✓ Database is up to date${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Found pending migrations${NC}"
    
    # Dry run mode
    if [ "$DRY_RUN" == "true" ]; then
        echo -e "\n${YELLOW}SQL to be executed:${NC}"
        alembic upgrade head --sql
        return 0
    fi
    
    # Confirmation prompt
    if [ "$FORCE" != "true" ]; then
        echo -e "\n${YELLOW}Do you want to apply these migrations? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo -e "${RED}Migration cancelled${NC}"
            exit 0
        fi
    fi
    
    # Apply migrations
    echo -e "\n${YELLOW}Applying migrations...${NC}"
    alembic upgrade head
    
    echo -e "${GREEN}✓ Migrations applied successfully${NC}"
    
    # Show new status
    echo -e "\n${YELLOW}New database revision:${NC}"
    alembic current
}

# Function to downgrade
downgrade_migration() {
    local REVISION=$1
    local DRY_RUN=$2
    local FORCE=$3
    
    echo -e "${BLUE}Downgrading Migration${NC}"
    echo "==================="
    
    if [ -z "$REVISION" ]; then
        echo -e "${RED}Error: Revision required for downgrade${NC}"
        usage
        exit 1
    fi
    
    # Show current revision
    echo -e "${YELLOW}Current revision:${NC}"
    alembic current
    
    # Dry run mode
    if [ "$DRY_RUN" == "true" ]; then
        echo -e "\n${YELLOW}SQL to be executed:${NC}"
        alembic downgrade $REVISION --sql
        return 0
    fi
    
    # Confirmation prompt
    if [ "$FORCE" != "true" ]; then
        echo -e "\n${RED}WARNING: This will downgrade the database to revision $REVISION${NC}"
        echo -e "${YELLOW}Do you want to continue? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo -e "${RED}Downgrade cancelled${NC}"
            exit 0
        fi
    fi
    
    # Create backup first
    create_backup
    
    # Apply downgrade
    echo -e "\n${YELLOW}Downgrading to revision $REVISION...${NC}"
    alembic downgrade $REVISION
    
    echo -e "${GREEN}✓ Downgrade completed successfully${NC}"
    
    # Show new status
    echo -e "\n${YELLOW}New database revision:${NC}"
    alembic current
}

# Parse command line arguments
COMMAND=""
REVISION=""
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        status|upgrade|downgrade|history|validate|backup)
            COMMAND=$1
            shift
            ;;
        -r|--revision)
            REVISION="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
check_database
validate_alembic

# Execute command
case $COMMAND in
    status)
        show_status
        ;;
    upgrade)
        apply_migrations $DRY_RUN $FORCE
        ;;
    downgrade)
        downgrade_migration $REVISION $DRY_RUN $FORCE
        ;;
    history)
        show_history
        ;;
    validate)
        validate_migrations
        ;;
    backup)
        create_backup
        ;;
    *)
        echo -e "${RED}Error: Command required${NC}"
        usage
        exit 1
        ;;
esac

echo -e "\n${GREEN}Migration tool completed successfully${NC}"