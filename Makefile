# Maverick-MCP Makefile
# Central command interface for agent-friendly development

.PHONY: help dev dev-stdio stop test test-all test-watch test-specific test-parallel test-cov lint format typecheck docs-check clean tail-log check setup redis-start redis-stop docker-up docker-down docker-logs

# Default target
help:
	@echo "Maverick-MCP Development Commands:"
	@echo ""
	@echo "  make dev          - Start development environment (Streamable-HTTP transport, default)"
	@echo "  make dev-stdio    - Start with STDIO transport (recommended for Claude Desktop)"
	@echo "  make stop         - Stop all services"
	@echo ""
	@echo "  make test         - Run unit tests (fast)"
	@echo "  make test-all     - Run all tests including integration"
	@echo "  make test-watch   - Auto-run tests on file changes"
	@echo "  make test-specific TEST=name - Run specific test"
	@echo "  make test-parallel - Run tests in parallel"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "  make lint         - Run code quality checks"
	@echo "  make format       - Auto-format code"
	@echo "  make typecheck    - Run type checking"
	@echo "  make docs-check   - Validate documentation catalog and links"
	@echo "  make check        - Run all checks (lint + type check)"
	@echo ""
	@echo "  make tail-log     - Follow backend logs"
	@echo ""
	@echo "  make clean        - Clean up generated files"
	@echo ""
	@echo "  make docker-up    - Start with Docker"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make docker-logs  - View Docker logs"

# Development commands
dev:
	@echo "Starting Maverick-MCP development environment (Streamable-HTTP transport)..."
	@uv run python -m maverick.server --transport http

dev-stdio:
	@echo "Starting Maverick-MCP development environment (STDIO transport)..."
	@uv run python -m maverick.server --transport stdio

stop:
	@echo "Stopping all services..."
	@pkill -f "maverick.server" || true
	@echo "All services stopped."

# Testing commands
test:
	@echo "Running unit tests..."
	@uv run pytest -v

test-all:
	@echo "Running all tests (including integration)..."
	@uv run pytest -v -m ""

test-watch:
	@echo "Starting test watcher..."
	@if ! uv pip show pytest-watch > /dev/null 2>&1; then \
		echo "Installing pytest-watch..."; \
		uv pip install pytest-watch; \
	fi
	@uv run ptw -- -v

test-specific:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-specific TEST=test_name"; \
		exit 1; \
	fi
	@echo "Running specific test: $(TEST)"
	@uv run pytest -v -k "$(TEST)"

test-parallel:
	@echo "Running tests in parallel..."
	@if ! uv pip show pytest-xdist > /dev/null 2>&1; then \
		echo "Installing pytest-xdist..."; \
		uv pip install pytest-xdist; \
	fi
	@uv run pytest -v -n auto

test-cov:
	@echo "Running tests with coverage..."
	@uv run pytest --cov=maverick --cov-report=html --cov-report=term

# Code quality commands
lint:
	@echo "Running linter..."
	@uv run --extra dev ruff check .
	@uv run lint-imports

format:
	@echo "Formatting code..."
	@uv run --extra dev ruff format .
	@uv run --extra dev ruff check . --fix

typecheck:
	@echo "Running type checker..."
	@uv run --extra dev pyright

docs-check:
	@echo "Checking documentation catalog..."
	@uv run python tools/check_docs_catalog.py

check: lint typecheck
	@echo "All checks passed!"

# Utility commands
tail-log:
	@echo "Following backend logs (Ctrl+C to stop)..."
	@tail -f backend.log

setup:
	@echo "Setting up Maverick-MCP..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please update with your configuration"; \
	fi
	@uv sync
	@echo "Setup complete! Run 'make dev' to start development."

clean:
	@echo "Cleaning up..."
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -rf .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete."

# Service management
redis-start:
	@echo "Starting Redis..."
	@if command -v brew &> /dev/null; then \
		brew services start redis; \
	else \
		redis-server --daemonize yes; \
	fi

redis-stop:
	@echo "Stopping Redis..."
	@if command -v brew &> /dev/null; then \
		brew services stop redis; \
	else \
		pkill redis-server || true; \
	fi

# Quick shortcuts
d: dev
ds: dev-stdio
t: test
l: lint
c: check

# Docker commands
docker-up:
	@echo "Starting Docker services..."
	@docker-compose up --build -d

docker-down:
	@echo "Stopping Docker services..."
	@docker-compose down

docker-logs:
	@echo "Following Docker logs (Ctrl+C to stop)..."
	@docker-compose logs -f
