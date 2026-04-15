# Maverick-MCP Makefile
# Central command interface for agent-friendly development

.PHONY: help dev dev-sse dev-http dev-stdio stop test test-all test-watch test-specific test-parallel test-cov test-speed test-speed-quick test-speed-emergency test-speed-comparison test-strategies test-smoke lint format typecheck clean tail-log backend check migrate setup redis-start redis-stop experiment experiment-once benchmark-parallel benchmark-speed docker-up docker-down docker-logs

# Default target
help:
	@echo "Maverick-MCP Development Commands:"
	@echo ""
	@echo "  make dev          - Start development environment (Streamable-HTTP transport, default)"
	@echo "  make dev-http     - Start with Streamable-HTTP transport (same as dev)"
	@echo "  make dev-sse      - Start with SSE transport (debug/inspector use)"
	@echo "  make dev-stdio    - Start with STDIO transport (recommended for Claude Desktop)"
	@echo "  make backend      - Start backend MCP server (dev mode)"
	@echo "  make stop         - Stop all services"
	@echo ""
	@echo "  make test         - Run unit tests (fast)"
	@echo "  make test-all     - Run all tests including integration"
	@echo "  make test-watch   - Auto-run tests on file changes"
	@echo "  make test-specific TEST=name - Run specific test"
	@echo "  make test-parallel - Run tests in parallel"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make test-fixes   - Validate MCP tool fixes are working"
	@echo "  make test-speed   - Run speed optimization validation tests"
	@echo "  make test-speed-quick - Quick speed validation for CI"
	@echo "  make test-speed-emergency - Emergency mode speed tests"
	@echo "  make test-speed-comparison - Before/after performance comparison"
	@echo "  make test-strategies - Validate ALL backtesting strategies"
	@echo ""
	@echo "  make lint         - Run code quality checks"
	@echo "  make format       - Auto-format code"
	@echo "  make typecheck    - Run type checking"
	@echo "  make check        - Run all checks (lint + type check)"
	@echo ""
	@echo "  make tail-log     - Follow backend logs"
	@echo ""
	@echo "  make experiment   - Watch and auto-run .py files"
	@echo "  make benchmark-parallel - Test parallel screening"
	@echo "  make benchmark-speed - Run comprehensive speed benchmark"
	@echo "  make migrate      - Run database migrations"
	@echo "  make setup        - Initial project setup"
	@echo "  make clean        - Clean up generated files"
	@echo ""
	@echo "  make docker-up    - Start with Docker"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make docker-logs  - View Docker logs"

# Development commands
dev:
	@echo "Starting Maverick-MCP development environment (Streamable-HTTP transport)..."
	@./scripts/dev.sh

dev-http:
	@echo "Starting Maverick-MCP development environment (Streamable-HTTP transport)..."
	@./scripts/dev.sh

dev-sse:
	@echo "Starting Maverick-MCP development environment (SSE transport)..."
	@MAVERICK_TRANSPORT=sse ./scripts/dev.sh

dev-stdio:
	@echo "Starting Maverick-MCP development environment (STDIO transport)..."
	@MAVERICK_TRANSPORT=stdio ./scripts/dev.sh

backend:
	@echo "Starting backend in development mode..."
	@./scripts/start-backend.sh --dev

stop:
	@echo "Stopping all services..."
	@pkill -f "maverick_mcp.api.server" || true
	@echo "All services stopped."

# Testing commands
test:
	@echo "Running unit tests..."
	@uv run pytest -v

test-all:
	@echo "Running all tests (including integration)..."
	@uv run pytest -v -m ""

test-smoke:
	@echo "Running dev.sh readiness smoke test..."
	@./scripts/smoke_test_dev.sh

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
	@echo "Running tests with coverage (sysmon core; see pyproject.toml)..."
	# COVERAGE_CORE=sysmon uses Python 3.12+'s PEP 669 monitoring instead of
	# sys.settrace, which sidesteps a circular-import collision between
	# coverage.py's trace hook and beartype's import-claw. Without this the
	# run crashes inside beartype.claw._clawimpload trying to import
	# _clawstate mid-initialisation. sysmon is also 2-5x faster. Requires
	# coverage >= 7.4 (already in our lockfile).
	@COVERAGE_CORE=sysmon uv run pytest --cov=maverick_mcp --cov-report=html --cov-report=term

test-fixes:
	@echo "Running MCP tool fixes validation..."
	@uv run python maverick_mcp/tests/test_mcp_tool_fixes.py

test-fixes-verbose:
	@echo "Running MCP tool fixes validation (verbose)..."
	@uv run python -u maverick_mcp/tests/test_mcp_tool_fixes.py

# Speed optimization testing commands
test-speed:
	@echo "Running speed optimization validation tests..."
	@uv run pytest -v tests/test_speed_optimization_validation.py

test-speed-quick:
	@echo "Running quick speed validation for CI..."
	@uv run python scripts/speed_benchmark.py --mode quick

test-speed-emergency:
	@echo "Running emergency mode speed tests..."
	@uv run python scripts/speed_benchmark.py --mode emergency

test-speed-comparison:
	@echo "Running before/after performance comparison..."
	@uv run python scripts/speed_benchmark.py --mode comparison

test-strategies:
	@echo "Validating ALL backtesting strategies with real market data..."
	@uv run python scripts/test_all_strategies.py

# Code quality commands
lint:
	@echo "Running linter..."
	@uv run --extra dev ruff check .

format:
	@echo "Formatting code..."
	@uv run --extra dev ruff format .
	@uv run --extra dev ruff check . --fix

typecheck:
	@echo "Running type checker..."
	@uv run --extra dev pyright

check-mcp-types:
	@echo "Checking MCP tool list[str] parameters use coercion aliases..."
	@uv run python scripts/check_mcp_list_types.py

check-otel-versions:
	@echo "Checking OpenTelemetry package versions are aligned in uv.lock..."
	@uv run --no-sync python scripts/check_otel_versions.py

# Warning-only lint surfaces. Exit 0 today so they surface gaps without
# breaking merges; flip to --strict once the audit roadmap (Phase 2 for
# descriptions, Phase 3 for router consolidation) has closed the backlog.
check-mcp-descriptions:
	@echo "Checking MCP @mcp.tool decorators have useful description= ..."
	@uv run python scripts/check_mcp_descriptions.py

check-router-variants:
	@echo "Checking router-variant sprawl (_enhanced/_parallel/_ddd/_pipeline)..."
	@uv run python scripts/check_router_variants.py

check: lint typecheck check-mcp-types check-otel-versions check-mcp-descriptions check-router-variants
	@echo "All checks passed!"

# Utility commands
tail-log:
	@echo "Following backend logs (Ctrl+C to stop)..."
	@tail -f backend.log

experiment:
	@echo "Starting experiment harness..."
	@python tools/experiment.py

experiment-once:
	@echo "Running experiments once..."
	@python tools/experiment.py --once

migrate:
	@echo "Running database migrations..."
	@./scripts/run-migrations.sh upgrade

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
dh: dev-http
ds: dev-stdio
b: backend
t: test
l: lint
c: check

# Performance testing
benchmark-parallel:
	@echo "Benchmarking parallel screening performance..."
	@python -c "from tools.quick_test import test_parallel_screening; import asyncio; asyncio.run(test_parallel_screening())"

benchmark-speed:
	@echo "Running comprehensive speed benchmark..."
	@uv run python scripts/speed_benchmark.py --mode full


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
