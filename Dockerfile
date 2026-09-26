# Dockerfile for Maverick-MCP
# Python-only MCP server
#
# Two stages on the same base image, so the venv's interpreter symlinks
# (/usr/local/bin/python3.12) resolve in both. The builder installs the
# locked dependencies with uv; the runtime stage copies only the finished
# venv, so it carries no uv and no build toolchain. Every locked package
# ships a Linux wheel, so neither stage needs a compiler.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

WORKDIR /app

# Install the dependencies first, without the project, so this layer stays
# cached until pyproject.toml or uv.lock changes. Ships the backtesting and
# research extras so the image has the full tool surface out of the box;
# drop --extra backtesting --extra research for a smaller, core-only image.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project \
    --extra backtesting --extra research

# Then install the project itself as a regular (non-editable) package.
COPY maverick ./maverick
RUN uv sync --frozen --no-dev --no-editable --extra backtesting --extra research


FROM python:3.12-slim

# MCP registry identity label (Docker MCP Catalog / GHCR discovery)
LABEL io.modelcontextprotocol.server.name="io.github.wshobson/maverick-mcp"

# Non-root user. /app is its working directory and must be writable: the
# default SQLite database (maverick.db) and cache (maverick_cache.db) are
# created there. The venv stays root-owned and read-only.
RUN groupadd -g 1000 maverick && \
    useradd -u 1000 -g maverick -s /bin/sh -m maverick && \
    install -d -o maverick -g maverick /app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER maverick

EXPOSE 8000

# No HEALTHCHECK: the new server exposes no HTTP /health endpoint (it is an
# MCP server, not a REST API). Container orchestrators should instead use
# process liveness or an MCP-aware probe.

# Start MCP server (streamable HTTP transport for container deployment)
CMD ["python", "-m", "maverick.server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
