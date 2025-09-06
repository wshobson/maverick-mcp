# Dockerfile for Maverick-MCP
# Python-only MCP server

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and TA-Lib
RUN apt-get update && apt-get install -yqq \
  build-essential \
  python3-dev \
  libpq-dev \
  wget \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Install and compile TA-Lib
ENV TALIB_DIR=/usr/local
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
  && tar -xzf ta-lib-0.6.4-src.tar.gz \
  && cd ta-lib-0.6.4/ \
  && ./configure --prefix=$TALIB_DIR \
  && make -j$(nproc) \
  && make install \
  && cd .. \
  && rm -rf ta-lib-0.6.4-src.tar.gz ta-lib-0.6.4/

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies
RUN uv sync --frozen

# Copy application code
COPY maverick_mcp ./maverick_mcp
COPY alembic ./alembic
COPY alembic.ini setup.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd -g 1000 maverick && \
    useradd -u 1000 -g maverick -s /bin/sh -m maverick && \
    chown -R maverick:maverick /app

USER maverick

EXPOSE 8000

# Start MCP server
CMD ["uv", "run", "python", "-m", "maverick_mcp.api.server", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
