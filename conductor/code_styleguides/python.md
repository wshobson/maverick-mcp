# Python Style Guide

This style guide is based on the existing project configuration in `pyproject.toml`.

## Tooling

| Tool     | Purpose                | Config Location            |
| -------- | ---------------------- | -------------------------- |
| ruff     | Linting & formatting   | `pyproject.toml`           |
| pyright  | Type checking          | `pyrightconfig.json`       |
| pytest   | Testing                | `pyproject.toml`           |
| uv       | Dependency management  | `pyproject.toml` / `uv.lock` |

## Formatting Rules

### Line Length

- **Maximum: 88 characters**
- Configured in `[tool.ruff]`

### Target Python Version

- **Python 3.12**
- Configured in `[tool.ruff]` and `[project.requires-python]`

## Imports

Ruff enforces import sorting (isort-compatible). Prefer grouping:

```python
# Standard library
from datetime import datetime

# Third-party
import pandas as pd

# Local
from maverick_mcp.core import indicators
```

## Type Hints

- Use modern type syntax (`list[str]`, `str | None`)
- Prefer clear, explicit return types for public functions

```python
from collections.abc import Sequence


def top_movers(tickers: Sequence[str], lookback_days: int = 5) -> list[str]:
    ...
```

## Error Handling

- Raise actionable errors (what’s missing and how to fix it)
- Avoid silent fallbacks when correctness depends on configuration (API keys, DB URL)

## Running Checks

```bash
make test
make lint
make typecheck
make check
```

