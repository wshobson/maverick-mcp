# Product Guidelines

## Voice and Tone

**Direct and analytical**

- Prefer concrete numbers and clear assumptions
- Avoid hype or “guaranteed returns” language
- Use consistent terminology (timeframes, units, tickers)
- When data is missing, say so and explain what is required

## Design Principles

### 1. Correctness and Transparency

Incorrect data is worse than no data.

- Prefer explicit sources, timestamps, and units
- Make calculations reproducible (show parameters and inputs)
- Fail loudly on invalid inputs; avoid silent fallbacks

### 2. Speed via Caching (Without Surprises)

- Cache expensive calls with sensible TTLs
- Make cache behavior observable (hit/miss where helpful)
- Ensure “freshness” expectations are clear in tool outputs

### 3. Deterministic, Composable Tools

- Tools should do one thing well and compose into workflows
- Keep outputs structured (tables, JSON-like sections, consistent keys)
- Avoid side effects unless explicitly requested (e.g., portfolio writes)

### 4. Personal-Use Simplicity

- Avoid authentication and multi-tenant complexity
- Keep configuration environment-variable driven
- Prefer SQLite by default; allow Postgres/Redis optionally

## Code Quality Standards

- Match existing project patterns and module boundaries
- Add tests for behavior changes and non-trivial logic
- Keep error handling explicit; include actionable error messages
- Use type hints where they materially improve readability

## User Experience Guidelines

- Outputs should be scannable (headings, bullet points, tables)
- Prefer “what to do next” suggestions on errors (missing API keys, etc.)
- Avoid unnecessary verbosity; offer drill-down when requested

