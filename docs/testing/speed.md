# Research Speed Testing

Research speed tests validate timeout-aware behavior for research agents and
provider-backed tools.

## Goals

- Keep quick research paths responsive.
- Avoid repeated long timeout failures.
- Validate emergency/fast-mode model selection.
- Confirm parallel processing does not regress default behavior.

## Commands

```bash
make test-speed
make test-speed-quick
make test-speed-emergency
make test-speed-comparison
make benchmark-speed
```

Direct script usage:

```bash
uv run python scripts/speed_benchmark.py --mode quick
uv run python scripts/speed_benchmark.py --mode emergency
uv run python scripts/speed_benchmark.py --query "Apple Inc analysis"
```

## Thresholds

Use thresholds from the test code as the source of truth. Docs should describe
the intent, but code owns exact pass/fail values.

## Policy

- Do not run slow benchmark suites as part of the default unit command.
- Keep CI speed checks focused and short.
- Record provider/model configuration when comparing before/after timings.
