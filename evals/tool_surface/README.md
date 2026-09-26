# Tool-surface traces

This harness records how Claude uses the Maverick MCP tools, for error
analysis only. A person reads the traces and describes the failures in their
own words. There are no judges, no scores, and no CI gate.

## Dimensions

`cases.json` holds 20 fixed queries. Each case has three dimensions:

- task: lookup, technicals, screening, bookkeeping, risk, backtest,
  journal-watchlist, or multi-step.
- request type: specified, vague, ambiguous, advice, state-changing, or
  unsupported.
- data state: `empty` (schema only), `seeded` (a five-position portfolio, a
  fixture screening snapshot, one watchlist, and four journal trades), or
  `edge` (the seeded data with an unusual query). `seed.py` builds these
  offline.

## Running it

You need a `claude` CLI that is logged in to a Claude subscription. Never use
an API key: the run refuses to start when `ANTHROPIC_API_KEY` or
`ANTHROPIC_AUTH_TOKEN` is set, and the make target unsets both.

```bash
uv sync --group evals --extra dev --extra backtesting --extra research
make eval-traces                              # $6.00 cap by default
make eval-traces ARGS="--budget-usd 3 --model claude-opus-5-5"
```

Before each prompt is sent, the harness connects the CLI and checks the
session. No API key may be in use, the login must be a Claude subscription,
only the `maverick` server may be connected, and the model must see exactly
the Maverick tools (no built-ins and no `research_*` tools). If any check
fails, the run stops and no prompt is sent. After the prompt, the init message
is checked again, and it must report `apiKeySource` as `none`.

The run starts with one smoke query (q01), which must also report a cost. The
other queries then run one at a time. Each query has its own $0.40 limit. A
query does not start when the spend so far plus its worst case would pass the
cap. The worst case is $0.40, or 1.5 times the smoke cost if that is larger.
No query is retried.

Each query gets a fresh copy of its seeded database in a temporary directory
outside the repository, and the server gets only `PATH`, `HOME`,
`DATABASE_URL`, and `CACHE_SQLITE_PATH`.

Agent SDK and `claude -p` usage draws from the account's Agent SDK credit or
usage credits, and it stops when they run out if reload is off.

## Output

A run writes `runs/<UTC timestamp>-<model>/traces/<id>.json`, one per query,
and `run.json` with the spend, the tools, and the cases run and skipped.

An `annotations.csv` in a run folder holds only the reviewer's own words.
Agents never write verdicts or notes there.
