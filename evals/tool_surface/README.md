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
make eval-traces ARGS="--cases evals/tool_surface/cases_batch2.json --budget-usd 2.50"
```

`--cases` picks the case file. `cases.json` (batch 1) runs in a fixed priority
order; any other file runs in file order, and its first case is the smoke query.

The budget cap is checked before each query starts. Each query also has a
$0.40 SDK budget, but the SDK checks it between turns, so one query can
overshoot it by the cost of its last turn (batch 2's b15 spent $0.77). The cap
check therefore reserves twice the per-query budget, or 1.5 times the smoke
query's cost if that is larger. That makes `--budget-usd` a planning cap, not
a hard ceiling. The hard ceiling is the Claude account itself: with usage
credits off, requests stop when the Agent SDK credit or balance runs out. Set
`--budget-usd` with margin below what the account has left.

Before each prompt is sent, the harness connects the CLI and checks the
session. No API key may be in use, the login must be a Claude subscription,
only the `maverick` server may be connected, and the model must see exactly
the Maverick tools (no built-ins and no `research_*` tools). If any check
fails, the run stops and no prompt is sent. After the prompt, the init message
is checked again, and it must report `apiKeySource` as `none`.

The run starts with one smoke query (the first case), which must also report
a cost. The other queries then run one at a time. A query does not start when
the spend so far plus its reservation (described above) would pass the cap. No
query is retried.

Each query gets a fresh copy of its seeded database in a temporary directory
outside the repository, and the server gets only `PATH`, `HOME`,
`DATABASE_URL`, and `CACHE_SQLITE_PATH`.

Agent SDK and `claude -p` usage draws from the account's Agent SDK credit or
usage credits, and it stops when they run out if reload is off.

## Output

A run writes `runs/<UTC timestamp>-<model>/traces/<id>.json`, one per query,
and `run.json` with the spend, the tools, and the cases run and skipped.

An `annotations.json` in a run folder holds only the reviewer's own words.
Agents never write verdicts or notes there.

## Reviewing

```bash
make eval-review                                  # newest run, port 8765
make eval-review EVAL_RUN=evals/tool_surface/runs/<run> ARGS="--port 8800"
```

This serves `evals/review/app.html` on 127.0.0.1 only. It calls no model or
API; the page loads marked.js and DOMPurify from cdnjs. For each trace the
reviewer sets a verdict (pass, fail, or defer), writes a trace note, and
selects text to attach span notes. The app saves `annotations.json` on every
change, writing through a temp file and a rename and keeping the previous
version as `annotations.json.bak`. Only the browser app writes this file.

An agent may later write `patterns.json` in the same folder: a DRAFT grouping
of the reviewer's notes into failure modes, which the app shows on its
Progress view for the reviewer to accept or edit. It names each mode and
points at notes by reference, never copying or rewording them:

```json
{"status": "draft", "created": "<ISO time>", "failure_modes": [
  {"name": "...", "description": "...",
   "notes": [{"trace_id": "q02", "span_id": "s-..."},
             {"trace_id": "q08", "span_id": null}]}]}
```

A `null` `span_id` points at that trace's trace note.

## In-session subagent traces (no Agent SDK spend)

Traces can also come from a Claude Code subagent spawned in an interactive
session, which runs on the session's subscription login and interactive plan
limits instead of the Agent SDK credit. The subagent is defined in
`agent/maverick-eval-client.md`:

- It sees only the 49 Maverick tools (no built-ins, no `research_*` tools).
- It runs `claude-opus-5-5` with `maxTurns: 8` and `omitClaudeMd: true`, and
  its system prompt is the same one line the SDK harness uses.
- Its own Maverick server (`agent_server.py`) starts when the subagent starts.
  The server reads the case from `.agent_case.json`, seeds a fresh database for
  the case's data state under `$TMPDIR`, keeps only PATH and HOME, and stops
  when the subagent finishes.

Workflow, driven from a Claude Code session in this repo:

```bash
make eval-agent-install   # once; restart Claude Code if .claude/agents is new
make eval-agent-case CASES=evals/tool_surface/cases_batch2.json CASE=b01
# spawn the maverick-eval-client subagent with the case query, verbatim
uv run python -m evals.tool_surface.agent_trace --transcript <agent-*.jsonl> \
    --cases evals/tool_surface/cases_batch2.json --case-id b01 --run <run dir>
```

Run one case at a time, because the pointer file names a single case. The
subagent's transcript is under
`~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`.

Differences from the SDK harness: Claude Code still attaches some session
context to a subagent (for example the date, an environment snapshot, and the
git status). Each trace lists those kinds in `injected_context`. There is no
per-query cost to record, and a rerun happens from a Claude Code session
rather than a `make` command. Use `make eval-traces` when a run must be
repeatable, such as checking whether a fix removed a failure mode.
