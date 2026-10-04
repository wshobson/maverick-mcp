# Tool-surface traces

This harness records how Claude uses the Maverick MCP tools, for error
analysis only. A person reads the traces and describes the failures in their
own words. There are no judges, no scores, and no CI gate.

## Dimensions

`cases.json` (batch 1), `cases_batch2.json` (batch 2), and `cases_batch3.json`
(batch 3) each hold 20 fixed queries. Each case has three dimensions, and
batch 3 adds a fourth, `gap`: what the request leaves out or gets wrong
(`none` for its controls). The three dimensions are:

- task: lookup, technicals, screening, bookkeeping, risk, backtest,
  journal-watchlist, or multi-step.
- request type: specified, vague, ambiguous, advice, state-changing, or
  unsupported.
- data state: `empty` (schema only), `seeded` (a five-position portfolio,
  nine symbols registered for the screener with no prices or screening
  results, one watchlist, and four synthetic journal trades), or `edge` (the seeded
  data with an unusual query). `seed.py` builds these offline. A trace that
  runs the screens fetches live prices for those nine symbols.

## Fixture provenance and correctness reruns

The current seed is `synthetic-v2-2026-10-04`. Every position and journal price
and date is a synthetic test input, not a historical market observation or a
real transaction. The basis is stored with each position's notes and each
journal entry's rationale, and in the watchlist description and item notes.
Comparisons with live quotes describe hypothetical gains or losses.

The NVDA position uses 120 shares at 180.005, for a Decimal total cost of
21,600.60. The closed NVDA journal example buys 50 at 100.005 and closes at
101.015, for 50.50 profit after aggregation. These values test sub-cent unit
prices; they make no claim about NVDA's price on the fixture dates. All other
seed values share the same synthetic basis in `seed.py`.

Offline tests build all three states twice and compare business records. Domain
APIs generate UUIDs and audit timestamps, so those identifiers/timestamps are
excluded from equivalence checks. Purchase, entry, and exit dates remain part
of the comparison. No market provider or model runs during seed construction.

`cases_correctness.json` prepares nine affected cases: q14/q15, b01/b05/b12/b15/
b18, and c15/c16. Each retains its original case file and ID, records the new
fixture version, and explicitly tells the client that returns are hypothetical.
This subset covers sizing, backtest output/metrics, cost basis, observed levels,
correlation, journal input, watchlist discovery, and parallel history requests.
The original case files and historical traces remain unchanged.

These reruns have not been executed. A future authorized run must record its
model, input case file, fixture version, server commit, and live-provider timing.
Any generated verdicts, including batch-3 labels, still require the owner's
review in the UI. This correction adds no model scores or invented human labels.

## Running it

Traces come from a Claude Code subagent spawned in an interactive session in
this repo. The subagent uses the session's own login. On a Claude
subscription login it runs within the plan limits and costs nothing beyond
the session. Nothing in this workflow can check which login the session
uses, so confirm it with `/status` before a run: a session on an API key
would bill that key. Each trace records `apiKeySource` as `unverified`. The
subagent is defined in `agent/maverick-eval-client.md`:

- It sees only the 50 Maverick tools (no built-ins, no `research_*` tools).
- It runs `claude-opus-5-5` with `maxTurns: 8` and `omitClaudeMd: true`, and
  its system prompt is one line.
- Its own Maverick server (`agent_server.py`) starts when the subagent starts.
  The server reads the case from `.agent_case.json`, seeds a fresh database for
  the case's data state under `$TMPDIR`, keeps only PATH and HOME, and stops
  when the subagent finishes.

Workflow, driven from a Claude Code session in this repo:

```bash
uv sync --extra dev --extra backtesting --extra research  # the server registers only installed extras
make eval-agent-install   # then restart Claude Code; a running session kept the old agent
make eval-agent-case CASES=evals/tool_surface/cases_batch2.json CASE=b01
# spawn the maverick-eval-client subagent with the case query, verbatim
uv run python -m evals.tool_surface.agent_trace --transcript <agent-*.jsonl> \
    --cases evals/tool_surface/cases_batch2.json --case-id b01 \
    --run evals/tool_surface/runs/<UTC timestamp>-claude-opus-5-5
```

Run one case at a time, because the pointer file names a single case. The
subagent's transcript is under
`~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`. The
converter refuses a transcript whose first request is not the case's query,
and it exits non-zero when the subagent called a non-Maverick or `research_*`
tool.

Claude Code still attaches some session context to a subagent (for example
the date, an environment snapshot, and the git status). Each trace lists those
kinds in `injected_context`. There is no per-query cost to record. Token usage
comes from the transcript, which records a response's usage as it starts, so
output token counts run low.

## Failure modes and judges

`failure_modes.md` groups the reviewer's notes into failure modes and says
where each fix belongs. A mode gets a model judge only when the failure needs
judgment; defects in Maverick or the fixtures are fixed instead and tracked
in `docs/exec-plans/tech-debt-tracker.md`.

A judge runs in-session, like the traces. Render its inputs, have subagents
judge the input files against the judge prompt and write
`{"<id>": {"critique": ..., "result": "Pass" | "Fail"}}`, then score
the verdicts against the reviewer's labels:

```bash
uv run python -m evals.tool_surface.judge inputs --mode acts-on-a-guess --out <dir>
uv run python -m evals.tool_surface.judge score --mode acts-on-a-guess --judgments <file>
```

A trace's label for a mode is Fail when `patterns.json` lists it under that
mode, and Pass otherwise. The prompt's few-shot traces are left out of
scoring. Results are kept in `judges/results/`.

`jev.py` asks Jev (TypeSafe) the same question about the rendered inputs
and writes judgments that `score` reads. It is a paid API that uses
`TYPESAFE_API_KEY` from `.env`, so run it only with the owner's approval.

## Output

Each converted case writes `runs/<run>/traces/<id>.json`. Name a run folder
`<UTC timestamp>-<model>` (for example `20261001T090000Z-claude-opus-5-5`)
so `make eval-review` picks the newest one. The two runs from 2026-09-26 were
recorded by an Agent SDK harness that has since been removed. Their traces
have the same shape, and each `run.json` records that run's spend, tools,
and cases.

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
The server reads the run's traces when it starts, so restart it to see
traces converted after that.

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
