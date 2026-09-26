"""Run the tool-surface cases through Claude on the Maverick MCP server and
write one trace per case. See README.md.

Needs the `evals` dependency group and a `claude` CLI logged in to a Claude
subscription. Never an API key: this refuses to start when ANTHROPIC_API_KEY
or ANTHROPIC_AUTH_TOKEN is set, and it never calls load_dotenv.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from evals.tool_surface import harness, seed
from evals.tool_surface.harness import HarnessAbort

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
SERVER_ARGS = ["-m", "maverick.server", "--transport", "stdio"]
SYSTEM_PROMPT = (
    "You are connected to the Maverick MCP server, which provides stock analysis tools."
)
DEFAULT_MODEL = "claude-opus-5-5"
PER_QUERY_BUDGET_USD = 0.40
MAX_TURNS = 8
SERVER_WAIT_SECONDS = 90
QUERY_TIMEOUT_SECONDS = 900
# q01 is the smoke query; the rest run in this order until the budget stops.
RUN_ORDER = (1, 7, 11, 19, 10, 13, 16, 17, 2, 3, 6, 12, 15, 18, 9, 14, 8, 4, 5, 20)
# Ends that are part of the trace. Any other error ends the run.
TRACE_STOPS = ("error_max_turns", "error_max_budget_usd")


@dataclass
class Run:
    model: str
    cap: float
    scratch: Path
    seeds: dict[str, Path]
    cli_env: dict[str, str]
    allowed: list[str]
    excluded: list[str]
    sha: str
    traces: Path
    spent: float = 0.0
    smoke_cost: float | None = None
    init: dict[str, Any] = field(default_factory=dict)


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    return out.stdout.strip()


def _refuse_dotenv_above(path: Path) -> None:
    """The server's find_dotenv walks up from its cwd; nothing may be there."""
    for directory in (path, *path.parents):
        if (directory / ".env").exists():
            raise HarnessAbort(f"{directory / '.env'} would load into the server")


async def _server_tools(scratch: Path, seeds: dict[str, Path]) -> list[str]:
    """List the tools the server advertises, started exactly as for a case."""
    probe = scratch / "probe"
    probe.mkdir()
    database = shutil.copyfile(seeds["empty"], probe / "maverick.db")
    env = harness.server_env(os.environ, database, probe / "maverick_cache.db")
    transport = StdioTransport(
        str(PYTHON), SERVER_ARGS, env=env, cwd=str(probe), log_file=probe / "log"
    )
    async with Client(transport, timeout=SERVER_WAIT_SECONDS) as client:
        return [tool.name for tool in await client.list_tools()]


def _options(
    run: Run, workdir: Path, server_env: dict[str, str], stderr: list[str]
) -> ClaudeAgentOptions:
    server = {"type": "stdio", "command": str(PYTHON), "args": SERVER_ARGS}
    return ClaudeAgentOptions(
        model=run.model,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={harness.SERVER_NAME: {**server, "env": server_env}},
        strict_mcp_config=True,
        tools=[],
        allowed_tools=run.allowed,
        disallowed_tools=[*run.excluded, *harness.EXTRA_DISALLOWED],
        permission_mode="dontAsk",
        setting_sources=[],
        skills=[],
        max_turns=MAX_TURNS,
        max_budget_usd=PER_QUERY_BUDGET_USD,
        cwd=workdir,
        env=run.cli_env,
        stderr=stderr.append,
    )


async def run_case(run: Run, case: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Run one query once. Returns its trace and a reason to stop, if any."""
    workdir = run.scratch / case["id"]
    workdir.mkdir()
    database = shutil.copyfile(run.seeds[case["data_state"]], workdir / "maverick.db")
    server_env = harness.server_env(os.environ, database, workdir / "maverick_cache.db")
    if banned := harness.banned_env_keys(run.cli_env, server_env):
        raise HarnessAbort(f"banned variables in the CLI or server env: {banned}")
    stderr: list[str] = []
    builder = harness.TraceBuilder()
    abort: str | None = None
    started, clock = datetime.now(UTC), time.monotonic()
    try:
        async with (
            asyncio.timeout(QUERY_TIMEOUT_SECONDS),
            ClaudeSDKClient(_options(run, workdir, server_env, stderr)) as client,
        ):
            # Checks auth, servers, and tools before the prompt goes out.
            abort = await harness.converse(
                client, case["query"], run.allowed, builder, SERVER_WAIT_SECONDS
            )
    except Exception as exc:  # recorded in the trace and run.json, never retried
        abort = f"{type(exc).__name__}: {exc}"
    result = builder.result
    if abort is None and (result is None or result.total_cost_usd is None):
        abort = "no cost reported, so the run cannot track spend"
    elif abort is None and result.is_error and result.subtype not in TRACE_STOPS:
        abort = f"query ended in an error: {result.subtype} {result.errors or ''}"
    trace = builder.trace(
        case,
        model=run.model,
        system_prompt=SYSTEM_PROMPT,
        wall_seconds=round(time.monotonic() - clock, 2),
        maverick_git_sha=run.sha,
        timestamp=started.isoformat(),
        cli_stderr=stderr,
    )
    if builder.init is not None and not run.init:
        run.init = builder.init
    return trace, abort


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


async def run_all(run: Run, cases: list[dict[str, Any]]) -> dict[str, Any]:
    ran: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    aborted: str | None = None
    for index, case in enumerate(cases):
        stop = aborted or harness.budget_stop_reason(
            run.spent, run.smoke_cost, run.cap, PER_QUERY_BUDGET_USD
        )
        if stop is not None:
            skipped += [{"id": c["id"], "reason": stop} for c in cases[index:]]
            break
        print(f"{case['id']}: {case['query']}", flush=True)
        trace, aborted = await run_case(run, case)
        _write_json(run.traces / f"{case['id']}.json", trace)
        cost = trace["notional_cost_usd"] or 0.0
        run.spent += cost
        if run.smoke_cost is None and aborted is None:
            run.smoke_cost = cost
        keys = ("notional_cost_usd", "num_turns", "result_subtype")
        ran.append({"id": case["id"], **{k: trace[k] for k in keys}})
        print(f"  ${cost:.4f}, total ${run.spent:.4f}, {aborted or 'ok'}", flush=True)
    return {"cases_run": ran, "cases_skipped": skipped, "aborted": aborted}


DEFAULT_CASES = HERE / "cases.json"


def _ordered_cases(path: Path) -> list[dict[str, Any]]:
    """Batch 1 (`cases.json`) runs in RUN_ORDER; any other file in file order."""
    cases = json.loads(path.read_text())
    is_default = path.resolve() == DEFAULT_CASES.resolve()
    order = [f"q{number:02d}" for number in RUN_ORDER] if is_default else None
    return harness.ordered_cases(cases, order)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget-usd", type=float, default=6.00)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="case file; any file but cases.json runs in file order",
    )
    args = parser.parse_args(argv)
    try:
        cli_env = harness.cli_env(os.environ)
    except HarnessAbort as exc:
        print(f"refusing to start: {exc}")
        return 2
    # The SDK merges its env over this process's, so keep only the allowlist.
    os.environ.clear()
    os.environ.update(cli_env)

    started, clock = datetime.now(UTC), time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix="maverick-evals-")).resolve()
    _refuse_dotenv_above(scratch)
    seeds = seed.build_all(scratch / "seeds")
    allowed, excluded = harness.split_server_tools(
        asyncio.run(_server_tools(scratch, seeds))
    )
    folder = HERE / "runs" / f"{started:%Y%m%dT%H%M%SZ}-{args.model}"
    run = Run(
        model=args.model,
        cap=args.budget_usd,
        scratch=scratch,
        seeds=seeds,
        cli_env=cli_env,
        allowed=allowed,
        excluded=excluded,
        sha=_git_sha(),
        traces=folder / "traces",
    )
    run.traces.mkdir(parents=True)
    summary = asyncio.run(run_all(run, _ordered_cases(args.cases)))
    _write_json(
        folder / "run.json",
        {
            "model": run.model,
            "cases_file": args.cases.name,
            "maverick_git_sha": run.sha,
            "apiKeySource": run.init.get("apiKeySource"),
            "tools_advertised": run.init.get("tools"),
            "tools_excluded": run.excluded,
            "redis_enabled": "REDIS_HOST" in run.cli_env,
            "budget_cap_usd": run.cap,
            "per_query_budget_usd": PER_QUERY_BUDGET_USD,
            "smoke_cost_usd": run.smoke_cost,
            "total_notional_cost_usd": round(run.spent, 6),
            **summary,
            "started_at": started.isoformat(),
            "wall_seconds": round(time.monotonic() - clock, 1),
            "scratch_dir": str(scratch),
        },
    )
    print(f"wrote {folder}; total ${run.spent:.4f}; aborted: {summary['aborted']}")
    return 1 if summary["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
