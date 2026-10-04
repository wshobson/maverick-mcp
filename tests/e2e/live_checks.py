"""Explicit live smoke lanes. Paid mode requires fresh human authorization."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from mcp_process import REPO, server_process
from optional_checks import expect, scenario


async def run(output, paid, reserved, skip_parser=False):
    """Run authorized live scenarios with isolated state and saved coverage."""
    output.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="maverick-e2e-live-", dir="/tmp"))
    rows = []
    if paid:
        values = dotenv_values(REPO / ".env")
        env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-6-luna",
            "LLM_API_KEY": values.get("LLM_API_KEY") or values.get("OPENAI_API_KEY"),
            "MAVERICK_E2E_SPEND_RESERVED": reserved,
            "EXA_API_KEY": values.get("EXA_API_KEY"),
            "RESEARCH_SEARCH_BACKEND": "exa",
        }
        assert all(env.values()), "Missing authorized credentials"
        env = {k: str(v) for k, v in env.items()}
        launcher = [
            str(REPO / ".venv/bin/python"),
            str(REPO / "tests/e2e/paid_launcher.py"),
        ]
    else:
        env, launcher = {}, None
    async with server_process(
        "http",
        state,
        output,
        label="live-paid" if paid else "live-yahoo",
        launcher=launcher,
        env_overrides=env,
    ) as server:
        async with server.connect() as client:
            if paid:
                calls = [
                    (
                        "backtesting_parse_strategy",
                        "live-parser",
                        {
                            "description": "Buy when 10 day SMA crosses above 20 day SMA; sell when it crosses below."
                        },
                    ),
                    (
                        "research_run_comprehensive",
                        "live-comprehensive",
                        {
                            "query": "Microsoft latest annual report revenue and risks",
                            "research_scope": "basic",
                            "persona": "conservative",
                        },
                    ),
                    (
                        "research_analyze_company",
                        "live-company",
                        {
                            "symbol": "MSFT",
                            "persona": "conservative",
                            "include_competitive_analysis": False,
                        },
                    ),
                    (
                        "research_analyze_sentiment",
                        "live-sentiment",
                        {"topic": "Microsoft earnings", "persona": "conservative"},
                    ),
                ]
                if skip_parser:
                    calls = calls[1:]
                for tool, label, args in calls:
                    check = (
                        (lambda v: expect(v["method"] == "llm", "live model parse"))
                        if tool.startswith("backtesting")
                        else (
                            lambda v: expect(
                                bool(v["citations"]), "live citations present"
                            )
                        )
                    )
                    value = await scenario(
                        client,
                        rows,
                        output,
                        "http",
                        tool,
                        label,
                        args,
                        mode="live",
                        assertions=check,
                    )
                    if value is None:
                        break  # Stop paid work at the first failed scenario.
            else:
                await scenario(
                    client,
                    rows,
                    output,
                    "http",
                    "market_data_get_quote",
                    "yahoo-quote",
                    {"ticker": "MSFT"},
                    mode="live",
                )
                await scenario(
                    client,
                    rows,
                    output,
                    "http",
                    "market_data_get_price_history",
                    "yahoo-history",
                    {
                        "ticker": "MSFT",
                        "start_date": "2026-09-01",
                        "end_date": "2026-09-30",
                    },
                    mode="live",
                    assertions=lambda v: expect(
                        v["record_count"] >= 20, "completed history sessions"
                    ),
                )
                await scenario(
                    client,
                    rows,
                    output,
                    "http",
                    "market_data_get_price_history",
                    "yahoo-history-cached",
                    {
                        "ticker": "MSFT",
                        "start_date": "2026-09-01",
                        "end_date": "2026-09-30",
                    },
                    mode="live",
                    assertions=lambda v: expect(
                        v["record_count"] >= 20, "cached history still complete"
                    ),
                )
    if paid and (state / "paid-usage.jsonl").exists():
        shutil.copyfile(state / "paid-usage.jsonl", output / "paid-usage.jsonl")
    failed = sum(r["status"] == "fail" for r in rows)
    print(
        json.dumps(
            {
                "passed": sum(r["status"] == "pass" for r in rows),
                "failed": sum(r["status"] == "fail" for r in rows),
            }
        )
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paid-authorized",
        action="store_true",
        help="Only after human authorizes the $1 maximum and credentials",
    )
    parser.add_argument(
        "--skip-parser",
        action="store_true",
        help="Reuse an already recorded live parser success",
    )
    parser.add_argument(
        "--reserved-usd",
        default="0",
        help="Prior conservative reservations under the same authorization",
    )
    args = parser.parse_args()
    asyncio.run(
        run(args.output, args.paid_authorized, args.reserved_usd, args.skip_parser)
    )
