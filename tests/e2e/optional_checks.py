"""Run bounded optional-domain scenarios across real MCP process transports."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from mcp_process import server_process
from research_provider import provider_server

ROOT = Path(__file__).resolve().parents[2]
DATES = {"start_date": "2023-01-03", "end_date": "2025-12-31"}


def payload(result):
    """Extract the domain payload while retaining protocol error text."""
    if result.structured_content is not None:
        return result.structured_content
    text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
    try:
        return json.loads(text)
    except ValueError:
        return {"protocol_error": text, "is_error": result.is_error}


def validate_bounds(value):
    """Check nested trade counts and configured response-size limits."""
    if isinstance(value, dict):
        if "trades" in value and isinstance(value["trades"], list):
            assert len(value["trades"]) <= 20
            assert value["trades_returned"] == len(value["trades"])
            assert value["trades_total"] >= value["trades_returned"]
        for key, item in value.items():
            if key in {"equity_curve", "drawdown_series"}:
                assert len(item) <= 60
            validate_bounds(item)
    elif isinstance(value, list):
        for item in value:
            validate_bounds(item)


async def scenario(
    client,
    rows,
    output,
    transport,
    tool,
    label,
    args,
    *,
    error=False,
    assertions=None,
    mode="synthetic",
    timeout=180,
):
    """Record one tool outcome, payload bounds, and scenario assertions."""
    started = time.monotonic()
    row = {
        "tool": tool,
        "transport": transport,
        "scenario": label,
        "arguments": args,
        "mode": mode,
        "expected": "structured error" if error else "successful structured response",
        "status": "fail",
    }
    try:
        result = await client.request(
            label, "call_tool", name=tool, arguments=args, read_timeout_seconds=timeout
        )
        value = payload(result)
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        (output / f"{transport}-{label}.json").write_text(encoded + "\n")
        row.update(
            response_bytes=len(encoded.encode()), evidence=f"{transport}-{label}.json"
        )
        failed = (
            result.is_error
            or value.get("status") == "error"
            or value.get("success") is False
        )
        assert bool(failed) == error, value
        if not error:
            assert value.get("status") == "success" or value.get("success") is True, (
                value
            )
            required = {
                "backtesting_compare_strategies": ("rankings", "best_overall"),
                "backtesting_train_ml_predictor": ("training_metrics", "data_points"),
                "backtesting_run_ml_strategy_backtest": (
                    "metrics",
                    "equity_curve",
                    "ml_metrics",
                ),
                "backtesting_analyze_market_regimes": (
                    "recent_regime_history",
                    "regime_counts",
                ),
            }
            for key in required.get(tool, ()):
                assert value.get(key), (key, value)
        validate_bounds(value)
        if assertions:
            assertions(value)
        row.update(
            status="pass",
            actual="expected error received"
            if error
            else "response and assertions passed",
        )
        return value
    except Exception as exc:
        row["actual"] = f"{type(exc).__name__}: {exc}"
        return None
    finally:
        row["duration_seconds"] = round(time.monotonic() - started, 4)
        rows.append(row)
        (output / "coverage.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(transport, label, row["status"], flush=True)


def expect(condition, description):
    """Raise an assertion with the scenario-specific explanation."""
    assert condition, description


async def backtesting(transport, output, rows):
    """Exercise backtesting tools with deterministic market-data fixtures."""
    state = Path(
        tempfile.mkdtemp(prefix=f"maverick-e2e-optional-{transport}-", dir="/tmp")
    )
    launcher = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "tests/e2e/fixture_provider.py"),
    ]
    async with server_process(
        transport, state, output / transport, label="backtesting", launcher=launcher
    ) as server:
        async with server.connect() as client:
            discovery = await client.request("discover", "list_tools")
            names = [t.name for t in discovery.tools]
            (output / f"{transport}-tools.json").write_text(
                json.dumps(names, indent=2) + "\n"
            )
            assert len(names) == 53
            calls = [
                (
                    "list_strategies",
                    {},
                    lambda v: expect(v["total_count"] == 12, "12 templates"),
                ),
                (
                    "run_backtest",
                    {"symbol": "AAPL", **DATES, "fast_period": 2, "slow_period": 3},
                    lambda v: expect(
                        v["trades_total"] > 20 and v["trades_truncated"],
                        "trades actually truncated",
                    ),
                ),
                (
                    "compare_strategies",
                    {"symbol": "AAPL", **DATES, "strategies": ["sma_cross", "rsi"]},
                    None,
                ),
                (
                    "backtest_portfolio",
                    {"symbols": ["AAPL", "MSFT", "BAD"], **DATES},
                    lambda v: expect(
                        v["failed"][0]["symbol"] == "BAD", "partial failure visible"
                    ),
                ),
                (
                    "optimize_strategy",
                    {
                        "symbol": "AAPL",
                        **DATES,
                        "optimization_level": "coarse",
                        "top_n": 2,
                    },
                    lambda v: expect(
                        v["total_combinations_tested"] <= 9, "bounded grid"
                    ),
                ),
                (
                    "walk_forward_analysis",
                    {
                        "symbol": "AAPL",
                        "start_date": "2023-01-03",
                        "end_date": "2024-09-30",
                        "window_size": 120,
                        "step_size": 120,
                    },
                    lambda v: expect(
                        v["periods_tested"] > 0, "at least one out of sample window"
                    ),
                ),
                (
                    "monte_carlo_simulation",
                    {"symbol": "AAPL", **DATES, "num_simulations": 20},
                    lambda v: expect(v["num_simulations"] == 20, "bounded simulations"),
                ),
                (
                    "train_ml_predictor",
                    {"symbol": "AAPL", **DATES, "n_estimators": 5, "max_depth": 3},
                    None,
                ),
                (
                    "run_ml_strategy_backtest",
                    {"symbol": "AAPL", **DATES, "n_estimators": 5, "max_depth": 3},
                    None,
                ),
                (
                    "analyze_market_regimes",
                    {"symbol": "AAPL", **DATES, "method": "kmeans"},
                    None,
                ),
                (
                    "create_strategy_ensemble",
                    {
                        "symbols": ["AAPL", "BAD", "SHORT", "MSFT", "SPY", "QQQ"],
                        **DATES,
                        "base_strategies": ["sma_cross", "rsi"],
                    },
                    lambda v: expect(
                        {s["reason"] for s in v["skipped_symbols"]}
                        >= {"fetch_failed", "insufficient_history", "symbol_limit"},
                        "all skipped reasons visible",
                    ),
                ),
                (
                    "parse_strategy",
                    {"description": "SMA crosses 10 and 20"},
                    lambda v: expect(
                        v["method"] == "simple_degraded",
                        "missing LLM is labeled degradation",
                    ),
                ),
            ]
            for name, args, check in calls:
                await scenario(
                    client,
                    rows,
                    output,
                    transport,
                    "backtesting_" + name,
                    name,
                    args,
                    assertions=check,
                )
            await scenario(
                client,
                rows,
                output,
                transport,
                "backtesting_run_backtest",
                "no-trades",
                {"symbol": "FLAT", **DATES},
                assertions=lambda v: expect(
                    v["metrics"]["profit_factor"] is None
                    and v["metrics"]["profit_factor_status"] == "no_trades",
                    "null no-trade factor",
                ),
            )
            await scenario(
                client,
                rows,
                output,
                transport,
                "backtesting_optimize_strategy",
                "nullable-optimization",
                {
                    "symbol": "FLAT",
                    **DATES,
                    "optimization_level": "coarse",
                    "optimization_metric": "profit_factor",
                },
                assertions=lambda v: expect(
                    v["best_metric_value"] is None
                    and v["best_metric_status"] == "no_trades",
                    "undefined optimization remains null",
                ),
            )
            errors = [
                ("run_backtest", {"symbol": "EMPTY", **DATES}),
                ("compare_strategies", {"symbol": "BAD", **DATES}),
                ("backtest_portfolio", {"symbols": ["BAD"], **DATES}),
                (
                    "optimize_strategy",
                    {"symbol": "AAPL", **DATES, "strategy": "nonexistent"},
                ),
                ("walk_forward_analysis", {"symbol": "BAD", **DATES, "step_size": 500}),
                (
                    "monte_carlo_simulation",
                    {"symbol": "EMPTY", **DATES, "num_simulations": 2},
                ),
                ("train_ml_predictor", {"symbol": "SHORT", **DATES, "n_estimators": 5}),
                (
                    "run_ml_strategy_backtest",
                    {"symbol": "SHORT", **DATES, "n_estimators": 5},
                ),
                ("analyze_market_regimes", {"symbol": "EMPTY", **DATES}),
                ("create_strategy_ensemble", {"symbols": ["EMPTY"], **DATES}),
                ("parse_strategy", {"description": {"invalid": "object"}}),
            ]
            for name, args in errors:
                await scenario(
                    client,
                    rows,
                    output,
                    transport,
                    "backtesting_" + name,
                    name + "-error",
                    args,
                    error=True,
                )
            await scenario(
                client,
                rows,
                output,
                transport,
                "backtesting_run_backtest",
                "invalid-dates",
                {"symbol": "AAPL", "start_date": "invalid"},
                error=True,
            )
            for name, args in [
                ("research_run_comprehensive", {"query": "Synthetic"}),
                ("research_analyze_company", {"symbol": "AAPL"}),
                ("research_analyze_sentiment", {"topic": "Synthetic"}),
            ]:
                await scenario(
                    client,
                    rows,
                    output,
                    transport,
                    name,
                    name + "-unconfigured",
                    args,
                    error=True,
                    assertions=lambda v: expect(
                        v["error_type"] == "not_configured", "typed configuration error"
                    ),
                )


async def research(transport, output, rows):
    """Exercise research and parsing through synthetic loopback providers."""
    with provider_server(output / f"{transport}-provider-requests.jsonl") as provider:
        base = f"http://127.0.0.1:{provider.server_port}"
        env = {
            "RESEARCH_SEARCH_BACKEND": "searxng",
            "SEARXNG_BASE_URL": base,
            "LLM_PROVIDER": "openai_compatible",
            "LLM_BASE_URL": base + "/v1",
            "LLM_API_KEY": "synthetic-not-a-secret",
            "LLM_MODEL": "fixture-model",
            "HTTP_RETRIES": "0",
            "DATA_PROVIDER_RATE_LIMIT": "1000",
        }
        for temperature in [None, "0.3"]:
            state = Path(
                tempfile.mkdtemp(
                    prefix=f"maverick-e2e-research-{transport}-", dir="/tmp"
                )
            )
            overrides = env | ({"LLM_TEMPERATURE": temperature} if temperature else {})
            async with server_process(
                transport,
                state,
                output / transport,
                label=f"research-{temperature}",
                env_overrides=overrides,
            ) as server:
                async with server.connect() as client:
                    before = len(provider.requests)
                    await scenario(
                        client,
                        rows,
                        output,
                        transport,
                        "backtesting_parse_strategy",
                        f"llm-parser-{temperature}",
                        {"description": "SMA crosses 10 and 20"},
                        assertions=lambda v: expect(
                            v["method"] == "llm", "model parse succeeds"
                        ),
                    )
                    bodies = [
                        r["body"]
                        for r in provider.requests[before:]
                        if r["method"] == "POST"
                    ]
                    assert bodies and all(
                        (
                            "temperature" not in b
                            if temperature is None
                            else b.get("temperature") == 0.3
                        )
                        for b in bodies
                    )
                    if temperature is not None:
                        continue
                    cases = [
                        (
                            "research_run_comprehensive",
                            "comprehensive",
                            {
                                "query": "Synthetic company outlook",
                                "research_scope": "basic",
                            },
                        ),
                        (
                            "research_analyze_company",
                            "company",
                            {"symbol": "SYNTH", "include_competitive_analysis": False},
                        ),
                        (
                            "research_analyze_company",
                            "competitive",
                            {"symbol": "SYNTH", "include_competitive_analysis": True},
                        ),
                        (
                            "research_analyze_sentiment",
                            "sentiment",
                            {"topic": "Synthetic sector"},
                        ),
                    ]
                    for tool, label, args in cases:
                        before = len(provider.requests)
                        await scenario(
                            client,
                            rows,
                            output,
                            transport,
                            tool,
                            label,
                            args,
                            assertions=lambda v: expect(
                                v["citations"][0]["url"]
                                == "https://example.org/synthetic-financial-report",
                                "citation preserved",
                            ),
                        )
                        queries = [
                            str(r.get("query"))
                            for r in provider.requests[before:]
                            if r["method"] == "GET"
                        ]
                        if label in {"company", "competitive"}:
                            assert any("competit" in q for q in queries) == (
                                label == "competitive"
                            ), queries
                    for mode in ["object_scores", "malformed_scores"]:
                        provider.mode = mode
                        await scenario(
                            client,
                            rows,
                            output,
                            transport,
                            "research_run_comprehensive",
                            mode,
                            {
                                "query": "Synthetic company outlook",
                                "research_scope": "basic",
                            },
                            assertions=lambda v: expect(
                                bool(v["citations"]),
                                "score normalization retains evidence",
                            ),
                        )
                    for mode in ["empty", "search_failure", "model_failure"]:
                        provider.mode = mode
                        for tool, label, args in cases:
                            if label == "competitive":
                                continue
                            await scenario(
                                client,
                                rows,
                                output,
                                transport,
                                tool,
                                f"{label}-{mode}",
                                args,
                                error=True,
                                assertions=(
                                    lambda v: expect(
                                        v["error_type"] == "insufficient_evidence",
                                        "no evidence cannot succeed",
                                    )
                                )
                                if mode != "model_failure"
                                else None,
                            )
                        if mode == "model_failure":
                            await scenario(
                                client,
                                rows,
                                output,
                                transport,
                                "backtesting_parse_strategy",
                                "parser-provider-failure",
                                {"description": "SMA 10 20"},
                                error=True,
                            )
                    provider.mode = "success"


async def main():
    """Run requested optional lanes across both transports and report failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lane", choices=["all", "backtesting", "research"], default="all"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for transport in ["stdio", "http"]:
        if args.lane in {"all", "backtesting"}:
            await backtesting(transport, args.output, rows)
        if args.lane in {"all", "research"}:
            await research(transport, args.output, rows)
    print(
        json.dumps(
            {
                "passed": sum(r["status"] == "pass" for r in rows),
                "failed": sum(r["status"] == "fail" for r in rows),
            }
        )
    )
    if any(r["status"] == "fail" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
