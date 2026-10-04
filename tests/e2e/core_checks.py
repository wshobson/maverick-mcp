"""Opt-in real-process MCP core workflows with deterministic external providers.

Run directly with the project Python. Each result has its own request/response
trace plus explicit expected behavior and assertion outcome in summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from mcp_process import server_process

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = [str(ROOT / ".venv/bin/python"), str(ROOT / "tests/e2e/fixture_provider.py")]
D = Decimal


def payload(result: Any) -> dict[str, Any]:
    """Extract a structured domain payload or retain the protocol error."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    for block in getattr(result, "content", []):
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                pass
    return {"protocol_error": result.model_dump(mode="json", by_alias=True)}


class Checks:
    def __init__(self, transport: str, evidence: Path) -> None:
        """Collect assertions and tool coverage for one transport."""
        self.transport = transport
        self.evidence = evidence
        self.rows: list[dict[str, Any]] = []
        self.tools: set[str] = set()

    def verify(self, label: str, expected: str, actual: Any, assertion: bool) -> None:
        """Record one assertion and immediately persist its outcome."""
        self.rows.append(
            {
                "label": label,
                "transport": self.transport,
                "expected": expected,
                "actual": actual,
                "assertion": bool(assertion),
                "status": "pass" if assertion else "fail",
            }
        )
        self.save()
        print(
            f"{self.transport} {label}: {'PASS' if assertion else 'FAIL'}", flush=True
        )

    def save(self) -> None:
        """Write the current assertion summary and exercised tool names."""
        self.evidence.mkdir(parents=True, exist_ok=True)
        (self.evidence / "summary.json").write_text(
            json.dumps(
                {
                    "transport": self.transport,
                    "provider": "Injected deterministic Yahoo/finviz sync boundary; real MCP process/CLI assembly",
                    "tools_exercised": sorted(self.tools),
                    "passed": sum(r["assertion"] for r in self.rows),
                    "failed": sum(not r["assertion"] for r in self.rows),
                    "checks": self.rows,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

    async def call(
        self,
        client: Any,
        label: str,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        status: str = "success",
        expected: str = "Successful domain result",
        check: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """Call a tool and verify its domain status and scenario predicate."""
        self.tools.add(tool)
        result = await client.request(
            label, "call_tool", name=tool, arguments=args or {}
        )
        data = payload(result)
        valid = data.get("status") == status
        error = None
        if check is not None:
            try:
                valid = valid and check(data)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                valid, error = False, repr(exc)
        self.verify(
            label,
            expected + f"; status={status}",
            {
                "tool": tool,
                "arguments": args or {},
                "payload": data,
                "assertion_error": error,
            },
            valid,
        )
        return data


def provider_calls(state: Path, operation: str, symbol: str) -> int:
    """Count matching calls captured at the synthetic provider boundary."""
    path = state / "provider-calls.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    return sum(
        row["operation"] == operation and row["symbol"] == symbol for row in rows
    )


def fixture_control(state: Path, **values: Any) -> None:
    """Set provider behavior for the next deterministic scenario."""
    (state / "fixture-control.json").write_text(json.dumps(values))


def price_rows(state: Path, symbol: str) -> list[tuple[Any, ...]]:
    """Read ordered stored bars to verify persistence independently."""
    with sqlite3.connect(state / "maverick.db") as conn:
        return conn.execute(
            "SELECT date, open, high, low, close, volume FROM md_price_bars WHERE stock_id=(SELECT id FROM md_stocks WHERE symbol=?) ORDER BY date",
            (symbol,),
        ).fetchall()


async def market_checks(c: Checks, client: Any, state: Path) -> None:
    """Verify market-data payloads, calendars, caches, and provider errors."""
    await c.call(
        client,
        "screens-empty-universe",
        "screening_run_screens",
        status="error",
        expected="Fresh database explains fetching history first",
        check=lambda p: "price history" in p["error"],
    )
    await c.call(
        client,
        "screens-empty-get",
        "screening_get_all",
        expected="No stored screening rows",
    )
    args = {"ticker": "aapl", "start_date": "2023-01-01", "end_date": "2026-09-30"}
    history = await c.call(
        client,
        "history-three-years",
        "market_data_get_price_history",
        args,
        expected="More than 900 ordered OHLCV exchange sessions",
        check=lambda p: (
            p["record_count"] > 900
            and p["columns"] == ["Open", "High", "Low", "Close", "Volume"]
            and p["index"] == sorted(p["index"])
        ),
    )
    before = provider_calls(state, "history", "AAPL")
    await c.call(
        client,
        "history-fresh-repeat",
        "market_data_get_price_history",
        args,
        expected="Fresh repeat equals cached snapshot",
        check=lambda p: p == history,
    )
    c.verify(
        "history-fresh-no-provider",
        "No repeat provider history call",
        {"before": before, "after": provider_calls(state, "history", "AAPL")},
        before == provider_calls(state, "history", "AAPL"),
    )
    await c.call(
        client,
        "history-batch-mixed",
        "market_data_get_price_history_batch",
        {
            "tickers": ["AAPL", "MSFT", "BAD", "BEAR"],
            "start_date": "2025-01-01",
            "end_date": "2026-09-30",
        },
        expected="Three successes and one isolated provider failure",
        check=lambda p: (
            p["success_count"] == 3
            and p["error_count"] == 1
            and p["results"]["BAD"]["status"] == "error"
        ),
    )
    await c.call(
        client,
        "history-malformed-date",
        "market_data_get_price_history",
        {"ticker": "AAPL", "start_date": "2026-02-30"},
        status="error",
    )
    await c.call(
        client,
        "history-reversed-date",
        "market_data_get_price_history",
        {"ticker": "AAPL", "start_date": "2026-09-30", "end_date": "2026-01-01"},
        expected="Reversed range yields zero rows under current contract",
        check=lambda p: p["record_count"] == 0,
    )
    await c.call(
        client,
        "history-unsupported-exchange",
        "market_data_get_price_history",
        {"ticker": "EXAMPLE.ZZ"},
        status="error",
        expected="Unsupported calendar explicitly rejected",
        check=lambda p: "calendar" in p["error"],
    )
    await c.call(
        client,
        "history-london-holiday",
        "market_data_get_price_history",
        {"ticker": "VOD.L", "start_date": "2026-07-03", "end_date": "2026-07-03"},
        expected="LSE trades on the US observed Independence Day",
        check=lambda p: p["record_count"] == 1,
    )
    await c.call(
        client,
        "history-us-holiday",
        "market_data_get_price_history",
        {"ticker": "AAPL", "start_date": "2026-07-03", "end_date": "2026-07-03"},
        expected="NYSE observed holiday returns zero rows",
        check=lambda p: p["record_count"] == 0,
    )
    await c.call(
        client,
        "history-tokyo",
        "market_data_get_price_history",
        {"ticker": "7203.T", "start_date": "2026-09-01", "end_date": "2026-09-30"},
        expected="JPX calendar history accepted",
        check=lambda p: p["record_count"] > 15,
    )
    await c.call(
        client,
        "quote-normalized",
        "market_data_get_quote",
        {"ticker": "aapl"},
        expected="Uppercase ticker and fixed fixture price",
        check=lambda p: p["symbol"] == "AAPL" and p["price"] == 150.25,
    )
    fixture_control(state, quotes={"AAPL": 160.25})
    await c.call(
        client,
        "quote-ttl-cache",
        "market_data_get_quote",
        {"ticker": "AAPL"},
        expected="Cached quote retains price until invalidated",
        check=lambda p: p["price"] == 150.25,
    )
    await c.call(
        client,
        "quote-clear-one",
        "market_data_clear_market_cache",
        {"ticker": "aapl"},
        expected="One quote invalidated",
        check=lambda p: p["entries_cleared"] == 1,
    )
    await c.call(
        client,
        "quote-refreshed",
        "market_data_get_quote",
        {"ticker": "AAPL"},
        expected="New provider quote visible after invalidation",
        check=lambda p: p["price"] == 160.25,
    )
    fixture_control(state)
    await c.call(
        client,
        "quote-clear-all",
        "market_data_clear_market_cache",
        expected="Clear all quote and overview caches",
    )
    await c.call(
        client, "quote-bad", "market_data_get_quote", {"ticker": "BAD"}, status="error"
    )
    await c.call(
        client,
        "quote-no-data",
        "market_data_get_quote",
        {"ticker": "EMPTY"},
        status="error",
        check=lambda p: "No quote data" in p["error"],
    )
    await c.call(
        client,
        "fundamentals",
        "market_data_get_stock_fundamentals",
        {"ticker": "AAPL"},
        expected="Mapped company and valuation values",
        check=lambda p: (
            p["company"]["sector"] == "Technology" and p["valuation"]["pe_ratio"] == 20
        ),
    )
    await c.call(
        client,
        "fundamentals-bad",
        "market_data_get_stock_fundamentals",
        {"ticker": "BAD"},
        status="error",
    )
    await c.call(
        client,
        "market-overview",
        "market_data_get_market_overview",
        expected="Six indices, eleven sectors, movers, VIX-derived elevated fear",
        check=lambda p: (
            len(p["indices"]) == 6
            and len(p["sectors"]) == 11
            and p["volatility"]["vix"] == 21
            and p["volatility"]["fear_level"] == "elevated"
            and bool(p["top_gainers"])
        ),
    )
    await c.call(
        client,
        "chart-links",
        "market_data_get_chart_links",
        {"ticker": "AAPL"},
        expected="Four external chart links",
        check=lambda p: (
            len(p["charts"]) == 4 and "AAPL" in p["charts"]["yahoo_finance"]
        ),
    )
    rows_before = price_rows(state, "AAPL")
    with sqlite3.connect(state / "maverick.db") as conn:
        conn.execute(
            "UPDATE md_stocks SET history_refreshed_at='2000-01-01' WHERE symbol='AAPL'"
        )
    before = provider_calls(state, "history", "AAPL")
    await c.call(
        client,
        "history-stale-refresh",
        "market_data_get_price_history",
        args,
        expected="Expired history refreshes while preserving values",
        check=lambda p: p == history,
    )
    c.verify(
        "history-stale-provider",
        "Expired history causes provider call",
        {"before": before, "after": provider_calls(state, "history", "AAPL")},
        provider_calls(state, "history", "AAPL") == before + 1,
    )
    with sqlite3.connect(state / "maverick.db") as conn:
        conn.execute(
            "UPDATE md_stocks SET history_refreshed_at='2000-01-01' WHERE symbol='AAPL'"
        )
    fixture_control(state, history_modes={"AAPL": "partial"})
    await c.call(
        client,
        "history-incomplete-refresh",
        "market_data_get_price_history",
        args,
        status="error",
        expected="Interior missing session refuses replacement",
        check=lambda p: "Incomplete price history" in p["error"],
    )
    c.verify(
        "history-incomplete-preserves-snapshot",
        "Every stored OHLCV value preserved after rejected refresh",
        {
            "row_count_before": len(rows_before),
            "row_count_after": len(price_rows(state, "AAPL")),
        },
        price_rows(state, "AAPL") == rows_before,
    )
    fixture_control(state)
    await c.call(
        client,
        "history-recovered",
        "market_data_get_price_history",
        args,
        check=lambda p: p == history,
    )


async def screening_technical_checks(c: Checks, client: Any) -> None:
    """Verify screen filters and indicators against fixture history."""
    await c.call(
        client,
        "screens-populated",
        "screening_run_screens",
        expected="All three screens run over populated universe",
        check=lambda p: (
            p["count"] == 3
            and all(r["symbols_screened"] >= 6 for r in p["results"].values())
        ),
    )
    for screen in ("bullish", "bearish", "supply_demand"):
        await c.call(
            client,
            f"screen-{screen}",
            f"screening_get_{screen}",
            expected="Typed screen results with count and reasons",
            check=lambda p, screen=screen: (
                p["count"] > 0
                and p["count"] == len(p["results"])
                and all(r["screen"] == screen and r["reason"] for r in p["results"])
            ),
        )
        await c.call(
            client,
            f"run-{screen}",
            "screening_run_screens",
            {"screen": screen},
            expected="Single screen preserves mapping response",
            check=lambda p, screen=screen: list(p["results"]) == [screen],
        )
    await c.call(client, "screens-all", "screening_get_all")
    await c.call(
        client,
        "screen-criteria",
        "screening_get_by_criteria",
        {
            "min_volume": 1_000_000,
            "max_price": 1000,
            "min_combined_score": 50,
            "limit": 5,
        },
        expected="AND criteria applied",
        check=lambda p: (
            p["count"] <= 5
            and all(
                r["close"] <= 1000
                and r["combined_score"] >= 50
                and r["indicators"]["volume"] >= 1_000_000
                for r in p["results"]
            )
        ),
    )
    await c.call(
        client,
        "screen-criteria-none",
        "screening_get_by_criteria",
        {"max_price": 0.001},
        expected="Impossible price excludes all rows",
        check=lambda p: p["count"] == 0,
    )
    await c.call(
        client,
        "screen-invalid",
        "screening_run_screens",
        {"screen": "unknown"},
        status="error",
    )
    await c.call(
        client,
        "rsi",
        "technical_get_rsi_analysis",
        {"ticker": "AAPL"},
        expected="Finite RSI in range 0 to 100",
        check=lambda p: 0 <= p["current"] <= 100,
    )
    await c.call(
        client,
        "macd",
        "technical_get_macd_analysis",
        {"ticker": "AAPL"},
        expected="Finite MACD histogram",
        check=lambda p: math.isfinite(p["histogram"]),
    )
    history = await c.call(
        client,
        "levels-source-history",
        "market_data_get_price_history",
        {
            "ticker": "AAPL",
            "start_date": (date.today() - timedelta(days=400)).isoformat(),
            "end_date": date.today().isoformat(),
        },
    )
    rows = history["data"][-30:]
    await c.call(
        client,
        "support-resistance",
        "technical_get_support_resistance",
        {"ticker": "AAPL"},
        expected="Observed 30-bar range exactly matches history extrema",
        check=lambda p: (
            p["method"] == "observed_range"
            and p["bars_analyzed"] == 30
            and p["support"] == [min(r[2] for r in rows)]
            and p["resistance"] == [max(r[1] for r in rows)]
        ),
    )
    days = 14
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = [
        row
        for index, row in zip(history["index"], history["data"], strict=True)
        if index[:10] >= cutoff
    ]
    levels = await c.call(
        client,
        "support-resistance-calendar-days",
        "technical_get_support_resistance",
        {"ticker": "AAPL", "days": days},
        expected="Requested calendar days select actual bars and extrema",
        check=lambda p: (
            p["bars_analyzed"] == len(rows)
            and p["support"] == [min(r[2] for r in rows)]
            and p["resistance"] == [max(r[1] for r in rows)]
        ),
    )
    await c.call(
        client,
        "full-technical",
        "technical_get_full_technical_analysis",
        {"ticker": "AAPL", "days": days},
        expected="Full analysis uses same observed range and warm-up >=200 bars",
        check=lambda p: (
            p["levels"]["bars_analyzed"] == levels["bars_analyzed"]
            and p["levels"]["support"] == levels["support"]
            and p["analysis_metadata"]["bars_analyzed"] >= 200
        ),
    )
    for tool in (
        "technical_get_rsi_analysis",
        "technical_get_macd_analysis",
        "technical_get_support_resistance",
        "technical_get_full_technical_analysis",
    ):
        await c.call(
            client,
            f"{tool}-empty",
            tool,
            {"ticker": "EMPTY"},
            status="error",
            expected="Empty history rejected explicitly",
            check=lambda p: "Insufficient" in p["error"],
        )
    await c.call(
        client,
        "full-technical-insufficient",
        "technical_get_full_technical_analysis",
        {"ticker": "SHORT"},
        status="error",
        expected="Five bars cannot yield 200-bar full analysis",
        check=lambda p: "Insufficient" in p["error"],
    )
    await c.call(
        client,
        "levels-invalid-days",
        "technical_get_support_resistance",
        {"ticker": "AAPL", "days": 0},
        status="error",
    )


async def portfolio_checks(c: Checks, client: Any, server: Any) -> None:
    """Verify Decimal accounting, risk checks, and concurrent writes."""
    await c.call(
        client,
        "portfolio-empty",
        "portfolio_get_my_portfolio",
        check=lambda p: p["metrics"]["position_count"] == 0,
    )
    await c.call(client, "risk-empty", "portfolio_get_risk_dashboard", status="empty")
    await c.call(client, "alerts-empty", "portfolio_get_risk_alerts", status="empty")
    await c.call(
        client,
        "portfolio-add",
        "portfolio_add_position",
        {
            "ticker": "aapl",
            "shares": 1.25,
            "purchase_price": 100.1234,
            "purchase_date": "2026-01-02",
        },
        expected="Fractional holding with Decimal cost",
        check=lambda p: (
            p["position"]["ticker"] == "AAPL"
            and D(p["position"]["total_cost"]) == D("125.1543")
        ),
    )
    total_cost = (D("125.1543") + D(".75") * D("110.5678")).quantize(
        D(".0001"), rounding=ROUND_HALF_UP
    )
    basis = (total_cost / D("2")).quantize(D(".0001"), rounding=ROUND_HALF_UP)
    await c.call(
        client,
        "portfolio-average",
        "portfolio_add_position",
        {"ticker": "AAPL", "shares": 0.75, "purchase_price": 110.5678},
        expected=f"Independent Decimal total={total_cost}, basis={basis}, shares=2",
        check=lambda p: (
            D(p["position"]["shares"]) == 2
            and D(p["position"]["total_cost"]) == total_cost
            and D(p["position"]["average_cost_basis"]) == basis
        ),
    )
    await c.call(
        client,
        "portfolio-add-second",
        "portfolio_add_position",
        {"ticker": "MSFT", "shares": 3.5, "purchase_price": 250.25},
    )
    await c.call(
        client,
        "portfolio-remove-part",
        "portfolio_remove_position",
        {"ticker": "AAPL", "shares": 0.25},
        check=lambda p: (
            D(p["shares_removed"]) == D(".25") and not p["position_fully_closed"]
        ),
    )
    remaining_cost = (D("1.75") * basis).quantize(D(".0001"), rounding=ROUND_HALF_UP)
    await c.call(
        client,
        "portfolio-values",
        "portfolio_get_my_portfolio",
        expected="Remaining average cost unchanged and live P&L matches independent math",
        check=lambda p: (
            D(p["positions"][0]["average_cost_basis"]) == basis
            and D(p["positions"][0]["total_cost"]) == remaining_cost
            and abs(
                p["metrics"]["total_value"]
                - float(
                    (D("1.75") * D("150.25") + D("3.5") * D("300.5")).quantize(
                        D(".01"), rounding=ROUND_HALF_UP
                    )
                )
            )
            < 1e-8
        ),
    )
    await c.call(
        client,
        "portfolio-over-remove-setup",
        "portfolio_add_position",
        {
            "ticker": "AAPL",
            "shares": 1,
            "purchase_price": 100,
            "portfolio_name": "Oversell",
        },
    )
    await c.call(
        client,
        "portfolio-over-remove",
        "portfolio_remove_position",
        {"ticker": "AAPL", "shares": 999, "portfolio_name": "Oversell"},
        expected="Removing more shares than held closes the position under documented contract",
        check=lambda p: p["position_fully_closed"] and D(p["shares_removed"]) == 1,
    )
    for key, val in (
        ("shares", 0),
        ("shares", -1),
        ("purchase_price", 0),
        ("purchase_date", "invalid"),
    ):
        await c.call(
            client,
            f"portfolio-invalid-{key}-{val}",
            "portfolio_add_position",
            {"ticker": "AAPL", "shares": 1, "purchase_price": 1, key: val},
            status="error",
        )
    await c.call(
        client,
        "clear-unconfirmed",
        "portfolio_clear_portfolio",
        status="error",
        expected="Explicit confirmation required",
    )
    await c.call(
        client,
        "portfolio-retained-after-clear-refusal",
        "portfolio_get_my_portfolio",
        check=lambda p: p["metrics"]["position_count"] == 2,
    )
    await c.call(
        client,
        "risk-adjusted",
        "portfolio_risk_adjusted_analysis",
        {"ticker": "AAPL", "risk_level": 50},
        expected="Positive ATR sizing plus existing position",
        check=lambda p: p["atr"] > 0 and p["existing_position"] is not None,
    )
    await c.call(
        client,
        "comparison-holdings",
        "portfolio_compare_tickers",
        expected="Omitted tickers expands portfolio holdings",
        check=lambda p: set(p["comparison"]) == {"AAPL", "MSFT"},
    )
    await c.call(
        client,
        "correlation-holdings",
        "portfolio_correlation_analysis",
        expected="Symmetric 2x2 correlation and unit diagonal",
        check=lambda p: (
            p["matrix"]["AAPL"]["AAPL"] == 1
            and p["matrix"]["AAPL"]["MSFT"] == p["matrix"]["MSFT"]["AAPL"]
            and p["data_points"] > 50
        ),
    )
    await c.call(
        client,
        "comparison-insufficient",
        "portfolio_compare_tickers",
        {"tickers": ["AAPL"]},
        status="error",
    )
    await c.call(
        client,
        "risk-dashboard",
        "portfolio_get_risk_dashboard",
        expected="Two positions, value and nonnegative VaR",
        check=lambda p: (
            p["position_count"] == 2
            and p["portfolio_var_99"] >= p["portfolio_var_95"] >= 0
            and abs(
                p["total_value"]
                - float(
                    (D("1.75") * D("150.25") + D("3.5") * D("300.5")).quantize(
                        D(".01"), rounding=ROUND_HALF_UP
                    )
                )
            )
            < 1e-8
        ),
    )
    await c.call(
        client,
        "risk-pretrade",
        "portfolio_check_position_risk",
        {"ticker": "AAPL", "shares": 2, "entry_price": 150.25},
        expected="Projected value adds exactly 300.5",
        check=lambda p: (
            abs(p["projected"]["total_value"] - p["current"]["total_value"] - 300.5)
            < 1e-8
        ),
    )
    await c.call(
        client,
        "regime-sizing",
        "portfolio_get_regime_adjusted_sizing",
        {"account_size": 10000, "entry_price": 100, "stop_loss": 95, "risk_pct": 2},
        expected="Risk-scaled whole shares match stop distance",
        check=lambda p: (
            p["shares"] == int(p["risk_amount"] / 5)
            and p["position_value"] == p["shares"] * 100
        ),
    )
    await c.call(
        client,
        "regime-equal-stop",
        "portfolio_get_regime_adjusted_sizing",
        {"account_size": 10000, "entry_price": 100, "stop_loss": 100},
        expected="Zero stop distance produces zero shares under existing contract",
        check=lambda p: p["shares"] == 0,
    )
    for label, overrides in [
        ("account", {"account_size": -10000}),
        ("price", {"entry_price": -100}),
        ("stop", {"stop_loss": -95}),
        ("risk", {"risk_pct": -2}),
        ("nan", {"account_size": "NaN"}),
    ]:
        await c.call(
            client,
            f"regime-invalid-{label}",
            "portfolio_get_regime_adjusted_sizing",
            {"account_size": 10000, "entry_price": 100, "stop_loss": 95, **overrides},
            status="error",
            expected="Invalid numeric sizing input rejected",
        )
    for label, overrides in [
        ("shares", {"shares": -1}),
        ("price", {"entry_price": 0}),
        ("nan", {"shares": "NaN"}),
    ]:
        await c.call(
            client,
            f"pretrade-invalid-{label}",
            "portfolio_check_position_risk",
            {"ticker": "AAPL", "shares": 1, "entry_price": 100, **overrides},
            status="error",
            expected="Invalid prospective position rejected without fabricating risk",
        )
    await c.call(
        client,
        "portfolio-nonfinite",
        "portfolio_add_position",
        {"ticker": "AAPL", "shares": "NaN", "purchase_price": 100},
        status="error",
    )
    await c.call(
        client,
        "risk-alerts",
        "portfolio_get_risk_alerts",
        expected="Concentration yields count-matched risk alerts",
        check=lambda p: p["alert_count"] == len(p["alerts"]) and p["alert_count"] > 0,
    )
    # Independent MCP clients/processes share only this test's persistent DB.
    async with (
        server_process(
            server.transport,
            server.state_dir,
            server.evidence_dir,
            label="concurrent-a",
            launcher=LAUNCHER,
        ) as process_a,
        server_process(
            server.transport,
            server.state_dir,
            server.evidence_dir,
            label="concurrent-b",
            launcher=LAUNCHER,
        ) as process_b,
    ):
        async with (
            process_a.connect("concurrent-a") as a,
            process_b.connect("concurrent-b") as b,
        ):
            await asyncio.gather(
                *(
                    c.call(
                        a if i % 2 == 0 else b,
                        f"concurrent-add-{i}",
                        "portfolio_add_position",
                        {
                            "ticker": "CONCURRENT",
                            "shares": 0.125,
                            "purchase_price": 10.125,
                            "portfolio_name": "Concurrency",
                        },
                    )
                    for i in range(8)
                )
            )
    await c.call(
        client,
        "concurrent-total",
        "portfolio_get_my_portfolio",
        {"portfolio_name": "Concurrency"},
        expected="Eight independent writes retain shares=1, total_cost=10.1248 after per-add storage rounding",
        check=lambda p: (
            D(p["positions"][0]["shares"]) == 1
            and D(p["positions"][0]["total_cost"]) == D("10.1248")
        ),
    )
    await c.call(
        client,
        "portfolio-remove-full",
        "portfolio_remove_position",
        {"ticker": "CONCURRENT", "portfolio_name": "Concurrency"},
        check=lambda p: p["position_fully_closed"],
    )
    await c.call(
        client,
        "portfolio-clear-confirmed",
        "portfolio_clear_portfolio",
        {"confirm": True, "portfolio_name": "Concurrency"},
        check=lambda p: p["positions_cleared"] == 0,
    )


async def watchlist_journal_checks(c: Checks, client: Any) -> dict[str, Any]:
    """Verify watchlist and journal workflows and return restart IDs."""
    await c.call(
        client,
        "watchlist-empty",
        "portfolio_watchlist_list",
        check=lambda p: p["count"] == 0,
    )
    watch = await c.call(
        client,
        "watchlist-create",
        "portfolio_watchlist_create",
        {"name": "E2E", "description": "Synthetic isolated portfolio watchlist"},
    )
    watch_id = watch["id"]
    await c.call(
        client,
        "watchlist-duplicate-name",
        "portfolio_watchlist_create",
        {"name": "E2E"},
        status="error",
    )
    await c.call(
        client,
        "watchlist-add",
        "portfolio_watchlist_add",
        {"watchlist_id": watch_id, "symbol": "aapl", "notes": "fixture"},
        check=lambda p: p["symbol"] == "AAPL",
    )
    await c.call(
        client,
        "watchlist-brief",
        "portfolio_watchlist_brief",
        {"watchlist_id": watch_id},
        check=lambda p: (
            p["count"] == 1
            and p["items"][0]["current_price"] == 150.25
            and p["items"][0]["notes"] == "fixture"
        ),
    )
    await c.call(
        client,
        "watchlist-remove",
        "portfolio_watchlist_remove",
        {"watchlist_id": watch_id, "symbol": "AAPL"},
        check=lambda p: p["removed"],
    )
    await c.call(
        client,
        "watchlist-remove-again",
        "portfolio_watchlist_remove",
        {"watchlist_id": watch_id, "symbol": "AAPL"},
        check=lambda p: not p["removed"],
    )
    await c.call(
        client,
        "watchlist-invalid-id",
        "portfolio_watchlist_add",
        {"watchlist_id": 99999, "symbol": "AAPL"},
        status="error",
    )
    await c.call(
        client,
        "watchlist-remove-invalid-id",
        "portfolio_watchlist_remove",
        {"watchlist_id": 99999, "symbol": "AAPL"},
        status="error",
    )
    await c.call(
        client,
        "watchlist-brief-invalid-id",
        "portfolio_watchlist_brief",
        {"watchlist_id": 99998},
        status="error",
    )
    await c.call(
        client,
        "watchlist-retained-add",
        "portfolio_watchlist_add",
        {"watchlist_id": watch_id, "symbol": "MSFT"},
    )
    await c.call(
        client,
        "watchlist-list",
        "portfolio_watchlist_list",
        check=lambda p: p["count"] == 1 and p["watchlists"][0]["id"] == watch_id,
    )
    cases = [
        ("long", 100.0, 110.0, 1.25),
        ("short", 100.0, 90.0, 0.75),
        ("long", 1.005, 1.006, 10000.0),
        ("long", 100.0, 90.0, 2.0),
    ]
    ids, pnls = [], []
    for index, (side, entry, exit_price, shares) in enumerate(cases):
        pnl = (
            (D(str(exit_price)) - D(str(entry)))
            * D(str(shares))
            * (1 if side == "long" else -1)
        ).quantize(D(".01"), rounding=ROUND_HALF_UP)
        trade = await c.call(
            client,
            f"journal-add-{index}",
            "portfolio_journal_add_trade",
            {
                "symbol": "AAPL",
                "side": side,
                "entry_price": entry,
                "shares": shares,
                "tags": ["e2e"],
                "entry_date": "2026-01-02T14:30:00Z",
            },
            expected="Entry date and subcent unit precision preserved",
            check=lambda p, entry=entry, shares=shares: (
                p["entry_price"] == entry
                and p["shares"] == shares
                and p["entry_date"].startswith("2026-01-02")
            ),
        )
        ids.append(trade["id"])
        pnls.append(pnl)
        await c.call(
            client,
            f"journal-close-{index}",
            "portfolio_journal_close_trade",
            {"entry_id": trade["id"], "exit_price": exit_price},
            expected=f"Independent Decimal realized P&L={pnl}",
            check=lambda p, pnl=pnl: D(str(p["pnl"])) == pnl,
        )
        await c.call(
            client,
            f"journal-review-{index}",
            "portfolio_journal_review",
            {"entry_id": trade["id"]},
            expected="Review matches realized P&L and side-aware percent",
            check=lambda p, pnl=pnl: (
                p["found"] and D(str(p["pnl"])) == pnl and p["pnl_pct"] is not None
            ),
        )
    await c.call(
        client,
        "journal-reclose",
        "portfolio_journal_close_trade",
        {"entry_id": ids[0], "exit_price": 120},
        status="error",
    )
    await c.call(
        client,
        "journal-invalid-date",
        "portfolio_journal_add_trade",
        {
            "symbol": "AAPL",
            "side": "long",
            "entry_price": 1,
            "shares": 1,
            "entry_date": "2026-99-99",
        },
        status="error",
    )
    for label, overrides in [
        ("side", {"side": "sideways"}),
        ("shares", {"shares": -1}),
        ("price", {"entry_price": 0}),
    ]:
        await c.call(
            client,
            f"journal-invalid-{label}",
            "portfolio_journal_add_trade",
            {
                "symbol": "AAPL",
                "side": "long",
                "entry_price": 1,
                "shares": 1,
                **overrides,
            },
            status="error",
        )
    await c.call(
        client,
        "journal-invalid-close",
        "portfolio_journal_close_trade",
        {"entry_id": 99999, "exit_price": -1},
        status="error",
    )
    await c.call(
        client,
        "journal-list",
        "portfolio_journal_list_trades",
        {"status": "closed", "strategy_tag": "e2e"},
        expected="Exactly four closed matching trades",
        check=lambda p: p["count"] == 4,
    )
    await c.call(
        client,
        "journal-missing-review",
        "portfolio_journal_review",
        {"entry_id": 99999},
        check=lambda p: not p["found"],
    )
    await c.call(
        client,
        "strategy-performance",
        "portfolio_get_strategy_performance",
        {"strategy_tag": "e2e"},
        expected=f"Aggregate P&L={sum(pnls)}, 3 wins 1 loss",
        check=lambda p: (
            p["found"]
            and D(str(p["total_pnl"])) == sum(pnls)
            and p["win_count"] == 3
            and p["loss_count"] == 1
        ),
    )
    await c.call(
        client,
        "strategy-comparison",
        "portfolio_get_strategy_performance",
        {"compare": True},
        check=lambda p: p["count"] == 1 and p["strategies"][0]["rank"] == 1,
    )
    await c.call(
        client,
        "strategy-missing-tag",
        "portfolio_get_strategy_performance",
        status="error",
    )
    for field in ("shares", "entry_price"):
        await c.call(
            client,
            f"journal-nonfinite-{field}",
            "portfolio_journal_add_trade",
            {
                "symbol": "AAPL",
                "side": "long",
                "shares": 1,
                "entry_price": 100,
                field: "NaN",
            },
            status="error",
        )
    future = await c.call(
        client,
        "journal-future-entry",
        "portfolio_journal_add_trade",
        {
            "symbol": "AAPL",
            "side": "long",
            "shares": 1,
            "entry_price": 100,
            "entry_date": "2099-01-01",
        },
    )
    await c.call(
        client,
        "journal-close-before-entry",
        "portfolio_journal_close_trade",
        {"entry_id": future["id"], "exit_price": 110},
        status="error",
        expected="Exit before entry is rejected",
    )
    return {"watchlist_id": watch_id, "journal_id": ids[2]}


async def run(transport: str, evidence_root: Path) -> bool:
    """Exercise every core tool and verify persistence after process restart."""
    state = Path(tempfile.mkdtemp(prefix=f"maverick-e2e-core-{transport}-"))
    evidence = evidence_root / transport
    c = Checks(transport, evidence)
    async with server_process(
        transport, state, evidence, label="core", launcher=LAUNCHER
    ) as server:
        async with server.connect("core") as client:
            tools = await client.request("core-tools", "list_tools")
            expected_tools = {
                t.name
                for t in tools.tools
                if t.name.startswith(
                    ("market_data_", "technical_", "screening_", "portfolio_")
                )
            }
            c.verify(
                "core-tool-count",
                "38 core tools registered",
                len(expected_tools),
                len(expected_tools) == 38,
            )
            await market_checks(c, client, state)
            await screening_technical_checks(c, client)
            await portfolio_checks(c, client, server)
            persisted = await watchlist_journal_checks(c, client)
    async with server_process(
        transport, state, evidence, label="restart", launcher=LAUNCHER
    ) as server:
        async with server.connect("restart") as client:
            await c.call(
                client,
                "restart-portfolio",
                "portfolio_get_my_portfolio",
                expected="Two holdings survive actual process restart",
                check=lambda p: p["metrics"]["position_count"] == 2,
            )
            await c.call(
                client,
                "restart-watchlist",
                "portfolio_watchlist_brief",
                {"watchlist_id": persisted["watchlist_id"]},
                check=lambda p: p["count"] == 1 and p["items"][0]["symbol"] == "MSFT",
            )
            await c.call(
                client,
                "restart-journal",
                "portfolio_journal_review",
                {"entry_id": persisted["journal_id"]},
                expected="Subcent realized P&L remains exactly ten dollars after restart",
                check=lambda p: p["found"] and p["pnl"] == 10,
            )
            await c.call(client, "restart-screen", "screening_get_all")
    c.verify(
        "all-core-tools-exercised",
        "Every discovered core tool exercised",
        {
            "expected": sorted(expected_tools),
            "missing": sorted(expected_tools - c.tools),
        },
        expected_tools <= c.tools,
    )
    (evidence / "state-location.json").write_text(
        json.dumps(
            {
                "state_dir": str(state),
                "database": str(state / "maverick.db"),
                "isolation": "Temporary test-only state; retained for debugging",
                "launcher": LAUNCHER,
            },
            indent=2,
        )
        + "\n"
    )
    shutil.copyfile(state / "provider-calls.jsonl", evidence / "provider-calls.jsonl")
    c.save()
    return all(r["assertion"] for r in c.rows)


async def main() -> None:
    """Run selected transports and fail if any core assertion fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=["stdio", "http", "both"], default="both"
    )
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "tests/e2e/evidence/2026-10-04/core"
    )
    args = parser.parse_args()
    transports = ["stdio", "http"] if args.transport == "both" else [args.transport]
    results = []
    for transport in transports:
        results.append(await run(transport, args.evidence))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
