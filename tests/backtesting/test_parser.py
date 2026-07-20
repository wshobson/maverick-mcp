"""Tests for `maverick.backtesting.strategies.parser.StrategyParser`.

`parse_simple` parity cases are ported from the legacy integration suite
(`tests/integration/test_mcp_tools.py::test_strategy_tools_integration`'s
`parse_test_cases`), with real assertions on the parsed `strategy_type`/
`parameters` instead of the legacy test's loose `isinstance(result, dict)`
checks -- the legacy test never pinned exact output because it exercised the
tool end-to-end; this module has direct access to `StrategyParser` so it can
assert precisely.

`parse_with_llm` is exercised via a `maverick.platform.llm.get_llm`
monkeypatch (no real langchain network calls, no sys.modules stubbing
required since this dev environment already has the `[research]` extra's
langchain packages installed -- the one test that genuinely imports
`langchain_core.prompts.PromptTemplate` is guarded with
`pytest.importorskip` so it degrades to a skip rather than a hard failure on
an environment without the extra).
"""

import json

import pytest

from maverick.backtesting.strategies.parser import StrategyParser

# ---------------------------------------------------------------------------
# parse_simple parity (ported from legacy tests/integration/test_mcp_tools.py)
# ---------------------------------------------------------------------------


def test_parse_simple_rsi_parity():
    parser = StrategyParser()
    result = parser.parse_simple("Buy when RSI is below 30 and sell when above 70")

    assert result == {
        "strategy_type": "rsi",
        "parameters": {"period": 14, "oversold": 30, "overbought": 70},
    }
    assert parser.validate_strategy(result) is True


def test_parse_simple_sma_cross_parity():
    parser = StrategyParser()
    result = parser.parse_simple("Use 10-day and 20-day moving average crossover")

    assert result == {
        "strategy_type": "sma_cross",
        "parameters": {"fast_period": 10, "slow_period": 20},
    }
    assert parser.validate_strategy(result) is True


def test_parse_simple_macd_parity():
    parser = StrategyParser()
    result = parser.parse_simple("MACD strategy with standard parameters")

    assert result == {
        "strategy_type": "macd",
        "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    }
    assert parser.validate_strategy(result) is True


def test_parse_simple_defaults_to_momentum_for_unmatched_description_parity():
    parser = StrategyParser()
    result = parser.parse_simple("Invalid strategy description that makes no sense")

    assert result == {
        "strategy_type": "momentum",
        "parameters": {"lookback": 20, "threshold": 0.05},
    }
    assert parser.validate_strategy(result) is True


def test_parse_simple_bollinger():
    parser = StrategyParser()
    result = parser.parse_simple("Bollinger bands with 20 period and 2.5 std dev")

    assert result == {
        "strategy_type": "bollinger",
        "parameters": {"period": 20, "std_dev": 2.5},
    }


def test_parse_simple_breakout():
    parser = StrategyParser()
    result = parser.parse_simple("Breakout channel with 55 and 20 lookback")

    assert result == {
        "strategy_type": "breakout",
        "parameters": {"lookback": 55, "exit_lookback": 20},
    }


# ---------------------------------------------------------------------------
# validate_strategy
# ---------------------------------------------------------------------------


def test_validate_strategy_rejects_unknown_strategy_type():
    parser = StrategyParser()
    assert parser.validate_strategy({"strategy_type": "not_a_real_strategy"}) is False


def test_validate_strategy_rejects_missing_required_params():
    parser = StrategyParser()
    config = {"strategy_type": "sma_cross", "parameters": {"fast_period": 10}}
    assert parser.validate_strategy(config) is False


# ---------------------------------------------------------------------------
# parse_with_llm: degrade path (no LLM configured / provider package missing)
# ---------------------------------------------------------------------------


async def test_parse_with_llm_degrades_when_not_configured(monkeypatch):
    def _raise_not_configured():
        raise ValueError("No LLM configured; set LLM_PROVIDER ...")

    monkeypatch.setattr("maverick.platform.llm.get_llm", _raise_not_configured)

    parser = StrategyParser()
    description = "MACD strategy with standard parameters"
    result = await parser.parse_with_llm(description)

    expected = parser.parse_simple(description)
    expected["method"] = "simple_degraded"
    assert result == expected


async def test_parse_with_llm_degrades_when_provider_package_missing(monkeypatch):
    def _raise_import_error():
        raise ImportError("langchain_anthropic is required ...")

    monkeypatch.setattr("maverick.platform.llm.get_llm", _raise_import_error)

    parser = StrategyParser()
    description = "Buy when RSI is below 30 and sell when above 70"
    result = await parser.parse_with_llm(description)

    expected = parser.parse_simple(description)
    expected["method"] = "simple_degraded"
    assert result == expected


async def test_parse_with_llm_degrades_on_non_json_response(monkeypatch):
    pytest.importorskip("langchain_core")

    class _FakeModel:
        async def ainvoke(self, _prompt):
            class _Response:
                content = "not valid json"

            return _Response()

    monkeypatch.setattr("maverick.platform.llm.get_llm", lambda: _FakeModel())

    parser = StrategyParser()
    description = "Use 10-day and 20-day moving average crossover"
    result = await parser.parse_with_llm(description)

    expected = parser.parse_simple(description)
    expected["method"] = "simple_degraded"
    assert result == expected


# ---------------------------------------------------------------------------
# parse_with_llm: successful model-backed parse (stubbed model)
# ---------------------------------------------------------------------------


async def test_parse_with_llm_returns_model_parsed_mapping(monkeypatch):
    pytest.importorskip("langchain_core")

    captured_prompts = []
    llm_payload = {
        "strategy_type": "sma_cross",
        "parameters": {"fast_period": 15, "slow_period": 45},
        "entry_logic": "Buy when fast SMA crosses above slow SMA",
        "exit_logic": "Sell when fast SMA crosses below slow SMA",
    }

    class _FakeModel:
        async def ainvoke(self, prompt):
            captured_prompts.append(prompt)

            class _Response:
                content = json.dumps(llm_payload)

            return _Response()

    monkeypatch.setattr("maverick.platform.llm.get_llm", lambda: _FakeModel())

    parser = StrategyParser()
    result = await parser.parse_with_llm(
        "A crossover strategy using a 15 day and 45 day simple moving average"
    )

    assert result == {**llm_payload, "method": "llm"}
    assert len(captured_prompts) == 1
    assert "sma_cross" in captured_prompts[0]


async def test_parse_with_llm_configured_llm_is_not_called_via_simple_path(
    monkeypatch,
):
    """`parse_simple` never touches `platform.llm` -- confirms the zero-dependency
    method truly has no LLM seam, independent of what's configured."""

    def _boom():
        raise AssertionError("get_llm() must not be called by parse_simple")

    monkeypatch.setattr("maverick.platform.llm.get_llm", _boom)

    parser = StrategyParser()
    result = parser.parse_simple("MACD strategy with standard parameters")

    assert result["strategy_type"] == "macd"
