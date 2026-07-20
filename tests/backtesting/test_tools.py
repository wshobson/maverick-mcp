"""Tests for `maverick.backtesting.tools`/`tools_ml`. No `importorskip`: `tools.py` never
imports vectorbt/sklearn (see its module docstring), so this file collects and runs identically
on a base install."""

from typing import Any

import pytest
from fastmcp import Client, FastMCP

from maverick.backtesting import tools
from maverick.backtesting.types import (
    EnsembleBacktestResult,
    EnsembleSummary,
    MarketRegimeAnalysis,
    MLBacktestResult,
    MLTrainingResult,
    MonteCarloResult,
    OptimizationResult,
    PortfolioBacktestMetrics,
    PortfolioBacktestResult,
    RunBacktestResult,
    SimpleBacktestMetrics,
    StrategyCatalog,
    StrategyComparisonResult,
    WalkForwardResult,
)


def _run_backtest_result() -> RunBacktestResult:
    from maverick.backtesting.types import (
        BacktestAnalysis,
        BacktestMetrics,
        RiskAssessment,
        TradeQuality,
    )

    metrics = BacktestMetrics(
        total_return=0.1,
        annual_return=0.1,
        sharpe_ratio=1.0,
        sortino_ratio=1.0,
        calmar_ratio=1.0,
        max_drawdown=-0.1,
        win_rate=0.5,
        profit_factor=1.5,
        expectancy=0.01,
        total_trades=10,
        winning_trades=5,
        losing_trades=5,
        avg_win=1.0,
        avg_loss=-1.0,
        best_trade=2.0,
        worst_trade=-2.0,
        avg_duration=5.0,
        kelly_criterion=0.1,
        recovery_factor=1.0,
        risk_reward_ratio=1.0,
    )
    return RunBacktestResult(
        symbol="AAPL",
        strategy="sma_cross",
        parameters={"fast_period": 10, "slow_period": 20},
        metrics=metrics,
        trades=[],
        equity_curve={"2023-01-01": 10000.0},
        drawdown_series={"2023-01-01": 0.0},
        start_date="2023-01-01",
        end_date="2023-06-01",
        initial_capital=10000.0,
        analysis=BacktestAnalysis(
            performance_grade="B",
            risk_assessment=RiskAssessment(
                risk_level="Low",
                max_drawdown=0.1,
                sortino_ratio=1.0,
                calmar_ratio=1.0,
                recovery_factor=1.0,
                risk_adjusted_return=1.0,
                downside_protection="Good",
            ),
            trade_quality=TradeQuality(
                quality="Good", total_trades=10, frequency="Low"
            ),
            strengths=["Good"],
            weaknesses=["None"],
            recommendations=["Keep going"],
            summary="Solid.",
        ),
    )


def _catalog() -> StrategyCatalog:
    from maverick.backtesting.types import StrategyCatalogEntry

    entry = StrategyCatalogEntry(
        type="sma_cross",
        name="SMA Crossover",
        description="desc",
        default_parameters={"fast_period": 10},
        optimization_ranges={"fast_period": [5, 10]},
    )
    return StrategyCatalog(
        available_strategies={"sma_cross": entry}, total_count=1, categories={}
    )


def _optimization_result() -> OptimizationResult:
    return OptimizationResult(
        symbol="AAPL",
        strategy="sma_cross",
        optimization_metric="sharpe_ratio",
        best_parameters={"fast_period": 10},
        best_metric_value=1.0,
        top_results=[],
        total_combinations_tested=9,
        valid_combinations=9,
    )


def _walk_forward_result() -> WalkForwardResult:
    return WalkForwardResult(
        symbol="AAPL",
        strategy="sma_cross",
        periods_tested=1,
        average_return=0.1,
        average_sharpe=1.0,
        average_drawdown=-0.1,
        consistency=1.0,
        walk_forward_results=[],
        summary="ok",
    )


def _monte_carlo_result() -> MonteCarloResult:
    return MonteCarloResult(
        num_simulations=100,
        expected_return=0.1,
        return_std=0.05,
        return_percentiles={"p5": -0.1, "p50": 0.1, "p95": 0.3},
        expected_drawdown=-0.1,
        drawdown_std=0.05,
        drawdown_percentiles={"p5": -0.3, "p50": -0.1, "p95": 0.0},
        probability_profit=0.6,
        var_95=-0.1,
        summary="ok",
    )


def _comparison_result() -> StrategyComparisonResult:
    from maverick.backtesting.types import StrategyComparisonRow

    row = StrategyComparisonRow(
        strategy="sma_cross",
        parameters={},
        total_return=0.1,
        sharpe_ratio=1.0,
        max_drawdown=0.1,
        win_rate=0.5,
        profit_factor=1.5,
        total_trades=10,
        grade="B",
        rank=1,
    )
    return StrategyComparisonResult(
        rankings=[row],
        best_overall=row,
        best_return=row,
        best_sharpe=row,
        best_drawdown=row,
        best_win_rate=row,
        summary="ok",
    )


def _portfolio_result() -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        portfolio_metrics=PortfolioBacktestMetrics(
            symbols_tested=2,
            total_return=0.1,
            average_sharpe=1.0,
            max_drawdown=-0.1,
            total_trades=20,
        ),
        individual_results=[],
        summary="ok",
    )


def _ml_backtest_result() -> MLBacktestResult:
    return MLBacktestResult(
        metrics=SimpleBacktestMetrics(
            total_return=0.1,
            annual_return=0.1,
            sharpe_ratio=1.0,
            max_drawdown=-0.1,
            win_rate=0.5,
            total_trades=10,
            profit_factor=1.5,
        ),
        trades=[],
        equity_curve={},
        drawdown_series={},
        ml_metrics={"strategy_type": "ml_predictor"},
    )


def _ml_training_result() -> MLTrainingResult:
    return MLTrainingResult(
        symbol="AAPL",
        model_type="random_forest",
        training_period="2022-01-01 to 2023-01-01",
        data_points=300,
        target_periods=5,
        return_threshold=0.02,
        model_parameters={"n_estimators": 100},
        training_metrics={},
    )


def _regime_result() -> MarketRegimeAnalysis:
    return MarketRegimeAnalysis(
        symbol="AAPL",
        analysis_period="2022-01-01 to 2023-01-01",
        method="hmm",
        n_regimes=3,
        regime_names={0: "Bear", 1: "Sideways", 2: "Bull"},
        current_regime=1,
        regime_counts={0: 1, 1: 1, 2: 1},
        regime_percentages={0: 33.3, 1: 33.3, 2: 33.3},
        average_regime_durations={0: 1.0, 1: 1.0, 2: 1.0},
        recent_regime_history=[],
        total_regime_switches=2,
    )


def _ensemble_result() -> EnsembleBacktestResult:
    return EnsembleBacktestResult(
        ensemble_summary=EnsembleSummary(
            symbols_tested=2,
            base_strategies=["sma_cross"],
            weighting_method="performance",
            average_return=0.1,
            total_trades=20,
            average_trades_per_symbol=10.0,
        ),
        individual_results=[],
        final_strategy_weights={},
        strategy_performance_analysis={},
    )


class StubService:
    """Fake matching `BacktestingService`'s full 11-method public surface."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple]] = {}
        self.results: dict[str, Any] = {
            "run_backtest": _run_backtest_result(),
            "optimize_strategy": _optimization_result(),
            "walk_forward_analysis": _walk_forward_result(),
            "monte_carlo_simulation": _monte_carlo_result(),
            "compare_strategies": _comparison_result(),
            "list_strategies": _catalog(),
            "backtest_portfolio": _portfolio_result(),
            "run_ml_strategy_backtest": _ml_backtest_result(),
            "train_ml_predictor": _ml_training_result(),
            "analyze_market_regimes": _regime_result(),
            "create_strategy_ensemble": _ensemble_result(),
        }
        self.raise_on: dict[str, Exception] = {}

    async def _call(self, name: str, args: tuple) -> Any:
        self.calls.setdefault(name, []).append(args)
        if name in self.raise_on:
            raise self.raise_on[name]
        return self.results[name]

    async def run_backtest(self, symbol, **kwargs):
        return await self._call("run_backtest", (symbol, kwargs))

    async def optimize_strategy(self, symbol, **kwargs):
        return await self._call("optimize_strategy", (symbol, kwargs))

    async def walk_forward_analysis(self, symbol, **kwargs):
        return await self._call("walk_forward_analysis", (symbol, kwargs))

    async def monte_carlo_simulation(self, symbol, **kwargs):
        return await self._call("monte_carlo_simulation", (symbol, kwargs))

    async def compare_strategies(self, symbol, **kwargs):
        return await self._call("compare_strategies", (symbol, kwargs))

    async def list_strategies(self):
        return await self._call("list_strategies", ())

    async def backtest_portfolio(self, symbols, **kwargs):
        return await self._call("backtest_portfolio", (symbols, kwargs))

    async def run_ml_strategy_backtest(self, symbol, **kwargs):
        return await self._call("run_ml_strategy_backtest", (symbol, kwargs))

    async def train_ml_predictor(self, symbol, **kwargs):
        return await self._call("train_ml_predictor", (symbol, kwargs))

    async def analyze_market_regimes(self, symbol, **kwargs):
        return await self._call("analyze_market_regimes", (symbol, kwargs))

    async def create_strategy_ensemble(self, symbols, **kwargs):
        return await self._call("create_strategy_ensemble", (symbols, kwargs))


@pytest.fixture
def stub_service():
    stub = StubService()
    tools.configure(stub)
    yield stub


# ---------------------------------------------------------------------------
# unconfigured service
# ---------------------------------------------------------------------------


async def test_unconfigured_service_returns_configure_error_payload():
    tools.configure(None)  # type: ignore[arg-type]

    result = await tools.backtesting_list_strategies()

    assert result == {
        "status": "error",
        "error": "backtesting.tools: configure(service) was not called",
    }


# ---------------------------------------------------------------------------
# success / error payload per tool
# ---------------------------------------------------------------------------


async def test_run_backtest_success(stub_service):
    result = await tools.backtesting_run_backtest("aapl", fast_period=10)

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"
    assert stub_service.calls["run_backtest"] == [
        (
            "aapl",
            {
                "strategy": "sma_cross",
                "start_date": None,
                "end_date": None,
                "initial_capital": 10000.0,
                "fast_period": 10,
                "slow_period": None,
                "period": None,
                "oversold": None,
                "overbought": None,
                "signal_period": None,
                "std_dev": None,
                "lookback": None,
                "threshold": None,
                "z_score_threshold": None,
                "breakout_factor": None,
            },
        )
    ]


async def test_run_backtest_error_payload(stub_service):
    stub_service.raise_on["run_backtest"] = ValueError("insufficient history")

    result = await tools.backtesting_run_backtest("AAPL")

    assert result == {"status": "error", "error": "insufficient history"}


async def test_optimize_strategy_success(stub_service):
    result = await tools.backtesting_optimize_strategy("AAPL")

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"


async def test_optimize_strategy_error_payload(stub_service):
    stub_service.raise_on["optimize_strategy"] = ValueError("boom")

    result = await tools.backtesting_optimize_strategy("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_walk_forward_analysis_success(stub_service):
    result = await tools.backtesting_walk_forward_analysis("AAPL")

    assert result["status"] == "success"
    assert result["periods_tested"] == 1


async def test_walk_forward_analysis_error_payload(stub_service):
    stub_service.raise_on["walk_forward_analysis"] = ValueError("boom")

    result = await tools.backtesting_walk_forward_analysis("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_monte_carlo_simulation_success(stub_service):
    result = await tools.backtesting_monte_carlo_simulation("AAPL")

    assert result["status"] == "success"
    assert result["num_simulations"] == 100


async def test_monte_carlo_simulation_error_payload(stub_service):
    stub_service.raise_on["monte_carlo_simulation"] = ValueError("boom")

    result = await tools.backtesting_monte_carlo_simulation("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_compare_strategies_success(stub_service):
    result = await tools.backtesting_compare_strategies(
        "AAPL", strategies=["sma_cross"]
    )

    assert result["status"] == "success"
    assert stub_service.calls["compare_strategies"][0][1]["strategies"] == ["sma_cross"]


async def test_compare_strategies_error_payload(stub_service):
    stub_service.raise_on["compare_strategies"] = ValueError("boom")

    result = await tools.backtesting_compare_strategies("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_list_strategies_success(stub_service):
    result = await tools.backtesting_list_strategies()

    assert result["status"] == "success"
    assert result["total_count"] == 1


async def test_list_strategies_error_payload(stub_service):
    stub_service.raise_on["list_strategies"] = ValueError("boom")

    result = await tools.backtesting_list_strategies()

    assert result == {"status": "error", "error": "boom"}


async def test_backtest_portfolio_success(stub_service):
    result = await tools.backtesting_backtest_portfolio(["AAPL", "MSFT"])

    assert result["status"] == "success"
    assert stub_service.calls["backtest_portfolio"][0][0] == ["AAPL", "MSFT"]


async def test_backtest_portfolio_error_payload(stub_service):
    stub_service.raise_on["backtest_portfolio"] = ValueError(
        "No symbols could be backtested"
    )

    result = await tools.backtesting_backtest_portfolio(["AAPL"])

    assert result == {"status": "error", "error": "No symbols could be backtested"}


async def test_run_ml_strategy_backtest_success(stub_service):
    result = await tools.backtesting_run_ml_strategy_backtest("AAPL")

    assert result["status"] == "success"
    assert result["ml_metrics"]["strategy_type"] == "ml_predictor"


async def test_run_ml_strategy_backtest_error_payload(stub_service):
    stub_service.raise_on["run_ml_strategy_backtest"] = ValueError("boom")

    result = await tools.backtesting_run_ml_strategy_backtest("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_train_ml_predictor_success(stub_service):
    result = await tools.backtesting_train_ml_predictor("AAPL")

    assert result["status"] == "success"
    assert result["symbol"] == "AAPL"


async def test_train_ml_predictor_error_payload(stub_service):
    stub_service.raise_on["train_ml_predictor"] = ValueError("boom")

    result = await tools.backtesting_train_ml_predictor("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_analyze_market_regimes_success(stub_service):
    result = await tools.backtesting_analyze_market_regimes("AAPL")

    assert result["status"] == "success"
    assert result["n_regimes"] == 3


async def test_analyze_market_regimes_error_payload(stub_service):
    stub_service.raise_on["analyze_market_regimes"] = ValueError("boom")

    result = await tools.backtesting_analyze_market_regimes("AAPL")

    assert result == {"status": "error", "error": "boom"}


async def test_create_strategy_ensemble_success(stub_service):
    result = await tools.backtesting_create_strategy_ensemble(["AAPL", "MSFT"])

    assert result["status"] == "success"
    assert stub_service.calls["create_strategy_ensemble"][0][0] == ["AAPL", "MSFT"]


async def test_create_strategy_ensemble_error_payload(stub_service):
    stub_service.raise_on["create_strategy_ensemble"] = ValueError("boom")

    result = await tools.backtesting_create_strategy_ensemble(["AAPL"])

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# backtesting_parse_strategy: no BacktestingService involved (see tools.py's
# module docstring on why tool 12 is still gated by the same extra guard).
# ---------------------------------------------------------------------------


async def test_parse_strategy_success_degrades_to_simple_when_no_llm_configured(
    monkeypatch,
):
    def _raise_not_configured():
        raise ValueError("No LLM configured; set LLM_PROVIDER ...")

    monkeypatch.setattr("maverick.platform.llm.get_llm", _raise_not_configured)

    result = await tools.backtesting_parse_strategy(
        "Buy when RSI is below 30 and sell when above 70"
    )

    assert result == {
        "success": True,
        "strategy": {
            "strategy_type": "rsi",
            "parameters": {"period": 14, "oversold": 30, "overbought": 70},
        },
        "method": "simple_degraded",
        "message": "Successfully parsed as rsi strategy",
    }


async def test_parse_strategy_reports_llm_method_on_success(monkeypatch):
    pytest.importorskip("langchain_core")

    class _FakeModel:
        async def ainvoke(self, _prompt):
            class _Response:
                content = (
                    '{"strategy_type": "macd", '
                    '"parameters": {"fast_period": 12, "slow_period": 26, '
                    '"signal_period": 9}}'
                )

            return _Response()

    monkeypatch.setattr("maverick.platform.llm.get_llm", lambda: _FakeModel())

    result = await tools.backtesting_parse_strategy("MACD with standard settings")

    assert result["success"] is True
    assert result["method"] == "llm"
    assert result["strategy"]["strategy_type"] == "macd"


async def test_parse_strategy_error_payload_on_unexpected_exception(monkeypatch):
    def _raise_unexpected(_self, _description):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "maverick.backtesting.strategies.parser.StrategyParser.parse_simple",
        _raise_unexpected,
    )
    monkeypatch.setattr(
        "maverick.platform.llm.get_llm",
        lambda: (_ for _ in ()).throw(ValueError("not configured")),
    )

    result = await tools.backtesting_parse_strategy("anything")

    assert result == {"status": "error", "error": "boom"}


# ---------------------------------------------------------------------------
# register: attaches 12 tools, all honestly read-only
# ---------------------------------------------------------------------------


_EXPECTED_TOOL_NAMES = {
    "backtesting_run_backtest",
    "backtesting_optimize_strategy",
    "backtesting_walk_forward_analysis",
    "backtesting_monte_carlo_simulation",
    "backtesting_compare_strategies",
    "backtesting_list_strategies",
    "backtesting_backtest_portfolio",
    "backtesting_run_ml_strategy_backtest",
    "backtesting_train_ml_predictor",
    "backtesting_analyze_market_regimes",
    "backtesting_create_strategy_ensemble",
    "backtesting_parse_strategy",
}


async def test_register_attaches_twelve_tools(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    registered = await mcp.list_tools()

    assert {tool.name for tool in registered} == _EXPECTED_TOOL_NAMES


async def test_register_marks_every_tool_read_only(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    for name in _EXPECTED_TOOL_NAMES:
        tool = await mcp.get_tool(name)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True


async def test_register_in_memory_client_round_trips_list_strategies(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool("backtesting_list_strategies", {})

    assert result.data["status"] == "success"
    assert result.data["total_count"] == 1


async def test_register_in_memory_client_round_trips_run_backtest(stub_service):
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "backtesting_run_backtest", {"symbol": "AAPL", "fast_period": 10}
        )

    assert result.data["status"] == "success"
    assert result.data["symbol"] == "AAPL"
    assert stub_service.calls["run_backtest"][0][0] == "AAPL"


async def test_register_in_memory_client_round_trips_parse_strategy(
    stub_service, monkeypatch
):
    monkeypatch.setattr(
        "maverick.platform.llm.get_llm",
        lambda: (_ for _ in ()).throw(ValueError("not configured")),
    )
    mcp = FastMCP("test")
    tools.register(mcp)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "backtesting_parse_strategy",
            {"description": "MACD strategy with standard parameters"},
        )

    assert result.data["success"] is True
    assert result.data["method"] == "simple_degraded"
    assert result.data["strategy"]["strategy_type"] == "macd"
