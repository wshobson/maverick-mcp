"""Pure backtest result analysis and Monte Carlo simulation. Third layer:
imports config and types.

Ports two things, both operating on `engine.py`'s typed `BacktestResult`
output rather than the legacy raw dicts:

- `maverick_mcp/backtesting/analysis.py`'s `BacktestAnalyzer`: grading,
  risk assessment, trade-quality analysis, strengths/weaknesses,
  recommendations, summaries, and `compare_strategies` (`analyze`,
  `_grade_performance`, `_assess_risk`, `_analyze_trades`,
  `_identify_strengths`, `_identify_weaknesses`,
  `_generate_recommendations`, `_generate_summary`, `compare_strategies`).
- `monte_carlo_simulation`, which despite the module name does NOT live in
  the legacy `analysis.py` -- it lives in
  `maverick_mcp/backtesting/optimization.py`'s
  `StrategyOptimizer.monte_carlo_simulation`. It is ported here because the
  task brief assigns "the monte-carlo simulation logic" to this module, and
  because it operates on the same `BacktestResult`/`TradeRecord` shapes as
  everything else here rather than on `StrategyOptimizer`'s
  engine-orchestration responsibilities (walk-forward analysis, param-grid
  generation), which are out of scope for this task.

Not ported: `BacktestAnalyzer.run_vectorbt_backtest`. It is a second,
lower-fidelity backtest execution path (producing `SimpleBacktestMetrics`
dict-shaped trades, per `types.py`'s module docstring) used only by the
ML-strategy tools (`run_ml_strategy_backtest`, `create_strategy_ensemble`),
whose underlying ML strategy code has not been ported into `maverick/` yet.
`engine.run_backtest` is this domain's one execution path going forward.

Two legacy behaviors are NOT preserved because they don't fit a typed
return contract: `compare_strategies([])` and `monte_carlo_simulation([])`
used to return `{"error": "..."}"` dicts; here they raise `ValueError`
instead, since `StrategyComparisonResult`/`MonteCarloResult` have no error
variant to construct from an empty input.
"""

import numpy as np

from maverick.backtesting.types import (
    BacktestAnalysis,
    BacktestMetrics,
    BacktestResult,
    MonteCarloResult,
    RiskAssessment,
    StrategyComparisonResult,
    StrategyComparisonRow,
    TradeQuality,
    TradeRecord,
)

_DEFAULT_CONFIDENCE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _grade_performance(metrics: BacktestMetrics) -> str:
    """Port of `BacktestAnalyzer._grade_performance`: a 100-point rubric
    (Sharpe 30, total return 25, win rate 20, max drawdown 15, profit
    factor 10) converted to an A-F letter grade."""
    score = 0

    sharpe = metrics.sharpe_ratio
    if sharpe >= 2.0:
        score += 30
    elif sharpe >= 1.5:
        score += 25
    elif sharpe >= 1.0:
        score += 20
    elif sharpe >= 0.5:
        score += 10
    else:
        score += 5

    total_return = metrics.total_return
    if total_return >= 0.50:
        score += 25
    elif total_return >= 0.30:
        score += 20
    elif total_return >= 0.15:
        score += 15
    elif total_return >= 0.05:
        score += 10
    elif total_return > 0:
        score += 5

    win_rate = metrics.win_rate
    if win_rate >= 0.60:
        score += 20
    elif win_rate >= 0.50:
        score += 15
    elif win_rate >= 0.40:
        score += 10
    else:
        score += 5

    max_dd = abs(metrics.max_drawdown)
    if max_dd <= 0.10:
        score += 15
    elif max_dd <= 0.20:
        score += 12
    elif max_dd <= 0.30:
        score += 8
    elif max_dd <= 0.40:
        score += 4

    profit_factor = metrics.profit_factor
    if profit_factor >= 2.0:
        score += 10
    elif profit_factor >= 1.5:
        score += 8
    elif profit_factor >= 1.2:
        score += 5
    elif profit_factor > 1.0:
        score += 3

    percentage = score
    if percentage >= 90:
        return "A"
    if percentage >= 80:
        return "B"
    if percentage >= 70:
        return "C"
    if percentage >= 60:
        return "D"
    return "F"


def _assess_risk(metrics: BacktestMetrics) -> RiskAssessment:
    """Port of `BacktestAnalyzer._assess_risk`."""
    max_dd = abs(metrics.max_drawdown)
    sortino = metrics.sortino_ratio
    sharpe = metrics.sharpe_ratio

    risk_level = "Low"
    if max_dd > 0.40:
        risk_level = "Very High"
    elif max_dd > 0.30:
        risk_level = "High"
    elif max_dd > 0.20:
        risk_level = "Medium"
    elif max_dd > 0.10:
        risk_level = "Low-Medium"

    return RiskAssessment(
        risk_level=risk_level,
        max_drawdown=max_dd,
        sortino_ratio=sortino,
        calmar_ratio=metrics.calmar_ratio,
        recovery_factor=metrics.recovery_factor,
        risk_adjusted_return=sortino if sortino > 0 else sharpe,
        downside_protection="Good"
        if sortino > 1.5
        else "Moderate"
        if sortino > 0.5
        else "Poor",
    )


def _analyze_trades(
    trades: list[TradeRecord], metrics: BacktestMetrics
) -> TradeQuality:
    """Port of `BacktestAnalyzer._analyze_trades`."""
    if not trades:
        return TradeQuality(quality="No trades", total_trades=0, frequency="None")

    total_trades = metrics.total_trades
    win_rate = metrics.win_rate

    if total_trades < 10:
        frequency = "Very Low"
    elif total_trades < 50:
        frequency = "Low"
    elif total_trades < 100:
        frequency = "Moderate"
    elif total_trades < 200:
        frequency = "High"
    else:
        frequency = "Very High"

    if win_rate >= 0.60 and metrics.profit_factor >= 1.5:
        quality = "Excellent"
    elif win_rate >= 0.50 and metrics.profit_factor >= 1.2:
        quality = "Good"
    elif win_rate >= 0.40:
        quality = "Average"
    else:
        quality = "Poor"

    return TradeQuality(
        quality=quality,
        total_trades=total_trades,
        frequency=frequency,
        win_rate=win_rate,
        avg_win=metrics.avg_win,
        avg_loss=metrics.avg_loss,
        best_trade=metrics.best_trade,
        worst_trade=metrics.worst_trade,
        avg_duration_days=metrics.avg_duration,
        risk_reward_ratio=metrics.risk_reward_ratio,
    )


def _identify_strengths(metrics: BacktestMetrics) -> list[str]:
    """Port of `BacktestAnalyzer._identify_strengths`."""
    strengths = []
    if metrics.sharpe_ratio >= 1.5:
        strengths.append("Excellent risk-adjusted returns")
    if metrics.win_rate >= 0.60:
        strengths.append("High win rate")
    if abs(metrics.max_drawdown) <= 0.15:
        strengths.append("Low maximum drawdown")
    if metrics.profit_factor >= 1.5:
        strengths.append("Strong profit factor")
    if metrics.sortino_ratio >= 2.0:
        strengths.append("Excellent downside protection")
    if metrics.calmar_ratio >= 1.0:
        strengths.append("Good return vs drawdown ratio")
    if metrics.recovery_factor >= 3.0:
        strengths.append("Quick drawdown recovery")
    if metrics.total_return >= 0.30:
        strengths.append("High total returns")
    return strengths if strengths else ["Consistent performance"]


def _identify_weaknesses(metrics: BacktestMetrics) -> list[str]:
    """Port of `BacktestAnalyzer._identify_weaknesses`."""
    weaknesses = []
    if metrics.sharpe_ratio < 0.5:
        weaknesses.append("Poor risk-adjusted returns")
    if metrics.win_rate < 0.40:
        weaknesses.append("Low win rate")
    if abs(metrics.max_drawdown) > 0.30:
        weaknesses.append("High maximum drawdown")
    if metrics.profit_factor < 1.0:
        weaknesses.append("Unprofitable trades overall")
    if metrics.total_trades < 10:
        weaknesses.append("Insufficient trade signals")
    if metrics.sortino_ratio < 0:
        weaknesses.append("Poor downside protection")
    if metrics.total_return < 0:
        weaknesses.append("Negative returns")
    return weaknesses if weaknesses else ["Room for optimization"]


def _generate_recommendations(metrics: BacktestMetrics) -> list[str]:
    """Port of `BacktestAnalyzer._generate_recommendations`."""
    recommendations = []
    if abs(metrics.max_drawdown) > 0.25:
        recommendations.append("Implement tighter stop-loss rules to reduce drawdowns")
    if metrics.win_rate < 0.45:
        recommendations.append("Refine entry signals to improve win rate")
    if metrics.total_trades < 20:
        recommendations.append(
            "Consider more sensitive parameters for increased signals"
        )
    elif metrics.total_trades > 200:
        recommendations.append("Filter signals to reduce overtrading")
    if metrics.risk_reward_ratio < 1.5:
        recommendations.append("Adjust exit strategy for better risk-reward ratio")
    if metrics.profit_factor < 1.2:
        recommendations.append(
            "Focus on cutting losses quicker and letting winners run"
        )
    if metrics.sharpe_ratio < 1.0:
        recommendations.append("Consider position sizing based on volatility")
    kelly = metrics.kelly_criterion
    if 0 < kelly < 0.25:
        recommendations.append(
            f"Consider position size of {kelly * 100:.1f}% based on Kelly Criterion"
        )
    return (
        recommendations
        if recommendations
        else ["Strategy performing well, consider live testing"]
    )


def _generate_summary(metrics: BacktestMetrics) -> str:
    """Port of `BacktestAnalyzer._generate_summary`."""
    total_return = metrics.total_return * 100
    sharpe = metrics.sharpe_ratio
    max_dd = abs(metrics.max_drawdown) * 100
    win_rate = metrics.win_rate * 100
    total_trades = metrics.total_trades

    summary = (
        f"The strategy generated a {total_return:.1f}% return with a Sharpe "
        f"ratio of {sharpe:.2f}. "
    )
    summary += (
        f"Maximum drawdown was {max_dd:.1f}% with a {win_rate:.1f}% win rate "
        f"across {total_trades} trades. "
    )
    if sharpe >= 1.5 and max_dd <= 20:
        summary += "Overall performance is excellent with strong risk-adjusted returns."
    elif sharpe >= 1.0 and max_dd <= 30:
        summary += "Performance is good with acceptable risk levels."
    elif sharpe >= 0.5:
        summary += "Performance is moderate and could benefit from optimization."
    else:
        summary += "Performance needs significant improvement before live trading."
    return summary


def analyze(result: BacktestResult) -> BacktestAnalysis:
    """Port of `BacktestAnalyzer.analyze`."""
    metrics = result.metrics
    trades = result.trades
    return BacktestAnalysis(
        performance_grade=_grade_performance(metrics),
        risk_assessment=_assess_risk(metrics),
        trade_quality=_analyze_trades(trades, metrics),
        strengths=_identify_strengths(metrics),
        weaknesses=_identify_weaknesses(metrics),
        recommendations=_generate_recommendations(metrics),
        summary=_generate_summary(metrics),
    )


def _generate_comparison_summary(comparisons: list[StrategyComparisonRow]) -> str:
    """Port of `BacktestAnalyzer._generate_comparison_summary`."""
    best = comparisons[0]
    summary = (
        f"The best performing strategy is {best.strategy} "
        f"with a Sharpe ratio of {best.sharpe_ratio:.2f} "
        f"and total return of {best.total_return * 100:.1f}%. "
    )
    if len(comparisons) > 1:
        summary += f"It outperformed {len(comparisons) - 1} other strategies tested."
    return summary


def compare_strategies(results: list[BacktestResult]) -> StrategyComparisonResult:
    """Port of `BacktestAnalyzer.compare_strategies`.

    Raises `ValueError` for an empty `results` list (see module docstring:
    the legacy `{"error": ...}` dict return has no typed equivalent).
    """
    if not results:
        raise ValueError("compare_strategies requires at least one result")

    comparisons = [
        StrategyComparisonRow(
            strategy=result.strategy,
            parameters=result.parameters,
            total_return=result.metrics.total_return,
            sharpe_ratio=result.metrics.sharpe_ratio,
            max_drawdown=abs(result.metrics.max_drawdown),
            win_rate=result.metrics.win_rate,
            profit_factor=result.metrics.profit_factor,
            total_trades=result.metrics.total_trades,
            grade=_grade_performance(result.metrics),
            rank=0,
        )
        for result in results
    ]
    comparisons.sort(key=lambda row: row.sharpe_ratio, reverse=True)
    for rank, row in enumerate(comparisons, 1):
        row.rank = rank

    best_return = max(comparisons, key=lambda row: row.total_return)
    best_sharpe = max(comparisons, key=lambda row: row.sharpe_ratio)
    best_drawdown = min(comparisons, key=lambda row: row.max_drawdown)
    best_win_rate = max(comparisons, key=lambda row: row.win_rate)

    return StrategyComparisonResult(
        rankings=comparisons,
        best_overall=comparisons[0],
        best_return=best_return,
        best_sharpe=best_sharpe,
        best_drawdown=best_drawdown,
        best_win_rate=best_win_rate,
        summary=_generate_comparison_summary(comparisons),
    )


def _generate_mc_summary(
    expected_return: float, var_95: float, prob_profit: float
) -> str:
    """Port of `StrategyOptimizer._generate_mc_summary`."""
    summary = (
        f"Monte Carlo simulation shows {expected_return * 100:.1f}% expected "
        f"return with {prob_profit * 100:.1f}% probability of profit. "
    )
    summary += f"95% Value at Risk is {abs(var_95) * 100:.1f}%. "
    if prob_profit >= 0.8 and expected_return > 0.10:
        summary += "Strategy shows strong probabilistic edge."
    elif prob_profit >= 0.6 and expected_return > 0:
        summary += "Strategy shows positive expectancy with moderate confidence."
    else:
        summary += "Strategy may not have sufficient edge for live trading."
    return summary


def monte_carlo_simulation(
    trades: list[TradeRecord],
    *,
    num_simulations: int = 1000,
    confidence_levels: list[float] | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """Bootstrap-resample `trades`' returns to estimate a distribution of
    total return / max drawdown outcomes.

    Port of `StrategyOptimizer.monte_carlo_simulation` (which, despite the
    name, lives in the legacy `optimization.py`, not `analysis.py` -- see
    module docstring). One deliberate deviation: the legacy version reads
    `np.random.choice` off numpy's mutable global random state (callers
    seed it beforehand via `np.random.seed(...)`); this version takes an
    explicit `seed` and constructs a private `np.random.default_rng(seed)`
    instead, so simulation determinism doesn't depend on -- or mutate --
    global interpreter state. `var_95` is the percentile at
    `confidence_levels[0]` (5% by default), matching the legacy code
    exactly: it is positional, not "whichever level is nearest 0.05".

    Raises `ValueError` if `trades` is empty (see module docstring: the
    legacy `{"error": ...}` dict return has no typed equivalent).
    """
    if not trades:
        raise ValueError("monte_carlo_simulation requires at least one trade")
    levels = (
        list(confidence_levels)
        if confidence_levels
        else list(_DEFAULT_CONFIDENCE_LEVELS)
    )

    trade_returns = np.array([t.return_ for t in trades], dtype=float)
    rng = np.random.default_rng(seed)

    simulated_returns = np.empty(num_simulations, dtype=float)
    simulated_drawdowns = np.empty(num_simulations, dtype=float)
    for i in range(num_simulations):
        sampled = rng.choice(trade_returns, size=len(trade_returns), replace=True)
        cumulative = np.cumprod(1 + sampled)
        simulated_returns[i] = cumulative[-1] - 1
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        simulated_drawdowns[i] = drawdown.min()

    return_percentiles = np.percentile(simulated_returns, np.array(levels) * 100)
    drawdown_percentiles = np.percentile(simulated_drawdowns, np.array(levels) * 100)
    level_keys = [f"p{int(cl * 100)}" for cl in levels]
    prob_profit = float(np.mean(simulated_returns > 0))
    expected_return = float(np.mean(simulated_returns))
    var_95 = float(return_percentiles[0])

    return MonteCarloResult(
        num_simulations=num_simulations,
        expected_return=expected_return,
        return_std=float(np.std(simulated_returns)),
        return_percentiles=dict(
            zip(level_keys, (float(v) for v in return_percentiles), strict=True)
        ),
        expected_drawdown=float(np.mean(simulated_drawdowns)),
        drawdown_std=float(np.std(simulated_drawdowns)),
        drawdown_percentiles=dict(
            zip(level_keys, (float(v) for v in drawdown_percentiles), strict=True)
        ),
        probability_profit=prob_profit,
        var_95=var_95,
        summary=_generate_mc_summary(expected_return, var_95, prob_profit),
    )
