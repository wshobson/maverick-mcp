"""
Validator Agent for backtesting results validation and robustness testing.

This agent performs walk-forward analysis, Monte Carlo simulation, and robustness
testing to validate optimization results and provide confidence-scored recommendations.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Any

from maverick_mcp.backtesting import StrategyOptimizer, VectorBTEngine
from maverick_mcp.workflows.state import BacktestingWorkflowState

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Intelligent validator for backtesting results and strategy robustness."""

    def __init__(
        self,
        vectorbt_engine: VectorBTEngine | None = None,
        strategy_optimizer: StrategyOptimizer | None = None,
    ):
        """Initialize validator agent.

        Args:
            vectorbt_engine: VectorBT backtesting engine
            strategy_optimizer: Strategy optimization engine
        """
        self.engine = vectorbt_engine or VectorBTEngine()
        self.optimizer = strategy_optimizer or StrategyOptimizer(self.engine)

        # Validation criteria for different regimes
        self.REGIME_VALIDATION_CRITERIA = {
            "trending": {
                "min_sharpe_ratio": 0.8,
                "max_drawdown_threshold": 0.25,
                "min_total_return": 0.10,
                "min_win_rate": 0.35,
                "stability_threshold": 0.7,
            },
            "ranging": {
                "min_sharpe_ratio": 1.0,  # Higher standard for ranging markets
                "max_drawdown_threshold": 0.15,
                "min_total_return": 0.05,
                "min_win_rate": 0.45,
                "stability_threshold": 0.8,
            },
            "volatile": {
                "min_sharpe_ratio": 0.6,  # Lower expectation in volatile markets
                "max_drawdown_threshold": 0.35,
                "min_total_return": 0.08,
                "min_win_rate": 0.30,
                "stability_threshold": 0.6,
            },
            "volatile_trending": {
                "min_sharpe_ratio": 0.7,
                "max_drawdown_threshold": 0.30,
                "min_total_return": 0.12,
                "min_win_rate": 0.35,
                "stability_threshold": 0.65,
            },
            "low_volume": {
                "min_sharpe_ratio": 0.9,
                "max_drawdown_threshold": 0.20,
                "min_total_return": 0.06,
                "min_win_rate": 0.40,
                "stability_threshold": 0.75,
            },
            "unknown": {
                "min_sharpe_ratio": 0.8,
                "max_drawdown_threshold": 0.20,
                "min_total_return": 0.08,
                "min_win_rate": 0.40,
                "stability_threshold": 0.7,
            },
        }

        # Robustness scoring weights
        self.ROBUSTNESS_WEIGHTS = {
            "walk_forward_consistency": 0.3,
            "parameter_sensitivity": 0.2,
            "monte_carlo_stability": 0.2,
            "out_of_sample_performance": 0.3,
        }

        logger.info("ValidatorAgent initialized")

    async def validate_strategies(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Validate optimized strategies through comprehensive testing.

        Args:
            state: Current workflow state with optimization results

        Returns:
            Updated state with validation results and final recommendations
        """
        start_time = datetime.now()

        try:
            logger.info(
                f"Validating {len(state.best_parameters)} strategies for {state.symbol}"
            )

            # Get validation criteria for current regime
            validation_criteria = self._get_validation_criteria(state.market_regime)

            # Perform validation for each strategy
            walk_forward_results = {}
            monte_carlo_results = {}
            out_of_sample_performance = {}
            robustness_scores = {}
            validation_warnings = []

            for strategy, parameters in state.best_parameters.items():
                try:
                    logger.info(f"Validating {strategy} strategy...")

                    # Walk-forward analysis
                    wf_result = await self._run_walk_forward_analysis(
                        state, strategy, parameters
                    )
                    walk_forward_results[strategy] = wf_result

                    # Monte Carlo simulation
                    mc_result = await self._run_monte_carlo_simulation(
                        state, strategy, parameters
                    )
                    monte_carlo_results[strategy] = mc_result

                    # Out-of-sample testing
                    oos_result = await self._run_out_of_sample_test(
                        state, strategy, parameters
                    )
                    out_of_sample_performance[strategy] = oos_result

                    # Calculate robustness score
                    robustness_score = self._calculate_robustness_score(
                        wf_result, mc_result, oos_result, validation_criteria
                    )
                    robustness_scores[strategy] = robustness_score

                    # Check for validation warnings
                    warnings = self._check_validation_warnings(
                        strategy, wf_result, mc_result, oos_result, validation_criteria
                    )
                    validation_warnings.extend(warnings)

                    logger.info(
                        f"Validated {strategy}: robustness score {robustness_score:.2f}"
                    )

                except Exception as e:
                    logger.error(f"Failed to validate {strategy}: {e}")
                    robustness_scores[strategy] = 0.0
                    validation_warnings.append(
                        f"{strategy}: Validation failed - {str(e)}"
                    )

            # Generate final recommendations
            final_ranking = self._generate_final_ranking(
                state.best_parameters, robustness_scores, state.strategy_rankings
            )

            # Select recommended strategy
            recommended_strategy, recommendation_confidence = (
                self._select_recommended_strategy(
                    final_ranking, robustness_scores, state.regime_confidence
                )
            )

            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(
                recommended_strategy,
                walk_forward_results,
                monte_carlo_results,
                validation_criteria,
            )

            # Update state
            state.walk_forward_results = walk_forward_results
            state.monte_carlo_results = monte_carlo_results
            state.out_of_sample_performance = out_of_sample_performance
            state.robustness_score = robustness_scores
            state.validation_warnings = validation_warnings
            state.final_strategy_ranking = final_ranking
            state.recommended_strategy = recommended_strategy
            state.recommended_parameters = state.best_parameters.get(
                recommended_strategy, {}
            )
            state.recommendation_confidence = recommendation_confidence
            state.risk_assessment = risk_assessment

            # Update workflow status
            state.workflow_status = "completed"
            state.current_step = "validation_completed"
            state.steps_completed.append("strategy_validation")

            # Record total execution time
            total_execution_time = (datetime.now() - start_time).total_seconds() * 1000
            state.total_execution_time_ms = (
                state.regime_analysis_time_ms
                + state.optimization_time_ms
                + total_execution_time
            )

            logger.info(
                f"Strategy validation completed for {state.symbol}: "
                f"Recommended {recommended_strategy} with confidence {recommendation_confidence:.2f}"
            )

            return state

        except Exception as e:
            error_info = {
                "step": "strategy_validation",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbol": state.symbol,
            }
            state.errors_encountered.append(error_info)

            # Fallback recommendation
            if state.best_parameters:
                fallback_strategy = list(state.best_parameters.keys())[0]
                state.recommended_strategy = fallback_strategy
                state.recommended_parameters = state.best_parameters[fallback_strategy]
                state.recommendation_confidence = 0.3
                state.fallback_strategies_used.append("validation_fallback")

            logger.error(f"Strategy validation failed for {state.symbol}: {e}")
            return state

    def _get_validation_criteria(self, regime: str) -> dict[str, Any]:
        """Get validation criteria based on market regime."""
        return self.REGIME_VALIDATION_CRITERIA.get(
            regime, self.REGIME_VALIDATION_CRITERIA["unknown"]
        )

    async def _run_walk_forward_analysis(
        self, state: BacktestingWorkflowState, strategy: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Run walk-forward analysis for strategy validation."""
        try:
            # Calculate walk-forward windows
            start_dt = datetime.strptime(state.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(state.end_date, "%Y-%m-%d")
            total_days = (end_dt - start_dt).days

            # Use appropriate window sizes based on data length
            if total_days > 500:  # ~2 years
                window_size = 252  # 1 year
                step_size = 63  # 3 months
            elif total_days > 250:  # ~1 year
                window_size = 126  # 6 months
                step_size = 42  # 6 weeks
            else:
                window_size = 63  # 3 months
                step_size = 21  # 3 weeks

            # Run walk-forward analysis using the optimizer
            wf_result = await self.optimizer.walk_forward_analysis(
                symbol=state.symbol,
                strategy_type=strategy,
                parameters=parameters,
                start_date=state.start_date,
                end_date=state.end_date,
                window_size=window_size,
                step_size=step_size,
            )

            return wf_result

        except Exception as e:
            logger.error(f"Walk-forward analysis failed for {strategy}: {e}")
            return {"error": str(e), "consistency_score": 0.0}

    async def _run_monte_carlo_simulation(
        self, state: BacktestingWorkflowState, strategy: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation for strategy validation."""
        try:
            # First run a backtest to get base results
            backtest_result = await self.engine.run_backtest(
                symbol=state.symbol,
                strategy_type=strategy,
                parameters=parameters,
                start_date=state.start_date,
                end_date=state.end_date,
                initial_capital=state.initial_capital,
            )

            # Run Monte Carlo simulation
            mc_result = await self.optimizer.monte_carlo_simulation(
                backtest_results=backtest_result,
                num_simulations=500,  # Reduced for performance
            )

            return mc_result

        except Exception as e:
            logger.error(f"Monte Carlo simulation failed for {strategy}: {e}")
            return {"error": str(e), "stability_score": 0.0}

    async def _run_out_of_sample_test(
        self, state: BacktestingWorkflowState, strategy: str, parameters: dict[str, Any]
    ) -> dict[str, float]:
        """Run out-of-sample testing on holdout data."""
        try:
            # Use last 30% of data as out-of-sample
            start_dt = datetime.strptime(state.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(state.end_date, "%Y-%m-%d")
            total_days = (end_dt - start_dt).days

            oos_days = int(total_days * 0.3)
            oos_start = end_dt - timedelta(days=oos_days)

            # Run backtest on out-of-sample period
            oos_result = await self.engine.run_backtest(
                symbol=state.symbol,
                strategy_type=strategy,
                parameters=parameters,
                start_date=oos_start.strftime("%Y-%m-%d"),
                end_date=state.end_date,
                initial_capital=state.initial_capital,
            )

            return {
                "total_return": oos_result["metrics"]["total_return"],
                "sharpe_ratio": oos_result["metrics"]["sharpe_ratio"],
                "max_drawdown": oos_result["metrics"]["max_drawdown"],
                "win_rate": oos_result["metrics"]["win_rate"],
                "total_trades": oos_result["metrics"]["total_trades"],
            }

        except Exception as e:
            logger.error(f"Out-of-sample test failed for {strategy}: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

    def _calculate_robustness_score(
        self,
        wf_result: dict[str, Any],
        mc_result: dict[str, Any],
        oos_result: dict[str, float],
        validation_criteria: dict[str, Any],
    ) -> float:
        """Calculate overall robustness score for a strategy."""
        scores = {}

        # Walk-forward consistency score
        if "consistency_score" in wf_result:
            scores["walk_forward_consistency"] = wf_result["consistency_score"]
        elif "error" not in wf_result and "periods" in wf_result:
            # Calculate consistency from period results
            period_returns = [
                p.get("total_return", 0) for p in wf_result.get("periods", [])
            ]
            if period_returns:
                # Lower std deviation relative to mean = higher consistency
                mean_return = statistics.mean(period_returns)
                std_return = (
                    statistics.stdev(period_returns) if len(period_returns) > 1 else 0
                )
                consistency = max(0, 1 - (std_return / max(abs(mean_return), 0.01)))
                scores["walk_forward_consistency"] = min(1.0, consistency)
            else:
                scores["walk_forward_consistency"] = 0.0
        else:
            scores["walk_forward_consistency"] = 0.0

        # Parameter sensitivity (inverse of standard error)
        scores["parameter_sensitivity"] = 0.7  # Default moderate sensitivity

        # Monte Carlo stability
        if "stability_score" in mc_result:
            scores["monte_carlo_stability"] = mc_result["stability_score"]
        elif "error" not in mc_result and "percentiles" in mc_result:
            # Calculate stability from percentile spread
            percentiles = mc_result["percentiles"]
            p10 = percentiles.get("10", 0)
            p90 = percentiles.get("90", 0)
            median = percentiles.get("50", 0)

            if median != 0:
                stability = 1 - abs(p90 - p10) / abs(median)
                scores["monte_carlo_stability"] = max(0, min(1, stability))
            else:
                scores["monte_carlo_stability"] = 0.0
        else:
            scores["monte_carlo_stability"] = 0.0

        # Out-of-sample performance score
        oos_score = 0.0
        if oos_result["sharpe_ratio"] >= validation_criteria["min_sharpe_ratio"]:
            oos_score += 0.3
        if (
            abs(oos_result["max_drawdown"])
            <= validation_criteria["max_drawdown_threshold"]
        ):
            oos_score += 0.3
        if oos_result["total_return"] >= validation_criteria["min_total_return"]:
            oos_score += 0.2
        if oos_result["win_rate"] >= validation_criteria["min_win_rate"]:
            oos_score += 0.2

        scores["out_of_sample_performance"] = oos_score

        # Calculate weighted robustness score
        robustness_score = sum(
            scores[component] * self.ROBUSTNESS_WEIGHTS[component]
            for component in self.ROBUSTNESS_WEIGHTS
        )

        return max(0.0, min(1.0, robustness_score))

    def _check_validation_warnings(
        self,
        strategy: str,
        wf_result: dict[str, Any],
        mc_result: dict[str, Any],
        oos_result: dict[str, float],
        validation_criteria: dict[str, Any],
    ) -> list[str]:
        """Check for validation warnings and concerns."""
        warnings = []

        # Walk-forward analysis warnings
        if "error" in wf_result:
            warnings.append(f"{strategy}: Walk-forward analysis failed")
        elif (
            wf_result.get("consistency_score", 0)
            < validation_criteria["stability_threshold"]
        ):
            warnings.append(
                f"{strategy}: Low walk-forward consistency ({wf_result.get('consistency_score', 0):.2f})"
            )

        # Monte Carlo warnings
        if "error" in mc_result:
            warnings.append(f"{strategy}: Monte Carlo simulation failed")
        elif mc_result.get("stability_score", 0) < 0.6:
            warnings.append(f"{strategy}: High Monte Carlo variability")

        # Out-of-sample warnings
        if oos_result["total_trades"] < 5:
            warnings.append(
                f"{strategy}: Very few out-of-sample trades ({oos_result['total_trades']})"
            )

        if oos_result["sharpe_ratio"] < validation_criteria["min_sharpe_ratio"]:
            warnings.append(
                f"{strategy}: Low out-of-sample Sharpe ratio ({oos_result['sharpe_ratio']:.2f})"
            )

        if (
            abs(oos_result["max_drawdown"])
            > validation_criteria["max_drawdown_threshold"]
        ):
            warnings.append(
                f"{strategy}: High out-of-sample drawdown ({oos_result['max_drawdown']:.2f})"
            )

        return warnings

    def _generate_final_ranking(
        self,
        best_parameters: dict[str, dict[str, Any]],
        robustness_scores: dict[str, float],
        strategy_rankings: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Generate final ranked recommendations."""
        rankings = []

        for strategy in best_parameters.keys():
            robustness = robustness_scores.get(strategy, 0.0)
            fitness = strategy_rankings.get(strategy, 0.5)

            # Combined score: 60% robustness, 40% initial fitness
            combined_score = robustness * 0.6 + fitness * 0.4

            rankings.append(
                {
                    "strategy": strategy,
                    "robustness_score": robustness,
                    "fitness_score": fitness,
                    "combined_score": combined_score,
                    "parameters": best_parameters[strategy],
                    "recommendation": self._get_recommendation_level(combined_score),
                }
            )

        # Sort by combined score
        rankings.sort(key=lambda x: x["combined_score"], reverse=True)

        return rankings

    def _get_recommendation_level(self, combined_score: float) -> str:
        """Get recommendation level based on combined score."""
        if combined_score >= 0.8:
            return "Highly Recommended"
        elif combined_score >= 0.6:
            return "Recommended"
        elif combined_score >= 0.4:
            return "Acceptable"
        else:
            return "Not Recommended"

    def _select_recommended_strategy(
        self,
        final_ranking: list[dict[str, Any]],
        robustness_scores: dict[str, float],
        regime_confidence: float,
    ) -> tuple[str, float]:
        """Select the final recommended strategy and calculate confidence."""
        if not final_ranking:
            return "sma_cross", 0.1  # Fallback

        # Select top strategy
        top_strategy = final_ranking[0]["strategy"]
        top_score = final_ranking[0]["combined_score"]

        # Calculate recommendation confidence
        confidence_factors = []

        # Score-based confidence
        confidence_factors.append(top_score)

        # Robustness-based confidence
        robustness = robustness_scores.get(top_strategy, 0.0)
        confidence_factors.append(robustness)

        # Regime confidence factor
        confidence_factors.append(regime_confidence)

        # Score separation from second-best
        if len(final_ranking) > 1:
            score_gap = top_score - final_ranking[1]["combined_score"]
            separation_confidence = min(score_gap * 2, 1.0)  # Scale to 0-1
            confidence_factors.append(separation_confidence)
        else:
            confidence_factors.append(0.5)  # Moderate confidence for single option

        # Calculate overall confidence
        recommendation_confidence = sum(confidence_factors) / len(confidence_factors)
        recommendation_confidence = max(0.1, min(0.95, recommendation_confidence))

        return top_strategy, recommendation_confidence

    def _perform_risk_assessment(
        self,
        recommended_strategy: str,
        walk_forward_results: dict[str, dict[str, Any]],
        monte_carlo_results: dict[str, dict[str, Any]],
        validation_criteria: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform comprehensive risk assessment of recommended strategy."""
        wf_result = walk_forward_results.get(recommended_strategy, {})
        mc_result = monte_carlo_results.get(recommended_strategy, {})

        risk_assessment = {
            "overall_risk_level": "Medium",
            "key_risks": [],
            "risk_mitigation": [],
            "confidence_intervals": {},
            "worst_case_scenario": {},
        }

        # Analyze walk-forward results for risk patterns
        if "periods" in wf_result:
            periods = wf_result["periods"]
            negative_periods = [p for p in periods if p.get("total_return", 0) < 0]

            if len(negative_periods) / len(periods) > 0.4:
                risk_assessment["key_risks"].append("High frequency of losing periods")
                risk_assessment["overall_risk_level"] = "High"

            max_period_loss = min([p.get("total_return", 0) for p in periods])
            if max_period_loss < -0.15:
                risk_assessment["key_risks"].append(
                    f"Severe single-period loss: {max_period_loss:.1%}"
                )

        # Analyze Monte Carlo results
        if "percentiles" in mc_result:
            percentiles = mc_result["percentiles"]
            worst_case = percentiles.get("5", 0)  # 5th percentile

            risk_assessment["worst_case_scenario"] = {
                "return_5th_percentile": worst_case,
                "probability": 0.05,
                "description": f"5% chance of returns below {worst_case:.1%}",
            }

            risk_assessment["confidence_intervals"] = {
                "90_percent_range": f"{percentiles.get('5', 0):.1%} to {percentiles.get('95', 0):.1%}",
                "median_return": f"{percentiles.get('50', 0):.1%}",
            }

        # Risk mitigation recommendations
        risk_assessment["risk_mitigation"] = [
            "Use position sizing based on volatility",
            "Implement stop-loss orders",
            "Monitor strategy performance regularly",
            "Consider diversification across multiple strategies",
        ]

        return risk_assessment
