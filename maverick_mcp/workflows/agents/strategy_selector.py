"""
Strategy Selector Agent for intelligent strategy recommendation.

This agent analyzes market regime and selects the most appropriate trading strategies
based on current market conditions, volatility, and trend characteristics.
"""

import logging
from datetime import datetime
from typing import Any

from maverick_mcp.backtesting.strategies.templates import (
    get_strategy_info,
    list_available_strategies,
)
from maverick_mcp.workflows.state import BacktestingWorkflowState

logger = logging.getLogger(__name__)


class StrategySelectorAgent:
    """Intelligent strategy selector based on market regime analysis."""

    def __init__(self):
        """Initialize strategy selector agent."""
        # Strategy fitness mapping for different market regimes
        self.REGIME_STRATEGY_FITNESS = {
            "trending": {
                "sma_cross": 0.9,
                "ema_cross": 0.85,
                "macd": 0.8,
                "breakout": 0.9,
                "momentum": 0.85,
                "rsi": 0.3,  # Poor for trending markets
                "bollinger": 0.4,
                "mean_reversion": 0.2,
            },
            "ranging": {
                "rsi": 0.9,
                "bollinger": 0.85,
                "mean_reversion": 0.9,
                "sma_cross": 0.3,  # Poor for ranging markets
                "ema_cross": 0.3,
                "breakout": 0.2,
                "momentum": 0.25,
                "macd": 0.5,
            },
            "volatile": {
                "breakout": 0.8,
                "momentum": 0.7,
                "volatility_breakout": 0.9,
                "bollinger": 0.7,
                "sma_cross": 0.4,
                "rsi": 0.6,
                "mean_reversion": 0.5,
                "macd": 0.5,
            },
            "volatile_trending": {
                "breakout": 0.85,
                "momentum": 0.8,
                "volatility_breakout": 0.9,
                "macd": 0.7,
                "ema_cross": 0.6,
                "sma_cross": 0.6,
                "rsi": 0.4,
                "bollinger": 0.6,
            },
            "low_volume": {
                "sma_cross": 0.7,
                "ema_cross": 0.7,
                "rsi": 0.6,
                "mean_reversion": 0.6,
                "breakout": 0.3,  # Poor for low volume
                "momentum": 0.4,
                "bollinger": 0.6,
                "macd": 0.6,
            },
            "low_volume_ranging": {
                "rsi": 0.8,
                "mean_reversion": 0.8,
                "bollinger": 0.7,
                "sma_cross": 0.5,
                "ema_cross": 0.5,
                "breakout": 0.2,
                "momentum": 0.3,
                "macd": 0.4,
            },
            "unknown": {
                # Balanced approach for unknown regimes
                "sma_cross": 0.6,
                "ema_cross": 0.6,
                "rsi": 0.6,
                "macd": 0.6,
                "bollinger": 0.6,
                "momentum": 0.5,
                "breakout": 0.5,
                "mean_reversion": 0.5,
            },
        }

        # Additional fitness adjustments based on market conditions
        self.CONDITION_ADJUSTMENTS = {
            "high_volatility": {
                "rsi": -0.1,
                "breakout": 0.1,
                "volatility_breakout": 0.15,
            },
            "low_volatility": {
                "mean_reversion": 0.1,
                "rsi": 0.1,
                "breakout": -0.1,
            },
            "high_volume": {
                "breakout": 0.1,
                "momentum": 0.1,
                "sma_cross": 0.05,
            },
            "low_volume": {
                "breakout": -0.15,
                "momentum": -0.1,
                "mean_reversion": 0.05,
            },
        }

        logger.info("StrategySelectorAgent initialized")

    async def select_strategies(
        self, state: BacktestingWorkflowState
    ) -> BacktestingWorkflowState:
        """Select optimal strategies based on market regime analysis.

        Args:
            state: Current workflow state with market regime analysis

        Returns:
            Updated state with strategy selection results
        """
        try:
            logger.info(
                f"Selecting strategies for {state['symbol']} in {state['market_regime']} regime"
            )

            # Get available strategies
            available_strategies = list_available_strategies()

            # Calculate strategy fitness scores
            strategy_rankings = self._calculate_strategy_fitness(
                state["market_regime"],
                state["market_conditions"],
                available_strategies,
                state["regime_confidence"],
            )

            # Select top strategies
            selected_strategies = self._select_top_strategies(
                strategy_rankings,
                user_preference=state["requested_strategy"],
                max_strategies=5,  # Limit to top 5 for optimization efficiency
            )

            # Generate strategy candidates with metadata
            candidates = self._generate_strategy_candidates(
                selected_strategies, available_strategies
            )

            # Create selection reasoning
            reasoning = self._generate_selection_reasoning(
                state["market_regime"],
                state["regime_confidence"],
                selected_strategies,
                state["market_conditions"],
            )

            # Calculate selection confidence
            selection_confidence = self._calculate_selection_confidence(
                strategy_rankings,
                selected_strategies,
                state["regime_confidence"],
            )

            # Update state
            state["candidate_strategies"] = candidates
            state["strategy_rankings"] = strategy_rankings
            state["selected_strategies"] = selected_strategies
            state["strategy_selection_reasoning"] = reasoning
            state["strategy_selection_confidence"] = selection_confidence

            # Update workflow status
            state["workflow_status"] = "optimizing"
            state["current_step"] = "strategy_selection_completed"
            state["steps_completed"].append("strategy_selection")

            logger.info(
                f"Strategy selection completed for {state['symbol']}: "
                f"Selected {len(selected_strategies)} strategies with confidence {selection_confidence:.2f}"
            )

            return state

        except Exception as e:
            error_info = {
                "step": "strategy_selection",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbol": state["symbol"],
            }
            state["errors_encountered"].append(error_info)

            # Fallback to basic strategy set
            fallback_strategies = ["sma_cross", "rsi", "macd"]
            state["selected_strategies"] = fallback_strategies
            state["strategy_selection_confidence"] = 0.3
            state["fallback_strategies_used"].append("strategy_selection_fallback")

            logger.error(f"Strategy selection failed for {state['symbol']}: {e}")
            return state

    def _calculate_strategy_fitness(
        self,
        regime: str,
        market_conditions: dict[str, Any],
        available_strategies: list[str],
        regime_confidence: float,
    ) -> dict[str, float]:
        """Calculate fitness scores for all available strategies."""
        fitness_scores = {}

        # Base fitness from regime mapping
        base_fitness = self.REGIME_STRATEGY_FITNESS.get(
            regime, self.REGIME_STRATEGY_FITNESS["unknown"]
        )

        for strategy in available_strategies:
            # Start with base fitness score
            score = base_fitness.get(strategy, 0.5)  # Default to neutral if not defined

            # Apply condition-based adjustments
            score = self._apply_condition_adjustments(
                score, strategy, market_conditions
            )

            # Weight by regime confidence
            # If low confidence, move scores toward neutral (0.5)
            confidence_weight = regime_confidence
            score = score * confidence_weight + 0.5 * (1 - confidence_weight)

            # Ensure score is in valid range
            fitness_scores[strategy] = max(0.0, min(1.0, score))

        return fitness_scores

    def _apply_condition_adjustments(
        self, base_score: float, strategy: str, market_conditions: dict[str, Any]
    ) -> float:
        """Apply market condition adjustments to base fitness score."""
        score = base_score

        # Get relevant conditions
        volatility_regime = market_conditions.get(
            "volatility_regime", "medium_volatility"
        )
        volume_regime = market_conditions.get("volume_regime", "normal_volume")

        # Apply volatility adjustments
        if volatility_regime in self.CONDITION_ADJUSTMENTS:
            adjustment = self.CONDITION_ADJUSTMENTS[volatility_regime].get(strategy, 0)
            score += adjustment

        # Apply volume adjustments
        if volume_regime in self.CONDITION_ADJUSTMENTS:
            adjustment = self.CONDITION_ADJUSTMENTS[volume_regime].get(strategy, 0)
            score += adjustment

        return score

    def _select_top_strategies(
        self,
        strategy_rankings: dict[str, float],
        user_preference: str | None = None,
        max_strategies: int = 5,
    ) -> list[str]:
        """Select top strategies based on fitness scores and user preferences."""
        # Sort strategies by fitness score
        sorted_strategies = sorted(
            strategy_rankings.items(), key=lambda x: x[1], reverse=True
        )

        selected = []

        # Always include user preference if specified and available
        if user_preference and user_preference in strategy_rankings:
            selected.append(user_preference)
            logger.info(f"Including user-requested strategy: {user_preference}")

        # Add top strategies up to limit
        for strategy, score in sorted_strategies:
            if len(selected) >= max_strategies:
                break
            if strategy not in selected and score > 0.4:  # Minimum threshold
                selected.append(strategy)

        # Ensure we have at least 2 strategies
        if len(selected) < 2:
            for strategy, _ in sorted_strategies:
                if strategy not in selected:
                    selected.append(strategy)
                if len(selected) >= 2:
                    break

        return selected

    def _generate_strategy_candidates(
        self, selected_strategies: list[str], available_strategies: list[str]
    ) -> list[dict[str, Any]]:
        """Generate detailed candidate information for selected strategies."""
        candidates = []

        for strategy in selected_strategies:
            if strategy in available_strategies:
                strategy_info = get_strategy_info(strategy)
                candidates.append(
                    {
                        "strategy": strategy,
                        "name": strategy_info.get("name", strategy.title()),
                        "description": strategy_info.get("description", ""),
                        "category": strategy_info.get("category", "unknown"),
                        "parameters": strategy_info.get("parameters", {}),
                        "risk_level": strategy_info.get("risk_level", "medium"),
                        "best_market_conditions": strategy_info.get(
                            "best_conditions", []
                        ),
                    }
                )

        return candidates

    def _generate_selection_reasoning(
        self,
        regime: str,
        regime_confidence: float,
        selected_strategies: list[str],
        market_conditions: dict[str, Any],
    ) -> str:
        """Generate human-readable reasoning for strategy selection."""
        reasoning_parts = []

        # Market regime reasoning
        reasoning_parts.append(
            f"Market regime identified as '{regime}' with {regime_confidence:.1%} confidence."
        )

        # Strategy selection reasoning
        if regime == "trending":
            reasoning_parts.append(
                "In trending markets, trend-following strategies like moving average crossovers "
                "and momentum strategies typically perform well."
            )
        elif regime == "ranging":
            reasoning_parts.append(
                "In ranging markets, mean-reversion strategies like RSI and Bollinger Bands "
                "are favored as they capitalize on price oscillations within a range."
            )
        elif regime == "volatile":
            reasoning_parts.append(
                "In volatile markets, breakout strategies and volatility-based approaches "
                "can capture large price movements effectively."
            )

        # Condition-specific reasoning
        volatility_regime = market_conditions.get("volatility_regime", "")
        if volatility_regime == "high_volatility":
            reasoning_parts.append(
                "High volatility conditions favor strategies that can handle larger price swings."
            )
        elif volatility_regime == "low_volatility":
            reasoning_parts.append(
                "Low volatility conditions favor mean-reversion and range-bound strategies."
            )

        volume_regime = market_conditions.get("volume_regime", "")
        if volume_regime == "low_volume":
            reasoning_parts.append(
                "Low volume conditions reduce reliability of breakout strategies and favor "
                "trend-following approaches with longer timeframes."
            )

        # Selected strategies summary
        reasoning_parts.append(
            f"Selected strategies: {', '.join(selected_strategies)} "
            f"based on their historical performance in similar market conditions."
        )

        return " ".join(reasoning_parts)

    def _calculate_selection_confidence(
        self,
        strategy_rankings: dict[str, float],
        selected_strategies: list[str],
        regime_confidence: float,
    ) -> float:
        """Calculate confidence in strategy selection."""
        if not selected_strategies or not strategy_rankings:
            return 0.0

        # Average fitness of selected strategies
        selected_scores = [strategy_rankings.get(s, 0.5) for s in selected_strategies]
        avg_selected_fitness = sum(selected_scores) / len(selected_scores)

        # Score spread (higher spread = more confident in selection)
        all_scores = list(strategy_rankings.values())
        score_std = (
            sum((s - sum(all_scores) / len(all_scores)) ** 2 for s in all_scores) ** 0.5
        )
        score_spread = (
            score_std / (sum(all_scores) / len(all_scores)) if all_scores else 0
        )

        # Combine factors
        fitness_confidence = avg_selected_fitness  # 0-1
        spread_confidence = min(score_spread, 1.0)  # Normalize spread

        # Weight by regime confidence
        total_confidence = (
            fitness_confidence * 0.5 + spread_confidence * 0.2 + regime_confidence * 0.3
        )

        return max(0.1, min(0.95, total_confidence))

    def get_strategy_compatibility_matrix(self) -> dict[str, dict[str, float]]:
        """Get compatibility matrix showing strategy fitness for each regime."""
        return self.REGIME_STRATEGY_FITNESS.copy()

    def explain_strategy_selection(
        self, regime: str, strategy: str, market_conditions: dict[str, Any]
    ) -> str:
        """Explain why a specific strategy is suitable for given conditions."""
        base_fitness = self.REGIME_STRATEGY_FITNESS.get(regime, {}).get(strategy, 0.5)

        explanations = {
            "sma_cross": {
                "trending": "SMA crossovers excel in trending markets by catching trend changes early.",
                "ranging": "SMA crossovers produce many false signals in ranging markets.",
            },
            "rsi": {
                "ranging": "RSI is ideal for ranging markets, buying oversold and selling overbought levels.",
                "trending": "RSI can remain overbought/oversold for extended periods in strong trends.",
            },
            "breakout": {
                "volatile": "Breakout strategies capitalize on high volatility and strong price moves.",
                "ranging": "Breakout strategies struggle in ranging markets with frequent false breakouts.",
            },
        }

        specific_explanation = explanations.get(strategy, {}).get(regime, "")

        return f"Strategy '{strategy}' has {base_fitness:.1%} fitness for '{regime}' markets. {specific_explanation}"
