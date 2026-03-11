"""Natural language strategy parser for VectorBT."""

import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from .templates import STRATEGY_TEMPLATES


class StrategyParser:
    """Parser for converting natural language to VectorBT strategies."""

    def __init__(self, llm: ChatAnthropic | None = None):
        """Initialize strategy parser.

        Args:
            llm: Language model for parsing (optional)
        """
        self.llm = llm
        self.templates = STRATEGY_TEMPLATES

    def parse_simple(self, description: str) -> dict[str, Any]:
        """Parse simple strategy descriptions without LLM.

        Args:
            description: Natural language strategy description

        Returns:
            Strategy configuration
        """
        description_lower = description.lower()

        # Try to match known strategy patterns
        if "sma" in description_lower or "moving average cross" in description_lower:
            return self._parse_sma_strategy(description)
        elif "rsi" in description_lower:
            return self._parse_rsi_strategy(description)
        elif "macd" in description_lower:
            return self._parse_macd_strategy(description)
        elif "bollinger" in description_lower or "band" in description_lower:
            return self._parse_bollinger_strategy(description)
        elif "momentum" in description_lower:
            return self._parse_momentum_strategy(description)
        elif "ema" in description_lower or "exponential" in description_lower:
            return self._parse_ema_strategy(description)
        elif "breakout" in description_lower or "channel" in description_lower:
            return self._parse_breakout_strategy(description)
        elif "mean reversion" in description_lower or "reversion" in description_lower:
            return self._parse_mean_reversion_strategy(description)
        else:
            # Default to momentum if no clear match
            return {
                "strategy_type": "momentum",
                "parameters": self.templates["momentum"]["parameters"],
            }

    def _parse_sma_strategy(self, description: str) -> dict[str, Any]:
        """Parse SMA crossover strategy from description."""
        # Extract numbers from description
        numbers = re.findall(r"\d+", description)

        params = dict(self.templates["sma_cross"]["parameters"])
        if len(numbers) >= 2:
            params["fast_period"] = int(numbers[0])
            params["slow_period"] = int(numbers[1])
        elif len(numbers) == 1:
            params["fast_period"] = int(numbers[0])

        return {
            "strategy_type": "sma_cross",
            "parameters": params,
        }

    def _parse_rsi_strategy(self, description: str) -> dict[str, Any]:
        """Parse RSI strategy from description."""
        numbers = re.findall(r"\d+", description)

        params = dict(self.templates["rsi"]["parameters"])

        # Look for period
        for _i, num in enumerate(numbers):
            num_val = int(num)
            # Period is typically 7-21
            if 5 <= num_val <= 30 and "period" not in params:
                params["period"] = num_val
            # Oversold is typically 20-35
            elif 15 <= num_val <= 35:
                params["oversold"] = num_val
            # Overbought is typically 65-85
            elif 65 <= num_val <= 85:
                params["overbought"] = num_val

        return {
            "strategy_type": "rsi",
            "parameters": params,
        }

    def _parse_macd_strategy(self, description: str) -> dict[str, Any]:
        """Parse MACD strategy from description."""
        numbers = re.findall(r"\d+", description)

        params = dict(self.templates["macd"]["parameters"])
        if len(numbers) >= 3:
            params["fast_period"] = int(numbers[0])
            params["slow_period"] = int(numbers[1])
            params["signal_period"] = int(numbers[2])

        return {
            "strategy_type": "macd",
            "parameters": params,
        }

    def _parse_bollinger_strategy(self, description: str) -> dict[str, Any]:
        """Parse Bollinger Bands strategy from description."""
        numbers = re.findall(r"\d+\.?\d*", description)

        params = dict(self.templates["bollinger"]["parameters"])
        for num in numbers:
            num_val = float(num)
            # Period is typically 10-30
            if num_val == int(num_val) and 5 <= num_val <= 50:
                params["period"] = int(num_val)
            # Std dev is typically 1.5-3.0
            elif 1.0 <= num_val <= 4.0:
                params["std_dev"] = num_val

        return {
            "strategy_type": "bollinger",
            "parameters": params,
        }

    def _parse_momentum_strategy(self, description: str) -> dict[str, Any]:
        """Parse momentum strategy from description."""
        numbers = re.findall(r"\d+\.?\d*", description)

        params = dict(self.templates["momentum"]["parameters"])
        for num in numbers:
            num_val = float(num)
            # Lookback is typically 10-50
            if num_val == int(num_val) and 5 <= num_val <= 100:
                params["lookback"] = int(num_val)
            # Threshold is typically 0.01-0.20
            elif 0.001 <= num_val <= 0.5:
                params["threshold"] = num_val
            # Handle percentage notation (e.g., "5%" -> 0.05)
            elif description[description.find(str(num)) + len(str(num))] == "%":
                params["threshold"] = num_val / 100

        return {
            "strategy_type": "momentum",
            "parameters": params,
        }

    def _parse_ema_strategy(self, description: str) -> dict[str, Any]:
        """Parse EMA crossover strategy from description."""
        numbers = re.findall(r"\d+", description)

        params = dict(self.templates["ema_cross"]["parameters"])
        if len(numbers) >= 2:
            params["fast_period"] = int(numbers[0])
            params["slow_period"] = int(numbers[1])
        elif len(numbers) == 1:
            params["fast_period"] = int(numbers[0])

        return {
            "strategy_type": "ema_cross",
            "parameters": params,
        }

    def _parse_breakout_strategy(self, description: str) -> dict[str, Any]:
        """Parse breakout strategy from description."""
        numbers = re.findall(r"\d+", description)

        params = dict(self.templates["breakout"]["parameters"])
        if len(numbers) >= 2:
            params["lookback"] = int(numbers[0])
            params["exit_lookback"] = int(numbers[1])
        elif len(numbers) == 1:
            params["lookback"] = int(numbers[0])

        return {
            "strategy_type": "breakout",
            "parameters": params,
        }

    def _parse_mean_reversion_strategy(self, description: str) -> dict[str, Any]:
        """Parse mean reversion strategy from description."""
        numbers = re.findall(r"\d+\.?\d*", description)

        params = dict(self.templates["mean_reversion"]["parameters"])
        for num in numbers:
            num_val = float(num)
            if num_val == int(num_val) and 5 <= num_val <= 100:
                params["ma_period"] = int(num_val)
            elif 0.001 <= num_val <= 0.2:
                if "entry" in description.lower():
                    params["entry_threshold"] = num_val
                elif "exit" in description.lower():
                    params["exit_threshold"] = num_val

        return {
            "strategy_type": "mean_reversion",
            "parameters": params,
        }

    async def parse_with_llm(self, description: str) -> dict[str, Any]:
        """Parse complex strategy descriptions using LLM.

        Args:
            description: Natural language strategy description

        Returns:
            Strategy configuration
        """
        if not self.llm:
            # Fall back to simple parsing
            return self.parse_simple(description)

        prompt = PromptTemplate(
            input_variables=["description", "available_strategies"],
            template="""
            Convert this trading strategy description into a structured format.

            Description: {description}

            Available strategy types:
            {available_strategies}

            Return a JSON object with:
            - strategy_type: one of the available types
            - parameters: dictionary of parameters for that strategy
            - entry_logic: description of entry conditions
            - exit_logic: description of exit conditions

            Example response:
            {{
                "strategy_type": "sma_cross",
                "parameters": {{
                    "fast_period": 10,
                    "slow_period": 20
                }},
                "entry_logic": "Buy when fast SMA crosses above slow SMA",
                "exit_logic": "Sell when fast SMA crosses below slow SMA"
            }}
            """,
        )

        available = "\n".join(
            [f"- {k}: {v['description']}" for k, v in self.templates.items()]
        )

        response = await self.llm.ainvoke(
            prompt.format(description=description, available_strategies=available)
        )

        # Parse JSON response
        import json

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fall back to simple parsing
            return self.parse_simple(description)

    def validate_strategy(self, config: dict[str, Any]) -> bool:
        """Validate strategy configuration.

        Args:
            config: Strategy configuration

        Returns:
            True if valid
        """
        strategy_type = config.get("strategy_type")
        if strategy_type not in self.templates:
            return False

        template = self.templates[strategy_type]
        required_params = set(template["parameters"].keys())
        provided_params = set(config.get("parameters", {}).keys())

        # Check if all required parameters are present
        return required_params.issubset(provided_params)
