"""
Risk management tools for position sizing, stop loss calculation, and portfolio risk analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from maverick_mcp.agents.base import PersonaAwareTool
from maverick_mcp.core.technical_analysis import calculate_atr
from maverick_mcp.providers.stock_data import StockDataProvider

logger = logging.getLogger(__name__)


class PositionSizeInput(BaseModel):
    """Input for position sizing calculations."""

    account_size: float = Field(description="Total account size in dollars")
    entry_price: float = Field(description="Planned entry price")
    stop_loss_price: float = Field(description="Stop loss price")
    risk_percentage: float = Field(
        default=2.0, description="Percentage of account to risk (default 2%)"
    )


class TechnicalStopsInput(BaseModel):
    """Input for technical stop calculations."""

    symbol: str = Field(description="Stock symbol")
    lookback_days: int = Field(default=20, description="Days to look back for analysis")
    atr_multiplier: float = Field(
        default=2.0, description="ATR multiplier for stop distance"
    )


class RiskMetricsInput(BaseModel):
    """Input for portfolio risk metrics."""

    symbols: list[str] = Field(description="List of symbols in portfolio")
    weights: list[float] | None = Field(
        default=None, description="Portfolio weights (equal weight if not provided)"
    )
    lookback_days: int = Field(
        default=252, description="Days for correlation calculation"
    )


class PositionSizeTool(PersonaAwareTool):
    """Calculate position size based on risk management rules."""

    name: str = "calculate_position_size"
    description: str = (
        "Calculate position size based on account risk, with Kelly Criterion "
        "and persona adjustments"
    )
    args_schema: type[BaseModel] = PositionSizeInput

    def _run(
        self,
        account_size: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percentage: float = 2.0,
    ) -> str:
        """Calculate position size synchronously."""
        try:
            # Basic risk calculation
            risk_amount = account_size * (risk_percentage / 100)
            price_risk = abs(entry_price - stop_loss_price)

            if price_risk == 0:
                return "Error: Entry and stop loss prices cannot be the same"

            # Calculate base position size
            base_shares = risk_amount / price_risk
            base_position_value = base_shares * entry_price

            # Apply persona adjustments
            adjusted_shares = self.adjust_for_risk(base_shares, "position_size")
            adjusted_value = adjusted_shares * entry_price

            # Calculate Kelly fraction if persona is set
            kelly_fraction = 0.25  # Default conservative Kelly
            if self.persona:
                risk_factor = sum(self.persona.risk_tolerance) / 100
                kelly_fraction = self._calculate_kelly_fraction(risk_factor)

            kelly_shares = base_shares * kelly_fraction
            kelly_value = kelly_shares * entry_price

            # Ensure position doesn't exceed max allocation
            max_position_pct = self.persona.position_size_max if self.persona else 0.10
            max_position_value = account_size * max_position_pct

            final_shares = min(adjusted_shares, kelly_shares)
            final_value = final_shares * entry_price

            if final_value > max_position_value:
                final_shares = max_position_value / entry_price
                final_value = max_position_value

            result = {
                "status": "success",
                "position_sizing": {
                    "recommended_shares": int(final_shares),
                    "position_value": round(final_value, 2),
                    "position_percentage": round((final_value / account_size) * 100, 2),
                    "risk_amount": round(risk_amount, 2),
                    "price_risk_per_share": round(price_risk, 2),
                    "r_multiple_target": round(
                        2.0 * price_risk / entry_price * 100, 2
                    ),  # 2R target
                },
                "calculations": {
                    "base_shares": int(base_shares),
                    "base_position_value": round(base_position_value, 2),
                    "kelly_shares": int(kelly_shares),
                    "kelly_value": round(kelly_value, 2),
                    "persona_adjusted_shares": int(adjusted_shares),
                    "persona_adjusted_value": round(adjusted_value, 2),
                    "kelly_fraction": round(kelly_fraction, 3),
                    "max_allowed_value": round(max_position_value, 2),
                },
            }

            # Add persona insights if available
            if self.persona:
                result["persona_insights"] = {
                    "investor_type": self.persona.name,
                    "risk_tolerance": self.persona.risk_tolerance,
                    "max_position_size": f"{self.persona.position_size_max * 100:.1f}%",
                    "suitable_for_profile": final_value <= max_position_value,
                }

            # Format for return
            formatted = self.format_for_persona(result)
            return str(formatted)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return f"Error calculating position size: {str(e)}"


class TechnicalStopsTool(PersonaAwareTool):
    """Calculate stop loss levels based on technical analysis."""

    name: str = "calculate_technical_stops"
    description: str = (
        "Calculate stop loss levels using ATR, support levels, and moving averages"
    )
    args_schema: type[BaseModel] = TechnicalStopsInput

    def _run(
        self, symbol: str, lookback_days: int = 20, atr_multiplier: float = 2.0
    ) -> str:
        """Calculate technical stops synchronously."""
        try:
            provider = StockDataProvider()

            # Get price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max(lookback_days * 2, 100))

            df = provider.get_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_cache=True,
            )

            if df.empty:
                return f"Error: No price data available for {symbol}"

            # Calculate technical levels
            current_price = df["Close"].iloc[-1]

            # ATR-based stop
            atr = calculate_atr(df, period=14)
            atr_value = atr.iloc[-1]
            atr_stop = current_price - (atr_value * atr_multiplier)

            # Support-based stops
            recent_lows = df["Low"].rolling(window=lookback_days).min()
            support_level = recent_lows.iloc[-1]

            # Moving average stops
            ma_20 = float(df["Close"].rolling(window=20).mean().iloc[-1])
            ma_50 = (
                float(df["Close"].rolling(window=50).mean().iloc[-1])
                if len(df) >= 50
                else None
            )

            # Swing low stop (lowest low in recent period)
            swing_low = df["Low"].iloc[-lookback_days:].min()

            # Apply persona adjustments
            if self.persona:
                atr_multiplier = self.adjust_for_risk(atr_multiplier, "stop_loss")
                atr_stop = current_price - (atr_value * atr_multiplier)

            stops = {
                "current_price": round(current_price, 2),
                "atr_stop": round(atr_stop, 2),
                "support_stop": round(support_level, 2),
                "swing_low_stop": round(swing_low, 2),
                "ma_20_stop": round(ma_20, 2),
                "ma_50_stop": round(ma_50, 2) if ma_50 else None,
                "atr_value": round(atr_value, 2),
                "stop_distances": {
                    "atr_stop_pct": round(
                        ((current_price - atr_stop) / current_price) * 100, 2
                    ),
                    "support_stop_pct": round(
                        ((current_price - support_level) / current_price) * 100, 2
                    ),
                    "swing_low_pct": round(
                        ((current_price - swing_low) / current_price) * 100, 2
                    ),
                },
            }

            # Recommend stop based on persona
            if self.persona:
                if self.persona.name == "Conservative":
                    recommended = max(atr_stop, ma_20)  # Tighter stop
                elif self.persona.name == "Day Trader":
                    recommended = atr_stop  # ATR-based for volatility
                else:
                    recommended = min(support_level, atr_stop)  # Balance
            else:
                recommended = atr_stop

            stops["recommended_stop"] = round(recommended, 2)
            stops["recommended_stop_pct"] = round(
                ((current_price - recommended) / current_price) * 100, 2
            )

            result = {
                "status": "success",
                "symbol": symbol,
                "technical_stops": stops,
                "analysis_period": lookback_days,
                "atr_multiplier": atr_multiplier,
            }

            # Format for persona
            formatted = self.format_for_persona(result)
            return str(formatted)

        except Exception as e:
            logger.error(f"Error calculating technical stops for {symbol}: {e}")
            return f"Error calculating technical stops: {str(e)}"


class RiskMetricsTool(PersonaAwareTool):
    """Calculate portfolio risk metrics including correlations and VaR."""

    name: str = "calculate_risk_metrics"
    description: str = (
        "Calculate portfolio risk metrics including correlation, beta, and VaR"
    )
    args_schema: type[BaseModel] = RiskMetricsInput  # type: ignore[assignment]

    def _run(
        self,
        symbols: list[str],
        weights: list[float] | None = None,
        lookback_days: int = 252,
    ) -> str:
        """Calculate risk metrics synchronously."""
        try:
            if not symbols:
                return "Error: No symbols provided"

            provider = StockDataProvider()

            # If no weights provided, use equal weight
            if weights is None:
                weights = [1.0 / len(symbols)] * len(symbols)
            elif len(weights) != len(symbols):
                return "Error: Number of weights must match number of symbols"

            # Normalize weights
            weights_array = np.array(weights)
            weights = list(weights_array / weights_array.sum())

            # Get price data for all symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)

            price_data = {}
            returns_data = {}

            for symbol in symbols:
                df = provider.get_stock_data(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    use_cache=True,
                )
                if not df.empty:
                    price_data[symbol] = df["Close"]
                    returns_data[symbol] = df["Close"].pct_change().dropna()

            if not returns_data:
                return "Error: No price data available for any symbols"

            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data).dropna()

            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()

            # Calculate portfolio metrics
            portfolio_returns = (returns_df * weights[: len(returns_df.columns)]).sum(
                axis=1
            )
            portfolio_std = portfolio_returns.std() * np.sqrt(252)  # Annualized

            # Calculate VaR (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)

            # Calculate portfolio beta (vs SPY)
            spy_df = provider.get_stock_data(
                "SPY",
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                use_cache=True,
            )
            if not spy_df.empty:
                spy_returns = spy_df["Close"].pct_change().dropna()
                # Align dates
                common_dates = portfolio_returns.index.intersection(spy_returns.index)
                if len(common_dates) > 0:
                    portfolio_beta = (
                        portfolio_returns[common_dates].cov(spy_returns[common_dates])
                        / spy_returns[common_dates].var()
                    )
                else:
                    portfolio_beta = None
            else:
                portfolio_beta = None

            # Build result
            result = {
                "status": "success",
                "portfolio_metrics": {
                    "annualized_volatility": round(portfolio_std * 100, 2),
                    "value_at_risk_95": round(var_95 * 100, 2),
                    "portfolio_beta": round(portfolio_beta, 2)
                    if portfolio_beta
                    else None,
                    "avg_correlation": round(
                        correlation_matrix.values[
                            np.triu_indices_from(correlation_matrix.values, k=1)
                        ].mean(),
                        3,
                    ),
                },
                "correlations": correlation_matrix.to_dict(),
                "weights": {
                    symbol: round(weight, 3)
                    for symbol, weight in zip(
                        symbols[: len(weights)], weights, strict=False
                    )
                },
                "risk_assessment": self._assess_portfolio_risk(
                    portfolio_std, var_95, correlation_matrix
                ),
            }

            # Format for persona
            formatted = self.format_for_persona(result)
            return str(formatted)

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return f"Error calculating risk metrics: {str(e)}"

    def _assess_portfolio_risk(
        self, volatility: float, var: float, correlation_matrix: pd.DataFrame
    ) -> dict[str, Any]:
        """Assess portfolio risk level."""
        risk_level = "Low"
        warnings = []

        # Check volatility
        if volatility > 0.25:  # 25% annual vol
            risk_level = "High"
            warnings.append("High portfolio volatility")
        elif volatility > 0.15:
            risk_level = "Moderate"

        # Check VaR
        if abs(var) > 0.10:  # 10% VaR
            warnings.append("High Value at Risk")

        # Check correlation
        avg_corr = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].mean()
        if avg_corr > 0.7:
            warnings.append("High correlation between holdings")

        return {
            "risk_level": risk_level,
            "warnings": warnings,
            "diversification_score": round(1 - avg_corr, 2),
        }
