"""
Template for creating new stock screening strategies.

Copy this file and modify it to create new screening strategies quickly.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from maverick_mcp.core.technical_analysis import (
    calculate_atr,
    calculate_rsi,
    calculate_sma,
)
from maverick_mcp.data.models import Stock, get_db
from maverick_mcp.providers.stock_data import StockDataProvider
from maverick_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class YourScreeningStrategy:
    """
    Your custom screening strategy.

    This strategy identifies stocks that meet specific criteria
    based on technical indicators and price action.
    """

    def __init__(
        self,
        min_price: float = 10.0,
        min_volume: int = 1_000_000,
        lookback_days: int = 90,
    ):
        """
        Initialize the screening strategy.

        Args:
            min_price: Minimum stock price to consider
            min_volume: Minimum average daily volume
            lookback_days: Number of days to analyze
        """
        self.min_price = min_price
        self.min_volume = min_volume
        self.lookback_days = lookback_days
        self.stock_provider = StockDataProvider()

    def calculate_score(self, symbol: str, data: pd.DataFrame) -> float:
        """
        Calculate a composite score for the stock.

        Args:
            symbol: Stock symbol
            data: Historical price data

        Returns:
            Score between 0 and 100
        """
        score = 0.0

        try:
            # Price above moving averages
            sma_20 = calculate_sma(data, 20).iloc[-1]
            sma_50 = calculate_sma(data, 50).iloc[-1]
            current_price = data["Close"].iloc[-1]

            if current_price > sma_20:
                score += 20
            if current_price > sma_50:
                score += 15

            # RSI in optimal range (not overbought/oversold)
            rsi = calculate_rsi(data, 14).iloc[-1]
            if 40 <= rsi <= 70:
                score += 20
            elif 30 <= rsi <= 80:
                score += 10

            # MACD bullish (using pandas_ta as alternative)
            try:
                import pandas_ta as ta

                macd = ta.macd(data["close"])
                if macd["MACD_12_26_9"].iloc[-1] > macd["MACDs_12_26_9"].iloc[-1]:
                    score += 15
            except ImportError:
                # Skip MACD if pandas_ta not available
                pass

            # Volume increasing
            avg_volume_recent = data["Volume"].iloc[-5:].mean()
            avg_volume_prior = data["Volume"].iloc[-20:-5].mean()
            if avg_volume_recent > avg_volume_prior * 1.2:
                score += 15

            # Price momentum
            price_change_1m = (current_price / data["Close"].iloc[-20] - 1) * 100
            if price_change_1m > 10:
                score += 15
            elif price_change_1m > 5:
                score += 10

            logger.debug(
                f"Score calculated for {symbol}: {score}",
                extra={
                    "symbol": symbol,
                    "price": current_price,
                    "rsi": rsi,
                    "score": score,
                },
            )

        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}")
            score = 0.0

        return min(score, 100.0)

    def screen_stocks(
        self,
        symbols: list[str] | None = None,
        min_score: float = 70.0,
    ) -> list[dict[str, Any]]:
        """
        Screen stocks based on the strategy criteria.

        Args:
            symbols: List of symbols to screen (None for all)
            min_score: Minimum score to include in results

        Returns:
            List of stocks meeting criteria with scores
        """
        results = []
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime(
            "%Y-%m-%d"
        )

        # Get list of symbols to screen
        if symbols is None:
            # Get all active stocks from database
            db = next(get_db())
            try:
                stocks = db.query(Stock).filter(Stock.is_active).all()
                symbols = [stock.symbol for stock in stocks]
            finally:
                db.close()

        logger.info(f"Screening {len(symbols)} stocks")

        # Screen each stock
        for symbol in symbols:
            try:
                # Get historical data
                data = self.stock_provider.get_stock_data(symbol, start_date, end_date)

                if len(data) < 50:  # Need enough data for indicators
                    continue

                # Check basic criteria
                current_price = data["Close"].iloc[-1]
                avg_volume = data["Volume"].iloc[-20:].mean()

                if current_price < self.min_price or avg_volume < self.min_volume:
                    continue

                # Calculate score
                score = self.calculate_score(symbol, data)

                if score >= min_score:
                    # Calculate additional metrics
                    atr = calculate_atr(data, 14).iloc[-1]
                    price_change_5d = (
                        data["Close"].iloc[-1] / data["Close"].iloc[-5] - 1
                    ) * 100

                    result = {
                        "symbol": symbol,
                        "score": round(score, 2),
                        "price": round(current_price, 2),
                        "volume": int(avg_volume),
                        "atr": round(atr, 2),
                        "price_change_5d": round(price_change_5d, 2),
                        "rsi": round(calculate_rsi(data, 14).iloc[-1], 2),
                        "above_sma_20": current_price
                        > calculate_sma(data, 20).iloc[-1],
                        "above_sma_50": current_price
                        > calculate_sma(data, 50).iloc[-1],
                    }

                    results.append(result)
                    logger.info(f"Stock passed screening: {symbol} (score: {score})")

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                continue

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Screening complete: {len(results)} stocks found")
        return results

    def get_entry_exit_levels(
        self, symbol: str, data: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calculate entry, stop loss, and target levels.

        Args:
            symbol: Stock symbol
            data: Historical price data

        Returns:
            Dictionary with entry, stop, and target levels
        """
        current_price = data["Close"].iloc[-1]
        atr = calculate_atr(data, 14).iloc[-1]

        # Find recent support/resistance
        recent_low = data["Low"].iloc[-20:].min()

        # Calculate levels
        entry = current_price
        stop_loss = max(current_price - (2 * atr), recent_low * 0.98)
        target1 = current_price + (2 * atr)
        target2 = current_price + (3 * atr)

        # Ensure minimum risk/reward
        risk = entry - stop_loss
        reward = target1 - entry
        if reward / risk < 2:
            target1 = entry + (2 * risk)
            target2 = entry + (3 * risk)

        return {
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "target1": round(target1, 2),
            "target2": round(target2, 2),
            "risk_reward_ratio": round(reward / risk, 2),
        }
