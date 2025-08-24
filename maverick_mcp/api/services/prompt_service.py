"""
Prompt service for MaverickMCP API.

Handles trading and investing prompts for technical analysis and stock screening.
Extracted from server.py to improve code organization and maintainability.
"""

from .base_service import BaseService


class PromptService(BaseService):
    """
    Service class for prompt operations.

    Provides trading and investing prompts for technical analysis and stock screening.
    """

    def register_tools(self):
        """Register prompt tools with MCP."""

        @self.mcp.prompt()
        def technical_analysis(ticker: str, timeframe: str = "daily") -> str:
            """
            Generate a comprehensive technical analysis prompt for a given stock.

            Args:
                ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
                timeframe: Analysis timeframe - "daily", "weekly", or "monthly"

            Returns:
                Formatted prompt for technical analysis
            """
            return self._technical_analysis_prompt(ticker, timeframe)

        @self.mcp.prompt()
        def stock_screening_report(strategy: str = "momentum") -> str:
            """
            Generate a stock screening analysis prompt based on specified strategy.

            Args:
                strategy: Screening strategy - "momentum", "value", "growth", "quality", or "dividend"

            Returns:
                Formatted prompt for stock screening analysis
            """
            return self._stock_screening_prompt(strategy)

    def _technical_analysis_prompt(self, ticker: str, timeframe: str = "daily") -> str:
        """Generate technical analysis prompt implementation."""
        # Validate inputs
        valid_timeframes = ["daily", "weekly", "monthly"]
        if timeframe not in valid_timeframes:
            timeframe = "daily"

        ticker = ticker.upper().strip()

        prompt = f"""
# Technical Analysis Request for {ticker}

Please provide a comprehensive technical analysis for **{ticker}** using {timeframe} timeframe data.

## Analysis Requirements:

### 1. Price Action Analysis
- Current price level and recent price movement
- Key support and resistance levels
- Trend direction (bullish, bearish, or sideways)
- Chart patterns (if any): triangles, flags, head & shoulders, etc.

### 2. Technical Indicators Analysis
Please analyze these key indicators:

**Moving Averages:**
- 20, 50, 200-period moving averages
- Price position relative to moving averages
- Moving average convergence/divergence signals

**Momentum Indicators:**
- RSI (14-period): overbought/oversold conditions
- MACD: signal line crossovers and histogram
- Stochastic oscillator: %K and %D levels

**Volume Analysis:**
- Recent volume trends
- Volume confirmation of price moves
- On-balance volume (OBV) trend

### 3. Market Context
- Overall market trend and {ticker}'s correlation
- Sector performance and relative strength
- Recent news or events that might impact the stock

### 4. Trading Recommendations
Based on the technical analysis, please provide:
- **Entry points**: Optimal buy/sell levels
- **Stop loss**: Risk management levels
- **Target prices**: Profit-taking levels
- **Time horizon**: Short-term, medium-term, or long-term outlook
- **Risk assessment**: High, medium, or low risk trade

### 5. Alternative Scenarios
- Bull case: What would drive the stock higher?
- Bear case: What are the key risks or downside catalysts?
- Base case: Most likely scenario given current technicals

## Additional Context:
- Timeframe: {timeframe.title()} analysis
- Analysis date: {self._get_current_date()}
- Please use the most recent market data available
- Consider both technical and fundamental factors if relevant

Please structure your analysis clearly and provide actionable insights for traders and investors.
"""

        self.log_tool_usage(
            "technical_analysis_prompt", ticker=ticker, timeframe=timeframe
        )
        return prompt.strip()

    def _stock_screening_prompt(self, strategy: str = "momentum") -> str:
        """Generate stock screening prompt implementation."""
        # Validate strategy
        valid_strategies = ["momentum", "value", "growth", "quality", "dividend"]
        if strategy not in valid_strategies:
            strategy = "momentum"

        strategy_configs = {
            "momentum": {
                "title": "Momentum Stock Screening",
                "description": "Identify stocks with strong price momentum and technical strength",
                "criteria": [
                    "Strong relative strength (RS rating > 80)",
                    "Price above 50-day and 200-day moving averages",
                    "Recent breakout from consolidation pattern",
                    "Volume surge on breakout",
                    "Positive earnings growth",
                    "Strong sector performance",
                ],
                "metrics": [
                    "Relative Strength Index (RSI)",
                    "Price rate of change (ROC)",
                    "Volume relative to average",
                    "Distance from moving averages",
                    "Earnings growth rate",
                    "Revenue growth rate",
                ],
            },
            "value": {
                "title": "Value Stock Screening",
                "description": "Find undervalued stocks with strong fundamentals",
                "criteria": [
                    "Low P/E ratio relative to industry",
                    "P/B ratio below 2.0",
                    "Debt-to-equity ratio below industry average",
                    "Positive free cash flow",
                    "Dividend yield above market average",
                    "Strong return on equity (ROE > 15%)",
                ],
                "metrics": [
                    "Price-to-Earnings (P/E) ratio",
                    "Price-to-Book (P/B) ratio",
                    "Price-to-Sales (P/S) ratio",
                    "Enterprise Value/EBITDA",
                    "Free cash flow yield",
                    "Return on equity (ROE)",
                ],
            },
            "growth": {
                "title": "Growth Stock Screening",
                "description": "Identify companies with accelerating growth metrics",
                "criteria": [
                    "Revenue growth > 20% annually",
                    "Earnings growth acceleration",
                    "Strong profit margins",
                    "Expanding market share",
                    "Innovation and competitive advantages",
                    "Strong management execution",
                ],
                "metrics": [
                    "Revenue growth rate",
                    "Earnings per share (EPS) growth",
                    "Profit margin trends",
                    "Return on invested capital (ROIC)",
                    "Price/Earnings/Growth (PEG) ratio",
                    "Market share metrics",
                ],
            },
            "quality": {
                "title": "Quality Stock Screening",
                "description": "Find high-quality companies with sustainable competitive advantages",
                "criteria": [
                    "Consistent earnings growth (5+ years)",
                    "Strong balance sheet (low debt)",
                    "High return on equity (ROE > 20%)",
                    "Wide economic moat",
                    "Stable or growing market share",
                    "Strong management track record",
                ],
                "metrics": [
                    "Return on equity (ROE)",
                    "Return on assets (ROA)",
                    "Debt-to-equity ratio",
                    "Interest coverage ratio",
                    "Earnings consistency",
                    "Free cash flow stability",
                ],
            },
            "dividend": {
                "title": "Dividend Stock Screening",
                "description": "Identify stocks with attractive and sustainable dividend yields",
                "criteria": [
                    "Dividend yield between 3-8%",
                    "Dividend growth history (5+ years)",
                    "Payout ratio below 60%",
                    "Strong free cash flow coverage",
                    "Stable or growing earnings",
                    "Defensive business model",
                ],
                "metrics": [
                    "Dividend yield",
                    "Dividend growth rate",
                    "Payout ratio",
                    "Free cash flow coverage",
                    "Dividend aristocrat status",
                    "Earnings stability",
                ],
            },
        }

        config = strategy_configs[strategy]

        prompt = f"""
# {config["title"]} Analysis Request

Please conduct a comprehensive {strategy} stock screening analysis to {config["description"]}.

## Screening Criteria:

### Primary Filters:
{chr(10).join(f"- {criteria}" for criteria in config["criteria"])}

### Key Metrics to Analyze:
{chr(10).join(f"- {metric}" for metric in config["metrics"])}

## Analysis Framework:

### 1. Market Environment Assessment
- Current market conditions and {strategy} stock performance
- Sector rotation trends favoring {strategy} strategies
- Economic factors supporting {strategy} investing
- Historical performance of {strategy} strategies in similar conditions

### 2. Stock Screening Process
Please apply the following methodology:
- **Universe**: Focus on large and mid-cap stocks (market cap > $2B)
- **Liquidity**: Average daily volume > 1M shares
- **Fundamental Screening**: Apply the primary filters listed above
- **Technical Validation**: Confirm with technical analysis
- **Risk Assessment**: Evaluate potential risks and catalysts

### 3. Top Stock Recommendations
For each recommended stock, provide:
- **Company overview**: Business model and competitive position
- **Why it fits the {strategy} criteria**: Specific metrics and rationale
- **Risk factors**: Key risks to monitor
- **Price targets**: Entry points and target prices
- **Position sizing**: Recommended allocation (1-5% portfolio weight)

### 4. Portfolio Construction
- **Diversification**: Spread across sectors and industries
- **Risk management**: Position sizing and stop-loss levels
- **Rebalancing**: When and how to adjust positions
- **Performance monitoring**: Key metrics to track

### 5. Implementation Strategy
- **Entry strategy**: Best practices for building positions
- **Timeline**: Short-term vs. long-term holding periods
- **Market timing**: Consider current market cycle
- **Tax considerations**: Tax-efficient implementation

## Additional Requirements:
- Screen date: {self._get_current_date()}
- Market cap focus: Large and mid-cap stocks
- Geographic focus: US markets (can include international if compelling)
- Minimum liquidity: $10M average daily volume
- Exclude recent IPOs (< 6 months) unless exceptionally compelling

## Output Format:
1. **Executive Summary**: Key findings and market outlook
2. **Top 10 Stock Recommendations**: Detailed analysis for each
3. **Sector Allocation**: Recommended sector weights
4. **Risk Assessment**: Portfolio-level risks and mitigation
5. **Performance Expectations**: Expected returns and timeline

Please provide actionable insights that can be immediately implemented in a {strategy}-focused investment strategy.
"""

        self.log_tool_usage("stock_screening_prompt", strategy=strategy)
        return prompt.strip()

    def _get_current_date(self) -> str:
        """Get current date in readable format."""
        from datetime import UTC, datetime

        return datetime.now(UTC).strftime("%B %d, %Y")
