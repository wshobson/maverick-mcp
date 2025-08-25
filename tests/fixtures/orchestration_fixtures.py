"""
Comprehensive test fixtures for orchestration testing.

Provides realistic mock data for LLM responses, API responses, market data,
and test scenarios for the SupervisorAgent and DeepResearchAgent orchestration system.
"""

import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

# ==============================================================================
# MOCK LLM RESPONSES
# ==============================================================================


class MockLLMResponses:
    """Realistic LLM responses for various orchestration scenarios."""

    @staticmethod
    def query_classification_response(
        category: str = "stock_investment_decision",
        confidence: float = 0.85,
        parallel_capable: bool = True,
    ) -> str:
        """Mock query classification response from LLM."""
        routing_agents_map = {
            "market_screening": ["market"],
            "technical_analysis": ["technical"],
            "stock_investment_decision": ["market", "technical"],
            "portfolio_analysis": ["market", "technical"],
            "deep_research": ["research"],
            "company_research": ["research"],
            "sentiment_analysis": ["research"],
            "risk_assessment": ["market", "technical"],
        }

        return json.dumps(
            {
                "category": category,
                "confidence": confidence,
                "required_agents": routing_agents_map.get(category, ["market"]),
                "complexity": "moderate" if confidence > 0.7 else "complex",
                "estimated_execution_time_ms": 45000
                if category == "deep_research"
                else 30000,
                "parallel_capable": parallel_capable,
                "reasoning": f"Query classified as {category} based on content analysis and intent detection.",
            }
        )

    @staticmethod
    def result_synthesis_response(
        persona: str = "moderate",
        agents_used: list[str] = None,
        confidence: float = 0.82,
    ) -> str:
        """Mock result synthesis response from LLM."""
        if agents_used is None:
            agents_used = ["market", "technical"]

        persona_focused_content = {
            "conservative": """
            Based on comprehensive analysis from our specialist agents, AAPL presents a balanced investment opportunity
            with strong fundamentals and reasonable risk profile. The market analysis indicates stable sector
            positioning with consistent dividend growth, while technical indicators suggest a consolidation phase
            with support at $170. For conservative investors, consider gradual position building with
            strict stop-loss at $165 to preserve capital. The risk-adjusted return profile aligns well
            with conservative portfolio objectives, offering both income stability and modest growth potential.
            """,
            "moderate": """
            Our multi-agent analysis reveals AAPL as a compelling investment opportunity with balanced risk-reward
            characteristics. Market screening identified strong fundamentals including 15% revenue growth and
            expanding services segment. Technical analysis shows bullish momentum with RSI at 58 and MACD
            trending positive. Entry points around $175-180 offer favorable risk-reward with targets at $200-210.
            Position sizing of 3-5% of portfolio aligns with moderate risk tolerance while capitalizing on
            the current uptrend momentum.
            """,
            "aggressive": """
            Multi-agent analysis identifies AAPL as a high-conviction growth play with exceptional upside potential.
            Market analysis reveals accelerating AI adoption driving hardware refresh cycles, while technical
            indicators signal strong breakout momentum above $185 resistance. The confluence of fundamental
            catalysts and technical setup supports aggressive position sizing up to 8-10% allocation.
            Target price of $220+ represents 25% upside with momentum likely to continue through earnings season.
            This represents a prime opportunity for growth-focused portfolios seeking alpha generation.
            """,
        }

        return persona_focused_content.get(
            persona, persona_focused_content["moderate"]
        ).strip()

    @staticmethod
    def content_analysis_response(
        sentiment: str = "bullish", confidence: float = 0.75, credibility: float = 0.8
    ) -> str:
        """Mock content analysis response from LLM."""
        return json.dumps(
            {
                "KEY_INSIGHTS": [
                    "Apple's Q4 earnings exceeded expectations with 15% revenue growth",
                    "Services segment continues to expand with 12% year-over-year growth",
                    "iPhone 15 sales showing strong adoption in key markets",
                    "Cash position remains robust at $165B supporting capital allocation",
                    "AI integration across product line driving next upgrade cycle",
                ],
                "SENTIMENT": {"direction": sentiment, "confidence": confidence},
                "RISK_FACTORS": [
                    "China market regulatory concerns persist",
                    "Supply chain dependencies in Taiwan and South Korea",
                    "Increasing competition in services market",
                    "Currency headwinds affecting international revenue",
                ],
                "OPPORTUNITIES": [
                    "AI-powered device upgrade cycle beginning",
                    "Vision Pro market penetration expanding",
                    "Services recurring revenue model strengthening",
                    "Emerging markets iPhone adoption accelerating",
                ],
                "CREDIBILITY": credibility,
                "RELEVANCE": 0.9,
                "SUMMARY": f"Comprehensive analysis suggests {sentiment} outlook for Apple with strong fundamentals and growth catalysts, though regulatory and competitive risks require monitoring.",
            }
        )

    @staticmethod
    def research_synthesis_response(persona: str = "moderate") -> str:
        """Mock research synthesis response for deep research agent."""
        synthesis_by_persona = {
            "conservative": """
            ## Executive Summary
            Apple represents a stable, dividend-paying technology stock suitable for conservative portfolios seeking
            balanced growth and income preservation.

            ## Key Findings
            • Consistent dividend growth averaging 8% annually over past 5 years
            • Strong balance sheet with $165B cash providing downside protection
            • Services revenue provides recurring income stream growing at 12% annually
            • P/E ratio of 28x reasonable for quality growth stock
            • Beta of 1.1 indicates moderate volatility relative to market
            • Debt-to-equity ratio of 0.3 shows conservative capital structure
            • Free cash flow yield of 3.2% supports dividend sustainability

            ## Investment Implications for Conservative Investors
            Apple's combination of dividend growth, balance sheet strength, and market leadership makes it suitable
            for conservative portfolios. The company's pivot to services provides recurring revenue stability while
            hardware sales offer moderate growth potential.

            ## Risk Considerations
            Primary risks include China market exposure (19% of revenue), technology obsolescence, and regulatory
            pressure on App Store policies. However, strong cash position provides significant downside protection.

            ## Recommended Actions
            Consider 2-3% portfolio allocation with gradual accumulation on pullbacks below $170.
            Appropriate stop-loss at $160 to limit downside risk.
            """,
            "moderate": """
            ## Executive Summary
            Apple presents a balanced investment opportunity combining growth potential with quality fundamentals,
            well-suited for diversified moderate-risk portfolios.

            ## Key Findings
            • Revenue growth acceleration to 15% driven by AI-enhanced products
            • Services segment margins expanding to 70%, improving overall profitability
            • Strong competitive moats in ecosystem and brand loyalty
            • Capital allocation balance between growth investment and shareholder returns
            • Technical indicators suggesting continued uptrend momentum
            • Valuation appears fair at current levels with room for multiple expansion
            • Market leadership position in premium smartphone and tablet segments

            ## Investment Implications for Moderate Investors
            Apple offers an attractive blend of stability and growth potential. The company's evolution toward
            services provides recurring revenue while hardware innovation drives periodic upgrade cycles.

            ## Risk Considerations
            Key risks include supply chain disruption, China regulatory issues, and increasing competition
            in services. Currency headwinds may impact international revenue growth.

            ## Recommended Actions
            Target 4-5% portfolio allocation with entry points between $175-185. Consider taking profits
            above $210 and adding on weakness below $170.
            """,
            "aggressive": """
            ## Executive Summary
            Apple stands at the forefront of the next technology revolution with AI integration across its ecosystem,
            presenting significant alpha generation potential for growth-focused investors.

            ## Key Findings
            • AI-driven product refresh cycle beginning with iPhone 15 Pro and Vision Pro launch
            • Services revenue trajectory accelerating with 18% growth potential
            • Market share expansion opportunities in emerging markets and enterprise
            • Vision Pro early adoption exceeding expectations, validating spatial computing thesis
            • Developer ecosystem strengthening with AI tools integration
            • Operating leverage improving with services mix shift
            • Stock momentum indicators showing bullish technical setup

            ## Investment Implications for Aggressive Investors
            Apple represents a high-conviction growth play with multiple expansion catalysts. The convergence
            of AI adoption, new product categories, and services growth creates exceptional upside potential.

            ## Risk Considerations
            Execution risk on Vision Pro adoption, competitive response from Android ecosystem, and
            regulatory pressure on App Store represent key downside risks requiring active monitoring.

            ## Recommended Actions
            Consider aggressive 8-10% allocation with momentum-based entry above $185 resistance.
            Target price $230+ over 12-month horizon with trailing stop at 15% to protect gains.
            """,
        }

        return synthesis_by_persona.get(
            persona, synthesis_by_persona["moderate"]
        ).strip()


# ==============================================================================
# MOCK EXA API RESPONSES
# ==============================================================================


class MockExaResponses:
    """Realistic Exa API responses for financial research."""

    @staticmethod
    def search_results_aapl() -> list[dict[str, Any]]:
        """Mock Exa search results for AAPL analysis."""
        return [
            {
                "url": "https://www.bloomberg.com/news/articles/2024-01-15/apple-earnings-beat",
                "title": "Apple Earnings Beat Expectations as iPhone Sales Surge",
                "content": "Apple Inc. reported quarterly revenue of $119.6 billion, surpassing analyst expectations as iPhone 15 sales showed strong momentum in key markets. The technology giant's services segment grew 12% year-over-year to $23.1 billion, demonstrating the recurring revenue model's strength. CEO Tim Cook highlighted AI integration across the product lineup as a key driver for the next upgrade cycle. Gross margins expanded to 45.9% compared to 43.3% in the prior year period, reflecting improved mix and operational efficiency. The company's cash position remains robust at $165.1 billion, providing flexibility for strategic investments and shareholder returns. China revenue declined 2% due to competitive pressures, though management expressed optimism about long-term opportunities in the region.",
                "summary": "Apple exceeded Q4 earnings expectations with strong iPhone 15 sales and services growth, while maintaining robust cash position and expanding margins despite China headwinds.",
                "highlights": [
                    "iPhone 15 strong sales momentum",
                    "Services grew 12% year-over-year",
                    "$165.1B cash position",
                ],
                "published_date": "2024-01-15T08:30:00Z",
                "author": "Mark Gurman",
                "score": 0.94,
                "provider": "exa",
            },
            {
                "url": "https://seekingalpha.com/article/4665432-apple-stock-analysis-ai-catalyst",
                "title": "Apple Stock: AI Integration Could Drive Next Super Cycle",
                "content": "Apple's integration of artificial intelligence across its ecosystem represents a potential catalyst for the next device super cycle. The company's on-device AI processing capabilities, enabled by the A17 Pro chip, position Apple uniquely in the mobile AI landscape. Industry analysts project AI-enhanced features could drive iPhone replacement cycles to accelerate from the current 3.5 years to approximately 2.8 years. The services ecosystem benefits significantly from AI integration, with enhanced Siri capabilities driving increased App Store engagement and subscription services adoption. Vision Pro early metrics suggest spatial computing adoption is tracking ahead of initial estimates, with developer interest surging 300% quarter-over-quarter. The convergence of AI, spatial computing, and services creates multiple revenue expansion vectors over the next 3-5 years.",
                "summary": "AI integration across Apple's ecosystem could accelerate device replacement cycles and expand services revenue through enhanced user engagement.",
                "highlights": [
                    "AI-driven replacement cycle acceleration",
                    "Vision Pro adoption tracking well",
                    "Services ecosystem AI benefits",
                ],
                "published_date": "2024-01-14T14:20:00Z",
                "author": "Tech Analyst Team",
                "score": 0.87,
                "provider": "exa",
            },
            {
                "url": "https://www.morningstar.com/stocks/aapl-valuation-analysis",
                "title": "Apple Valuation Analysis: Fair Value Assessment",
                "content": "Our discounted cash flow analysis suggests Apple's fair value ranges between $185-195 per share, indicating the stock trades near intrinsic value at current levels. The company's transition toward higher-margin services revenue supports multiple expansion, though hardware cycle dependency introduces valuation volatility. Key valuation drivers include services attach rates (currently 85% of active devices), gross margin trajectory (target 47-48% long-term), and capital allocation efficiency. The dividend yield of 0.5% appears sustainable with strong free cash flow generation of $95+ billion annually. Compared to technology peers, Apple trades at a 15% premium to the sector median, justified by superior return on invested capital and cash generation capabilities.",
                "summary": "DCF analysis places Apple's fair value at $185-195, with current valuation supported by services transition and strong cash generation.",
                "highlights": [
                    "Fair value $185-195 range",
                    "Services driving multiple expansion",
                    "Strong free cash flow $95B+",
                ],
                "published_date": "2024-01-13T11:45:00Z",
                "author": "Sarah Chen",
                "score": 0.91,
                "provider": "exa",
            },
            {
                "url": "https://www.reuters.com/technology/apple-china-challenges-2024-01-12",
                "title": "Apple Faces Growing Competition in China Market",
                "content": "Apple confronts intensifying competition in China as local brands gain market share and regulatory scrutiny increases. Huawei's Mate 60 Pro launch has resonated strongly with Chinese consumers, contributing to Apple's 2% revenue decline in Greater China for Q4. The Chinese government's restrictions on iPhone use in government agencies signal potential broader policy shifts. Despite challenges, Apple maintains premium market leadership with 47% share in smartphones priced above $600. Management highlighted ongoing investments in local partnerships and supply chain relationships to navigate the complex regulatory environment. The company's services revenue in China grew 8% despite hardware headwinds, demonstrating ecosystem stickiness among existing users.",
                "summary": "Apple faces competitive and regulatory challenges in China, though maintains premium market leadership and growing services revenue.",
                "highlights": [
                    "China revenue down 2%",
                    "Regulatory iPhone restrictions",
                    "Premium segment leadership maintained",
                ],
                "published_date": "2024-01-12T16:15:00Z",
                "author": "Reuters Technology Team",
                "score": 0.89,
                "provider": "exa",
            },
        ]

    @staticmethod
    def search_results_market_sentiment() -> list[dict[str, Any]]:
        """Mock Exa results for market sentiment analysis."""
        return [
            {
                "url": "https://www.cnbc.com/2024/01/16/market-outlook-tech-stocks",
                "title": "Tech Stocks Rally on AI Optimism Despite Rate Concerns",
                "content": "Technology stocks surged 2.3% as artificial intelligence momentum overcame Federal Reserve policy concerns. Investors rotated into AI-beneficiary names including Apple, Microsoft, and Nvidia following strong earnings guidance across the sector. The Technology Select Sector SPDR ETF (XLK) reached new 52-week highs despite 10-year Treasury yields hovering near 4.5%. Institutional flows show $12.8 billion net inflows to technology funds over the past month, the strongest since early 2023. Options activity indicates continued bullish sentiment with call volume exceeding puts by 1.8:1 across major tech names. Analyst upgrades accelerated with 67% of tech stocks carrying buy ratings versus 52% sector average.",
                "summary": "Tech stocks rally on AI optimism with strong institutional inflows and bullish options activity despite interest rate headwinds.",
                "highlights": [
                    "Tech sector +2.3%",
                    "$12.8B institutional inflows",
                    "Call/put ratio 1.8:1",
                ],
                "published_date": "2024-01-16T09:45:00Z",
                "author": "CNBC Markets Team",
                "score": 0.92,
                "provider": "exa",
            },
            {
                "url": "https://finance.yahoo.com/news/vix-fear-greed-market-sentiment",
                "title": "VIX Falls to Multi-Month Lows as Fear Subsides",
                "content": "The VIX volatility index dropped to 13.8, the lowest level since November 2021, signaling reduced market anxiety and increased risk appetite among investors. The CNN Fear & Greed Index shifted to 'Greed' territory at 72, up from 'Neutral' just two weeks ago. Credit spreads tightened across investment-grade and high-yield markets, with IG spreads at 85 basis points versus 110 in December. Equity put/call ratios declined to 0.45, indicating overwhelming bullish positioning. Margin debt increased 8% month-over-month as investors leverage up for continued market gains.",
                "summary": "Market sentiment indicators show reduced fear and increased greed with VIX at multi-month lows and bullish positioning accelerating.",
                "highlights": [
                    "VIX at 13.8 multi-month low",
                    "Fear & Greed at 72",
                    "Margin debt up 8%",
                ],
                "published_date": "2024-01-16T14:30:00Z",
                "author": "Market Sentiment Team",
                "score": 0.88,
                "provider": "exa",
            },
        ]

    @staticmethod
    def search_results_empty() -> list[dict[str, Any]]:
        """Mock empty Exa search results for testing edge cases."""
        return []

    @staticmethod
    def search_results_low_quality() -> list[dict[str, Any]]:
        """Mock low-quality Exa search results for credibility testing."""
        return [
            {
                "url": "https://sketchy-site.com/apple-prediction",
                "title": "AAPL Will 100X - Trust Me Bro Analysis",
                "content": "Apple stock is going to the moon because reasons. My uncle works at Apple and says they're releasing iPhones made of gold next year. This is not financial advice but also definitely is financial advice. Buy now or cry later. Diamond hands to the moon rockets.",
                "summary": "Questionable analysis with unsubstantiated claims about Apple's prospects.",
                "highlights": [
                    "Gold iPhones coming",
                    "100x returns predicted",
                    "Uncle insider info",
                ],
                "published_date": "2024-01-16T23:59:00Z",
                "author": "Random Internet User",
                "score": 0.12,
                "provider": "exa",
            }
        ]


# ==============================================================================
# MOCK TAVILY API RESPONSES
# ==============================================================================


class MockTavilyResponses:
    """Realistic Tavily API responses for web search."""

    @staticmethod
    def search_results_aapl() -> dict[str, Any]:
        """Mock Tavily search response for AAPL analysis."""
        return {
            "query": "Apple stock analysis AAPL investment outlook",
            "follow_up_questions": [
                "What are Apple's main revenue drivers?",
                "How does Apple compare to competitors?",
                "What are the key risks for Apple stock?",
            ],
            "answer": "Apple (AAPL) shows strong fundamentals with growing services revenue and AI integration opportunities, though faces competition in China and regulatory pressures.",
            "results": [
                {
                    "title": "Apple Stock Analysis: Strong Fundamentals Despite Headwinds",
                    "url": "https://www.fool.com/investing/2024/01/15/apple-stock-analysis",
                    "content": "Apple's latest quarter demonstrated the resilience of its business model, with services revenue hitting a new record and iPhone sales exceeding expectations. The company's focus on artificial intelligence integration across its product ecosystem positions it well for future growth cycles. However, investors should monitor China market dynamics and App Store regulatory challenges that could impact long-term growth trajectories.",
                    "raw_content": "Apple Inc. (AAPL) continues to demonstrate strong business fundamentals in its latest quarterly report, with services revenue reaching new records and iPhone sales beating analyst expectations across key markets. The technology giant has strategically positioned itself at the forefront of artificial intelligence integration, with on-device AI processing capabilities that differentiate its products from competitors. Looking ahead, the company's ecosystem approach and services transition provide multiple growth vectors, though challenges in China and regulatory pressures on App Store policies require careful monitoring. The stock's current valuation appears reasonable given the company's cash generation capabilities and market position.",
                    "published_date": "2024-01-15",
                    "score": 0.89,
                },
                {
                    "title": "Tech Sector Outlook: AI Revolution Drives Growth",
                    "url": "https://www.barrons.com/articles/tech-outlook-ai-growth",
                    "content": "The technology sector stands at the beginning of a multi-year artificial intelligence transformation that could reshape revenue models and competitive dynamics. Companies with strong AI integration capabilities, including Apple, Microsoft, and Google, are positioned to benefit from this shift. Apple's approach of on-device AI processing provides privacy advantages and reduces cloud infrastructure costs compared to competitors relying heavily on cloud-based AI services.",
                    "raw_content": "The technology sector is experiencing a fundamental transformation as artificial intelligence capabilities become central to product differentiation and user experience. Companies that can effectively integrate AI while maintaining user privacy and system performance are likely to capture disproportionate value creation over the next 3-5 years. Apple's strategy of combining custom silicon with on-device AI processing provides competitive advantages in both performance and privacy, potentially driving accelerated device replacement cycles and services engagement. This positions Apple favorably compared to competitors relying primarily on cloud-based AI infrastructure.",
                    "published_date": "2024-01-14",
                    "score": 0.85,
                },
                {
                    "title": "Investment Analysis: Apple's Services Transformation",
                    "url": "https://www.investopedia.com/apple-services-analysis",
                    "content": "Apple's transformation from a hardware-centric to services-enabled company continues to gain momentum, with services revenue now representing over 22% of total revenue and growing at double-digit rates. This shift toward recurring revenue streams improves business model predictability and supports higher valuation multiples. The company's services ecosystem benefits from its large installed base and strong customer loyalty metrics.",
                    "raw_content": "Apple Inc.'s strategic evolution toward a services-centric business model represents one of the most successful corporate transformations in technology sector history. The company has leveraged its installed base of over 2 billion active devices to create a thriving services ecosystem encompassing the App Store, Apple Music, iCloud, Apple Pay, and various subscription services. This services revenue now exceeds $85 billion annually and continues growing at rates exceeding 10% year-over-year, providing both revenue diversification and margin enhancement. The recurring nature of services revenue creates more predictable cash flows and justifies premium valuation multiples compared to pure hardware companies.",
                    "published_date": "2024-01-13",
                    "score": 0.91,
                },
            ],
            "response_time": 1.2,
        }

    @staticmethod
    def search_results_market_sentiment() -> dict[str, Any]:
        """Mock Tavily search response for market sentiment analysis."""
        return {
            "query": "stock market sentiment investor mood analysis 2024",
            "follow_up_questions": [
                "What are current market sentiment indicators?",
                "How do investors feel about tech stocks?",
                "What factors are driving market optimism?",
            ],
            "answer": "Current market sentiment shows cautious optimism with reduced volatility and increased risk appetite, driven by AI enthusiasm and strong corporate earnings despite interest rate concerns.",
            "results": [
                {
                    "title": "Market Sentiment Indicators Signal Bullish Mood",
                    "url": "https://www.marketwatch.com/story/market-sentiment-bullish",
                    "content": "Multiple sentiment indicators suggest investors have shifted from defensive to risk-on positioning as 2024 progresses. The VIX volatility index has declined to multi-month lows while institutional money flows accelerate into equities. Credit markets show tightening spreads and increased issuance activity, reflecting improved risk appetite across asset classes.",
                    "raw_content": "A comprehensive analysis of market sentiment indicators reveals a significant shift in investor psychology over the past month. The CBOE Volatility Index (VIX) has dropped below 14, its lowest level since late 2021, indicating reduced fear and increased complacency among options traders. Simultaneously, the American Association of Individual Investors (AAII) sentiment survey shows bullish respondents outnumbering bearish by a 2:1 margin, the widest spread since early 2023. Institutional flows data from EPFR shows $45 billion in net inflows to equity funds over the past four weeks, with technology and growth sectors receiving disproportionate allocation.",
                    "published_date": "2024-01-16",
                    "score": 0.93,
                },
                {
                    "title": "Investor Psychology: Fear of Missing Out Returns",
                    "url": "https://www.wsj.com/markets/stocks/fomo-returns-markets",
                    "content": "The fear of missing out (FOMO) mentality has returned to equity markets as investors chase performance and increase leverage. Margin debt has increased significantly while cash positions at major brokerages have declined to multi-year lows. This shift in behavior suggests sentiment has moved from cautious to optimistic, though some analysts warn of potential overextension.",
                    "raw_content": "Behavioral indicators suggest a fundamental shift in investor psychology from the cautious stance that characterized much of 2023 to a more aggressive, opportunity-seeking mindset. NYSE margin debt has increased 15% over the past two months, reaching $750 billion as investors leverage up to participate in market gains. Cash positions at major discount brokerages have declined to just 3.2% of assets, compared to 5.8% during peak uncertainty in October 2023. Options market activity shows call volume exceeding put volume by the widest margin in 18 months, with particular strength in technology and AI-related names.",
                    "published_date": "2024-01-15",
                    "score": 0.88,
                },
            ],
            "response_time": 1.4,
        }

    @staticmethod
    def search_results_error() -> dict[str, Any]:
        """Mock Tavily error response for testing error handling."""
        return {
            "error": "rate_limit_exceeded",
            "message": "API rate limit exceeded. Please try again later.",
            "retry_after": 60,
        }


# ==============================================================================
# MOCK MARKET DATA
# ==============================================================================


class MockMarketData:
    """Realistic market data for testing financial analysis."""

    @staticmethod
    def stock_price_history(
        symbol: str = "AAPL", days: int = 100, current_price: float = 185.0
    ) -> pd.DataFrame:
        """Generate realistic stock price history."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Generate realistic price movement
        np.random.seed(42)  # Consistent data for testing
        returns = np.random.normal(
            0.0008, 0.02, len(dates)
        )  # ~0.2% daily return, 2% volatility

        # Start with a base price and apply returns
        base_price = current_price * 0.9  # Start 10% lower
        prices = [base_price]

        for return_val in returns[1:]:
            next_price = prices[-1] * (1 + return_val)
            prices.append(max(next_price, 50))  # Floor price at $50

        # Create OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices, strict=False)):
            # Generate realistic OHLC from close price
            volatility = abs(np.random.normal(0, 0.015))  # Intraday volatility

            high = close_price * (1 + volatility)
            low = close_price * (1 - volatility)

            # Determine open based on previous close with gap
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, 0.005)  # Small gap
                open_price = prices[i - 1] * (1 + gap)

            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Generate volume
            base_volume = 50_000_000  # Base volume
            volume_multiplier = np.random.uniform(0.5, 2.0)
            volume = int(base_volume * volume_multiplier)

            data.append(
                {
                    "Date": date,
                    "Open": round(open_price, 2),
                    "High": round(high, 2),
                    "Low": round(low, 2),
                    "Close": round(close_price, 2),
                    "Volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        return df

    @staticmethod
    def technical_indicators(symbol: str = "AAPL") -> dict[str, Any]:
        """Mock technical indicators for a stock."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "rsi": {
                "value": 58.3,
                "signal": "neutral",
                "interpretation": "Neither overbought nor oversold",
            },
            "macd": {
                "value": 2.15,
                "signal_line": 1.89,
                "histogram": 0.26,
                "signal": "bullish",
                "interpretation": "MACD above signal line suggests bullish momentum",
            },
            "bollinger_bands": {
                "upper": 192.45,
                "middle": 185.20,
                "lower": 177.95,
                "position": "middle",
                "squeeze": False,
            },
            "moving_averages": {
                "sma_20": 183.45,
                "sma_50": 178.90,
                "sma_200": 172.15,
                "ema_12": 184.80,
                "ema_26": 181.30,
            },
            "support_resistance": {
                "support_levels": [175.00, 170.50, 165.25],
                "resistance_levels": [190.00, 195.75, 200.50],
                "current_level": "between_support_resistance",
            },
            "volume_analysis": {
                "average_volume": 52_000_000,
                "current_volume": 68_000_000,
                "relative_volume": 1.31,
                "volume_trend": "increasing",
            },
        }

    @staticmethod
    def market_overview() -> dict[str, Any]:
        """Mock market overview data."""
        return {
            "timestamp": datetime.now(),
            "indices": {
                "SPY": {"price": 485.30, "change": +2.15, "change_pct": +0.44},
                "QQQ": {"price": 412.85, "change": +5.42, "change_pct": +1.33},
                "IWM": {"price": 195.67, "change": -1.23, "change_pct": -0.62},
                "VIX": {"price": 13.8, "change": -1.2, "change_pct": -8.0},
            },
            "sector_performance": {
                "Technology": +1.85,
                "Healthcare": +0.45,
                "Financial Services": -0.32,
                "Consumer Cyclical": +0.78,
                "Industrials": -0.15,
                "Energy": -1.22,
                "Utilities": +0.33,
                "Real Estate": +0.91,
                "Materials": -0.67,
                "Consumer Defensive": +0.12,
                "Communication Services": +1.34,
            },
            "market_breadth": {
                "advancers": 1845,
                "decliners": 1230,
                "unchanged": 125,
                "new_highs": 89,
                "new_lows": 12,
                "up_volume": 8.2e9,
                "down_volume": 4.1e9,
            },
            "sentiment_indicators": {
                "fear_greed_index": 72,
                "vix_level": "low",
                "put_call_ratio": 0.45,
                "margin_debt_trend": "increasing",
            },
        }


# ==============================================================================
# TEST QUERY EXAMPLES
# ==============================================================================


class TestQueries:
    """Realistic user queries for different classification categories."""

    MARKET_SCREENING = [
        "Find me momentum stocks in the technology sector with strong earnings growth",
        "Screen for dividend-paying stocks with yields above 3% and consistent payout history",
        "Show me small-cap stocks with high revenue growth and low debt levels",
        "Find stocks breaking out of consolidation patterns with increasing volume",
        "Screen for value stocks trading below book value with improving fundamentals",
    ]

    COMPANY_RESEARCH = [
        "Analyze Apple's competitive position in the smartphone market",
        "Research Tesla's battery technology advantages and manufacturing scale",
        "Provide comprehensive analysis of Microsoft's cloud computing strategy",
        "Analyze Amazon's e-commerce margins and AWS growth potential",
        "Research Nvidia's AI chip market dominance and competitive threats",
    ]

    TECHNICAL_ANALYSIS = [
        "Analyze AAPL's chart patterns and provide entry/exit recommendations",
        "What do the technical indicators say about SPY's short-term direction?",
        "Analyze TSLA's support and resistance levels for swing trading",
        "Show me the RSI and MACD signals for QQQ",
        "Identify chart patterns in the Nasdaq that suggest market direction",
    ]

    SENTIMENT_ANALYSIS = [
        "What's the current market sentiment around tech stocks?",
        "Analyze investor sentiment toward electric vehicle companies",
        "How are traders feeling about the Fed's interest rate policy?",
        "What's the mood in crypto markets right now?",
        "Analyze sentiment around bank stocks after recent earnings",
    ]

    PORTFOLIO_ANALYSIS = [
        "Optimize my portfolio allocation for moderate risk tolerance",
        "Analyze the correlation between my holdings and suggest diversification",
        "Review my portfolio for sector concentration risk",
        "Suggest rebalancing strategy for my retirement portfolio",
        "Analyze my portfolio's beta and suggest hedging strategies",
    ]

    RISK_ASSESSMENT = [
        "Calculate appropriate position size for AAPL given my $100k account",
        "What's the maximum drawdown risk for a 60/40 portfolio?",
        "Analyze the tail risk in my growth stock positions",
        "Calculate VaR for my current portfolio allocation",
        "Assess concentration risk in my tech-heavy portfolio",
    ]

    @classmethod
    def get_random_query(cls, category: str) -> str:
        """Get a random query from the specified category."""
        queries_map = {
            "market_screening": cls.MARKET_SCREENING,
            "company_research": cls.COMPANY_RESEARCH,
            "technical_analysis": cls.TECHNICAL_ANALYSIS,
            "sentiment_analysis": cls.SENTIMENT_ANALYSIS,
            "portfolio_analysis": cls.PORTFOLIO_ANALYSIS,
            "risk_assessment": cls.RISK_ASSESSMENT,
        }

        queries = queries_map.get(category, cls.MARKET_SCREENING)
        return np.random.choice(queries)


# ==============================================================================
# PERSONA-SPECIFIC FIXTURES
# ==============================================================================


class PersonaFixtures:
    """Persona-specific test data and responses."""

    @staticmethod
    def conservative_investor_data() -> dict[str, Any]:
        """Data for conservative investor persona testing."""
        return {
            "persona": "conservative",
            "characteristics": [
                "capital preservation",
                "income generation",
                "low volatility",
                "dividend focus",
            ],
            "risk_tolerance": 0.3,
            "preferred_sectors": ["Utilities", "Consumer Defensive", "Healthcare"],
            "analysis_focus": [
                "dividend yield",
                "debt levels",
                "stability",
                "downside protection",
            ],
            "position_sizing": {
                "max_single_position": 0.05,  # 5% max
                "stop_loss_multiplier": 1.5,
                "target_volatility": 0.12,
            },
            "sample_recommendations": [
                "Consider gradual position building with strict risk management",
                "Focus on dividend-paying stocks with consistent payout history",
                "Maintain defensive positioning until market clarity improves",
                "Prioritize capital preservation over aggressive growth",
            ],
        }

    @staticmethod
    def moderate_investor_data() -> dict[str, Any]:
        """Data for moderate investor persona testing."""
        return {
            "persona": "moderate",
            "characteristics": [
                "balanced growth",
                "diversification",
                "moderate risk",
                "long-term focus",
            ],
            "risk_tolerance": 0.6,
            "preferred_sectors": [
                "Technology",
                "Healthcare",
                "Financial Services",
                "Industrials",
            ],
            "analysis_focus": [
                "risk-adjusted returns",
                "diversification",
                "growth potential",
                "fundamentals",
            ],
            "position_sizing": {
                "max_single_position": 0.08,  # 8% max
                "stop_loss_multiplier": 2.0,
                "target_volatility": 0.18,
            },
            "sample_recommendations": [
                "Balance growth opportunities with risk management",
                "Consider diversified allocation across sectors and market caps",
                "Target 4-6% position sizing for high-conviction ideas",
                "Monitor both technical and fundamental indicators",
            ],
        }

    @staticmethod
    def aggressive_investor_data() -> dict[str, Any]:
        """Data for aggressive investor persona testing."""
        return {
            "persona": "aggressive",
            "characteristics": [
                "high growth",
                "momentum",
                "concentrated positions",
                "active trading",
            ],
            "risk_tolerance": 0.9,
            "preferred_sectors": [
                "Technology",
                "Communication Services",
                "Consumer Cyclical",
            ],
            "analysis_focus": [
                "growth potential",
                "momentum",
                "catalysts",
                "alpha generation",
            ],
            "position_sizing": {
                "max_single_position": 0.15,  # 15% max
                "stop_loss_multiplier": 3.0,
                "target_volatility": 0.25,
            },
            "sample_recommendations": [
                "Consider concentrated positions in high-conviction names",
                "Target momentum stocks with strong catalysts",
                "Use 10-15% position sizing for best opportunities",
                "Focus on alpha generation over risk management",
            ],
        }


# ==============================================================================
# EDGE CASE AND ERROR FIXTURES
# ==============================================================================


class EdgeCaseFixtures:
    """Fixtures for testing edge cases and error conditions."""

    @staticmethod
    def api_failure_responses() -> dict[str, Any]:
        """Mock API failure responses for error handling testing."""
        return {
            "exa_rate_limit": {
                "error": "rate_limit_exceeded",
                "message": "You have exceeded your API rate limit",
                "retry_after": 3600,
                "status_code": 429,
            },
            "tavily_unauthorized": {
                "error": "unauthorized",
                "message": "Invalid API key provided",
                "status_code": 401,
            },
            "llm_timeout": {
                "error": "timeout",
                "message": "Request timed out after 30 seconds",
                "status_code": 408,
            },
            "network_error": {
                "error": "network_error",
                "message": "Unable to connect to external service",
                "status_code": 503,
            },
        }

    @staticmethod
    def conflicting_agent_results() -> dict[str, dict[str, Any]]:
        """Mock conflicting results from different agents for synthesis testing."""
        return {
            "market": {
                "recommendation": "BUY",
                "confidence": 0.85,
                "reasoning": "Strong fundamentals and sector rotation into technology",
                "target_price": 210.0,
                "sentiment": "bullish",
            },
            "technical": {
                "recommendation": "SELL",
                "confidence": 0.78,
                "reasoning": "Bearish divergence in RSI and approaching strong resistance",
                "target_price": 165.0,
                "sentiment": "bearish",
            },
            "research": {
                "recommendation": "HOLD",
                "confidence": 0.72,
                "reasoning": "Mixed signals from fundamental analysis and market conditions",
                "target_price": 185.0,
                "sentiment": "neutral",
            },
        }

    @staticmethod
    def incomplete_data() -> dict[str, Any]:
        """Mock incomplete or missing data scenarios."""
        return {
            "missing_price_data": {
                "symbol": "AAPL",
                "error": "Price data not available for requested timeframe",
                "available_data": None,
            },
            "partial_search_results": {
                "results_found": 2,
                "results_expected": 10,
                "provider_errors": ["exa_timeout", "tavily_rate_limit"],
                "partial_data": True,
            },
            "llm_partial_response": {
                "analysis": "Partial analysis completed before",
                "truncated": True,
                "completion_percentage": 0.6,
            },
        }

    @staticmethod
    def malformed_data() -> dict[str, Any]:
        """Mock malformed or invalid data for error testing."""
        return {
            "invalid_json": '{"analysis": "incomplete json"',  # Missing closing brace
            "wrong_schema": {
                "unexpected_field": "value",
                "missing_required_field": None,
            },
            "invalid_dates": {
                "published_date": "not-a-date",
                "timestamp": "invalid-timestamp",
            },
            "invalid_numbers": {"confidence": "not-a-number", "price": "invalid-price"},
        }


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================


@pytest.fixture
def mock_llm_responses():
    """Fixture providing mock LLM responses."""
    return MockLLMResponses()


@pytest.fixture
def mock_exa_responses():
    """Fixture providing mock Exa API responses."""
    return MockExaResponses()


@pytest.fixture
def mock_tavily_responses():
    """Fixture providing mock Tavily API responses."""
    return MockTavilyResponses()


@pytest.fixture
def mock_market_data():
    """Fixture providing mock market data."""
    return MockMarketData()


@pytest.fixture
def test_queries():
    """Fixture providing test queries."""
    return TestQueries()


@pytest.fixture
def persona_fixtures():
    """Fixture providing persona-specific data."""
    return PersonaFixtures()


@pytest.fixture
def edge_case_fixtures():
    """Fixture providing edge case test data."""
    return EdgeCaseFixtures()


@pytest.fixture(params=["conservative", "moderate", "aggressive"])
def investor_persona(request):
    """Parametrized fixture for testing across all investor personas."""
    return request.param


@pytest.fixture(
    params=[
        "market_screening",
        "company_research",
        "technical_analysis",
        "sentiment_analysis",
    ]
)
def query_category(request):
    """Parametrized fixture for testing across all query categories."""
    return request.param


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def create_mock_llm_with_responses(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns specific responses in order."""
    mock_llm = MagicMock()

    # Create AIMessage objects for each response
    ai_messages = [AIMessage(content=response) for response in responses]
    mock_llm.ainvoke.side_effect = ai_messages

    return mock_llm


def create_mock_agent_result(
    agent_type: str,
    confidence: float = 0.8,
    recommendation: str = "BUY",
    additional_data: dict[str, Any] = None,
) -> dict[str, Any]:
    """Create a mock agent result with realistic structure."""
    base_result = {
        "status": "success",
        "agent_type": agent_type,
        "confidence_score": confidence,
        "recommendation": recommendation,
        "timestamp": datetime.now(),
        "execution_time_ms": np.random.uniform(1000, 5000),
    }

    if additional_data:
        base_result.update(additional_data)

    return base_result


def create_realistic_stock_data(
    symbol: str = "AAPL", price: float = 185.0, volume: int = 50_000_000
) -> dict[str, Any]:
    """Create realistic stock data for testing."""
    return {
        "symbol": symbol,
        "current_price": price,
        "volume": volume,
        "market_cap": 2_850_000_000_000,  # $2.85T for AAPL
        "pe_ratio": 28.5,
        "dividend_yield": 0.005,
        "beta": 1.1,
        "52_week_high": 198.23,
        "52_week_low": 164.08,
        "average_volume": 48_000_000,
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }


# Export main classes for easy importing
__all__ = [
    "MockLLMResponses",
    "MockExaResponses",
    "MockTavilyResponses",
    "MockMarketData",
    "TestQueries",
    "PersonaFixtures",
    "EdgeCaseFixtures",
    "create_mock_llm_with_responses",
    "create_mock_agent_result",
    "create_realistic_stock_data",
]
