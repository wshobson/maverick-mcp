"""
Deep research tools for comprehensive financial research and analysis.

This module provides MCP tools for the DeepResearchAgent, enabling comprehensive
financial research through web search, content analysis, and AI-powered insights.
"""

import logging
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.config.settings import get_settings
from maverick_mcp.providers.llm_factory import get_llm
from maverick_mcp.utils.orchestration_logging import (
    log_performance_metrics,
    log_tool_invocation,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize LLM and agent
llm = get_llm()
research_agent = None


def get_research_agent() -> DeepResearchAgent:
    """Get or create the research agent singleton."""
    global research_agent
    if research_agent is None:
        research_agent = DeepResearchAgent(
            llm=llm,
            persona="moderate",  # Default persona, can be overridden
            max_sources=50,
            research_depth="comprehensive",
        )
    return research_agent


class ResearchRequest(BaseModel):
    """Request model for research operations."""

    query: str = Field(description="Research query or topic")
    persona: str | None = Field(default="moderate", description="Investor persona")
    research_scope: str | None = Field(
        default="comprehensive", description="Research scope"
    )
    max_sources: int | None = Field(
        default=50, description="Maximum sources to analyze"
    )
    timeframe: str | None = Field(default="1m", description="Time frame for search")
    session_id: str | None = Field(default=None, description="Session identifier")


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""

    topic: str = Field(description="Topic for sentiment analysis")
    timeframe: str | None = Field(default="1w", description="Time frame for analysis")
    persona: str | None = Field(default="moderate", description="Investor persona")
    session_id: str | None = Field(default=None, description="Session identifier")


class CompanyResearchRequest(BaseModel):
    """Request model for company research."""

    symbol: str = Field(description="Stock symbol")
    include_competitive_analysis: bool | None = Field(
        default=True, description="Include competitive analysis"
    )
    persona: str | None = Field(default="moderate", description="Investor persona")
    session_id: str | None = Field(default=None, description="Session identifier")


def create_research_router() -> FastMCP:
    """Create and configure the research router."""

    mcp = FastMCP("Deep Research Tools")

    @mcp.tool()
    async def comprehensive_research(request: ResearchRequest) -> dict[str, Any]:
        """
        Perform comprehensive research on any financial topic using web search and AI analysis.

        This tool provides deep research capabilities including:
        - Multi-source web search (news, reports, analysis)
        - Content analysis and sentiment detection
        - Source credibility assessment
        - Persona-aware insights and recommendations
        - Citation management and fact validation

        Perfect for researching stocks, sectors, market trends, company analysis,
        and any financial topic requiring comprehensive investigation.

        Args:
            request: Research parameters including query, persona, scope, and timeframe

        Returns:
            Comprehensive research results with insights, sentiment, and recommendations
        """
        # Log tool invocation
        log_tool_invocation(
            "comprehensive_research",
            {
                "query": request.query[:100],  # Truncate for logging
                "persona": request.persona,
                "research_scope": request.research_scope,
                "max_sources": request.max_sources,
                "timeframe": request.timeframe,
            },
        )

        start_time = datetime.now()

        try:
            agent = get_research_agent()

            # Set persona if provided
            if request.persona and request.persona in [
                "conservative",
                "moderate",
                "aggressive",
                "day_trader",
            ]:
                agent.persona = agent.persona.__class__.get(request.persona)

            # Generate session ID if not provided
            session_id = request.session_id or f"research_{datetime.now().timestamp()}"

            # Perform research
            result = await agent.research_topic(
                query=request.query,
                session_id=session_id,
                research_scope=request.research_scope,
                max_sources=request.max_sources,
                timeframe=request.timeframe,
            )

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log performance metrics
            log_performance_metrics(
                "comprehensive_research",
                {
                    "execution_time_ms": execution_time,
                    "sources_analyzed": result.get("sources_found", 0),
                    "confidence_score": result.get("research_confidence", 0.0),
                    "success": "error" not in result,
                },
            )

            # Format response for MCP
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "query": request.query,
                    "timestamp": datetime.now().isoformat(),
                }

            return {
                "success": True,
                "query": request.query,
                "research_results": {
                    "summary": result.get("content", "Research completed"),
                    "confidence_score": result.get("research_confidence", 0.0),
                    "sources_analyzed": result.get("sources_found", 0),
                    "persona_insights": result.get("persona_insights", {}),
                    "key_themes": result.get("content_analysis", {}).get(
                        "key_themes", []
                    ),
                    "sentiment": result.get("content_analysis", {}).get(
                        "consensus_view", {}
                    ),
                    "actionable_insights": result.get("actionable_insights", []),
                },
                "research_metadata": {
                    "persona": request.persona,
                    "scope": request.research_scope,
                    "timeframe": request.timeframe,
                    "max_sources": request.max_sources,
                    "execution_time_ms": execution_time,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Research error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": request.query,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.tool()
    async def analyze_market_sentiment(
        request: SentimentAnalysisRequest,
    ) -> dict[str, Any]:
        """
        Analyze market sentiment for stocks, sectors, or market trends.

        Performs targeted sentiment analysis by:
        - Searching recent news and social media
        - Analyzing sentiment across multiple sources
        - Identifying consensus vs contrarian views
        - Providing persona-specific sentiment interpretation
        - Tracking sentiment trends over time

        Ideal for understanding market mood, investor sentiment, and potential
        contrarian opportunities.

        Args:
            request: Sentiment analysis parameters

        Returns:
            Detailed sentiment analysis with directional bias and confidence scores
        """
        # Log tool invocation
        log_tool_invocation(
            "analyze_market_sentiment",
            {
                "topic": request.topic[:100],
                "timeframe": request.timeframe,
                "persona": request.persona,
            },
        )

        start_time = datetime.now()

        try:
            agent = get_research_agent()

            # Set persona if provided
            if request.persona and request.persona in [
                "conservative",
                "moderate",
                "aggressive",
                "day_trader",
            ]:
                agent.persona = agent.persona.__class__.get(request.persona)

            # Generate session ID if not provided
            session_id = request.session_id or f"sentiment_{datetime.now().timestamp()}"

            # Perform sentiment analysis
            result = await agent.analyze_market_sentiment(
                topic=request.topic, session_id=session_id, timeframe=request.timeframe
            )

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log performance metrics
            log_performance_metrics(
                "analyze_market_sentiment",
                {
                    "execution_time_ms": execution_time,
                    "sources_analyzed": result.get("sources_found", 0),
                    "confidence_score": result.get("research_confidence", 0.0),
                    "success": "error" not in result,
                },
            )

            # Format response
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "topic": request.topic,
                    "timestamp": datetime.now().isoformat(),
                }

            return {
                "success": True,
                "topic": request.topic,
                "sentiment_analysis": {
                    "overall_sentiment": result.get("content_analysis", {}).get(
                        "consensus_view", {}
                    ),
                    "sentiment_confidence": result.get("research_confidence", 0.0),
                    "sentiment_themes": result.get("content_analysis", {}).get(
                        "key_themes", []
                    ),
                    "contrarian_indicators": result.get("content_analysis", {}).get(
                        "contrarian_views", []
                    ),
                    "source_diversity": len(result.get("processed_sources", [])),
                    "persona_interpretation": result.get("persona_insights", {}),
                },
                "analysis_metadata": {
                    "timeframe": request.timeframe,
                    "persona": request.persona,
                    "sources_analyzed": result.get("sources_found", 0),
                    "execution_time_ms": execution_time,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": request.topic,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.tool()
    async def research_company_comprehensive(
        request: CompanyResearchRequest,
    ) -> dict[str, Any]:
        """
        Perform comprehensive company research and fundamental analysis.

        Conducts deep-dive company research including:
        - Financial performance and metrics analysis
        - Business model and competitive positioning
        - Growth prospects and future outlook
        - Risk factors and challenges
        - Industry and sector context
        - Analyst opinions and price targets
        - Recent news and developments

        Perfect for investment decision-making, due diligence, and comprehensive
        company evaluation.

        Args:
            request: Company research parameters

        Returns:
            Comprehensive company analysis with financial insights and recommendations
        """
        # Log tool invocation
        log_tool_invocation(
            "research_company_comprehensive",
            {
                "symbol": request.symbol,
                "include_competitive_analysis": request.include_competitive_analysis,
                "persona": request.persona,
            },
        )

        start_time = datetime.now()

        try:
            agent = get_research_agent()

            # Set persona if provided
            if request.persona and request.persona in [
                "conservative",
                "moderate",
                "aggressive",
                "day_trader",
            ]:
                agent.persona = agent.persona.__class__.get(request.persona)

            # Generate session ID if not provided
            session_id = (
                request.session_id
                or f"company_{request.symbol}_{datetime.now().timestamp()}"
            )

            # Perform company research
            result = await agent.research_company_comprehensive(
                symbol=request.symbol,
                session_id=session_id,
                include_competitive_analysis=request.include_competitive_analysis,
            )

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log performance metrics
            log_performance_metrics(
                "research_company_comprehensive",
                {
                    "execution_time_ms": execution_time,
                    "sources_analyzed": result.get("sources_found", 0),
                    "confidence_score": result.get("research_confidence", 0.0),
                    "success": "error" not in result,
                    "include_competitive": request.include_competitive_analysis,
                },
            )

            # Format response
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "symbol": request.symbol,
                    "timestamp": datetime.now().isoformat(),
                }

            return {
                "success": True,
                "symbol": request.symbol,
                "company_research": {
                    "executive_summary": result.get(
                        "content", "Company research completed"
                    ),
                    "research_confidence": result.get("research_confidence", 0.0),
                    "fundamental_insights": result.get("persona_insights", {}),
                    "business_analysis": result.get("content_analysis", {}).get(
                        "insights", []
                    ),
                    "competitive_position": result.get("content_analysis", {}).get(
                        "key_themes", []
                    ),
                    "growth_outlook": result.get("content_analysis", {}).get(
                        "consensus_view", {}
                    ),
                    "risk_factors": result.get("content_analysis", {}).get(
                        "contrarian_views", []
                    ),
                    "analyst_sentiment": result.get("content_analysis", {}).get(
                        "sentiment_scores", {}
                    ),
                },
                "research_metadata": {
                    "include_competitive_analysis": request.include_competitive_analysis,
                    "persona": request.persona,
                    "sources_analyzed": result.get("sources_found", 0),
                    "research_scope": "comprehensive",
                    "execution_time_ms": execution_time,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Company research error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": request.symbol,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.tool()
    async def search_financial_news(
        query: str,
        timeframe: str = "1w",
        max_results: int = 20,
        persona: str = "moderate",
    ) -> dict[str, Any]:
        """
        Search for recent financial news and analysis on any topic.

        Provides targeted news search with:
        - Multi-source news aggregation
        - Relevance scoring and filtering
        - Source credibility assessment
        - Summary generation
        - Persona-aware result prioritization

        Great for staying updated on market news, company developments,
        and sector trends.

        Args:
            query: Search query for financial news
            timeframe: Time frame for news search (1d, 1w, 1m)
            max_results: Maximum number of results to return
            persona: Investor persona for result filtering

        Returns:
            Curated financial news results with summaries and relevance scores
        """
        try:
            agent = get_research_agent()

            # Use the search_financial_news tool from the agent
            if hasattr(agent, "tools") and agent.tools:
                for tool in agent.tools:
                    if hasattr(tool, "name") and "search_financial_news" in tool.name:
                        result = await tool.arun(
                            query=query, timeframe=timeframe, max_results=max_results
                        )

                        if "error" in result:
                            return {
                                "success": False,
                                "error": result["error"],
                                "query": query,
                            }

                        return {
                            "success": True,
                            "query": query,
                            "news_results": result.get("results", []),
                            "total_found": result.get("total_found", 0),
                            "timeframe": timeframe,
                            "persona": persona,
                            "timestamp": datetime.now().isoformat(),
                        }

            # Fallback: use basic research
            result = await agent.research_topic(
                query=f"{query} news",
                session_id=f"news_{datetime.now().timestamp()}",
                research_scope="basic",
                max_sources=max_results,
                timeframe=timeframe,
            )

            return {
                "success": True,
                "query": query,
                "news_results": result.get("processed_sources", [])[:max_results],
                "total_found": len(result.get("processed_sources", [])),
                "timeframe": timeframe,
                "persona": persona,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"News search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.tool()
    async def validate_research_claims(
        claims: list[str], context: str | None = None
    ) -> dict[str, Any]:
        """
        Validate research claims and statements through fact-checking.

        Performs fact validation by:
        - Cross-referencing multiple authoritative sources
        - Checking for contradictory information
        - Assessing claim reliability and confidence
        - Identifying potential bias or misinformation
        - Providing evidence-based validation scores

        Useful for verifying research findings, analyst claims, and
        investment thesis validation.

        Args:
            claims: List of claims or statements to validate
            context: Optional context for validation

        Returns:
            Validation results with confidence scores and supporting evidence
        """
        try:
            agent = get_research_agent()

            validation_results = []

            for claim in claims:
                # Create validation query
                validation_query = f"fact check validate: {claim}"
                if context:
                    validation_query += f" in context of {context}"

                # Perform focused research for validation
                result = await agent.research_topic(
                    query=validation_query,
                    session_id=f"validation_{datetime.now().timestamp()}",
                    research_scope="standard",
                    max_sources=20,
                    timeframe="1m",
                )

                # Extract validation metrics
                validation_results.append(
                    {
                        "claim": claim,
                        "validation_confidence": result.get("research_confidence", 0.5),
                        "supporting_sources": len(result.get("processed_sources", [])),
                        "validation_summary": result.get(
                            "content", "Validation completed"
                        ),
                        "consensus_support": result.get("content_analysis", {}).get(
                            "consensus_view", {}
                        ),
                        "contradictory_evidence": result.get(
                            "content_analysis", {}
                        ).get("contrarian_views", []),
                    }
                )

            # Calculate overall validation score
            avg_confidence = (
                sum(r["validation_confidence"] for r in validation_results)
                / len(validation_results)
                if validation_results
                else 0
            )

            return {
                "success": True,
                "validation_results": validation_results,
                "overall_confidence": avg_confidence,
                "claims_validated": len(claims),
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Claim validation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "claims": claims,
                "timestamp": datetime.now().isoformat(),
            }

    return mcp


# Create the router instance
research_router = create_research_router()
