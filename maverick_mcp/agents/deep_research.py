"""
DeepResearchAgent implementation using 2025 LangGraph patterns.

Provides comprehensive financial research capabilities with web search,
content analysis, sentiment detection, and source validation.
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from maverick_mcp.agents.base import PersonaAwareAgent
from maverick_mcp.agents.circuit_breaker import circuit_manager
from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import (
    WebSearchError,
)
from maverick_mcp.memory.stores import ConversationStore
from maverick_mcp.utils.orchestration_logging import (
    get_orchestration_logger,
    log_agent_execution,
    log_method_call,
    log_performance_metrics,
    log_synthesis_operation,
)
from maverick_mcp.utils.parallel_research import (
    ParallelResearchConfig,
    ParallelResearchOrchestrator,
    ResearchResult,
    ResearchTask,
    TaskDistributionEngine,
)
from maverick_mcp.workflows.state import DeepResearchState

logger = logging.getLogger(__name__)
settings = get_settings()

# Research depth levels with different scopes
RESEARCH_DEPTH_LEVELS = {
    "basic": {
        "max_sources": 3,
        "max_searches": 2,
        "analysis_depth": "summary",
        "validation_required": False,
    },
    "standard": {
        "max_sources": 8,
        "max_searches": 4,
        "analysis_depth": "detailed",
        "validation_required": True,
    },
    "comprehensive": {
        "max_sources": 15,
        "max_searches": 6,
        "analysis_depth": "comprehensive",
        "validation_required": True,
    },
    "exhaustive": {
        "max_sources": 25,
        "max_searches": 10,
        "analysis_depth": "exhaustive",
        "validation_required": True,
    },
}

# Persona-specific research focus areas
PERSONA_RESEARCH_FOCUS = {
    "conservative": {
        "keywords": [
            "dividend",
            "stability",
            "risk",
            "debt",
            "cash flow",
            "established",
        ],
        "sources": [
            "sec filings",
            "annual reports",
            "rating agencies",
            "dividend history",
        ],
        "risk_focus": "downside protection",
        "time_horizon": "long-term",
    },
    "moderate": {
        "keywords": ["growth", "value", "balance", "diversification", "fundamentals"],
        "sources": ["financial statements", "analyst reports", "industry analysis"],
        "risk_focus": "risk-adjusted returns",
        "time_horizon": "medium-term",
    },
    "aggressive": {
        "keywords": ["growth", "momentum", "opportunity", "innovation", "expansion"],
        "sources": [
            "news",
            "earnings calls",
            "industry trends",
            "competitive analysis",
        ],
        "risk_focus": "upside potential",
        "time_horizon": "short to medium-term",
    },
    "day_trader": {
        "keywords": [
            "catalysts",
            "earnings",
            "news",
            "volume",
            "volatility",
            "momentum",
        ],
        "sources": ["breaking news", "social sentiment", "earnings announcements"],
        "risk_focus": "short-term risks",
        "time_horizon": "intraday to weekly",
    },
}


class WebSearchProvider:
    """Base class for web search providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = None  # Implement rate limiting

    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Perform web search and return results."""
        raise NotImplementedError

    async def get_content(self, url: str) -> dict[str, Any]:
        """Extract content from URL."""
        raise NotImplementedError


class ExaSearchProvider(WebSearchProvider):
    """Exa AI search provider for high-quality research content."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from exa_py import Exa

            self.client = Exa(api_key=api_key)
        except ImportError:
            raise ImportError("exa-py library required for ExaSearchProvider")

    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Search using Exa AI for high-quality content."""
        try:
            circuit_breaker = await circuit_manager.get_or_create("exa_search")

            async def _search():
                response = self.client.search_and_contents(
                    query=query,
                    num_results=num_results,
                    text=True,
                    highlights=True,
                    summary=True,
                )

                results = []
                for result in response.results:
                    results.append(
                        {
                            "url": result.url,
                            "title": result.title,
                            "content": result.text[:2000] if result.text else "",
                            "summary": result.summary,
                            "highlights": result.highlights,
                            "published_date": result.published_date,
                            "author": result.author,
                            "score": result.score,
                            "provider": "exa",
                        }
                    )

                return results

            return await circuit_breaker.call(_search)

        except Exception as e:
            logger.error(f"Exa search error: {e}")
            raise WebSearchError(f"Exa search failed: {str(e)}")


class TavilySearchProvider(WebSearchProvider):
    """Tavily search provider for comprehensive web search."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from tavily import TavilyClient

            self.client = TavilyClient(api_key=api_key)
        except ImportError:
            raise ImportError("tavily-python library required for TavilySearchProvider")

    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Search using Tavily for comprehensive web results."""
        try:
            circuit_breaker = await circuit_manager.get_or_create("tavily_search")

            async def _search():
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=num_results,
                    include_domains=None,
                    exclude_domains=[
                        "facebook.com",
                        "twitter.com",
                    ],  # Filter social media for financial research
                    include_answer=True,
                    include_raw_content=True,
                )

                results = []
                for result in response.get("results", []):
                    results.append(
                        {
                            "url": result.get("url"),
                            "title": result.get("title"),
                            "content": result.get("content", "")[:2000],
                            "raw_content": result.get("raw_content", "")[:3000],
                            "published_date": result.get("published_date"),
                            "score": result.get("score", 0.5),
                            "provider": "tavily",
                        }
                    )

                return results

            return await circuit_breaker.call(_search)

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            raise WebSearchError(f"Tavily search failed: {str(e)}")


class ContentAnalyzer:
    """AI-powered content analysis for research results."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def analyze_content(
        self, content: str, persona: str, analysis_focus: str = "general"
    ) -> dict[str, Any]:
        """Analyze content with AI for insights, sentiment, and relevance."""

        persona_focus = PERSONA_RESEARCH_FOCUS.get(
            persona, PERSONA_RESEARCH_FOCUS["moderate"]
        )

        analysis_prompt = f"""
        Analyze this financial content from the perspective of a {persona} investor.

        Content to analyze:
        {content[:3000]}  # Limit content length

        Focus Areas: {", ".join(persona_focus["keywords"])}
        Risk Focus: {persona_focus["risk_focus"]}
        Time Horizon: {persona_focus["time_horizon"]}

        Provide analysis in the following structure:

        1. KEY_INSIGHTS: 3-5 bullet points of most important insights
        2. SENTIMENT: Overall sentiment (bullish/bearish/neutral) with confidence (0-1)
        3. RISK_FACTORS: Key risks identified relevant to {persona} investors
        4. OPPORTUNITIES: Investment opportunities or catalysts identified
        5. CREDIBILITY: Assessment of source credibility (0-1 score)
        6. RELEVANCE: How relevant is this to {persona} investment strategy (0-1 score)
        7. SUMMARY: 2-3 sentence summary for {persona} investors

        Format as JSON with clear structure.
        """

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a financial content analyst. Return only valid JSON."
                    ),
                    HumanMessage(content=analysis_prompt),
                ]
            )

            # Parse AI response
            import json

            analysis = json.loads(response.content.strip())

            return {
                "insights": analysis.get("KEY_INSIGHTS", []),
                "sentiment": {
                    "direction": analysis.get("SENTIMENT", {}).get(
                        "direction", "neutral"
                    ),
                    "confidence": analysis.get("SENTIMENT", {}).get("confidence", 0.5),
                },
                "risk_factors": analysis.get("RISK_FACTORS", []),
                "opportunities": analysis.get("OPPORTUNITIES", []),
                "credibility_score": analysis.get("CREDIBILITY", 0.5),
                "relevance_score": analysis.get("RELEVANCE", 0.5),
                "summary": analysis.get("SUMMARY", ""),
                "analysis_timestamp": datetime.now(),
            }

        except Exception as e:
            logger.warning(f"AI content analysis failed: {e}, using fallback")
            return self._fallback_analysis(content, persona)

    def _fallback_analysis(self, content: str, persona: str) -> dict[str, Any]:
        """Fallback analysis using keyword matching."""
        persona_focus = PERSONA_RESEARCH_FOCUS.get(
            persona, PERSONA_RESEARCH_FOCUS["moderate"]
        )

        content_lower = content.lower()

        # Simple sentiment analysis
        positive_words = [
            "growth",
            "increase",
            "profit",
            "success",
            "opportunity",
            "strong",
        ]
        negative_words = ["decline", "loss", "risk", "problem", "concern", "weak"]

        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            sentiment = "bullish"
            confidence = 0.6
        elif negative_count > positive_count:
            sentiment = "bearish"
            confidence = 0.6
        else:
            sentiment = "neutral"
            confidence = 0.5

        # Relevance scoring based on keywords
        keyword_matches = sum(
            1 for keyword in persona_focus["keywords"] if keyword in content_lower
        )
        relevance_score = min(keyword_matches / len(persona_focus["keywords"]), 1.0)

        return {
            "insights": [f"Fallback analysis for {persona} investor perspective"],
            "sentiment": {"direction": sentiment, "confidence": confidence},
            "risk_factors": ["Unable to perform detailed risk analysis"],
            "opportunities": ["Unable to identify specific opportunities"],
            "credibility_score": 0.5,
            "relevance_score": relevance_score,
            "summary": f"Content analysis for {persona} investor using fallback method",
            "analysis_timestamp": datetime.now(),
            "fallback_used": True,
        }


class DeepResearchAgent(PersonaAwareAgent):
    """
    Deep research agent using 2025 LangGraph patterns.

    Provides comprehensive financial research with web search, content analysis,
    sentiment detection, and source validation.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        persona: str = "moderate",
        checkpointer: MemorySaver | None = None,
        ttl_hours: int = 24,  # Research results cached longer
        exa_api_key: str | None = None,
        tavily_api_key: str | None = None,
        default_depth: str = "standard",
        max_sources: int | None = None,
        research_depth: str | None = None,
        enable_parallel_execution: bool = True,
        parallel_config: ParallelResearchConfig | None = None,
    ):
        """Initialize deep research agent."""

        # Initialize search providers
        self.search_providers = []

        if exa_api_key:
            try:
                self.search_providers.append(ExaSearchProvider(exa_api_key))
                logger.info("Initialized Exa search provider")
            except ImportError as e:
                logger.warning(f"Failed to initialize Exa provider: {e}")

        if tavily_api_key:
            try:
                self.search_providers.append(TavilySearchProvider(tavily_api_key))
                logger.info("Initialized Tavily search provider")
            except ImportError as e:
                logger.warning(f"Failed to initialize Tavily provider: {e}")

        if not self.search_providers:
            logger.warning(
                "No search providers available - research capabilities will be limited"
            )

        # Configuration
        self.default_depth = research_depth or default_depth
        self.max_sources = max_sources or RESEARCH_DEPTH_LEVELS.get(
            self.default_depth, {}
        ).get("max_sources", 10)
        self.content_analyzer = ContentAnalyzer(llm)

        # Parallel execution configuration
        self.enable_parallel_execution = enable_parallel_execution
        self.parallel_config = parallel_config or ParallelResearchConfig(
            max_concurrent_agents=settings.data_limits.max_parallel_agents,
            timeout_per_agent=300,
            enable_fallbacks=True,
            rate_limit_delay=1.0,
        )
        self.parallel_orchestrator = ParallelResearchOrchestrator(self.parallel_config)
        self.task_distributor = TaskDistributionEngine()

        # Get research-specific tools
        research_tools = self._get_research_tools()

        # Initialize base class
        super().__init__(
            llm=llm,
            tools=research_tools,
            persona=persona,
            checkpointer=checkpointer or MemorySaver(),
            ttl_hours=ttl_hours,
        )

        # Initialize components
        self.conversation_store = ConversationStore(ttl_hours=ttl_hours)

        logger.info(
            f"DeepResearchAgent initialized with {len(self.search_providers)} search providers, "
            f"parallel execution: {self.enable_parallel_execution}"
        )

    def get_state_schema(self) -> type:
        """Return DeepResearchState schema."""
        return DeepResearchState

    def _get_research_tools(self) -> list[BaseTool]:
        """Get tools specific to research capabilities."""
        tools = []

        @tool
        async def web_search_financial(
            query: str, num_results: int = 10, provider: str = "auto"
        ) -> dict[str, Any]:
            """Search the web for financial information using available providers."""
            return await self._perform_web_search(query, num_results, provider)

        @tool
        async def analyze_company_fundamentals(
            symbol: str, depth: str = "standard"
        ) -> dict[str, Any]:
            """Research company fundamentals including financials, competitive position, and outlook."""
            return await self._research_company_fundamentals(symbol, depth)

        @tool
        async def analyze_market_sentiment(
            topic: str, timeframe: str = "7d"
        ) -> dict[str, Any]:
            """Analyze market sentiment around a topic using news and social signals."""
            return await self._analyze_market_sentiment(topic, timeframe)

        @tool
        async def validate_research_claims(
            claims: list[str], sources: list[str]
        ) -> dict[str, Any]:
            """Validate research claims against multiple sources for fact-checking."""
            return await self._validate_claims(claims, sources)

        tools.extend(
            [
                web_search_financial,
                analyze_company_fundamentals,
                analyze_market_sentiment,
                validate_research_claims,
            ]
        )

        return tools

    def _build_graph(self):
        """Build research workflow graph with multi-step research process."""
        workflow = StateGraph(DeepResearchState)

        # Core research workflow nodes
        workflow.add_node("plan_research", self._plan_research)
        workflow.add_node("execute_searches", self._execute_searches)
        workflow.add_node("analyze_content", self._analyze_content)
        workflow.add_node("validate_sources", self._validate_sources)
        workflow.add_node("synthesize_findings", self._synthesize_findings)
        workflow.add_node("generate_citations", self._generate_citations)

        # Specialized research nodes
        workflow.add_node("sentiment_analysis", self._sentiment_analysis)
        workflow.add_node("fundamental_analysis", self._fundamental_analysis)
        workflow.add_node("competitive_analysis", self._competitive_analysis)

        # Quality control nodes
        workflow.add_node("fact_validation", self._fact_validation)
        workflow.add_node("source_credibility", self._source_credibility)

        # Define workflow edges
        workflow.add_edge(START, "plan_research")
        workflow.add_edge("plan_research", "execute_searches")
        workflow.add_edge("execute_searches", "analyze_content")

        # Conditional routing based on research type
        workflow.add_conditional_edges(
            "analyze_content",
            self._route_specialized_analysis,
            {
                "sentiment": "sentiment_analysis",
                "fundamental": "fundamental_analysis",
                "competitive": "competitive_analysis",
                "validation": "validate_sources",
                "synthesis": "synthesize_findings",
            },
        )

        # Specialized analysis flows
        workflow.add_edge("sentiment_analysis", "validate_sources")
        workflow.add_edge("fundamental_analysis", "validate_sources")
        workflow.add_edge("competitive_analysis", "validate_sources")

        # Quality control flow
        workflow.add_edge("validate_sources", "fact_validation")
        workflow.add_edge("fact_validation", "source_credibility")
        workflow.add_edge("source_credibility", "synthesize_findings")

        # Final steps
        workflow.add_edge("synthesize_findings", "generate_citations")
        workflow.add_edge("generate_citations", END)

        return workflow.compile(checkpointer=self.checkpointer)

    @log_method_call(component="DeepResearchAgent", include_timing=True)
    async def research_comprehensive(
        self,
        topic: str,
        session_id: str,
        depth: str | None = None,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Comprehensive research on a financial topic.

        Args:
            topic: Research topic or company/symbol
            session_id: Session identifier
            depth: Research depth (basic/standard/comprehensive/exhaustive)
            focus_areas: Specific areas to focus on
            timeframe: Time range for research
            **kwargs: Additional parameters

        Returns:
            Comprehensive research results with analysis and citations
        """
        start_time = datetime.now()
        depth = depth or self.default_depth

        # Initialize research state
        initial_state = {
            "messages": [HumanMessage(content=f"Research: {topic}")],
            "persona": self.persona.name,
            "session_id": session_id,
            "timestamp": datetime.now(),
            "research_topic": topic,
            "research_depth": depth,
            "focus_areas": focus_areas
            or PERSONA_RESEARCH_FOCUS[self.persona.name.lower()]["keywords"],
            "timeframe": timeframe,
            "search_queries": [],
            "search_results": [],
            "analyzed_content": [],
            "validated_sources": [],
            "research_findings": [],
            "sentiment_analysis": {},
            "source_credibility_scores": {},
            "citations": [],
            "research_status": "planning",
            "research_confidence": 0.0,
            "source_diversity_score": 0.0,
            "fact_validation_results": [],
            "execution_time_ms": 0.0,
            "api_calls_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            # Legacy fields
            "token_count": 0,
            "error": None,
            "analyzed_stocks": {},
            "key_price_levels": {},
            "last_analysis_time": {},
            "conversation_context": {},
        }

        # Add additional parameters
        initial_state.update(kwargs)

        # Set up orchestration logging
        orchestration_logger = get_orchestration_logger("DeepResearchAgent")
        orchestration_logger.set_request_context(
            session_id=session_id,
            research_topic=topic[:50],  # Truncate for logging
            research_depth=depth,
        )

        # Check if parallel execution is enabled and requested
        use_parallel = kwargs.get(
            "use_parallel_execution", self.enable_parallel_execution
        )

        orchestration_logger.info(
            "🔍 RESEARCH_START",
            execution_mode="parallel" if use_parallel else "sequential",
            focus_areas=focus_areas[:3] if focus_areas else None,
            timeframe=timeframe,
        )

        if use_parallel:
            orchestration_logger.info("🚀 PARALLEL_EXECUTION_SELECTED")
            try:
                result = await self._execute_parallel_research(
                    topic=topic,
                    session_id=session_id,
                    depth=depth,
                    focus_areas=focus_areas,
                    timeframe=timeframe,
                    initial_state=initial_state,
                    start_time=start_time,
                    **kwargs,
                )
                orchestration_logger.info("✅ PARALLEL_EXECUTION_SUCCESS")
                return result
            except Exception as e:
                orchestration_logger.warning(
                    "⚠️ PARALLEL_FALLBACK_TRIGGERED",
                    error=str(e),
                    fallback_mode="sequential",
                )
                # Fall through to sequential execution

        # Execute research workflow (sequential)
        orchestration_logger.info("🔄 SEQUENTIAL_EXECUTION_START")
        try:
            result = await self.graph.ainvoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": session_id,
                        "checkpoint_ns": "deep_research",
                    }
                },
            )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result["execution_time_ms"] = execution_time

            return self._format_research_response(result)

        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": (datetime.now() - start_time).total_seconds()
                * 1000,
                "agent_type": "deep_research",
            }

    # Workflow node implementations

    async def _plan_research(self, state: DeepResearchState) -> Command:
        """Plan research strategy based on topic and persona."""
        topic = state["research_topic"]
        depth_config = RESEARCH_DEPTH_LEVELS[state["research_depth"]]
        persona_focus = PERSONA_RESEARCH_FOCUS[self.persona.name.lower()]

        # Generate search queries based on topic and persona
        search_queries = await self._generate_search_queries(
            topic, persona_focus, depth_config
        )

        return Command(
            goto="execute_searches",
            update={"search_queries": search_queries, "research_status": "searching"},
        )

    async def _execute_searches(self, state: DeepResearchState) -> Command:
        """Execute web searches using available providers."""
        search_queries = state["search_queries"]
        depth_config = RESEARCH_DEPTH_LEVELS[state["research_depth"]]

        all_results = []

        # Execute searches in parallel across providers
        for query in search_queries[: depth_config["max_searches"]]:
            for provider in self.search_providers:
                try:
                    results = await provider.search(query, num_results=5)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(
                        f"Search failed for {query} with provider {type(provider).__name__}: {e}"
                    )

        # Deduplicate and limit results
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if (
                result["url"] not in seen_urls
                and len(unique_results) < depth_config["max_sources"]
            ):
                unique_results.append(result)
                seen_urls.add(result["url"])

        return Command(
            goto="analyze_content",
            update={"search_results": unique_results, "research_status": "analyzing"},
        )

    async def _analyze_content(self, state: DeepResearchState) -> Command:
        """Analyze search results using AI content analysis."""
        search_results = state["search_results"]
        analyzed_content = []

        # Analyze each piece of content
        for result in search_results:
            if result.get("content"):
                analysis = await self.content_analyzer.analyze_content(
                    content=result["content"],
                    persona=self.persona.name.lower(),
                    analysis_focus=state["research_depth"],
                )

                analyzed_content.append({**result, "analysis": analysis})

        return Command(
            goto="validate_sources",
            update={
                "analyzed_content": analyzed_content,
                "research_status": "validating",
            },
        )

    def _route_specialized_analysis(self, state: DeepResearchState) -> str:
        """Route to specialized analysis based on research focus."""
        focus_areas = state.get("focus_areas", [])

        if any(word in focus_areas for word in ["sentiment", "news", "social"]):
            return "sentiment"
        elif any(
            word in focus_areas for word in ["fundamental", "financial", "earnings"]
        ):
            return "fundamental"
        elif any(word in focus_areas for word in ["competitive", "market", "industry"]):
            return "competitive"
        else:
            return "validation"

    async def _validate_sources(self, state: DeepResearchState) -> Command:
        """Validate source credibility and filter results."""
        analyzed_content = state["analyzed_content"]
        validated_sources = []
        credibility_scores = {}

        for content in analyzed_content:
            # Calculate credibility score based on multiple factors
            credibility_score = self._calculate_source_credibility(content)
            credibility_scores[content["url"]] = credibility_score

            # Only include sources above credibility threshold
            if credibility_score >= 0.6:  # Configurable threshold
                validated_sources.append(content)

        return Command(
            goto="synthesize_findings",
            update={
                "validated_sources": validated_sources,
                "source_credibility_scores": credibility_scores,
                "research_status": "synthesizing",
            },
        )

    async def _synthesize_findings(self, state: DeepResearchState) -> Command:
        """Synthesize research findings into coherent insights."""
        validated_sources = state["validated_sources"]

        # Generate synthesis using LLM
        synthesis_prompt = self._build_synthesis_prompt(validated_sources, state)

        synthesis_response = await self.llm.ainvoke(
            [
                SystemMessage(content="You are a financial research synthesizer."),
                HumanMessage(content=synthesis_prompt),
            ]
        )

        research_findings = {
            "synthesis": synthesis_response.content,
            "key_insights": self._extract_key_insights(validated_sources),
            "overall_sentiment": self._calculate_overall_sentiment(validated_sources),
            "risk_assessment": self._assess_risks(validated_sources),
            "investment_implications": self._derive_investment_implications(
                validated_sources
            ),
            "confidence_score": self._calculate_research_confidence(validated_sources),
        }

        return Command(
            goto="generate_citations",
            update={
                "research_findings": research_findings,
                "research_confidence": research_findings["confidence_score"],
                "research_status": "completing",
            },
        )

    async def _generate_citations(self, state: DeepResearchState) -> Command:
        """Generate proper citations for all sources."""
        validated_sources = state["validated_sources"]

        citations = []
        for i, source in enumerate(validated_sources, 1):
            citation = {
                "id": i,
                "title": source.get("title", "Untitled"),
                "url": source["url"],
                "published_date": source.get("published_date"),
                "author": source.get("author"),
                "credibility_score": state["source_credibility_scores"].get(
                    source["url"], 0.5
                ),
                "relevance_score": source.get("analysis", {}).get(
                    "relevance_score", 0.5
                ),
            }
            citations.append(citation)

        return Command(
            goto="__end__",
            update={"citations": citations, "research_status": "completed"},
        )

    # Helper methods

    async def _generate_search_queries(
        self, topic: str, persona_focus: dict[str, Any], depth_config: dict[str, Any]
    ) -> list[str]:
        """Generate search queries optimized for the research topic and persona."""

        base_queries = [
            f"{topic} financial analysis",
            f"{topic} investment research",
            f"{topic} market outlook",
        ]

        # Add persona-specific queries
        persona_queries = [
            f"{topic} {keyword}" for keyword in persona_focus["keywords"][:3]
        ]

        # Add source-specific queries
        source_queries = [
            f"{topic} {source}" for source in persona_focus["sources"][:2]
        ]

        all_queries = base_queries + persona_queries + source_queries
        return all_queries[: depth_config["max_searches"]]

    def _calculate_source_credibility(self, content: dict[str, Any]) -> float:
        """Calculate credibility score for a source."""
        score = 0.5  # Base score

        # Domain credibility
        url = content.get("url", "")
        if any(domain in url for domain in [".gov", ".edu", ".org"]):
            score += 0.2
        elif any(
            domain in url
            for domain in [
                "sec.gov",
                "investopedia.com",
                "bloomberg.com",
                "reuters.com",
            ]
        ):
            score += 0.3

        # Publication date recency
        pub_date = content.get("published_date")
        if pub_date:
            try:
                date_obj = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                days_old = (datetime.now() - date_obj).days
                if days_old < 30:
                    score += 0.1
                elif days_old < 90:
                    score += 0.05
            except (ValueError, TypeError, AttributeError):
                pass

        # Content analysis credibility
        if "analysis" in content:
            analysis_cred = content["analysis"].get("credibility_score", 0.5)
            score = (score + analysis_cred) / 2

        return min(score, 1.0)

    def _build_synthesis_prompt(
        self, sources: list[dict[str, Any]], state: DeepResearchState
    ) -> str:
        """Build synthesis prompt for final research output."""
        topic = state["research_topic"]
        persona = self.persona.name

        prompt = f"""
        Synthesize comprehensive research findings on '{topic}' for a {persona} investor.

        Research Sources ({len(sources)} validated sources):
        """

        for i, source in enumerate(sources, 1):
            analysis = source.get("analysis", {})
            prompt += f"\n{i}. {source.get('title', 'Unknown Title')}"
            prompt += f"   - Insights: {', '.join(analysis.get('insights', [])[:2])}"
            prompt += f"   - Sentiment: {analysis.get('sentiment', {}).get('direction', 'neutral')}"
            prompt += f"   - Credibility: {state['source_credibility_scores'].get(source['url'], 0.5):.2f}"

        prompt += f"""

        Please provide a comprehensive synthesis that includes:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (5-7 bullet points)
        3. Investment Implications for {persona} investors
        4. Risk Considerations
        5. Recommended Actions
        6. Confidence Level and reasoning

        Tailor the analysis specifically for {persona} investment characteristics and risk tolerance.
        """

        return prompt

    def _extract_key_insights(self, sources: list[dict[str, Any]]) -> list[str]:
        """Extract and consolidate key insights from all sources."""
        all_insights = []
        for source in sources:
            analysis = source.get("analysis", {})
            insights = analysis.get("insights", [])
            all_insights.extend(insights)

        # Simple deduplication (could be enhanced with semantic similarity)
        unique_insights = list(dict.fromkeys(all_insights))
        return unique_insights[:10]  # Return top 10 insights

    def _calculate_overall_sentiment(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate overall sentiment from all sources."""
        sentiments = []
        weights = []

        for source in sources:
            analysis = source.get("analysis", {})
            sentiment = analysis.get("sentiment", {})

            # Convert sentiment to numeric value
            direction = sentiment.get("direction", "neutral")
            if direction == "bullish":
                sentiment_value = 1
            elif direction == "bearish":
                sentiment_value = -1
            else:
                sentiment_value = 0

            confidence = sentiment.get("confidence", 0.5)
            credibility = source.get("credibility_score", 0.5)

            sentiments.append(sentiment_value)
            weights.append(confidence * credibility)

        if not sentiments:
            return {"direction": "neutral", "confidence": 0.5, "consensus": 0.5}

        # Weighted average sentiment
        weighted_sentiment = sum(
            s * w for s, w in zip(sentiments, weights, strict=False)
        ) / sum(weights)

        # Convert back to direction
        if weighted_sentiment > 0.2:
            overall_direction = "bullish"
        elif weighted_sentiment < -0.2:
            overall_direction = "bearish"
        else:
            overall_direction = "neutral"

        # Calculate consensus (how much sources agree)
        sentiment_variance = sum(weights) / len(sentiments) if sentiments else 0
        consensus = 1 - sentiment_variance if sentiment_variance < 1 else 0

        return {
            "direction": overall_direction,
            "confidence": abs(weighted_sentiment),
            "consensus": consensus,
            "source_count": len(sentiments),
        }

    def _assess_risks(self, sources: list[dict[str, Any]]) -> list[str]:
        """Consolidate risk assessments from all sources."""
        all_risks = []
        for source in sources:
            analysis = source.get("analysis", {})
            risks = analysis.get("risk_factors", [])
            all_risks.extend(risks)

        # Deduplicate and return top risks
        unique_risks = list(dict.fromkeys(all_risks))
        return unique_risks[:8]

    def _derive_investment_implications(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Derive investment implications based on research findings."""
        opportunities = []
        threats = []

        for source in sources:
            analysis = source.get("analysis", {})
            opps = analysis.get("opportunities", [])
            risks = analysis.get("risk_factors", [])

            opportunities.extend(opps)
            threats.extend(risks)

        return {
            "opportunities": list(dict.fromkeys(opportunities))[:5],
            "threats": list(dict.fromkeys(threats))[:5],
            "recommended_action": self._recommend_action(sources),
            "time_horizon": PERSONA_RESEARCH_FOCUS[self.persona.name.lower()][
                "time_horizon"
            ],
        }

    def _recommend_action(self, sources: list[dict[str, Any]]) -> str:
        """Recommend investment action based on research findings."""
        overall_sentiment = self._calculate_overall_sentiment(sources)

        if (
            overall_sentiment["direction"] == "bullish"
            and overall_sentiment["confidence"] > 0.7
        ):
            if self.persona.name.lower() == "conservative":
                return "Consider gradual position building with proper risk management"
            else:
                return "Consider initiating position with appropriate position sizing"
        elif (
            overall_sentiment["direction"] == "bearish"
            and overall_sentiment["confidence"] > 0.7
        ):
            return "Exercise caution - consider waiting for better entry or avoiding"
        else:
            return "Monitor closely - mixed signals suggest waiting for clarity"

    def _calculate_research_confidence(self, sources: list[dict[str, Any]]) -> float:
        """Calculate overall confidence in research findings."""
        if not sources:
            return 0.0

        # Factors that increase confidence
        source_count_factor = min(
            len(sources) / 10, 1.0
        )  # More sources = higher confidence

        avg_credibility = sum(
            source.get("credibility_score", 0.5) for source in sources
        ) / len(sources)

        avg_relevance = sum(
            source.get("analysis", {}).get("relevance_score", 0.5) for source in sources
        ) / len(sources)

        # Diversity of sources (different domains)
        unique_domains = len(
            {source["url"].split("/")[2] for source in sources if "url" in source}
        )
        diversity_factor = min(unique_domains / 5, 1.0)

        # Combine factors
        confidence = (
            source_count_factor + avg_credibility + avg_relevance + diversity_factor
        ) / 4

        return round(confidence, 2)

    def _format_research_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Format research response for consistent output."""
        return {
            "status": "success",
            "agent_type": "deep_research",
            "persona": result.get("persona"),
            "research_topic": result.get("research_topic"),
            "research_depth": result.get("research_depth"),
            "findings": result.get("research_findings", {}),
            "sources_analyzed": len(result.get("validated_sources", [])),
            "confidence_score": result.get("research_confidence", 0.0),
            "citations": result.get("citations", []),
            "execution_time_ms": result.get("execution_time_ms", 0.0),
            "search_queries_used": result.get("search_queries", []),
            "source_diversity": result.get("source_diversity_score", 0.0),
        }

    # Specialized research analysis methods
    async def _sentiment_analysis(self, state: DeepResearchState) -> Command:
        """Perform specialized sentiment analysis."""
        logger.info("Performing sentiment analysis")

        # For now, route to content analysis with sentiment focus
        original_focus = state.get("focus_areas", [])
        state["focus_areas"] = ["market_sentiment", "sentiment", "mood"]
        result = await self._analyze_content(state)
        state["focus_areas"] = original_focus  # Restore original focus
        return result

    async def _fundamental_analysis(self, state: DeepResearchState) -> Command:
        """Perform specialized fundamental analysis."""
        logger.info("Performing fundamental analysis")

        # For now, route to content analysis with fundamental focus
        original_focus = state.get("focus_areas", [])
        state["focus_areas"] = ["fundamentals", "financials", "valuation"]
        result = await self._analyze_content(state)
        state["focus_areas"] = original_focus  # Restore original focus
        return result

    async def _competitive_analysis(self, state: DeepResearchState) -> Command:
        """Perform specialized competitive analysis."""
        logger.info("Performing competitive analysis")

        # For now, route to content analysis with competitive focus
        original_focus = state.get("focus_areas", [])
        state["focus_areas"] = ["competitive_landscape", "market_share", "competitors"]
        result = await self._analyze_content(state)
        state["focus_areas"] = original_focus  # Restore original focus
        return result

    async def _fact_validation(self, state: DeepResearchState) -> Command:
        """Perform fact validation on research findings."""
        logger.info("Performing fact validation")

        # For now, route to source validation
        return await self._validate_sources(state)

    async def _source_credibility(self, state: DeepResearchState) -> Command:
        """Assess source credibility and reliability."""
        logger.info("Assessing source credibility")

        # For now, route to source validation
        return await self._validate_sources(state)

    async def research_company_comprehensive(
        self,
        symbol: str,
        session_id: str,
        include_competitive_analysis: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Comprehensive company research.

        Args:
            symbol: Stock symbol to research
            session_id: Session identifier
            include_competitive_analysis: Whether to include competitive analysis
            **kwargs: Additional parameters

        Returns:
            Comprehensive company research results
        """
        topic = f"{symbol} company financial analysis and outlook"
        if include_competitive_analysis:
            kwargs["focus_areas"] = kwargs.get("focus_areas", []) + [
                "competitive_analysis",
                "market_position",
            ]

        return await self.research_comprehensive(
            topic=topic, session_id=session_id, **kwargs
        )

    async def research_topic(
        self,
        query: str,
        session_id: str,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        **kwargs,
    ) -> dict[str, Any]:
        """
        General topic research.

        Args:
            query: Research query or topic
            session_id: Session identifier
            focus_areas: Specific areas to focus on
            timeframe: Time range for research
            **kwargs: Additional parameters

        Returns:
            Research results for the given topic
        """
        return await self.research_comprehensive(
            topic=query,
            session_id=session_id,
            focus_areas=focus_areas,
            timeframe=timeframe,
            **kwargs,
        )

    async def analyze_market_sentiment(
        self, topic: str, session_id: str, timeframe: str = "7d", **kwargs
    ) -> dict[str, Any]:
        """
        Analyze market sentiment around a topic.

        Args:
            topic: Topic to analyze sentiment for
            session_id: Session identifier
            timeframe: Time range for analysis
            **kwargs: Additional parameters

        Returns:
            Market sentiment analysis results
        """
        return await self.research_comprehensive(
            topic=f"market sentiment analysis: {topic}",
            session_id=session_id,
            focus_areas=["sentiment", "market_mood", "investor_sentiment"],
            timeframe=timeframe,
            **kwargs,
        )

    # Parallel Execution Implementation

    @log_method_call(component="DeepResearchAgent", include_timing=True)
    async def _execute_parallel_research(
        self,
        topic: str,
        session_id: str,
        depth: str,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        initial_state: dict[str, Any] | None = None,
        start_time: datetime | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute research using parallel subagent execution.

        Args:
            topic: Research topic
            session_id: Session identifier
            depth: Research depth level
            focus_areas: Specific focus areas
            timeframe: Research timeframe
            initial_state: Initial state for backward compatibility
            start_time: Start time for execution measurement
            **kwargs: Additional parameters

        Returns:
            Research results in same format as sequential execution
        """
        orchestration_logger = get_orchestration_logger("ParallelExecution")
        orchestration_logger.set_request_context(session_id=session_id)

        try:
            # Generate research tasks using task distributor
            orchestration_logger.info("🎯 TASK_DISTRIBUTION_START")
            research_tasks = self.task_distributor.distribute_research_tasks(
                topic=topic, session_id=session_id, focus_areas=focus_areas
            )

            orchestration_logger.info(
                "📋 TASKS_GENERATED",
                task_count=len(research_tasks),
                task_types=[t.task_type for t in research_tasks],
            )

            # Execute tasks in parallel
            orchestration_logger.info("🚀 PARALLEL_ORCHESTRATION_START")
            research_result = (
                await self.parallel_orchestrator.execute_parallel_research(
                    tasks=research_tasks,
                    research_executor=self._execute_subagent_task,
                    synthesis_callback=self._synthesize_parallel_results,
                )
            )

            # Log parallel execution metrics
            log_performance_metrics(
                "ParallelExecution",
                {
                    "total_tasks": research_result.successful_tasks
                    + research_result.failed_tasks,
                    "successful_tasks": research_result.successful_tasks,
                    "failed_tasks": research_result.failed_tasks,
                    "parallel_efficiency": research_result.parallel_efficiency,
                    "execution_time": research_result.total_execution_time,
                },
            )

            # Convert parallel results to expected format
            orchestration_logger.info("🔄 RESULT_FORMATTING_START")
            formatted_result = await self._format_parallel_research_response(
                research_result=research_result,
                topic=topic,
                session_id=session_id,
                depth=depth,
                initial_state=initial_state,
                start_time=start_time,
            )

            orchestration_logger.info(
                "✅ PARALLEL_RESEARCH_COMPLETE",
                result_confidence=formatted_result.get("confidence_score", 0.0),
                sources_analyzed=formatted_result.get("sources_analyzed", 0),
            )

            return formatted_result

        except Exception as e:
            orchestration_logger.error("❌ PARALLEL_RESEARCH_FAILED", error=str(e))
            raise  # Re-raise to trigger fallback to sequential

    async def _execute_subagent_task(self, task: ResearchTask) -> dict[str, Any]:
        """
        Execute a single research task using specialized subagent.

        Args:
            task: ResearchTask to execute

        Returns:
            Research results from specialized subagent
        """
        with log_agent_execution(
            task.task_type, task.task_id, task.focus_areas
        ) as agent_logger:
            agent_logger.info(
                "🎯 SUBAGENT_ROUTING",
                target_topic=task.target_topic[:50],
                focus_count=len(task.focus_areas),
                priority=task.priority,
            )

            # Route to appropriate subagent based on task type
            if task.task_type == "fundamental":
                subagent = FundamentalResearchAgent(self)
                return await subagent.execute_research(task)
            elif task.task_type == "technical":
                subagent = TechnicalResearchAgent(self)
                return await subagent.execute_research(task)
            elif task.task_type == "sentiment":
                subagent = SentimentResearchAgent(self)
                return await subagent.execute_research(task)
            elif task.task_type == "competitive":
                subagent = CompetitiveResearchAgent(self)
                return await subagent.execute_research(task)
            else:
                # Default to fundamental analysis
                agent_logger.warning("⚠️ UNKNOWN_TASK_TYPE", fallback="fundamental")
                subagent = FundamentalResearchAgent(self)
                return await subagent.execute_research(task)

    async def _synthesize_parallel_results(
        self, task_results: dict[str, ResearchTask]
    ) -> dict[str, Any]:
        """
        Synthesize results from multiple parallel research tasks.

        Args:
            task_results: Dictionary of task IDs to ResearchTask objects

        Returns:
            Synthesized research insights
        """
        synthesis_logger = get_orchestration_logger("ResultSynthesis")

        log_synthesis_operation(
            "parallel_research_synthesis",
            len(task_results),
            f"Synthesizing from {len(task_results)} research tasks",
        )

        # Extract successful results
        successful_results = {
            task_id: task.result
            for task_id, task in task_results.items()
            if task.status == "completed" and task.result
        }

        synthesis_logger.info(
            "📊 SYNTHESIS_INPUT_ANALYSIS",
            total_tasks=len(task_results),
            successful_tasks=len(successful_results),
            failed_tasks=len(task_results) - len(successful_results),
        )

        if not successful_results:
            synthesis_logger.warning("⚠️ NO_SUCCESSFUL_RESULTS")
            return {
                "synthesis": "No research results available for synthesis",
                "confidence_score": 0.0,
            }

        all_insights = []
        all_risks = []
        all_opportunities = []
        sentiment_scores = []
        credibility_scores = []

        # Aggregate results from all successful tasks
        for task_id, task in task_results.items():
            if task.status == "completed" and task.result:
                task_type = task_id.split("_")[-1] if "_" in task_id else "unknown"
                synthesis_logger.debug(
                    "🔍 PROCESSING_TASK_RESULT",
                    task_id=task_id,
                    task_type=task_type,
                    has_insights="insights" in task.result
                    if isinstance(task.result, dict)
                    else False,
                )

                result = task.result

                # Extract insights
                insights = result.get("insights", [])
                all_insights.extend(insights)

                # Extract risks and opportunities
                risks = result.get("risk_factors", [])
                opportunities = result.get("opportunities", [])
                all_risks.extend(risks)
                all_opportunities.extend(opportunities)

                # Extract sentiment
                sentiment = result.get("sentiment", {})
                if sentiment:
                    sentiment_scores.append(sentiment)

                # Extract credibility
                credibility = result.get("credibility_score", 0.5)
                credibility_scores.append(credibility)

        # Calculate overall metrics
        overall_sentiment = self._calculate_aggregated_sentiment(sentiment_scores)
        average_credibility = (
            sum(credibility_scores) / len(credibility_scores)
            if credibility_scores
            else 0.5
        )

        # Generate synthesis using LLM
        synthesis_prompt = self._build_parallel_synthesis_prompt(
            task_results, all_insights, all_risks, all_opportunities, overall_sentiment
        )

        try:
            synthesis_response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a financial research synthesizer. Combine insights from multiple specialized research agents."
                    ),
                    HumanMessage(content=synthesis_prompt),
                ]
            )

            synthesis_text = synthesis_response.content
            synthesis_logger.info("🧠 LLM_SYNTHESIS_SUCCESS")
        except Exception as e:
            synthesis_logger.warning(
                "⚠️ LLM_SYNTHESIS_FAILED", error=str(e), fallback="text_fallback"
            )
            synthesis_text = self._generate_fallback_synthesis(
                all_insights, overall_sentiment
            )

        synthesis_result = {
            "synthesis": synthesis_text,
            "key_insights": list(dict.fromkeys(all_insights))[
                :10
            ],  # Deduplicate and limit
            "overall_sentiment": overall_sentiment,
            "risk_assessment": list(dict.fromkeys(all_risks))[:8],
            "investment_implications": {
                "opportunities": list(dict.fromkeys(all_opportunities))[:5],
                "threats": list(dict.fromkeys(all_risks))[:5],
                "recommended_action": self._derive_parallel_recommendation(
                    overall_sentiment
                ),
            },
            "confidence_score": average_credibility,
            "task_breakdown": {
                task_id: {
                    "type": task.task_type,
                    "status": task.status,
                    "execution_time": (task.end_time - task.start_time)
                    if task.start_time and task.end_time
                    else 0,
                }
                for task_id, task in task_results.items()
            },
        }

        synthesis_logger.info(
            "✅ SYNTHESIS_COMPLETE",
            insights_count=len(all_insights),
            overall_confidence=average_credibility,
            sentiment_direction=synthesis_result["overall_sentiment"]["direction"],
        )

        return synthesis_result

    async def _format_parallel_research_response(
        self,
        research_result: ResearchResult,
        topic: str,
        session_id: str,
        depth: str,
        initial_state: dict[str, Any],
        start_time: datetime,
    ) -> dict[str, Any]:
        """Format parallel research results to match expected sequential format."""

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Extract synthesis from research result
        synthesis = research_result.synthesis or {}

        # Create citations from task results
        citations = []
        all_sources = []
        citation_id = 1

        for _task_id, task in research_result.task_results.items():
            if task.status == "completed" and task.result:
                sources = task.result.get("sources", [])
                for source in sources:
                    citation = {
                        "id": citation_id,
                        "title": source.get("title", "Unknown Title"),
                        "url": source.get("url", ""),
                        "published_date": source.get("published_date"),
                        "author": source.get("author"),
                        "credibility_score": source.get("credibility_score", 0.5),
                        "relevance_score": source.get("relevance_score", 0.5),
                        "research_type": task.task_type,
                    }
                    citations.append(citation)
                    all_sources.append(source)
                    citation_id += 1

        return {
            "status": "success",
            "agent_type": "deep_research",
            "execution_mode": "parallel",
            "persona": initial_state.get("persona"),
            "research_topic": topic,
            "research_depth": depth,
            "findings": synthesis,
            "sources_analyzed": len(all_sources),
            "confidence_score": synthesis.get("confidence_score", 0.0),
            "citations": citations,
            "execution_time_ms": execution_time,
            "parallel_execution_stats": {
                "total_tasks": len(research_result.task_results),
                "successful_tasks": research_result.successful_tasks,
                "failed_tasks": research_result.failed_tasks,
                "parallel_efficiency": research_result.parallel_efficiency,
                "task_breakdown": synthesis.get("task_breakdown", {}),
            },
            "search_queries_used": [],  # Will be populated by subagents
            "source_diversity": len({source.get("url", "") for source in all_sources})
            / max(len(all_sources), 1),
        }

    # Helper methods for parallel execution

    def _calculate_aggregated_sentiment(
        self, sentiment_scores: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate overall sentiment from multiple sentiment scores."""
        if not sentiment_scores:
            return {"direction": "neutral", "confidence": 0.5}

        # Convert sentiment directions to numeric values
        numeric_scores = []
        confidences = []

        for sentiment in sentiment_scores:
            direction = sentiment.get("direction", "neutral")
            confidence = sentiment.get("confidence", 0.5)

            if direction == "bullish":
                numeric_scores.append(1 * confidence)
            elif direction == "bearish":
                numeric_scores.append(-1 * confidence)
            else:
                numeric_scores.append(0)

            confidences.append(confidence)

        # Calculate weighted average
        avg_score = sum(numeric_scores) / len(numeric_scores)
        avg_confidence = sum(confidences) / len(confidences)

        # Convert back to direction
        if avg_score > 0.2:
            direction = "bullish"
        elif avg_score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "confidence": avg_confidence,
            "consensus": 1 - abs(avg_score) if abs(avg_score) < 1 else 0,
            "source_count": len(sentiment_scores),
        }

    def _build_parallel_synthesis_prompt(
        self,
        task_results: dict[str, ResearchTask],
        all_insights: list[str],
        all_risks: list[str],
        all_opportunities: list[str],
        overall_sentiment: dict[str, Any],
    ) -> str:
        """Build synthesis prompt for parallel research results."""
        successful_tasks = [
            task for task in task_results.values() if task.status == "completed"
        ]

        prompt = f"""
        Synthesize comprehensive research findings from {len(successful_tasks)} specialized research agents.

        Research Task Results:
        """

        for task in successful_tasks:
            if task.result:
                prompt += f"\n{task.task_type.upper()} RESEARCH:"
                prompt += f"  - Status: {task.status}"
                prompt += f"  - Key Insights: {', '.join(task.result.get('insights', [])[:3])}"
                prompt += f"  - Sentiment: {task.result.get('sentiment', {}).get('direction', 'neutral')}"

        prompt += f"""

        AGGREGATED DATA:
        - Total Insights: {len(all_insights)}
        - Risk Factors: {len(all_risks)}
        - Opportunities: {len(all_opportunities)}
        - Overall Sentiment: {overall_sentiment.get("direction")} (confidence: {overall_sentiment.get("confidence", 0.5):.2f})

        Please provide a comprehensive synthesis that includes:
        1. Executive Summary (2-3 sentences)
        2. Key Findings from all research areas
        3. Investment Implications for {self.persona.name} investors
        4. Risk Assessment and Mitigation
        5. Recommended Actions based on parallel analysis
        6. Confidence Level and reasoning

        Focus on insights that are supported by multiple research agents and highlight any contradictions.
        """

        return prompt

    def _generate_fallback_synthesis(
        self, insights: list[str], sentiment: dict[str, Any]
    ) -> str:
        """Generate fallback synthesis when LLM synthesis fails."""
        return f"""
        Research synthesis generated from {len(insights)} insights.

        Overall sentiment: {sentiment.get("direction", "neutral")} with {sentiment.get("confidence", 0.5):.2f} confidence.

        Key insights identified:
        {chr(10).join(f"- {insight}" for insight in insights[:5])}

        This is a fallback synthesis due to LLM processing limitations.
        """

    def _derive_parallel_recommendation(self, sentiment: dict[str, Any]) -> str:
        """Derive investment recommendation from parallel analysis."""
        direction = sentiment.get("direction", "neutral")
        confidence = sentiment.get("confidence", 0.5)

        if direction == "bullish" and confidence > 0.7:
            return "Strong buy signal based on parallel analysis from multiple research angles"
        elif direction == "bullish" and confidence > 0.5:
            return "Consider position building with appropriate risk management"
        elif direction == "bearish" and confidence > 0.7:
            return "Exercise significant caution - multiple research areas show negative signals"
        elif direction == "bearish" and confidence > 0.5:
            return "Monitor closely - mixed to negative signals suggest waiting"
        else:
            return "Neutral stance recommended - parallel analysis shows mixed signals"


# Specialized Subagent Classes


class BaseSubagent:
    """Base class for specialized research subagents."""

    def __init__(self, parent_agent: "DeepResearchAgent"):
        self.parent = parent_agent
        self.llm = parent_agent.llm
        self.search_providers = parent_agent.search_providers
        self.content_analyzer = parent_agent.content_analyzer
        self.persona = parent_agent.persona
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute_research(self, task: ResearchTask) -> dict[str, Any]:
        """Execute research task - to be implemented by subclasses."""
        raise NotImplementedError

    async def _perform_specialized_search(
        self, topic: str, specialized_queries: list[str], max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Perform specialized web search for this subagent type."""
        all_results = []

        for query in specialized_queries:
            for provider in self.search_providers:
                try:
                    results = await provider.search(
                        query, num_results=max_results // len(specialized_queries)
                    )
                    all_results.extend(results)
                except Exception as e:
                    self.logger.warning(f"Search failed for {query}: {e}")

        # Deduplicate results
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.get("url") not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)

        return unique_results[:max_results]

    async def _analyze_search_results(
        self, results: list[dict[str, Any]], analysis_focus: str
    ) -> list[dict[str, Any]]:
        """Analyze search results with specialized focus."""
        analyzed_results = []

        for result in results:
            if result.get("content"):
                try:
                    analysis = await self.content_analyzer.analyze_content(
                        content=result["content"],
                        persona=self.persona.name.lower(),
                        analysis_focus=analysis_focus,
                    )

                    # Add source credibility
                    credibility_score = self._calculate_source_credibility(result)
                    analysis["credibility_score"] = credibility_score

                    analyzed_results.append(
                        {
                            **result,
                            "analysis": analysis,
                            "credibility_score": credibility_score,
                        }
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Content analysis failed for {result.get('url', 'unknown')}: {e}"
                    )

        return analyzed_results

    def _calculate_source_credibility(self, source: dict[str, Any]) -> float:
        """Calculate credibility score for a source - reuse from parent."""
        return self.parent._calculate_source_credibility(source)


class FundamentalResearchAgent(BaseSubagent):
    """Specialized agent for fundamental financial analysis."""

    async def execute_research(self, task: ResearchTask) -> dict[str, Any]:
        """Execute fundamental analysis research."""
        self.logger.info(f"Executing fundamental research for: {task.target_topic}")

        # Generate fundamental-specific search queries
        queries = self._generate_fundamental_queries(task.target_topic)

        # Perform specialized search
        search_results = await self._perform_specialized_search(
            topic=task.target_topic, specialized_queries=queries, max_results=8
        )

        # Analyze results with fundamental focus
        analyzed_results = await self._analyze_search_results(
            search_results, analysis_focus="fundamental_analysis"
        )

        # Extract fundamental-specific insights
        insights = []
        risks = []
        opportunities = []
        sources = []

        for result in analyzed_results:
            analysis = result.get("analysis", {})
            insights.extend(analysis.get("insights", []))
            risks.extend(analysis.get("risk_factors", []))
            opportunities.extend(analysis.get("opportunities", []))
            sources.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "credibility_score": result.get("credibility_score", 0.5),
                    "published_date": result.get("published_date"),
                    "author": result.get("author"),
                }
            )

        return {
            "research_type": "fundamental",
            "insights": list(dict.fromkeys(insights))[:8],  # Deduplicate
            "risk_factors": list(dict.fromkeys(risks))[:6],
            "opportunities": list(dict.fromkeys(opportunities))[:6],
            "sentiment": self._calculate_fundamental_sentiment(analyzed_results),
            "credibility_score": self._calculate_average_credibility(analyzed_results),
            "sources": sources,
            "focus_areas": [
                "earnings",
                "valuation",
                "financial_health",
                "growth_prospects",
            ],
        }

    def _generate_fundamental_queries(self, topic: str) -> list[str]:
        """Generate fundamental analysis specific queries."""
        return [
            f"{topic} earnings report financial results",
            f"{topic} revenue growth profit margins",
            f"{topic} balance sheet debt ratio financial health",
            f"{topic} valuation PE ratio price earnings",
            f"{topic} cash flow dividend payout",
        ]

    def _calculate_fundamental_sentiment(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate sentiment specific to fundamental analysis."""
        sentiments = []
        for result in results:
            analysis = result.get("analysis", {})
            sentiment = analysis.get("sentiment", {})
            if sentiment:
                sentiments.append(sentiment)

        if not sentiments:
            return {"direction": "neutral", "confidence": 0.5}

        # Simple aggregation for now
        bullish_count = sum(1 for s in sentiments if s.get("direction") == "bullish")
        bearish_count = sum(1 for s in sentiments if s.get("direction") == "bearish")

        if bullish_count > bearish_count:
            return {"direction": "bullish", "confidence": 0.7}
        elif bearish_count > bullish_count:
            return {"direction": "bearish", "confidence": 0.7}
        else:
            return {"direction": "neutral", "confidence": 0.5}

    def _calculate_average_credibility(self, results: list[dict[str, Any]]) -> float:
        """Calculate average credibility of sources."""
        if not results:
            return 0.5

        credibility_scores = [r.get("credibility_score", 0.5) for r in results]
        return sum(credibility_scores) / len(credibility_scores)


class TechnicalResearchAgent(BaseSubagent):
    """Specialized agent for technical analysis research."""

    async def execute_research(self, task: ResearchTask) -> dict[str, Any]:
        """Execute technical analysis research."""
        self.logger.info(f"Executing technical research for: {task.target_topic}")

        queries = self._generate_technical_queries(task.target_topic)
        search_results = await self._perform_specialized_search(
            topic=task.target_topic, specialized_queries=queries, max_results=6
        )

        analyzed_results = await self._analyze_search_results(
            search_results, analysis_focus="technical_analysis"
        )

        # Extract technical-specific insights
        insights = []
        risks = []
        opportunities = []
        sources = []

        for result in analyzed_results:
            analysis = result.get("analysis", {})
            insights.extend(analysis.get("insights", []))
            risks.extend(analysis.get("risk_factors", []))
            opportunities.extend(analysis.get("opportunities", []))
            sources.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "credibility_score": result.get("credibility_score", 0.5),
                    "published_date": result.get("published_date"),
                    "author": result.get("author"),
                }
            )

        return {
            "research_type": "technical",
            "insights": list(dict.fromkeys(insights))[:8],
            "risk_factors": list(dict.fromkeys(risks))[:6],
            "opportunities": list(dict.fromkeys(opportunities))[:6],
            "sentiment": self._calculate_technical_sentiment(analyzed_results),
            "credibility_score": self._calculate_average_credibility(analyzed_results),
            "sources": sources,
            "focus_areas": [
                "price_action",
                "chart_patterns",
                "technical_indicators",
                "support_resistance",
            ],
        }

    def _generate_technical_queries(self, topic: str) -> list[str]:
        """Generate technical analysis specific queries."""
        return [
            f"{topic} technical analysis chart pattern",
            f"{topic} price target support resistance",
            f"{topic} RSI MACD technical indicators",
            f"{topic} breakout trend analysis",
            f"{topic} volume analysis price movement",
        ]

    def _calculate_technical_sentiment(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate sentiment specific to technical analysis."""
        # Similar to fundamental but focused on technical indicators
        sentiments = [
            r.get("analysis", {}).get("sentiment", {})
            for r in results
            if r.get("analysis")
        ]
        sentiments = [s for s in sentiments if s]

        if not sentiments:
            return {"direction": "neutral", "confidence": 0.5}

        bullish_count = sum(1 for s in sentiments if s.get("direction") == "bullish")
        bearish_count = sum(1 for s in sentiments if s.get("direction") == "bearish")

        if bullish_count > bearish_count:
            return {"direction": "bullish", "confidence": 0.6}
        elif bearish_count > bullish_count:
            return {"direction": "bearish", "confidence": 0.6}
        else:
            return {"direction": "neutral", "confidence": 0.5}

    def _calculate_average_credibility(self, results: list[dict[str, Any]]) -> float:
        """Calculate average credibility of sources."""
        if not results:
            return 0.5
        credibility_scores = [r.get("credibility_score", 0.5) for r in results]
        return sum(credibility_scores) / len(credibility_scores)


class SentimentResearchAgent(BaseSubagent):
    """Specialized agent for market sentiment analysis."""

    async def execute_research(self, task: ResearchTask) -> dict[str, Any]:
        """Execute sentiment analysis research."""
        self.logger.info(f"Executing sentiment research for: {task.target_topic}")

        queries = self._generate_sentiment_queries(task.target_topic)
        search_results = await self._perform_specialized_search(
            topic=task.target_topic, specialized_queries=queries, max_results=10
        )

        analyzed_results = await self._analyze_search_results(
            search_results, analysis_focus="sentiment_analysis"
        )

        # Extract sentiment-specific insights
        insights = []
        risks = []
        opportunities = []
        sources = []

        for result in analyzed_results:
            analysis = result.get("analysis", {})
            insights.extend(analysis.get("insights", []))
            risks.extend(analysis.get("risk_factors", []))
            opportunities.extend(analysis.get("opportunities", []))
            sources.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "credibility_score": result.get("credibility_score", 0.5),
                    "published_date": result.get("published_date"),
                    "author": result.get("author"),
                }
            )

        return {
            "research_type": "sentiment",
            "insights": list(dict.fromkeys(insights))[:8],
            "risk_factors": list(dict.fromkeys(risks))[:6],
            "opportunities": list(dict.fromkeys(opportunities))[:6],
            "sentiment": self._calculate_market_sentiment(analyzed_results),
            "credibility_score": self._calculate_average_credibility(analyzed_results),
            "sources": sources,
            "focus_areas": [
                "market_sentiment",
                "analyst_opinions",
                "news_sentiment",
                "social_sentiment",
            ],
        }

    def _generate_sentiment_queries(self, topic: str) -> list[str]:
        """Generate sentiment analysis specific queries."""
        return [
            f"{topic} analyst rating recommendation upgrade downgrade",
            f"{topic} market sentiment investor opinion",
            f"{topic} news sentiment positive negative",
            f"{topic} social sentiment reddit twitter discussion",
            f"{topic} institutional investor sentiment",
        ]

    def _calculate_market_sentiment(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate overall market sentiment."""
        sentiments = [
            r.get("analysis", {}).get("sentiment", {})
            for r in results
            if r.get("analysis")
        ]
        sentiments = [s for s in sentiments if s]

        if not sentiments:
            return {"direction": "neutral", "confidence": 0.5}

        # Weighted by confidence
        weighted_scores = []
        total_confidence = 0

        for sentiment in sentiments:
            direction = sentiment.get("direction", "neutral")
            confidence = sentiment.get("confidence", 0.5)

            if direction == "bullish":
                weighted_scores.append(1 * confidence)
            elif direction == "bearish":
                weighted_scores.append(-1 * confidence)
            else:
                weighted_scores.append(0)

            total_confidence += confidence

        if not weighted_scores:
            return {"direction": "neutral", "confidence": 0.5}

        avg_score = sum(weighted_scores) / len(weighted_scores)
        avg_confidence = total_confidence / len(sentiments)

        if avg_score > 0.3:
            return {"direction": "bullish", "confidence": avg_confidence}
        elif avg_score < -0.3:
            return {"direction": "bearish", "confidence": avg_confidence}
        else:
            return {"direction": "neutral", "confidence": avg_confidence}

    def _calculate_average_credibility(self, results: list[dict[str, Any]]) -> float:
        """Calculate average credibility of sources."""
        if not results:
            return 0.5
        credibility_scores = [r.get("credibility_score", 0.5) for r in results]
        return sum(credibility_scores) / len(credibility_scores)


class CompetitiveResearchAgent(BaseSubagent):
    """Specialized agent for competitive and industry analysis."""

    async def execute_research(self, task: ResearchTask) -> dict[str, Any]:
        """Execute competitive analysis research."""
        self.logger.info(f"Executing competitive research for: {task.target_topic}")

        queries = self._generate_competitive_queries(task.target_topic)
        search_results = await self._perform_specialized_search(
            topic=task.target_topic, specialized_queries=queries, max_results=8
        )

        analyzed_results = await self._analyze_search_results(
            search_results, analysis_focus="competitive_analysis"
        )

        # Extract competitive-specific insights
        insights = []
        risks = []
        opportunities = []
        sources = []

        for result in analyzed_results:
            analysis = result.get("analysis", {})
            insights.extend(analysis.get("insights", []))
            risks.extend(analysis.get("risk_factors", []))
            opportunities.extend(analysis.get("opportunities", []))
            sources.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "credibility_score": result.get("credibility_score", 0.5),
                    "published_date": result.get("published_date"),
                    "author": result.get("author"),
                }
            )

        return {
            "research_type": "competitive",
            "insights": list(dict.fromkeys(insights))[:8],
            "risk_factors": list(dict.fromkeys(risks))[:6],
            "opportunities": list(dict.fromkeys(opportunities))[:6],
            "sentiment": self._calculate_competitive_sentiment(analyzed_results),
            "credibility_score": self._calculate_average_credibility(analyzed_results),
            "sources": sources,
            "focus_areas": [
                "competitive_position",
                "market_share",
                "industry_trends",
                "competitive_advantages",
            ],
        }

    def _generate_competitive_queries(self, topic: str) -> list[str]:
        """Generate competitive analysis specific queries."""
        return [
            f"{topic} market share competitive position industry",
            f"{topic} competitors comparison competitive advantage",
            f"{topic} industry analysis market trends",
            f"{topic} competitive landscape market dynamics",
            f"{topic} industry outlook sector performance",
        ]

    def _calculate_competitive_sentiment(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate sentiment specific to competitive positioning."""
        sentiments = [
            r.get("analysis", {}).get("sentiment", {})
            for r in results
            if r.get("analysis")
        ]
        sentiments = [s for s in sentiments if s]

        if not sentiments:
            return {"direction": "neutral", "confidence": 0.5}

        # Focus on competitive strength indicators
        bullish_count = sum(1 for s in sentiments if s.get("direction") == "bullish")
        bearish_count = sum(1 for s in sentiments if s.get("direction") == "bearish")

        if bullish_count > bearish_count:
            return {"direction": "bullish", "confidence": 0.6}
        elif bearish_count > bullish_count:
            return {"direction": "bearish", "confidence": 0.6}
        else:
            return {"direction": "neutral", "confidence": 0.5}

    def _calculate_average_credibility(self, results: list[dict[str, Any]]) -> float:
        """Calculate average credibility of sources."""
        if not results:
            return 0.5
        credibility_scores = [r.get("credibility_score", 0.5) for r in results]
        return sum(credibility_scores) / len(credibility_scores)
