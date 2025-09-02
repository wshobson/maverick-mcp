"""
Optimized Deep Research Agent with LLM-side optimizations to prevent timeouts.

This module integrates the comprehensive LLM optimization strategies including:
- Adaptive model selection based on time constraints
- Progressive token budgeting with confidence tracking
- Parallel LLM processing with intelligent load balancing
- Optimized prompt engineering for speed
- Early termination based on confidence thresholds
- Content filtering to reduce processing overhead
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from maverick_mcp.agents.deep_research import (
    PERSONA_RESEARCH_FOCUS,
    RESEARCH_DEPTH_LEVELS,
    ContentAnalyzer,
    DeepResearchAgent,
)
from maverick_mcp.providers.openrouter_provider import OpenRouterProvider, TaskType
from maverick_mcp.utils.llm_optimization import (
    AdaptiveModelSelector,
    ConfidenceTracker,
    IntelligentContentFilter,
    OptimizedPromptEngine,
    ParallelLLMProcessor,
    ProgressiveTokenBudgeter,
)
from maverick_mcp.utils.orchestration_logging import (
    get_orchestration_logger,
    log_method_call,
    log_performance_metrics,
)
from maverick_mcp.utils.parallel_research import (
    ParallelResearchConfig,
)

logger = logging.getLogger(__name__)


class OptimizedContentAnalyzer(ContentAnalyzer):
    """Enhanced ContentAnalyzer with LLM optimizations."""

    def __init__(self, openrouter_provider: OpenRouterProvider):
        # Initialize with OpenRouter provider instead of single LLM
        self.openrouter_provider = openrouter_provider
        self.model_selector = AdaptiveModelSelector(openrouter_provider)
        self.prompt_engine = OptimizedPromptEngine()
        self.parallel_processor = ParallelLLMProcessor(openrouter_provider)

    async def analyze_content_optimized(
        self,
        content: str,
        persona: str,
        analysis_focus: str = "general",
        time_budget_seconds: float = 30.0,
        current_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Analyze content with time-optimized LLM selection and prompting."""

        if not content or not content.strip():
            return self._create_empty_analysis()

        # Calculate content complexity
        complexity_score = self.model_selector.calculate_task_complexity(
            content, TaskType.SENTIMENT_ANALYSIS, [analysis_focus]
        )

        # Select optimal model for time budget
        model_config = self.model_selector.select_model_for_time_budget(
            task_type=TaskType.SENTIMENT_ANALYSIS,
            time_remaining_seconds=time_budget_seconds,
            complexity_score=complexity_score,
            content_size_tokens=len(content) // 4,  # Rough token estimate
            current_confidence=current_confidence,
        )

        # Create optimized prompt
        optimized_prompt = self.prompt_engine.get_optimized_prompt(
            prompt_type="content_analysis",
            time_remaining=time_budget_seconds,
            confidence_level=current_confidence,
            content=content[: model_config.max_tokens * 3],  # Limit content size
            persona=persona,
            focus_areas=analysis_focus,
        )

        # Execute with optimized LLM
        try:
            llm = self.openrouter_provider.get_llm(
                model_override=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
            )

            start_time = time.time()
            response = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are a financial content analyst. Return structured JSON analysis."
                        ),
                        HumanMessage(content=optimized_prompt),
                    ]
                ),
                timeout=model_config.timeout_seconds,
            )

            execution_time = time.time() - start_time

            # Parse response
            analysis = self._parse_optimized_response(response.content, persona)
            analysis["execution_time"] = execution_time
            analysis["model_used"] = model_config.model_id
            analysis["optimization_applied"] = True

            return analysis

        except TimeoutError:
            logger.warning(
                f"Content analysis timed out after {model_config.timeout_seconds}s"
            )
            return self._fallback_analysis(content, persona)
        except Exception as e:
            logger.warning(f"Optimized content analysis failed: {e}")
            return self._fallback_analysis(content, persona)

    async def batch_analyze_content(
        self,
        sources: list[dict],
        persona: str,
        analysis_type: str,
        time_budget_seconds: float,
        current_confidence: float = 0.0,
    ) -> list[dict]:
        """Analyze multiple sources using parallel processing."""

        return await self.parallel_processor.parallel_content_analysis(
            sources=sources,
            analysis_type=analysis_type,
            persona=persona,
            time_budget_seconds=time_budget_seconds,
            current_confidence=current_confidence,
        )

    def _parse_optimized_response(
        self, response_content: str, persona: str
    ) -> dict[str, Any]:
        """Parse LLM response with fallback handling."""

        try:
            # Try to parse as JSON first
            import json

            if response_content.strip().startswith("{"):
                return json.loads(response_content.strip())
        except Exception:
            pass

        # Try structured text parsing
        try:
            return self._parse_structured_response(response_content, persona)
        except Exception:
            # Final fallback
            return self._fallback_analysis(response_content, persona)

    def _parse_structured_response(self, response: str, persona: str) -> dict[str, Any]:
        """Parse structured text response."""

        import re

        # Extract sentiment
        sentiment_match = re.search(
            r"sentiment:?\s*(\w+)[,\s]*(?:confidence:?\s*([\d.]+))?", response.lower()
        )
        if sentiment_match:
            direction = sentiment_match.group(1).lower()
            confidence = float(sentiment_match.group(2) or 0.6)

            # Normalize sentiment terms
            if direction in ["bull", "bullish", "positive", "buy"]:
                direction = "bullish"
            elif direction in ["bear", "bearish", "negative", "sell"]:
                direction = "bearish"
            else:
                direction = "neutral"
        else:
            direction = "neutral"
            confidence = 0.5

        # Extract insights
        insights = []
        insight_patterns = [
            r"insight:?\s*([^\n.]+)",
            r"key point:?\s*([^\n.]+)",
            r"finding:?\s*([^\n.]+)",
        ]
        for pattern in insight_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            insights.extend([m.strip() for m in matches if m.strip()])

        # Extract risks and opportunities
        risks = re.findall(r"risk:?\s*([^\n.]+)", response, re.IGNORECASE)
        opportunities = re.findall(
            r"opportunit(?:y|ies):?\s*([^\n.]+)", response, re.IGNORECASE
        )

        # Extract scores
        relevance_match = re.search(r"relevance:?\s*([\d.]+)", response.lower())
        relevance_score = float(relevance_match.group(1)) if relevance_match else 0.6

        credibility_match = re.search(r"credibility:?\s*([\d.]+)", response.lower())
        credibility_score = (
            float(credibility_match.group(1)) if credibility_match else 0.7
        )

        return {
            "insights": insights[:5],
            "sentiment": {"direction": direction, "confidence": confidence},
            "risk_factors": [r.strip() for r in risks[:3]],
            "opportunities": [o.strip() for o in opportunities[:3]],
            "credibility_score": credibility_score,
            "relevance_score": relevance_score,
            "summary": f"Analysis for {persona} investor using optimized processing",
            "analysis_timestamp": datetime.now(),
            "structured_parsing": True,
        }

    def _create_empty_analysis(self) -> dict[str, Any]:
        """Create empty analysis for invalid content."""
        return {
            "insights": [],
            "sentiment": {"direction": "neutral", "confidence": 0.0},
            "risk_factors": [],
            "opportunities": [],
            "credibility_score": 0.0,
            "relevance_score": 0.0,
            "summary": "No content to analyze",
            "analysis_timestamp": datetime.now(),
            "empty_content": True,
        }


class OptimizedDeepResearchAgent(DeepResearchAgent):
    """
    Deep research agent with comprehensive LLM-side optimizations to prevent timeouts.

    Integrates all optimization strategies:
    - Adaptive model selection
    - Progressive token budgeting
    - Parallel LLM processing
    - Optimized prompting
    - Early termination
    - Content filtering
    """

    def __init__(
        self,
        openrouter_provider: OpenRouterProvider,
        persona: str = "moderate",
        checkpointer: MemorySaver | None = None,
        ttl_hours: int = 24,
        exa_api_key: str | None = None,
        tavily_api_key: str | None = None,
        default_depth: str = "standard",
        max_sources: int | None = None,
        research_depth: str | None = None,
        enable_parallel_execution: bool = True,
        parallel_config: ParallelResearchConfig | None = None,
        optimization_enabled: bool = True,
    ):
        """Initialize optimized deep research agent."""

        self.openrouter_provider = openrouter_provider
        self.optimization_enabled = optimization_enabled

        # Initialize optimization components
        if optimization_enabled:
            self.model_selector = AdaptiveModelSelector(openrouter_provider)
            self.token_budgeter = None  # Will be created per request
            self.prompt_engine = OptimizedPromptEngine()
            self.confidence_tracker = None  # Will be created per request
            self.content_filter = IntelligentContentFilter()
            self.parallel_processor = ParallelLLMProcessor(openrouter_provider)

            # Replace content analyzer with optimized version
            self.optimized_analyzer = OptimizedContentAnalyzer(openrouter_provider)

        # Initialize base class with dummy LLM (we'll use OpenRouter provider instead)
        dummy_llm = openrouter_provider.get_llm(TaskType.GENERAL)

        super().__init__(
            llm=dummy_llm,
            persona=persona,
            checkpointer=checkpointer,
            ttl_hours=ttl_hours,
            exa_api_key=exa_api_key,
            tavily_api_key=tavily_api_key,
            default_depth=default_depth,
            max_sources=max_sources,
            research_depth=research_depth,
            enable_parallel_execution=enable_parallel_execution,
            parallel_config=parallel_config,
        )

        logger.info(
            f"OptimizedDeepResearchAgent initialized with optimization_enabled={optimization_enabled}"
        )

    @log_method_call(component="OptimizedDeepResearchAgent", include_timing=True)
    async def research_comprehensive(
        self,
        topic: str,
        session_id: str,
        depth: str | None = None,
        focus_areas: list[str] | None = None,
        timeframe: str = "30d",
        time_budget_seconds: float = 120.0,  # Default 2 minutes
        target_confidence: float = 0.75,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Comprehensive research with LLM optimizations to prevent timeouts.

        Args:
            topic: Research topic or company/symbol
            session_id: Session identifier
            depth: Research depth (basic/standard/comprehensive/exhaustive)
            focus_areas: Specific areas to focus on
            timeframe: Time range for research
            time_budget_seconds: Maximum time allowed for research
            target_confidence: Target confidence level for early termination
            **kwargs: Additional parameters

        Returns:
            Comprehensive research results with optimization metrics
        """

        if not self.optimization_enabled:
            # Fall back to parent implementation
            return await super().research_comprehensive(
                topic, session_id, depth, focus_areas, timeframe, **kwargs
            )

        # Check if search providers are available
        if not self.search_providers:
            return {
                "error": "Research functionality unavailable - no search providers configured",
                "details": "Please configure EXA_API_KEY or TAVILY_API_KEY environment variables",
                "topic": topic,
                "optimization_enabled": self.optimization_enabled,
            }

        start_time = time.time()
        depth = depth or self.default_depth

        # Initialize optimization components for this request
        self.token_budgeter = ProgressiveTokenBudgeter(
            total_time_budget_seconds=time_budget_seconds,
            confidence_target=target_confidence,
        )
        self.confidence_tracker = ConfidenceTracker(
            target_confidence=target_confidence,
            min_sources=3,
            max_sources=RESEARCH_DEPTH_LEVELS[depth]["max_sources"],
        )

        orchestration_logger = get_orchestration_logger("OptimizedDeepResearchAgent")
        orchestration_logger.set_request_context(
            session_id=session_id,
            topic=topic[:50],
            time_budget=time_budget_seconds,
            target_confidence=target_confidence,
        )

        orchestration_logger.info(
            "🚀 OPTIMIZED_RESEARCH_START",
            depth=depth,
            focus_areas=focus_areas,
            timeframe=timeframe,
            optimization_features=[
                "adaptive_model_selection",
                "progressive_token_budgeting",
                "parallel_processing",
                "optimized_prompts",
                "early_termination",
                "content_filtering",
            ],
        )

        try:
            # Phase 1: Search and Content Filtering
            orchestration_logger.info("📋 PHASE_1_SEARCH_START")
            search_time_budget = min(
                time_budget_seconds * 0.2, 30
            )  # 20% of budget, max 30s

            search_results = await self._optimized_search_phase(
                topic, depth, focus_areas, search_time_budget
            )

            orchestration_logger.info(
                "✅ PHASE_1_COMPLETE",
                raw_sources=len(search_results.get("raw_results", [])),
                filtered_sources=len(search_results.get("filtered_sources", [])),
            )

            # Phase 2: Content Analysis with Parallel Processing
            remaining_time = time_budget_seconds - (time.time() - start_time)
            if remaining_time < 10:
                orchestration_logger.warning(
                    "⚠️ TIME_CONSTRAINT_CRITICAL", remaining=remaining_time
                )
                return self._create_emergency_response(
                    topic, search_results, start_time
                )

            orchestration_logger.info(
                "🔬 PHASE_2_ANALYSIS_START", remaining_time=remaining_time
            )
            analysis_time_budget = remaining_time * 0.7  # 70% of remaining time

            analysis_results = await self._optimized_analysis_phase(
                search_results["filtered_sources"],
                topic,
                focus_areas,
                analysis_time_budget,
            )

            orchestration_logger.info(
                "✅ PHASE_2_COMPLETE",
                sources_analyzed=len(analysis_results["analyzed_sources"]),
                final_confidence=analysis_results["final_confidence"],
                early_terminated=analysis_results.get("early_terminated", False),
            )

            # Phase 3: Synthesis with Remaining Time
            remaining_time = time_budget_seconds - (time.time() - start_time)
            if remaining_time < 5:
                # Skip synthesis if very little time left
                synthesis_results = {
                    "synthesis": "Time constraints prevented full synthesis"
                }
            else:
                orchestration_logger.info(
                    "🧠 PHASE_3_SYNTHESIS_START", remaining_time=remaining_time
                )
                synthesis_results = await self._optimized_synthesis_phase(
                    analysis_results["analyzed_sources"], topic, remaining_time
                )
                orchestration_logger.info("✅ PHASE_3_COMPLETE")

            # Compile final results
            execution_time = time.time() - start_time
            final_results = self._compile_optimized_results(
                topic=topic,
                session_id=session_id,
                depth=depth,
                search_results=search_results,
                analysis_results=analysis_results,
                synthesis_results=synthesis_results,
                execution_time=execution_time,
                time_budget=time_budget_seconds,
            )

            # Log performance metrics
            log_performance_metrics(
                "OptimizedDeepResearchAgent",
                {
                    "total_execution_time": execution_time,
                    "time_budget_used_pct": (execution_time / time_budget_seconds)
                    * 100,
                    "sources_processed": len(analysis_results["analyzed_sources"]),
                    "final_confidence": analysis_results["final_confidence"],
                    "optimization_enabled": True,
                    "phases_completed": 3,
                },
            )

            orchestration_logger.info(
                "🎉 OPTIMIZED_RESEARCH_COMPLETE",
                execution_time=f"{execution_time:.2f}s",
                budget_utilized=f"{(execution_time / time_budget_seconds) * 100:.1f}%",
                final_confidence=analysis_results["final_confidence"],
            )

            return final_results

        except Exception as e:
            execution_time = time.time() - start_time
            orchestration_logger.error(
                "💥 OPTIMIZED_RESEARCH_FAILED",
                error=str(e),
                execution_time=f"{execution_time:.2f}s",
            )

            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": execution_time * 1000,
                "agent_type": "optimized_deep_research",
                "optimization_enabled": True,
                "topic": topic,
            }

    async def _optimized_search_phase(
        self, topic: str, depth: str, focus_areas: list[str], time_budget_seconds: float
    ) -> dict[str, Any]:
        """Execute search phase with content filtering."""

        # Generate search queries (reuse parent logic)
        persona_focus = PERSONA_RESEARCH_FOCUS[self.persona.name.lower()]
        search_queries = await self._generate_search_queries(
            topic, persona_focus, RESEARCH_DEPTH_LEVELS[depth]
        )

        # Execute searches (reuse parent logic but with time limits)
        all_results = []
        max_searches = min(len(search_queries), 4)  # Limit searches for speed

        search_tasks = []
        for query in search_queries[:max_searches]:
            for provider in self.search_providers[
                :1
            ]:  # Use only first provider for speed
                task = self._search_with_timeout(
                    provider, query, time_budget_seconds / max_searches
                )
                search_tasks.append(task)

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect valid results
        for result in search_results:
            if isinstance(result, list):
                all_results.extend(result)

        # Apply intelligent content filtering
        current_confidence = 0.0  # Starting confidence
        research_focus = focus_areas[0] if focus_areas else "fundamental"

        filtered_sources = await self.content_filter.filter_and_prioritize_sources(
            sources=all_results,
            research_focus=research_focus,
            time_budget=time_budget_seconds,
            current_confidence=current_confidence,
        )

        return {
            "raw_results": all_results,
            "filtered_sources": filtered_sources,
            "search_queries": search_queries[:max_searches],
            "filtering_applied": True,
        }

    async def _search_with_timeout(
        self, provider, query: str, timeout: float
    ) -> list[dict]:
        """Execute search with timeout."""
        try:
            return await asyncio.wait_for(
                provider.search(query, num_results=5), timeout=timeout
            )
        except TimeoutError:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.warning(f"Search failed for {query}: {e}")
            return []

    async def _optimized_analysis_phase(
        self,
        sources: list[dict],
        topic: str,
        focus_areas: list[str],
        time_budget_seconds: float,
    ) -> dict[str, Any]:
        """Execute content analysis with optimizations and early termination."""

        if not sources:
            return {
                "analyzed_sources": [],
                "final_confidence": 0.0,
                "early_terminated": False,
                "termination_reason": "no_sources",
            }

        analyzed_sources = []
        current_confidence = 0.0
        sources_to_process = sources.copy()

        # Calculate time per source
        time_per_source = time_budget_seconds / len(sources_to_process)

        # Use batch processing if time allows
        if len(sources_to_process) > 3 and time_per_source < 8:
            # Use parallel batch processing
            logger.info(
                f"Using parallel batch processing for {len(sources_to_process)} sources"
            )

            analyzed_sources = await self.optimized_analyzer.batch_analyze_content(
                sources=sources_to_process,
                persona=self.persona.name.lower(),
                analysis_type=focus_areas[0] if focus_areas else "general",
                time_budget_seconds=time_budget_seconds,
                current_confidence=current_confidence,
            )

            # Calculate final confidence from batch results
            confidence_sum = 0
            for source in analyzed_sources:
                analysis = source.get("analysis", {})
                sentiment = analysis.get("sentiment", {})
                source_confidence = sentiment.get("confidence", 0.5)
                credibility = analysis.get("credibility_score", 0.5)
                confidence_sum += source_confidence * credibility

            final_confidence = (
                confidence_sum / len(analyzed_sources) if analyzed_sources else 0.0
            )

            return {
                "analyzed_sources": analyzed_sources,
                "final_confidence": final_confidence,
                "early_terminated": False,
                "termination_reason": "batch_processing_complete",
                "processing_mode": "parallel_batch",
            }

        else:
            # Use sequential processing with early termination
            logger.info(
                f"Using sequential processing with early termination for {len(sources_to_process)} sources"
            )

            for _, source in enumerate(sources_to_process):
                remaining_time = time_budget_seconds - (
                    len(analyzed_sources) * time_per_source
                )

                if remaining_time < 5:  # Reserve minimum time
                    break

                # Analyze source with optimizations
                analysis_result = (
                    await self.optimized_analyzer.analyze_content_optimized(
                        content=source.get("content", ""),
                        persona=self.persona.name.lower(),
                        analysis_focus=focus_areas[0] if focus_areas else "general",
                        time_budget_seconds=min(
                            remaining_time / 2, 15
                        ),  # Max 15s per source
                        current_confidence=current_confidence,
                    )
                )

                # Add analysis to source
                source["analysis"] = analysis_result
                analyzed_sources.append(source)

                # Update confidence tracker
                credibility_score = analysis_result.get("credibility_score", 0.5)
                confidence_update = self.confidence_tracker.update_confidence(
                    analysis_result, credibility_score
                )

                current_confidence = confidence_update["current_confidence"]

                # Check for early termination
                if not confidence_update["should_continue"]:
                    logger.info(
                        f"Early termination triggered after {len(analyzed_sources)} sources. "
                        f"Reason: {confidence_update['early_termination_reason']}"
                    )
                    return {
                        "analyzed_sources": analyzed_sources,
                        "final_confidence": current_confidence,
                        "early_terminated": True,
                        "termination_reason": confidence_update[
                            "early_termination_reason"
                        ],
                        "processing_mode": "sequential_early_termination",
                    }

            return {
                "analyzed_sources": analyzed_sources,
                "final_confidence": current_confidence,
                "early_terminated": False,
                "termination_reason": "all_sources_processed",
                "processing_mode": "sequential_complete",
            }

    async def _optimized_synthesis_phase(
        self, analyzed_sources: list[dict], topic: str, time_budget_seconds: float
    ) -> dict[str, Any]:
        """Execute synthesis with optimized model selection."""

        if not analyzed_sources:
            return {"synthesis": "No sources available for synthesis"}

        # Select optimal model for synthesis
        combined_content = "\n".join(
            [str(source.get("analysis", {})) for source in analyzed_sources[:5]]
        )

        complexity_score = self.model_selector.calculate_task_complexity(
            combined_content, TaskType.RESULT_SYNTHESIS
        )

        model_config = self.model_selector.select_model_for_time_budget(
            task_type=TaskType.RESULT_SYNTHESIS,
            time_remaining_seconds=time_budget_seconds,
            complexity_score=complexity_score,
            content_size_tokens=len(combined_content) // 4,
        )

        # Create optimized synthesis prompt
        synthesis_prompt = self.prompt_engine.create_time_optimized_synthesis_prompt(
            sources=analyzed_sources,
            persona=self.persona.name,
            time_remaining=time_budget_seconds,
            current_confidence=0.8,  # Assume good confidence at synthesis stage
        )

        # Execute synthesis
        try:
            llm = self.openrouter_provider.get_llm(
                model_override=model_config.model_id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
            )

            response = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are a financial research synthesizer."
                        ),
                        HumanMessage(content=synthesis_prompt),
                    ]
                ),
                timeout=model_config.timeout_seconds,
            )

            return {
                "synthesis": response.content,
                "model_used": model_config.model_id,
                "synthesis_optimized": True,
            }

        except Exception as e:
            logger.warning(f"Optimized synthesis failed: {e}")
            return {
                "synthesis": f"Synthesis of {len(analyzed_sources)} sources completed with basic processing due to constraints.",
                "fallback_used": True,
            }

    def _create_emergency_response(
        self, topic: str, search_results: dict, start_time: float
    ) -> dict[str, Any]:
        """Create emergency response when time is critically low."""

        execution_time = time.time() - start_time
        source_count = len(search_results.get("filtered_sources", []))

        return {
            "status": "partial_success",
            "agent_type": "optimized_deep_research",
            "emergency_mode": True,
            "topic": topic,
            "sources_found": source_count,
            "execution_time_ms": execution_time * 1000,
            "findings": {
                "synthesis": f"Emergency mode: Found {source_count} relevant sources for {topic}. "
                "Full analysis was prevented by time constraints.",
                "confidence_score": 0.3,
                "sources_analyzed": source_count,
            },
            "optimization_metrics": {
                "time_budget_exceeded": True,
                "phases_completed": 1,
                "emergency_fallback": True,
            },
        }

    def _compile_optimized_results(
        self,
        topic: str,
        session_id: str,
        depth: str,
        search_results: dict,
        analysis_results: dict,
        synthesis_results: dict,
        execution_time: float,
        time_budget: float,
    ) -> dict[str, Any]:
        """Compile final optimized research results."""

        analyzed_sources = analysis_results["analyzed_sources"]

        # Create citations
        citations = []
        for i, source in enumerate(analyzed_sources, 1):
            analysis = source.get("analysis", {})
            citation = {
                "id": i,
                "title": source.get("title", f"Source {i}"),
                "url": source.get("url", ""),
                "published_date": source.get("published_date"),
                "credibility_score": analysis.get("credibility_score", 0.5),
                "relevance_score": analysis.get("relevance_score", 0.5),
                "optimized_analysis": analysis.get("optimization_applied", False),
            }
            citations.append(citation)

        return {
            "status": "success",
            "agent_type": "optimized_deep_research",
            "optimization_enabled": True,
            "persona": self.persona.name,
            "research_topic": topic,
            "research_depth": depth,
            "findings": {
                "synthesis": synthesis_results.get(
                    "synthesis", "No synthesis available"
                ),
                "confidence_score": analysis_results["final_confidence"],
                "early_terminated": analysis_results.get("early_terminated", False),
                "termination_reason": analysis_results.get("termination_reason"),
                "processing_mode": analysis_results.get("processing_mode", "unknown"),
            },
            "sources_analyzed": len(analyzed_sources),
            "citations": citations,
            "execution_time_ms": execution_time * 1000,
            "optimization_metrics": {
                "time_budget_seconds": time_budget,
                "time_used_seconds": execution_time,
                "time_utilization_pct": (execution_time / time_budget) * 100,
                "sources_found": len(search_results.get("raw_results", [])),
                "sources_filtered": len(search_results.get("filtered_sources", [])),
                "sources_processed": len(analyzed_sources),
                "content_filtering_applied": search_results.get(
                    "filtering_applied", False
                ),
                "parallel_processing_used": "batch"
                in analysis_results.get("processing_mode", ""),
                "synthesis_optimized": synthesis_results.get(
                    "synthesis_optimized", False
                ),
                "optimization_features_used": [
                    "adaptive_model_selection",
                    "progressive_token_budgeting",
                    "content_filtering",
                    "optimized_prompts",
                ]
                + (
                    ["parallel_processing"]
                    if "batch" in analysis_results.get("processing_mode", "")
                    else []
                )
                + (
                    ["early_termination"]
                    if analysis_results.get("early_terminated")
                    else []
                ),
            },
            "search_queries_used": search_results.get("search_queries", []),
            "session_id": session_id,
        }


# Factory function for easy integration
def create_optimized_research_agent(
    openrouter_api_key: str,
    persona: str = "moderate",
    time_budget_seconds: float = 120.0,
    target_confidence: float = 0.75,
    **kwargs,
) -> OptimizedDeepResearchAgent:
    """Create an optimized deep research agent with recommended settings."""

    openrouter_provider = OpenRouterProvider(openrouter_api_key)

    return OptimizedDeepResearchAgent(
        openrouter_provider=openrouter_provider,
        persona=persona,
        optimization_enabled=True,
        **kwargs,
    )
