"""Parallel LLM batch processing with adaptive load balancing."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from maverick_mcp.providers.openrouter_provider import (
    OpenRouterProvider,
    TaskType,
)
from maverick_mcp.utils.llm.model_selection import (
    AdaptiveModelSelector,
    ModelConfiguration,
)
from maverick_mcp.utils.orchestration_logging import (
    get_orchestration_logger,
    log_method_call,
)


class ParallelLLMProcessor:
    """Handles parallel LLM operations with intelligent load balancing."""

    def __init__(
        self,
        openrouter_provider: OpenRouterProvider,
        max_concurrent: int = 5,  # OPTIMIZATION: Increased from 3
    ):
        self.provider = openrouter_provider
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.BoundedSemaphore(
            max_concurrent
        )  # OPTIMIZATION: Use BoundedSemaphore
        self.model_selector = AdaptiveModelSelector(openrouter_provider)
        self.orchestration_logger = get_orchestration_logger("ParallelLLMProcessor")
        # OPTIMIZATION: Track active requests for better coordination
        self._active_requests = 0
        self._request_lock = asyncio.Lock()

    @log_method_call(component="ParallelLLMProcessor", include_timing=True)
    async def parallel_content_analysis(
        self,
        sources: list[dict],
        analysis_type: str,
        persona: str,
        time_budget_seconds: float,
        current_confidence: float = 0.0,
    ) -> list[dict]:
        """Analyze multiple sources in parallel with adaptive optimization."""

        if not sources:
            return []

        self.orchestration_logger.set_request_context(
            analysis_type=analysis_type,
            source_count=len(sources),
            time_budget=time_budget_seconds,
        )

        # Calculate complexity for all sources
        combined_content = "\n".join(
            [source.get("content", "")[:1000] for source in sources[:5]]
        )
        overall_complexity = self.model_selector.calculate_task_complexity(
            combined_content,
            TaskType.SENTIMENT_ANALYSIS
            if analysis_type == "sentiment"
            else TaskType.MARKET_ANALYSIS,
        )

        # Determine optimal batching strategy
        model_config = self.model_selector.select_model_for_time_budget(
            task_type=TaskType.SENTIMENT_ANALYSIS
            if analysis_type == "sentiment"
            else TaskType.MARKET_ANALYSIS,
            time_remaining_seconds=time_budget_seconds,
            complexity_score=overall_complexity,
            content_size_tokens=len(combined_content) // 4,
            current_confidence=current_confidence,
        )

        # Create batches based on model configuration
        batches = self._create_optimal_batches(
            sources, model_config.parallel_batch_size
        )

        self.orchestration_logger.info(
            "🔄 PARALLEL_ANALYSIS_START",
            total_sources=len(sources),
            batch_count=len(batches),
        )

        # OPTIMIZATION: Process batches using create_task for immediate parallelism
        running_tasks = []
        for i, batch in enumerate(batches):
            # Create task immediately without awaiting
            task_future = asyncio.create_task(
                self._analyze_source_batch(
                    batch=batch,
                    batch_id=i,
                    analysis_type=analysis_type,
                    persona=persona,
                    model_config=model_config,
                    overall_complexity=overall_complexity,
                )
            )
            running_tasks.append((i, task_future))  # Track batch ID with future

            # OPTIMIZATION: Minimal stagger to prevent API overload
            if i < len(batches) - 1:  # Don't delay after last batch
                await asyncio.sleep(0.01)  # 10ms micro-delay

        # OPTIMIZATION: Use as_completed for progressive result handling
        batch_results = [None] * len(batches)  # Pre-allocate results list
        timeout_at = time.time() + (time_budget_seconds * 0.9)

        try:
            for batch_id, task_future in running_tasks:
                remaining_time = timeout_at - time.time()
                if remaining_time <= 0:
                    raise TimeoutError()

                try:
                    result = await asyncio.wait_for(task_future, timeout=remaining_time)
                    batch_results[batch_id] = result
                except Exception as e:
                    batch_results[batch_id] = e
        except TimeoutError:
            self.orchestration_logger.warning(
                "⏰ PARALLEL_ANALYSIS_TIMEOUT", timeout=time_budget_seconds
            )
            return self._create_fallback_results(sources)

        # Flatten and process results
        final_results = []
        successful_batches = 0
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                self.orchestration_logger.warning(
                    "⚠️ BATCH_FAILED", batch_id=i, error=str(batch_result)
                )
                # Add fallback results for failed batch
                final_results.extend(self._create_fallback_results(batches[i]))
            else:
                final_results.extend(batch_result)
                successful_batches += 1

        self.orchestration_logger.info(
            "✅ PARALLEL_ANALYSIS_COMPLETE",
            successful_batches=successful_batches,
            results_count=len(final_results),
        )

        return final_results

    def _create_optimal_batches(
        self, sources: list[dict], batch_size: int
    ) -> list[list[dict]]:
        """Create optimal batches for parallel processing."""
        if batch_size <= 1:
            return [[source] for source in sources]

        batches = []
        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            batches.append(batch)

        return batches

    async def _analyze_source_batch(
        self,
        batch: list[dict],
        batch_id: int,
        analysis_type: str,
        persona: str,
        model_config: ModelConfiguration,
        overall_complexity: float,
    ) -> list[dict]:
        """Analyze a batch of sources with optimized LLM call."""

        # OPTIMIZATION: Track active requests for better coordination
        async with self._request_lock:
            self._active_requests += 1

        try:
            # OPTIMIZATION: Acquire semaphore without blocking other task creation
            await self.semaphore.acquire()
            try:
                # Create batch analysis prompt
                batch_prompt = self._create_batch_analysis_prompt(
                    batch, analysis_type, persona, model_config.max_tokens
                )

                # Get LLM instance
                llm = self.provider.get_llm(
                    model_override=model_config.model_id,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                )

                # Execute with timeout
                start_time = time.time()
                result = await asyncio.wait_for(
                    llm.ainvoke(
                        [
                            SystemMessage(
                                content="You are a financial analyst. Provide structured, concise analysis."
                            ),
                            HumanMessage(content=batch_prompt),
                        ]
                    ),
                    timeout=model_config.timeout_seconds,
                )

                execution_time = time.time() - start_time

                # Parse batch results
                parsed_results = self._parse_batch_analysis_result(
                    result.content, batch
                )

                self.orchestration_logger.debug(
                    "✨ BATCH_SUCCESS",
                    batch_id=batch_id,
                    duration=f"{execution_time:.2f}s",
                )

                return parsed_results

            except TimeoutError:
                self.orchestration_logger.warning(
                    "⏰ BATCH_TIMEOUT",
                    batch_id=batch_id,
                    timeout=model_config.timeout_seconds,
                )
                return self._create_fallback_results(batch)
            except Exception as e:
                self.orchestration_logger.error(
                    "💥 BATCH_ERROR", batch_id=batch_id, error=str(e)
                )
                return self._create_fallback_results(batch)
            finally:
                # OPTIMIZATION: Always release semaphore
                self.semaphore.release()
        finally:
            # OPTIMIZATION: Track active requests
            async with self._request_lock:
                self._active_requests -= 1

    def _create_batch_analysis_prompt(
        self, batch: list[dict], analysis_type: str, persona: str, max_tokens: int
    ) -> str:
        """Create optimized prompt for batch analysis."""

        # Determine prompt style based on token budget
        if max_tokens < 800:
            style = "ultra_concise"
        elif max_tokens < 1500:
            style = "concise"
        else:
            style = "detailed"

        prompt_templates = {
            "ultra_concise": """URGENT BATCH ANALYSIS - {analysis_type} for {persona} investor.

Analyze {source_count} sources. For EACH source, provide:
SOURCE_N: SENTIMENT:Bull/Bear/Neutral|CONFIDENCE:0-1|INSIGHT:one key point|RISK:main risk

{sources}

Keep total response under 500 words.""",
            "concise": """BATCH ANALYSIS - {analysis_type} for {persona} investor perspective.

Analyze these {source_count} sources. For each source provide:
- Sentiment: Bull/Bear/Neutral + confidence (0-1)
- Key insight (1 sentence)
- Main risk (1 sentence)
- Relevance score (0-1)

{sources}

Format consistently. Target ~100 words per source.""",
            "detailed": """Comprehensive {analysis_type} analysis for {persona} investor.

Analyze these {source_count} sources with structured output for each:

{sources}

For each source provide:
1. Sentiment (direction, confidence 0-1, brief reasoning)
2. Key insights (2-3 main points)
3. Risk factors (1-2 key risks)
4. Opportunities (1-2 opportunities if any)
5. Credibility assessment (0-1 score)
6. Relevance score (0-1)

Maintain {persona} investor perspective throughout.""",
        }

        # Format sources for prompt
        sources_text = ""
        for i, source in enumerate(batch, 1):
            content = source.get("content", "")[:1500]  # Limit content length
            title = source.get("title", f"Source {i}")
            sources_text += f"\nSOURCE {i} - {title}:\n{content}\n{'---' * 20}\n"

        template = prompt_templates[style]
        return template.format(
            analysis_type=analysis_type,
            persona=persona,
            source_count=len(batch),
            sources=sources_text.strip(),
        )

    def _parse_batch_analysis_result(
        self, result_content: str, batch: list[dict]
    ) -> list[dict]:
        """Parse LLM batch analysis result into structured data."""

        results = []

        # Try structured parsing first
        source_sections = re.split(r"\n(?:SOURCE\s+\d+|---+)", result_content)

        if len(source_sections) >= len(batch):
            # Structured parsing successful
            for _i, (source, section) in enumerate(
                zip(batch, source_sections[1 : len(batch) + 1], strict=False)
            ):
                parsed = self._parse_source_analysis(section, source)
                results.append(parsed)
        else:
            # Fallback to simple parsing
            for i, source in enumerate(batch):
                fallback_analysis = self._create_simple_fallback_analysis(
                    result_content, source, i
                )
                results.append(fallback_analysis)

        return results

    def _parse_source_analysis(self, analysis_text: str, source: dict) -> dict:
        """Parse analysis text for a single source."""

        # Extract sentiment
        sentiment_match = re.search(
            r"sentiment:?\s*(\w+)[,\s]*(?:confidence:?\s*([\d.]+))?",
            analysis_text.lower(),
        )
        if sentiment_match:
            direction = sentiment_match.group(1).lower()
            confidence = float(sentiment_match.group(2) or 0.5)

            # Map common sentiment terms
            if direction in ["bull", "bullish", "positive"]:
                direction = "bullish"
            elif direction in ["bear", "bearish", "negative"]:
                direction = "bearish"
            else:
                direction = "neutral"
        else:
            direction = "neutral"
            confidence = 0.5

        # Extract other information
        insights = self._extract_insights(analysis_text)
        risks = self._extract_risks(analysis_text)
        opportunities = self._extract_opportunities(analysis_text)

        # Extract scores
        relevance_match = re.search(r"relevance:?\s*([\d.]+)", analysis_text.lower())
        relevance_score = float(relevance_match.group(1)) if relevance_match else 0.6

        credibility_match = re.search(
            r"credibility:?\s*([\d.]+)", analysis_text.lower()
        )
        credibility_score = (
            float(credibility_match.group(1)) if credibility_match else 0.7
        )

        return {
            **source,
            "analysis": {
                "insights": insights,
                "sentiment": {"direction": direction, "confidence": confidence},
                "risk_factors": risks,
                "opportunities": opportunities,
                "credibility_score": credibility_score,
                "relevance_score": relevance_score,
                "analysis_timestamp": datetime.now(),
                "batch_processed": True,
            },
        }

    def _extract_insights(self, text: str) -> list[str]:
        """Extract insights from analysis text."""
        insights = []

        # Look for insight patterns
        insight_patterns = [
            r"insight:?\s*([^.\n]+)",
            r"key point:?\s*([^.\n]+)",
            r"main finding:?\s*([^.\n]+)",
        ]

        for pattern in insight_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            insights.extend([m.strip() for m in matches if m.strip()])

        # If no structured insights found, extract bullet points
        if not insights:
            bullet_matches = re.findall(r"[•\-\*]\s*([^.\n]+)", text)
            insights.extend([m.strip() for m in bullet_matches if m.strip()][:3])

        return insights[:5]  # Limit to 5 insights

    def _extract_risks(self, text: str) -> list[str]:
        """Extract risk factors from analysis text."""
        risk_patterns = [
            r"risk:?\s*([^.\n]+)",
            r"concern:?\s*([^.\n]+)",
            r"headwind:?\s*([^.\n]+)",
        ]

        risks = []
        for pattern in risk_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            risks.extend([m.strip() for m in matches if m.strip()])

        return risks[:3]

    def _extract_opportunities(self, text: str) -> list[str]:
        """Extract opportunities from analysis text."""
        opp_patterns = [
            r"opportunit(?:y|ies):?\s*([^.\n]+)",
            r"catalyst:?\s*([^.\n]+)",
            r"tailwind:?\s*([^.\n]+)",
        ]

        opportunities = []
        for pattern in opp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            opportunities.extend([m.strip() for m in matches if m.strip()])

        return opportunities[:3]

    def _create_simple_fallback_analysis(
        self, full_analysis: str, source: dict, index: int
    ) -> dict:
        """Create simple fallback analysis when parsing fails."""

        # Basic sentiment analysis from text
        analysis_lower = full_analysis.lower()

        positive_words = ["positive", "bullish", "strong", "growth", "opportunity"]
        negative_words = ["negative", "bearish", "weak", "decline", "risk"]

        pos_count = sum(1 for word in positive_words if word in analysis_lower)
        neg_count = sum(1 for word in negative_words if word in analysis_lower)

        if pos_count > neg_count:
            sentiment = "bullish"
            confidence = 0.6
        elif neg_count > pos_count:
            sentiment = "bearish"
            confidence = 0.6
        else:
            sentiment = "neutral"
            confidence = 0.5

        return {
            **source,
            "analysis": {
                "insights": [f"Analysis based on source content (index {index})"],
                "sentiment": {"direction": sentiment, "confidence": confidence},
                "risk_factors": ["Unable to extract specific risks"],
                "opportunities": ["Unable to extract specific opportunities"],
                "credibility_score": 0.5,
                "relevance_score": 0.5,
                "analysis_timestamp": datetime.now(),
                "fallback_used": True,
                "batch_processed": True,
            },
        }

    def _create_fallback_results(self, sources: list[dict]) -> list[dict]:
        """Create fallback results when batch processing fails."""
        results = []
        for source in sources:
            fallback_result = {
                **source,
                "analysis": {
                    "insights": ["Analysis failed - using fallback"],
                    "sentiment": {"direction": "neutral", "confidence": 0.3},
                    "risk_factors": ["Analysis timeout - unable to assess risks"],
                    "opportunities": [],
                    "credibility_score": 0.5,
                    "relevance_score": 0.5,
                    "analysis_timestamp": datetime.now(),
                    "fallback_used": True,
                    "batch_timeout": True,
                },
            }
            results.append(fallback_result)
        return results
