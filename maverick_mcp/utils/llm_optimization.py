"""
LLM-side optimizations for research agents to prevent timeouts.

This module provides comprehensive optimization strategies including:
- Adaptive model selection based on time constraints
- Progressive token budgeting with confidence tracking
- Parallel LLM processing with intelligent load balancing
- Optimized prompt engineering for speed
- Early termination based on confidence thresholds
- Content filtering to reduce processing overhead
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from maverick_mcp.providers.openrouter_provider import (
    OpenRouterProvider,
    TaskType,
)
from maverick_mcp.utils.orchestration_logging import (
    get_orchestration_logger,
    log_method_call,
)

logger = logging.getLogger(__name__)


class ResearchPhase(str, Enum):
    """Research phases for token allocation."""

    SEARCH = "search"
    CONTENT_ANALYSIS = "content_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


class ModelConfiguration(BaseModel):
    """Configuration for model selection with time optimization."""

    model_id: str = Field(description="OpenRouter model identifier")
    max_tokens: int = Field(description="Maximum output tokens")
    temperature: float = Field(description="Model temperature")
    timeout_seconds: float = Field(description="Request timeout")
    parallel_batch_size: int = Field(
        default=1, description="Sources per batch for this model"
    )


class TokenAllocation(BaseModel):
    """Token allocation for a research phase."""

    input_tokens: int = Field(description="Maximum input tokens")
    output_tokens: int = Field(description="Maximum output tokens")
    per_source_tokens: int = Field(description="Tokens per source")
    emergency_reserve: int = Field(description="Emergency reserve tokens")
    timeout_seconds: float = Field(description="Processing timeout")


class AdaptiveModelSelector:
    """Intelligent model selection based on time budgets and task complexity."""

    def __init__(self, openrouter_provider: OpenRouterProvider):
        self.provider = openrouter_provider
        self.performance_cache = {}  # Cache model performance metrics

    def select_model_for_time_budget(
        self,
        task_type: TaskType,
        time_remaining_seconds: float,
        complexity_score: float,
        content_size_tokens: int,
        confidence_threshold: float = 0.8,
        current_confidence: float = 0.0,
    ) -> ModelConfiguration:
        """Select optimal model based on available time and requirements."""

        # Time pressure categories with adaptive thresholds
        if time_remaining_seconds < 10:
            return self._select_emergency_model(task_type, content_size_tokens)
        elif time_remaining_seconds < 25:
            return self._select_fast_quality_model(task_type, complexity_score)
        elif time_remaining_seconds < 45:
            return self._select_balanced_model(
                task_type, complexity_score, current_confidence
            )
        else:
            return self._select_optimal_model(
                task_type, complexity_score, confidence_threshold
            )

    def _select_emergency_model(
        self, task_type: TaskType, content_size: int
    ) -> ModelConfiguration:
        """Ultra-fast models for time-critical situations."""
        # Prioritize speed over everything else - use fastest available models
        if content_size > 20000:  # Large content needs fast + capable models
            return ModelConfiguration(
                model_id="google/gemini-2.5-flash",  # 199 tokens/sec - fastest available
                max_tokens=min(800, content_size // 25),  # Adaptive token limit
                temperature=0.1,  # Lower temp for faster, more focused response
                timeout_seconds=8,
                parallel_batch_size=4,  # Increased for faster model
            )
        else:
            return ModelConfiguration(
                model_id="openai/gpt-4o-mini",  # 126 tokens/sec - excellent speed/cost balance
                max_tokens=min(500, content_size // 20),
                temperature=0.05,  # Minimal temperature for fastest response
                timeout_seconds=6,
                parallel_batch_size=5,  # Increased for faster processing
            )

    def _select_fast_quality_model(
        self, task_type: TaskType, complexity_score: float
    ) -> ModelConfiguration:
        """Balance speed and quality for time-constrained situations."""
        if complexity_score > 0.7 or task_type == TaskType.COMPLEX_REASONING:
            # Complex tasks - use fast model with good quality
            return ModelConfiguration(
                model_id="openai/gpt-4o-mini",  # 126 tokens/sec + good quality
                max_tokens=1200,
                temperature=0.2,
                timeout_seconds=18,  # Reduced timeout for speed
                parallel_batch_size=3,  # Increased for faster model
            )
        else:
            # Simple tasks - use the fastest model available
            return ModelConfiguration(
                model_id="google/gemini-2.5-flash",  # 199 tokens/sec - fastest
                max_tokens=1000,
                temperature=0.2,
                timeout_seconds=12,  # Reduced timeout for fastest model
                parallel_batch_size=4,  # Increased for faster processing
            )

    def _select_balanced_model(
        self, task_type: TaskType, complexity_score: float, current_confidence: float
    ) -> ModelConfiguration:
        """Standard mode with cost-effectiveness focus."""
        # If confidence is already high, use fastest models for validation
        if current_confidence > 0.7:
            return ModelConfiguration(
                model_id="google/gemini-2.5-flash",  # 199 tokens/sec - fastest validation
                max_tokens=1500,
                temperature=0.25,
                timeout_seconds=20,  # Reduced for fastest model
                parallel_batch_size=4,  # Increased for speed
            )

        # Standard balanced approach - prioritize speed-optimized models
        if task_type in [TaskType.DEEP_RESEARCH, TaskType.RESULT_SYNTHESIS]:
            return ModelConfiguration(
                model_id="openai/gpt-4o-mini",  # Speed + quality balance for research
                max_tokens=2000,
                temperature=0.3,
                timeout_seconds=25,  # Reduced for faster model
                parallel_batch_size=3,  # Increased for speed
            )
        else:
            return ModelConfiguration(
                model_id="google/gemini-2.5-flash",  # Fastest for general tasks
                max_tokens=1500,
                temperature=0.25,
                timeout_seconds=20,  # Reduced for fastest model
                parallel_batch_size=4,  # Increased for speed
            )

    def _select_optimal_model(
        self, task_type: TaskType, complexity_score: float, confidence_threshold: float
    ) -> ModelConfiguration:
        """Comprehensive mode for complex analysis."""
        # Use premium models for the most complex tasks when time allows
        if complexity_score > 0.8 and task_type == TaskType.DEEP_RESEARCH:
            return ModelConfiguration(
                model_id="google/gemini-2.5-pro",
                max_tokens=3000,
                temperature=0.3,
                timeout_seconds=45,
                parallel_batch_size=1,  # Deep thinking models work better individually
            )

        # High-quality cost-effective models for standard comprehensive analysis
        return ModelConfiguration(
            model_id="anthropic/claude-sonnet-4",
            max_tokens=2500,
            temperature=0.3,
            timeout_seconds=40,
            parallel_batch_size=2,
        )

    def calculate_task_complexity(
        self, content: str, task_type: TaskType, focus_areas: list[str] = None
    ) -> float:
        """Calculate complexity score based on content and task requirements."""
        if not content:
            return 0.3  # Default low complexity

        content_lower = content.lower()

        # Financial complexity indicators
        complexity_indicators = {
            "financial_jargon": len(
                re.findall(
                    r"\b(?:ebitda|dcf|roic?|wacc|beta|volatility|sharpe)\b",
                    content_lower,
                )
            ),
            "numerical_data": len(re.findall(r"\$?[\d,]+\.?\d*[%kmbKMB]?", content)),
            "comparative_analysis": len(
                re.findall(
                    r"\b(?:versus|compared|relative|outperform|underperform)\b",
                    content_lower,
                )
            ),
            "temporal_analysis": len(
                re.findall(r"\b(?:quarterly|q[1-4]|fy|yoy|qoq|annual)\b", content_lower)
            ),
            "market_terms": len(
                re.findall(
                    r"\b(?:bullish|bearish|catalyst|headwind|tailwind)\b", content_lower
                )
            ),
            "technical_terms": len(
                re.findall(
                    r"\b(?:support|resistance|breakout|rsi|macd|sma|ema)\b",
                    content_lower,
                )
            ),
        }

        # Calculate base complexity
        total_indicators = sum(complexity_indicators.values())
        content_length = len(content.split())
        base_complexity = min(total_indicators / max(content_length / 100, 1), 1.0)

        # Task-specific complexity adjustments
        task_multipliers = {
            TaskType.DEEP_RESEARCH: 1.4,
            TaskType.COMPLEX_REASONING: 1.6,
            TaskType.RESULT_SYNTHESIS: 1.2,
            TaskType.TECHNICAL_ANALYSIS: 1.3,
            TaskType.SENTIMENT_ANALYSIS: 0.8,
            TaskType.QUICK_ANSWER: 0.5,
        }

        # Focus area adjustments
        focus_multiplier = 1.0
        if focus_areas:
            complex_focus_areas = [
                "competitive_analysis",
                "fundamental_analysis",
                "complex_reasoning",
            ]
            if any(area in focus_areas for area in complex_focus_areas):
                focus_multiplier = 1.2

        final_complexity = (
            base_complexity * task_multipliers.get(task_type, 1.0) * focus_multiplier
        )
        return min(final_complexity, 1.0)


class ProgressiveTokenBudgeter:
    """Manages token budgets across research phases with time awareness."""

    def __init__(
        self, total_time_budget_seconds: float, confidence_target: float = 0.75
    ):
        self.total_time_budget = total_time_budget_seconds
        self.confidence_target = confidence_target
        self.phase_budgets = self._calculate_base_phase_budgets()
        self.time_started = time.time()

    def _calculate_base_phase_budgets(self) -> dict[ResearchPhase, int]:
        """Calculate base token budgets for each research phase."""
        # Allocate tokens based on typical phase requirements
        if self.total_time_budget < 30:
            # Emergency mode - minimal tokens
            return {
                ResearchPhase.SEARCH: 500,
                ResearchPhase.CONTENT_ANALYSIS: 2000,
                ResearchPhase.SYNTHESIS: 800,
                ResearchPhase.VALIDATION: 300,
            }
        elif self.total_time_budget < 60:
            # Fast mode
            return {
                ResearchPhase.SEARCH: 1000,
                ResearchPhase.CONTENT_ANALYSIS: 4000,
                ResearchPhase.SYNTHESIS: 1500,
                ResearchPhase.VALIDATION: 500,
            }
        else:
            # Standard mode
            return {
                ResearchPhase.SEARCH: 1500,
                ResearchPhase.CONTENT_ANALYSIS: 6000,
                ResearchPhase.SYNTHESIS: 2500,
                ResearchPhase.VALIDATION: 1000,
            }

    def allocate_tokens_for_phase(
        self,
        phase: ResearchPhase,
        sources_count: int,
        current_confidence: float,
        complexity_score: float = 0.5,
    ) -> TokenAllocation:
        """Allocate tokens for a research phase based on current state."""

        time_elapsed = time.time() - self.time_started
        time_remaining = max(0, self.total_time_budget - time_elapsed)

        base_budget = self.phase_budgets[phase]

        # Confidence-based scaling
        if current_confidence > self.confidence_target:
            # High confidence - focus on validation with fewer tokens
            confidence_multiplier = 0.7
        elif current_confidence < 0.4:
            # Low confidence - increase token usage if time allows
            confidence_multiplier = 1.3 if time_remaining > 30 else 0.9
        else:
            confidence_multiplier = 1.0

        # Time pressure scaling
        time_multiplier = self._calculate_time_multiplier(time_remaining)

        # Complexity scaling
        complexity_multiplier = 0.8 + (complexity_score * 0.4)  # Range: 0.8 to 1.2

        # Source count scaling (diminishing returns)
        if sources_count > 0:
            source_multiplier = min(1.0 + (sources_count - 3) * 0.05, 1.3)
        else:
            source_multiplier = 1.0

        # Calculate final budget
        final_budget = int(
            base_budget
            * confidence_multiplier
            * time_multiplier
            * complexity_multiplier
            * source_multiplier
        )

        # Calculate timeout based on available time and token budget
        base_timeout = min(time_remaining * 0.8, 45)  # Max 45 seconds per phase
        adjusted_timeout = base_timeout * (final_budget / base_budget) ** 0.5

        return TokenAllocation(
            input_tokens=min(int(final_budget * 0.75), 15000),  # Cap input tokens
            output_tokens=min(int(final_budget * 0.25), 3000),  # Cap output tokens
            per_source_tokens=final_budget // max(sources_count, 1)
            if sources_count > 0
            else final_budget,
            emergency_reserve=200,  # Always keep emergency reserve
            timeout_seconds=max(adjusted_timeout, 5),  # Minimum 5 seconds
        )

    def _calculate_time_multiplier(self, time_remaining: float) -> float:
        """Scale token budget based on time pressure."""
        if time_remaining < 5:
            return 0.2  # Extreme emergency mode
        elif time_remaining < 15:
            return 0.4  # Emergency mode
        elif time_remaining < 30:
            return 0.7  # Time-constrained
        elif time_remaining < 60:
            return 0.9  # Slightly reduced
        else:
            return 1.0  # Full budget available


class ParallelLLMProcessor:
    """Handles parallel LLM operations with intelligent load balancing."""

    def __init__(
        self, openrouter_provider: OpenRouterProvider, max_concurrent: int = 3
    ):
        self.provider = openrouter_provider
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model_selector = AdaptiveModelSelector(openrouter_provider)
        self.orchestration_logger = get_orchestration_logger("ParallelLLMProcessor")

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
            model_id=model_config.model_id,
            parallel_batch_size=model_config.parallel_batch_size,
        )

        # Process batches in parallel
        tasks = []
        for i, batch in enumerate(batches):
            task = self._analyze_source_batch(
                batch=batch,
                batch_id=i,
                analysis_type=analysis_type,
                persona=persona,
                model_config=model_config,
                overall_complexity=overall_complexity,
            )
            tasks.append(task)

        # Execute with timeout
        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=time_budget_seconds * 0.9,
            )
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
            total_batches=len(batches),
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
        """Analyze a batch of sources with a single LLM call."""

        async with self.semaphore:
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
                    execution_time=f"{execution_time:.2f}s",
                    sources_processed=len(batch),
                    model=model_config.model_id,
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
            for i, (source, section) in enumerate(
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


class OptimizedPromptEngine:
    """Creates optimized prompts for different time constraints and confidence levels."""

    def __init__(self):
        self.prompt_cache = {}  # Cache for generated prompts

        self.prompt_templates = {
            "emergency": {
                "content_analysis": """URGENT: Quick 3-point analysis of financial content for {persona} investor.

Content: {content}

Provide ONLY:
1. SENTIMENT: Bull/Bear/Neutral + confidence (0-1)
2. KEY_RISK: Primary risk factor
3. KEY_OPPORTUNITY: Main opportunity (if any)

Format: SENTIMENT:Bull|0.8 KEY_RISK:Market volatility KEY_OPPORTUNITY:Earnings growth
Max 50 words total. No explanations.""",
                "synthesis": """URGENT: 2-sentence summary from {source_count} sources for {persona} investor.

Key findings: {key_points}

Provide: 1) Overall sentiment direction 2) Primary investment implication
Max 40 words total.""",
            },
            "fast": {
                "content_analysis": """Quick financial analysis for {persona} investor - 5 points max.

Content: {content}

Provide concisely:
• Sentiment: Bull/Bear/Neutral (confidence 0-1)
• Key insight (1 sentence)
• Main risk (1 sentence)
• Main opportunity (1 sentence)
• Relevance score (0-1)

Target: Under 150 words total.""",
                "synthesis": """Synthesize research findings for {persona} investor.

Sources: {source_count} | Key insights: {insights}

4-part summary:
1. Overall sentiment + confidence
2. Top 2 opportunities
3. Top 2 risks
4. Recommended action

Limit: 200 words max.""",
            },
            "standard": {
                "content_analysis": """Financial content analysis for {persona} investor.

Content: {content}
Focus areas: {focus_areas}

Structured analysis:
- Sentiment (direction, confidence 0-1, brief reasoning)
- Key insights (3-5 bullet points)
- Risk factors (2-3 main risks)
- Opportunities (2-3 opportunities)
- Credibility assessment (0-1)
- Relevance score (0-1)

Target: 300-500 words.""",
                "synthesis": """Comprehensive research synthesis for {persona} investor.

Research Summary:
- Sources analyzed: {source_count}
- Key insights: {insights}
- Time horizon: {time_horizon}

Provide detailed analysis:
1. Executive Summary (2-3 sentences)
2. Key Findings (5-7 bullet points)
3. Investment Implications
4. Risk Assessment
5. Recommended Actions
6. Confidence Level + reasoning

Tailor specifically for {persona} investment characteristics.""",
            },
        }

    def get_optimized_prompt(
        self,
        prompt_type: str,
        time_remaining: float,
        confidence_level: float,
        **context,
    ) -> str:
        """Generate optimized prompt based on time constraints and confidence."""

        # Create cache key
        cache_key = f"{prompt_type}_{time_remaining:.0f}_{confidence_level:.1f}_{hash(str(sorted(context.items())))}"

        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        # Select template based on time pressure
        if time_remaining < 15:
            template_category = "emergency"
        elif time_remaining < 45:
            template_category = "fast"
        else:
            template_category = "standard"

        template = self.prompt_templates[template_category].get(prompt_type)

        if not template:
            # Fallback to fast template
            template = self.prompt_templates["fast"].get(
                prompt_type, "Analyze the content quickly and provide key insights."
            )

        # Add confidence-based instructions
        confidence_instructions = ""
        if confidence_level > 0.7:
            confidence_instructions = "\n\nNOTE: High confidence already achieved. Focus on validation and contradictory evidence."
        elif confidence_level < 0.4:
            confidence_instructions = "\n\nNOTE: Low confidence. Look for strong supporting evidence to build confidence."

        # Format template with context
        formatted_prompt = template.format(**context) + confidence_instructions

        # Cache the result
        self.prompt_cache[cache_key] = formatted_prompt

        return formatted_prompt

    def create_time_optimized_synthesis_prompt(
        self,
        sources: list[dict],
        persona: str,
        time_remaining: float,
        current_confidence: float,
    ) -> str:
        """Create synthesis prompt optimized for available time."""

        # Extract key information from sources
        insights = []
        sentiments = []
        for source in sources:
            analysis = source.get("analysis", {})
            insights.extend(analysis.get("insights", [])[:2])  # Limit per source
            sentiment = analysis.get("sentiment", {})
            if sentiment:
                sentiments.append(sentiment.get("direction", "neutral"))

        # Prepare context
        context = {
            "persona": persona,
            "source_count": len(sources),
            "insights": "; ".join(insights[:8]),  # Top 8 insights
            "time_horizon": "short-term" if time_remaining < 30 else "medium-term",
        }

        return self.get_optimized_prompt(
            "synthesis", time_remaining, current_confidence, **context
        )


class ConfidenceTracker:
    """Tracks research confidence and triggers early termination when appropriate."""

    def __init__(
        self,
        target_confidence: float = 0.75,
        min_sources: int = 3,
        max_sources: int = 15,
    ):
        self.target_confidence = target_confidence
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.confidence_history = []
        self.evidence_history = []
        self.source_count = 0
        self.last_significant_improvement = 0
        self.sentiment_votes = {"bullish": 0, "bearish": 0, "neutral": 0}

    def update_confidence(
        self, new_evidence: dict, source_credibility: float
    ) -> dict[str, Any]:
        """Update confidence based on new evidence and return continuation decision."""

        self.source_count += 1

        # Store evidence
        self.evidence_history.append(
            {
                "evidence": new_evidence,
                "credibility": source_credibility,
                "timestamp": datetime.now(),
            }
        )

        # Update sentiment voting
        sentiment = new_evidence.get("sentiment", {})
        direction = sentiment.get("direction", "neutral")
        confidence = sentiment.get("confidence", 0.5)

        # Weight vote by source credibility and sentiment confidence
        vote_weight = source_credibility * confidence
        self.sentiment_votes[direction] += vote_weight

        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(
            new_evidence, source_credibility
        )

        # Update confidence using Bayesian-style updating
        current_confidence = self._update_bayesian_confidence(evidence_strength)
        self.confidence_history.append(current_confidence)

        # Check for significant improvement
        if len(self.confidence_history) >= 2:
            improvement = current_confidence - self.confidence_history[-2]
            if improvement > 0.1:  # 10% improvement
                self.last_significant_improvement = self.source_count

        # Make continuation decision
        should_continue = self._should_continue_research(current_confidence)

        return {
            "current_confidence": current_confidence,
            "should_continue": should_continue,
            "sources_processed": self.source_count,
            "confidence_trend": self._calculate_confidence_trend(),
            "early_termination_reason": None
            if should_continue
            else self._get_termination_reason(current_confidence),
            "sentiment_consensus": self._calculate_sentiment_consensus(),
        }

    def _calculate_evidence_strength(self, evidence: dict, credibility: float) -> float:
        """Calculate the strength of new evidence."""

        # Base strength from sentiment confidence
        sentiment = evidence.get("sentiment", {})
        sentiment_confidence = sentiment.get("confidence", 0.5)

        # Adjust for source credibility
        credibility_adjusted = sentiment_confidence * credibility

        # Factor in evidence richness
        insights_count = len(evidence.get("insights", []))
        risk_factors_count = len(evidence.get("risk_factors", []))
        opportunities_count = len(evidence.get("opportunities", []))

        # Evidence richness score (0-1)
        evidence_richness = min(
            (insights_count + risk_factors_count + opportunities_count) / 12, 1.0
        )

        # Relevance factor
        relevance_score = evidence.get("relevance_score", 0.5)

        # Final evidence strength calculation
        final_strength = credibility_adjusted * (
            0.5 + 0.3 * evidence_richness + 0.2 * relevance_score
        )

        return min(final_strength, 1.0)

    def _update_bayesian_confidence(self, evidence_strength: float) -> float:
        """Update confidence using Bayesian approach."""

        if not self.confidence_history:
            # First evidence - base confidence
            return evidence_strength

        # Current prior
        prior = self.confidence_history[-1]

        # Bayesian update with evidence strength as likelihood
        # Simple approximation: weighted average with decay
        decay_factor = 0.9 ** (self.source_count - 1)  # Diminishing returns

        updated = prior * decay_factor + evidence_strength * (1 - decay_factor)

        # Ensure within bounds
        return max(0.1, min(updated, 0.95))

    def _should_continue_research(self, current_confidence: float) -> bool:
        """Determine if research should continue based on multiple factors."""

        # Always process minimum sources
        if self.source_count < self.min_sources:
            return True

        # Stop at maximum sources
        if self.source_count >= self.max_sources:
            return False

        # High confidence reached
        if current_confidence >= self.target_confidence:
            return False

        # Check for diminishing returns
        if self.source_count - self.last_significant_improvement > 4:
            # No significant improvement in last 4 sources
            return False

        # Check sentiment consensus
        consensus_score = self._calculate_sentiment_consensus()
        if consensus_score > 0.8 and self.source_count >= 5:
            # Strong consensus with adequate sample
            return False

        # Check confidence plateau
        if len(self.confidence_history) >= 3:
            recent_change = abs(current_confidence - self.confidence_history[-3])
            if recent_change < 0.03:  # Less than 3% change in last 3 sources
                return False

        return True

    def _calculate_confidence_trend(self) -> str:
        """Calculate the trend in confidence over recent sources."""

        if len(self.confidence_history) < 3:
            return "insufficient_data"

        recent = self.confidence_history[-3:]

        # Calculate trend
        if recent[-1] > recent[0] + 0.05:
            return "increasing"
        elif recent[-1] < recent[0] - 0.05:
            return "decreasing"
        else:
            return "stable"

    def _calculate_sentiment_consensus(self) -> float:
        """Calculate how much sources agree on sentiment."""

        total_votes = sum(self.sentiment_votes.values())
        if total_votes == 0:
            return 0.0

        # Calculate consensus as max vote share
        max_votes = max(self.sentiment_votes.values())
        consensus = max_votes / total_votes

        return consensus

    def _get_termination_reason(self, current_confidence: float) -> str:
        """Get reason for early termination."""

        if current_confidence >= self.target_confidence:
            return "target_confidence_reached"
        elif self.source_count >= self.max_sources:
            return "max_sources_reached"
        elif self._calculate_sentiment_consensus() > 0.8:
            return "strong_consensus"
        elif self.source_count - self.last_significant_improvement > 4:
            return "diminishing_returns"
        else:
            return "confidence_plateau"


class IntelligentContentFilter:
    """Pre-filters and prioritizes content to reduce LLM processing overhead."""

    def __init__(self):
        self.relevance_keywords = {
            "fundamental": {
                "high": [
                    "earnings",
                    "revenue",
                    "profit",
                    "ebitda",
                    "cash flow",
                    "debt",
                    "valuation",
                ],
                "medium": [
                    "balance sheet",
                    "income statement",
                    "financial",
                    "quarterly",
                    "annual",
                ],
                "context": ["company", "business", "financial results", "guidance"],
            },
            "technical": {
                "high": [
                    "price",
                    "chart",
                    "trend",
                    "support",
                    "resistance",
                    "breakout",
                ],
                "medium": ["volume", "rsi", "macd", "moving average", "pattern"],
                "context": ["technical analysis", "trading", "momentum"],
            },
            "sentiment": {
                "high": ["rating", "upgrade", "downgrade", "buy", "sell", "hold"],
                "medium": ["analyst", "recommendation", "target price", "outlook"],
                "context": ["opinion", "sentiment", "market mood"],
            },
            "competitive": {
                "high": [
                    "market share",
                    "competitor",
                    "competitive advantage",
                    "industry",
                ],
                "medium": ["peer", "comparison", "market position", "sector"],
                "context": ["competitive landscape", "industry analysis"],
            },
        }

        self.domain_credibility_scores = {
            "reuters.com": 0.95,
            "bloomberg.com": 0.95,
            "wsj.com": 0.90,
            "ft.com": 0.90,
            "marketwatch.com": 0.85,
            "cnbc.com": 0.80,
            "yahoo.com": 0.75,
            "seekingalpha.com": 0.80,
            "fool.com": 0.70,
            "investing.com": 0.75,
        }

    async def filter_and_prioritize_sources(
        self,
        sources: list[dict],
        research_focus: str,
        time_budget: float,
        target_source_count: int | None = None,
        current_confidence: float = 0.0,
    ) -> list[dict]:
        """Filter and prioritize sources based on relevance, quality, and time constraints."""

        if not sources:
            return []

        # Determine target count based on time budget and confidence
        if target_source_count is None:
            target_source_count = self._calculate_optimal_source_count(
                time_budget, current_confidence, len(sources)
            )

        # Quick relevance scoring without LLM
        scored_sources = []
        for source in sources:
            relevance_score = self._calculate_relevance_score(source, research_focus)
            credibility_score = self._get_source_credibility(source)
            recency_score = self._calculate_recency_score(source.get("published_date"))

            # Combined score with weights
            combined_score = (
                relevance_score * 0.5 + credibility_score * 0.3 + recency_score * 0.2
            )

            if combined_score > 0.3:  # Relevance threshold
                scored_sources.append((combined_score, source))

        # Sort by combined score
        scored_sources.sort(key=lambda x: x[0], reverse=True)

        # Select diverse sources
        selected_sources = self._select_diverse_sources(
            scored_sources, target_source_count, research_focus
        )

        # Pre-process content for faster LLM processing
        processed_sources = []
        for score, source in selected_sources:
            processed_source = self._preprocess_content(
                source, research_focus, time_budget
            )
            processed_source["relevance_score"] = score
            processed_sources.append(processed_source)

        return processed_sources

    def _calculate_optimal_source_count(
        self, time_budget: float, current_confidence: float, available_sources: int
    ) -> int:
        """Calculate optimal number of sources to process given constraints."""

        # Base count from time budget
        if time_budget < 20:
            base_count = 3
        elif time_budget < 40:
            base_count = 6
        elif time_budget < 80:
            base_count = 10
        else:
            base_count = 15

        # Adjust for confidence level
        if current_confidence > 0.7:
            # High confidence - fewer sources needed
            confidence_multiplier = 0.7
        elif current_confidence < 0.4:
            # Low confidence - more sources helpful
            confidence_multiplier = 1.2
        else:
            confidence_multiplier = 1.0

        # Final calculation
        target_count = int(base_count * confidence_multiplier)

        # Ensure we don't exceed available sources
        return min(target_count, available_sources, 20)  # Cap at 20

    def _calculate_relevance_score(self, source: dict, research_focus: str) -> float:
        """Calculate relevance score using keyword matching and heuristics."""

        content = source.get("content", "").lower()
        title = source.get("title", "").lower()

        if not content and not title:
            return 0.0

        focus_keywords = self.relevance_keywords.get(research_focus, {})

        # High-value keywords
        high_keywords = focus_keywords.get("high", [])
        high_score = sum(1 for keyword in high_keywords if keyword in content) / max(
            len(high_keywords), 1
        )

        # Medium-value keywords
        medium_keywords = focus_keywords.get("medium", [])
        medium_score = sum(
            1 for keyword in medium_keywords if keyword in content
        ) / max(len(medium_keywords), 1)

        # Context keywords
        context_keywords = focus_keywords.get("context", [])
        context_score = sum(
            1 for keyword in context_keywords if keyword in content
        ) / max(len(context_keywords), 1)

        # Title relevance (titles are more focused)
        title_high_score = sum(
            1 for keyword in high_keywords if keyword in title
        ) / max(len(high_keywords), 1)

        # Combine scores with weights
        relevance_score = (
            high_score * 0.4
            + medium_score * 0.25
            + context_score * 0.15
            + title_high_score * 0.2
        )

        # Boost for very relevant titles
        if any(keyword in title for keyword in high_keywords):
            relevance_score *= 1.2

        return min(relevance_score, 1.0)

    def _get_source_credibility(self, source: dict) -> float:
        """Calculate source credibility based on domain and other factors."""

        url = source.get("url", "").lower()

        # Domain-based credibility
        domain_score = 0.5  # Default
        for domain, score in self.domain_credibility_scores.items():
            if domain in url:
                domain_score = score
                break

        # Boost for specific high-quality indicators
        if any(indicator in url for indicator in [".gov", ".edu", "sec.gov"]):
            domain_score = min(domain_score + 0.2, 1.0)

        # Penalty for low-quality indicators
        if any(indicator in url for indicator in ["blog", "forum", "reddit"]):
            domain_score *= 0.8

        return domain_score

    def _calculate_recency_score(self, published_date: str) -> float:
        """Calculate recency score based on publication date."""

        if not published_date:
            return 0.5  # Default for unknown dates

        try:
            # Parse date (handle various formats)
            if "T" in published_date:
                pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            else:
                pub_date = datetime.strptime(published_date, "%Y-%m-%d")

            # Calculate days old
            days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days

            # Scoring based on age
            if days_old <= 1:
                return 1.0  # Very recent
            elif days_old <= 7:
                return 0.9  # Recent
            elif days_old <= 30:
                return 0.7  # Fairly recent
            elif days_old <= 90:
                return 0.5  # Moderately old
            else:
                return 0.3  # Old

        except (ValueError, TypeError):
            return 0.5  # Default for unparseable dates

    def _select_diverse_sources(
        self,
        scored_sources: list[tuple[float, dict]],
        target_count: int,
        research_focus: str,
    ) -> list[tuple[float, dict]]:
        """Select diverse sources to avoid redundancy."""

        if len(scored_sources) <= target_count:
            return scored_sources

        selected = []
        used_domains = set()

        # First pass: select high-scoring diverse sources
        for score, source in scored_sources:
            if len(selected) >= target_count:
                break

            url = source.get("url", "")
            domain = self._extract_domain(url)

            # Ensure diversity by domain (max 2 from same domain initially)
            domain_count = sum(
                1
                for _, s in selected
                if self._extract_domain(s.get("url", "")) == domain
            )

            if domain_count < 2 or len(selected) < target_count // 2:
                selected.append((score, source))
                used_domains.add(domain)

        # Second pass: fill remaining slots with best remaining sources
        remaining_needed = target_count - len(selected)
        if remaining_needed > 0:
            remaining_sources = scored_sources[len(selected) :]
            selected.extend(remaining_sources[:remaining_needed])

        return selected[:target_count]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            if "//" in url:
                domain = url.split("//")[1].split("/")[0]
                return domain.replace("www.", "")
            return url
        except:
            return url

    def _preprocess_content(
        self, source: dict, research_focus: str, time_budget: float
    ) -> dict:
        """Pre-process content to optimize for LLM analysis."""

        content = source.get("content", "")
        if not content:
            return source

        # Determine content length limit based on time budget
        if time_budget < 30:
            max_length = 800  # Emergency mode
        elif time_budget < 60:
            max_length = 1200  # Fast mode
        else:
            max_length = 2000  # Standard mode

        # If content is already short enough, return as-is
        if len(content) <= max_length:
            source_copy = source.copy()
            source_copy["original_length"] = len(content)
            source_copy["filtered"] = False
            return source_copy

        # Extract most relevant sentences/paragraphs
        sentences = re.split(r"[.!?]+", content)
        focus_keywords = self.relevance_keywords.get(research_focus, {})
        all_keywords = (
            focus_keywords.get("high", [])
            + focus_keywords.get("medium", [])
            + focus_keywords.get("context", [])
        )

        # Score sentences by keyword relevance
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue

            sentence_lower = sentence.lower()
            keyword_count = sum(
                1 for keyword in all_keywords if keyword in sentence_lower
            )

            # Boost for financial numbers and percentages
            has_numbers = bool(re.search(r"\$?[\d,]+\.?\d*[%kmbKMB]?", sentence))
            number_boost = 0.5 if has_numbers else 0

            sentence_score = keyword_count + number_boost
            if sentence_score > 0:
                scored_sentences.append((sentence_score, sentence.strip()))

        # Sort by relevance and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Build filtered content
        filtered_content = ""
        for score, sentence in scored_sentences:
            if len(filtered_content) + len(sentence) > max_length:
                break
            filtered_content += sentence + ". "

        # If no relevant sentences found, take first part of original content
        if not filtered_content:
            filtered_content = content[:max_length]

        # Create processed source
        source_copy = source.copy()
        source_copy["content"] = filtered_content.strip()
        source_copy["original_length"] = len(content)
        source_copy["filtered_length"] = len(filtered_content)
        source_copy["filtered"] = True
        source_copy["compression_ratio"] = len(filtered_content) / len(content)

        return source_copy


# Export main classes for integration
__all__ = [
    "AdaptiveModelSelector",
    "ProgressiveTokenBudgeter",
    "ParallelLLMProcessor",
    "OptimizedPromptEngine",
    "ConfidenceTracker",
    "IntelligentContentFilter",
    "ModelConfiguration",
    "TokenAllocation",
    "ResearchPhase",
]
