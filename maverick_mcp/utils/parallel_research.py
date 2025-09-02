"""
Parallel Research Execution Utilities

This module provides infrastructure for spawning and managing parallel research
subagents for comprehensive financial analysis.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from ..agents.circuit_breaker import circuit_breaker
from ..config.settings import get_settings
from .orchestration_logging import (
    get_orchestration_logger,
    log_agent_execution,
    log_method_call,
    log_parallel_execution,
    log_performance_metrics,
    log_resource_usage,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class ParallelResearchConfig:
    """Configuration for parallel research operations."""

    def __init__(
        self,
        max_concurrent_agents: int = 4,
        timeout_per_agent: int = 180,  # 3 minutes per agent for thorough research
        enable_fallbacks: bool = False,  # Disabled by default for speed
        rate_limit_delay: float = 0.5,  # Reduced delay for faster parallelization
    ):
        self.max_concurrent_agents = max_concurrent_agents
        self.timeout_per_agent = timeout_per_agent
        self.enable_fallbacks = enable_fallbacks
        self.rate_limit_delay = rate_limit_delay


class ResearchTask:
    """Represents a single research task for parallel execution."""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        target_topic: str,
        focus_areas: list[str],
        priority: int = 1,
        timeout: int | None = None,
    ):
        self.task_id = task_id
        self.task_type = task_type  # fundamental, technical, sentiment, competitive
        self.target_topic = target_topic
        self.focus_areas = focus_areas
        self.priority = priority
        self.timeout = timeout
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.status: str = "pending"  # pending, running, completed, failed
        self.result: dict[str, Any] | None = None
        self.error: str | None = None


class ResearchResult:
    """Aggregated results from parallel research execution."""

    def __init__(self):
        self.task_results: dict[str, ResearchTask] = {}
        self.synthesis: dict[str, Any] | None = None
        self.total_execution_time: float = 0.0
        self.successful_tasks: int = 0
        self.failed_tasks: int = 0
        self.parallel_efficiency: float = 0.0


class ParallelResearchOrchestrator:
    """Orchestrates parallel research agent execution."""

    def __init__(self, config: ParallelResearchConfig | None = None):
        self.config = config or ParallelResearchConfig()
        self.active_tasks: dict[str, ResearchTask] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        self.orchestration_logger = get_orchestration_logger("ParallelOrchestrator")

        # Log initialization
        self.orchestration_logger.info(
            "🎛️ ORCHESTRATOR_INIT",
            max_agents=self.config.max_concurrent_agents,
            timeout_per_agent=self.config.timeout_per_agent,
            fallbacks_enabled=self.config.enable_fallbacks,
        )

    @log_method_call(component="ParallelOrchestrator", include_timing=True)
    async def execute_parallel_research(
        self,
        tasks: list[ResearchTask],
        research_executor,
        synthesis_callback: Callable[..., Any] | None = None,
    ) -> ResearchResult:
        """
        Execute multiple research tasks in parallel with intelligent coordination.

        Args:
            tasks: List of research tasks to execute
            research_executor: Function to execute individual research tasks
            synthesis_callback: Optional function to synthesize results

        Returns:
            ResearchResult with aggregated results and synthesis
        """
        self.orchestration_logger.set_request_context(
            session_id=tasks[0].task_id.split("_")[0] if tasks else "unknown",
            task_count=len(tasks),
        )

        # Log task overview
        task_types = [task.task_type for task in tasks]
        self.orchestration_logger.info(
            "📋 TASK_OVERVIEW",
            task_count=len(tasks),
            task_types=task_types,
            max_concurrent=self.config.max_concurrent_agents,
        )

        start_time = time.time()

        # Create result container
        result = ResearchResult()

        with log_parallel_execution(
            "ParallelOrchestrator", "research execution", len(tasks)
        ) as exec_logger:
            try:
                # Prepare tasks for execution
                exec_logger.info("🔧 TASK_PREPARATION", total_tasks=len(tasks))
                prepared_tasks = await self._prepare_tasks(tasks)
                exec_logger.info(
                    "✅ TASKS_PREPARED", prepared_count=len(prepared_tasks)
                )

                # Execute tasks in parallel with concurrency control
                task_coroutines = [
                    self._execute_single_task(task, research_executor)
                    for task in prepared_tasks
                ]

                # Add rate limiting between task starts
                if self.config.rate_limit_delay > 0:
                    exec_logger.info(
                        "⏱️ APPLYING_RATE_LIMITING", delay=self.config.rate_limit_delay
                    )
                    await self._stagger_task_starts(task_coroutines)

                # Log parallel execution start
                exec_logger.info(
                    "🚀 PARALLEL_EXECUTION_START", concurrent_tasks=len(task_coroutines)
                )

                # Wait for all tasks to complete
                completed_tasks = await asyncio.gather(
                    *task_coroutines, return_exceptions=True
                )

                exec_logger.info(
                    "🏁 PARALLEL_EXECUTION_COMPLETE",
                    completed_count=len(completed_tasks),
                )

                # Process results
                result = await self._process_task_results(
                    prepared_tasks, completed_tasks, start_time
                )

                # Log performance metrics
                log_performance_metrics(
                    "ParallelOrchestrator",
                    {
                        "total_tasks": len(tasks),
                        "successful_tasks": result.successful_tasks,
                        "failed_tasks": result.failed_tasks,
                        "parallel_efficiency": result.parallel_efficiency,
                        "total_duration": result.total_execution_time,
                    },
                )

                # Synthesize results if callback provided
                if synthesis_callback and result.successful_tasks > 0:
                    exec_logger.info(
                        "🧠 SYNTHESIS_START", successful_tasks=result.successful_tasks
                    )
                    try:
                        synthesis_start = time.time()
                        result.synthesis = await synthesis_callback(result.task_results)
                        synthesis_duration = time.time() - synthesis_start
                        exec_logger.info(
                            "✅ SYNTHESIS_SUCCESS",
                            duration=f"{synthesis_duration:.3f}s",
                        )
                    except Exception as e:
                        exec_logger.error("❌ SYNTHESIS_FAILED", error=str(e))
                        result.synthesis = {"error": f"Synthesis failed: {str(e)}"}
                else:
                    exec_logger.info(
                        "⏭️ SYNTHESIS_SKIPPED",
                        reason="no_callback_or_no_successful_tasks",
                    )

                return result

            except Exception as e:
                exec_logger.error("💥 PARALLEL_EXECUTION_FAILED", error=str(e))
                result.total_execution_time = time.time() - start_time
                return result

    async def _prepare_tasks(self, tasks: list[ResearchTask]) -> list[ResearchTask]:
        """Prepare tasks for execution by setting timeouts and priorities."""
        prepared = []

        for task in sorted(tasks, key=lambda t: t.priority, reverse=True):
            # Set default timeout if not specified
            if not task.timeout:
                task.timeout = self.config.timeout_per_agent

            # Set task to pending status
            task.status = "pending"
            self.active_tasks[task.task_id] = task
            prepared.append(task)

        return prepared[: self.config.max_concurrent_agents]

    @circuit_breaker("parallel_research_task", failure_threshold=2, recovery_timeout=30)
    async def _execute_single_task(
        self, task: ResearchTask, research_executor
    ) -> ResearchTask:
        """Execute a single research task with error handling."""
        async with self._semaphore:  # Concurrency control
            task.start_time = time.time()
            task.status = "running"

            with log_agent_execution(
                task.task_type, task.task_id, task.focus_areas
            ) as agent_logger:
                try:
                    agent_logger.info(
                        "🎯 TASK_EXECUTION_START",
                        timeout=task.timeout,
                        priority=task.priority,
                        focus_areas=task.focus_areas,
                    )

                    # Execute the research with timeout
                    result = await asyncio.wait_for(
                        research_executor(task), timeout=task.timeout
                    )

                    task.result = result
                    task.status = "completed"
                    task.end_time = time.time()

                    # Log successful completion with metrics
                    execution_time = task.end_time - task.start_time
                    agent_logger.info(
                        "✨ TASK_EXECUTION_SUCCESS",
                        execution_time=f"{execution_time:.3f}s",
                        result_size=len(str(result)) if result else 0,
                    )

                    # Log resource usage if available
                    if isinstance(result, dict) and "metrics" in result:
                        log_resource_usage(
                            f"{task.task_type}Agent",
                            api_calls=result["metrics"].get("api_calls"),
                            cache_hits=result["metrics"].get("cache_hits"),
                        )

                    return task

                except TimeoutError:
                    task.error = f"Task timeout after {task.timeout}s"
                    task.status = "failed"
                    agent_logger.error(
                        "⏰ TASK_TIMEOUT",
                        timeout=task.timeout,
                        elapsed_time=f"{time.time() - task.start_time:.3f}s",
                    )

                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
                    agent_logger.error(
                        "💥 TASK_EXECUTION_FAILED",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

                finally:
                    task.end_time = time.time()

                return task

    async def _stagger_task_starts(self, coroutines: list[Coroutine]):
        """Add delays between task starts to avoid overwhelming APIs."""
        if len(coroutines) <= 1:
            return

        # Add progressive delays to task starts
        for i in range(1, len(coroutines)):
            await asyncio.sleep(self.config.rate_limit_delay * i)

    async def _process_task_results(
        self, tasks: list[ResearchTask], completed_tasks: list[Any], start_time: float
    ) -> ResearchResult:
        """Process and aggregate results from completed tasks."""
        result = ResearchResult()
        result.total_execution_time = time.time() - start_time

        for task in tasks:
            result.task_results[task.task_id] = task

            if task.status == "completed":
                result.successful_tasks += 1
            else:
                result.failed_tasks += 1

        # Calculate parallel efficiency
        if result.total_execution_time > 0:
            total_sequential_time = sum(
                (task.end_time or 0) - (task.start_time or 0)
                for task in tasks
                if task.start_time
            )
            result.parallel_efficiency = (
                (total_sequential_time / result.total_execution_time)
                if total_sequential_time > 0
                else 0.0
            )

        logger.info(
            f"Parallel research completed: {result.successful_tasks} successful, "
            f"{result.failed_tasks} failed, {result.parallel_efficiency:.2f}x speedup"
        )

        return result


class TaskDistributionEngine:
    """Intelligent task distribution for research topics."""

    TASK_TYPES = {
        "fundamental": {
            "keywords": [
                "earnings",
                "revenue",
                "profit",
                "cash flow",
                "debt",
                "valuation",
            ],
            "focus_areas": ["financials", "fundamentals", "earnings", "balance_sheet"],
        },
        "technical": {
            "keywords": [
                "price",
                "chart",
                "trend",
                "support",
                "resistance",
                "momentum",
            ],
            "focus_areas": ["technical_analysis", "chart_patterns", "indicators"],
        },
        "sentiment": {
            "keywords": [
                "sentiment",
                "news",
                "analyst",
                "opinion",
                "rating",
                "recommendation",
            ],
            "focus_areas": ["market_sentiment", "analyst_ratings", "news_sentiment"],
        },
        "competitive": {
            "keywords": [
                "competitor",
                "market share",
                "industry",
                "competitive",
                "peers",
            ],
            "focus_areas": [
                "competitive_analysis",
                "industry_analysis",
                "market_position",
            ],
        },
    }

    @log_method_call(component="TaskDistributionEngine", include_timing=True)
    def distribute_research_tasks(
        self, topic: str, session_id: str, focus_areas: list[str] | None = None
    ) -> list[ResearchTask]:
        """
        Intelligently distribute a research topic into specialized tasks.

        Args:
            topic: Main research topic
            session_id: Session identifier for tracking
            focus_areas: Optional specific areas to focus on

        Returns:
            List of specialized research tasks
        """
        distribution_logger = get_orchestration_logger("TaskDistributionEngine")
        distribution_logger.set_request_context(session_id=session_id)

        distribution_logger.info(
            "🎯 TASK_DISTRIBUTION_START",
            topic=topic[:100],
            focus_areas=focus_areas,
            session_id=session_id,
        )

        tasks = []
        topic_lower = topic.lower()

        # Determine which task types are relevant
        relevant_types = self._analyze_topic_relevance(topic_lower, focus_areas)

        # Log relevance analysis results
        distribution_logger.info(
            "🧠 RELEVANCE_ANALYSIS",
            **{
                f"{task_type}_score": f"{score:.2f}"
                for task_type, score in relevant_types.items()
            },
        )

        # Create tasks for relevant types
        created_tasks = []
        for task_type, score in relevant_types.items():
            if score > 0.3:  # Relevance threshold
                task = ResearchTask(
                    task_id=f"{session_id}_{task_type}",
                    task_type=task_type,
                    target_topic=topic,
                    focus_areas=self.TASK_TYPES[task_type]["focus_areas"],
                    priority=int(score * 10),  # Convert to 1-10 priority
                )
                tasks.append(task)
                created_tasks.append(
                    {
                        "type": task_type,
                        "priority": task.priority,
                        "score": score,
                        "focus_areas": task.focus_areas[:3],  # Limit for logging
                    }
                )

        # Log created tasks
        if created_tasks:
            distribution_logger.info(
                "✅ TASKS_CREATED",
                task_count=len(created_tasks),
                task_details=created_tasks,
            )

        # Ensure at least one task (fallback to fundamental analysis)
        if not tasks:
            distribution_logger.warning(
                "⚠️ NO_RELEVANT_TASKS_FOUND - using fallback",
                threshold=0.3,
                max_score=max(relevant_types.values()) if relevant_types else 0,
            )

            fallback_task = ResearchTask(
                task_id=f"{session_id}_fundamental",
                task_type="fundamental",
                target_topic=topic,
                focus_areas=["general_analysis"],
                priority=5,
            )
            tasks.append(fallback_task)

            distribution_logger.info(
                "🔄 FALLBACK_TASK_CREATED", task_type="fundamental"
            )

        # Final summary
        task_summary = {
            "total_tasks": len(tasks),
            "task_types": [t.task_type for t in tasks],
            "avg_priority": sum(t.priority for t in tasks) / len(tasks) if tasks else 0,
        }

        distribution_logger.info("🎉 TASK_DISTRIBUTION_COMPLETE", **task_summary)

        return tasks

    def _analyze_topic_relevance(
        self, topic: str, focus_areas: list[str] | None = None
    ) -> dict[str, float]:
        """Analyze topic relevance to different research types."""
        relevance_scores = {}

        for task_type, config in self.TASK_TYPES.items():
            score = 0.0

            # Score based on keywords in topic
            keyword_matches = sum(
                1 for keyword in config["keywords"] if keyword in topic
            )
            score += keyword_matches / len(config["keywords"]) * 0.6

            # Score based on focus areas
            if focus_areas:
                focus_matches = sum(
                    1
                    for focus in focus_areas
                    if any(area in focus.lower() for area in config["focus_areas"])
                )
                score += focus_matches / len(config["focus_areas"]) * 0.4
            else:
                # Default relevance for common research types
                score += {
                    "fundamental": 0.8,
                    "sentiment": 0.6,
                    "technical": 0.4,
                    "competitive": 0.5,
                }.get(task_type, 0.3)

            relevance_scores[task_type] = min(score, 1.0)

        return relevance_scores


# Export key classes for easy import
__all__ = [
    "ParallelResearchConfig",
    "ResearchTask",
    "ResearchResult",
    "ParallelResearchOrchestrator",
    "TaskDistributionEngine",
]
