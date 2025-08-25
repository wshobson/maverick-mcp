"""
Parallel Research Execution Utilities

This module provides infrastructure for spawning and managing parallel research
subagents for comprehensive financial analysis.
"""

import asyncio
import logging
import time
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..agents.circuit_breaker import circuit_breaker
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ParallelResearchConfig:
    """Configuration for parallel research operations."""
    
    def __init__(
        self,
        max_concurrent_agents: int = 4,
        timeout_per_agent: int = 300,  # 5 minutes per agent
        enable_fallbacks: bool = True,
        rate_limit_delay: float = 1.0,  # Delay between agent starts
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
        focus_areas: List[str],
        priority: int = 1,
        timeout: Optional[int] = None,
    ):
        self.task_id = task_id
        self.task_type = task_type  # fundamental, technical, sentiment, competitive
        self.target_topic = target_topic
        self.focus_areas = focus_areas
        self.priority = priority
        self.timeout = timeout
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "pending"  # pending, running, completed, failed
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None


class ResearchResult:
    """Aggregated results from parallel research execution."""
    
    def __init__(self):
        self.task_results: Dict[str, ResearchTask] = {}
        self.synthesis: Optional[Dict[str, Any]] = None
        self.total_execution_time: float = 0.0
        self.successful_tasks: int = 0
        self.failed_tasks: int = 0
        self.parallel_efficiency: float = 0.0


class ParallelResearchOrchestrator:
    """Orchestrates parallel research agent execution."""
    
    def __init__(self, config: Optional[ParallelResearchConfig] = None):
        self.config = config or ParallelResearchConfig()
        self.active_tasks: Dict[str, ResearchTask] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        
    async def execute_parallel_research(
        self,
        tasks: List[ResearchTask],
        research_executor, 
        synthesis_callback: Optional[callable] = None
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
        logger.info(f"Starting parallel research execution with {len(tasks)} tasks")
        start_time = time.time()
        
        # Create result container
        result = ResearchResult()
        
        # Prepare tasks for execution
        prepared_tasks = await self._prepare_tasks(tasks)
        
        try:
            # Execute tasks in parallel with concurrency control
            task_coroutines = [
                self._execute_single_task(task, research_executor)
                for task in prepared_tasks
            ]
            
            # Add rate limiting between task starts
            if self.config.rate_limit_delay > 0:
                await self._stagger_task_starts(task_coroutines)
            
            # Wait for all tasks to complete
            completed_tasks = await asyncio.gather(
                *task_coroutines, 
                return_exceptions=True
            )
            
            # Process results
            result = await self._process_task_results(
                prepared_tasks, completed_tasks, start_time
            )
            
            # Synthesize results if callback provided
            if synthesis_callback and result.successful_tasks > 0:
                try:
                    result.synthesis = await synthesis_callback(result.task_results)
                except Exception as e:
                    logger.error(f"Result synthesis failed: {e}")
                    result.synthesis = {"error": f"Synthesis failed: {str(e)}"}
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel research execution failed: {e}")
            result.total_execution_time = time.time() - start_time
            return result
    
    async def _prepare_tasks(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
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
            
        return prepared[:self.config.max_concurrent_agents]
    
    @circuit_breaker("parallel_research_task", failure_threshold=2, recovery_timeout=30)
    async def _execute_single_task(
        self, 
        task: ResearchTask, 
        research_executor
    ) -> ResearchTask:
        """Execute a single research task with error handling."""
        async with self._semaphore:  # Concurrency control
            task.start_time = time.time()
            task.status = "running"
            
            try:
                logger.info(f"Starting research task: {task.task_id} ({task.task_type})")
                
                # Execute the research with timeout
                result = await asyncio.wait_for(
                    research_executor(task),
                    timeout=task.timeout
                )
                
                task.result = result
                task.status = "completed"
                task.end_time = time.time()
                
                logger.info(
                    f"Task {task.task_id} completed in "
                    f"{task.end_time - task.start_time:.2f}s"
                )
                
                return task
                
            except asyncio.TimeoutError:
                task.error = f"Task timeout after {task.timeout}s"
                task.status = "failed"
                logger.error(f"Task {task.task_id} timed out")
                
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                logger.error(f"Task {task.task_id} failed: {e}")
                
            finally:
                task.end_time = time.time()
                
            return task
    
    async def _stagger_task_starts(self, coroutines: List[Coroutine]):
        """Add delays between task starts to avoid overwhelming APIs."""
        if len(coroutines) <= 1:
            return
            
        # Add progressive delays to task starts
        for i in range(1, len(coroutines)):
            await asyncio.sleep(self.config.rate_limit_delay * i)
    
    async def _process_task_results(
        self, 
        tasks: List[ResearchTask], 
        completed_tasks: List[Any], 
        start_time: float
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
                for task in tasks if task.start_time
            )
            result.parallel_efficiency = (
                total_sequential_time / result.total_execution_time
            ) if total_sequential_time > 0 else 0.0
        
        logger.info(
            f"Parallel research completed: {result.successful_tasks} successful, "
            f"{result.failed_tasks} failed, {result.parallel_efficiency:.2f}x speedup"
        )
        
        return result


class TaskDistributionEngine:
    """Intelligent task distribution for research topics."""
    
    TASK_TYPES = {
        "fundamental": {
            "keywords": ["earnings", "revenue", "profit", "cash flow", "debt", "valuation"],
            "focus_areas": ["financials", "fundamentals", "earnings", "balance_sheet"]
        },
        "technical": {
            "keywords": ["price", "chart", "trend", "support", "resistance", "momentum"],
            "focus_areas": ["technical_analysis", "chart_patterns", "indicators"]
        },
        "sentiment": {
            "keywords": ["sentiment", "news", "analyst", "opinion", "rating", "recommendation"],
            "focus_areas": ["market_sentiment", "analyst_ratings", "news_sentiment"]
        },
        "competitive": {
            "keywords": ["competitor", "market share", "industry", "competitive", "peers"],
            "focus_areas": ["competitive_analysis", "industry_analysis", "market_position"]
        }
    }
    
    def distribute_research_tasks(
        self, 
        topic: str, 
        session_id: str,
        focus_areas: Optional[List[str]] = None
    ) -> List[ResearchTask]:
        """
        Intelligently distribute a research topic into specialized tasks.
        
        Args:
            topic: Main research topic
            session_id: Session identifier for tracking
            focus_areas: Optional specific areas to focus on
            
        Returns:
            List of specialized research tasks
        """
        tasks = []
        topic_lower = topic.lower()
        
        # Determine which task types are relevant
        relevant_types = self._analyze_topic_relevance(topic_lower, focus_areas)
        
        for task_type, score in relevant_types.items():
            if score > 0.3:  # Relevance threshold
                task = ResearchTask(
                    task_id=f"{session_id}_{task_type}",
                    task_type=task_type,
                    target_topic=topic,
                    focus_areas=self.TASK_TYPES[task_type]["focus_areas"],
                    priority=int(score * 10)  # Convert to 1-10 priority
                )
                tasks.append(task)
        
        # Ensure at least one task (fallback to fundamental analysis)
        if not tasks:
            tasks.append(ResearchTask(
                task_id=f"{session_id}_fundamental",
                task_type="fundamental",
                target_topic=topic,
                focus_areas=["general_analysis"],
                priority=5
            ))
        
        logger.info(f"Distributed '{topic}' into {len(tasks)} parallel tasks")
        return tasks
    
    def _analyze_topic_relevance(
        self, 
        topic: str, 
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Analyze topic relevance to different research types."""
        relevance_scores = {}
        
        for task_type, config in self.TASK_TYPES.items():
            score = 0.0
            
            # Score based on keywords in topic
            keyword_matches = sum(
                1 for keyword in config["keywords"] 
                if keyword in topic
            )
            score += keyword_matches / len(config["keywords"]) * 0.6
            
            # Score based on focus areas
            if focus_areas:
                focus_matches = sum(
                    1 for focus in focus_areas 
                    if any(area in focus.lower() for area in config["focus_areas"])
                )
                score += focus_matches / len(config["focus_areas"]) * 0.4
            else:
                # Default relevance for common research types
                score += {"fundamental": 0.8, "sentiment": 0.6, "technical": 0.4, "competitive": 0.5}.get(task_type, 0.3)
            
            relevance_scores[task_type] = min(score, 1.0)
        
        return relevance_scores


# Export key classes for easy import
__all__ = [
    "ParallelResearchConfig",
    "ResearchTask", 
    "ResearchResult",
    "ParallelResearchOrchestrator",
    "TaskDistributionEngine"
]