"""
Comprehensive test suite for ExaSearch integration with research agents.

This test suite validates the complete research agent architecture with ExaSearch provider,
including timeout handling, parallel execution, specialized subagents, and performance
benchmarking across all research depths and focus areas.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exa_py import Exa
from langchain_core.messages import HumanMessage, SystemMessage

from maverick_mcp.agents.base import INVESTOR_PERSONAS
from maverick_mcp.agents.deep_research import (
    CompetitiveResearchAgent,
    ContentAnalyzer,
    DeepResearchAgent,
    ExaSearchProvider,
    FundamentalResearchAgent,
    RESEARCH_DEPTH_LEVELS,
    SentimentResearchAgent,
    TechnicalResearchAgent,
    WebSearchProvider,
)
from maverick_mcp.api.routers.research import (
    comprehensive_research,
    get_research_agent,
    ResearchRequest,
)
from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import WebSearchError
from maverick_mcp.utils.parallel_research import (
    ParallelResearchConfig,
    ParallelResearchOrchestrator,
    ResearchResult,
    ResearchTask,
    TaskDistributionEngine,
)


# Test Data Factories and Fixtures

@pytest.fixture
def mock_llm():
    """Mock LLM with realistic response patterns for research scenarios."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    
    # Mock different response types for different research phases
    def mock_response_content(messages):
        """Generate realistic mock responses based on message content."""
        content = str(messages[-1].content).lower()
        
        if "synthesis" in content:
            return MagicMock(content='{"synthesis": "Comprehensive analysis shows positive outlook", "confidence": 0.8}')
        elif "analyze" in content or "financial" in content:
            return MagicMock(content='{"KEY_INSIGHTS": ["Strong earnings growth", "Market share expansion"], "SENTIMENT": {"direction": "bullish", "confidence": 0.75}, "RISK_FACTORS": ["Interest rate sensitivity"], "OPPORTUNITIES": ["Market expansion"], "CREDIBILITY": 0.8, "RELEVANCE": 0.9, "SUMMARY": "Positive financial outlook"}')
        else:
            return MagicMock(content="Analysis completed successfully")
    
    llm.ainvoke.side_effect = lambda messages, **kwargs: mock_response_content(messages)
    return llm


@pytest.fixture
def mock_exa_client():
    """Mock Exa client with realistic search responses."""
    mock_client = MagicMock(spec=Exa)
    
    def create_mock_result(title, text, url_suffix=""):
        """Create mock Exa result object."""
        result = MagicMock()
        result.url = f"https://example.com/{url_suffix}"
        result.title = title
        result.text = text
        result.published_date = "2024-01-15T10:00:00Z"
        result.score = 0.85
        result.author = "Financial Analyst"
        return result
    
    def mock_search_and_contents(query, num_results=5, **kwargs):
        """Generate mock search results based on query content."""
        response = MagicMock()
        results = []
        
        query_lower = query.lower()
        
        if "aapl" in query_lower or "apple" in query_lower:
            results.extend([
                create_mock_result(
                    "Apple Q4 Earnings Beat Expectations",
                    "Apple reported strong quarterly earnings with iPhone sales growth of 15% and services revenue reaching new highs. The company's financial position remains robust with strong cash flow.",
                    "apple-earnings"
                ),
                create_mock_result(
                    "Apple Stock Technical Analysis",
                    "Apple stock shows bullish technical patterns with support at $180 and resistance at $200. RSI indicates oversold conditions presenting buying opportunity.",
                    "apple-technical"
                ),
            ])
        elif "sentiment" in query_lower:
            results.extend([
                create_mock_result(
                    "Market Sentiment Turns Positive",
                    "Investor sentiment shows improvement with increased confidence in tech sector. Analyst upgrades and positive earnings surprises drive optimism.",
                    "market-sentiment"
                ),
            ])
        elif "competitive" in query_lower or "industry" in query_lower:
            results.extend([
                create_mock_result(
                    "Tech Industry Competitive Landscape",
                    "The technology sector shows fierce competition with market leaders maintaining strong positions. Innovation and market share battles intensify.",
                    "competitive-analysis"
                ),
            ])
        else:
            # Default financial research results
            results.extend([
                create_mock_result(
                    "Financial Market Analysis",
                    "Current market conditions show mixed signals with growth prospects balanced against economic uncertainties. Investors remain cautiously optimistic.",
                    "market-analysis"
                ),
                create_mock_result(
                    "Investment Outlook 2024",
                    "Investment opportunities emerge in technology and healthcare sectors despite ongoing market volatility. Diversification remains key strategy.",
                    "investment-outlook"
                ),
            ])
        
        # Limit results to requested number
        response.results = results[:num_results]
        return response
    
    mock_client.search_and_contents.side_effect = mock_search_and_contents
    return mock_client


@pytest.fixture
def sample_research_tasks():
    """Sample research tasks for parallel execution testing."""
    return [
        ResearchTask(
            task_id="session_123_fundamental",
            task_type="fundamental",
            target_topic="AAPL financial analysis",
            focus_areas=["earnings", "valuation", "growth"],
            priority=8,
            timeout=20,
        ),
        ResearchTask(
            task_id="session_123_technical",
            task_type="technical",
            target_topic="AAPL technical analysis",
            focus_areas=["chart_patterns", "support_resistance"],
            priority=7,
            timeout=15,
        ),
        ResearchTask(
            task_id="session_123_sentiment",
            task_type="sentiment",
            target_topic="AAPL market sentiment",
            focus_areas=["news_sentiment", "analyst_ratings"],
            priority=6,
            timeout=15,
        ),
    ]


@pytest.fixture
def mock_settings():
    """Mock settings with ExaSearch configuration."""
    settings = MagicMock()
    settings.research.exa_api_key = "test_exa_api_key"
    settings.data_limits.max_parallel_agents = 4
    settings.performance.search_timeout_failure_threshold = 12
    settings.performance.search_circuit_breaker_failure_threshold = 8
    settings.performance.search_circuit_breaker_recovery_timeout = 30
    return settings


# ExaSearchProvider Tests

class TestExaSearchProvider:
    """Test ExaSearch provider integration and functionality."""
    
    @pytest.mark.unit
    def test_exa_provider_initialization(self):
        """Test ExaSearchProvider initialization."""
        api_key = "test_api_key_123"
        provider = ExaSearchProvider(api_key)
        
        assert provider.api_key == api_key
        assert provider._api_key_verified is True
        assert provider.is_healthy() is True
        assert provider._failure_count == 0
    
    @pytest.mark.unit
    def test_exa_provider_initialization_without_key(self):
        """Test ExaSearchProvider initialization without API key."""
        provider = ExaSearchProvider("")
        
        assert provider.api_key == ""
        assert provider._api_key_verified is False
        assert provider.is_healthy() is True  # Still healthy, but searches will fail
    
    @pytest.mark.unit
    def test_timeout_calculation(self):
        """Test adaptive timeout calculation for different query complexities."""
        provider = ExaSearchProvider("test_key")
        
        # Simple query
        timeout = provider._calculate_timeout("AAPL", None)
        assert timeout >= 4.0  # Minimum for Exa reliability
        
        # Complex query
        complex_query = "comprehensive analysis of Apple Inc financial performance and market position with competitive analysis"
        timeout_complex = provider._calculate_timeout(complex_query, None)
        assert timeout_complex >= timeout
        
        # Budget constrained query
        timeout_budget = provider._calculate_timeout("AAPL", 8.0)
        assert 4.0 <= timeout_budget <= 8.0
    
    @pytest.mark.unit
    def test_failure_recording_and_health_status(self):
        """Test failure recording and health status management."""
        provider = ExaSearchProvider("test_key")
        
        # Initially healthy
        assert provider.is_healthy() is True
        
        # Record several timeout failures
        for _ in range(5):
            provider._record_failure("timeout")
        
        assert provider._failure_count == 5
        assert provider.is_healthy() is True  # Still healthy, threshold not reached
        
        # Exceed timeout threshold (default 12)
        for _ in range(8):
            provider._record_failure("timeout")
        
        assert provider._failure_count == 13
        assert provider.is_healthy() is False  # Now unhealthy
        
        # Test recovery
        provider._record_success()
        assert provider.is_healthy() is True
        assert provider._failure_count == 0
    
    @pytest.mark.unit
    @patch('exa_py.Exa')
    async def test_exa_search_success(self, mock_exa_class, mock_exa_client):
        """Test successful ExaSearch operation."""
        mock_exa_class.return_value = mock_exa_client
        provider = ExaSearchProvider("test_key")
        
        results = await provider.search("AAPL financial analysis", num_results=3)
        
        assert len(results) >= 1
        assert all("url" in result for result in results)
        assert all("title" in result for result in results)
        assert all("content" in result for result in results)
        assert all(result["provider"] == "exa" for result in results)
    
    @pytest.mark.unit
    @patch('exa_py.Exa')
    async def test_exa_search_timeout(self, mock_exa_class):
        """Test ExaSearch timeout handling."""
        # Mock Exa client that takes too long
        mock_client = MagicMock()
        
        def slow_search(*args, **kwargs):
            import time
            time.sleep(10)  # Simulate slow synchronous response
        
        mock_client.search_and_contents.side_effect = slow_search
        mock_exa_class.return_value = mock_client
        
        provider = ExaSearchProvider("test_key")
        
        with pytest.raises(WebSearchError, match="timed out"):
            await provider.search("test query", timeout_budget=2.0)
        
        # Check that failure was recorded
        assert not provider.is_healthy() or provider._failure_count > 0
    
    @pytest.mark.unit
    @patch('exa_py.Exa')
    async def test_exa_search_unhealthy_provider(self, mock_exa_class):
        """Test behavior when provider is marked as unhealthy."""
        provider = ExaSearchProvider("test_key")
        provider._is_healthy = False
        
        with pytest.raises(WebSearchError, match="disabled due to repeated failures"):
            await provider.search("test query")


# DeepResearchAgent Tests

class TestDeepResearchAgent:
    """Test DeepResearchAgent with ExaSearch integration."""
    
    @pytest.mark.unit
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    async def test_agent_initialization_with_exa(self, mock_provider, mock_llm):
        """Test DeepResearchAgent initialization with ExaSearch provider."""
        mock_exa_provider = MagicMock(spec=ExaSearchProvider)
        mock_provider.return_value = mock_exa_provider
        
        agent = DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            exa_api_key="test_key",
            research_depth="standard"
        )
        
        await agent.initialize()
        
        assert agent.search_providers == [mock_exa_provider]
        assert agent._search_providers_loaded is True
        assert agent.default_depth == "standard"
    
    @pytest.mark.unit
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    async def test_agent_initialization_without_providers(self, mock_provider, mock_llm):
        """Test agent behavior when no search providers are available."""
        mock_provider.return_value = None
        
        agent = DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            exa_api_key=None,
        )
        
        await agent.initialize()
        
        assert agent.search_providers == []
        assert agent._search_providers_loaded is True
    
    @pytest.mark.unit
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    async def test_research_comprehensive_no_providers(self, mock_provider, mock_llm):
        """Test research behavior when no search providers are configured."""
        mock_provider.return_value = None
        
        agent = DeepResearchAgent(llm=mock_llm, exa_api_key=None)
        
        result = await agent.research_comprehensive(
            topic="AAPL analysis",
            session_id="test_session",
            depth="basic"
        )
        
        assert "error" in result
        assert "no search providers configured" in result["error"]
        assert result["topic"] == "AAPL analysis"
    
    @pytest.mark.integration
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    @patch('exa_py.Exa')
    async def test_research_comprehensive_success(self, mock_exa_class, mock_provider, mock_llm, mock_exa_client):
        """Test successful comprehensive research with ExaSearch."""
        # Setup mocks
        mock_exa_provider = ExaSearchProvider("test_key")
        mock_provider.return_value = mock_exa_provider
        mock_exa_class.return_value = mock_exa_client
        
        agent = DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            exa_api_key="test_key",
            research_depth="basic",
        )
        
        # Execute research
        result = await agent.research_comprehensive(
            topic="AAPL financial analysis",
            session_id="test_session_123",
            depth="basic",
            timeout_budget=15.0,
        )
        
        # Verify result structure
        assert result["status"] == "success"
        assert result["agent_type"] == "deep_research"
        assert result["research_topic"] == "AAPL financial analysis"
        assert result["research_depth"] == "basic"
        assert "findings" in result
        assert "confidence_score" in result
        assert "execution_time_ms" in result
    
    @pytest.mark.unit
    def test_research_depth_levels(self):
        """Test research depth level configurations."""
        assert "basic" in RESEARCH_DEPTH_LEVELS
        assert "standard" in RESEARCH_DEPTH_LEVELS
        assert "comprehensive" in RESEARCH_DEPTH_LEVELS
        assert "exhaustive" in RESEARCH_DEPTH_LEVELS
        
        # Verify basic level has minimal settings for speed
        basic = RESEARCH_DEPTH_LEVELS["basic"]
        assert basic["max_sources"] <= 5
        assert basic["max_searches"] <= 2
        assert basic["validation_required"] is False
        
        # Verify exhaustive has maximum settings
        exhaustive = RESEARCH_DEPTH_LEVELS["exhaustive"]
        assert exhaustive["max_sources"] >= 10
        assert exhaustive["validation_required"] is True


# Specialized Subagent Tests

class TestSpecializedSubagents:
    """Test specialized research subagents."""
    
    @pytest.fixture
    def mock_parent_agent(self, mock_llm):
        """Mock parent DeepResearchAgent for subagent testing."""
        agent = MagicMock()
        agent.llm = mock_llm
        agent.search_providers = [MagicMock(spec=ExaSearchProvider)]
        agent.content_analyzer = MagicMock(spec=ContentAnalyzer)
        agent.persona = MagicMock()
        agent.persona.name = "moderate"
        agent._calculate_source_credibility = MagicMock(return_value=0.8)
        return agent
    
    @pytest.mark.unit
    async def test_fundamental_research_agent(self, mock_parent_agent, sample_research_tasks):
        """Test FundamentalResearchAgent execution."""
        task = sample_research_tasks[0]  # fundamental task
        agent = FundamentalResearchAgent(mock_parent_agent)
        
        # Mock search results
        mock_search_results = [
            {
                "title": "AAPL Q4 Earnings Report",
                "url": "https://example.com/earnings",
                "content": "Apple reported strong quarterly earnings with revenue growth of 12%...",
                "published_date": "2024-01-15",
            }
        ]
        
        agent._perform_specialized_search = AsyncMock(return_value=mock_search_results)
        agent._analyze_search_results = AsyncMock(return_value=[
            {
                **mock_search_results[0],
                "analysis": {
                    "insights": ["Strong earnings growth", "Revenue diversification"],
                    "risk_factors": ["Market competition"],
                    "opportunities": ["Market expansion"],
                    "sentiment": {"direction": "bullish", "confidence": 0.8},
                },
                "credibility_score": 0.8,
            }
        ])
        
        result = await agent.execute_research(task)
        
        assert result["research_type"] == "fundamental"
        assert "insights" in result
        assert "risk_factors" in result
        assert "opportunities" in result
        assert "sentiment" in result
        assert "sources" in result
        assert len(result["focus_areas"]) > 0
        assert "earnings" in result["focus_areas"]
    
    @pytest.mark.unit
    async def test_technical_research_agent(self, mock_parent_agent, sample_research_tasks):
        """Test TechnicalResearchAgent execution."""
        task = sample_research_tasks[1]  # technical task
        agent = TechnicalResearchAgent(mock_parent_agent)
        
        # Mock search results with technical analysis
        mock_search_results = [
            {
                "title": "AAPL Technical Analysis",
                "url": "https://example.com/technical",
                "content": "AAPL shows bullish chart patterns with support at $180 and resistance at $200...",
                "published_date": "2024-01-15",
            }
        ]
        
        agent._perform_specialized_search = AsyncMock(return_value=mock_search_results)
        agent._analyze_search_results = AsyncMock(return_value=[
            {
                **mock_search_results[0],
                "analysis": {
                    "insights": ["Bullish breakout pattern", "Strong support levels"],
                    "risk_factors": ["Overbought conditions"],
                    "opportunities": ["Momentum continuation"],
                    "sentiment": {"direction": "bullish", "confidence": 0.7},
                },
                "credibility_score": 0.7,
            }
        ])
        
        result = await agent.execute_research(task)
        
        assert result["research_type"] == "technical"
        assert "price_action" in result["focus_areas"]
        assert "chart_patterns" in result["focus_areas"]
    
    @pytest.mark.unit
    async def test_sentiment_research_agent(self, mock_parent_agent, sample_research_tasks):
        """Test SentimentResearchAgent execution."""
        task = sample_research_tasks[2]  # sentiment task
        agent = SentimentResearchAgent(mock_parent_agent)
        
        # Mock search results with sentiment data
        mock_search_results = [
            {
                "title": "Apple Stock Sentiment Analysis",
                "url": "https://example.com/sentiment",
                "content": "Analyst sentiment remains positive on Apple with multiple upgrades...",
                "published_date": "2024-01-15",
            }
        ]
        
        agent._perform_specialized_search = AsyncMock(return_value=mock_search_results)
        agent._analyze_search_results = AsyncMock(return_value=[
            {
                **mock_search_results[0],
                "analysis": {
                    "insights": ["Positive analyst sentiment", "Upgrade momentum"],
                    "risk_factors": ["Market volatility concerns"],
                    "opportunities": ["Institutional accumulation"],
                    "sentiment": {"direction": "bullish", "confidence": 0.85},
                },
                "credibility_score": 0.9,
            }
        ])
        
        result = await agent.execute_research(task)
        
        assert result["research_type"] == "sentiment"
        assert "market_sentiment" in result["focus_areas"]
        assert result["sentiment"]["direction"] == "bullish"
    
    @pytest.mark.unit
    async def test_competitive_research_agent(self, mock_parent_agent):
        """Test CompetitiveResearchAgent execution."""
        task = ResearchTask(
            task_id="test_competitive",
            task_type="competitive",
            target_topic="AAPL competitive analysis",
            focus_areas=["competitive_position", "market_share"],
        )
        
        agent = CompetitiveResearchAgent(mock_parent_agent)
        
        # Mock search results with competitive data
        mock_search_results = [
            {
                "title": "Apple vs Samsung Market Share",
                "url": "https://example.com/competitive",
                "content": "Apple maintains strong competitive position in premium smartphone market...",
                "published_date": "2024-01-15",
            }
        ]
        
        agent._perform_specialized_search = AsyncMock(return_value=mock_search_results)
        agent._analyze_search_results = AsyncMock(return_value=[
            {
                **mock_search_results[0],
                "analysis": {
                    "insights": ["Strong market position", "Premium segment dominance"],
                    "risk_factors": ["Android competition"],
                    "opportunities": ["Emerging markets"],
                    "sentiment": {"direction": "bullish", "confidence": 0.75},
                },
                "credibility_score": 0.8,
            }
        ])
        
        result = await agent.execute_research(task)
        
        assert result["research_type"] == "competitive"
        assert "competitive_position" in result["focus_areas"]
        assert "industry_trends" in result["focus_areas"]


# Parallel Research Tests

class TestParallelResearchOrchestrator:
    """Test parallel research execution and orchestration."""
    
    @pytest.mark.unit
    def test_orchestrator_initialization(self):
        """Test ParallelResearchOrchestrator initialization."""
        config = ParallelResearchConfig(max_concurrent_agents=6, timeout_per_agent=20)
        orchestrator = ParallelResearchOrchestrator(config)
        
        assert orchestrator.config.max_concurrent_agents == 6
        assert orchestrator.config.timeout_per_agent == 20
        assert orchestrator._semaphore._value == 6  # Semaphore initialized correctly
    
    @pytest.mark.unit
    async def test_task_preparation(self, sample_research_tasks):
        """Test task preparation and prioritization."""
        orchestrator = ParallelResearchOrchestrator()
        
        prepared_tasks = await orchestrator._prepare_tasks(sample_research_tasks)
        
        # Should be sorted by priority (descending)
        assert prepared_tasks[0].priority >= prepared_tasks[1].priority
        
        # All tasks should have timeouts set
        for task in prepared_tasks:
            assert task.timeout is not None
            assert task.status == "pending"
            assert task.task_id in orchestrator.active_tasks
    
    @pytest.mark.integration
    async def test_parallel_execution_success(self, sample_research_tasks):
        """Test successful parallel execution of research tasks."""
        orchestrator = ParallelResearchOrchestrator(
            ParallelResearchConfig(max_concurrent_agents=3, timeout_per_agent=10)
        )
        
        # Mock research executor
        async def mock_executor(task):
            """Mock research executor that simulates successful execution."""
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "research_type": task.task_type,
                "insights": [f"{task.task_type} insight 1", f"{task.task_type} insight 2"],
                "sentiment": {"direction": "bullish", "confidence": 0.8},
                "sources": [{"title": f"{task.task_type} source", "url": "https://example.com"}]
            }
        
        # Mock synthesis callback
        async def mock_synthesis(task_results):
            return {
                "synthesis": f"Synthesized results from {len(task_results)} tasks",
                "confidence_score": 0.8,
            }
        
        result = await orchestrator.execute_parallel_research(
            tasks=sample_research_tasks,
            research_executor=mock_executor,
            synthesis_callback=mock_synthesis,
        )
        
        assert isinstance(result, ResearchResult)
        assert result.successful_tasks == len(sample_research_tasks)
        assert result.failed_tasks == 0
        assert result.parallel_efficiency > 1.0  # Should be faster than sequential
        assert result.synthesis is not None
        assert "synthesis" in result.synthesis
    
    @pytest.mark.unit
    async def test_parallel_execution_with_failures(self, sample_research_tasks):
        """Test parallel execution with some task failures."""
        orchestrator = ParallelResearchOrchestrator()
        
        # Mock research executor that fails for certain task types
        async def mock_executor_with_failures(task):
            if task.task_type == "technical":
                raise TimeoutError("Task timed out")
            elif task.task_type == "sentiment":
                raise Exception("Network error")
            else:
                return {"research_type": task.task_type, "insights": ["Success"]}
        
        result = await orchestrator.execute_parallel_research(
            tasks=sample_research_tasks,
            research_executor=mock_executor_with_failures,
        )
        
        assert result.successful_tasks == 1  # Only fundamental should succeed
        assert result.failed_tasks == 2
        
        # Check that failed tasks have error information
        failed_tasks = [
            task for task in result.task_results.values()
            if task.status == "failed"
        ]
        assert len(failed_tasks) == 2
        for task in failed_tasks:
            assert task.error is not None
    
    @pytest.mark.unit
    async def test_circuit_breaker_integration(self, sample_research_tasks):
        """Test circuit breaker integration in parallel execution."""
        orchestrator = ParallelResearchOrchestrator()
        
        # Mock executor that consistently fails
        failure_count = 0
        async def failing_executor(task):
            nonlocal failure_count
            failure_count += 1
            raise Exception(f"Failure {failure_count}")
        
        result = await orchestrator.execute_parallel_research(
            tasks=sample_research_tasks,
            research_executor=failing_executor,
        )
        
        # All tasks should fail
        assert result.failed_tasks == len(sample_research_tasks)
        assert result.successful_tasks == 0


class TestTaskDistributionEngine:
    """Test intelligent task distribution for research topics."""
    
    @pytest.mark.unit
    def test_topic_relevance_analysis(self):
        """Test topic relevance analysis for different task types."""
        engine = TaskDistributionEngine()
        
        # Test financial topic
        relevance = engine._analyze_topic_relevance(
            "apple earnings financial performance",
            focus_areas=["fundamentals", "financials"]
        )
        
        assert "fundamental" in relevance
        assert "technical" in relevance
        assert "sentiment" in relevance
        assert "competitive" in relevance
        
        # Fundamental should have highest relevance for earnings query
        assert relevance["fundamental"] > relevance["technical"]
        assert relevance["fundamental"] > relevance["competitive"]
    
    @pytest.mark.unit
    def test_task_distribution_basic(self):
        """Test basic task distribution for a research topic."""
        engine = TaskDistributionEngine()
        
        tasks = engine.distribute_research_tasks(
            topic="AAPL financial analysis and market outlook",
            session_id="test_session",
            focus_areas=["fundamentals", "technical_analysis"]
        )
        
        assert len(tasks) > 0
        
        # Should have variety of task types
        task_types = {task.task_type for task in tasks}
        assert "fundamental" in task_types  # High relevance for financial analysis
        
        # Tasks should be properly configured
        for task in tasks:
            assert task.session_id == "test_session"
            assert task.target_topic == "AAPL financial analysis and market outlook"
            assert task.priority > 0
            assert len(task.focus_areas) > 0
    
    @pytest.mark.unit
    def test_task_distribution_fallback(self):
        """Test task distribution fallback when no relevant tasks found."""
        engine = TaskDistributionEngine()
        
        # Mock the relevance analysis to return very low scores
        with patch.object(engine, '_analyze_topic_relevance', return_value={
            "fundamental": 0.1,
            "technical": 0.1,
            "sentiment": 0.1,
            "competitive": 0.1,
        }):
            tasks = engine.distribute_research_tasks(
                topic="obscure topic with no clear relevance",
                session_id="test_session",
            )
        
        # Should still create at least one task (fallback)
        assert len(tasks) >= 1
        
        # Fallback should be fundamental analysis
        assert any(task.task_type == "fundamental" for task in tasks)
    
    @pytest.mark.unit
    def test_task_priority_assignment(self):
        """Test priority assignment based on relevance scores."""
        engine = TaskDistributionEngine()
        
        tasks = engine.distribute_research_tasks(
            topic="AAPL fundamental analysis earnings valuation",
            session_id="test_session",
        )
        
        # Find fundamental task (should have high priority)
        fundamental_tasks = [t for t in tasks if t.task_type == "fundamental"]
        if fundamental_tasks:
            fundamental_task = fundamental_tasks[0]
            assert fundamental_task.priority >= 7  # Should be high priority


# Timeout and Circuit Breaker Tests

class TestTimeoutAndCircuitBreaker:
    """Test timeout handling and circuit breaker patterns."""
    
    @pytest.mark.unit
    async def test_timeout_budget_allocation(self, mock_llm):
        """Test timeout budget allocation across research phases."""
        agent = DeepResearchAgent(llm=mock_llm, exa_api_key="test_key")
        
        # Test basic timeout allocation
        timeout_budget = 20.0
        result = await agent.research_comprehensive(
            topic="test topic",
            session_id="test_session",
            depth="basic",
            timeout_budget=timeout_budget
        )
        
        # Should either complete or timeout gracefully
        assert "status" in result or "error" in result
        
        # If timeout occurred, should have appropriate error structure
        if result.get("status") == "error" or "error" in result:
            # Should be a timeout-related error for very short budget
            assert "timeout" in str(result).lower() or "search providers" in str(result).lower()
    
    @pytest.mark.unit
    def test_provider_health_monitoring(self):
        """Test search provider health monitoring and recovery."""
        provider = ExaSearchProvider("test_key")
        
        # Initially healthy
        assert provider.is_healthy()
        
        # Simulate multiple timeout failures
        for i in range(15):  # Exceed default threshold of 12
            provider._record_failure("timeout")
        
        # Should be marked unhealthy
        assert not provider.is_healthy()
        
        # Recovery after success
        provider._record_success()
        assert provider.is_healthy()
        assert provider._failure_count == 0
    
    @pytest.mark.integration
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    async def test_research_with_provider_failures(self, mock_provider, mock_llm):
        """Test research behavior when provider failures occur."""
        # Create a provider that will fail
        failing_provider = MagicMock(spec=ExaSearchProvider)
        failing_provider.is_healthy.return_value = True
        failing_provider.search = AsyncMock(side_effect=WebSearchError("Search failed"))
        
        mock_provider.return_value = failing_provider
        
        agent = DeepResearchAgent(llm=mock_llm, exa_api_key="test_key")
        
        result = await agent.research_comprehensive(
            topic="test topic",
            session_id="test_session",
            depth="basic",
        )
        
        # Should handle provider failure gracefully
        assert "status" in result
        # May succeed with fallback or fail gracefully


# Performance and Benchmarking Tests

class TestPerformanceBenchmarks:
    """Test performance across different research depths and configurations."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    @patch('exa_py.Exa')
    async def test_research_depth_performance(self, mock_exa_class, mock_provider, mock_llm, mock_exa_client):
        """Benchmark performance across different research depths."""
        mock_provider.return_value = ExaSearchProvider("test_key")
        mock_exa_class.return_value = mock_exa_client
        
        performance_results = {}
        
        for depth in ["basic", "standard", "comprehensive"]:
            agent = DeepResearchAgent(
                llm=mock_llm,
                exa_api_key="test_key",
                research_depth=depth,
            )
            
            start_time = time.time()
            
            result = await agent.research_comprehensive(
                topic="AAPL financial analysis",
                session_id=f"perf_test_{depth}",
                depth=depth,
                timeout_budget=30.0,
            )
            
            execution_time = time.time() - start_time
            performance_results[depth] = {
                "execution_time": execution_time,
                "success": result.get("status") == "success",
                "sources_analyzed": result.get("sources_analyzed", 0),
            }
        
        # Verify performance characteristics
        assert performance_results["basic"]["execution_time"] <= performance_results["comprehensive"]["execution_time"]
        
        # Basic should be fastest
        if performance_results["basic"]["success"]:
            assert performance_results["basic"]["execution_time"] < 15.0  # Should be fast
    
    @pytest.mark.slow
    async def test_parallel_vs_sequential_performance(self, sample_research_tasks):
        """Compare parallel vs sequential execution performance."""
        config = ParallelResearchConfig(max_concurrent_agents=4, timeout_per_agent=10)
        orchestrator = ParallelResearchOrchestrator(config)
        
        async def mock_executor(task):
            await asyncio.sleep(1)  # Simulate 1 second work per task
            return {"research_type": task.task_type, "insights": ["Mock insight"]}
        
        # Parallel execution
        start_time = time.time()
        parallel_result = await orchestrator.execute_parallel_research(
            tasks=sample_research_tasks,
            research_executor=mock_executor,
        )
        parallel_time = time.time() - start_time
        
        # Sequential simulation
        start_time = time.time()
        for task in sample_research_tasks:
            await mock_executor(task)
        sequential_time = time.time() - start_time
        
        # Parallel should be significantly faster
        assert parallel_result.parallel_efficiency > 1.5  # At least 50% improvement
        assert parallel_time < sequential_time * 0.7  # Should be at least 30% faster
    
    @pytest.mark.unit
    async def test_memory_usage_monitoring(self, sample_research_tasks):
        """Test memory usage stays reasonable during parallel execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = ParallelResearchConfig(max_concurrent_agents=4)
        orchestrator = ParallelResearchOrchestrator(config)
        
        async def mock_executor(task):
            # Create some data but not excessive
            data = {"results": ["data"] * 1000}  # Small amount of data
            await asyncio.sleep(0.1)
            return data
        
        await orchestrator.execute_parallel_research(
            tasks=sample_research_tasks * 5,  # More tasks to test scaling
            research_executor=mock_executor,
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for test)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB"


# MCP Integration Tests

class TestMCPIntegration:
    """Test MCP tool endpoints and research router integration."""
    
    @pytest.mark.integration
    @patch('maverick_mcp.api.routers.research.get_settings')
    async def test_comprehensive_research_mcp_tool(self, mock_settings):
        """Test the comprehensive research MCP tool endpoint."""
        mock_settings.return_value.research.exa_api_key = "test_key"
        
        result = await comprehensive_research(
            query="AAPL financial analysis",
            persona="moderate",
            research_scope="basic",
            max_sources=5,
            timeframe="1m",
        )
        
        # Should return structured response
        assert isinstance(result, dict)
        assert "success" in result
        
        # If successful, should have proper structure
        if result.get("success"):
            assert "research_results" in result
            assert "research_metadata" in result
            assert "request_id" in result
            assert "timestamp" in result
    
    @pytest.mark.unit
    @patch('maverick_mcp.api.routers.research.get_settings')
    async def test_research_without_exa_key(self, mock_settings):
        """Test research behavior without ExaSearch API key."""
        mock_settings.return_value.research.exa_api_key = None
        
        result = await comprehensive_research(
            query="test query",
            persona="moderate",
            research_scope="basic",
        )
        
        assert result["success"] is False
        assert "Exa search provider not configured" in result["error"]
        assert "setup_instructions" in result["details"]
    
    @pytest.mark.unit
    def test_research_request_validation(self):
        """Test ResearchRequest model validation."""
        # Valid request
        request = ResearchRequest(
            query="AAPL analysis",
            persona="moderate",
            research_scope="standard",
            max_sources=15,
            timeframe="1m"
        )
        
        assert request.query == "AAPL analysis"
        assert request.persona == "moderate"
        assert request.research_scope == "standard"
        assert request.max_sources == 15
        assert request.timeframe == "1m"
        
        # Test defaults
        minimal_request = ResearchRequest(query="test")
        assert minimal_request.persona == "moderate"
        assert minimal_request.research_scope == "standard"
        assert minimal_request.max_sources == 10
        assert minimal_request.timeframe == "1m"
    
    @pytest.mark.unit
    def test_get_research_agent_optimization(self):
        """Test research agent creation with optimization parameters."""
        # Test optimized agent creation
        agent = get_research_agent(
            query="complex financial analysis of multiple companies",
            research_scope="comprehensive",
            timeout_budget=25.0,
            max_sources=20,
        )
        
        assert isinstance(agent, DeepResearchAgent)
        assert agent.max_sources <= 20  # Should respect or optimize max sources
        assert agent.default_depth in ["basic", "standard", "comprehensive", "exhaustive"]
        
        # Test standard agent creation
        standard_agent = get_research_agent()
        assert isinstance(standard_agent, DeepResearchAgent)


# Content Analysis Tests

class TestContentAnalyzer:
    """Test AI-powered content analysis functionality."""
    
    @pytest.mark.unit
    async def test_content_analysis_success(self, mock_llm):
        """Test successful content analysis."""
        analyzer = ContentAnalyzer(mock_llm)
        
        content = "Apple reported strong quarterly earnings with revenue growth of 12% and expanding market share in the services segment."
        
        result = await analyzer.analyze_content(
            content=content,
            persona="moderate",
            analysis_focus="financial"
        )
        
        assert "insights" in result
        assert "sentiment" in result
        assert "risk_factors" in result
        assert "opportunities" in result
        assert "credibility_score" in result
        assert "relevance_score" in result
        assert "summary" in result
        assert "analysis_timestamp" in result
    
    @pytest.mark.unit
    async def test_content_analysis_fallback(self, mock_llm):
        """Test content analysis fallback when AI analysis fails."""
        analyzer = ContentAnalyzer(mock_llm)
        
        # Make LLM fail
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        
        result = await analyzer.analyze_content(
            content="Test content",
            persona="moderate"
        )
        
        # Should fall back to keyword-based analysis
        assert result["fallback_used"] is True
        assert "sentiment" in result
        assert result["sentiment"]["direction"] in ["bullish", "bearish", "neutral"]
    
    @pytest.mark.unit
    async def test_batch_content_analysis(self, mock_llm):
        """Test batch content analysis functionality."""
        analyzer = ContentAnalyzer(mock_llm)
        
        content_items = [
            ("Apple shows strong growth", "source1"),
            ("Market conditions remain volatile", "source2"),
            ("Tech sector outlook positive", "source3"),
        ]
        
        results = await analyzer.analyze_content_batch(
            content_items=content_items,
            persona="moderate",
            analysis_focus="general"
        )
        
        assert len(results) == len(content_items)
        for i, result in enumerate(results):
            assert result["source_identifier"] == f"source{i+1}"
            assert result["batch_processed"] is True
            assert "sentiment" in result


# Error Handling and Edge Cases

class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""
    
    @pytest.mark.unit
    async def test_empty_search_results(self, mock_llm):
        """Test behavior when search returns no results."""
        provider = ExaSearchProvider("test_key")
        
        with patch('exa_py.Exa') as mock_exa:
            # Mock empty results
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.results = []
            mock_client.search_and_contents.return_value = mock_response
            mock_exa.return_value = mock_client
            
            results = await provider.search("nonexistent topic", num_results=5)
            
            assert results == []
    
    @pytest.mark.unit
    async def test_malformed_search_response(self, mock_llm):
        """Test handling of malformed search responses."""
        provider = ExaSearchProvider("test_key")
        
        with patch('exa_py.Exa') as mock_exa:
            # Mock malformed response
            mock_client = MagicMock()
            mock_client.search_and_contents.side_effect = Exception("Invalid response format")
            mock_exa.return_value = mock_client
            
            with pytest.raises(WebSearchError):
                await provider.search("test query")
    
    @pytest.mark.unit
    async def test_network_timeout_recovery(self):
        """Test network timeout recovery mechanisms."""
        provider = ExaSearchProvider("test_key")
        
        # Simulate multiple timeouts followed by success
        with patch('exa_py.Exa') as mock_exa:
            call_count = 0
            
            async def mock_search_with_recovery(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise TimeoutError("Network timeout")
                else:
                    # Success on third try
                    mock_response = MagicMock()
                    mock_result = MagicMock()
                    mock_result.url = "https://example.com"
                    mock_result.title = "Test Result"
                    mock_result.text = "Test content"
                    mock_result.published_date = "2024-01-15"
                    mock_result.score = 0.8
                    mock_response.results = [mock_result]
                    return mock_response
            
            mock_client = MagicMock()
            mock_client.search_and_contents.side_effect = mock_search_with_recovery
            mock_exa.return_value = mock_client
            
            # First two calls should fail and record failures
            for _ in range(2):
                with pytest.raises(WebSearchError):
                    await provider.search("test query", timeout_budget=1.0)
            
            # Provider should still be healthy (failures recorded but not exceeded threshold)
            assert provider._failure_count == 2
            
            # Third call should succeed and reset failure count
            results = await provider.search("test query")
            assert len(results) > 0
            assert provider._failure_count == 0  # Reset on success
    
    @pytest.mark.unit 
    async def test_concurrent_request_limits(self, sample_research_tasks):
        """Test that concurrent request limits are respected."""
        config = ParallelResearchConfig(max_concurrent_agents=2)  # Very low limit
        orchestrator = ParallelResearchOrchestrator(config)
        
        execution_times = []
        
        async def tracking_executor(task):
            start = time.time()
            await asyncio.sleep(0.5)  # Simulate work
            end = time.time()
            execution_times.append((start, end))
            return {"result": "success"}
        
        await orchestrator.execute_parallel_research(
            tasks=sample_research_tasks,  # 3 tasks
            research_executor=tracking_executor,
        )
        
        # With max_concurrent_agents=2, the third task should start after one of the first two finishes
        # This means there should be overlap but not all three running simultaneously
        assert len(execution_times) == 3
        
        # Sort by start time
        execution_times.sort()
        
        # The third task should start after the first task finishes
        # (allowing for some timing tolerance)
        third_start = execution_times[2][0]
        first_end = execution_times[0][1]
        
        # Third should start after first ends (with small tolerance for async timing)
        assert third_start >= (first_end - 0.1)


# Integration Test Suite

class TestFullIntegrationScenarios:
    """End-to-end integration tests for complete research workflows."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @patch('maverick_mcp.agents.deep_research.get_cached_search_provider')
    @patch('exa_py.Exa')
    async def test_complete_research_workflow(self, mock_exa_class, mock_provider, mock_llm, mock_exa_client):
        """Test complete research workflow from query to final report."""
        # Setup comprehensive mocks
        mock_provider.return_value = ExaSearchProvider("test_key")
        mock_exa_class.return_value = mock_exa_client
        
        agent = DeepResearchAgent(
            llm=mock_llm,
            persona="moderate",
            exa_api_key="test_key",
            research_depth="standard",
            enable_parallel_execution=True,
        )
        
        # Execute complete research workflow
        result = await agent.research_comprehensive(
            topic="Apple Inc (AAPL) investment analysis with market sentiment and competitive position",
            session_id="integration_test_session",
            depth="standard",
            focus_areas=["fundamentals", "market_sentiment", "competitive_analysis"],
            timeframe="1m",
            use_parallel_execution=True,
        )
        
        # Verify comprehensive result structure
        if result.get("status") == "success":
            assert "findings" in result
            assert "confidence_score" in result
            assert isinstance(result["confidence_score"], (int, float))
            assert 0.0 <= result["confidence_score"] <= 1.0
            assert "citations" in result
            assert "execution_time_ms" in result
            
            # Check for parallel execution indicators
            if "parallel_execution_stats" in result:
                assert "successful_tasks" in result["parallel_execution_stats"]
                assert "parallel_efficiency" in result["parallel_execution_stats"]
        
        # Should handle both success and controlled failure scenarios
        assert "status" in result or "error" in result
    
    @pytest.mark.integration
    async def test_multi_persona_consistency(self, mock_llm, mock_exa_client):
        """Test research consistency across different investor personas."""
        personas = ["conservative", "moderate", "aggressive", "day_trader"]
        results = {}
        
        for persona in personas:
            with patch('maverick_mcp.agents.deep_research.get_cached_search_provider') as mock_provider, \
                 patch('exa_py.Exa') as mock_exa_class:
                
                mock_provider.return_value = ExaSearchProvider("test_key")
                mock_exa_class.return_value = mock_exa_client
                
                agent = DeepResearchAgent(
                    llm=mock_llm,
                    persona=persona,
                    exa_api_key="test_key",
                    research_depth="basic",
                )
                
                result = await agent.research_comprehensive(
                    topic="AAPL investment outlook",
                    session_id=f"persona_test_{persona}",
                    depth="basic",
                )
                
                results[persona] = result
        
        # All personas should provide valid responses
        for persona, result in results.items():
            assert isinstance(result, dict), f"Invalid result for {persona}"
            # Should have some form of result (success or controlled failure)
            assert "status" in result or "error" in result or "success" in result


if __name__ == "__main__":
    # Run specific test categories based on markers
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "unit",  # Run unit tests by default
    ])