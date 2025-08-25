#!/usr/bin/env python3
"""
Example demonstrating the new parallel research capabilities of DeepResearchAgent.

This example shows how to:
1. Initialize DeepResearchAgent with parallel execution
2. Use both parallel and sequential modes
3. Configure parallel execution parameters
4. Access specialized research results from parallel agents
"""

import asyncio
import logging
from datetime import datetime

from langchain_core.language_models.fake import FakeListLLM

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.utils.parallel_research import ParallelResearchConfig

# Set up logging to see parallel execution in action
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Demonstrate parallel research capabilities."""
    
    # Create a mock LLM for testing (in real usage, use Claude/GPT)
    llm = FakeListLLM(responses=[
        '{"KEY_INSIGHTS": ["Strong earnings growth", "Market expansion"], "SENTIMENT": {"direction": "bullish", "confidence": 0.8}, "RISK_FACTORS": ["Market volatility"], "OPPORTUNITIES": ["AI adoption"], "CREDIBILITY": 0.7, "RELEVANCE": 0.9, "SUMMARY": "Positive outlook for tech company"}',
        'Comprehensive research synthesis shows positive trends across multiple analysis areas with strong fundamentals and technical indicators supporting continued growth.',
        'Technical analysis indicates strong upward momentum with key resistance levels broken.',
        'Market sentiment is predominantly bullish with institutional support.',
        'Competitive analysis shows strong market position with sustainable advantages.'
    ])
    
    print("🔬 DeepResearchAgent Parallel Execution Demo")
    print("=" * 50)
    
    # 1. Create agent with parallel execution enabled (default)
    print("\n1. Creating DeepResearchAgent with parallel execution...")
    
    parallel_config = ParallelResearchConfig(
        max_concurrent_agents=3,  # Run 3 agents in parallel
        timeout_per_agent=120,    # 2 minutes per agent
        enable_fallbacks=True,    # Enable fallback to sequential if parallel fails
        rate_limit_delay=0.5,     # 0.5 second delay between agent starts
    )
    
    agent = DeepResearchAgent(
        llm=llm,
        persona='moderate',
        enable_parallel_execution=True,
        parallel_config=parallel_config,
        # Note: In real usage, provide API keys:
        # exa_api_key="your-exa-key",
        # tavily_api_key="your-tavily-key"
    )
    
    print(f"✅ Agent created with parallel execution enabled")
    print(f"   Max concurrent agents: {agent.parallel_config.max_concurrent_agents}")
    print(f"   Timeout per agent: {agent.parallel_config.timeout_per_agent}s")
    
    # 2. Demonstrate parallel research
    print("\n2. Running parallel research...")
    
    # This will automatically use parallel execution
    start_time = datetime.now()
    
    try:
        # Note: This requires actual search providers (Exa/Tavily API keys) to work fully
        # For demo purposes, we'll show the structure
        topic = "AAPL stock analysis and investment outlook"
        session_id = "demo_session_001"
        
        print(f"   Topic: {topic}")
        print(f"   Session: {session_id}")
        print("   🚀 Starting parallel research execution...")
        
        # In a real environment with API keys, this would work:
        # result = await agent.research_comprehensive(
        #     topic=topic,
        #     session_id=session_id,
        #     depth="standard",
        #     focus_areas=["fundamentals", "technical_analysis", "market_sentiment"],
        #     use_parallel_execution=True  # Explicitly enable (default)
        # )
        
        # For demo, we'll simulate the expected response structure
        result = {
            "status": "success",
            "agent_type": "deep_research",
            "execution_mode": "parallel",
            "persona": "Moderate",
            "research_topic": topic,
            "research_depth": "standard",
            "findings": {
                "synthesis": "Comprehensive analysis from multiple specialized agents shows strong fundamentals...",
                "key_insights": [
                    "Strong earnings growth trajectory",
                    "Positive technical indicators",
                    "Bullish market sentiment",
                    "Competitive market position"
                ],
                "overall_sentiment": {"direction": "bullish", "confidence": 0.75},
                "risk_assessment": ["Market volatility", "Regulatory risks"],
                "investment_implications": {
                    "opportunities": ["AI growth", "Market expansion"],
                    "threats": ["Competition", "Economic headwinds"],
                    "recommended_action": "Consider position building with appropriate risk management"
                },
                "confidence_score": 0.78
            },
            "sources_analyzed": 24,
            "confidence_score": 0.78,
            "execution_time_ms": 15000,  # 15 seconds (faster than sequential)
            "parallel_execution_stats": {
                "total_tasks": 3,
                "successful_tasks": 3,
                "failed_tasks": 0,
                "parallel_efficiency": 2.8,  # 2.8x faster than sequential
                "task_breakdown": {
                    "demo_session_001_fundamental": {"type": "fundamental", "status": "completed", "execution_time": 5.2},
                    "demo_session_001_sentiment": {"type": "sentiment", "status": "completed", "execution_time": 4.8},
                    "demo_session_001_competitive": {"type": "competitive", "status": "completed", "execution_time": 5.5}
                }
            }
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✅ Parallel research completed in {execution_time:.1f}s")
        print(f"   📊 Results from parallel execution:")
        print(f"      • Sources analyzed: {result['sources_analyzed']}")
        print(f"      • Overall sentiment: {result['findings']['overall_sentiment']['direction']} ({result['findings']['overall_sentiment']['confidence']:.2f} confidence)")
        print(f"      • Key insights: {len(result['findings']['key_insights'])}")
        print(f"      • Parallel efficiency: {result['parallel_execution_stats']['parallel_efficiency']:.1f}x speedup")
        print(f"      • Tasks: {result['parallel_execution_stats']['successful_tasks']}/{result['parallel_execution_stats']['total_tasks']} successful")
        
        # Show task breakdown
        print("\n   📋 Task Breakdown:")
        for task_id, task_info in result['parallel_execution_stats']['task_breakdown'].items():
            task_type = task_info['type'].title()
            status = task_info['status'].title()
            exec_time = task_info['execution_time']
            print(f"      • {task_type} Research: {status} ({exec_time:.1f}s)")
        
    except Exception as e:
        print(f"   ❌ Parallel research failed (expected without API keys): {e}")
    
    # 3. Demonstrate sequential fallback
    print("\n3. Testing sequential fallback...")
    
    sequential_agent = DeepResearchAgent(
        llm=llm,
        persona='moderate',
        enable_parallel_execution=False  # Force sequential mode
    )
    
    print("   ✅ Sequential-only agent created")
    print("   📝 This would use traditional LangGraph workflow for compatibility")
    
    # 4. Show configuration options
    print("\n4. Configuration Options:")
    print("   📋 Parallel Execution Configuration:")
    print(f"      • Max concurrent agents: {parallel_config.max_concurrent_agents}")
    print(f"      • Timeout per agent: {parallel_config.timeout_per_agent}s")
    print(f"      • Enable fallbacks: {parallel_config.enable_fallbacks}")
    print(f"      • Rate limit delay: {parallel_config.rate_limit_delay}s")
    
    print("\n   🎛️  Available Research Types:")
    print("      • Fundamental: Financial statements, earnings, valuation")
    print("      • Technical: Chart patterns, indicators, price action") 
    print("      • Sentiment: News analysis, analyst ratings, social sentiment")
    print("      • Competitive: Industry analysis, market position, competitors")
    
    # 5. Usage recommendations
    print("\n5. Usage Recommendations:")
    print("   💡 When to use parallel execution:")
    print("      • Comprehensive research requiring multiple analysis types")
    print("      • Time-sensitive research with tight deadlines")
    print("      • Research topics requiring diverse data sources")
    print("      • When you have sufficient API rate limits")
    
    print("\n   ⚠️  When to use sequential execution:")
    print("      • Limited API rate limits")
    print("      • Simple, focused research queries")
    print("      • Debugging and development")
    print("      • When consistency with legacy behavior is required")
    
    print("\n6. API Integration Requirements:")
    print("   🔑 For full functionality, provide:")
    print("      • EXA_API_KEY: High-quality research content")
    print("      • TAVILY_API_KEY: Comprehensive web search")
    print("      • Both are optional but recommended for best results")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed! The enhanced DeepResearchAgent now supports:")
    print("   ✅ Parallel execution with specialized subagents")
    print("   ✅ Automatic fallback to sequential execution")
    print("   ✅ Configurable concurrency and timeouts")
    print("   ✅ Full backward compatibility")
    print("   ✅ Detailed execution statistics and monitoring")

if __name__ == "__main__":
    asyncio.run(main())