"""
DeepResearchAgent Integration Example

This example demonstrates how to use the DeepResearchAgent with the SupervisorAgent
for comprehensive financial research capabilities.
"""

import asyncio
import logging

from maverick_mcp.agents.deep_research import DeepResearchAgent
from maverick_mcp.agents.market_analysis import MarketAnalysisAgent
from maverick_mcp.agents.supervisor import SupervisorAgent
from maverick_mcp.agents.technical_analysis import TechnicalAnalysisAgent
from maverick_mcp.providers.llm_factory import get_llm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_standalone_research():
    """Example of using DeepResearchAgent standalone."""

    print("🔍 DeepResearchAgent Standalone Example")
    print("=" * 50)

    # Initialize LLM and agent
    llm = get_llm()
    research_agent = DeepResearchAgent(
        llm=llm,
        persona="moderate",  # Conservative, Moderate, Aggressive, Day Trader
        max_sources=30,
        research_depth="comprehensive",
    )

    # Example 1: Company Research
    print("\n📊 Example 1: Comprehensive Company Research")
    print("-" * 40)

    try:
        result = await research_agent.research_company_comprehensive(
            symbol="AAPL",
            session_id="company_research_demo",
            include_competitive_analysis=True,
        )

        print("✅ Research completed for AAPL")
        print(f"📈 Confidence Score: {result.get('research_confidence', 0):.2f}")
        print(f"📰 Sources Analyzed: {result.get('sources_found', 0)}")

        if "persona_insights" in result:
            insights = result["persona_insights"]
            print(
                f"🎯 Persona Insights: {len(insights.get('prioritized_insights', []))} relevant insights"
            )
            print(
                f"⚠️ Risk Assessment: {insights.get('risk_assessment', {}).get('risk_acceptable', 'Unknown')}"
            )
            print(
                f"💡 Recommended Action: {insights.get('recommended_action', 'No recommendation')}"
            )

    except Exception as e:
        print(f"❌ Error in company research: {e}")

    # Example 2: Market Sentiment Analysis
    print("\n📈 Example 2: Market Sentiment Analysis")
    print("-" * 40)

    try:
        result = await research_agent.analyze_market_sentiment(
            topic="artificial intelligence stocks",
            session_id="sentiment_analysis_demo",
            timeframe="1w",
        )

        print("✅ Sentiment analysis completed")

        if "content_analysis" in result:
            analysis = result["content_analysis"]
            consensus = analysis.get("consensus_view", {})
            themes = analysis.get("key_themes", [])

            print(
                f"📊 Overall Sentiment: {consensus.get('direction', 'neutral').title()}"
            )
            print(f"🔒 Confidence: {consensus.get('confidence', 0):.2f}")
            print(f"🔑 Key Themes: {len(themes)} themes identified")

            if themes:
                for i, theme in enumerate(themes[:3], 1):
                    print(
                        f"   {i}. {theme.get('theme', 'Unknown')} (relevance: {theme.get('relevance', 0):.2f})"
                    )

    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")

    # Example 3: Custom Research Query
    print("\n🔍 Example 3: Custom Research Query")
    print("-" * 40)

    try:
        result = await research_agent.research_topic(
            query="impact of Federal Reserve interest rate decisions on tech stocks",
            session_id="custom_research_demo",
            research_scope="comprehensive",
            max_sources=25,
            timeframe="1m",
        )

        print("✅ Custom research completed")
        print(f"📊 Research Confidence: {result.get('research_confidence', 0):.2f}")

        if "content_analysis" in result:
            analysis = result["content_analysis"]
            insights = analysis.get("insights", [])
            print(f"💡 Insights Generated: {len(insights)}")

            # Show top 3 insights
            for i, insight in enumerate(insights[:3], 1):
                insight_text = insight.get("insight", "No insight text")[:100] + "..."
                confidence = insight.get("confidence", 0)
                print(f"   {i}. {insight_text} (confidence: {confidence:.2f})")

    except Exception as e:
        print(f"❌ Error in custom research: {e}")


async def example_supervisor_integration():
    """Example of using DeepResearchAgent with SupervisorAgent."""

    print("\n🎛️ SupervisorAgent Integration Example")
    print("=" * 50)

    # Initialize LLM
    llm = get_llm()

    # Create specialized agents
    market_agent = MarketAnalysisAgent(llm=llm, persona="moderate")
    technical_agent = TechnicalAnalysisAgent(llm=llm, persona="moderate")
    research_agent = DeepResearchAgent(llm=llm, persona="moderate")

    # Create supervisor with all agents
    supervisor = SupervisorAgent(
        llm=llm,
        agents={
            "market": market_agent,
            "technical": technical_agent,
            "research": research_agent,  # Key integration point
        },
        persona="moderate",
        routing_strategy="llm_powered",
        synthesis_mode="weighted",
    )

    # Example coordination scenarios
    test_queries = [
        {
            "query": "Should I invest in MSFT? I want comprehensive analysis including recent news and competitive position",
            "expected_routing": ["technical", "research"],
            "description": "Investment decision requiring technical + research",
        },
        {
            "query": "What's the current market sentiment on renewable energy stocks?",
            "expected_routing": ["research"],
            "description": "Pure sentiment analysis research",
        },
        {
            "query": "Find me high-momentum stocks with strong fundamentals",
            "expected_routing": ["market", "research"],
            "description": "Screening + fundamental research",
        },
    ]

    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📋 Test Case {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected Routing: {test_case['expected_routing']}")
        print("-" * 60)

        try:
            result = await supervisor.coordinate_agents(
                query=test_case["query"], session_id=f"supervisor_demo_{i}"
            )

            if result.get("status") == "success":
                agents_used = result.get("agents_used", [])
                confidence = result.get("confidence_score", 0)
                execution_time = result.get("execution_time_ms", 0)
                conflicts_resolved = result.get("conflicts_resolved", 0)

                print("✅ Coordination successful")
                print(f"🤖 Agents Used: {agents_used}")
                print(f"📊 Confidence Score: {confidence:.2f}")
                print(f"⏱️ Execution Time: {execution_time:.0f}ms")
                print(f"🔧 Conflicts Resolved: {conflicts_resolved}")

                # Show synthesis result
                synthesis = (
                    result.get("synthesis", "No synthesis available")[:200] + "..."
                )
                print(f"📝 Synthesis Preview: {synthesis}")

            else:
                print(f"❌ Coordination failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error in coordination: {e}")


async def example_persona_adaptation():
    """Example showing how research adapts to different investor personas."""

    print("\n👥 Persona Adaptation Example")
    print("=" * 50)

    llm = get_llm()
    personas = ["conservative", "moderate", "aggressive", "day_trader"]
    query = "Should I invest in Tesla (TSLA)?"

    for persona in personas:
        print(f"\n🎭 Persona: {persona.title()}")
        print("-" * 30)

        try:
            research_agent = DeepResearchAgent(
                llm=llm,
                persona=persona,
                max_sources=20,  # Smaller sample for demo
                research_depth="standard",
            )

            result = await research_agent.research_topic(
                query=query,
                session_id=f"persona_demo_{persona}",
                research_scope="standard",
                timeframe="2w",
            )

            if "persona_insights" in result:
                insights = result["persona_insights"]
                risk_assessment = insights.get("risk_assessment", {})
                action = insights.get("recommended_action", "No action")
                alignment = insights.get("persona_alignment_score", 0)

                print(f"📊 Persona Alignment: {alignment:.2f}")
                print(
                    f"⚠️ Risk Acceptable: {risk_assessment.get('risk_acceptable', 'Unknown')}"
                )
                print(f"💡 Recommended Action: {action}")

                # Show risk factors for conservative investors
                if persona == "conservative" and risk_assessment.get("risk_factors"):
                    print(f"🚨 Risk Factors ({len(risk_assessment['risk_factors'])}):")
                    for factor in risk_assessment["risk_factors"][:2]:
                        print(f"   • {factor[:80]}...")

            else:
                print("⚠️ No persona insights available")

        except Exception as e:
            print(f"❌ Error for {persona}: {e}")


async def example_research_tools_mcp():
    """Example showing MCP tool integration."""

    print("\n🔧 MCP Tools Integration Example")
    print("=" * 50)

    # Note: This is a conceptual example - actual MCP tool usage would be through Claude Desktop
    print("📚 Available Research Tools:")
    print("1. comprehensive_research - Deep research on any financial topic")
    print("2. analyze_market_sentiment - Market sentiment analysis")
    print("3. research_company_comprehensive - Company fundamental analysis")
    print("4. search_financial_news - News search and analysis")
    print("5. validate_research_claims - Fact-checking and validation")

    # Example tool configurations for Claude Desktop
    print("\n📋 Claude Desktop Configuration Example:")
    print("```json")
    print("{")
    print('  "mcpServers": {')
    print('    "maverick-research": {')
    print('      "command": "npx",')
    print('      "args": ["-y", "mcp-remote", "http://localhost:8000/research"]')
    print("    }")
    print("  }")
    print("}")
    print("```")

    print("\n💬 Example Claude Desktop Prompts:")
    examples = [
        "Research Tesla's competitive position in the EV market with comprehensive analysis",
        "Analyze current market sentiment for renewable energy stocks over the past week",
        "Perform fundamental analysis of Apple (AAPL) including business model and growth prospects",
        "Search for recent financial news about Federal Reserve policy changes",
        "Validate the claim that 'AI stocks outperformed the market by 20% this quarter'",
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")


async def main():
    """Run all examples."""

    print("🚀 DeepResearchAgent Comprehensive Examples")
    print("=" * 60)
    print("This demo showcases the DeepResearchAgent capabilities")
    print("including standalone usage, SupervisorAgent integration,")
    print("persona adaptation, and MCP tool integration.")
    print("=" * 60)

    try:
        # Run examples
        await example_standalone_research()
        await example_supervisor_integration()
        await example_persona_adaptation()
        await example_research_tools_mcp()

        print("\n✅ All examples completed successfully!")
        print("\n📖 Next Steps:")
        print("1. Set up EXA_API_KEY and TAVILY_API_KEY environment variables")
        print("2. Configure Claude Desktop with the research MCP server")
        print("3. Test with real queries through Claude Desktop")
        print("4. Customize personas and research parameters as needed")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo execution failed")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
