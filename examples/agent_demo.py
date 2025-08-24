"""
Demo script showing how to use the LangGraph agents.
"""

import asyncio
import os
from datetime import datetime

# For local testing without full LLM
from langchain_core.language_models import FakeListLLM
from langchain_openai import ChatOpenAI

from maverick_mcp.agents.market_analysis import MarketAnalysisAgent


async def demo_market_analysis():
    """Demonstrate market analysis agent."""

    # Use OpenAI if API key is available, otherwise use fake LLM
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        print("Using OpenAI GPT-4")
    else:
        # Fake LLM for testing without API keys
        llm: ChatOpenAI = FakeListLLM(  # type: ignore
            responses=[
                "Based on market screening, I found 3 momentum stocks: NVDA, AAPL, and MSFT.",
                "The market breadth is positive with advancing stocks outnumbering declining ones.",
                "Sector rotation shows strength in technology and healthcare.",
            ]
        )
        print("Using FakeListLLM for testing")

    # Create agents with different personas
    personas = ["conservative", "moderate", "aggressive"]

    for persona in personas:
        print(f"\n{'=' * 60}")
        print(f"Testing {persona.upper()} investor persona")
        print(f"{'=' * 60}")

        # Create agent
        agent = MarketAnalysisAgent(llm=llm, persona=persona, ttl_hours=1)

        # Test queries
        queries = [
            "Find me the top momentum stocks for today",
            "What sectors are showing strength?",
            "Screen for stocks suitable for my risk profile",
        ]

        session_id = f"demo_{persona}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for query in queries:
            print(f"\nQuery: {query}")

            try:
                result = await agent.ainvoke(query, session_id)

                if result.get("status") == "success":
                    print(f"Summary: {result['results']['summary']}")

                    if result["results"].get("screened_stocks"):
                        print(f"Found {result['results']['stock_count']} stocks")
                        print("Top picks:", result["results"]["screened_stocks"][:3])
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"Exception: {str(e)}")

        print(f"\nRisk parameters for {persona}:")
        print(f"- Risk tolerance: {agent.persona.risk_tolerance}")
        print(f"- Max position size: {agent.persona.position_size_max * 100:.1f}%")
        print(f"- Stop loss multiplier: {agent.persona.stop_loss_multiplier}")


async def demo_tool_registry():
    """Demonstrate tool registry functionality."""
    from maverick_mcp.langchain_tools import get_tool_registry

    print("\n" + "=" * 60)
    print("TOOL REGISTRY DEMO")
    print("=" * 60)

    registry = get_tool_registry()

    # List all tools
    print("\nRegistered tools:")
    for name, info in registry.list_tools().items():
        print(f"- {name}: {info['description'][:60]}...")
        print(f"  Source: {info['source']}, Persona-aware: {info['persona_aware']}")

    # Get tools by category
    screening_tools = registry.get_tools_by_category("screening")
    print(f"\nScreening tools: {len(screening_tools)}")

    # Get persona-aware tools
    persona_tools = registry.get_persona_aware_tools()
    print(f"Persona-aware tools: {len(persona_tools)}")


async def demo_memory_ttl():
    """Demonstrate memory TTL functionality."""
    from maverick_mcp.memory import ConversationStore, MemorySaver

    print("\n" + "=" * 60)
    print("MEMORY TTL DEMO")
    print("=" * 60)

    # Test checkpointer TTL
    MemorySaver()  # Using basic MemorySaver for demo

    # Note: MemorySaver is used internally by LangGraph agents
    # This is just a demonstration of memory concepts
    print("\nMemorySaver is used internally by agents for state management")
    print("Memory TTL and cleanup would be handled by the agent infrastructure")

    # Test conversation store
    store = ConversationStore(ttl_hours=1)
    store.save_analysis(
        session_id="test_session",
        symbol="AAPL",
        analysis_type="technical",
        data={"price": 150.0, "trend": "bullish"},
    )

    # Retrieve analysis
    analysis = store.get_analysis("test_session", "AAPL", "technical")
    print(f"\nStored analysis: {analysis}")


async def main():
    """Run all demos."""
    print("MAVERICK-MCP LANGGRAPH AGENT DEMO")
    print("=" * 80)

    # Run demos
    await demo_tool_registry()
    await demo_memory_ttl()
    await demo_market_analysis()

    print("\n" + "=" * 80)
    print("Demo completed!")


if __name__ == "__main__":
    # Create examples directory
    os.makedirs("examples", exist_ok=True)

    # Run async main
    asyncio.run(main())
