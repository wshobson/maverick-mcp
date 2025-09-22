#!/usr/bin/env python3
"""
Test script for enhanced financial search capabilities in DeepResearchAgent.

This script demonstrates the improved Exa client usage for financial records search
with different strategies and optimizations.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maverick_mcp.agents.deep_research import DeepResearchAgent, ExaSearchProvider


async def test_financial_search_strategies():
    """Test different financial search strategies."""

    # Initialize the search provider
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        print("❌ EXA_API_KEY environment variable not set")
        return

    print("🔍 Testing Enhanced Financial Search Capabilities")
    print("=" * 60)

    # Test queries for different financial scenarios
    test_queries = [
        ("AAPL financial performance", "Apple stock analysis"),
        ("Tesla quarterly earnings 2024", "Tesla earnings report"),
        ("Microsoft revenue growth", "Microsoft financial growth"),
        ("S&P 500 market analysis", "Market index analysis"),
        ("Federal Reserve interest rates", "Fed policy analysis"),
    ]

    # Test different search strategies
    strategies = ["hybrid", "authoritative", "comprehensive", "auto"]

    provider = ExaSearchProvider(exa_api_key)

    for query, description in test_queries:
        print(f"\n📊 Testing Query: {description}")
        print(f"   Query: '{query}'")
        print("-" * 40)

        for strategy in strategies:
            try:
                start_time = datetime.now()

                # Test the enhanced financial search
                results = await provider.search_financial(
                    query=query, num_results=5, strategy=strategy
                )

                duration = (datetime.now() - start_time).total_seconds()

                print(f"  🎯 Strategy: {strategy.upper()}")
                print(f"     Results: {len(results)}")
                print(f"     Duration: {duration:.2f}s")

                if results:
                    # Show top result with enhanced metadata
                    top_result = results[0]
                    print("     Top Result:")
                    print(f"       Title: {top_result.get('title', 'N/A')[:80]}...")
                    print(f"       Domain: {top_result.get('domain', 'N/A')}")
                    print(
                        f"       Financial Relevance: {top_result.get('financial_relevance', 0):.2f}"
                    )
                    print(
                        f"       Authoritative: {top_result.get('is_authoritative', False)}"
                    )
                    print(f"       Score: {top_result.get('score', 0):.2f}")

                print()

            except Exception as e:
                print(f"  ❌ Strategy {strategy} failed: {str(e)}")
                print()


async def test_query_enhancement():
    """Test the financial query enhancement feature."""

    print("\n🔧 Testing Query Enhancement")
    print("=" * 40)

    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        print("❌ EXA_API_KEY environment variable not set")
        return

    provider = ExaSearchProvider(exa_api_key)

    # Test queries that should be enhanced
    test_queries = [
        "AAPL",  # Stock symbol
        "Tesla company",  # Company name
        "Microsoft analysis",  # Analysis request
        "Amazon earnings financial",  # Already has financial context
    ]

    for query in test_queries:
        enhanced = provider._enhance_financial_query(query)
        print(f"Original: '{query}'")
        print(f"Enhanced: '{enhanced}'")
        print(f"Changed: {'Yes' if enhanced != query else 'No'}")
        print()


async def test_financial_relevance_scoring():
    """Test the financial relevance scoring system."""

    print("\n📈 Testing Financial Relevance Scoring")
    print("=" * 45)

    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        print("❌ EXA_API_KEY environment variable not set")
        return

    provider = ExaSearchProvider(exa_api_key)

    # Mock result objects for testing
    class MockResult:
        def __init__(self, url, title, text, published_date=None):
            self.url = url
            self.title = title
            self.text = text
            self.published_date = published_date

    test_results = [
        MockResult(
            "https://sec.gov/filing/aapl-10k-2024",
            "Apple Inc. Annual Report (Form 10-K)",
            "Apple Inc. reported quarterly earnings of $1.50 per share, with revenue of $95 billion for the quarter ending March 31, 2024.",
            "2024-01-15T00:00:00Z",
        ),
        MockResult(
            "https://bloomberg.com/news/apple-stock-analysis",
            "Apple Stock Analysis: Strong Financial Performance",
            "Apple's financial performance continues to show strong growth with increased market cap and dividend distributions.",
            "2024-01-10T00:00:00Z",
        ),
        MockResult(
            "https://example.com/random-article",
            "Random Article About Technology",
            "This is just a random article about technology trends without specific financial information.",
            "2024-01-01T00:00:00Z",
        ),
    ]

    for i, result in enumerate(test_results, 1):
        relevance = provider._calculate_financial_relevance(result)
        is_auth = provider._is_authoritative_source(result.url)
        domain = provider._extract_domain(result.url)

        print(f"Result {i}:")
        print(f"  URL: {result.url}")
        print(f"  Domain: {domain}")
        print(f"  Title: {result.title}")
        print(f"  Financial Relevance: {relevance:.2f}")
        print(f"  Authoritative: {is_auth}")
        print()


async def test_deep_research_agent_integration():
    """Test the integration with DeepResearchAgent."""

    print("\n🤖 Testing DeepResearchAgent Integration")
    print("=" * 45)

    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        print("❌ EXA_API_KEY environment variable not set")
        return

    try:
        # Initialize the agent
        agent = DeepResearchAgent(
            llm=None,  # Will be set by initialize if needed
            persona="financial_analyst",
            exa_api_key=exa_api_key,
        )

        await agent.initialize()

        # Test the enhanced financial search tool
        result = await agent._perform_financial_search(
            query="Apple quarterly earnings Q4 2024",
            num_results=3,
            provider="exa",
            strategy="authoritative",
        )

        print(f"Search Results: {result.get('total_results', 0)} found")
        print(f"Strategy Used: {result.get('search_strategy', 'N/A')}")
        print(f"Duration: {result.get('search_duration', 0):.2f}s")
        print(f"Enhanced Search: {result.get('enhanced_search', False)}")

        if result.get("results"):
            print("\nTop Result:")
            top = result["results"][0]
            print(f"  Title: {top.get('title', 'N/A')[:80]}...")
            print(f"  Financial Relevance: {top.get('financial_relevance', 0):.2f}")
            print(f"  Authoritative: {top.get('is_authoritative', False)}")

    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")


async def main():
    """Run all tests."""

    print("🚀 Enhanced Financial Search Testing Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        await test_financial_search_strategies()
        await test_query_enhancement()
        await test_financial_relevance_scoring()
        await test_deep_research_agent_integration()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback

        traceback.print_exc()

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
