#!/usr/bin/env python3
"""
Simple test script to verify the orchestration tools are working correctly.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maverick_mcp.api.routers.agents import (
    deep_research_financial,
    list_available_agents,
    orchestrated_analysis,
)


async def test_list_available_agents():
    """Test the list_available_agents function."""
    print("🧪 Testing list_available_agents...")
    try:
        result = list_available_agents()
        print(f"✅ Success: {result['status']}")
        print(f"📊 Available agents: {list(result['agents'].keys())}")
        print(f"🎭 Available personas: {result['personas']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_orchestrated_analysis():
    """Test the orchestrated_analysis function with a simple query."""
    print("\n🧪 Testing orchestrated_analysis...")
    try:
        result = await orchestrated_analysis(
            query="What's the technical outlook for Apple stock?",
            persona="moderate",
            routing_strategy="rule_based",  # Use rule_based to avoid LLM calls
            max_agents=2,
        )
        print(f"✅ Success: {result['status']}")
        if result["status"] == "success":
            print(f"📈 Agent Type: {result.get('agent_type', 'unknown')}")
            print(f"🎭 Persona: {result.get('persona', 'unknown')}")
            print(f"⏱️ Execution Time: {result.get('execution_time_ms', 0):.2f}ms")
        return result["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_deep_research_financial():
    """Test the deep_research_financial function."""
    print("\n🧪 Testing deep_research_financial...")
    try:
        result = await deep_research_financial(
            research_topic="Apple Inc",
            persona="moderate",
            research_depth="basic",  # Use basic depth to minimize processing
            timeframe="7d",
        )
        print(f"✅ Success: {result['status']}")
        if result["status"] == "success":
            print(f"🔍 Agent Type: {result.get('agent_type', 'unknown')}")
            print(f"📚 Research Topic: {result.get('research_topic', 'unknown')}")
        return result["status"] == "success"
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing Orchestration Tools\n" + "=" * 50)

    # Test 1: List available agents
    test1_passed = await test_list_available_agents()

    # Test 2: Orchestrated analysis
    test2_passed = await test_orchestrated_analysis()

    # Test 3: Deep research
    test3_passed = await test_deep_research_financial()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  List Available Agents: {'✅' if test1_passed else '❌'}")
    print(f"  Orchestrated Analysis: {'✅' if test2_passed else '❌'}")
    print(f"  Deep Research: {'✅' if test3_passed else '❌'}")

    total_passed = sum([test1_passed, test2_passed, test3_passed])
    print(f"\n🎯 Total: {total_passed}/3 tests passed")

    if total_passed == 3:
        print("🎉 All orchestration tools are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed - check the errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
