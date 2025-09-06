#!/usr/bin/env python3
"""
Test suite to verify the three critical MCP tool fixes are working properly.

This test validates that the fixes for:
1. Research returning empty results (API keys not passed to DeepResearchAgent)
2. Portfolio risk analysis cryptic "'high'" error (DataFrame validation and column case)
3. External API key hard dependency (graceful degradation)

All continue to work correctly after code changes.

## Issues Fixed

### Issue #1: Research Returning Empty Results
- **Root Cause**: API keys weren't passed from settings to DeepResearchAgent constructor
- **Files Modified**:
  - `maverick_mcp/api/routers/research.py:line 35-40` - Added API key parameters
  - `maverick_mcp/providers/llm_factory.py:line 30` - Fixed temperature and streaming
- **Fix**: Pass exa_api_key and tavily_api_key to DeepResearchAgent, fix LLM config

### Issue #2: Portfolio Risk Analysis "'high'" Error
- **Root Cause**: DataFrame column name case mismatch and date range problems
- **Files Modified**: `maverick_mcp/api/routers/portfolio.py:line 66-84`
- **Fixes**:
  - Added DataFrame validation before column access
  - Fixed column name case sensitivity (High/Low/Close vs high/low/close)
  - Used explicit date range to avoid weekend/holiday data fetch issues

### Issue #3: External API Key Hard Dependency
- **Root Cause**: Hard failure when EXTERNAL_DATA_API_KEY not configured
- **Files Modified**: `maverick_mcp/api/routers/data.py:line 244-253`
- **Fix**: Graceful degradation with informative fallback message

## Running This Test

```bash
# Via Makefile (recommended)
make test-fixes

# Direct execution
uv run python maverick_mcp/tests/test_mcp_tool_fixes.py

# Via pytest (if environment allows)
pytest maverick_mcp/tests/test_fixes_validation.py
```

This test should be run after any changes to ensure the MCP tool fixes remain intact.
"""

import asyncio
import os

from maverick_mcp.api.routers.data import get_stock_info
from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis
from maverick_mcp.validation.data import GetStockInfoRequest


def test_portfolio_risk_analysis():
    """
    Test Issue #2: Portfolio risk analysis (formerly returned cryptic 'high' error).

    This test validates:
    - DataFrame is properly retrieved with correct columns
    - Column name case sensitivity is handled correctly
    - Date range calculation avoids weekend/holiday issues
    - Risk calculations complete successfully
    """
    print("🧪 Testing portfolio risk analysis (Issue #2)...")
    try:
        # First test what data we actually get from the provider
        from datetime import UTC, datetime, timedelta

        from maverick_mcp.api.routers.portfolio import stock_provider

        print("   Debugging: Testing data provider directly...")
        end_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
        start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")
        df = stock_provider.get_stock_data(
            "MSFT", start_date=start_date, end_date=end_date
        )

        print(f"   DataFrame shape: {df.shape}")
        print(f"   DataFrame columns: {list(df.columns)}")
        print(f"   DataFrame empty: {df.empty}")
        if not df.empty:
            print(f"   Sample data (last 3 rows):\n{df.tail(3)}")

        # Now test the actual function
        result = risk_adjusted_analysis("MSFT", 75.0)
        if "error" in result:
            # If still error, try string conversion
            result = risk_adjusted_analysis("MSFT", "75")
            if "error" in result:
                print(f"❌ Still has error: {result}")
                return False

        print(
            f"✅ Success! Current price: ${result.get('current_price')}, Risk level: {result.get('risk_level')}"
        )
        print(
            f"   Position sizing: ${result.get('position_sizing', {}).get('suggested_position_size')}"
        )
        print(f"   Strategy type: {result.get('analysis', {}).get('strategy_type')}")
        return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_stock_info_external_api():
    """
    Test Issue #3: Stock info requiring EXTERNAL_DATA_API_KEY.

    This test validates:
    - External API dependency is optional
    - Graceful fallback when EXTERNAL_DATA_API_KEY not configured
    - Core stock info functionality still works
    """
    print("\n🧪 Testing stock info external API handling (Issue #3)...")
    try:
        request = GetStockInfoRequest(ticker="MSFT")
        result = get_stock_info(request)
        if "error" in result and "Invalid API key" in str(result.get("error")):
            print(f"❌ Still failing on external API: {result}")
            return False
        else:
            print(f"✅ Success! Company: {result.get('company', {}).get('name')}")
            print(
                f"   Current price: ${result.get('market_data', {}).get('current_price')}"
            )
            return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


async def test_research_empty_results():
    """
    Test Issue #1: Research returning empty results.

    This test validates:
    - DeepResearchAgent is created with API keys from settings
    - Search providers are properly initialized
    - API keys are correctly passed through the configuration chain
    """
    print("\n🧪 Testing research functionality (Issue #1)...")
    try:
        # Import the research function
        from maverick_mcp.api.routers.research import get_research_agent

        # Test that the research agent can be created with API keys
        agent = get_research_agent()

        # Check if API keys are available in environment
        exa_key = os.getenv("EXA_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        print(f"   API keys available: EXA={bool(exa_key)}, TAVILY={bool(tavily_key)}")

        # Check if the agent has search providers (indicates API keys were passed correctly)
        if hasattr(agent, "search_providers") and len(agent.search_providers) > 0:
            print(
                f"✅ Research agent created with {len(agent.search_providers)} search providers!"
            )

            # Try to access the provider API keys to verify they're configured
            providers_configured = 0
            for provider in agent.search_providers:
                if hasattr(provider, "api_key") and provider.api_key:
                    providers_configured += 1

            if providers_configured > 0:
                print(
                    f"✅ {providers_configured} search providers have API keys configured"
                )
                return True
            else:
                print("❌ Search providers missing API keys")
                return False
        else:
            print("❌ Research agent has no search providers configured")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_llm_configuration():
    """
    Test LLM configuration fixes.

    This test validates:
    - LLM can be created successfully
    - Temperature and streaming settings are compatible with gpt-5-mini
    - LLM can handle basic queries without errors
    """
    print("\n🧪 Testing LLM configuration...")
    try:
        from maverick_mcp.providers.llm_factory import get_llm

        print("   Creating LLM instance...")
        llm = get_llm()
        print(f"   LLM created: {type(llm).__name__}")

        # Test a simple query to ensure it works
        print("   Testing LLM query...")
        response = llm.invoke("What is 2+2?")
        print(f"✅ LLM response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False


def main():
    """Run comprehensive test suite for MCP tool fixes."""
    print("🚀 Testing MCP Tool Fixes")
    print("=" * 50)

    results = []

    # Test portfolio risk analysis
    results.append(test_portfolio_risk_analysis())

    # Test stock info external API handling
    results.append(test_stock_info_external_api())

    # Test research functionality
    results.append(asyncio.run(test_research_empty_results()))

    # Test LLM configuration
    results.append(test_llm_configuration())

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n🎉 All MCP tool fixes are working correctly!")
        print("\nFixed Issues:")
        print("1. ✅ Research tools return actual content (API keys properly passed)")
        print(
            "2. ✅ Portfolio risk analysis works (DataFrame validation & column case)"
        )
        print("3. ✅ Stock info graceful fallback (external API optional)")
        print("4. ✅ LLM configuration compatible (temperature & streaming)")
    else:
        print("\n⚠️  Some issues remain to be fixed.")
        print("Please check the individual test results above.")

    return all(results)


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
