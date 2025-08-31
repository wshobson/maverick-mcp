"""
Pytest-compatible test suite for MCP tool fixes.

This test validates that the fixes for:
1. Research returning empty results (API keys not passed to DeepResearchAgent)  
2. Portfolio risk analysis cryptic "'high'" error (DataFrame validation and column case)
3. External API key hard dependency (graceful degradation)

All continue to work correctly after code changes.
"""

import os
import pytest
from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis, stock_provider
from maverick_mcp.api.routers.data import get_stock_info
from maverick_mcp.validation.data import GetStockInfoRequest
from datetime import UTC, datetime, timedelta


@pytest.mark.integration
@pytest.mark.external
def test_portfolio_risk_analysis_fix():
    """
    Test Issue #2: Portfolio risk analysis DataFrame validation and column case fix.
    
    Validates:
    - DataFrame is properly retrieved with correct columns
    - Column name case sensitivity is handled correctly  
    - Date range calculation avoids weekend/holiday issues
    - Risk calculations complete successfully
    """
    # Test data provider directly first
    end_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
    start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")
    df = stock_provider.get_stock_data('MSFT', start_date=start_date, end_date=end_date)
    
    # Verify DataFrame has expected structure
    assert not df.empty, "DataFrame should not be empty"
    assert df.shape[0] > 200, "Should have substantial historical data"
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
    
    # Test the actual portfolio risk analysis function
    result = risk_adjusted_analysis('MSFT', 75.0)
    
    # Verify successful result structure
    assert 'error' not in result, f"Should not have error: {result}"
    assert 'current_price' in result, "Should include current price"
    assert 'risk_level' in result, "Should include risk level"
    assert 'position_sizing' in result, "Should include position sizing"
    assert 'analysis' in result, "Should include analysis"
    
    # Verify data types and ranges
    assert isinstance(result['current_price'], (int, float)), "Current price should be numeric"
    assert result['current_price'] > 0, "Current price should be positive"
    assert result['risk_level'] == 75.0, "Risk level should match input"
    
    position_size = result['position_sizing']['suggested_position_size']
    assert isinstance(position_size, (int, float)), "Position size should be numeric"
    assert position_size > 0, "Position size should be positive"


@pytest.mark.integration
@pytest.mark.database
def test_stock_info_external_api_graceful_fallback():
    """
    Test Issue #3: External API graceful fallback handling.
    
    Validates:
    - External API dependency is optional
    - Graceful fallback when EXTERNAL_DATA_API_KEY not configured
    - Core stock info functionality still works
    """
    request = GetStockInfoRequest(ticker='MSFT')
    result = get_stock_info(request)
    
    # Should not have hard errors about missing API keys
    if 'error' in result:
        assert 'Invalid API key' not in str(result.get('error')), \
            f"Should not have hard API key error: {result}"
    
    # Should have basic company information
    assert 'company' in result, "Should include company information"
    assert 'market_data' in result, "Should include market data"
    
    company = result.get('company', {})
    assert company.get('name'), "Should have company name"
    
    market_data = result.get('market_data', {})
    current_price = market_data.get('current_price')
    if current_price:
        assert isinstance(current_price, (int, float)), "Price should be numeric"
        assert current_price > 0, "Price should be positive"


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.asyncio
async def test_research_agent_api_key_configuration():
    """
    Test Issue #1: Research agent API key configuration fix.
    
    Validates:
    - DeepResearchAgent is created with API keys from settings
    - Search providers are properly initialized
    - API keys are correctly passed through the configuration chain
    """
    from maverick_mcp.api.routers.research import get_research_agent
    
    # Check environment has required API keys
    exa_key = os.getenv('EXA_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not (exa_key and tavily_key):
        pytest.skip("EXA_API_KEY and TAVILY_API_KEY required for research test")
    
    # Create research agent
    agent = get_research_agent()
    
    # Verify agent has search providers
    assert hasattr(agent, 'search_providers'), "Agent should have search_providers"
    assert len(agent.search_providers) > 0, "Should have at least one search provider"
    
    # Verify providers have API keys configured
    providers_configured = 0
    for provider in agent.search_providers:
        if hasattr(provider, 'api_key') and provider.api_key:
            providers_configured += 1
    
    assert providers_configured > 0, "At least one search provider should have API key configured"
    assert providers_configured >= 2, "Should have both EXA and Tavily providers configured"


@pytest.mark.unit
def test_llm_configuration_compatibility():
    """
    Test LLM configuration fixes.
    
    Validates:
    - LLM can be created successfully
    - Temperature and streaming settings are compatible with gpt-5-mini
    - LLM can handle basic queries without errors
    """
    from maverick_mcp.providers.llm_factory import get_llm
    
    # Test LLM creation
    llm = get_llm()
    assert llm is not None, "LLM should be created successfully"
    
    # Test basic query to ensure configuration is working
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        response = llm.invoke("What is 2+2?")
        assert response is not None, "LLM should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert '4' in response.content, "LLM should correctly answer 2+2=4"
    else:
        pytest.skip("OPENAI_API_KEY required for LLM test")


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.database
def test_all_mcp_fixes_integration():
    """
    Integration test to verify all three MCP tool fixes work together.
    
    This is a comprehensive test that ensures all fixes are compatible
    and work correctly in combination.
    """
    # Test 1: Portfolio analysis
    portfolio_result = risk_adjusted_analysis('AAPL', 50.0)
    assert 'error' not in portfolio_result, "Portfolio analysis should work"
    
    # Test 2: Stock info
    request = GetStockInfoRequest(ticker='AAPL')
    stock_info_result = get_stock_info(request)
    assert 'company' in stock_info_result, "Stock info should work"
    
    # Test 3: Research agent (if API keys available)
    exa_key = os.getenv('EXA_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if exa_key and tavily_key:
        from maverick_mcp.api.routers.research import get_research_agent
        agent = get_research_agent()
        assert len(agent.search_providers) >= 2, "Research agent should have providers"
    
    # Test 4: LLM configuration
    from maverick_mcp.providers.llm_factory import get_llm
    llm = get_llm()
    assert llm is not None, "LLM should be configured correctly"