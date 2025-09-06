#!/bin/bash
"""
Integration Test Runner for MaverickMCP Orchestration Tools

This script runs the comprehensive integration test suite with proper environment setup
and provides clear output for validation of all orchestration capabilities.
"""

set -e  # Exit on any error

echo "🚀 MaverickMCP Orchestration Integration Test Runner"
echo "=================================================="

# Check if we're in the right directory
if [[ ! -f "test_orchestration_complete.py" ]]; then
    echo "❌ Error: Must run from tests/integration directory"
    exit 1
fi

# Navigate to project root for proper imports
cd "$(dirname "$0")/../.."

# Check Python environment
echo "🔍 Checking Python environment..."
if command -v uv >/dev/null 2>&1; then
    echo "✅ Using uv for Python environment"
    PYTHON_CMD="uv run python"
elif [[ -f ".venv/bin/activate" ]]; then
    echo "✅ Using virtual environment"
    source .venv/bin/activate
    PYTHON_CMD="python"
else
    echo "⚠️  No virtual environment detected, using system Python"
    PYTHON_CMD="python"
fi

# Check required dependencies
echo "🔍 Checking dependencies..."
$PYTHON_CMD -c "import maverick_mcp; print('✅ maverick_mcp package found')" || {
    echo "❌ maverick_mcp package not installed. Run 'make setup' first."
    exit 1
}

# Check if MCP server dependencies are available
$PYTHON_CMD -c "from maverick_mcp.api.routers.agents import orchestrated_analysis; print('✅ Orchestration tools available')" || {
    echo "❌ Orchestration tools not available. Check agent dependencies."
    exit 1
}

# Set up test environment
echo "🛠️  Setting up test environment..."

# Check for API keys (optional)
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠️  OPENAI_API_KEY not set - tests will use mock responses"
else
    echo "✅ OPENAI_API_KEY found"
fi

if [[ -z "$EXA_API_KEY" ]]; then
    echo "⚠️  EXA_API_KEY not set - deep research may have limited functionality"
else
    echo "✅ EXA_API_KEY found"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "🧪 Starting comprehensive integration tests..."
echo "   This will test all orchestration capabilities including:"
echo "   - agents_orchestrated_analysis with multiple personas/routing"
echo "   - agents_deep_research_financial with various depths/focus areas"  
echo "   - agents_compare_multi_agent_analysis with different combinations"
echo "   - Error handling and edge cases"
echo "   - Concurrent execution performance"
echo "   - Memory usage monitoring"
echo ""

# Run the comprehensive test suite
$PYTHON_CMD tests/integration/test_orchestration_complete.py

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "=================================================="

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo "🎉 ALL INTEGRATION TESTS PASSED!"
    echo "   The orchestration tools are working correctly and ready for production use."
elif [[ $TEST_EXIT_CODE -eq 1 ]]; then
    echo "⚠️  SOME TESTS FAILED"
    echo "   Check the test output above and log files for details."
elif [[ $TEST_EXIT_CODE -eq 130 ]]; then
    echo "🛑 TESTS INTERRUPTED BY USER"
else
    echo "💥 TEST SUITE EXECUTION FAILED"
    echo "   Check the error output and ensure all dependencies are properly installed."
fi

echo ""
echo "📊 Test artifacts:"
echo "   - Detailed logs: integration_test_*.log"
echo "   - JSON results: integration_test_results_*.json"
echo ""

exit $TEST_EXIT_CODE