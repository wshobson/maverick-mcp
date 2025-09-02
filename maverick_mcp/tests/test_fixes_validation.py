"""
Simple test to validate MCP tool fixes are working.

This test runs the comprehensive fix validation script
and ensures it passes all checks.
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_mcp_tool_fixes_validation():
    """
    Test that all MCP tool fixes are working by running the validation script.

    This test executes the comprehensive test script and verifies all fixes pass.
    """
    # Get the path to the test script
    test_script = Path(__file__).parent / "test_mcp_tool_fixes.py"

    # Run the test script
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout
    )

    # Check that the script succeeded
    assert result.returncode == 0, (
        f"MCP tool fixes validation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Verify expected success messages are in output
    output = result.stdout
    assert "🎉 All MCP tool fixes are working correctly!" in output, (
        "Expected success message not found"
    )
    assert "✅ Passed: 4/4" in output, "Expected 4/4 tests to pass"
    assert "❌ Failed: 0/4" in output, "Expected 0/4 tests to fail"

    # Verify individual fixes
    assert "✅ Research tools return actual content" in output, (
        "Research fix not validated"
    )
    assert "✅ Portfolio risk analysis works" in output, "Portfolio fix not validated"
    assert "✅ Stock info graceful fallback" in output, "Stock info fix not validated"
    assert "✅ LLM configuration compatible" in output, "LLM fix not validated"


if __name__ == "__main__":
    # Allow running this test directly
    test_mcp_tool_fixes_validation()
