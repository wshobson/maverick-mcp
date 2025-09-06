#!/usr/bin/env python3
"""
Configuration checker for orchestrated agent setup.
Verifies that all required dependencies and configurations are available.
"""

import os
import sys
from typing import Any


def check_dependencies() -> dict[str, Any]:
    """Check if all required dependencies are available."""
    results = {"dependencies": {}, "status": "success", "missing": []}

    # Check core dependencies
    deps_to_check = [
        ("langchain_core", "LangChain core"),
        ("langgraph", "LangGraph"),
        ("fastmcp", "FastMCP"),
        ("exa_py", "Exa AI search (optional)"),
        ("tavily", "Tavily search (optional)"),
    ]

    for module, description in deps_to_check:
        try:
            __import__(module)
            results["dependencies"][module] = {
                "status": "available",
                "description": description,
            }
        except ImportError as e:
            results["dependencies"][module] = {
                "status": "missing",
                "description": description,
                "error": str(e),
            }
            if module not in ["exa_py", "tavily"]:  # Optional dependencies
                results["missing"].append(module)
                results["status"] = "error"

    return results


def check_environment_variables() -> dict[str, Any]:
    """Check environment variables for API keys."""
    results = {"environment": {}, "status": "success", "warnings": []}

    # Required variables
    required_vars = [
        ("TIINGO_API_KEY", "Stock data provider", True),
    ]

    # Optional variables
    optional_vars = [
        ("OPENAI_API_KEY", "OpenAI LLM provider", False),
        ("ANTHROPIC_API_KEY", "Anthropic LLM provider", False),
        ("EXA_API_KEY", "Exa search provider", False),
        ("TAVILY_API_KEY", "Tavily search provider", False),
        ("FRED_API_KEY", "Economic data provider", False),
    ]

    all_vars = required_vars + optional_vars

    for var_name, description, required in all_vars:
        value = os.getenv(var_name)
        if value:
            results["environment"][var_name] = {
                "status": "configured",
                "description": description,
                "has_value": bool(value and value.strip()),
            }
        else:
            results["environment"][var_name] = {
                "status": "not_configured",
                "description": description,
                "required": required,
            }

            if required:
                results["status"] = "error"
            else:
                results["warnings"].append(
                    f"{var_name} not configured - {description} will not be available"
                )

    return results


def check_agent_imports() -> dict[str, Any]:
    """Check if agent classes can be imported successfully."""
    results = {"agents": {}, "status": "success", "errors": []}

    agents_to_check = [
        ("maverick_mcp.agents.market_analysis", "MarketAnalysisAgent"),
        ("maverick_mcp.agents.supervisor", "SupervisorAgent"),
        ("maverick_mcp.agents.deep_research", "DeepResearchAgent"),
    ]

    for module_path, class_name in agents_to_check:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            results["agents"][class_name] = {
                "status": "importable",
                "module": module_path,
            }
        except Exception as e:
            results["agents"][class_name] = {
                "status": "error",
                "module": module_path,
                "error": str(e),
            }
            results["errors"].append(f"{class_name}: {str(e)}")
            results["status"] = "error"

    return results


def main():
    """Run configuration checks."""
    print("🔍 Checking MaverickMCP Orchestration Configuration...")
    print("=" * 60)

    # Check dependencies
    dep_results = check_dependencies()
    print("\n📦 Dependencies:")
    for dep, info in dep_results["dependencies"].items():
        status_icon = "✅" if info["status"] == "available" else "❌"
        print(f"  {status_icon} {dep}: {info['description']}")
        if info["status"] == "missing":
            print(f"    Error: {info['error']}")

    # Check environment variables
    env_results = check_environment_variables()
    print("\n🔧 Environment Variables:")
    for var, info in env_results["environment"].items():
        if info["status"] == "configured":
            print(f"  ✅ {var}: {info['description']}")
        else:
            icon = "❌" if info.get("required") else "⚠️ "
            print(f"  {icon} {var}: {info['description']} (not configured)")

    # Check agent imports
    agent_results = check_agent_imports()
    print("\n🤖 Agent Classes:")
    for agent, info in agent_results["agents"].items():
        status_icon = "✅" if info["status"] == "importable" else "❌"
        print(f"  {status_icon} {agent}: {info['module']}")
        if info["status"] == "error":
            print(f"    Error: {info['error']}")

    # Summary
    print("\n" + "=" * 60)

    all_status = [dep_results["status"], env_results["status"], agent_results["status"]]
    overall_status = "error" if "error" in all_status else "success"

    if overall_status == "success":
        print("✅ Configuration check PASSED!")
        print("\nOrchestrated agents are ready to use.")

        if env_results["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in env_results["warnings"]:
                print(f"  • {warning}")
    else:
        print("❌ Configuration check FAILED!")
        print("\nPlease fix the errors above before using orchestrated agents.")

        if dep_results["missing"]:
            print(f"\nMissing dependencies: {', '.join(dep_results['missing'])}")
            print("Run: uv sync")

        if env_results["status"] == "error":
            print("\nMissing required environment variables.")
            print("Copy .env.example to .env and configure required API keys.")

        if agent_results["errors"]:
            print("\nAgent import errors:")
            for error in agent_results["errors"]:
                print(f"  • {error}")

    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
