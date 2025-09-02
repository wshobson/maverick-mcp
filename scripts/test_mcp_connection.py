#!/usr/bin/env python3
"""
MCP Connection Testing Script

Tests MCP server connection stability and tool registration consistency
to diagnose disappearing tools in Claude Desktop.
"""

import asyncio
import json
import sys
import time
from typing import Any

import httpx


async def test_sse_connection(
    host: str = "localhost", port: int = 8003
) -> dict[str, Any]:
    """Test SSE connection stability."""
    results = {"transport": "sse", "tests": []}

    try:
        url = f"http://{host}:{port}/sse"

        # Test 1: Basic connection
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            results["tests"].append(
                {
                    "test": "basic_connection",
                    "status": "success" if response.status_code == 200 else "failed",
                    "status_code": response.status_code,
                    "details": response.text[:200]
                    if response.status_code != 200
                    else "Connected",
                }
            )

        # Test 2: Multiple rapid connections (simulating mcp-remote)
        connection_results = []
        for i in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    connection_results.append(
                        {
                            "attempt": i + 1,
                            "status_code": response.status_code,
                            "success": response.status_code == 200,
                        }
                    )
                await asyncio.sleep(0.1)  # Brief delay between connections
            except Exception as e:
                connection_results.append(
                    {"attempt": i + 1, "error": str(e), "success": False}
                )

        results["tests"].append(
            {
                "test": "multiple_connections",
                "results": connection_results,
                "success_rate": sum(1 for r in connection_results if r.get("success"))
                / len(connection_results),
            }
        )

    except Exception as e:
        results["error"] = str(e)

    return results


async def test_streamable_http_connection(
    host: str = "localhost", port: int = 8003
) -> dict[str, Any]:
    """Test streamable-http connection stability."""
    results = {"transport": "streamable-http", "tests": []}

    try:
        url = f"http://{host}:{port}/mcp"

        # Test 1: Basic connection
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json={"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
            )
            results["tests"].append(
                {
                    "test": "basic_connection",
                    "status": "success" if response.status_code == 200 else "failed",
                    "status_code": response.status_code,
                    "response": response.text[:200],
                }
            )

        # Test 2: Tool listing
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )

            if response.status_code == 200:
                data = response.json()
                tools_count = len(data.get("result", {}).get("tools", []))
                results["tests"].append(
                    {
                        "test": "tool_listing",
                        "status": "success",
                        "tools_count": tools_count,
                        "has_research_tools": any(
                            "research" in tool.get("name", "")
                            for tool in data.get("result", {}).get("tools", [])
                        ),
                    }
                )
            else:
                results["tests"].append(
                    {
                        "test": "tool_listing",
                        "status": "failed",
                        "status_code": response.status_code,
                        "response": response.text[:200],
                    }
                )

    except Exception as e:
        results["error"] = str(e)

    return results


async def test_connection_persistence(
    host: str = "localhost", port: int = 8003, duration: int = 30
) -> dict[str, Any]:
    """Test connection persistence over time."""
    results = {"test": "connection_persistence", "duration": duration, "samples": []}

    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{host}:{port}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": int(time.time()),
                        "method": "tools/list",
                        "params": {},
                    },
                )

                sample = {
                    "timestamp": time.time(),
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                }

                if response.status_code == 200:
                    data = response.json()
                    sample["tools_count"] = len(data.get("result", {}).get("tools", []))

                results["samples"].append(sample)

        except Exception as e:
            results["samples"].append(
                {"timestamp": time.time(), "error": str(e), "success": False}
            )

        await asyncio.sleep(1)  # Sample every second

    # Calculate statistics
    successful_samples = [s for s in results["samples"] if s.get("success")]
    results["statistics"] = {
        "total_samples": len(results["samples"]),
        "successful_samples": len(successful_samples),
        "success_rate": len(successful_samples) / len(results["samples"])
        if results["samples"]
        else 0,
        "avg_tools_count": sum(s.get("tools_count", 0) for s in successful_samples)
        / len(successful_samples)
        if successful_samples
        else 0,
    }

    return results


async def main():
    """Run comprehensive MCP connection tests."""
    print("🔧 MCP Connection Stability Test")
    print("=" * 50)

    host = "localhost"
    port = 8003

    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{host}:{port}/health")
            print(f"✅ Server is running on {host}:{port}")
    except Exception:
        print(f"❌ Server not running on {host}:{port}")
        print("   Start server with: make dev")
        return

    print("\n📊 Running Connection Tests...")

    # Test SSE transport
    print("\n1. Testing SSE Transport...")
    sse_results = await test_sse_connection(host, port)
    print(f"   SSE Connection: {'✅' if not sse_results.get('error') else '❌'}")

    # Test streamable-http transport
    print("\n2. Testing Streamable-HTTP Transport...")
    http_results = await test_streamable_http_connection(host, port)
    print(f"   HTTP Connection: {'✅' if not http_results.get('error') else '❌'}")

    # Check tool counts
    for test in http_results.get("tests", []):
        if test.get("test") == "tool_listing":
            tools_count = test.get("tools_count", 0)
            has_research = test.get("has_research_tools", False)
            print(
                f"   Tools Found: {tools_count} ({'✅ Research tools present' if has_research else '⚠️  No research tools'})"
            )

    # Test connection persistence
    print("\n3. Testing Connection Persistence (30s)...")
    persistence_results = await test_connection_persistence(host, port, 30)
    stats = persistence_results["statistics"]
    print(
        f"   Success Rate: {stats['success_rate']:.1%} ({stats['successful_samples']}/{stats['total_samples']} samples)"
    )
    print(f"   Avg Tools: {stats['avg_tools_count']:.0f}")

    # Generate report
    report = {
        "timestamp": time.time(),
        "server": f"{host}:{port}",
        "results": {
            "sse": sse_results,
            "streamable_http": http_results,
            "persistence": persistence_results,
        },
    }

    # Save detailed results
    with open("mcp_connection_test_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n📄 Detailed results saved to: mcp_connection_test_results.json")

    # Recommendations
    print("\n💡 Recommendations:")
    if stats["success_rate"] < 0.9:
        print("   ⚠️  Connection instability detected")
        print("   → Consider using STDIO transport for Claude Desktop")
        print("   → Check server logs for connection errors")

    if http_results.get("tests", [{}])[1].get("tools_count", 0) == 0:
        print("   ⚠️  Tools not registering properly")
        print("   → Check connection manager initialization")
        print("   → Verify tool registration in server logs")

    print("\n🔗 For Claude Desktop, try STDIO connection:")
    print('   "command": "uv",')
    print(
        '   "args": ["run", "python", "-m", "maverick_mcp.api.server", "--transport", "stdio"],'
    )
    print(f'   "cwd": "{sys.path[0]}"')


if __name__ == "__main__":
    asyncio.run(main())
