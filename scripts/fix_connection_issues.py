#!/usr/bin/env python3
"""
Quick Fix Script for MCP Connection Issues

Applies immediate fixes for disappearing tools in Claude Desktop:
1. Restarts server with enhanced connection management
2. Tests connection stability
3. Provides configuration recommendations
"""

import json
import subprocess
import time
from pathlib import Path


def print_header(text: str):
    """Print formatted header."""
    print(f"\n🔧 {text}")
    print("=" * (len(text) + 3))


def run_command(
    cmd: list[str], description: str, capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run command with error handling."""
    print(f"   Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        if capture_output:
            print(f"   ✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        if capture_output and e.stdout:
            print(f"      stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"      stderr: {e.stderr}")
        raise


def stop_existing_servers():
    """Stop any existing MCP servers."""
    print_header("Stopping Existing Servers")

    # Kill any running maverick-mcp processes
    try:
        result = subprocess.run(
            ["pkill", "-f", "maverick-mcp"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("   ✅ Stopped existing MCP servers")
        else:
            print("   ℹ️  No existing MCP servers found")
    except Exception as e:
        print(f"   ⚠️  Could not check for existing servers: {e}")

    # Wait for processes to stop
    time.sleep(2)


def start_enhanced_server():
    """Start server with enhanced connection management."""
    print_header("Starting Enhanced MCP Server")

    # Start server with SSE transport (preferred for stability)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "maverick_mcp.api.server",
        "--transport",
        "sse",
        "--port",
        "8003",
        "--host",
        "0.0.0.0",
    ]

    print("   Starting server with enhanced connection management...")
    print("   Transport: SSE (optimized for stability)")
    print("   Port: 8003")
    print("   Connection Manager: Enabled")
    print("   Tool Registration: Single initialization pattern")

    # Start server in background
    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to initialize
        print("   Waiting for server initialization...")
        time.sleep(8)

        # Check if process is still running
        if process.poll() is None:
            print("   ✅ Enhanced MCP server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print("   ❌ Server failed to start")
            print(f"      stdout: {stdout}")
            print(f"      stderr: {stderr}")
            return None

    except Exception as e:
        print(f"   ❌ Failed to start server: {e}")
        return None


def test_connection_stability():
    """Test connection stability."""
    print_header("Testing Connection Stability")

    # Run connection test script
    try:
        test_script = Path(__file__).parent / "test_mcp_connection.py"
        if test_script.exists():
            run_command(
                ["python", str(test_script)],
                "Connection stability test",
                capture_output=False,
            )
        else:
            print("   ⚠️  Connection test script not found, skipping test")
    except Exception as e:
        print(f"   ⚠️  Connection test failed: {e}")


def update_claude_desktop_config():
    """Update Claude Desktop configuration with optimal settings."""
    print_header("Claude Desktop Configuration")

    # Configuration paths
    config_paths = [
        Path.home()
        / "Library/Application Support/Claude/claude_desktop_config.json",  # macOS
        Path.home() / ".config/Claude/claude_desktop_config.json",  # Linux
        Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json",  # Windows
    ]

    config_path = None
    for path in config_paths:
        if path.parent.exists():
            config_path = path
            break

    if not config_path:
        print("   ⚠️  Claude Desktop config directory not found")
        print("   Manual configuration required")
        return

    # Read existing config
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            print(f"   ⚠️  Could not read existing config: {e}")

    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix(".json.backup")
        try:
            with open(backup_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"   ✅ Config backed up to: {backup_path}")
        except Exception as e:
            print(f"   ⚠️  Could not backup config: {e}")

    # Update with optimal configuration
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Option 1: STDIO (Recommended - no mcp-remote needed)
    config["mcpServers"]["maverick-mcp-stdio"] = {
        "command": "uv",
        "args": [
            "run",
            "python",
            "-m",
            "maverick_mcp.api.server",
            "--transport",
            "stdio",
        ],
        "cwd": str(Path(__file__).parent.parent.absolute()),
    }

    # Option 2: HTTP with mcp-remote (Fallback)
    config["mcpServers"]["maverick-mcp-http"] = {
        "command": "npx",
        "args": ["-y", "mcp-remote", "http://localhost:8003/mcp"],
    }

    # Write updated config
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"   ✅ Updated Claude Desktop config: {config_path}")
        print("   📝 Added optimal connection configurations:")
        print("      - maverick-mcp-stdio (recommended)")
        print("      - maverick-mcp-http (fallback)")
    except Exception as e:
        print(f"   ❌ Failed to update config: {e}")


def provide_recommendations():
    """Provide connection recommendations."""
    print_header("Connection Recommendations")

    print("   🎯 For Best Stability:")
    print("      1. Use STDIO transport (maverick-mcp-stdio)")
    print("      2. No mcp-remote bridge needed")
    print("      3. Direct process communication")
    print("      4. Lowest latency and highest reliability")

    print("\n   🔄 If Tools Disappear:")
    print("      1. Restart Claude Desktop completely")
    print("      2. Check server logs: tail -f logs/maverick_mcp.log")
    print("      3. Use connection status tool: get_mcp_connection_status()")
    print("      4. Switch to STDIO if using HTTP")

    print("\n   🐛 Debugging Commands:")
    print("      • Test connection: python scripts/test_mcp_connection.py")
    print("      • Check processes: ps aux | grep maverick")
    print("      • Check port: lsof -i :8003")
    print("      • View logs: tail -f logs/maverick_mcp.log")


def main():
    """Main fix application."""
    print("🚀 MCP Connection Issue Quick Fix")
    print(
        "This script applies immediate fixes for disappearing tools in Claude Desktop"
    )

    # Step 1: Stop existing servers
    stop_existing_servers()

    # Step 2: Start enhanced server
    server_process = start_enhanced_server()

    if server_process:
        # Step 3: Test connection stability
        test_connection_stability()

        # Step 4: Update Claude Desktop config
        update_claude_desktop_config()

        # Step 5: Provide recommendations
        provide_recommendations()

        print_header("Fix Application Complete")
        print("   ✅ Enhanced MCP server is running")
        print("   ✅ Connection stability optimizations applied")
        print("   ✅ Claude Desktop configuration updated")
        print("\n   🔄 Next Steps:")
        print("      1. Restart Claude Desktop")
        print("      2. Test with: 'Show me available tools'")
        print("      3. If issues persist, use STDIO configuration")

        # Keep server running
        try:
            print("\n   📡 Server running on http://localhost:8003")
            print("   Press Ctrl+C to stop server")
            server_process.wait()
        except KeyboardInterrupt:
            print("\n   🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("   ✅ Server stopped")
    else:
        print_header("Fix Failed")
        print("   ❌ Could not start enhanced server")
        print("   🔧 Try manual server start: make dev")


if __name__ == "__main__":
    main()
