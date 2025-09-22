#!/usr/bin/env python3
"""
Example demonstrating MaverickMCP monitoring and observability features.

This example shows how to:
1. Enable monitoring and tracing
2. Use monitoring utilities in your code
3. Access monitoring endpoints
4. View metrics and traces
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

# Set environment variables for monitoring
os.environ.update(
    {
        "OTEL_TRACING_ENABLED": "true",
        "JAEGER_ENDPOINT": "http://localhost:14268/api/traces",
        "SENTRY_DSN": "",  # Optional: add your Sentry DSN
    }
)

from maverick_mcp.utils.logging import get_logger
from maverick_mcp.utils.monitoring import (
    track_cache_operation,
    track_external_api_call,
    track_tool_usage,
    update_performance_metrics,
)
from maverick_mcp.utils.tracing import (
    trace_cache_operation,
    trace_database_query,
    trace_external_api_call,
    trace_operation,
)

logger = get_logger(__name__)


@asynccontextmanager
async def example_database_operation(query_type: str, table: str):
    """Example of monitoring a database operation."""
    with trace_database_query(query_type, table, f"SELECT * FROM {table}"):
        start_time = time.time()
        try:
            # Simulate database operation
            await asyncio.sleep(0.1)
            yield "mock_result"

            # Track successful operation
            duration = time.time() - start_time
            from maverick_mcp.utils.monitoring import track_database_query

            track_database_query(query_type, table, duration, "success")

        except Exception:
            # Track failed operation
            duration = time.time() - start_time
            from maverick_mcp.utils.monitoring import track_database_query

            track_database_query(query_type, table, duration, "error")
            raise


@asynccontextmanager
async def example_external_api_call(service: str, endpoint: str):
    """Example of monitoring an external API call."""
    with trace_external_api_call(service, endpoint):
        start_time = time.time()
        try:
            # Simulate external API call
            await asyncio.sleep(0.2)
            status_code = 200
            yield {"data": "mock_response"}

            # Track successful API call
            duration = time.time() - start_time
            track_external_api_call(service, endpoint, "GET", status_code, duration)

        except Exception as e:
            # Track failed API call
            duration = time.time() - start_time
            track_external_api_call(
                service, endpoint, "GET", 500, duration, str(type(e).__name__)
            )
            raise


@asynccontextmanager
async def example_cache_operation(key: str):
    """Example of monitoring a cache operation."""
    with trace_cache_operation("get", "redis"):
        time.time()
        try:
            # Simulate cache operation
            await asyncio.sleep(0.01)
            hit = True  # Simulate cache hit
            yield "cached_value"

            # Track cache operation
            track_cache_operation("redis", "get", hit, key.split(":")[0])

        except Exception:
            # Track cache miss/error
            track_cache_operation("redis", "get", False, key.split(":")[0])
            raise


async def example_tool_execution(tool_name: str, user_id: str):
    """Example of monitoring a tool execution."""
    with trace_operation(
        f"tool.{tool_name}", {"tool.name": tool_name, "user.id": user_id}
    ):
        start_time = time.time()

        try:
            # Simulate tool execution with some operations
            logger.info(f"Executing tool: {tool_name}", extra={"user_id": user_id})

            # Example: Database query
            async with example_database_operation("SELECT", "stocks") as db_result:
                logger.info(f"Database query result: {db_result}")

            # Example: External API call
            async with example_external_api_call(
                "yahoo_finance", "/quote/AAPL"
            ) as api_result:
                logger.info(f"API call result: {api_result}")

            # Example: Cache operation
            async with example_cache_operation("stock:AAPL:price") as cache_result:
                logger.info(f"Cache result: {cache_result}")

            # Simulate processing time
            await asyncio.sleep(0.5)

            # Track successful tool execution
            duration = time.time() - start_time
            track_tool_usage(
                tool_name=tool_name,
                user_id=user_id,
                duration=duration,
                status="success",
                complexity="standard",
            )

            return {
                "status": "success",
                "data": "Tool execution completed",
                "duration_ms": int(duration * 1000),
            }

        except Exception as e:
            # Track failed tool execution
            duration = time.time() - start_time
            from maverick_mcp.utils.monitoring import track_tool_error

            track_tool_error(tool_name, type(e).__name__, "standard")

            logger.error(f"Tool execution failed: {tool_name}", exc_info=True)
            raise


async def demonstrate_monitoring():
    """Demonstrate various monitoring features."""
    logger.info("Starting monitoring demonstration...")

    # Initialize monitoring (this would normally be done by the server)
    from maverick_mcp.utils.monitoring import initialize_monitoring
    from maverick_mcp.utils.tracing import initialize_tracing

    initialize_monitoring()
    initialize_tracing()

    # Example 1: Tool execution monitoring
    logger.info("=== Example 1: Tool Execution Monitoring ===")
    result = await example_tool_execution("get_stock_data", "user123")
    print(f"Tool result: {result}")

    # Example 2: Performance metrics
    logger.info("=== Example 2: Performance Metrics ===")
    update_performance_metrics()
    print("Performance metrics updated")

    # Example 3: Multiple tool executions for metrics
    logger.info("=== Example 3: Multiple Tool Executions ===")
    tools = ["get_technical_analysis", "screen_stocks", "get_portfolio_data"]
    users = ["user123", "user456", "user789"]

    for i in range(5):
        tool = tools[i % len(tools)]
        user = users[i % len(users)]
        try:
            result = await example_tool_execution(tool, user)
            print(f"Tool {tool} for {user}: {result['status']}")
        except Exception as e:
            print(f"Tool {tool} for {user}: FAILED - {e}")

        # Small delay between executions
        await asyncio.sleep(0.1)

    # Example 4: Error scenarios
    logger.info("=== Example 4: Error Scenarios ===")
    try:
        # Simulate a tool that fails
        with trace_operation("tool.failing_tool", {"tool.name": "failing_tool"}):
            raise ValueError("Simulated tool failure")
    except ValueError as e:
        logger.error(f"Expected error: {e}")
        from maverick_mcp.utils.monitoring import track_tool_error

        track_tool_error("failing_tool", "ValueError", "standard")

    # Example 5: Security events
    logger.info("=== Example 5: Security Events ===")
    from maverick_mcp.utils.monitoring import (
        track_authentication,
        track_security_violation,
    )

    # Simulate authentication attempts
    track_authentication("bearer_token", "success", "Mozilla/5.0")
    track_authentication("bearer_token", "failure", "suspicious-bot/1.0")

    # Simulate security violation
    track_security_violation("invalid_token", "high")

    print("Security events tracked")

    # Example 6: Business metrics
    logger.info("=== Example 6: Business Metrics ===")
    from maverick_mcp.utils.monitoring import (
        track_user_session,
        update_active_users,
    )

    # Simulate engagement events
    track_user_session("registered", "api_key", duration=360.0)
    track_user_session("anonymous", "public", duration=120.0)
    update_active_users(daily_count=42, monthly_count=156)

    print("Business metrics tracked")

    logger.info("Monitoring demonstration completed!")
    print("\n=== Check Your Monitoring Stack ===")
    print("1. Prometheus metrics: http://localhost:9090")
    print("2. Grafana dashboards: http://localhost:3000")
    print("3. Jaeger traces: http://localhost:16686")
    print("4. MaverickMCP metrics: http://localhost:8000/metrics")
    print("5. MaverickMCP health: http://localhost:8000/health")


def print_monitoring_setup_instructions():
    """Print instructions for setting up the monitoring stack."""
    print("=== MaverickMCP Monitoring Setup ===")
    print()
    print("1. Start the monitoring stack:")
    print("   cd monitoring/")
    print("   docker-compose -f docker-compose.monitoring.yml up -d")
    print()
    print("2. Install monitoring dependencies:")
    print("   pip install prometheus-client opentelemetry-distro sentry-sdk")
    print()
    print("3. Set environment variables:")
    print("   export OTEL_TRACING_ENABLED=true")
    print("   export JAEGER_ENDPOINT=http://localhost:14268/api/traces")
    print()
    print("4. Start MaverickMCP:")
    print("   make dev")
    print()
    print("5. Access monitoring services:")
    print("   - Grafana: http://localhost:3000 (admin/admin)")
    print("   - Prometheus: http://localhost:9090")
    print("   - Jaeger: http://localhost:16686")
    print("   - MaverickMCP metrics: http://localhost:8000/metrics")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        print_monitoring_setup_instructions()
    else:
        # Run the monitoring demonstration
        asyncio.run(demonstrate_monitoring())
