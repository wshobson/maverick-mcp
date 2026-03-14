"""Tests for the health_tools router — register_health_tools inner functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _register_and_get_tools():
    """Register health tools on a mock MCP and return the inner functions."""
    from maverick_mcp.api.routers.health_tools import register_health_tools

    tools = {}
    mcp = MagicMock()

    def tool_decorator(**kwargs):
        def decorator(func):
            name = kwargs.get("name", func.__name__)
            tools[name] = func
            return func

        return decorator

    # handle @mcp.tool() (no kwargs) — side_effect returns a decorator
    mcp.tool = MagicMock(side_effect=lambda **kw: tool_decorator(**kw))
    # also handle @mcp.tool() with no arguments at all
    mcp.tool.return_value = lambda func: tools.update({func.__name__: func}) or func

    register_health_tools(mcp)
    return tools


@pytest.fixture(scope="module")
def health_tools():
    return _register_and_get_tools()


class TestGetSystemHealth:
    @pytest.mark.asyncio
    async def test_success(self, health_tools):
        fn = health_tools["get_system_health"]
        mock_status = {"status": "healthy", "components": {}}
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_detailed_health_status=AsyncMock(return_value=mock_status)
                )
            },
        ):
            result = await fn()
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_exception(self, health_tools):
        fn = health_tools["get_system_health"]
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_detailed_health_status=AsyncMock(
                        side_effect=RuntimeError("boom")
                    )
                )
            },
        ):
            result = await fn()
            assert result["status"] == "error"
            assert "boom" in result["error"]


class TestGetComponentStatus:
    @pytest.mark.asyncio
    async def test_specific_component(self, health_tools):
        fn = health_tools["get_component_status"]
        comp = MagicMock()
        comp.__dict__ = {"status": "healthy", "latency": 5}
        mock_status = {"components": {"redis": comp}}
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_detailed_health_status=AsyncMock(return_value=mock_status)
                )
            },
        ):
            result = await fn(component_name="redis")
            assert result["status"] == "success"
            assert result["component"] == "redis"

    @pytest.mark.asyncio
    async def test_missing_component(self, health_tools):
        fn = health_tools["get_component_status"]
        mock_status = {"components": {"redis": MagicMock()}}
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_detailed_health_status=AsyncMock(return_value=mock_status)
                )
            },
        ):
            result = await fn(component_name="nonexistent")
            assert result["status"] == "error"
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_all_components(self, health_tools):
        fn = health_tools["get_component_status"]
        comp = MagicMock()
        comp.__dict__ = {"status": "healthy"}
        mock_status = {"components": {"redis": comp, "db": comp}}
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_detailed_health_status=AsyncMock(return_value=mock_status)
                )
            },
        ):
            result = await fn(component_name=None)
            assert result["status"] == "success"
            assert result["total_components"] == 2


class TestGetCircuitBreakerStatus:
    @pytest.mark.asyncio
    async def test_success(self, health_tools):
        fn = health_tools["get_circuit_breaker_status"]
        mock_cb = {
            "tiingo": {"state": "closed"},
            "fred": {"state": "open"},
        }
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.utils.circuit_breaker": MagicMock(
                    get_all_circuit_breaker_status=MagicMock(return_value=mock_cb)
                )
            },
        ):
            result = await fn()
            assert result["status"] == "success"
            assert result["summary"]["total_breakers"] == 2
            assert result["summary"]["states"]["closed"] == 1
            assert result["summary"]["states"]["open"] == 1


class TestGetResourceUsage:
    @pytest.mark.asyncio
    async def test_success(self, health_tools):
        fn = health_tools["get_resource_usage"]
        resource = MagicMock()
        resource.cpu_percent = 45.0
        resource.memory_percent = 60.0
        resource.disk_percent = 55.0
        resource.__dict__ = {
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "disk_percent": 55.0,
        }
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_resource_usage=MagicMock(return_value=resource)
                )
            },
        ):
            result = await fn()
            assert result["status"] == "success"
            assert result["alerts"]["high_cpu"] is False
            assert result["alerts"]["high_memory"] is False


class TestResetCircuitBreaker:
    @pytest.mark.asyncio
    async def test_success(self, health_tools):
        fn = health_tools["reset_circuit_breaker"]
        mock_manager = MagicMock()
        mock_manager.reset_breaker.return_value = True
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.utils.circuit_breaker": MagicMock(
                    get_circuit_breaker_manager=MagicMock(return_value=mock_manager)
                )
            },
        ):
            result = await fn(breaker_name="tiingo")
            assert result["status"] == "success"
            assert "tiingo" in result["message"]

    @pytest.mark.asyncio
    async def test_failure(self, health_tools):
        fn = health_tools["reset_circuit_breaker"]
        mock_manager = MagicMock()
        mock_manager.reset_breaker.return_value = False
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.utils.circuit_breaker": MagicMock(
                    get_circuit_breaker_manager=MagicMock(return_value=mock_manager)
                )
            },
        ):
            result = await fn(breaker_name="unknown")
            assert result["status"] == "error"


class TestRunHealthDiagnostics:
    @pytest.mark.asyncio
    async def test_success(self, health_tools):
        fn = health_tools["run_health_diagnostics"]
        comp_healthy = MagicMock()
        comp_healthy.status = "healthy"
        mock_health = {
            "status": "healthy",
            "components": {"redis": comp_healthy},
            "resource_usage": {"memory_percent": 50, "cpu_percent": 30},
        }
        with patch.dict(
            "sys.modules",
            {
                "maverick_mcp.api.routers.health_enhanced": MagicMock(
                    _get_detailed_health_status=AsyncMock(return_value=mock_health)
                ),
                "maverick_mcp.utils.circuit_breaker": MagicMock(
                    get_all_circuit_breaker_status=MagicMock(return_value={})
                ),
                "maverick_mcp.monitoring.health_monitor": MagicMock(
                    get_monitoring_status=MagicMock(return_value={"running": True})
                ),
            },
        ):
            result = await fn()
            assert result["status"] == "success"
            assert result["data"]["overall_health_score"] == 100
