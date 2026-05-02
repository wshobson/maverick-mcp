"""Unit tests for the service registry."""

from __future__ import annotations

import pytest

from maverick_mcp.services.registry import ServiceRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ServiceRegistry:
    return ServiceRegistry()


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


def test_register_and_get_returns_service(registry: ServiceRegistry) -> None:
    service = object()
    registry.register("my_service", service)
    assert registry.get("my_service") is service


def test_get_unknown_name_raises_key_error(registry: ServiceRegistry) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")


def test_register_duplicate_raises_value_error(registry: ServiceRegistry) -> None:
    registry.register("svc", object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("svc", object())


def test_register_with_replace_true_overwrites(registry: ServiceRegistry) -> None:
    original = object()
    replacement = object()
    registry.register("svc", original)
    registry.register("svc", replacement, replace=True)
    assert registry.get("svc") is replacement


# ---------------------------------------------------------------------------
# get_optional
# ---------------------------------------------------------------------------


def test_get_optional_returns_service_when_registered(
    registry: ServiceRegistry,
) -> None:
    service = object()
    registry.register("opt_svc", service)
    assert registry.get_optional("opt_svc") is service


def test_get_optional_returns_none_when_not_registered(
    registry: ServiceRegistry,
) -> None:
    assert registry.get_optional("nonexistent") is None


# ---------------------------------------------------------------------------
# has
# ---------------------------------------------------------------------------


def test_has_returns_true_when_registered(registry: ServiceRegistry) -> None:
    registry.register("existing", object())
    assert registry.has("existing") is True


def test_has_returns_false_when_not_registered(registry: ServiceRegistry) -> None:
    assert registry.has("missing") is False


# ---------------------------------------------------------------------------
# list_services
# ---------------------------------------------------------------------------


def test_list_services_empty_registry(registry: ServiceRegistry) -> None:
    assert registry.list_services() == []


def test_list_services_returns_all_names(registry: ServiceRegistry) -> None:
    registry.register("alpha", object())
    registry.register("beta", object())
    registry.register("gamma", object())
    assert registry.list_services() == ["alpha", "beta", "gamma"]


def test_list_services_after_replace_same_name(registry: ServiceRegistry) -> None:
    registry.register("svc", object())
    registry.register("svc", object(), replace=True)
    # Should still appear once
    assert registry.list_services() == ["svc"]


def test_register_accepts_any_value_type(registry: ServiceRegistry) -> None:
    registry.register("int_val", 42)
    registry.register("dict_val", {"key": "value"})
    registry.register("none_val", None)
    assert registry.get("int_val") == 42
    assert registry.get("dict_val") == {"key": "value"}
    assert registry.get("none_val") is None
