"""The session protocol in evals/tool_surface/harness.py: the pre-prompt
handshake check, and that `converse` never sends a prompt when it fails.
`FakeClient` records every call and never starts a CLI."""

from typing import Any

import pytest

from evals.tool_surface import harness

from ._fakes import (
    ALLOWED,
    AssistantMessage,
    FakeClient,
    P,
    ResultMessage,
    SystemMessage,
    TextBlock,
    account,
    init_data,
)

CONNECTED = [{"name": "maverick", "status": "connected"}]


def _usage(mcp: list[str], system: list[str] | None = None) -> dict[str, Any]:
    return {
        "systemTools": [{"name": name} for name in system or []],
        "mcpTools": [{"name": name} for name in mcp],
    }


def _problems(info: dict[str, Any], servers: list[dict[str, Any]], usage: Any) -> list:
    return harness.handshake_problems(info, servers, usage, ALLOWED)


class TestHandshakeProblems:
    def test_subscription_session_passes(self) -> None:
        assert _problems({"account": account()}, CONNECTED, _usage(ALLOWED)) == []

    @pytest.mark.parametrize("source", ["ANTHROPIC_API_KEY", "apiKeyHelper"])
    def test_any_api_key_in_the_account_fails(self, source: str) -> None:
        info = {"account": account(apiKeySource=source)}
        problems = _problems(info, CONNECTED, _usage(ALLOWED))
        assert problems == [f"an API key is in use (apiKeySource {source!r})"]

    @pytest.mark.parametrize(
        "info",
        [None, {}, {"account": account(subscriptionType=None)}],
    )
    def test_missing_subscription_login_fails(self, info: Any) -> None:
        problems = _problems(info, CONNECTED, _usage(ALLOWED))
        assert len(problems) == 1 and "no subscription login" in problems[0]

    def test_third_party_provider_fails(self) -> None:
        info = {"account": {"apiProvider": "bedrock"}}
        assert "no subscription login" in _problems(info, CONNECTED, _usage(ALLOWED))[0]

    def test_built_in_research_and_extra_server_all_fail(self) -> None:
        usage = _usage([*ALLOWED, f"{P}research_run_comprehensive"], ["Bash"])
        servers = [*CONNECTED, {"name": "other", "status": "connected"}]
        problems = _problems({"account": account()}, servers, usage)
        assert len(problems) == 3
        assert problems[0] == "built-in tools advertised: ['Bash']"
        assert "research_run_comprehensive" in problems[1]
        assert "MCP servers" in problems[2]


async def _converse(client: FakeClient, wait: float = 5.0) -> tuple[str | None, Any]:
    builder = harness.TraceBuilder()
    abort = await harness.converse(client, "What's NVDA's P/E?", ALLOWED, builder, wait)
    return abort, builder


@pytest.mark.parametrize(
    "client",
    [
        FakeClient(info={"account": account(apiKeySource="ANTHROPIC_API_KEY")}),
        FakeClient(info={"account": {"apiProvider": "firstParty"}}),
        FakeClient(usage=_usage(ALLOWED, ["Bash"])),
        FakeClient(usage=_usage(ALLOWED[:1])),
        FakeClient(servers=[{"name": "maverick", "status": "failed"}]),
        FakeClient(servers=[{"name": "maverick", "status": "pending"}]),
    ],
    ids=["api-key", "no-plan", "built-in", "missing-tool", "failed", "pending"],
)
async def test_no_prompt_is_sent_when_the_session_check_fails(
    client: FakeClient,
) -> None:
    abort, builder = await _converse(client, wait=-1.0)

    assert abort is not None
    assert abort.startswith("session check failed before any prompt")
    assert client.prompts == []
    assert "query" not in client.calls and "receive_response" not in client.calls
    assert builder.events == [] and builder.result is None


async def test_prompt_goes_out_only_after_every_check() -> None:
    result = ResultMessage("success", 900, False, 2, 0.01, "28.5")
    client = FakeClient(stream=[SystemMessage("init", init_data()), result])

    abort, builder = await _converse(client)

    assert abort is None
    assert client.calls == [
        "get_mcp_status",
        "get_server_info",
        "get_context_usage",
        "query",
        "receive_response",
    ]
    assert client.prompts == ["What's NVDA's P/E?"]
    assert builder.result is result


async def test_the_init_message_is_checked_again_after_the_prompt() -> None:
    init = SystemMessage("init", init_data(apiKeySource="ANTHROPIC_API_KEY"))
    late = AssistantMessage([TextBlock("never collected")])
    client = FakeClient(stream=[init, late])

    abort, builder = await _converse(client)

    assert abort is not None and abort.startswith("init check failed")
    assert "apiKeySource is 'ANTHROPIC_API_KEY'" in abort
    assert builder.init == init.data and builder.events == []
