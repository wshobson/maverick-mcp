"""Tests for `maverick.research.agents.analyzer.ContentAnalyzer`.

Fully mocked: no network, no real LLM. `pytest.importorskip("langgraph")`
gates the whole module even though `analyzer.py` itself only needs
`langchain_core` -- consistent with the brief's guard for `tests/research/
test_agents*.py`, and `langgraph` pulls in `langchain_core` as a
dependency anyway so the skip condition is equivalent.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from maverick.research.agents.analyzer import ContentAnalyzer  # noqa: E402

from ._fakes import FakeChatModel  # noqa: E402


def _valid_analysis_json() -> str:
    return json.dumps(
        {
            "KEY_INSIGHTS": ["Revenue grew 20%", "Margins expanded"],
            "SENTIMENT": {"direction": "bullish", "confidence": 0.8},
            "RISK_FACTORS": ["Rising input costs"],
            "OPPORTUNITIES": ["New product line"],
            "CREDIBILITY": 0.9,
            "RELEVANCE": 0.85,
            "SUMMARY": "Strong quarter with margin expansion.",
        }
    )


def test_analyze_content_parses_valid_json() -> None:
    llm = FakeChatModel(responder=lambda _messages: _valid_analysis_json())
    analyzer = ContentAnalyzer(llm)

    import asyncio

    result = asyncio.run(analyzer.analyze_content("some content", "moderate"))

    assert result["insights"] == ["Revenue grew 20%", "Margins expanded"]
    assert result["sentiment"] == {"direction": "bullish", "confidence": 0.8}
    assert result["risk_factors"] == ["Rising input costs"]
    assert result["opportunities"] == ["New product line"]
    assert result["credibility_score"] == 0.9
    assert result["relevance_score"] == 0.85
    assert "fallback_used" not in result


def test_analyze_content_falls_back_on_invalid_json() -> None:
    llm = FakeChatModel(responder=lambda _messages: "not json at all")
    analyzer = ContentAnalyzer(llm)

    import asyncio

    result = asyncio.run(
        analyzer.analyze_content("strong growth and profit ahead", "aggressive")
    )

    assert result["fallback_used"] is True
    # Fallback keyword scan: "growth" and "profit" are both positive words.
    assert result["sentiment"]["direction"] == "bullish"


def test_analyze_content_fallback_scores_negative_keywords_bearish() -> None:
    llm = FakeChatModel(responder=lambda _messages: "not json")
    analyzer = ContentAnalyzer(llm)

    import asyncio

    result = asyncio.run(
        analyzer.analyze_content("decline and loss expected, major concern", "moderate")
    )

    assert result["fallback_used"] is True
    assert result["sentiment"]["direction"] == "bearish"


def test_persona_conditioning_appears_in_prompt() -> None:
    """Persona focus (risk_focus / time_horizon / keywords) must appear in
    the prompt the model actually receives -- pinned per this task's
    brief ("persona conditioning appears in the prompts the fake model
    receives")."""
    llm = FakeChatModel(responder=lambda _messages: _valid_analysis_json())
    analyzer = ContentAnalyzer(llm)

    import asyncio

    asyncio.run(analyzer.analyze_content("content", "conservative"))

    assert len(llm.captured_prompts) == 1
    messages = llm.captured_prompts[0]
    assert any(isinstance(m, SystemMessage) for m in messages)
    human = next(m for m in messages if isinstance(m, HumanMessage))
    assert "conservative investor" in human.content
    assert "downside protection" in human.content  # conservative's risk_focus
    assert "long-term" in human.content  # conservative's time_horizon
    assert "dividend" in human.content  # conservative's first keyword


def test_analyze_content_batch_preserves_order_and_tags_metadata() -> None:
    llm = FakeChatModel(responder=lambda _messages: _valid_analysis_json())
    analyzer = ContentAnalyzer(llm)

    items = [
        ("content one", "url-1"),
        ("content two", "url-2"),
        ("content three", "url-3"),
    ]

    import asyncio

    results = asyncio.run(analyzer.analyze_content_batch(items, "moderate"))

    assert [r["source_identifier"] for r in results] == ["url-1", "url-2", "url-3"]
    assert all(r["batch_processed"] for r in results)


def test_analyze_content_batch_uses_fallback_for_failed_items() -> None:
    def flaky_responder(messages: list) -> str:
        human = next(m.content for m in messages if isinstance(m, HumanMessage))
        if "bad" in human:
            raise RuntimeError("simulated LLM failure")
        return _valid_analysis_json()

    llm = FakeChatModel(responder=flaky_responder)
    analyzer = ContentAnalyzer(llm)

    import asyncio

    items = [("good content", "url-good"), ("bad content", "url-bad")]
    results = asyncio.run(analyzer.analyze_content_batch(items, "moderate"))

    by_id = {r["source_identifier"]: r for r in results}
    assert "fallback_used" not in by_id["url-good"]
    assert by_id["url-bad"]["fallback_used"] is True


def test_analyze_content_batch_empty_input_returns_empty() -> None:
    llm = FakeChatModel(responder=lambda _messages: _valid_analysis_json())
    analyzer = ContentAnalyzer(llm)

    import asyncio

    assert asyncio.run(analyzer.analyze_content_batch([], "moderate")) == []


@pytest.mark.parametrize("field", ["CREDIBILITY", "RELEVANCE", "confidence"])
@pytest.mark.parametrize("score", [0.8, "0.8", {"score": 0.8}, {"score": "0.8"}])
def test_analyze_content_normalizes_provider_scores_before_synthesis(field, score):
    """Synthetic object-score reproduction of the observed float-plus-dict error.

    The live provider response body was not captured; its exact JSON shape is
    unknown. This fixture exercises the inferred analyzer-to-synthesis path.
    """
    import asyncio

    from maverick.research.agents.synthesis import calculate_source_credibility

    analysis = json.loads(_valid_analysis_json())
    if field == "confidence":
        analysis["SENTIMENT"][field] = score
    else:
        analysis[field] = score
    analyzer = ContentAnalyzer(
        FakeChatModel(responder=lambda _messages: json.dumps(analysis))
    )
    result = asyncio.run(analyzer.analyze_content("some content", "moderate"))

    assert "fallback_used" not in result
    actual = {
        "CREDIBILITY": result["credibility_score"],
        "RELEVANCE": result["relevance_score"],
        "confidence": result["sentiment"]["confidence"],
    }[field]
    assert isinstance(actual, float)
    assert actual == 0.8
    credibility = calculate_source_credibility({"analysis": result})
    assert isinstance(credibility, float)
    assert 0 <= credibility <= 1


@pytest.mark.parametrize("field", ["CREDIBILITY", "RELEVANCE", "confidence"])
@pytest.mark.parametrize(
    "invalid_score",
    [None, True, [], {}, {"score": {}}, "high", -0.1, 1.1, float("nan"), float("inf")],
)
def test_analyze_content_uses_predictable_fallback_for_invalid_scores(
    field, invalid_score
):
    import asyncio

    analysis = json.loads(_valid_analysis_json())
    if field == "confidence":
        analysis["SENTIMENT"][field] = invalid_score
    else:
        analysis[field] = invalid_score
    analyzer = ContentAnalyzer(
        FakeChatModel(responder=lambda _messages: json.dumps(analysis))
    )
    result = asyncio.run(
        analyzer.analyze_content("strong growth and profit ahead", "moderate")
    )

    assert result["fallback_used"] is True
    assert result["credibility_score"] == 0.5
    assert result["sentiment"] == {"direction": "bullish", "confidence": 0.6}
    assert 0 <= result["relevance_score"] <= 1


def test_analyze_content_batch_isolates_malformed_scores():
    import asyncio

    invalid = json.loads(_valid_analysis_json())
    invalid["CREDIBILITY"] = {"score": "unavailable"}

    def responder(messages):
        human = next(m.content for m in messages if isinstance(m, HumanMessage))
        return json.dumps(invalid) if "bad source" in human else _valid_analysis_json()

    analyzer = ContentAnalyzer(FakeChatModel(responder=responder))
    results = asyncio.run(
        analyzer.analyze_content_batch(
            [("good source", "good"), ("bad source", "bad")], "moderate"
        )
    )

    assert [result["source_identifier"] for result in results] == ["good", "bad"]
    assert "fallback_used" not in results[0]
    assert results[1]["fallback_used"] is True
