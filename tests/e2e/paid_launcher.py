"""Opt-in, budget-limited launcher for authorized live provider testing.

This test instrumentation retains real SDK/provider responses but bounds model
output, disables model retries and reserves cost before each request. It is
explicitly not the unmodified CLI. Keys only enter through the process env.
Pricing checked 2026-10-04: GPT-6 Luna standard input $0.10/M, output $0.50/M;
Exa auto search $0.007 plus $0.001 per text result. Conservative reservations
are $0.01/model call and $0.02/search; their combined cap is $0.90.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import exa_py
import langchain_openai

RESERVED = Decimal(os.environ.get("MAVERICK_E2E_SPEND_RESERVED", "0"))
COUNTS = {"model": 0, "search": 0}
OUTPUT = Path("paid-usage.jsonl")
OriginalModel = langchain_openai.ChatOpenAI
original_search = exa_py.AsyncExa.search


def record(**data):
    """Append reservation and provider-usage evidence without credentials."""
    with OUTPUT.open("a") as stream:
        stream.write(json.dumps(data, default=str) + "\n")


def reserve(kind):
    """Reserve conservative cost before a call and enforce the shared cap."""
    global RESERVED
    amount = Decimal("0.01" if kind == "model" else "0.02")
    if RESERVED + amount > Decimal("0.90"):
        raise RuntimeError("Authorized test budget exhausted before provider call")
    RESERVED += amount
    COUNTS[kind] += 1
    record(event="reserved", kind=kind, reserved_usd=RESERVED, counts=COUNTS.copy())


class BudgetModel(OriginalModel):
    def __init__(self, **kwargs):
        """Restrict the authorized model, output, retries, and timeout."""
        if kwargs.get("model") != "gpt-6-luna":
            raise ValueError("Paid test is authorized only for gpt-6-luna")
        kwargs.update(
            max_tokens=1024, max_retries=0, reasoning_effort="none", timeout=35
        )
        super().__init__(**kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        # UTF-8 byte length is a conservative token ceiling for these text prompts.
        """Bound input, reserve cost, and capture actual model usage."""
        if len(str(input).encode()) > 50_000:
            raise ValueError("Paid test input exceeds the bounded request size")
        reserve("model")
        result = await super().ainvoke(input, config=config, **kwargs)
        record(
            event="model_response",
            usage=result.usage_metadata,
            response_metadata=result.response_metadata,
            content=result.content,
        )
        return result


async def budget_search(self, query, **kwargs):
    """Bound search scope, reserve cost, and close the provider client."""
    if kwargs.get("type") != "auto" or not 0 < kwargs.get("num_results", 0) <= 10:
        raise ValueError("Paid test allows only auto search with at most 10 results")
    if kwargs.get("contents") != {"text": {"max_characters": 5000}}:
        raise ValueError("Paid test allows bounded text only")
    reserve("search")
    try:
        result = await original_search(self, query, **kwargs)
        record(
            event="search_response",
            query=query,
            results=len(result.results),
            cost=getattr(result, "cost_dollars", None),
        )
        return result
    finally:
        if self._client is not None:
            await self._client.aclose()


if __name__ == "__main__":
    langchain_openai.ChatOpenAI = BudgetModel  # ty: ignore[invalid-assignment]
    exa_py.AsyncExa.search = budget_search
    from maverick.server.app import main

    main()
