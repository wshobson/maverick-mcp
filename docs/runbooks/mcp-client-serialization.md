# MCP Client Argument Serialization

**Audience:** developers adding or debugging FastMCP tools in `maverick_mcp/api/routers/`.
**TL;DR:** Never declare `list[str]` on a `@mcp.tool()` function. Use `StrList` / `OptionalStrList` from `maverick_mcp.utils.mcp_types`.

## Why

FastMCP builds a Pydantic v2 `TypeAdapter` directly from each tool function's signature. Pydantic strict mode rejects type coercion by default. Some MCP clients do **not** send list arguments as JSON arrays — they JSON-stringify the array first and send a string. Claude Desktop via `mcp-remote` is the best-documented offender; some IDE integrations exhibit the same behavior.

**Failure shape:**

```
ValidationError: 1 validation error for call[<tool_name>]
<param>
  Input should be a valid list [type=list_type, input_value='["ANET","MRVL"]', input_type=str]
```

The giveaway is `input_type=str` with a value whose text starts with `[` — the client stringified the list.

## Fix

Use the shared coercion aliases:

```python
from maverick_mcp.utils.mcp_types import StrList, OptionalStrList

@mcp.tool()
def my_tool(
    symbols: StrList,                 # required list
    tags: OptionalStrList = None,     # optional list
):
    ...
```

Both aliases apply a `BeforeValidator` that accepts:

| Input | Result |
|---|---|
| `["A","B"]` (native list) | `["A", "B"]` (unchanged) |
| `'["A","B"]'` (JSON-encoded string) | `["A", "B"]` (parsed) |
| `"AAPL"` (bare scalar) | `["AAPL"]` (wrapped) |
| `None` | `None` (for `OptionalStrList` only) |

The JSON schema the client sees still advertises `array`, so well-behaved clients are unaffected — the coercion only runs when input arrives as a `str`.

## Enforcement

- **Static check**: `scripts/check_mcp_list_types.py` scans every router and fails on bare `list[str]`. Runs automatically via `make check` (CI and pre-push).
- **Contract tests**: `tests/test_mcp_list_coercion.py` pins the behavior of the aliases across all 6 known input shapes. Any regression in the `BeforeValidator` breaks CI.

## Adding New Coerced Types

If a new parameter needs the same tolerance for a different element type (`list[int]`, `list[float]`, etc.), extend `maverick_mcp/utils/mcp_types.py`:

```python
IntList = Annotated[list[int], BeforeValidator(_coerce_json_list)]
OptionalIntList = Annotated[list[int] | None, BeforeValidator(_coerce_json_list)]
```

The `_coerce_json_list` validator is element-type-agnostic — Pydantic's downstream validation handles per-element coercion (`"1"` → `1`). Add parametrized tests to `tests/test_mcp_list_coercion.py` for the new alias.

Then update `scripts/check_mcp_list_types.py` so it also flags bare `list[int]` etc. — the script currently only scans for `list[str]` because that was the only known vector. Extend `_is_list_str_annotation` or generalize it if you add multiple coerced types.

## Exempt Code

These are intentionally **not** scanned by the check script because they do not go through FastMCP's signature-based validation:

- **`screening_parallel.py`** — FastAPI router with Pydantic `BaseModel` request bodies. FastAPI deserializes JSON arrays natively.
- **`performance.py`** — FastAPI router with `Field`-annotated body models.
- **Any `dataclass` field** — not a function parameter.

If you convert one of those FastAPI endpoints into an MCP tool, remove it from the `FASTAPI_ONLY` set in `scripts/check_mcp_list_types.py`.

## History

- **2026-04-14** — `get_upcoming_catalysts` in `watchlist.py` failed with `list_type` error when Claude Desktop stringified `["ANET","MRVL"]`. Investigation revealed 10 other latent call sites and one pre-existing manual fix in `compare_strategies` (`backtesting.py`). Unified into the shared aliases. This runbook and the enforcement script were added to prevent recurrence.
