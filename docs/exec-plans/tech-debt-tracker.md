# Tech debt tracker

One line per item. Remove the line in the same change that removes the debt.

| Item | Where | Phase to fix |
| --- | --- | --- |
| `is_auth_enabled` and `AUTH_ENABLED` survive in the provider config interface | `maverick_mcp/providers/` | cutover |
| `setup.py` duplicates hatchling and parses pyproject by hand | repo root | packaging |
| Wheel build uses `include = ["*.py"]` instead of explicit packages | `pyproject.toml` | packaging |
| `server.json` declares only remote transports and no package installs | repo root | distribution |
| Dockerfile is single-stage and ships build toolchain in the final image | `Dockerfile` | distribution |
| Two agent abstractions exist (`agents/` and `workflows/agents/`) | legacy tree | research port |
| Five LLM and search vendors are reachable from research paths | `providers/llm_factory.py` | research port |
| Default pytest filter deselects 664 tests; review the marker policy | `pyproject.toml` | cutover |
| MCP Apps chart rendering | new server | deferred |
| Tasks extension for long-running backtests | new server | deferred |
| `test_in_memory_server.py` hangs reading the `health://` resource via the in-memory client; quarantined `integration`, no root cause yet | `maverick_mcp/tests/` | cutover |
| `test_models_functional.py` fixture bypasses lazy schema creation and fails on a fresh CI database; needs a fixture rewrite; quarantined `integration` | `maverick_mcp/tests/` | cutover |
| `test_mcp_tool_fixes.py` is a vacuous duplicate of `test_mcp_tool_fixes_pytest.py`; deletion candidate | `maverick_mcp/tests/` | cutover |
| `application/commands/` and `application/screening/` are unimported by production code | `maverick_mcp/application/` | cutover |
| Two typecheckers disagree: CI gates on ty, make check runs pyright; retire one or document ty as the gate | `Makefile`, `.github/workflows/ci.yml` | maintainer decision |
