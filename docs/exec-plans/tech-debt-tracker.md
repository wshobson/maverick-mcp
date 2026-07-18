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
