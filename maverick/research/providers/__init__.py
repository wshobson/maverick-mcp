"""Web search providers (`WebSearchProvider`, `ExaSearchProvider`). Third-layer sibling: imports config and types.

No parallel circuit-breaker manager lives here: `exa.py`'s module docstring
records the comparison against `maverick.platform.http`'s `CircuitBreaker`/
`get_breaker` and why the platform breaker is used directly instead of
porting a research-scoped one.

This file imports nothing itself (no re-exports), by design: it is
imported whenever any sibling submodule is (ordinary Python package
semantics), including transitively by the service tier, so it must stay
importable on a base install with no `research` extra. `exa.py`'s `exa_py`
import is lazy for the same reason -- see that module's docstring.
"""
