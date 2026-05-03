"""Backward-compatible re-exports for the LLM optimization helpers.

The implementation moved to the focused submodules in
``maverick_mcp/utils/llm/``. Existing import paths
(``from maverick_mcp.utils.llm_optimization import ...``) continue to
work unchanged via this shim. New code should prefer importing from
``maverick_mcp.utils.llm`` directly.
"""

from __future__ import annotations

from maverick_mcp.utils.llm import (
    AdaptiveModelSelector,
    ConfidenceTracker,
    IntelligentContentFilter,
    ModelConfiguration,
    OptimizedPromptEngine,
    ParallelLLMProcessor,
    ProgressiveTokenBudgeter,
    ResearchPhase,
    TokenAllocation,
)

__all__ = [
    "AdaptiveModelSelector",
    "ConfidenceTracker",
    "IntelligentContentFilter",
    "ModelConfiguration",
    "OptimizedPromptEngine",
    "ParallelLLMProcessor",
    "ProgressiveTokenBudgeter",
    "ResearchPhase",
    "TokenAllocation",
]
