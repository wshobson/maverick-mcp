"""Focused submodules extracted from `maverick_mcp/utils/llm_optimization.py`.

The re-export at this level (and in the legacy `llm_optimization` shim)
preserves the public import surface so callers can migrate gradually:

    from maverick_mcp.utils.llm import AdaptiveModelSelector
    # ...is equivalent to...
    from maverick_mcp.utils.llm_optimization import AdaptiveModelSelector
"""

from maverick_mcp.utils.llm.confidence import ConfidenceTracker
from maverick_mcp.utils.llm.content_filter import IntelligentContentFilter
from maverick_mcp.utils.llm.model_selection import (
    AdaptiveModelSelector,
    ModelConfiguration,
)
from maverick_mcp.utils.llm.parallel import ParallelLLMProcessor
from maverick_mcp.utils.llm.prompts import OptimizedPromptEngine
from maverick_mcp.utils.llm.token_budgeter import (
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
