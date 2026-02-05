"""
LLM Optimization Configuration for Research Agents.

This module provides configuration settings and presets for different optimization scenarios
to prevent research agent timeouts while maintaining quality.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from maverick_mcp.providers.openrouter_provider import TaskType


class OptimizationMode(StrEnum):
    """Optimization modes for different use cases."""

    EMERGENCY = "emergency"  # <20s - Ultra-fast, minimal quality
    FAST = "fast"  # 20-60s - Fast with reasonable quality
    BALANCED = "balanced"  # 60-180s - Balance speed and quality
    COMPREHENSIVE = "comprehensive"  # 180s+ - Full quality, time permitting


class ResearchComplexity(StrEnum):
    """Research complexity levels."""

    SIMPLE = "simple"  # Basic queries, single focus
    MODERATE = "moderate"  # Multi-faceted analysis
    COMPLEX = "complex"  # Deep analysis, multiple dimensions
    EXPERT = "expert"  # Highly specialized, technical


@dataclass
class OptimizationPreset:
    """Configuration preset for optimization settings."""

    # Model Selection Settings
    prefer_fast: bool = True
    prefer_cheap: bool = True
    prefer_quality: bool = False

    # Token Budgeting
    max_input_tokens: int = 8000
    max_output_tokens: int = 2000
    emergency_reserve_tokens: int = 200

    # Time Management
    search_time_allocation_pct: float = 0.20  # 20% for search
    analysis_time_allocation_pct: float = 0.60  # 60% for analysis
    synthesis_time_allocation_pct: float = 0.20  # 20% for synthesis

    # Content Processing
    max_sources: int = 10
    max_content_length_per_source: int = 2000
    parallel_batch_size: int = 3

    # Early Termination
    target_confidence: float = 0.75
    min_sources_before_termination: int = 3
    diminishing_returns_threshold: float = 0.05
    consensus_threshold: float = 0.8

    # Quality vs Speed Trade-offs
    use_content_filtering: bool = True
    use_parallel_processing: bool = True
    use_early_termination: bool = True
    use_optimized_prompts: bool = True


class OptimizationPresets:
    """Predefined optimization presets for common scenarios."""

    EMERGENCY = OptimizationPreset(
        # Ultra-fast settings for <20 seconds
        prefer_fast=True,
        prefer_cheap=True,
        prefer_quality=False,
        max_input_tokens=2000,
        max_output_tokens=500,
        max_sources=3,
        max_content_length_per_source=800,
        parallel_batch_size=5,  # Aggressive batching
        target_confidence=0.6,  # Lower bar
        min_sources_before_termination=2,
        search_time_allocation_pct=0.15,
        analysis_time_allocation_pct=0.70,
        synthesis_time_allocation_pct=0.15,
    )

    FAST = OptimizationPreset(
        # Fast settings for 20-60 seconds
        prefer_fast=True,
        prefer_cheap=True,
        prefer_quality=False,
        max_input_tokens=4000,
        max_output_tokens=1000,
        max_sources=6,
        max_content_length_per_source=1200,
        parallel_batch_size=3,
        target_confidence=0.70,
        min_sources_before_termination=3,
    )

    BALANCED = OptimizationPreset(
        # Balanced settings for 60-180 seconds
        prefer_fast=False,
        prefer_cheap=True,
        prefer_quality=False,
        max_input_tokens=8000,
        max_output_tokens=2000,
        max_sources=10,
        max_content_length_per_source=2000,
        parallel_batch_size=2,
        target_confidence=0.75,
        min_sources_before_termination=3,
    )

    COMPREHENSIVE = OptimizationPreset(
        # Comprehensive settings for 180+ seconds
        prefer_fast=False,
        prefer_cheap=False,
        prefer_quality=True,
        max_input_tokens=12000,
        max_output_tokens=3000,
        max_sources=15,
        max_content_length_per_source=3000,
        parallel_batch_size=1,  # Less batching for quality
        target_confidence=0.80,
        min_sources_before_termination=5,
        use_early_termination=False,  # Allow full processing
        search_time_allocation_pct=0.25,
        analysis_time_allocation_pct=0.55,
        synthesis_time_allocation_pct=0.20,
    )

    @classmethod
    def get_preset(cls, mode: OptimizationMode) -> OptimizationPreset:
        """Get preset by optimization mode."""
        preset_map = {
            OptimizationMode.EMERGENCY: cls.EMERGENCY,
            OptimizationMode.FAST: cls.FAST,
            OptimizationMode.BALANCED: cls.BALANCED,
            OptimizationMode.COMPREHENSIVE: cls.COMPREHENSIVE,
        }
        return preset_map[mode]

    @classmethod
    def get_adaptive_preset(
        cls,
        time_budget_seconds: float,
        complexity: ResearchComplexity = ResearchComplexity.MODERATE,
        current_confidence: float = 0.0,
    ) -> OptimizationPreset:
        """Get adaptive preset based on time budget and complexity."""

        # Base mode selection by time
        if time_budget_seconds < 20:
            base_mode = OptimizationMode.EMERGENCY
        elif time_budget_seconds < 60:
            base_mode = OptimizationMode.FAST
        elif time_budget_seconds < 180:
            base_mode = OptimizationMode.BALANCED
        else:
            base_mode = OptimizationMode.COMPREHENSIVE

        # Get base preset
        preset = cls.get_preset(base_mode)

        # Adjust for complexity
        complexity_adjustments = {
            ResearchComplexity.SIMPLE: {
                "max_sources": int(preset.max_sources * 0.7),
                "target_confidence": preset.target_confidence - 0.1,
                "prefer_cheap": True,
            },
            ResearchComplexity.MODERATE: {
                # No adjustments - use base preset
            },
            ResearchComplexity.COMPLEX: {
                "max_sources": int(preset.max_sources * 1.3),
                "target_confidence": preset.target_confidence + 0.05,
                "max_input_tokens": int(preset.max_input_tokens * 1.2),
            },
            ResearchComplexity.EXPERT: {
                "max_sources": int(preset.max_sources * 1.5),
                "target_confidence": preset.target_confidence + 0.1,
                "max_input_tokens": int(preset.max_input_tokens * 1.4),
                "prefer_quality": True,
            },
        }

        # Apply complexity adjustments
        adjustments = complexity_adjustments.get(complexity, {})
        for key, value in adjustments.items():
            setattr(preset, key, value)

        # Adjust for current confidence
        if current_confidence > 0.6:
            # Already have good confidence, can be more aggressive with speed
            preset.target_confidence = max(preset.target_confidence - 0.1, 0.6)
            preset.max_sources = int(preset.max_sources * 0.8)
            preset.prefer_fast = True

        return preset


class ModelSelectionStrategy:
    """Strategies for model selection in different scenarios."""

    TIME_CRITICAL_MODELS = [
        "google/gemini-2.5-flash",  # 199 tokens/sec - FASTEST
        "openai/gpt-4o-mini",  # 126 tokens/sec - Most cost-effective
        "openai/gpt-5-nano",  # 180 tokens/sec - High speed
        "anthropic/claude-3.5-haiku",  # 65.6 tokens/sec - Fallback
    ]

    BALANCED_MODELS = [
        "google/gemini-2.5-flash",  # 199 tokens/sec - Speed-optimized
        "openai/gpt-4o-mini",  # 126 tokens/sec - Cost & speed balance
        "deepseek/deepseek-r1",  # 90+ tokens/sec - Good value
        "anthropic/claude-sonnet-4",  # High quality when needed
        "google/gemini-2.5-pro",  # Comprehensive analysis
        "openai/gpt-5",  # Fallback option
    ]

    QUALITY_MODELS = [
        "google/gemini-2.5-pro",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-sonnet-4",
    ]

    @classmethod
    def get_model_priority(
        cls,
        time_remaining: float,
        task_type: TaskType,
        complexity: ResearchComplexity = ResearchComplexity.MODERATE,
    ) -> list[str]:
        """Get prioritized model list for selection."""

        if time_remaining < 30:
            # Emergency mode: ultra-fast models for <30s timeouts (prioritize speed)
            return cls.TIME_CRITICAL_MODELS[:2]  # Use only the 2 fastest models
        elif time_remaining < 60:
            # Mix of fast and balanced models (speed-first approach)
            return cls.TIME_CRITICAL_MODELS[:3] + cls.BALANCED_MODELS[:2]
        elif complexity in [ResearchComplexity.COMPLEX, ResearchComplexity.EXPERT]:
            return cls.QUALITY_MODELS + cls.BALANCED_MODELS
        else:
            return cls.BALANCED_MODELS + cls.TIME_CRITICAL_MODELS


class PromptOptimizationSettings:
    """Settings for prompt optimization strategies."""

    # Template selection based on time constraints
    EMERGENCY_MAX_WORDS = {"content_analysis": 50, "synthesis": 40, "validation": 30}

    FAST_MAX_WORDS = {"content_analysis": 150, "synthesis": 200, "validation": 100}

    STANDARD_MAX_WORDS = {"content_analysis": 500, "synthesis": 800, "validation": 300}

    # Confidence-based prompt modifications
    HIGH_CONFIDENCE_ADDITIONS = [
        "Focus on validation and contradictory evidence since confidence is already high.",
        "Look for edge cases and potential risks that may have been missed.",
        "Verify consistency across sources and identify any conflicting information.",
    ]

    LOW_CONFIDENCE_ADDITIONS = [
        "Look for strong supporting evidence to build confidence in findings.",
        "Identify the most credible sources and weight them appropriately.",
        "Focus on consensus indicators and corroborating evidence.",
    ]

    @classmethod
    def get_word_limit(cls, prompt_type: str, time_remaining: float) -> int:
        """Get word limit for prompt type based on time remaining."""

        if time_remaining < 15:
            return cls.EMERGENCY_MAX_WORDS.get(prompt_type, 50)
        elif time_remaining < 45:
            return cls.FAST_MAX_WORDS.get(prompt_type, 150)
        else:
            return cls.STANDARD_MAX_WORDS.get(prompt_type, 500)

    @classmethod
    def get_confidence_instruction(cls, confidence_level: float) -> str:
        """Get confidence-based instruction addition."""

        if confidence_level > 0.7:
            import random

            return random.choice(cls.HIGH_CONFIDENCE_ADDITIONS)
        elif confidence_level < 0.4:
            import random

            return random.choice(cls.LOW_CONFIDENCE_ADDITIONS)
        else:
            return ""


class OptimizationConfig:
    """Main configuration class for LLM optimizations."""

    def __init__(
        self,
        mode: OptimizationMode = OptimizationMode.BALANCED,
        complexity: ResearchComplexity = ResearchComplexity.MODERATE,
        time_budget_seconds: float = 120.0,
        target_confidence: float = 0.75,
        custom_preset: OptimizationPreset | None = None,
    ):
        """Initialize optimization configuration.

        Args:
            mode: Optimization mode preset
            complexity: Research complexity level
            time_budget_seconds: Total time budget
            target_confidence: Target confidence threshold
            custom_preset: Custom preset overriding mode selection
        """
        self.mode = mode
        self.complexity = complexity
        self.time_budget_seconds = time_budget_seconds
        self.target_confidence = target_confidence

        # Get optimization preset
        if custom_preset:
            self.preset = custom_preset
        else:
            self.preset = OptimizationPresets.get_adaptive_preset(
                time_budget_seconds, complexity, 0.0
            )

        # Override target confidence if specified
        if target_confidence != 0.75:  # Non-default value
            self.preset.target_confidence = target_confidence

    def get_phase_time_budget(self, phase: str) -> float:
        """Get time budget for specific research phase."""

        allocation_map = {
            "search": self.preset.search_time_allocation_pct,
            "analysis": self.preset.analysis_time_allocation_pct,
            "synthesis": self.preset.synthesis_time_allocation_pct,
        }

        return self.time_budget_seconds * allocation_map.get(phase, 0.33)

    def should_use_optimization(self, optimization_name: str) -> bool:
        """Check if specific optimization should be used."""

        optimization_map = {
            "content_filtering": self.preset.use_content_filtering,
            "parallel_processing": self.preset.use_parallel_processing,
            "early_termination": self.preset.use_early_termination,
            "optimized_prompts": self.preset.use_optimized_prompts,
        }

        return optimization_map.get(optimization_name, True)

    def get_model_selection_params(self) -> dict[str, Any]:
        """Get model selection parameters."""

        return {
            "prefer_fast": self.preset.prefer_fast,
            "prefer_cheap": self.preset.prefer_cheap,
            "prefer_quality": self.preset.prefer_quality,
            "max_tokens": self.preset.max_output_tokens,
            "complexity": self.complexity,
        }

    def get_token_allocation_params(self) -> dict[str, Any]:
        """Get token allocation parameters."""

        return {
            "max_input_tokens": self.preset.max_input_tokens,
            "max_output_tokens": self.preset.max_output_tokens,
            "emergency_reserve": self.preset.emergency_reserve_tokens,
        }

    def get_content_filtering_params(self) -> dict[str, Any]:
        """Get content filtering parameters."""

        return {
            "max_sources": self.preset.max_sources,
            "max_content_length": self.preset.max_content_length_per_source,
            "enabled": self.preset.use_content_filtering,
        }

    def get_parallel_processing_params(self) -> dict[str, Any]:
        """Get parallel processing parameters."""

        return {
            "batch_size": self.preset.parallel_batch_size,
            "enabled": self.preset.use_parallel_processing,
        }

    def get_early_termination_params(self) -> dict[str, Any]:
        """Get early termination parameters."""

        return {
            "target_confidence": self.preset.target_confidence,
            "min_sources": self.preset.min_sources_before_termination,
            "diminishing_returns_threshold": self.preset.diminishing_returns_threshold,
            "consensus_threshold": self.preset.consensus_threshold,
            "enabled": self.preset.use_early_termination,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        return {
            "mode": self.mode.value,
            "complexity": self.complexity.value,
            "time_budget_seconds": self.time_budget_seconds,
            "target_confidence": self.target_confidence,
            "preset": {
                "prefer_fast": self.preset.prefer_fast,
                "prefer_cheap": self.preset.prefer_cheap,
                "prefer_quality": self.preset.prefer_quality,
                "max_input_tokens": self.preset.max_input_tokens,
                "max_output_tokens": self.preset.max_output_tokens,
                "max_sources": self.preset.max_sources,
                "parallel_batch_size": self.preset.parallel_batch_size,
                "target_confidence": self.preset.target_confidence,
                "optimizations_enabled": {
                    "content_filtering": self.preset.use_content_filtering,
                    "parallel_processing": self.preset.use_parallel_processing,
                    "early_termination": self.preset.use_early_termination,
                    "optimized_prompts": self.preset.use_optimized_prompts,
                },
            },
        }


# Convenience functions for common configurations


def create_emergency_config(time_budget: float = 15.0) -> OptimizationConfig:
    """Create emergency optimization configuration."""
    return OptimizationConfig(
        mode=OptimizationMode.EMERGENCY,
        time_budget_seconds=time_budget,
        target_confidence=0.6,
    )


def create_fast_config(time_budget: float = 45.0) -> OptimizationConfig:
    """Create fast optimization configuration."""
    return OptimizationConfig(
        mode=OptimizationMode.FAST,
        time_budget_seconds=time_budget,
        target_confidence=0.7,
    )


def create_balanced_config(time_budget: float = 120.0) -> OptimizationConfig:
    """Create balanced optimization configuration."""
    return OptimizationConfig(
        mode=OptimizationMode.BALANCED,
        time_budget_seconds=time_budget,
        target_confidence=0.75,
    )


def create_comprehensive_config(time_budget: float = 300.0) -> OptimizationConfig:
    """Create comprehensive optimization configuration."""
    return OptimizationConfig(
        mode=OptimizationMode.COMPREHENSIVE,
        time_budget_seconds=time_budget,
        target_confidence=0.8,
    )


def create_adaptive_config(
    time_budget_seconds: float,
    complexity: ResearchComplexity = ResearchComplexity.MODERATE,
    current_confidence: float = 0.0,
) -> OptimizationConfig:
    """Create adaptive configuration based on runtime parameters."""

    # Auto-select mode based on time budget
    if time_budget_seconds < 20:
        mode = OptimizationMode.EMERGENCY
    elif time_budget_seconds < 60:
        mode = OptimizationMode.FAST
    elif time_budget_seconds < 180:
        mode = OptimizationMode.BALANCED
    else:
        mode = OptimizationMode.COMPREHENSIVE

    return OptimizationConfig(
        mode=mode,
        complexity=complexity,
        time_budget_seconds=time_budget_seconds,
        target_confidence=0.75 - (0.15 if current_confidence > 0.6 else 0),
    )
