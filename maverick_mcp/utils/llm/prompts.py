"""Time- and confidence-adaptive prompt template engine."""

from __future__ import annotations


class OptimizedPromptEngine:
    """Creates optimized prompts for different time constraints and confidence levels."""

    def __init__(self):
        self.prompt_cache = {}  # Cache for generated prompts

        self.prompt_templates = {
            "emergency": {
                "content_analysis": """URGENT: Quick 3-point analysis of financial content for {persona} investor.

Content: {content}

Provide ONLY:
1. SENTIMENT: Bull/Bear/Neutral + confidence (0-1)
2. KEY_RISK: Primary risk factor
3. KEY_OPPORTUNITY: Main opportunity (if any)

Format: SENTIMENT:Bull|0.8 KEY_RISK:Market volatility KEY_OPPORTUNITY:Earnings growth
Max 50 words total. No explanations.""",
                "synthesis": """URGENT: 2-sentence summary from {source_count} sources for {persona} investor.

Key findings: {key_points}

Provide: 1) Overall sentiment direction 2) Primary investment implication
Max 40 words total.""",
            },
            "fast": {
                "content_analysis": """Quick financial analysis for {persona} investor - 5 points max.

Content: {content}

Provide concisely:
• Sentiment: Bull/Bear/Neutral (confidence 0-1)
• Key insight (1 sentence)
• Main risk (1 sentence)
• Main opportunity (1 sentence)
• Relevance score (0-1)

Target: Under 150 words total.""",
                "synthesis": """Synthesize research findings for {persona} investor.

Sources: {source_count} | Key insights: {insights}

4-part summary:
1. Overall sentiment + confidence
2. Top 2 opportunities
3. Top 2 risks
4. Recommended action

Limit: 200 words max.""",
            },
            "standard": {
                "content_analysis": """Financial content analysis for {persona} investor.

Content: {content}
Focus areas: {focus_areas}

Structured analysis:
- Sentiment (direction, confidence 0-1, brief reasoning)
- Key insights (3-5 bullet points)
- Risk factors (2-3 main risks)
- Opportunities (2-3 opportunities)
- Credibility assessment (0-1)
- Relevance score (0-1)

Target: 300-500 words.""",
                "synthesis": """Comprehensive research synthesis for {persona} investor.

Research Summary:
- Sources analyzed: {source_count}
- Key insights: {insights}
- Time horizon: {time_horizon}

Provide detailed analysis:
1. Executive Summary (2-3 sentences)
2. Key Findings (5-7 bullet points)
3. Investment Implications
4. Risk Assessment
5. Recommended Actions
6. Confidence Level + reasoning

Tailor specifically for {persona} investment characteristics.""",
            },
        }

    def get_optimized_prompt(
        self,
        prompt_type: str,
        time_remaining: float,
        confidence_level: float,
        **context,
    ) -> str:
        """Generate optimized prompt based on time constraints and confidence."""

        # Create cache key
        cache_key = f"{prompt_type}_{time_remaining:.0f}_{confidence_level:.1f}_{hash(str(sorted(context.items())))}"

        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        # Select template based on time pressure
        if time_remaining < 15:
            template_category = "emergency"
        elif time_remaining < 45:
            template_category = "fast"
        else:
            template_category = "standard"

        template = self.prompt_templates[template_category].get(prompt_type)

        if not template:
            # Fallback to fast template
            template = self.prompt_templates["fast"].get(
                prompt_type, "Analyze the content quickly and provide key insights."
            )

        # Add confidence-based instructions
        confidence_instructions = ""
        if confidence_level > 0.7:
            confidence_instructions = "\n\nNOTE: High confidence already achieved. Focus on validation and contradictory evidence."
        elif confidence_level < 0.4:
            confidence_instructions = "\n\nNOTE: Low confidence. Look for strong supporting evidence to build confidence."

        # Format template with context
        formatted_prompt = template.format(**context) + confidence_instructions

        # Cache the result
        self.prompt_cache[cache_key] = formatted_prompt

        return formatted_prompt

    def create_time_optimized_synthesis_prompt(
        self,
        sources: list[dict],
        persona: str,
        time_remaining: float,
        current_confidence: float,
    ) -> str:
        """Create synthesis prompt optimized for available time."""

        # Extract key information from sources
        insights = []
        sentiments = []
        for source in sources:
            analysis = source.get("analysis", {})
            insights.extend(analysis.get("insights", [])[:2])  # Limit per source
            sentiment = analysis.get("sentiment", {})
            if sentiment:
                sentiments.append(sentiment.get("direction", "neutral"))

        # Prepare context
        context = {
            "persona": persona,
            "source_count": len(sources),
            "insights": "; ".join(insights[:8]),  # Top 8 insights
            "key_points": "; ".join(insights[:8]),  # For backward compatibility
            "time_horizon": "short-term" if time_remaining < 30 else "medium-term",
        }

        return self.get_optimized_prompt(
            "synthesis", time_remaining, current_confidence, **context
        )
