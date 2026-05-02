"""Pre-filter and prioritize content sources before LLM processing."""

from __future__ import annotations

import re
from datetime import datetime


class IntelligentContentFilter:
    """Pre-filters and prioritizes content to reduce LLM processing overhead."""

    def __init__(self):
        self.relevance_keywords = {
            "fundamental": {
                "high": [
                    "earnings",
                    "revenue",
                    "profit",
                    "ebitda",
                    "cash flow",
                    "debt",
                    "valuation",
                ],
                "medium": [
                    "balance sheet",
                    "income statement",
                    "financial",
                    "quarterly",
                    "annual",
                ],
                "context": ["company", "business", "financial results", "guidance"],
            },
            "technical": {
                "high": [
                    "price",
                    "chart",
                    "trend",
                    "support",
                    "resistance",
                    "breakout",
                ],
                "medium": ["volume", "rsi", "macd", "moving average", "pattern"],
                "context": ["technical analysis", "trading", "momentum"],
            },
            "sentiment": {
                "high": ["rating", "upgrade", "downgrade", "buy", "sell", "hold"],
                "medium": ["analyst", "recommendation", "target price", "outlook"],
                "context": ["opinion", "sentiment", "market mood"],
            },
            "competitive": {
                "high": [
                    "market share",
                    "competitor",
                    "competitive advantage",
                    "industry",
                ],
                "medium": ["peer", "comparison", "market position", "sector"],
                "context": ["competitive landscape", "industry analysis"],
            },
        }

        self.domain_credibility_scores = {
            "reuters.com": 0.95,
            "bloomberg.com": 0.95,
            "wsj.com": 0.90,
            "ft.com": 0.90,
            "marketwatch.com": 0.85,
            "cnbc.com": 0.80,
            "yahoo.com": 0.75,
            "seekingalpha.com": 0.80,
            "fool.com": 0.70,
            "investing.com": 0.75,
        }

    async def filter_and_prioritize_sources(
        self,
        sources: list[dict],
        research_focus: str,
        time_budget: float,
        target_source_count: int | None = None,
        current_confidence: float = 0.0,
    ) -> list[dict]:
        """Filter and prioritize sources based on relevance, quality, and time constraints."""

        if not sources:
            return []

        # Determine target count based on time budget and confidence
        if target_source_count is None:
            target_source_count = self._calculate_optimal_source_count(
                time_budget, current_confidence, len(sources)
            )

        # Quick relevance scoring without LLM
        scored_sources = []
        for source in sources:
            relevance_score = self._calculate_relevance_score(source, research_focus)
            credibility_score = self._get_source_credibility(source)
            recency_score = self._calculate_recency_score(source.get("published_date"))

            # Combined score with weights
            combined_score = (
                relevance_score * 0.5 + credibility_score * 0.3 + recency_score * 0.2
            )

            if combined_score > 0.3:  # Relevance threshold
                scored_sources.append((combined_score, source))

        # Sort by combined score
        scored_sources.sort(key=lambda x: x[0], reverse=True)

        # Select diverse sources
        selected_sources = self._select_diverse_sources(
            scored_sources, target_source_count, research_focus
        )

        # Pre-process content for faster LLM processing
        processed_sources = []
        for score, source in selected_sources:
            processed_source = self._preprocess_content(
                source, research_focus, time_budget
            )
            processed_source["relevance_score"] = score
            processed_sources.append(processed_source)

        return processed_sources

    def _calculate_optimal_source_count(
        self, time_budget: float, current_confidence: float, available_sources: int
    ) -> int:
        """Calculate optimal number of sources to process given constraints."""

        # Base count from time budget
        if time_budget < 20:
            base_count = 3
        elif time_budget < 40:
            base_count = 6
        elif time_budget < 80:
            base_count = 10
        else:
            base_count = 15

        # Adjust for confidence level
        if current_confidence > 0.7:
            # High confidence - fewer sources needed
            confidence_multiplier = 0.7
        elif current_confidence < 0.4:
            # Low confidence - more sources helpful
            confidence_multiplier = 1.2
        else:
            confidence_multiplier = 1.0

        # Final calculation
        target_count = int(base_count * confidence_multiplier)

        # Ensure we don't exceed available sources
        return min(target_count, available_sources, 20)  # Cap at 20

    def _calculate_relevance_score(self, source: dict, research_focus: str) -> float:
        """Calculate relevance score using keyword matching and heuristics."""

        content = source.get("content", "").lower()
        title = source.get("title", "").lower()

        if not content and not title:
            return 0.0

        focus_keywords = self.relevance_keywords.get(research_focus, {})

        # High-value keywords
        high_keywords = focus_keywords.get("high", [])
        high_score = sum(1 for keyword in high_keywords if keyword in content) / max(
            len(high_keywords), 1
        )

        # Medium-value keywords
        medium_keywords = focus_keywords.get("medium", [])
        medium_score = sum(
            1 for keyword in medium_keywords if keyword in content
        ) / max(len(medium_keywords), 1)

        # Context keywords
        context_keywords = focus_keywords.get("context", [])
        context_score = sum(
            1 for keyword in context_keywords if keyword in content
        ) / max(len(context_keywords), 1)

        # Title relevance (titles are more focused)
        title_high_score = sum(
            1 for keyword in high_keywords if keyword in title
        ) / max(len(high_keywords), 1)

        # Combine scores with weights
        relevance_score = (
            high_score * 0.4
            + medium_score * 0.25
            + context_score * 0.15
            + title_high_score * 0.2
        )

        # Boost for very relevant titles
        if any(keyword in title for keyword in high_keywords):
            relevance_score *= 1.2

        return min(relevance_score, 1.0)

    def _get_source_credibility(self, source: dict) -> float:
        """Calculate source credibility based on domain and other factors."""

        url = source.get("url", "").lower()

        # Domain-based credibility
        domain_score = 0.5  # Default
        for domain, score in self.domain_credibility_scores.items():
            if domain in url:
                domain_score = score
                break

        # Boost for specific high-quality indicators
        if any(indicator in url for indicator in [".gov", ".edu", "sec.gov"]):
            domain_score = min(domain_score + 0.2, 1.0)

        # Penalty for low-quality indicators
        if any(indicator in url for indicator in ["blog", "forum", "reddit"]):
            domain_score *= 0.8

        return domain_score

    def _calculate_recency_score(self, published_date: str) -> float:
        """Calculate recency score based on publication date."""

        if not published_date:
            return 0.5  # Default for unknown dates

        try:
            # Parse date (handle various formats)
            if "T" in published_date:
                pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            else:
                pub_date = datetime.strptime(published_date, "%Y-%m-%d")

            # Calculate days old
            days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days

            # Scoring based on age
            if days_old <= 1:
                return 1.0  # Very recent
            elif days_old <= 7:
                return 0.9  # Recent
            elif days_old <= 30:
                return 0.7  # Fairly recent
            elif days_old <= 90:
                return 0.5  # Moderately old
            else:
                return 0.3  # Old

        except (ValueError, TypeError):
            return 0.5  # Default for unparseable dates

    def _select_diverse_sources(
        self,
        scored_sources: list[tuple[float, dict]],
        target_count: int,
        research_focus: str,
    ) -> list[tuple[float, dict]]:
        """Select diverse sources to avoid redundancy."""

        if len(scored_sources) <= target_count:
            return scored_sources

        selected = []
        used_domains = set()

        # First pass: select high-scoring diverse sources
        for score, source in scored_sources:
            if len(selected) >= target_count:
                break

            url = source.get("url", "")
            domain = self._extract_domain(url)

            # Ensure diversity by domain (max 2 from same domain initially)
            domain_count = sum(
                1
                for _, s in selected
                if self._extract_domain(s.get("url", "")) == domain
            )

            if domain_count < 2 or len(selected) < target_count // 2:
                selected.append((score, source))
                used_domains.add(domain)

        # Second pass: fill remaining slots with best remaining sources
        remaining_needed = target_count - len(selected)
        if remaining_needed > 0:
            remaining_sources = scored_sources[len(selected) :]
            selected.extend(remaining_sources[:remaining_needed])

        return selected[:target_count]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            if "//" in url:
                domain = url.split("//")[1].split("/")[0]
                return domain.replace("www.", "")
            return url
        except Exception:
            return url

    def _preprocess_content(
        self, source: dict, research_focus: str, time_budget: float
    ) -> dict:
        """Pre-process content to optimize for LLM analysis."""

        content = source.get("content", "")
        if not content:
            return source

        # Determine content length limit based on time budget
        if time_budget < 30:
            max_length = 800  # Emergency mode
        elif time_budget < 60:
            max_length = 1200  # Fast mode
        else:
            max_length = 2000  # Standard mode

        # If content is already short enough, return as-is
        if len(content) <= max_length:
            source_copy = source.copy()
            source_copy["original_length"] = len(content)
            source_copy["filtered"] = False
            return source_copy

        # Extract most relevant sentences/paragraphs
        sentences = re.split(r"[.!?]+", content)
        focus_keywords = self.relevance_keywords.get(research_focus, {})
        all_keywords = (
            focus_keywords.get("high", [])
            + focus_keywords.get("medium", [])
            + focus_keywords.get("context", [])
        )

        # Score sentences by keyword relevance
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue

            sentence_lower = sentence.lower()
            keyword_count = sum(
                1 for keyword in all_keywords if keyword in sentence_lower
            )

            # Boost for financial numbers and percentages
            has_numbers = bool(re.search(r"\$?[\d,]+\.?\d*[%kmbKMB]?", sentence))
            number_boost = 0.5 if has_numbers else 0

            sentence_score = keyword_count + number_boost
            if sentence_score > 0:
                scored_sentences.append((sentence_score, sentence.strip()))

        # Sort by relevance and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Build filtered content
        filtered_content = ""
        for _score, sentence in scored_sentences:
            if len(filtered_content) + len(sentence) > max_length:
                break
            filtered_content += sentence + ". "

        # If no relevant sentences found, take first part of original content
        if not filtered_content:
            filtered_content = content[:max_length]

        # Create processed source
        source_copy = source.copy()
        source_copy["content"] = filtered_content.strip()
        source_copy["original_length"] = len(content)
        source_copy["filtered_length"] = len(filtered_content)
        source_copy["filtered"] = True
        source_copy["compression_ratio"] = len(filtered_content) / len(content)

        return source_copy
