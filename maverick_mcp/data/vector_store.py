"""
Research vector store for semantic caching and retrieval.

Provides embedded vector storage using ChromaDB (optional) for:
- Storing research results with metadata for later retrieval
- Semantic similarity search with temporal decay scoring
- Content deduplication to avoid redundant research

ChromaDB is an optional dependency. The system gracefully falls back
to no caching when ChromaDB is not installed.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from maverick_mcp.config.settings import get_settings

logger = logging.getLogger("maverick_mcp.data.vector_store")

# Conditional ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment, misc]

# Collection name for research documents
_COLLECTION_NAME = "research_documents"


@dataclass
class ResearchResult:
    """A single research result retrieved from the vector store."""

    content: str
    ticker: str
    topic: str
    source_url: str
    source_date: datetime
    credibility_score: float
    similarity_score: float = 0.0
    temporal_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class _StoreStats:
    """Internal bookkeeping for store operations."""

    total_stored: int = 0
    total_searches: int = 0
    total_dedup_checks: int = 0
    cache_hits: int = 0


class ResearchVectorStore:
    """
    Lightweight vector store for research content using ChromaDB.

    If ChromaDB is not available, all operations gracefully degrade:
    - ``store_research`` becomes a no-op
    - ``search_similar`` returns an empty list
    - ``deduplicate`` returns False (content is treated as new)
    """

    def __init__(
        self,
        storage_path: str | None = None,
        half_life_days: int | None = None,
    ) -> None:
        settings = get_settings()
        vs_settings = settings.vector_store

        self._storage_path = storage_path or vs_settings.storage_path
        self._half_life_days = half_life_days or vs_settings.temporal_half_life_days
        self._enabled = vs_settings.enabled
        self._stats = _StoreStats()

        # Eagerly initialise if available
        self._client: Any | None = None
        self._collection: Any | None = None

        if not self._enabled:
            logger.info("Research vector store disabled via configuration")
            return

        if not CHROMADB_AVAILABLE:
            logger.info(
                "ChromaDB not installed -- research vector store disabled. "
                "Install with: pip install 'maverick_mcp[vectors]'"
            )
            return

        try:
            self._init_chromadb()
        except Exception:
            logger.exception("Failed to initialise ChromaDB -- vector store disabled")
            self._client = None
            self._collection = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_chromadb(self) -> None:
        """Create or open a persistent ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            return

        # Ensure storage directory exists
        os.makedirs(self._storage_path, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._storage_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB vector store initialised at %s (%d documents)",
            self._storage_path,
            self._collection.count(),
        )

    @property
    def is_available(self) -> bool:
        """Return True when the backing store is ready for use."""
        return self._collection is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_research(
        self,
        ticker: str,
        topic: str,
        content: str,
        source_url: str,
        source_date: datetime,
        credibility_score: float,
    ) -> bool:
        """
        Embed and store a research document.

        Returns True when the document was successfully stored, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            doc_id = self._make_id(source_url, content)

            # Build metadata (ChromaDB stores flat key/value pairs)
            metadata: dict[str, str | int | float | bool] = {
                "ticker": ticker.upper(),
                "topic": topic,
                "source_url": source_url,
                "source_date_iso": source_date.isoformat(),
                "source_date_ts": source_date.timestamp(),
                "credibility_score": float(credibility_score),
                "stored_at_iso": datetime.now(UTC).isoformat(),
            }

            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata],
            )

            self._stats.total_stored += 1
            logger.debug(
                "Stored research doc %s for %s (topic=%s)", doc_id[:12], ticker, topic
            )
            return True

        except Exception:
            logger.exception("Failed to store research document")
            return False

    def search_similar(
        self,
        query: str,
        ticker: str | None = None,
        top_k: int = 5,
        max_age_days: int | None = None,
    ) -> list[ResearchResult]:
        """
        Semantic search with optional ticker filter and temporal decay.

        Results are ranked by ``combined_score = similarity * temporal_decay``.
        """
        if not self.is_available:
            return []

        settings = get_settings()
        if max_age_days is None:
            max_age_days = settings.vector_store.max_age_days

        self._stats.total_searches += 1

        try:
            # Build optional where-filter
            where_filter = self._build_where_filter(ticker, max_age_days)

            # Query ChromaDB -- request more than top_k because temporal
            # re-ranking may shuffle the order.
            query_limit = min(top_k * 3, 100)

            query_kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": query_limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            results = self._collection.query(**query_kwargs)

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            # Convert and score
            research_results = self._process_query_results(results)

            # Sort by combined score (descending) and truncate
            research_results.sort(key=lambda r: r.combined_score, reverse=True)
            top_results = research_results[:top_k]

            if top_results:
                self._stats.cache_hits += 1
                logger.debug(
                    "Vector search returned %d results (best score=%.3f)",
                    len(top_results),
                    top_results[0].combined_score,
                )

            return top_results

        except Exception:
            logger.exception("Vector search failed")
            return []

    def deduplicate(
        self,
        content: str,
        ticker: str,
        threshold: float | None = None,
    ) -> bool:
        """
        Check whether substantially similar content already exists.

        Returns True when a duplicate (similarity >= threshold) is found.
        """
        if not self.is_available:
            return False

        settings = get_settings()
        if threshold is None:
            threshold = settings.vector_store.dedup_threshold

        self._stats.total_dedup_checks += 1

        try:
            results = self._collection.query(
                query_texts=[content],
                n_results=1,
                where={"ticker": ticker.upper()},
                include=["distances"],
            )

            if (
                not results
                or not results.get("distances")
                or not results["distances"][0]
            ):
                return False

            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity: similarity = 1 - (distance / 2)
            distance = results["distances"][0][0]
            similarity = 1.0 - (distance / 2.0)

            is_duplicate = similarity >= threshold
            if is_duplicate:
                logger.debug(
                    "Duplicate detected for %s (similarity=%.3f >= %.3f)",
                    ticker,
                    similarity,
                    threshold,
                )
            return is_duplicate

        except Exception:
            logger.exception("Deduplication check failed")
            return False

    def get_stats(self) -> dict[str, int]:
        """Return basic usage statistics."""
        stats = {
            "total_stored": self._stats.total_stored,
            "total_searches": self._stats.total_searches,
            "total_dedup_checks": self._stats.total_dedup_checks,
            "cache_hits": self._stats.cache_hits,
            "chromadb_available": CHROMADB_AVAILABLE,
            "store_enabled": self._enabled,
            "store_ready": self.is_available,
        }
        if self.is_available:
            try:
                stats["document_count"] = self._collection.count()
            except Exception:
                stats["document_count"] = -1
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(source_url: str, content: str) -> str:
        """Deterministic document ID from URL + content hash."""
        raw = f"{source_url}:{hashlib.sha256(content.encode()).hexdigest()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _build_where_filter(
        self,
        ticker: str | None,
        max_age_days: int,
    ) -> dict[str, Any] | None:
        """Build a ChromaDB ``where`` filter dict."""
        conditions: list[dict[str, Any]] = []

        if ticker:
            conditions.append({"ticker": {"$eq": ticker.upper()}})

        # Filter by max age
        cutoff_ts = datetime.now(UTC).timestamp() - max_age_days * 86400
        conditions.append({"source_date_ts": {"$gte": cutoff_ts}})

        if len(conditions) == 1:
            return conditions[0]
        if len(conditions) > 1:
            return {"$and": conditions}
        return None

    def _temporal_decay(self, source_date_ts: float) -> float:
        """
        Calculate temporal decay using exponential decay.

        ``score = exp(-days_old / half_life)``
        """
        now_ts = datetime.now(UTC).timestamp()
        days_old = max((now_ts - source_date_ts) / 86400.0, 0.0)
        return math.exp(-days_old / self._half_life_days)

    def _process_query_results(
        self,
        results: dict[str, Any],
    ) -> list[ResearchResult]:
        """Convert raw ChromaDB query output into scored ResearchResult objects."""
        research_results: list[ResearchResult] = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, _doc_id in enumerate(ids):
            meta = metadatas[i]
            content = documents[i] or ""

            # Cosine similarity from distance
            distance = distances[i]
            similarity = 1.0 - (distance / 2.0)

            # Temporal decay
            source_date_ts = meta.get("source_date_ts", 0.0)
            temporal_score = self._temporal_decay(source_date_ts)

            # Combined score
            combined = similarity * temporal_score

            # Parse source_date back to datetime
            try:
                source_date = datetime.fromisoformat(meta.get("source_date_iso", ""))
            except (ValueError, TypeError):
                source_date = datetime.fromtimestamp(source_date_ts, tz=UTC)

            research_results.append(
                ResearchResult(
                    content=content,
                    ticker=meta.get("ticker", ""),
                    topic=meta.get("topic", ""),
                    source_url=meta.get("source_url", ""),
                    source_date=source_date,
                    credibility_score=meta.get("credibility_score", 0.0),
                    similarity_score=similarity,
                    temporal_score=temporal_score,
                    combined_score=combined,
                )
            )

        return research_results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_store: ResearchVectorStore | None = None


def get_vector_store() -> ResearchVectorStore:
    """
    Return (and lazily create) the module-level ResearchVectorStore singleton.

    Safe to call even when ChromaDB is not installed -- the returned store
    will simply no-op on all operations.
    """
    global _global_store
    if _global_store is None:
        _global_store = ResearchVectorStore()
    return _global_store
