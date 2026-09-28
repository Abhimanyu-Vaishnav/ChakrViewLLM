"""
Persistent Memory Retriever for ChakrView (Step 16).

Combines:
- Lexical matching (token overlap)
- Semantic vector similarity (via Step 14 NeuralSemanticEmbeddingProvider or reference embedder)
- Recency decay weighting
- Importance & Confidence weighting
- Strict temporal validity and owner isolation

Integrates cleanly with Step 13/14 UnifiedRetriever and Step 15 CognitiveController.
"""

from dataclasses import dataclass, field
import math
import re
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.memory.record import MemoryRecord, MemoryType, MemoryValidity
from chakrview.memory.store import MemoryStore
from chakrview.memory.deduplication import MemoryDeduplicator
from chakrview.runtime.retrieval import RetrievalCandidate, RetrievalSourceType


@dataclass
class MemoryRetrievalCandidate:
    """
    Structured outcome of a persistent memory search.
    """
    memory_id: str
    content: str
    memory_type: MemoryType
    lexical_score: float
    semantic_score: float
    recency_score: float
    importance: float
    confidence: float
    combined_score: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_retrieval_candidate(self) -> RetrievalCandidate:
        """Convert to runtime RetrievalCandidate for UnifiedRetriever integration."""
        return RetrievalCandidate(
            candidate_id=self.memory_id,
            text=self.content,
            source_type=RetrievalSourceType.MEMORY,
            source_id=self.provenance.get("source_id") or self.memory_id,
            score=self.combined_score,
            retrieval_method="hybrid" if self.semantic_score > 0 else "lexical",
            metadata={
                "memory_id": self.memory_id,
                "memory_type": self.memory_type.value,
                "lexical_score": self.lexical_score,
                "semantic_score": self.semantic_score,
                "importance": self.importance,
                "confidence": self.confidence,
            },
            provenance={
                "memory_id": self.memory_id,
                "memory_type": self.memory_type.value,
                "importance": self.importance,
                "confidence": self.confidence,
                **self.provenance,
            },
        )


class PersistentMemoryRetriever:
    """
    Multi-faceted memory retriever honoring privacy, temporal bounds, and weighted scoring.
    """

    def __init__(
        self,
        embedding_provider: Optional[Any] = None,
        lexical_weight: float = 0.35,
        semantic_weight: float = 0.35,
        recency_weight: float = 0.10,
        importance_weight: float = 0.10,
        confidence_weight: float = 0.10,
        half_life_days: float = 30.0,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.confidence_weight = confidence_weight
        self.half_life_seconds = half_life_days * 86400.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        clean = text.lower().replace("_", " ")
        return re.findall(r"\b[a-zA-Z0-9]{2,}\b", clean)

    def _compute_lexical_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        q_set = set(query_tokens)
        d_set = set(doc_tokens)
        intersection = q_set.intersection(d_set)
        if not intersection:
            return 0.0
        # Jaccard + query coverage combination
        jaccard = len(intersection) / len(q_set.union(d_set))
        coverage = len(intersection) / len(q_set)
        return 0.5 * jaccard + 0.5 * coverage

    def _compute_recency_score(self, created_at: float, current_time: float) -> float:
        age_seconds = max(0.0, current_time - created_at)
        # Exponential half-life decay
        decay = math.exp(-0.693 * (age_seconds / self.half_life_seconds))
        return min(1.0, max(0.0, decay))

    def retrieve(
        self,
        query: str,
        owner_id: str,
        store: MemoryStore,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        as_of_timestamp: Optional[float] = None,
    ) -> List[MemoryRetrievalCandidate]:
        """
        Execute multi-criteria hybrid retrieval over active memories for owner_id.
        """
        now = time.time() if as_of_timestamp is None else as_of_timestamp

        # 1. Fetch active records for owner
        records = store.list_records(
            owner_id=owner_id,
            memory_type=memory_type,
            validity=MemoryValidity.ACTIVE,
            limit=500,
        )
        if not records:
            return []

        # Filter strictly temporally valid records
        active_records = [r for r in records if r.is_valid_at(now)]
        if not active_records:
            return []

        query_tokens = self._tokenize(query)

        # 2. Compute query embedding if provider available
        query_embedding: Optional[List[float]] = None
        if self.embedding_provider is not None:
            try:
                emb = self.embedding_provider.embed_query(query)
                if hasattr(emb, "tolist"):
                    query_embedding = emb.tolist()
                elif isinstance(emb, list):
                    query_embedding = emb
            except Exception:
                query_embedding = None

        candidates: List[MemoryRetrievalCandidate] = []

        for rec in active_records:
            # Lexical score
            doc_tokens = self._tokenize(rec.content)
            lex_score = self._compute_lexical_score(query_tokens, doc_tokens)

            # Semantic score
            sem_score = 0.0
            if query_embedding is not None and rec.embedding is not None:
                sem_score = max(0.0, MemoryDeduplicator.cosine_similarity(query_embedding, rec.embedding))

            # Recency score
            rec_score = self._compute_recency_score(rec.temporal.created_at, now)

            # Combined weighted score
            combined = (
                self.lexical_weight * lex_score +
                self.semantic_weight * sem_score +
                self.recency_weight * rec_score +
                self.importance_weight * rec.importance +
                self.confidence_weight * rec.confidence
            )

            candidates.append(MemoryRetrievalCandidate(
                memory_id=rec.memory_id,
                content=rec.content,
                memory_type=rec.memory_type,
                lexical_score=lex_score,
                semantic_score=sem_score,
                recency_score=rec_score,
                importance=rec.importance,
                confidence=rec.confidence,
                combined_score=combined,
                provenance=rec.provenance.to_dict(),
                tags=rec.tags,
            ))

        # Sort descending by combined_score
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates[:top_k]
