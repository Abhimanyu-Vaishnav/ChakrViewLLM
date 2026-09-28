"""
Hybrid Memory & Semantic Retrieval Foundation for ChakrView (Step 13).

Provides a modular, lightweight retrieval architecture combining:
1. Lexical retrieval (BM25 term-frequency and saturation)
2. Semantic vector retrieval (embedding provider abstraction + vector index)
3. Deterministic score fusion (normalized weighted combination / RRF)
4. Memory + Knowledge source orchestration without merging underlying storage
5. Strict provenance preservation and passive data containment

Architectural Guarantees:
- Model remains strictly stateless; runtime coordinates retrieval.
- Zero heavyweight external vector database dependencies (no FAISS, Chroma, Pinecone).
- Replaceable embedding provider interface for future neural model drop-in.
- Deterministic reference embedding implementation for testable, reproducible vector math.
- Complete backward compatibility with Step 11 RAG and Step 12 conversational memory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import math
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union

from chakrview.runtime.knowledge import (
    KnowledgeChunk,
    KnowledgeIndex,
    BM25KnowledgeIndex,
    Retriever,
    LexicalRetriever,
    KnowledgeProvenance,
)
from chakrview.runtime.memory import (
    MemoryItem,
    MemoryType,
    WorkingMemory,
)


class RetrievalSourceType(str, Enum):
    """Origin category of a retrieved candidate."""
    KNOWLEDGE = "knowledge"
    DOCUMENT = "document"
    MEMORY = "memory"
    CONVERSATION_SUMMARY = "conversation_summary"


@dataclass
class RetrievalCandidate:
    """
    Unified representation of a retrieved item across knowledge and memory sources.
    
    Attributes:
        candidate_id: Unique identifier for this candidate item.
        text: Raw text content retrieved.
        source_type: Category of origin (knowledge, memory, conversation_summary).
        source_id: Parent document ID or memory session ID.
        score: Unified retrieval relevance score in [0.0, 1.0].
        retrieval_method: Method employed ("lexical", "semantic", "hybrid", "heuristic").
        token_count: Estimated token footprint for context budgeting.
        metadata: Associated provenance attributes (titles, tags, section offsets).
        provenance: Optional structured provenance tracking.
    """
    candidate_id: str
    text: str
    source_type: RetrievalSourceType
    source_id: str
    score: float
    retrieval_method: str = "lexical"
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.token_count <= 0 and self.text:
            self.token_count = max(1, int(len(self.text.split()) * 1.3))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalCandidate":
        d = dict(data)
        if isinstance(d.get("source_type"), str):
            d["source_type"] = RetrievalSourceType(d["source_type"])
        return cls(**d)

    @classmethod
    def from_knowledge_chunk(
        cls,
        chunk: KnowledgeChunk,
        score: float,
        retrieval_method: str = "bm25",
    ) -> "RetrievalCandidate":
        """Convert a KnowledgeChunk into a unified RetrievalCandidate."""
        prov = KnowledgeProvenance(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            source_id=chunk.metadata.get("source_id", "knowledge_doc"),
            content_hash=chunk.chunk_hash,
            retrieval_score=score,
            text_preview=chunk.text[:80] + ("..." if len(chunk.text) > 80 else ""),
            metadata=dict(chunk.metadata),
        )
        return cls(
            candidate_id=chunk.chunk_id,
            text=chunk.text,
            source_type=RetrievalSourceType.KNOWLEDGE,
            source_id=chunk.doc_id,
            score=score,
            retrieval_method=retrieval_method,
            token_count=chunk.token_count,
            metadata=dict(chunk.metadata),
            provenance=prov.to_dict(),
        )

    @classmethod
    def from_memory_item(
        cls,
        item: MemoryItem,
        score: float,
        retrieval_method: str = "memory_heuristic",
    ) -> "RetrievalCandidate":
        """Convert a short-term MemoryItem into a unified RetrievalCandidate."""
        src_type = (
            RetrievalSourceType.CONVERSATION_SUMMARY
            if item.memory_type == MemoryType.SUMMARY
            else RetrievalSourceType.MEMORY
        )
        return cls(
            candidate_id=item.memory_id,
            text=item.content,
            source_type=src_type,
            source_id=item.source_turn_id or "session_working_memory",
            score=score,
            retrieval_method=retrieval_method,
            token_count=item.token_estimate,
            metadata={
                "memory_type": item.memory_type.value,
                "importance": item.importance,
                "created_sequence": item.created_sequence,
                **item.metadata,
            },
            provenance={
                "memory_id": item.memory_id,
                "memory_type": item.memory_type.value,
                "source_turn_id": item.source_turn_id,
                "created_sequence": item.created_sequence,
            },
        )


@dataclass
class RetrievalQuery:
    """
    Search request parameters for the hybrid retrieval subsystem.
    
    Attributes:
        text: Query string.
        top_k: Maximum candidates to return.
        score_threshold: Minimum acceptable relevance score.
        source_types: Optional filter restricting sources (knowledge, memory, summary).
        lexical_weight: Weight for BM25 term overlap in hybrid fusion.
        semantic_weight: Weight for vector similarity in hybrid fusion.
        metadata_filters: Optional key-value constraints on candidate metadata.
    """
    text: str
    top_k: int = 5
    score_threshold: Optional[float] = None
    source_types: Optional[List[RetrievalSourceType]] = None
    lexical_weight: float = 0.5
    semantic_weight: float = 0.5
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.lexical_weight < 0.0 or self.semantic_weight < 0.0:
            raise ValueError("Fusion weights must be non-negative.")
        if self.lexical_weight == 0.0 and self.semantic_weight == 0.0:
            raise ValueError("At least one fusion weight must be greater than zero.")


@dataclass
class RetrievalResult:
    """
    Structured outcome of a unified retrieval operation.
    
    Attributes:
        query: Original search string.
        candidates: Chronologically or relevance-ordered candidate items.
        total_candidates_scanned: Number of records evaluated across indices.
        execution_time_ms: Measured latency in milliseconds.
        retrieval_method: Applied routing strategy ("lexical", "semantic", "hybrid", "unified").
        metadata: Execution accounting.
    """
    query: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    total_candidates_scanned: int = 0
    execution_time_ms: float = 0.0
    retrieval_method: str = "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_candidates_scanned": self.total_candidates_scanned,
            "execution_time_ms": self.execution_time_ms,
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata,
        }


class EmbeddingProvider(ABC):
    """
    Abstract interface for generating text embeddings.
    
    Establishes the contract for semantic search, vector indexing, and
    cosine similarity without coupling the runtime to a specific ML framework.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert a single text string into a dense vector."""
        pass

    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Batch vectorize multiple text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of emitted embedding vectors."""
        pass

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier of the embedding model/provider."""
        pass

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Compute deterministic cosine similarity between two float vectors.
        
        Returns scalar in [-1.0, 1.0]. Returns 0.0 for zero-norm vectors.
        """
        if len(v1) != len(v2):
            raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for a, b in zip(v1, v2):
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0
        sim = dot / (math.sqrt(norm1) * math.sqrt(norm2))
        # Clamp to [-1.0, 1.0] against precision overflow
        return max(-1.0, min(1.0, float(sim)))


class DeterministicHashEmbeddingProvider(EmbeddingProvider):
    """
    Lightweight, deterministic reference embedding provider.
    
    CRITICAL NOTICE:
    This reference implementation is an architectural test harness for validating
    vector math, dimensionality, cosine similarity, and provider hot-swapping.
    It does NOT provide neural semantic understanding and must NOT be characterized as such.
    
    Mechanism:
    Computes signed feature projections over word and character 3-grams using
    deterministic SHA-256 hash buckets, followed by L2 vector normalization.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_id(self) -> str:
        return f"deterministic_hash_v1_d{self._dim}"

    def embed_text(self, text: str) -> List[float]:
        clean = text.strip().lower()
        if not clean:
            return [0.0] * self._dim

        vec = [0.0] * self._dim
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", clean)

        # 1. Word token projections
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[bucket] += sign * 1.0

        # 2. Character 3-gram projections for subword robustness
        for i in range(len(clean) - 2):
            trigram = clean[i:i + 3]
            digest = hashlib.sha256(trigram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[bucket] += sign * 0.35

        # 3. L2 Normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [float(x / norm) for x in vec]
        return vec

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class VectorIndex(ABC):
    """
    Abstract interface for in-memory or external vector storage and retrieval.
    """

    @abstractmethod
    def add_vector(
        self,
        item_id: str,
        vector: List[float],
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store item embedding and metadata."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search by vector similarity.
        Returns list of (item_id, similarity_score, metadata) ordered by score.
        """
        pass

    @abstractmethod
    def remove(self, item_id: str) -> bool:
        """Delete an item by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count of stored vectors."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored vectors."""
        pass


class InMemoryVectorIndex(VectorIndex):
    """
    Pure-Python, deterministic in-memory vector index.
    
    Zero external dependencies. Employs exact cosine similarity with deterministic
    tie-breaking on (-score, item_id).
    """

    def __init__(self, index_id: str = "vector_in_memory_default") -> None:
        self.index_id = index_id
        self._vectors: Dict[str, List[float]] = {}
        self._texts: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add_vector(
        self,
        item_id: str,
        vector: List[float],
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not item_id:
            raise ValueError("item_id cannot be empty.")
        self._vectors[item_id] = list(vector)
        self._texts[item_id] = text
        self._metadata[item_id] = dict(metadata or {})

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._vectors:
            return []

        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for item_id, vec in self._vectors.items():
            sim = EmbeddingProvider.cosine_similarity(query_vector, vec)
            if score_threshold is not None and sim < score_threshold:
                continue
            meta = dict(self._metadata.get(item_id, {}))
            meta["text"] = self._texts.get(item_id, "")
            scored.append((item_id, sim, meta))

        # Deterministic sorting: highest score first, tie-break alphabetical by item_id
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    def remove(self, item_id: str) -> bool:
        if item_id in self._vectors:
            del self._vectors[item_id]
            self._texts.pop(item_id, None)
            self._metadata.pop(item_id, None)
            return True
        return False

    def count(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()
        self._texts.clear()
        self._metadata.clear()


class HybridRetriever(Retriever):
    """
    Unified Hybrid Retriever combining Okapi BM25 Lexical search and Dense Vector search.
    
    Formula for Score Fusion:
    Given:
        S_lex = BM25 lexical score >= 0.0
        S_sem = Cosine similarity in [-1.0, 1.0]
    
    1. Normalization:
       Normalized S_lex: S_lex / max(S_lex_set) (clipped to [0.0, 1.0])
       Normalized S_sem: max(0.0, S_sem) (non-negative cosine similarity)
       
    2. Fusion:
       S_hybrid = w_lex * Norm(S_lex) + w_sem * Norm(S_sem)
       Where w_lex + w_sem = 1.0
       
    Fallback Guarantee:
    - If semantic search has no results, pure normalized lexical score is utilized.
    - If lexical search has no matches, pure normalized semantic score is utilized.
    - Deterministic ordering: (-S_hybrid, chunk_id).
    """

    def __init__(
        self,
        lexical_index: KnowledgeIndex,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_index: Optional[VectorIndex] = None,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        self.lexical_index = lexical_index
        self.embedding_provider = embedding_provider or DeterministicHashEmbeddingProvider()
        self.vector_index = vector_index or InMemoryVectorIndex()
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """Synchronously index chunks in both lexical and semantic vector stores."""
        lex_added = self.lexical_index.add_chunks(chunks)
        for chunk in chunks:
            vec = self.embedding_provider.embed_text(chunk.text)
            self.vector_index.add_vector(
                item_id=chunk.chunk_id,
                vector=vec,
                text=chunk.text,
                metadata={
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_hash": chunk.chunk_hash,
                    "token_count": chunk.token_count,
                    **chunk.metadata,
                },
            )
        return lex_added

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Retriever interface implementation returning (KnowledgeChunk, score) tuples.
        100% compatible with Step 11/12 PromptContextBuilder and InferenceSession.
        """
        candidates = self.retrieve_candidates(RetrievalQuery(text=query, top_k=top_k))
        results: List[Tuple[KnowledgeChunk, float]] = []

        for cand in candidates.candidates:
            # Reconstruct or fetch KnowledgeChunk
            chunk = KnowledgeChunk(
                chunk_id=cand.candidate_id,
                doc_id=cand.source_id,
                chunk_index=cand.metadata.get("chunk_index", 0),
                text=cand.text,
                token_count=cand.token_count,
                metadata=dict(cand.metadata),
                chunk_hash=cand.metadata.get("chunk_hash", ""),
            )
            results.append((chunk, cand.score))

        return results

    def retrieve_candidates(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Execute full hybrid score fusion and emit structured RetrievalResult.
        """
        t0 = time.perf_counter()
        q_text = query.text.strip()
        if not q_text:
            return RetrievalResult(query=query.text, retrieval_method="hybrid")

        # 1. Lexical Retrieval (BM25)
        lex_results = self.lexical_index.search(q_text, top_k=query.top_k * 2)

        # 2. Semantic Vector Retrieval
        q_vec = self.embedding_provider.embed_text(q_text)
        vec_results = self.vector_index.search(q_vec, top_k=query.top_k * 2)

        # Map by chunk_id
        chunk_map: Dict[str, Dict[str, Any]] = {}
        lex_scores: Dict[str, float] = {}
        for chunk, score in lex_results:
            cid = chunk.chunk_id
            lex_scores[cid] = score
            chunk_map[cid] = {
                "chunk": chunk,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "token_count": chunk.token_count,
                "metadata": dict(chunk.metadata),
                "chunk_hash": chunk.chunk_hash,
            }

        sem_scores: Dict[str, float] = {}
        for cid, sim, meta in vec_results:
            sem_scores[cid] = sim
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": None,
                    "text": meta.get("text", ""),
                    "doc_id": meta.get("doc_id", "doc_unknown"),
                    "token_count": meta.get("token_count", 0),
                    "metadata": dict(meta),
                    "chunk_hash": meta.get("chunk_hash", ""),
                }

        # 3. Normalization Factors
        max_lex = max(lex_scores.values()) if lex_scores else 1.0
        if max_lex <= 0.0:
            max_lex = 1.0

        all_candidate_ids = set(lex_scores.keys()).union(sem_scores.keys())
        scored_candidates: List[RetrievalCandidate] = []

        w_lex = query.lexical_weight
        w_sem = query.semantic_weight
        w_total = w_lex + w_sem
        w_lex /= w_total
        w_sem /= w_total

        has_lex = len(lex_scores) > 0
        has_sem = len(sem_scores) > 0

        for cid in all_candidate_ids:
            raw_lex = lex_scores.get(cid, 0.0)
            raw_sem = sem_scores.get(cid, 0.0)

            norm_lex = max(0.0, raw_lex / max_lex)
            norm_sem = max(0.0, raw_sem)  # cosine similarity non-negative clamp

            # Handle graceful fallback if one subsystem yielded zero results
            if has_lex and has_sem:
                final_score = (w_lex * norm_lex) + (w_sem * norm_sem)
                method = "hybrid"
            elif has_lex:
                final_score = norm_lex
                method = "lexical_fallback"
            else:
                final_score = norm_sem
                method = "semantic_fallback"

            if query.score_threshold is not None and final_score < query.score_threshold:
                continue

            info = chunk_map[cid]
            cand = RetrievalCandidate(
                candidate_id=cid,
                text=info["text"],
                source_type=RetrievalSourceType.KNOWLEDGE,
                source_id=info["doc_id"],
                score=float(final_score),
                retrieval_method=method,
                token_count=info["token_count"],
                metadata={
                    "raw_lexical_score": raw_lex,
                    "raw_semantic_score": raw_sem,
                    **info["metadata"],
                },
                provenance={
                    "doc_id": info["doc_id"],
                    "chunk_id": cid,
                    "content_hash": info["chunk_hash"],
                    "retrieval_score": final_score,
                },
            )
            scored_candidates.append(cand)

        # Deterministic sorting: highest score first, tie-break alphabetical on candidate_id
        scored_candidates.sort(key=lambda x: (-x.score, x.candidate_id))
        selected = scored_candidates[:query.top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            query=q_text,
            candidates=selected,
            total_candidates_scanned=len(all_candidate_ids),
            execution_time_ms=elapsed_ms,
            retrieval_method="hybrid" if (has_lex and has_sem) else "fallback",
            metadata={
                "lexical_candidates_count": len(lex_scores),
                "semantic_candidates_count": len(sem_scores),
                "lexical_weight": w_lex,
                "semantic_weight": w_sem,
            },
        )


class UnifiedRetriever:
    """
    Retrieval Orchestration Layer unifying Working Memory and Knowledge Documents.
    
    Architectural Contract:
    - Does NOT merge underlying storage models (WorkingMemory stays in memory.py,
      KnowledgeIndex stays in knowledge.py).
    - Preserves independent governance, lifespan, and eviction policies for each source.
    - Emits a unified list of RetrievalCandidate items with explicit RetrievalSourceType.
    """

    def __init__(
        self,
        knowledge_retriever: Optional[Retriever] = None,
        working_memory: Optional[WorkingMemory] = None,
        memory_priority_boost: float = 0.1,
    ) -> None:
        self.knowledge_retriever = knowledge_retriever
        self.working_memory = working_memory
        self.memory_priority_boost = memory_priority_boost

    def retrieve(
        self,
        query: str,
        top_k_knowledge: int = 3,
        top_k_memory: int = 3,
        active_domain: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Query both knowledge and memory sources, yielding unified, ranked candidates.
        """
        t0 = time.perf_counter()
        q_text = query.strip()
        candidates: List[RetrievalCandidate] = []
        total_scanned = 0

        # 1. Search Knowledge Source
        if self.knowledge_retriever is not None and q_text:
            if isinstance(self.knowledge_retriever, HybridRetriever):
                h_res = self.knowledge_retriever.retrieve_candidates(
                    RetrievalQuery(text=q_text, top_k=top_k_knowledge)
                )
                candidates.extend(h_res.candidates)
                total_scanned += h_res.total_candidates_scanned
            else:
                raw_k = self.knowledge_retriever.retrieve(q_text, top_k=top_k_knowledge)
                for chunk, score in raw_k:
                    candidates.append(
                        RetrievalCandidate.from_knowledge_chunk(chunk, score, retrieval_method="lexical")
                    )
                total_scanned += len(raw_k)

        # 2. Search Working Memory Source
        if self.working_memory is not None and q_text:
            raw_mem = self.working_memory.retrieve_relevant(
                query=q_text,
                top_k=top_k_memory,
                active_domain=active_domain,
            )
            for mem_item, score in raw_mem:
                # Add optional boost to active short-term conversational context
                adjusted_score = min(1.0, score + self.memory_priority_boost)
                candidates.append(
                    RetrievalCandidate.from_memory_item(
                        mem_item,
                        adjusted_score,
                        retrieval_method="memory_heuristic",
                    )
                )
            total_scanned += len(raw_mem)

        # 3. Deterministic Sorting: highest score first, tie-break alphabetical on candidate_id
        candidates.sort(key=lambda x: (-x.score, x.candidate_id))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            query=q_text,
            candidates=candidates,
            total_candidates_scanned=total_scanned,
            execution_time_ms=elapsed_ms,
            retrieval_method="unified_orchestration",
            metadata={
                "active_domain": active_domain,
                "memory_priority_boost": self.memory_priority_boost,
            },
        )


# Step 14 Neural Semantic Provider Re-export
from chakrview.semantic.provider import NeuralSemanticEmbeddingProvider
