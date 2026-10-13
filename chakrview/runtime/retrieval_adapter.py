"""
Robust Retrieval and Knowledge Adapter for ChakrView.
Location: chakrview/runtime/retrieval_adapter.py

Wraps HybridRetriever (BM25 + Dense Vector Index) with:
1. Deduplication and ranking thresholding (> min_score).
2. Provenance retention (doc_id, chunk_id, content_hash).
3. Exception safety (empty indexes, missing terms, empty queries).
4. Multi-document batch ingestion and index rebuilding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.knowledge import BM25KnowledgeIndex, KnowledgeChunk
from chakrview.runtime.retrieval import (
    HybridRetriever,
    DeterministicHashEmbeddingProvider,
    InMemoryVectorIndex,
)


@dataclass
class RetrievedEvidence:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    retrieval_method: str  # "bm25", "hybrid", "vector"
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
        }


class RetrievalAdapter:
    """
    Production-grade adapter for HybridRetriever with strict provenance,
    deduplication, and score thresholding.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.20,
        embedding_dim: int = 64,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        self.min_score_threshold = min_score_threshold
        self.embedding_dim = embedding_dim
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

        self.lex_index = BM25KnowledgeIndex()
        self.embed_provider = DeterministicHashEmbeddingProvider(dimension=embedding_dim)
        self.vec_index = InMemoryVectorIndex()
        self.retriever = HybridRetriever(
            lexical_index=self.lex_index,
            embedding_provider=self.embed_provider,
            vector_index=self.vec_index,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )
        self._doc_registry: Dict[str, str] = {}
        self._total_chunks = 0

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    def ingest_document(self, doc_id: str, text: str, chunk_size: int = 250) -> List[str]:
        """Ingests a document, splits into chunks, and updates both BM25 and Vector indices."""
        clean_text = text.strip()
        if not clean_text:
            return []

        self._doc_registry[doc_id] = clean_text
        # Simple paragraph or sentence chunking
        raw_chunks = [c.strip() for c in clean_text.split("\n\n") if c.strip()]
        if not raw_chunks:
            raw_chunks = [clean_text]

        chunk_objs: List[KnowledgeChunk] = []
        chunk_ids: List[str] = []

        for idx, c_text in enumerate(raw_chunks):
            cid = f"{doc_id}_c{idx}"
            chunk_ids.append(cid)
            chunk = KnowledgeChunk(
                doc_id=doc_id,
                chunk_id=cid,
                chunk_index=idx,
                text=c_text,
                metadata={"source_doc": doc_id},
            )
            chunk_objs.append(chunk)
            # Add to dense vector index
            vec = self.embed_provider.embed_text(c_text)
            self.vec_index.add_vector(cid, vec, text=c_text, metadata={"doc_id": doc_id})

        self.lex_index.add_chunks(chunk_objs)
        self._total_chunks += len(chunk_objs)
        return chunk_ids

    def search(self, query: str, top_k: int = 3) -> List[RetrievedEvidence]:
        """Executes hybrid retrieval, filters by threshold, and deduplicates chunks."""
        q_clean = query.strip()
        if not q_clean or self._total_chunks == 0:
            return []

        try:
            from chakrview.runtime.retrieval import RetrievalQuery
            cand_res = self.retriever.retrieve_candidates(RetrievalQuery(text=q_clean, top_k=top_k))
            candidates = cand_res.candidates
        except Exception:
            return []

        results: List[RetrievedEvidence] = []
        seen_hashes = set()

        for cand in candidates:
            raw_lex = cand.metadata.get("raw_lexical_score", 0.0)
            if self.embed_provider.provider_id.startswith("deterministic_hash") and raw_lex <= 0.0:
                continue

            if cand.score < self.min_score_threshold:
                continue

            c_hash = hashlib.sha256(cand.text.encode("utf-8")).hexdigest()
            if c_hash in seen_hashes:
                continue
            seen_hashes.add(c_hash)

            results.append(
                RetrievedEvidence(
                    chunk_id=cand.candidate_id,
                    doc_id=cand.source_id,
                    text=cand.text,
                    score=float(cand.score),
                    retrieval_method=cand.retrieval_method,
                    metadata=dict(cand.metadata),
                    provenance_hash=c_hash,
                )
            )

        return results

    def clear(self) -> None:
        """Resets the indices."""
        self.lex_index = BM25KnowledgeIndex()
        self.vec_index = InMemoryVectorIndex()
        self.retriever = HybridRetriever(
            lexical_index=self.lex_index,
            embedding_provider=self.embed_provider,
            vector_index=self.vec_index,
            lexical_weight=self.lexical_weight,
            semantic_weight=self.semantic_weight,
        )
        self._doc_registry.clear()
        self._total_chunks = 0
