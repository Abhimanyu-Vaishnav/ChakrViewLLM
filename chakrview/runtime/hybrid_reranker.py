"""
ChakrView Hybrid Retrieval Re-Ranker.
Location: chakrview/runtime/hybrid_reranker.py

Architecture:
1. Candidate Pool Generation: High-recall lexical BM25 retrieval over ingested corpus.
2. Neural Bi-Encoder Re-Ranking: Scores lexical candidates using ChakrBiEncoder representation.
3. Calibrated Score Fusion:
   Score_final = (1 - alpha) * Score_lexical_norm + alpha * Score_neural_sim
4. Provenance Preservation: Preserves raw lexical and neural scores, chunk IDs, and source document hashes.
5. Safe Lexical Fallback: When neural provider is unavailable, disabled, or fails, falls back to BM25 scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import math

from chakrview.runtime.knowledge import BM25KnowledgeIndex, KnowledgeChunk
from chakrview.runtime.retrieval import EmbeddingProvider, DeterministicHashEmbeddingProvider
from chakrview.runtime.retrieval_adapter import RetrievedEvidence


@dataclass
class RerankedEvidence:
    chunk_id: str
    doc_id: str
    text: str
    lexical_score: float
    neural_score: float
    final_score: float
    retrieval_rank: int
    rerank_delta: int  # change in position (+ improved, - degraded, 0 unchanged)
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "lexical_score": round(self.lexical_score, 4),
            "neural_score": round(self.neural_score, 4),
            "final_score": round(self.final_score, 4),
            "retrieval_rank": self.retrieval_rank,
            "rerank_delta": self.rerank_delta,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
        }


class HybridReranker:
    """
    Two-stage hybrid retriever:
    Stage 1: BM25 candidate generation (high recall).
    Stage 2: Learned bi-encoder re-ranking (fine-grained discriminability).
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        rerank_alpha: float = 0.4,
        min_abstention_threshold: float = 0.15,
        enable_reranking: bool = True,
    ) -> None:
        self.lex_index = BM25KnowledgeIndex()
        self.embedding_provider = embedding_provider or DeterministicHashEmbeddingProvider(dimension=64)
        self.rerank_alpha = rerank_alpha
        self.min_abstention_threshold = min_abstention_threshold
        self.enable_reranking = enable_reranking
        self._doc_chunks: Dict[str, KnowledgeChunk] = {}
        self._total_chunks = 0

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    def ingest_document(self, doc_id: str, text: str) -> List[str]:
        clean_text = text.strip()
        if not clean_text:
            return []

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
            self._doc_chunks[cid] = chunk

        self.lex_index.add_chunks(chunk_objs)
        self._total_chunks += len(chunk_objs)
        return chunk_ids

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 3,
        candidate_pool_size: int = 10,
    ) -> List[RerankedEvidence]:
        """
        Retrieves candidate pool using BM25, then optionally applies learned re-ranking.
        """
        q_clean = query.strip()
        if not q_clean or self._total_chunks == 0:
            return []

        # Stage 1: BM25 Candidate Retrieval
        bm25_hits = self.lex_index.search(q_clean, top_k=candidate_pool_size)
        if not bm25_hits:
            return []

        # Extract candidates
        candidates = []
        max_lex = max(score for _, score in bm25_hits) if bm25_hits else 1.0
        if max_lex <= 0.0:
            max_lex = 1.0

        for rank, (chunk, raw_score) in enumerate(bm25_hits):
            norm_lex = raw_score / max_lex
            candidates.append({
                "chunk": chunk,
                "raw_lex": raw_score,
                "norm_lex": norm_lex,
                "initial_rank": rank,
            })

        # Stage 2: Neural Re-Ranking
        q_vec = None
        if self.enable_reranking and self.embedding_provider is not None:
            try:
                q_vec = self.embedding_provider.embed_text(q_clean)
            except Exception:
                q_vec = None

        scored_results: List[Dict[str, Any]] = []
        for cand in candidates:
            chunk = cand["chunk"]
            neural_sim = 0.0
            if q_vec is not None:
                try:
                    c_vec = self.embedding_provider.embed_text(chunk.text)
                    # Dot product between normalized unit vectors = cosine similarity
                    dot = sum(a * b for a, b in zip(q_vec, c_vec))
                    neural_sim = max(0.0, min(1.0, (dot + 1.0) / 2.0))
                except Exception:
                    neural_sim = 0.0

            if self.enable_reranking and q_vec is not None:
                final_score = (1.0 - self.rerank_alpha) * cand["norm_lex"] + self.rerank_alpha * neural_sim
            else:
                final_score = cand["norm_lex"]

            scored_results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "lexical_score": cand["raw_lex"],
                "neural_score": neural_sim,
                "final_score": final_score,
                "initial_rank": cand["initial_rank"],
            })

        # Sort by final score descending, breaking ties by initial BM25 rank
        scored_results.sort(key=lambda x: (x["final_score"], -x["initial_rank"]), reverse=True)

        final_evidence: List[RerankedEvidence] = []
        for new_rank, item in enumerate(scored_results[:top_k]):
            if item["final_score"] < self.min_abstention_threshold:
                continue
            delta = item["initial_rank"] - new_rank  # positive if moved up
            evidence = RerankedEvidence(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                lexical_score=item["lexical_score"],
                neural_score=item["neural_score"],
                final_score=item["final_score"],
                retrieval_rank=new_rank,
                rerank_delta=delta,
            )
            final_evidence.append(evidence)

        return final_evidence
