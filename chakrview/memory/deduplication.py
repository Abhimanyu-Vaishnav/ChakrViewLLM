"""
Multi-Tier Deterministic Memory Deduplication for ChakrView (Step 16).

Pipeline:
1. Exact string match
2. Normalized match (whitespace collapsed, case folded, punctuation stripped)
3. Semantic vector cosine similarity (using Step 14 NeuralSemanticEmbeddingProvider or reference embedder)

Prevents uncontrolled memory fragmentation without silently deleting conflicting records.
"""

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Dict, List, Optional, Any, Tuple

from chakrview.memory.record import MemoryRecord, MemoryValidity


class MatchLevel(str, Enum):
    """Level of duplicate match detected."""
    EXACT = "exact"
    NORMALIZED = "normalized"
    SEMANTIC = "semantic"
    NONE = "none"


@dataclass
class DeduplicationResult:
    """
    Structured outcome of a duplicate detection pass.
    """
    is_duplicate: bool
    match_level: MatchLevel
    matched_record_id: Optional[str] = None
    similarity_score: float = 0.0
    notes: str = ""


class MemoryDeduplicator:
    """
    Multi-tier duplicate detector for persistent memory records.
    """

    def __init__(self, semantic_threshold: float = 0.88) -> None:
        self.semantic_threshold = semantic_threshold

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize string: lowercase, strip punctuation, collapse whitespace."""
        clean = text.lower()
        clean = re.sub(r"[^\w\s]", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            return 0.0
        return dot / (norm1 * norm2)

    def check_duplicate(
        self,
        candidate_content: str,
        existing_records: List[MemoryRecord],
        candidate_embedding: Optional[List[float]] = None,
        only_active: bool = True,
    ) -> DeduplicationResult:
        """
        Evaluate candidate against existing records using 3-tier cascade.
        """
        cand_raw = candidate_content.strip()
        cand_norm = self.normalize_text(cand_raw)

        # Filter active records if requested
        pool = [
            r for r in existing_records
            if not only_active or r.validity == MemoryValidity.ACTIVE
        ]

        # 1. Tier 1: Exact Match
        for rec in pool:
            if rec.content.strip() == cand_raw:
                return DeduplicationResult(
                    is_duplicate=True,
                    match_level=MatchLevel.EXACT,
                    matched_record_id=rec.memory_id,
                    similarity_score=1.0,
                    notes=f"Exact match with record '{rec.memory_id}'.",
                )

        # 2. Tier 2: Normalized Match
        for rec in pool:
            rec_norm = self.normalize_text(rec.content)
            if rec_norm == cand_norm:
                return DeduplicationResult(
                    is_duplicate=True,
                    match_level=MatchLevel.NORMALIZED,
                    matched_record_id=rec.memory_id,
                    similarity_score=0.98,
                    notes=f"Normalized match with record '{rec.memory_id}'.",
                )

        # 3. Tier 3: Semantic Cosine Match (if embeddings available)
        if candidate_embedding is not None:
            best_sim = 0.0
            best_rec_id = None
            for rec in pool:
                if rec.embedding is not None:
                    sim = self.cosine_similarity(candidate_embedding, rec.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_rec_id = rec.memory_id

            if best_sim >= self.semantic_threshold and best_rec_id is not None:
                return DeduplicationResult(
                    is_duplicate=True,
                    match_level=MatchLevel.SEMANTIC,
                    matched_record_id=best_rec_id,
                    similarity_score=best_sim,
                    notes=f"Semantic duplicate (similarity {best_sim:.3f} >= {self.semantic_threshold}) with record '{best_rec_id}'.",
                )

        return DeduplicationResult(
            is_duplicate=False,
            match_level=MatchLevel.NONE,
            matched_record_id=None,
            similarity_score=0.0,
            notes="No duplicate detected.",
        )
