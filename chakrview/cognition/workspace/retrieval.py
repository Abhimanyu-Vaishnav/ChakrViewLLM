"""
ChakrView Step 58: Explainable Deterministic Memory Retrieval.

Defines:
- MemoryRetrievalResult: Dataclass wrapping matched memory with explainable score components.
- ExplainableMemoryRetriever: Deterministic ranking system that matches active task objectives,
  task families, and diagnoses against episodic and semantic memory stores.
  Features:
  - Explainable scoring breakdown (family match, diagnosis match, evidence confidence, mismatch penalty).
  - Pure CPU-first deterministic execution.
  - Explainable audit logging for why a memory was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from chakrview.learning.experience import ExperienceRecord
from chakrview.cognition.workspace.consolidation import SemanticMemoryEntry, MemoryConsolidator


@dataclass
class MemoryRetrievalResult:
    """
    Explainable result wrapper containing the memory item and score breakdown.
    """
    memory_type: str  # "SEMANTIC" or "EPISODIC"
    memory_id: str
    reusable_content: str
    score: float
    family_match_score: float
    diagnosis_match_score: float
    evidence_score: float
    penalty_score: float
    explanation: str
    raw_item: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplainableMemoryRetriever:
    """
    Retrieves and ranks relevant memories deterministically with full explanation.
    """

    def __init__(
        self,
        weight_family: float = 0.40,
        weight_diagnosis: float = 0.40,
        weight_evidence: float = 0.20,
        penalty_mismatch: float = 0.50,
    ) -> None:
        self.w_fam = weight_family
        self.w_diag = weight_diagnosis
        self.w_ev = weight_evidence
        self.w_pen = penalty_mismatch

    def retrieve(
        self,
        task_family: str,
        objective: str,
        current_diagnosis: Optional[str] = None,
        task_id: Optional[str] = None,
        consolidator: Optional[MemoryConsolidator] = None,
        top_k: int = 3,
    ) -> List[MemoryRetrievalResult]:
        """
        Rank all available episodic and semantic memories.
        Returns ranked list of MemoryRetrievalResults with explainable scores.
        """
        if consolidator is None:
            return []

        results: List[MemoryRetrievalResult] = []
        target_family = task_family.lower().strip()
        diag_str = (current_diagnosis or "").lower().strip()

        # 1. Rank Semantic Patterns first (higher abstraction)
        for pattern in consolidator.semantic_store.values():
            res = self._score_semantic_pattern(pattern, target_family, objective, diag_str)
            if res.score > 0.0:
                results.append(res)

        # 2. Rank Episodic Experiences
        for exp in consolidator.episodic_store.values():
            res = self._score_episodic_record(exp, target_family, objective, diag_str, task_id=task_id)
            if res.score > 0.0:
                results.append(res)

        # Sort descending by score, tie-breaking deterministically by memory_id
        results.sort(key=lambda r: (r.score, r.memory_id), reverse=True)
        return results[:top_k]

    def _score_semantic_pattern(
        self,
        pattern: SemanticMemoryEntry,
        target_family: str,
        objective: str,
        diagnosis: str,
    ) -> MemoryRetrievalResult:
        # Family match
        fam_score = 1.0 if pattern.task_family.lower() == target_family else 0.0

        # Diagnosis / archetype token overlap (Jaccard-like)
        diag_score = self._compute_token_overlap(pattern.problem_archetype.lower(), diagnosis) if diagnosis else 0.5

        # Evidence confidence (scaled by count, capped at 1.0)
        ev_score = min(1.0, pattern.evidence_count / 5.0)

        # Boundary condition penalty check
        penalty = 0.0
        for boundary in pattern.boundary_conditions:
            if "restricted to task family" in boundary.lower() and fam_score == 0.0:
                penalty += self.w_pen

        final_score = (self.w_fam * fam_score) + (self.w_diag * diag_score) + (self.w_ev * ev_score) - penalty
        final_score = max(0.0, round(final_score, 4))

        explanation = (
            f"Semantic Pattern '{pattern.pattern_name}' (ID: {pattern.pattern_id}): "
            f"family_match={fam_score:.2f}, diag_match={diag_score:.2f}, "
            f"evidence_count={pattern.evidence_count} (ev_score={ev_score:.2f}), "
            f"penalty={penalty:.2f} -> final_score={final_score:.4f}"
        )

        return MemoryRetrievalResult(
            memory_type="SEMANTIC",
            memory_id=pattern.pattern_id,
            reusable_content=pattern.solution_strategy,
            score=final_score,
            family_match_score=fam_score,
            diagnosis_match_score=diag_score,
            evidence_score=ev_score,
            penalty_score=penalty,
            explanation=explanation,
            raw_item=pattern.to_dict(),
        )

    def _score_episodic_record(
        self,
        record: ExperienceRecord,
        target_family: str,
        objective: str,
        diagnosis: str,
        task_id: Optional[str] = None,
    ) -> MemoryRetrievalResult:
        # Family match
        fam_score = 1.0 if record.task_family.lower() == target_family else 0.0

        # Task ID exact re-encounter bonus
        task_id_bonus = 0.30 if task_id and record.task_id == task_id else 0.0

        # Diagnosis match
        rec_diag = (record.diagnosis or "").lower()
        diag_score = self._compute_token_overlap(rec_diag, diagnosis) if diagnosis else 0.3

        # Evidence confidence: single verified episode
        ev_score = 0.20 if record.verified else 0.0

        # Penalty if task family completely different
        penalty = self.w_pen if fam_score == 0.0 else 0.0

        final_score = (self.w_fam * fam_score) + (self.w_diag * diag_score) + (self.w_ev * ev_score) + task_id_bonus - penalty
        final_score = max(0.0, round(final_score, 4))

        explanation = (
            f"Episodic Record '{record.task_id}' (ID: {record.experience_id}): "
            f"family_match={fam_score:.2f}, task_id_bonus={task_id_bonus:.2f}, diag_match={diag_score:.2f}, "
            f"verified={record.verified} (ev_score={ev_score:.2f}), "
            f"penalty={penalty:.2f} -> final_score={final_score:.4f}"
        )

        return MemoryRetrievalResult(
            memory_type="EPISODIC",
            memory_id=record.experience_id,
            reusable_content=record.reusable_pattern,
            score=final_score,
            family_match_score=fam_score,
            diagnosis_match_score=diag_score,
            evidence_score=ev_score,
            penalty_score=penalty,
            explanation=explanation,
            raw_item=record.to_dict(),
        )


    @staticmethod
    def _compute_token_overlap(s1: str, s2: str) -> float:
        """Deterministic Jaccard token overlap between two strings."""
        if not s1 or not s2:
            return 0.0
        tokens1 = set(s1.replace(":", " ").replace(";", " ").replace(".", " ").split())
        tokens2 = set(s2.replace(":", " ").replace(";", " ").replace(".", " ").split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0.0
