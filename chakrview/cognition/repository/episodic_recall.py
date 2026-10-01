"""
ChakrView Step 66: Episodic Memory Recall Loop, Structural Relevance & Deterministic Validity Verification.

Implements controlled cognitive episodic memory recall around the frozen neural core:
- MemoryRecallRequest: Strongly typed query representing task parameters, structural targets, and repo state.
- MemoryRecallStatus: Explicit recall lifecycle state (RECALLABLE, REJECTED_STALE, REJECTED_SUPERSEDED,
  CONFLICTED, NEGATIVE_BOUNDARY, UNAVAILABLE, ABSTAIN).
- RecalledMemoryItem: Individual evaluated memory candidate with score, classification, and provenance.
- RecalledContextBundle: Budget-constrained, deterministically ordered collection of positive patterns
  and negative boundaries ready for safe injection into the neural proposal boundary.
- EpisodicMemoryRecallCoordinator: Orchestrates deterministic candidate retrieval, structural relevance scoring,
  validity/staleness verification, conflict resolution, negative boundary exposure, and budget enforcement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.change_detector import RepositoryChangeDetector
from chakrview.cognition.repository.impact_analyzer import (
    RepositoryImpactAnalyzer,
    MemoryValidityStatus,
    MemoryRevalidationDecision,
)
from chakrview.cognition.repository.context_store import EvidenceRecord, EpistemicState


class MemoryRecallStatus(Enum):
    """Explicit status classification for evaluated memory candidates."""
    RECALLABLE = auto()            # Valid, active, relevant positive solution pattern
    NEGATIVE_BOUNDARY = auto()     # Valid negative failure pattern / boundary constraint
    REJECTED_STALE = auto()        # Memory invalidated by structural / dependency drift
    REJECTED_SUPERSEDED = auto()   # Memory superseded by newer verified episode
    CONFLICTED = auto()            # Conflicting active memories or positive vs negative collision
    UNAVAILABLE = auto()           # Target modules or prerequisites absent from repo state
    ABSTAIN = auto()               # Cross-domain negative transfer or ungrounded ambiguity


@dataclass(frozen=True)
class MemoryRecallRequest:
    """Explicit, strongly typed query request for episodic memory recall."""
    task_description: str
    task_family: str
    target_files: List[str] = field(default_factory=list)
    target_symbols: List[str] = field(default_factory=list)
    repository_pattern: Optional[str] = None
    symptom_signature: Optional[str] = None
    root_cause_signature: Optional[str] = None
    dependency_signature: Optional[str] = None
    repository_fingerprint: Optional[str] = None
    domain_tags: Set[str] = field(default_factory=set)
    language: str = "python"
    framework: str = "standard_library"


@dataclass(frozen=True)
class RecallBudget:
    """Strict limits to prevent unbounded memory injection into neural context."""
    max_candidate_memories: int = 10
    max_recalled_memories: int = 3
    max_negative_boundaries: int = 3
    relevance_threshold: float = 0.30
    conflict_delta_threshold: float = 0.05


@dataclass
class RecalledMemoryItem:
    """Detailed evaluation and audit record for an individual memory candidate."""
    record: RepositorySemanticRecord
    status: MemoryRecallStatus
    relevance_score: float
    rationale: str
    signal_breakdown: Dict[str, float] = field(default_factory=dict)
    provenance_evidence: Optional[EvidenceRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.record.memory_id,
            "status": self.status.name,
            "relevance_score": round(self.relevance_score, 4),
            "rationale": self.rationale,
            "signal_breakdown": {k: round(v, 4) for k, v in self.signal_breakdown.items()},
            "provenance_evidence": self.provenance_evidence.to_dict() if self.provenance_evidence else None,
        }


@dataclass
class RecalledContextBundle:
    """Budgeted, admissible episodic context exposed to the neural proposal boundary."""
    task_family: str
    positive_memories: List[RecalledMemoryItem] = field(default_factory=list)
    negative_boundaries: List[RecalledMemoryItem] = field(default_factory=list)
    rejected_items: List[RecalledMemoryItem] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: Optional[str] = None
    total_candidates_evaluated: int = 0
    duration_ms: float = 0.0

    @property
    def has_positive_guidance(self) -> bool:
        return len(self.positive_memories) > 0

    @property
    def has_negative_boundaries(self) -> bool:
        return len(self.negative_boundaries) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_family": self.task_family,
            "positive_memories": [item.to_dict() for item in self.positive_memories],
            "negative_boundaries": [item.to_dict() for item in self.negative_boundaries],
            "rejected_items": [item.to_dict() for item in self.rejected_items],
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "duration_ms": round(self.duration_ms, 2),
        }


def _token_jaccard(a: str, b: str) -> float:
    """Deterministic token Jaccard similarity."""
    toks_a = set(x.lower().strip(".,:-_") for x in a.split() if len(x) > 1)
    toks_b = set(x.lower().strip(".,:-_") for x in b.split() if len(x) > 1)
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


class EpisodicMemoryRecallCoordinator:
    """
    Coordinates deterministic episodic memory recall:
    1. Retrieves candidate records from RepositoryMemoryIndex using structured filters.
    2. Computes explainable, deterministic structural relevance scores.
    3. Verifies validity, freshness, and structural integrity against repository state.
    4. Detects and resolves conflicts, stale pointers, and superseded versions.
    5. Categorizes and exposes negative boundary constraints.
    6. Enforces recall budgeting, deterministic ranking, and passive context preparation.
    """

    def __init__(
        self,
        memory_index: RepositoryMemoryIndex,
        budget: Optional[RecallBudget] = None,
    ) -> None:
        self.memory_index = memory_index
        self.budget = budget or RecallBudget()

    def recall(
        self,
        request: MemoryRecallRequest,
        repo_state: Optional[RepositoryState] = None,
        reference_state: Optional[RepositoryState] = None,
    ) -> RecalledContextBundle:
        """Execute the deterministic episodic memory recall pipeline."""
        start_t = time.perf_counter()

        # 0. Check cross-domain negative transfer
        if request.domain_tags and request.task_family not in request.domain_tags:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            return RecalledContextBundle(
                task_family=request.task_family,
                abstained=True,
                abstain_reason=f"Cross-domain negative transfer blocked: task family '{request.task_family}' not in domain tags {sorted(request.domain_tags)}",
                duration_ms=elapsed,
            )

        # 1. Candidate Retrieval from RepositoryMemoryIndex
        candidates = self._retrieve_candidates(request)
        if not candidates:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            return RecalledContextBundle(
                task_family=request.task_family,
                abstained=False,
                abstain_reason="No matching candidate memories in index",
                duration_ms=elapsed,
            )

        # 2. Structural Relevance Evaluation & Classification
        evaluated_items: List[RecalledMemoryItem] = []
        for cand in candidates[: self.budget.max_candidate_memories]:
            item = self._evaluate_candidate(cand, request, repo_state, reference_state)
            evaluated_items.append(item)

        # 3. Conflict Detection across top positive candidates
        evaluated_items = self._arbitrate_conflicts(evaluated_items)

        # 4. Partition by status and apply budget limits
        positives: List[RecalledMemoryItem] = []
        negatives: List[RecalledMemoryItem] = []
        rejected: List[RecalledMemoryItem] = []

        # Sort deterministically: highest relevance score, then memory_id
        evaluated_items.sort(key=lambda it: (-it.relevance_score, it.record.memory_id))

        for it in evaluated_items:
            if it.status == MemoryRecallStatus.RECALLABLE:
                if len(positives) < self.budget.max_recalled_memories and it.relevance_score >= self.budget.relevance_threshold:
                    positives.append(it)
                else:
                    rejected.append(it)
            elif it.status == MemoryRecallStatus.NEGATIVE_BOUNDARY:
                if len(negatives) < self.budget.max_negative_boundaries:
                    negatives.append(it)
                else:
                    rejected.append(it)
            else:
                rejected.append(it)

        elapsed = (time.perf_counter() - start_t) * 1000.0

        return RecalledContextBundle(
            task_family=request.task_family,
            positive_memories=positives,
            negative_boundaries=negatives,
            rejected_items=rejected,
            abstained=False,
            total_candidates_evaluated=len(evaluated_items),
            duration_ms=elapsed,
        )

    def _retrieve_candidates(self, request: MemoryRecallRequest) -> List[RepositorySemanticRecord]:
        """Deterministic candidate search using task_family and affected_modules."""
        # Query by task_family first
        by_family = self.memory_index.candidate_set(task_family=request.task_family)

        # Query by affected modules if specified
        by_modules: List[RepositorySemanticRecord] = []
        if request.target_files:
            by_modules = self.memory_index.candidate_set(affected_modules=request.target_files)

        # Merge deterministically
        seen: Dict[str, RepositorySemanticRecord] = {}
        for rec in by_family:
            seen[rec.memory_id] = rec
        for rec in by_modules:
            seen[rec.memory_id] = rec

        result = list(seen.values())
        result.sort(key=lambda r: r.memory_id)
        return result

    def _evaluate_candidate(
        self,
        record: RepositorySemanticRecord,
        request: MemoryRecallRequest,
        repo_state: Optional[RepositoryState],
        reference_state: Optional[RepositoryState],
    ) -> RecalledMemoryItem:
        """Evaluate validity, freshness, and structural relevance for a memory record."""
        # 1. Superseded check
        if not record.active_version or record.superseded_by:
            return RecalledMemoryItem(
                record=record,
                status=MemoryRecallStatus.REJECTED_SUPERSEDED,
                relevance_score=0.0,
                rationale=f"REJECTED_SUPERSEDED: Memory version superseded by {record.superseded_by}",
            )

        # 2. Check existence of affected modules in repository state
        if repo_state and record.affected_modules:
            missing_modules = [m for m in record.affected_modules if m not in repo_state.files]
            if missing_modules:
                return RecalledMemoryItem(
                    record=record,
                    status=MemoryRecallStatus.UNAVAILABLE,
                    relevance_score=0.0,
                    rationale=f"UNAVAILABLE: Required module(s) {missing_modules} absent from repository state",
                )

        # 3. Check staleness via change detection and impact analysis if reference_state is provided
        if repo_state and reference_state and repo_state.state_fingerprint != reference_state.state_fingerprint:
            diff = RepositoryChangeDetector.detect_changes(reference_state, repo_state)
            directly_modified = sorted(list(diff.file_changes.keys()))
            transitive_affected: Set[str] = set()
            for m in directly_modified:
                transitive_affected.update(repo_state.dependency_graph.get_downstream_dependents(m))

            reval = RepositoryImpactAnalyzer.revalidate_memory(
                state=repo_state,
                diff=diff,
                memory=record,
                directly_modified=directly_modified,
                transitive_affected=transitive_affected,
            )
            if reval.status in (MemoryValidityStatus.STALE, MemoryValidityStatus.INVALID):
                return RecalledMemoryItem(
                    record=record,
                    status=MemoryRecallStatus.REJECTED_STALE,
                    relevance_score=0.0,
                    rationale=f"REJECTED_STALE: Memory invalidated by drift ({reval.rationale})",
                )

        # 4. Check repository fingerprint mismatch if requested
        if request.repository_fingerprint and repo_state and request.repository_fingerprint != repo_state.state_fingerprint:
            # Fingerprint drift: evaluate whether memory module was touched
            if any(m in record.affected_modules for m in repo_state.files.keys()):
                # Not fatal if non-conflicting, but record in provenance
                pass

        # 5. Compute deterministic structural relevance score
        score, signals = self._compute_relevance(record, request)

        # 6. Distinguish negative boundary vs positive solution
        is_negative = (
            record.solution_pattern == "DO_NOT_APPLY"
            or any("FAILURE_BOUNDARY" in b for b in record.known_boundaries)
            or record.failed_episodes > 0 and record.successful_episodes == 0
        )

        status = MemoryRecallStatus.NEGATIVE_BOUNDARY if is_negative else MemoryRecallStatus.RECALLABLE
        rationale = (
            f"Admitted as negative boundary constraint (failures={record.failed_episodes})"
            if is_negative
            else f"Admitted as relevant solution pattern (score={score:.4f})"
        )

        # 7. Attach EvidenceRecord provenance
        prov_evidence = EvidenceRecord(
            source_type="SEMANTIC_MEMORY",
            file_path=record.affected_modules[0] if record.affected_modules else "",
            symbol_name=request.target_symbols[0] if request.target_symbols else None,
            structural_digest=record.memory_id,
            retrieval_reason=f"Recalled pattern: {record.repository_pattern}",
            epistemic_state=EpistemicState.KNOWN,
        )

        return RecalledMemoryItem(
            record=record,
            status=status,
            relevance_score=score,
            rationale=rationale,
            signal_breakdown=signals,
            provenance_evidence=prov_evidence,
        )

    def _compute_relevance(
        self,
        record: RepositorySemanticRecord,
        request: MemoryRecallRequest,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Deterministic, bounded relevance scoring:
        - Task Family match: 0.35
        - Module Overlap: 0.25
        - Repository Pattern / Symptom similarity: 0.20
        - Dependency Signature match: 0.10
        - Evidence / Success History: 0.10
        Total theoretical max = 1.00.
        """
        signals: Dict[str, float] = {}

        # 1. Task Family
        fam_score = 1.0 if record.task_family == request.task_family else 0.0
        signals["task_family"] = fam_score * 0.35

        # 2. Module Overlap
        req_mods = set(request.target_files)
        rec_mods = set(record.affected_modules)
        if req_mods and rec_mods:
            overlap = len(req_mods & rec_mods) / len(req_mods | rec_mods)
        elif not req_mods and rec_mods:
            overlap = 0.5
        else:
            overlap = 0.0
        signals["module_overlap"] = overlap * 0.25

        # 3. Structural Pattern & Symptom
        pat_sim = 0.0
        if request.repository_pattern:
            pat_sim = _token_jaccard(request.repository_pattern, record.repository_pattern)
        elif request.symptom_signature:
            pat_sim = _token_jaccard(request.symptom_signature, record.symptom_signature)
        else:
            pat_sim = _token_jaccard(request.task_description, record.repository_pattern)
        signals["pattern_similarity"] = pat_sim * 0.20

        # 4. Dependency Signature
        dep_score = 0.0
        if request.dependency_signature and record.dependency_signature:
            dep_score = 1.0 if request.dependency_signature in record.dependency_signature else _token_jaccard(request.dependency_signature, record.dependency_signature)
        signals["dependency_match"] = dep_score * 0.10

        # 5. Provenance / Historical Evidence
        ev_score = min(1.0, record.successful_episodes / 3.0) if record.successful_episodes > 0 else 0.5
        signals["evidence_confidence"] = ev_score * 0.10

        total_score = sum(signals.values())
        return total_score, signals

    def _arbitrate_conflicts(self, items: List[RecalledMemoryItem]) -> List[RecalledMemoryItem]:
        """Detect and arbitrate conflicts between recallable candidates."""
        positives = [it for it in items if it.status == MemoryRecallStatus.RECALLABLE]
        negatives = [it for it in items if it.status == MemoryRecallStatus.NEGATIVE_BOUNDARY]

        # 1. Conflict between positive candidate and negative boundary on same target module
        for pos in positives:
            pos_mods = set(pos.record.affected_modules)
            for neg in negatives:
                neg_mods = set(neg.record.affected_modules)
                if pos_mods & neg_mods:
                    # Check if negative boundary directly forbids positive solution
                    if any("DO_NOT_APPLY" in b or "FAILURE_BOUNDARY" in b for b in neg.record.known_boundaries):
                        pos.status = MemoryRecallStatus.CONFLICTED
                        pos.rationale = f"CONFLICTED: Collides with negative boundary {neg.record.memory_id} on modules {sorted(list(pos_mods & neg_mods))}"

        # 2. Conflict between top two positive candidates with close scores but differing solutions
        active_pos = [p for p in positives if p.status == MemoryRecallStatus.RECALLABLE]
        if len(active_pos) >= 2:
            active_pos.sort(key=lambda p: -p.relevance_score)
            top1, top2 = active_pos[0], active_pos[1]
            score_diff = abs(top1.relevance_score - top2.relevance_score)
            if score_diff < self.budget.conflict_delta_threshold:
                if top1.record.solution_pattern != top2.record.solution_pattern and set(top1.record.affected_modules) == set(top2.record.affected_modules):
                    # Flag both as conflicted to avoid subjective selection
                    top1.status = MemoryRecallStatus.CONFLICTED
                    top1.rationale = f"CONFLICTED: Score delta ({score_diff:.4f}) within conflict threshold {self.budget.conflict_delta_threshold} against {top2.record.memory_id}"
                    top2.status = MemoryRecallStatus.CONFLICTED
                    top2.rationale = f"CONFLICTED: Score delta ({score_diff:.4f}) within conflict threshold {self.budget.conflict_delta_threshold} against {top1.record.memory_id}"

        return items
