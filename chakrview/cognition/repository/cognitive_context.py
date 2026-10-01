"""
ChakrView Step 67: Unified Cognitive Context Composition Layer.

Composes grounded repository evidence, recalled episodic memories, verified
negative boundary constraints, and current task specifications into a deterministically
ordered, budget-enforced, passive CognitiveContextBundle ready for consumption by
the passive NeuralProposalAdapter without mutation privileges, execution access,
or memory-write authority.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.context_store import (
    RepositoryContextStore,
    GroundedContextRetriever,
    GroundedContextBundle,
    EvidenceRecord,
    EpistemicState,
)
from chakrview.cognition.repository.episodic_recall import (
    EpisodicMemoryRecallCoordinator,
    MemoryRecallRequest,
    RecalledContextBundle,
    RecalledMemoryItem,
    MemoryRecallStatus,
    RecallBudget,
)


class CognitiveContextSource(Enum):
    """Source classification for items entering the unified cognitive context."""
    REPOSITORY_EVIDENCE = auto()     # Static AST / dependency / test fact from current repo
    EPISODIC_MEMORY = auto()         # Recalled positive episodic solution pattern
    NEGATIVE_BOUNDARY = auto()       # Recalled or verified negative failure boundary / prohibition
    TASK_CONSTRAINT = auto()         # Explicit specification requirement or constraint


class CognitiveContextStatus(Enum):
    """Integrity and validity status of a unified cognitive context item."""
    ACTIVE = auto()                  # Verified, unconflicted, grounded active fact or rule
    NEGATIVE = auto()                # Valid negative boundary (DO_NOT_APPLY / prohibition)
    CONFLICTED = auto()              # Conflict detected with repository fact, negative boundary, or task constraint
    STALE = auto()                   # Structural drift or invalidity detected
    SUPERSEDED = auto()              # Superseded by newer confirmed pattern
    UNAVAILABLE = auto()             # Prerequisite target missing from active repository
    ABSTAIN = auto()                 # Ambiguity, cross-domain negative transfer, or ungrounded claim


@dataclass(frozen=True)
class CognitiveContextBudget:
    """Strict resource bounds to prevent context explosion and maintain deterministic composition."""
    max_evidence_items: int = 15
    max_positive_memories: int = 3
    max_negative_boundaries: int = 3
    max_total_items: int = 25
    max_symbols: int = 20
    max_modules: int = 10


@dataclass(frozen=True)
class CognitiveContextRequest:
    """
    Strongly typed specification for composing unified cognitive context.
    """
    task_description: str
    task_family: str
    target_files: Tuple[str, ...] = field(default_factory=tuple)
    target_symbols: Tuple[str, ...] = field(default_factory=tuple)
    allowed_files: Tuple[str, ...] = field(default_factory=tuple)
    explicit_constraints: Tuple[str, ...] = field(default_factory=tuple)
    repository_fingerprint: Optional[str] = None
    dependency_signature: Optional[str] = None
    language: str = "python"
    framework: str = "standard_library"
    budget: CognitiveContextBudget = field(default_factory=CognitiveContextBudget)


@dataclass(frozen=True)
class CognitiveContextItem:
    """
    An individual provenance-tracked context element.
    Every item carries complete grounding and epistemic metadata.
    """
    item_id: str
    source_type: CognitiveContextSource
    source_identifier: str
    module_reference: Optional[str]
    evidence_fingerprint: str
    status: CognitiveContextStatus
    deterministic_reason: str
    epistemic_state: EpistemicState
    originating_episode: Optional[str] = None
    symbol_reference: Optional[str] = None
    content_payload: str = ""
    ordering_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_type": self.source_type.name,
            "source_identifier": self.source_identifier,
            "module_reference": self.module_reference,
            "symbol_reference": self.symbol_reference,
            "evidence_fingerprint": self.evidence_fingerprint,
            "status": self.status.name,
            "deterministic_reason": self.deterministic_reason,
            "epistemic_state": self.epistemic_state.name,
            "originating_episode": self.originating_episode,
            "content_payload": self.content_payload,
            "ordering_key": self.ordering_key,
        }


@dataclass
class CognitiveContextTelemetry:
    """Resource and timing telemetry for context composition."""
    composition_time_ms: float = 0.0
    evidence_items_count: int = 0
    positive_memories_count: int = 0
    negative_boundaries_count: int = 0
    conflicts_quarantined_count: int = 0
    total_items_count: int = 0
    context_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition_time_ms": round(self.composition_time_ms, 2),
            "evidence_items_count": self.evidence_items_count,
            "positive_memories_count": self.positive_memories_count,
            "negative_boundaries_count": self.negative_boundaries_count,
            "conflicts_quarantined_count": self.conflicts_quarantined_count,
            "total_items_count": self.total_items_count,
            "context_fingerprint": self.context_fingerprint,
        }


@dataclass
class CognitiveContextBundle:
    """
    Composed, budget-enforced, deterministically ordered cognitive context.
    Separates positive guidance, negative boundary prohibitions, and quarantined conflicts.
    """
    request: CognitiveContextRequest
    repository_fingerprint: str
    positive_evidence: List[CognitiveContextItem] = field(default_factory=list)
    positive_memories: List[CognitiveContextItem] = field(default_factory=list)
    negative_boundaries: List[CognitiveContextItem] = field(default_factory=list)
    conflicted_items: List[CognitiveContextItem] = field(default_factory=list)
    excluded_items: List[CognitiveContextItem] = field(default_factory=list)
    active_symbols: List[str] = field(default_factory=list)
    active_modules: List[str] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: Optional[str] = None
    telemetry: CognitiveContextTelemetry = field(default_factory=CognitiveContextTelemetry)

    @property
    def has_positive_context(self) -> bool:
        return (len(self.positive_evidence) + len(self.positive_memories)) > 0

    @property
    def has_negative_boundaries(self) -> bool:
        return len(self.negative_boundaries) > 0

    @property
    def total_active_items(self) -> int:
        return len(self.positive_evidence) + len(self.positive_memories) + len(self.negative_boundaries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": {
                "task_description": self.request.task_description,
                "task_family": self.request.task_family,
                "target_files": list(self.request.target_files),
                "target_symbols": list(self.request.target_symbols),
                "allowed_files": list(self.request.allowed_files),
                "explicit_constraints": list(self.request.explicit_constraints),
            },
            "repository_fingerprint": self.repository_fingerprint,
            "positive_evidence": [it.to_dict() for it in self.positive_evidence],
            "positive_memories": [it.to_dict() for it in self.positive_memories],
            "negative_boundaries": [it.to_dict() for it in self.negative_boundaries],
            "conflicted_items": [it.to_dict() for it in self.conflicted_items],
            "excluded_items": [it.to_dict() for it in self.excluded_items],
            "active_symbols": list(self.active_symbols),
            "active_modules": list(self.active_modules),
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "telemetry": self.telemetry.to_dict(),
        }

    def to_neural_context(self) -> Dict[str, Any]:
        """
        Exposes a strictly passive, read-only dictionary view to the NeuralProposalAdapter.
        Guarantees zero mutation privileges, zero callbacks, and zero execution handles.
        """
        return {
            "task_description": self.request.task_description,
            "task_family": self.request.task_family,
            "target_files": tuple(self.request.target_files),
            "target_symbols": tuple(self.active_symbols),
            "allowed_files": tuple(self.request.allowed_files),
            "explicit_constraints": tuple(self.request.explicit_constraints),
            "repository_fingerprint": self.repository_fingerprint,
            "positive_evidence": tuple(
                {
                    "item_id": it.item_id,
                    "source": it.source_identifier,
                    "module": it.module_reference,
                    "symbol": it.symbol_reference,
                    "content": it.content_payload,
                    "reason": it.deterministic_reason,
                }
                for it in self.positive_evidence
            ),
            "positive_solution_patterns": tuple(
                {
                    "memory_id": it.source_identifier,
                    "target_modules": (it.module_reference,) if it.module_reference else (),
                    "pattern": it.content_payload,
                    "provenance_episode": it.originating_episode,
                    "reason": it.deterministic_reason,
                }
                for it in self.positive_memories
            ),
            "negative_boundaries": tuple(
                {
                    "boundary_id": it.source_identifier,
                    "module": it.module_reference,
                    "prohibition": it.content_payload,
                    "reason": it.deterministic_reason,
                }
                for it in self.negative_boundaries
            ),
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "context_fingerprint": self.telemetry.context_fingerprint,
        }


def _compute_digest(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


class UnifiedCognitiveContextComposer:
    """
    Deterministic composition engine that unites Step 64 repository grounding
    and Step 66 episodic memory recall under strict budget and provenance constraints.
    """

    def __init__(
        self,
        context_store: RepositoryContextStore,
        recall_coordinator: Optional[EpisodicMemoryRecallCoordinator] = None,
    ) -> None:
        self.context_store = context_store
        self.recall_coordinator = recall_coordinator

    def compose_context(
        self,
        request: CognitiveContextRequest,
    ) -> CognitiveContextBundle:
        start_t = time.perf_counter()
        budget = request.budget

        repo_state = self.context_store.current_state
        if not repo_state:
            raise RuntimeError("RepositoryContextStore has not been synchronized with active repository state.")

        repo_fp = repo_state.state_fingerprint

        # 1. Verification of repository fingerprint if explicitly asserted
        if request.repository_fingerprint and request.repository_fingerprint != repo_fp:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            bundle = CognitiveContextBundle(
                request=request,
                repository_fingerprint=repo_fp,
                abstained=True,
                abstain_reason=f"Repository fingerprint mismatch: requested {request.repository_fingerprint} vs current {repo_fp}",
            )
            bundle.telemetry = CognitiveContextTelemetry(
                composition_time_ms=elapsed,
                context_fingerprint=_compute_digest(f"abstained_mismatch_{repo_fp}"),
            )
            return bundle

        # 2. Gather Step 64 Grounded Evidence
        retriever = GroundedContextRetriever(
            context_store=self.context_store,
            memory_index=self.recall_coordinator.memory_index if self.recall_coordinator else None,
        )
        grounded_bundle = retriever.retrieve_context(
            task_description=request.task_description,
            target_domain=request.task_family,
        )

        # 3. Gather Step 66 Recalled Episodic Context
        recalled_bundle: Optional[RecalledContextBundle] = None
        if self.recall_coordinator:
            recall_req = MemoryRecallRequest(
                task_description=request.task_description,
                task_family=request.task_family,
                target_files=list(request.target_files),
                target_symbols=list(request.target_symbols),
                dependency_signature=request.dependency_signature,
                repository_fingerprint=repo_fp,
                language=request.language,
                framework=request.framework,
            )
            recalled_bundle = self.recall_coordinator.recall(
                request=recall_req,
                repo_state=repo_state,
            )

        # 4. Form candidate items with complete provenance
        evidence_items: List[CognitiveContextItem] = []
        for idx, ev in enumerate(grounded_bundle.evidence_records):
            item_id = f"ev_{ev.source_type.lower()}_{idx}"
            # Stable ordering key: source_type, file_path, symbol_name or empty
            ord_key = f"1_ev:{ev.source_type}:{ev.file_path}:{ev.symbol_name or ''}"
            evidence_items.append(
                CognitiveContextItem(
                    item_id=item_id,
                    source_type=CognitiveContextSource.REPOSITORY_EVIDENCE,
                    source_identifier=f"{ev.source_type}:{ev.file_path}",
                    module_reference=ev.file_path,
                    symbol_reference=ev.symbol_name,
                    evidence_fingerprint=ev.structural_digest or _compute_digest(f"{ev.file_path}:{ev.symbol_name}"),
                    status=CognitiveContextStatus.ACTIVE if ev.epistemic_state == EpistemicState.KNOWN else CognitiveContextStatus.STALE,
                    deterministic_reason=ev.retrieval_reason or "Verified grounded repository evidence",
                    epistemic_state=ev.epistemic_state,
                    content_payload=f"Grounded evidence in {ev.file_path}" + (f"::{ev.symbol_name}" if ev.symbol_name else ""),
                    ordering_key=ord_key,
                )
            )

        # Negative boundaries from episodic recall
        negative_items: List[CognitiveContextItem] = []
        if recalled_bundle:
            for neg in recalled_bundle.negative_boundaries:
                rec = neg.record
                item_id = f"neg_{rec.memory_id}"
                ord_key = f"2_neg:{rec.memory_id}:{rec.task_family}"
                primary_mod = rec.affected_modules[0] if rec.affected_modules else None
                boundary_text = "; ".join(rec.known_boundaries) if rec.known_boundaries else rec.solution_pattern
                negative_items.append(
                    CognitiveContextItem(
                        item_id=item_id,
                        source_type=CognitiveContextSource.NEGATIVE_BOUNDARY,
                        source_identifier=rec.memory_id,
                        module_reference=primary_mod,
                        evidence_fingerprint=_compute_digest(f"{rec.memory_id}:{boundary_text}"),
                        status=CognitiveContextStatus.NEGATIVE,
                        deterministic_reason=f"Verified negative failure boundary: {neg.rationale}",
                        epistemic_state=EpistemicState.KNOWN,
                        originating_episode=f"ep_{rec.memory_id}",
                        content_payload=f"DO_NOT_APPLY: {boundary_text}",
                        ordering_key=ord_key,
                    )
                )

        # Positive candidate memories
        positive_candidate_items: List[CognitiveContextItem] = []
        conflicted_items: List[CognitiveContextItem] = []
        excluded_items: List[CognitiveContextItem] = []

        if recalled_bundle:
            # First register any rejected or conflicted items from recall
            for rej in recalled_bundle.rejected_items:
                rec = rej.record
                item_id = f"rej_{rec.memory_id}"
                ord_key = f"5_rej:{rec.memory_id}"
                stat = CognitiveContextStatus.STALE
                if rej.status == MemoryRecallStatus.REJECTED_SUPERSEDED:
                    stat = CognitiveContextStatus.SUPERSEDED
                elif rej.status == MemoryRecallStatus.CONFLICTED:
                    stat = CognitiveContextStatus.CONFLICTED
                elif rej.status == MemoryRecallStatus.UNAVAILABLE:
                    stat = CognitiveContextStatus.UNAVAILABLE
                elif rej.status == MemoryRecallStatus.ABSTAIN:
                    stat = CognitiveContextStatus.ABSTAIN

                c_item = CognitiveContextItem(
                    item_id=item_id,
                    source_type=CognitiveContextSource.EPISODIC_MEMORY,
                    source_identifier=rec.memory_id,
                    module_reference=rec.affected_modules[0] if rec.affected_modules else None,
                    evidence_fingerprint=_compute_digest(f"{rec.memory_id}:{rec.solution_pattern}"),
                    status=stat,
                    deterministic_reason=rej.rationale,
                    epistemic_state=EpistemicState.CONTRADICTED if stat == CognitiveContextStatus.CONFLICTED else EpistemicState.STALE,
                    originating_episode=f"ep_{rec.memory_id}",
                    content_payload=rec.solution_pattern,
                    ordering_key=ord_key,
                )
                if stat == CognitiveContextStatus.CONFLICTED:
                    conflicted_items.append(c_item)
                else:
                    excluded_items.append(c_item)

            for pos in recalled_bundle.positive_memories:
                rec = pos.record
                item_id = f"pos_{rec.memory_id}"
                ord_key = f"3_pos:{rec.memory_id}:{rec.task_family}"
                primary_mod = rec.affected_modules[0] if rec.affected_modules else None

                # Perform deep conflict arbitration against:
                # 1) Current repository state facts (does target module exist in current repo?)
                # 2) Verified negative boundaries
                # 3) Explicit task constraints
                conflict_reason = None

                # Check repo existence
                for aff_m in rec.affected_modules:
                    if not self.context_store.file_exists(aff_m):
                        conflict_reason = f"Module '{aff_m}' in memory {rec.memory_id} does not exist in repository"
                        break

                # Check collision with negative boundaries
                if not conflict_reason:
                    for neg_it in negative_items:
                        if neg_it.module_reference and neg_it.module_reference in rec.affected_modules:
                            if "DO_NOT_APPLY" in neg_it.content_payload or "FAILURE" in neg_it.content_payload:
                                conflict_reason = f"Memory collides with verified negative boundary {neg_it.source_identifier} on {neg_it.module_reference}"
                                break

                # Check collision with explicit task constraints
                if not conflict_reason:
                    for constraint in request.explicit_constraints:
                        c_upper = constraint.upper()
                        # If constraint explicitly forbids modification of target module or pattern
                        if any(f"FORBID_{m.upper()}" in c_upper or f"NO_{m.upper()}" in c_upper for m in rec.affected_modules):
                            conflict_reason = f"Explicit task constraint '{constraint}' forbids memory target module"
                            break

                if conflict_reason:
                    conflicted_items.append(
                        CognitiveContextItem(
                            item_id=f"conf_{rec.memory_id}",
                            source_type=CognitiveContextSource.EPISODIC_MEMORY,
                            source_identifier=rec.memory_id,
                            module_reference=primary_mod,
                            evidence_fingerprint=_compute_digest(f"{rec.memory_id}:{rec.solution_pattern}"),
                            status=CognitiveContextStatus.CONFLICTED,
                            deterministic_reason=f"CONFLICT DETECTED: {conflict_reason}",
                            epistemic_state=EpistemicState.CONTRADICTED,
                            originating_episode=f"ep_{rec.memory_id}",
                            content_payload=rec.solution_pattern,
                            ordering_key=f"4_conf:{rec.memory_id}",
                        )
                    )
                else:
                    positive_candidate_items.append(
                        CognitiveContextItem(
                            item_id=item_id,
                            source_type=CognitiveContextSource.EPISODIC_MEMORY,
                            source_identifier=rec.memory_id,
                            module_reference=primary_mod,
                            evidence_fingerprint=_compute_digest(f"{rec.memory_id}:{rec.solution_pattern}"),
                            status=CognitiveContextStatus.ACTIVE,
                            deterministic_reason=pos.rationale or f"Recallable positive episodic memory with relevance score {pos.relevance_score:.4f}",
                            epistemic_state=EpistemicState.KNOWN,
                            originating_episode=f"ep_{rec.memory_id}",
                            content_payload=rec.solution_pattern,
                            ordering_key=ord_key,
                        )
                    )

        # 5. Deterministic sorting and budget enforcement
        # Deterministic sorting keys:
        evidence_items.sort(key=lambda it: it.ordering_key)
        negative_items.sort(key=lambda it: it.ordering_key)
        positive_candidate_items.sort(key=lambda it: it.ordering_key)
        conflicted_items.sort(key=lambda it: it.ordering_key)
        excluded_items.sort(key=lambda it: it.ordering_key)

        # Enforce module and symbol tracking
        collected_symbols: Set[str] = set(request.target_symbols)
        collected_modules: Set[str] = set(request.target_files)

        for ev in evidence_items:
            if ev.symbol_reference and len(collected_symbols) < budget.max_symbols:
                collected_symbols.add(ev.symbol_reference)
            if ev.module_reference and len(collected_modules) < budget.max_modules:
                collected_modules.add(ev.module_reference)

        # Apply budgets strictly
        budgeted_evidence = evidence_items[: budget.max_evidence_items]
        budgeted_negatives = negative_items[: budget.max_negative_boundaries]
        budgeted_positives = positive_candidate_items[: budget.max_positive_memories]

        # Enforce max_total_items across all active sections
        while (len(budgeted_evidence) + len(budgeted_negatives) + len(budgeted_positives)) > budget.max_total_items:
            if len(budgeted_evidence) > budget.max_evidence_items // 2:
                budgeted_evidence.pop()
            elif len(budgeted_positives) > 1:
                budgeted_positives.pop()
            else:
                budgeted_negatives.pop()

        # Check for abstention: if recall coordinator explicitly abstained
        abstained = False
        abstain_reason = None
        if recalled_bundle and recalled_bundle.abstained:
            abstained = True
            abstain_reason = recalled_bundle.abstain_reason

        # Build context fingerprint
        fp_hasher = hashlib.sha256()
        fp_hasher.update(repo_fp.encode("utf-8"))
        fp_hasher.update(request.task_description.encode("utf-8"))
        for it in budgeted_evidence:
            fp_hasher.update(it.ordering_key.encode("utf-8"))
        for it in budgeted_positives:
            fp_hasher.update(it.ordering_key.encode("utf-8"))
        for it in budgeted_negatives:
            fp_hasher.update(it.ordering_key.encode("utf-8"))
        context_fingerprint = fp_hasher.hexdigest()[:16]

        elapsed = (time.perf_counter() - start_t) * 1000.0

        telemetry = CognitiveContextTelemetry(
            composition_time_ms=elapsed,
            evidence_items_count=len(budgeted_evidence),
            positive_memories_count=len(budgeted_positives),
            negative_boundaries_count=len(budgeted_negatives),
            conflicts_quarantined_count=len(conflicted_items),
            total_items_count=len(budgeted_evidence) + len(budgeted_positives) + len(budgeted_negatives),
            context_fingerprint=context_fingerprint,
        )

        return CognitiveContextBundle(
            request=request,
            repository_fingerprint=repo_fp,
            positive_evidence=budgeted_evidence,
            positive_memories=budgeted_positives,
            negative_boundaries=budgeted_negatives,
            conflicted_items=conflicted_items,
            excluded_items=excluded_items,
            active_symbols=sorted(list(collected_symbols)),
            active_modules=sorted(list(collected_modules)),
            abstained=abstained,
            abstain_reason=abstain_reason,
            telemetry=telemetry,
        )
