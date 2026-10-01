"""
ChakrView Step 71: Structured Reasoning Subsystem.

Defines:
- EpistemicCategory: Strict epistemological partitions (FACT, MEMORY, INFERENCE, HYPOTHESIS, PROPOSAL, UNCERTAINTY, UNKNOWN).
- EpistemicConfidenceState: Grounded assessment states (KNOWN, SUPPORTED, INFERRED, CONTESTED, UNCERTAIN, UNKNOWN, ABSTAINED).
- ReasoningClaim: Provenance-tracked structured statement with supporting/contradicting IDs.
- AssumptionRecord: Explicit tracking of assumptions, their criticality, and validation status.
- ObservationRecord: Concrete observation derived from repository facts or runtime checks.
- InvestigationRequirement: Structured definition of missing knowledge requiring future acquisition.
- StructuredReasoningArtifact: Inspected, deterministic reasoning product with zero private chain-of-thought exposure.
- StructuredReasoningEngine: Transforms CognitiveContextBundle and ProposalContract into a structured reasoning artifact.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextBundle,
    CognitiveContextItem,
    CognitiveContextSource,
    CognitiveContextStatus,
)
from chakrview.cognition.repository.neural_proposal import (
    ProposalContract,
    ProposalValidationStatus,
    EpistemicPartition,
    StructuredEpistemicClaim,
)


class EpistemicCategory(Enum):
    """
    Authoritative classification of reasoning elements.
    Model-generated statements never automatically become FACT.
    """
    FACT = auto()           # Grounded in direct static repository analysis (AST, dependencies, tests)
    MEMORY = auto()         # Grounded in recalled, verified historical episodic patterns
    INFERENCE = auto()      # Deductive or inductive conclusion drawn from facts and memories
    HYPOTHESIS = auto()     # Tentative explanation or proposed direction under evaluation
    PROPOSAL = auto()       # Concrete action, patch, or modification specification
    UNCERTAINTY = auto()    # Explicitly recognized parameter or boundary that is unresolved
    UNKNOWN = auto()        # Absence of required information; requires investigation


class EpistemicConfidenceState(Enum):
    """
    Distinguishes the evidence foundation of a claim without collapsing into fake probability scalars.
    """
    KNOWN = auto()          # Directly verified against active repository source
    SUPPORTED = auto()      # Corroborated by verified episodic memories or multiple facts
    INFERRED = auto()       # Logically derived from known facts, pending empirical validation
    CONTESTED = auto()      # Subject to conflicting evidence or negative boundary collision
    UNCERTAIN = auto()      # Incomplete information; boundary known but value unverified
    UNKNOWN = auto()        # Required evidence missing entirely
    ABSTAINED = auto()      # Unresolvable ambiguity, safety risk, or cross-domain negative transfer


@dataclass(frozen=True)
class ReasoningClaim:
    """
    A single inspectable claim within a structured reasoning artifact.
    Must maintain 100% provenance linkage to repository evidence or memories.
    """
    claim_id: str
    category: EpistemicCategory
    statement: str
    confidence_state: EpistemicConfidenceState
    supporting_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    supporting_memory_ids: Tuple[str, ...] = field(default_factory=tuple)
    contradicting_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    target_file: Optional[str] = None
    target_symbol: Optional[str] = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            hasher = hashlib.sha256()
            hasher.update(self.category.name.encode("utf-8"))
            hasher.update(self.statement.encode("utf-8"))
            hasher.update(self.confidence_state.name.encode("utf-8"))
            for sid in sorted(self.supporting_evidence_ids):
                hasher.update(sid.encode("utf-8"))
            for mid in sorted(self.supporting_memory_ids):
                hasher.update(mid.encode("utf-8"))
            object.__setattr__(self, "fingerprint", hasher.hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "category": self.category.name,
            "statement": self.statement,
            "confidence_state": self.confidence_state.name,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "supporting_memory_ids": list(self.supporting_memory_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class AssumptionRecord:
    """
    Explicitly tracks an assumption made during reasoning.
    """
    assumption_id: str
    description: str
    criticality: str  # "HIGH", "MEDIUM", "LOW"
    is_empirically_verified: bool = False
    verification_requirement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "description": self.description,
            "criticality": self.criticality,
            "is_empirically_verified": self.is_empirically_verified,
            "verification_requirement": self.verification_requirement,
        }


@dataclass(frozen=True)
class ObservationRecord:
    """
    Direct concrete observation extracted from the active context or repository store.
    """
    observation_id: str
    source_id: str
    content: str
    module_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "content": self.content,
            "module_reference": self.module_reference,
        }


@dataclass(frozen=True)
class InvestigationRequirement:
    """
    Specifies missing evidence or knowledge gaps for future investigation (Step 74+ interface).
    Never fabricates fake web searches or hallucinated sources.
    """
    requirement_id: str
    question: str
    knowledge_gap: str
    required_evidence_type: str
    target_module: Optional[str] = None
    urgency: str = "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "question": self.question,
            "knowledge_gap": self.knowledge_gap,
            "required_evidence_type": self.required_evidence_type,
            "target_module": self.target_module,
            "urgency": self.urgency,
        }


@dataclass(frozen=True)
class StructuredReasoningArtifact:
    """
    Canonical, deterministic reasoning artifact.
    Replaces private chain-of-thought with structured, inspectable epistemic components.
    """
    artifact_id: str
    task_id: str
    context_fingerprint: str
    repository_fingerprint: str
    claims: Tuple[ReasoningClaim, ...]
    assumptions: Tuple[AssumptionRecord, ...]
    observations: Tuple[ObservationRecord, ...]
    candidate_conclusions: Tuple[str, ...]
    unresolved_questions: Tuple[str, ...]
    investigation_requirements: Tuple[InvestigationRequirement, ...]
    overall_confidence: EpistemicConfidenceState
    is_abstained: bool
    abstain_reason: Optional[str]
    artifact_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint:
            hasher = hashlib.sha256()
            hasher.update(self.task_id.encode("utf-8"))
            hasher.update(self.context_fingerprint.encode("utf-8"))
            hasher.update(self.repository_fingerprint.encode("utf-8"))
            hasher.update(self.overall_confidence.name.encode("utf-8"))
            for c in sorted(self.claims, key=lambda x: x.claim_id):
                hasher.update(c.fingerprint.encode("utf-8"))
            for a in sorted(self.assumptions, key=lambda x: x.assumption_id):
                hasher.update(a.description.encode("utf-8"))
            object.__setattr__(self, "artifact_fingerprint", hasher.hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "context_fingerprint": self.context_fingerprint,
            "repository_fingerprint": self.repository_fingerprint,
            "claims": [c.to_dict() for c in self.claims],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "observations": [o.to_dict() for o in self.observations],
            "candidate_conclusions": list(self.candidate_conclusions),
            "unresolved_questions": list(self.unresolved_questions),
            "investigation_requirements": [r.to_dict() for r in self.investigation_requirements],
            "overall_confidence": self.overall_confidence.name,
            "is_abstained": self.is_abstained,
            "abstain_reason": self.abstain_reason,
            "artifact_fingerprint": self.artifact_fingerprint,
        }


class StructuredReasoningEngine:
    """
    Deterministic reasoning coordinator operating over CognitiveContextBundle and ProposalContract.
    Strictly separates facts, memories, hypotheses, and proposals without exposing chain-of-thought.
    """

    @classmethod
    def reason(
        cls,
        context_bundle: CognitiveContextBundle,
        proposal: Optional[ProposalContract] = None,
        context_store: Optional[RepositoryContextStore] = None,
    ) -> StructuredReasoningArtifact:
        task_id = context_bundle.request.task_description
        ctx_fp = context_bundle.telemetry.context_fingerprint
        repo_fp = context_bundle.repository_fingerprint

        # Fail-closed check if context bundle itself was abstained
        if context_bundle.abstained:
            return StructuredReasoningArtifact(
                artifact_id=f"reason_abstain_{ctx_fp[:8]}",
                task_id=task_id,
                context_fingerprint=ctx_fp,
                repository_fingerprint=repo_fp,
                claims=(),
                assumptions=(),
                observations=(),
                candidate_conclusions=(),
                unresolved_questions=("Context was abstained at source layer.",),
                investigation_requirements=(),
                overall_confidence=EpistemicConfidenceState.ABSTAINED,
                is_abstained=True,
                abstain_reason=context_bundle.abstain_reason or "Context abstained",
            )

        claims: List[ReasoningClaim] = []
        observations: List[ObservationRecord] = []
        assumptions: List[AssumptionRecord] = []
        unresolved: List[str] = []
        investigations: List[InvestigationRequirement] = []
        conclusions: List[str] = []

        # 1. Ingest positive repository evidence as FACT
        ev_counter = 1
        for ev in context_bundle.positive_evidence:
            obs = ObservationRecord(
                observation_id=f"obs_ev_{ev_counter:02d}",
                source_id=ev.item_id,
                content=ev.content_payload,
                module_reference=ev.module_reference,
            )
            observations.append(obs)

            claim = ReasoningClaim(
                claim_id=f"claim_fact_{ev_counter:02d}",
                category=EpistemicCategory.FACT,
                statement=f"Repository fact in {ev.module_reference or 'workspace'}: {ev.content_payload}",
                confidence_state=EpistemicConfidenceState.KNOWN,
                supporting_evidence_ids=(ev.item_id,),
                target_file=ev.module_reference,
            )
            claims.append(claim)
            ev_counter += 1

        # 2. Ingest recalled episodic memories as MEMORY
        mem_counter = 1
        for mem in context_bundle.positive_memories:
            obs = ObservationRecord(
                observation_id=f"obs_mem_{mem_counter:02d}",
                source_id=mem.source_identifier,
                content=mem.content_payload,
                module_reference=mem.module_reference,
            )
            observations.append(obs)

            claim = ReasoningClaim(
                claim_id=f"claim_mem_{mem_counter:02d}",
                category=EpistemicCategory.MEMORY,
                statement=f"Recalled solution pattern: {mem.content_payload}",
                confidence_state=EpistemicConfidenceState.SUPPORTED,
                supporting_memory_ids=(mem.source_identifier,),
                target_file=mem.module_reference,
            )
            claims.append(claim)
            mem_counter += 1

        # 3. Ingest negative boundaries as UNCERTAINTY / CONSTRAINTS
        neg_counter = 1
        for neg in context_bundle.negative_boundaries:
            claim = ReasoningClaim(
                claim_id=f"claim_neg_{neg_counter:02d}",
                category=EpistemicCategory.UNCERTAINTY,
                statement=f"Prohibition boundary: {neg.content_payload}",
                confidence_state=EpistemicConfidenceState.KNOWN,
                supporting_evidence_ids=(neg.source_identifier,),
            )
            claims.append(claim)
            neg_counter += 1

        # 4. Ingest ProposalContract if provided
        if proposal is not None:
            # Check proposal validation status
            if proposal.validation_status == ProposalValidationStatus.CONFLICTED:
                conclusions.append(f"Proposal conflicted: {', '.join(proposal.validation_reasons)}")
                return StructuredReasoningArtifact(
                    artifact_id=f"reason_conflicted_{proposal.proposal_id[:8]}",
                    task_id=task_id,
                    context_fingerprint=ctx_fp,
                    repository_fingerprint=repo_fp,
                    claims=tuple(claims),
                    assumptions=tuple(assumptions),
                    observations=tuple(observations),
                    candidate_conclusions=tuple(conclusions),
                    unresolved_questions=("Proposal conflicted with negative boundary or task constraint.",),
                    investigation_requirements=(),
                    overall_confidence=EpistemicConfidenceState.CONTESTED,
                    is_abstained=True,
                    abstain_reason="Proposal is conflicted with negative boundary or constraints.",
                )

            prop_counter = 1
            for path, patch_code in sorted(proposal.proposed_changes.items()):
                prop_claim = ReasoningClaim(
                    claim_id=f"claim_prop_{prop_counter:02d}",
                    category=EpistemicCategory.PROPOSAL,
                    statement=f"Proposed patch for {path}",
                    confidence_state=EpistemicConfidenceState.INFERRED,
                    supporting_evidence_ids=proposal.supporting_evidence_ids,
                    supporting_memory_ids=proposal.supporting_memory_ids,
                    target_file=path,
                )
                claims.append(prop_claim)
                prop_counter += 1

            # Ingest claims from proposal
            for idx, c in enumerate(proposal.claims, 1):
                cat = EpistemicCategory.INFERENCE
                if c.partition == EpistemicPartition.FACT_EVIDENCE:
                    cat = EpistemicCategory.FACT
                elif c.partition == EpistemicPartition.MEMORY:
                    cat = EpistemicCategory.MEMORY
                elif c.partition == EpistemicPartition.PROPOSAL:
                    cat = EpistemicCategory.PROPOSAL
                elif c.partition == EpistemicPartition.UNCERTAINTY:
                    cat = EpistemicCategory.UNCERTAINTY

                supp_ev = (c.supporting_id,) if c.supporting_id and "ev" in c.supporting_id else ()
                supp_mem = (c.supporting_id,) if c.supporting_id and "mem" in c.supporting_id else ()

                claims.append(ReasoningClaim(
                    claim_id=f"claim_contract_{idx:02d}",
                    category=cat,
                    statement=c.statement,
                    confidence_state=EpistemicConfidenceState.INFERRED if cat == EpistemicCategory.INFERENCE else EpistemicConfidenceState.SUPPORTED,
                    supporting_evidence_ids=supp_ev,
                    supporting_memory_ids=supp_mem,
                ))

            conclusions.append(f"Derived grounded patch for {', '.join(proposal.target_files)}")

        # 5. Check for missing evidence or knowledge gaps
        if not context_bundle.positive_evidence and not context_bundle.positive_memories:
            unresolved.append("No positive repository evidence or episodic memories found for task.")
            investigations.append(InvestigationRequirement(
                requirement_id="inv_gap_01",
                question=f"How should '{task_id}' be implemented without local repository evidence?",
                knowledge_gap="Missing baseline implementation or test specification.",
                required_evidence_type="REPOSITORY_AST_OR_SPEC",
                target_module=context_bundle.request.target_files[0] if context_bundle.request.target_files else None,
            ))
            return StructuredReasoningArtifact(
                artifact_id=f"reason_unknown_{ctx_fp[:8]}",
                task_id=task_id,
                context_fingerprint=ctx_fp,
                repository_fingerprint=repo_fp,
                claims=tuple(claims),
                assumptions=tuple(assumptions),
                observations=tuple(observations),
                candidate_conclusions=("Insufficient evidence to form a deterministic proposal.",),
                unresolved_questions=tuple(unresolved),
                investigation_requirements=tuple(investigations),
                overall_confidence=EpistemicConfidenceState.UNKNOWN,
                is_abstained=True,
                abstain_reason="Insufficient evidence: context contains zero positive facts or memories.",
            )

        # Baseline assumption: Local workspace files conform to static AST specifications
        assumptions.append(AssumptionRecord(
            assumption_id="asm_01",
            description="Existing code syntax conforms to standard Python runtime grammar.",
            criticality="HIGH",
            is_empirically_verified=True,
            verification_requirement="Pass AST parse check.",
        ))

        overall_conf = EpistemicConfidenceState.SUPPORTED if context_bundle.positive_memories else EpistemicConfidenceState.INFERRED
        if not claims:
            overall_conf = EpistemicConfidenceState.UNKNOWN

        artifact_id = f"reason_{hashlib.sha256((task_id + ctx_fp).encode('utf-8')).hexdigest()[:10]}"

        return StructuredReasoningArtifact(
            artifact_id=artifact_id,
            task_id=task_id,
            context_fingerprint=ctx_fp,
            repository_fingerprint=repo_fp,
            claims=tuple(claims),
            assumptions=tuple(assumptions),
            observations=tuple(observations),
            candidate_conclusions=tuple(conclusions) if conclusions else ("Reasoning artifact constructed.",),
            unresolved_questions=tuple(unresolved),
            investigation_requirements=tuple(investigations),
            overall_confidence=overall_conf,
            is_abstained=False,
            abstain_reason=None,
        )
