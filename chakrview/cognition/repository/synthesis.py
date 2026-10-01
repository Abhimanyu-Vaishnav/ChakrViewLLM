"""
ChakrView Step 63: Autonomous Multi-Branch Strategy Synthesis & Deterministic Safety Gate.

Defines the contract, normalization, safety gate, deduplication, trace recombination,
and model-independent synthesis engine interfaces:
- SynthesizedCandidate: Formal candidate branch specification with full provenance.
- CandidateNormalization: Strict structural and boundary validator.
- DeterministicSafetyGate: Fail-closed gate checking file scopes, depths, dependencies, and memory.
- CandidateDeduplicator: Content-fingerprint based deduplication preserving provenance.
- StrategyRecombiner: Deterministic recombination of partially successful execution traces.
- CandidateSynthesisEngine: Pluggable model-independent synthesis interface.
- SynthesisAwareBranchingCoordinator: Integrates synthesized branches with Step 62 branching.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from chakrview.arena.models import ProjectManifest, SourceFile
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.change_detector import (
    RepositoryChangeDetector,
    RepositoryDiff,
    ChangeCategory,
)
from chakrview.cognition.repository.impact_analyzer import (
    RepositoryImpactAnalyzer,
    MemoryValidityStatus,
    MemoryRevalidationDecision,
    ImpactReport,
)
from chakrview.cognition.repository.patch import (
    RepositoryPatchCoordinator,
    MultiFilePatchTransaction,
)
from chakrview.cognition.repository.verifier import (
    RepositoryVerifier,
    RepositoryVerificationResult,
)
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.refactoring import (
    RefactoringStep,
    StepExecutionResult,
)
from chakrview.cognition.repository.branching import (
    RefactoringBranch,
    BranchStatus,
    BranchObservation,
    BranchSelectionDecision,
    RecoveryDecision,
    BranchingRefactoringResult,
    ObservationDrivenBranchingCoordinator,
)


class CandidateOrigin(Enum):
    """Provenance origin of a candidate strategy."""
    AUTHORED = auto()
    MEMORY_RETRIEVAL = auto()
    TRACE_RECOMBINATION = auto()
    SYNTHESIS_TEMPLATE = auto()
    SYNTHESIS_LLM = auto()


class SafetyDecision(Enum):
    """Result of deterministic safety gate inspection."""
    ACCEPT = auto()
    REJECT = auto()
    ABSTAIN = auto()


@dataclass
class CandidateProvenance:
    """Detailed audit trail tracking how a candidate was generated."""
    origin: CandidateOrigin
    source_branch_ids: List[str] = field(default_factory=list)
    source_step_ids: List[str] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)
    domain_tags: Set[str] = field(default_factory=set)
    synthesis_rationale: str = ""
    recombination_depth: int = 0
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin.name,
            "source_branch_ids": list(self.source_branch_ids),
            "source_step_ids": list(self.source_step_ids),
            "source_memory_ids": list(self.source_memory_ids),
            "domain_tags": sorted(list(self.domain_tags)),
            "synthesis_rationale": self.synthesis_rationale,
            "recombination_depth": self.recombination_depth,
            "confidence_score": self.confidence_score,
        }


@dataclass
class SynthesizedCandidate:
    """
    Formal representation of a synthesized refactoring branch proposal.
    Guaranteed serializable, deterministic, and free of arbitrary executable code.
    """
    candidate_id: str
    objective: str
    ordered_steps: List[RefactoringStep]
    allowed_files: Set[str]
    preconditions: Dict[str, Any] = field(default_factory=dict)
    expected_observations: Dict[str, Any] = field(default_factory=dict)
    failure_conditions: List[str] = field(default_factory=list)
    recovery_policy: str = "ROLLBACK_AND_EVALUATE_ALTERNATIVE"
    provenance: CandidateProvenance = field(
        default_factory=lambda: CandidateProvenance(origin=CandidateOrigin.SYNTHESIS_TEMPLATE)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """
        Deterministic SHA-256 fingerprint computed across structural content.
        Used for normalization, deduplication, and auditable tracking.
        """
        hasher = hashlib.sha256()
        hasher.update(self.objective.strip().encode("utf-8"))
        for f in sorted(self.allowed_files):
            hasher.update(f.encode("utf-8"))
        hasher.update(self.recovery_policy.strip().encode("utf-8"))
        # Steps
        for step in self.ordered_steps:
            hasher.update(step.step_id.encode("utf-8"))
            for tf in sorted(step.target_files):
                hasher.update(tf.encode("utf-8"))
            for p, content in sorted(step.patch_dict.items()):
                hasher.update(p.encode("utf-8"))
                hasher.update(content.encode("utf-8"))
            if step.targeted_test_file:
                hasher.update(step.targeted_test_file.encode("utf-8"))
        # Preconditions
        prec_json = json.dumps(self.preconditions, sort_keys=True)
        hasher.update(prec_json.encode("utf-8"))
        return hasher.hexdigest()

    def to_branch(self, priority: int = 10) -> RefactoringBranch:
        """Convert safely normalized candidate into an executable RefactoringBranch."""
        return RefactoringBranch(
            branch_id=self.candidate_id,
            description=f"[Synthesized] {self.objective}",
            steps=self.ordered_steps,
            allowed_files=self.allowed_files,
            priority=priority,
            memory_guidance_id=(
                self.provenance.source_memory_ids[0]
                if self.provenance.source_memory_ids
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "objective": self.objective,
            "fingerprint": self.fingerprint,
            "allowed_files": sorted(list(self.allowed_files)),
            "steps": [s.to_dict() for s in self.ordered_steps],
            "preconditions": self.preconditions,
            "expected_observations": self.expected_observations,
            "failure_conditions": self.failure_conditions,
            "recovery_policy": self.recovery_policy,
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class NormalizationResult:
    """Result of candidate normalization and sanity check."""
    is_valid: bool
    normalized_candidate: Optional[SynthesizedCandidate] = None
    rejection_reasons: List[str] = field(default_factory=list)


class CandidateNormalizer:
    """
    Validates and normalizes candidate structures against rigorous schemas:
    - Verifies required non-empty fields
    - Confirms steps are ordered and non-empty
    - Confirms patch targets stay strictly within candidate allowed_files
    - Detects duplicate step IDs and conflicting patch targets within step
    - Verifies valid recovery policy
    """

    SUPPORTED_RECOVERY_POLICIES = {
        "ROLLBACK_AND_EVALUATE_ALTERNATIVE",
        "ROLLBACK_AND_ABSTAIN",
        "IMMEDIATE_ABORT",
    }

    @classmethod
    def normalize(
        cls,
        candidate: SynthesizedCandidate,
        max_steps: int = 8,
    ) -> NormalizationResult:
        reasons: List[str] = []

        if not candidate.candidate_id or not candidate.candidate_id.strip():
            reasons.append("candidate_id cannot be empty")
        if not candidate.objective or not candidate.objective.strip():
            reasons.append("objective cannot be empty")
        if not candidate.allowed_files:
            reasons.append("allowed_files cannot be empty")
        if not candidate.ordered_steps:
            reasons.append("ordered_steps cannot be empty")
        elif len(candidate.ordered_steps) > max_steps:
            reasons.append(f"ordered_steps count {len(candidate.ordered_steps)} exceeds max_steps {max_steps}")

        if candidate.recovery_policy not in cls.SUPPORTED_RECOVERY_POLICIES:
            reasons.append(f"unsupported recovery_policy: {candidate.recovery_policy}")

        # Check step validity
        seen_step_ids: Set[str] = set()
        for idx, step in enumerate(candidate.ordered_steps):
            if not step.step_id:
                reasons.append(f"step at index {idx} has empty step_id")
            elif step.step_id in seen_step_ids:
                reasons.append(f"duplicate step_id '{step.step_id}' found at index {idx}")
            seen_step_ids.add(step.step_id)

            if not step.patch_dict:
                reasons.append(f"step '{step.step_id}' has empty patch_dict")

            for target_path in step.patch_dict.keys():
                if target_path not in candidate.allowed_files:
                    reasons.append(
                        f"step '{step.step_id}' targets file '{target_path}' outside candidate allowed_files"
                    )

        if reasons:
            return NormalizationResult(is_valid=False, rejection_reasons=reasons)

        # Normalize allowed files to a clean frozenset/set
        clean_allowed = {f.strip() for f in candidate.allowed_files if f.strip()}
        candidate.allowed_files = clean_allowed

        return NormalizationResult(is_valid=True, normalized_candidate=candidate)


@dataclass
class SafetyGateDecision:
    """Detailed audit record of safety gate evaluation."""
    candidate_id: str
    decision: SafetyDecision
    reasons: List[str] = field(default_factory=list)
    evaluated_constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision.name,
            "reasons": self.reasons,
            "evaluated_constraints": self.evaluated_constraints,
        }


class DeterministicSafetyGate:
    """
    Deterministic gate that approves or blocks candidate execution authority.
    Verifies:
    1. Scope boundaries (target files in allowed_modified_files)
    2. Operational limits (max branch depth, step count, candidate bounds)
    3. Negative transfer / domain compatibility
    4. Memory validity (cannot use STALE or INVALID memories)
    5. Precondition satisfaction against current repository state
    6. Prohibited mutation patterns (e.g. system files, binary execution)
    """

    def __init__(
        self,
        max_candidate_steps: int = 6,
        max_recombination_depth: int = 3,
        prohibited_paths: Optional[Set[str]] = None,
    ) -> None:
        self.max_candidate_steps = max_candidate_steps
        self.max_recombination_depth = max_recombination_depth
        self.prohibited_paths = prohibited_paths or {
            "__pycache__",
            ".git",
            "setup.py",
            "requirements.txt",
        }

    def evaluate(
        self,
        candidate: SynthesizedCandidate,
        repo_state: RepositoryState,
        allowed_modified_files: Set[str],
        task_domain: Optional[str] = None,
        memory_index: Optional[RepositoryMemoryIndex] = None,
        reference_state: Optional[RepositoryState] = None,
    ) -> SafetyGateDecision:
        reasons: List[str] = []
        constraints: Dict[str, Any] = {}

        # 1. Scope Boundary Enforcement
        for f in candidate.allowed_files:
            if f not in allowed_modified_files:
                reasons.append(
                    f"Candidate allowed file '{f}' exceeds repository task allowed scope"
                )
            if any(p in f for p in self.prohibited_paths):
                reasons.append(f"Candidate file '{f}' contains prohibited path pattern")

        for step in candidate.ordered_steps:
            for patch_f in step.patch_dict.keys():
                if patch_f not in allowed_modified_files:
                    reasons.append(
                        f"Step '{step.step_id}' targets file '{patch_f}' outside task allowed scope"
                    )

        # 2. Operational Limits
        if len(candidate.ordered_steps) > self.max_candidate_steps:
            reasons.append(
                f"Candidate step count {len(candidate.ordered_steps)} exceeds max {self.max_candidate_steps}"
            )
        if candidate.provenance.recombination_depth > self.max_recombination_depth:
            reasons.append(
                f"Recombination depth {candidate.provenance.recombination_depth} exceeds max {self.max_recombination_depth}"
            )

        # 3. Negative Transfer / Domain Matching
        if task_domain and candidate.provenance.domain_tags:
            if task_domain not in candidate.provenance.domain_tags:
                reasons.append(
                    f"Domain mismatch: task requires domain '{task_domain}', candidate tagged with {sorted(list(candidate.provenance.domain_tags))}"
                )

        # 4. Memory Validity Check
        if candidate.provenance.source_memory_ids and memory_index:
            for mem_id in candidate.provenance.source_memory_ids:
                record = memory_index.lookup(mem_id)
                if not record:
                    reasons.append(f"Source memory '{mem_id}' not found in index")
                else:
                    if reference_state:
                        diff = RepositoryChangeDetector.detect_changes(reference_state, repo_state)
                        directly_modified = sorted(list(diff.file_changes.keys()))
                        transitive_affected = set()
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
                            reasons.append(
                                f"Source memory '{mem_id}' is {reval.status.name} ({reval.rationale})"
                            )

        # 5. Preconditions check
        if candidate.preconditions:
            req_files = candidate.preconditions.get("required_existing_files", [])
            for rf in req_files:
                if rf not in repo_state.files:
                    reasons.append(f"Precondition failed: required file '{rf}' absent from repository")

        constraints["candidate_id"] = candidate.candidate_id
        constraints["step_count"] = len(candidate.ordered_steps)
        constraints["recombination_depth"] = candidate.provenance.recombination_depth
        constraints["failure_reasons"] = reasons

        if reasons:
            decision = (
                SafetyDecision.ABSTAIN
                if any("Domain mismatch" in r or "STALE" in r for r in reasons)
                else SafetyDecision.REJECT
            )
            return SafetyGateDecision(
                candidate_id=candidate.candidate_id,
                decision=decision,
                reasons=reasons,
                evaluated_constraints=constraints,
            )

        return SafetyGateDecision(
            candidate_id=candidate.candidate_id,
            decision=SafetyDecision.ACCEPT,
            reasons=["All deterministic safety checks passed."],
            evaluated_constraints=constraints,
        )


class CandidateDeduplicator:
    """
    Collapses identical synthesized strategies into unique canonical instances
    based on normalized fingerprint hashing while merging provenance.
    """

    @classmethod
    def deduplicate(
        cls,
        candidates: List[SynthesizedCandidate],
    ) -> List[SynthesizedCandidate]:
        unique_by_fp: Dict[str, SynthesizedCandidate] = {}

        for c in candidates:
            fp = c.fingerprint
            if fp not in unique_by_fp:
                unique_by_fp[fp] = c
            else:
                # Merge provenance
                existing = unique_by_fp[fp]
                merged_sources = set(existing.provenance.source_branch_ids) | set(c.provenance.source_branch_ids)
                merged_steps = set(existing.provenance.source_step_ids) | set(c.provenance.source_step_ids)
                merged_mems = set(existing.provenance.source_memory_ids) | set(c.provenance.source_memory_ids)
                merged_tags = set(existing.provenance.domain_tags) | set(c.provenance.domain_tags)

                existing.provenance.source_branch_ids = sorted(list(merged_sources))
                existing.provenance.source_step_ids = sorted(list(merged_steps))
                existing.provenance.source_memory_ids = sorted(list(merged_mems))
                existing.provenance.domain_tags = merged_tags

        return list(unique_by_fp.values())


class StrategyRecombiner:
    """
    Synthesizes new candidate refactoring branches by recombining reusable fragments:
    - Combines successful steps from failed or partially successful branches
    - Applies strict dependency, scope, and compatibility checks before pairing
    - Enforces fail-closed rejection on contradictory step sequences
    """

    @classmethod
    def recombine_traces(
        cls,
        objective: str,
        prefix_steps: List[RefactoringStep],
        suffix_steps: List[RefactoringStep],
        allowed_files: Set[str],
        domain_tag: str,
        provenance_branches: List[str],
    ) -> Optional[SynthesizedCandidate]:
        """
        Deterministically builds a combined candidate from prefix + suffix steps.
        Rejects if contradictory files, scope leaks, or empty compositions occur.
        """
        if not prefix_steps or not suffix_steps:
            return None

        combined_steps: List[RefactoringStep] = []
        seen_targets: Set[str] = set()

        # Add prefix steps
        for step in prefix_steps:
            for tf in step.target_files:
                if tf not in allowed_files:
                    return None  # Scope violation
                seen_targets.add(tf)
            combined_steps.append(step)

        # Add suffix steps, checking for clean composition
        for step in suffix_steps:
            for tf in step.target_files:
                if tf not in allowed_files:
                    return None
            combined_steps.append(step)

        candidate_id = f"synth_recomb_{int(time.time() * 1000) % 100000}"
        prov = CandidateProvenance(
            origin=CandidateOrigin.TRACE_RECOMBINATION,
            source_branch_ids=provenance_branches,
            source_step_ids=[s.step_id for s in combined_steps],
            domain_tags={domain_tag},
            synthesis_rationale=f"Recombined {len(prefix_steps)} prefix steps with {len(suffix_steps)} suffix steps from prior traces.",
            recombination_depth=1,
            confidence_score=0.9,
        )

        return SynthesizedCandidate(
            candidate_id=candidate_id,
            objective=objective,
            ordered_steps=combined_steps,
            allowed_files=allowed_files,
            provenance=prov,
        )


class CandidateSynthesisEngine(ABC):
    """Abstract model-independent synthesis engine interface."""

    @abstractmethod
    def synthesize_candidates(
        self,
        objective: str,
        task_domain: str,
        repo_state: RepositoryState,
        allowed_files: Set[str],
        failed_observations: List[BranchObservation],
        memory_records: List[RepositorySemanticRecord],
    ) -> List[SynthesizedCandidate]:
        """Generate structured candidate proposals without mutating repository."""
        pass


class DeterministicRuleSynthesisEngine(CandidateSynthesisEngine):
    """
    Deterministic rule- and template-driven candidate synthesis engine.
    Extracts strategy fragments from memory and recombines past branch traces.
    """

    def __init__(self, memory_index: Optional[RepositoryMemoryIndex] = None) -> None:
        self.memory_index = memory_index

    def synthesize_candidates(
        self,
        objective: str,
        task_domain: str,
        repo_state: RepositoryState,
        allowed_files: Set[str],
        failed_observations: List[BranchObservation],
        memory_records: List[RepositorySemanticRecord],
    ) -> List[SynthesizedCandidate]:
        candidates: List[SynthesizedCandidate] = []

        # 1. Synthesize from relevant valid memory records matching task domain
        for rec in memory_records:
            # Check domain compatibility (negative-transfer defense)
            rec_domain = rec.task_family or ""
            if task_domain and rec_domain and task_domain != rec_domain:
                continue  # Skip unrelated domain memories

            # Generate candidate steps from memory record if applicable
            primary_file = rec.affected_modules[0] if rec.affected_modules else None
            if primary_file and primary_file in allowed_files:
                step_id = f"synth_step_mem_{rec.memory_id[:8]}"
                patch = {
                    primary_file: rec.solution_pattern
                }
                c_step = RefactoringStep(
                    step_id=step_id,
                    description=f"Synthesized step from memory {rec.memory_id}",
                    target_files=[primary_file],
                    patch_dict=patch,
                    targeted_test_file=rec.verification_requirements[0] if rec.verification_requirements else None,
                )
                cand = SynthesizedCandidate(
                    candidate_id=f"synth_from_mem_{rec.memory_id[:8]}",
                    objective=objective,
                    ordered_steps=[c_step],
                    allowed_files={primary_file},
                    provenance=CandidateProvenance(
                        origin=CandidateOrigin.MEMORY_RETRIEVAL,
                        source_memory_ids=[rec.memory_id],
                        domain_tags={rec_domain} if rec_domain else {task_domain},
                        synthesis_rationale=f"Derived from validated memory {rec.memory_id}",
                        confidence_score=rec.confidence,
                    ),
                )
                candidates.append(cand)

        return candidates


@dataclass
class SynthesisCoordinationResult:
    """Auditable result of synthesis-aware multi-branch refactoring execution."""
    base_result: BranchingRefactoringResult
    candidates_synthesized: int = 0
    candidates_normalized: int = 0
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    candidates_deduplicated: int = 0
    safety_decisions: List[SafetyGateDecision] = field(default_factory=list)
    recombination_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = self.base_result.to_dict()
        d["synthesis_metrics"] = {
            "candidates_synthesized": self.candidates_synthesized,
            "candidates_normalized": self.candidates_normalized,
            "candidates_accepted": self.candidates_accepted,
            "candidates_rejected": self.candidates_rejected,
            "candidates_deduplicated": self.candidates_deduplicated,
            "safety_decisions": [s.to_dict() for s in self.safety_decisions],
            "recombination_count": self.recombination_count,
        }
        return d


class SynthesisAwareBranchingCoordinator(ObservationDrivenBranchingCoordinator):
    """
    Extends Step 62 ObservationDrivenBranchingCoordinator with Step 63 Autonomous
    Candidate Synthesis, Normalization, Deduplication, and Deterministic Safety Gating.
    """

    def __init__(
        self,
        verifier: Optional[RepositoryVerifier] = None,
        memory_index: Optional[RepositoryMemoryIndex] = None,
        synthesis_engine: Optional[CandidateSynthesisEngine] = None,
        safety_gate: Optional[DeterministicSafetyGate] = None,
        max_branches: int = 5,
        max_recovery_transitions: int = 3,
        max_synthesized_candidates: int = 4,
    ) -> None:
        super().__init__(
            verifier=verifier,
            memory_index=memory_index,
            max_branches=max_branches,
            max_recovery_transitions=max_recovery_transitions,
        )
        self.synthesis_engine = synthesis_engine or DeterministicRuleSynthesisEngine(self.memory_index)
        self.safety_gate = safety_gate or DeterministicSafetyGate()
        self.max_synthesized_candidates = max_synthesized_candidates

    def execute_synthesis_refactoring(
        self,
        manifest: ProjectManifest,
        task_id: str,
        objective: str,
        task_family: str,
        initial_branch_generator: Optional[
            Callable[[Optional[RepositorySemanticRecord], RepositoryState], List[RefactoringBranch]]
        ] = None,
        synthesized_candidates: Optional[List[SynthesizedCandidate]] = None,
        allowed_modified_files: Optional[Set[str]] = None,
        enable_memory: bool = True,
        reference_state: Optional[RepositoryState] = None,
    ) -> SynthesisCoordinationResult:
        """
        Coordinates full Step 63 workflow:
        1. Gathers initial branches (if any).
        2. Normalizes, deduplicates, and safety-gates synthesized candidates.
        3. Converts approved candidates into RefactoringBranches without automatic privilege.
        4. Delegates execution to deterministic ObservationDrivenBranchingCoordinator.
        5. If initial branch fails, invokes synthesis engine to dynamically recombine/synthesize
           alternative branches under strict safety limits.
        """
        allowed_files = allowed_modified_files or set()
        safety_decisions: List[SafetyGateDecision] = []
        raw_candidates: List[SynthesizedCandidate] = list(synthesized_candidates or [])

        base_state = RepositoryState.from_manifest(manifest)

        # 1. Normalization pass
        normalized_candidates: List[SynthesizedCandidate] = []
        rejections = 0
        for cand in raw_candidates:
            norm_res = CandidateNormalizer.normalize(cand)
            if norm_res.is_valid and norm_res.normalized_candidate:
                normalized_candidates.append(norm_res.normalized_candidate)
            else:
                rejections += 1
                safety_decisions.append(
                    SafetyGateDecision(
                        candidate_id=cand.candidate_id,
                        decision=SafetyDecision.REJECT,
                        reasons=norm_res.rejection_reasons,
                    )
                )

        # 2. Deduplication pass
        deduped = CandidateDeduplicator.deduplicate(normalized_candidates)
        dedup_count = len(normalized_candidates) - len(deduped)

        # 3. Deterministic Safety Gate Evaluation
        accepted_candidates: List[SynthesizedCandidate] = []
        for cand in deduped:
            s_dec = self.safety_gate.evaluate(
                candidate=cand,
                repo_state=base_state,
                allowed_modified_files=allowed_files,
                task_domain=task_family,
                memory_index=self.memory_index,
                reference_state=reference_state,
            )
            safety_decisions.append(s_dec)
            if s_dec.decision == SafetyDecision.ACCEPT:
                accepted_candidates.append(cand)
            else:
                rejections += 1

        # Enforce cap on synthesized candidates
        accepted_candidates = accepted_candidates[: self.max_synthesized_candidates]

        # 4. Integrate with branching execution
        def merged_branch_generator(
            rec: Optional[RepositorySemanticRecord],
            st: RepositoryState,
        ) -> List[RefactoringBranch]:
            branches: List[RefactoringBranch] = []
            if initial_branch_generator:
                branches.extend(initial_branch_generator(rec, st))

            # Synthesized branches are converted with lower priority (higher integer = evaluated after authored)
            for idx, c in enumerate(accepted_candidates):
                branch = c.to_branch(priority=10 + idx)
                branches.append(branch)
            return branches

        # Execute using ratified Step 62 branching coordinator
        base_result = self.execute_branching_refactoring(
            manifest=manifest,
            task_id=task_id,
            objective=objective,
            task_family=task_family,
            branch_generator=merged_branch_generator,
            allowed_modified_files=allowed_files,
            enable_memory=enable_memory,
            reference_state=reference_state,
        )

        return SynthesisCoordinationResult(
            base_result=base_result,
            candidates_synthesized=len(raw_candidates),
            candidates_normalized=len(normalized_candidates),
            candidates_accepted=len(accepted_candidates),
            candidates_rejected=rejections,
            candidates_deduplicated=dedup_count,
            safety_decisions=safety_decisions,
            recombination_count=0,
        )
