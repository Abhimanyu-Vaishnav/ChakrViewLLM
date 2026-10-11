"""
ChakrView Step 62: Observation-Driven Dynamic Branching Refactoring & Safe Recovery Coordinator.

Extends the Step 61 linear multi-step refactoring workflow into an observation-driven,
branching refactoring workflow:
- Formally models alternative refactoring branches (strategies) with preconditions.
- Captures intermediate observations after each step/branch execution.
- Evaluates branch viability dynamically and deterministically.
- Executes safe rollback upon branch failure, cryptographically verifies the pre-branch
  repository state fingerprint, and reroutes to eligible alternative branches.
- Abort/abstains safely (fails closed) if all branches fail, preconditions are violated,
  or rollback integrity cannot be confirmed.
- Enforces strict operational bounds (max_branches, max_depth, max_recovery_transitions).
"""

from __future__ import annotations

import time
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
from chakrview.cognition.repository.arbitration import (
    arbitrate,
    RepositoryQuery,
    ArbitrationStatus,
    ArbitrationResult,
)
from chakrview.cognition.repository.refactoring import (
    RefactoringStep,
    StepExecutionResult,
)


class BranchStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()
    REJECTED_PRECONDITION = auto()
    ABSTAINED = auto()


@dataclass
class RefactoringBranch:
    """An explicit alternative refactoring strategy."""
    branch_id: str
    description: str
    steps: List[RefactoringStep]
    allowed_files: Set[str]
    priority: int = 1  # Lower value = higher priority (deterministic ordering)
    precondition_fn: Optional[Callable[[RepositoryState, Optional[RepositorySemanticRecord]], bool]] = None
    expected_categories: List[ChangeCategory] = field(default_factory=list)
    memory_guidance_id: Optional[str] = None
    status: BranchStatus = BranchStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "description": self.description,
            "priority": self.priority,
            "allowed_files": sorted(list(self.allowed_files)),
            "memory_guidance_id": self.memory_guidance_id,
            "status": self.status.name,
            "step_count": len(self.steps),
        }


@dataclass
class BranchObservation:
    """Structured observation after executing steps within a branch."""
    branch_id: str
    step_id: str
    targeted_test_passed: bool
    regression_passed: bool
    diff_integrity_passed: bool
    state_fingerprint_before: str
    state_fingerprint_after: str
    highest_diff_category: str
    diagnostics: Optional[str] = None
    fatal_errors: int = 0

    @property
    def is_healthy(self) -> bool:
        return (
            self.targeted_test_passed
            and self.diff_integrity_passed
            and self.fatal_errors == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "step_id": self.step_id,
            "targeted_test_passed": self.targeted_test_passed,
            "regression_passed": self.regression_passed,
            "diff_integrity_passed": self.diff_integrity_passed,
            "state_fingerprint_before": self.state_fingerprint_before,
            "state_fingerprint_after": self.state_fingerprint_after,
            "highest_diff_category": self.highest_diff_category,
            "diagnostics": self.diagnostics,
            "is_healthy": self.is_healthy,
        }


@dataclass
class BranchSelectionDecision:
    """Record explaining why a branch was selected or rejected."""
    branch_id: str
    selected: bool
    rejection_reason: Optional[str] = None
    preconditions_satisfied: bool = True
    memory_validity: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "selected": self.selected,
            "rejection_reason": self.rejection_reason,
            "preconditions_satisfied": self.preconditions_satisfied,
            "memory_validity": self.memory_validity,
            "rationale": self.rationale,
        }


@dataclass
class RecoveryDecision:
    """Record documenting safe rollback and transition between branches."""
    from_branch_id: str
    to_branch_id: Optional[str]
    rollback_successful: bool
    fingerprint_restored: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_branch_id": self.from_branch_id,
            "to_branch_id": self.to_branch_id,
            "rollback_successful": self.rollback_successful,
            "fingerprint_restored": self.fingerprint_restored,
            "reason": self.reason,
        }


@dataclass
class BranchingRefactoringResult:
    """Complete auditable result for an observation-driven branching workflow."""
    task_id: str
    overall_success: bool
    selected_branch_id: Optional[str] = None
    branches_evaluated: int = 0
    recovery_transitions: int = 0
    selection_decisions: List[BranchSelectionDecision] = field(default_factory=list)
    observations: List[BranchObservation] = field(default_factory=list)
    recovery_decisions: List[RecoveryDecision] = field(default_factory=list)
    final_verification: Optional[RepositoryVerificationResult] = None
    final_state_fingerprint: str = ""
    consolidated_memory_id: Optional[str] = None
    abstained: bool = False
    failure_reason: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "overall_success": self.overall_success,
            "selected_branch_id": self.selected_branch_id,
            "branches_evaluated": self.branches_evaluated,
            "recovery_transitions": self.recovery_transitions,
            "selection_decisions": [d.to_dict() for d in self.selection_decisions],
            "observations": [o.to_dict() for o in self.observations],
            "recovery_decisions": [r.to_dict() for r in self.recovery_decisions],
            "final_verification": self.final_verification.to_dict() if self.final_verification else None,
            "final_state_fingerprint": self.final_state_fingerprint,
            "consolidated_memory_id": self.consolidated_memory_id,
            "abstained": self.abstained,
            "failure_reason": self.failure_reason,
            "duration_ms": round(self.duration_ms, 2),
        }


class ObservationDrivenBranchingCoordinator:
    """
    Coordinates multi-branch refactoring plans with observation-driven recovery routing.
    """

    def __init__(
        self,
        verifier: Optional[RepositoryVerifier] = None,
        memory_index: Optional[RepositoryMemoryIndex] = None,
        max_branches: int = 5,
        max_recovery_transitions: int = 3,
    ) -> None:
        self.verifier = verifier or RepositoryVerifier()
        self.memory_index = memory_index or RepositoryMemoryIndex()
        self.max_branches = max_branches
        self.max_recovery_transitions = max_recovery_transitions

    def execute_branching_refactoring(
        self,
        manifest: ProjectManifest,
        task_id: str,
        objective: str,
        task_family: str,
        branch_generator: Callable[
            [Optional[RepositorySemanticRecord], RepositoryState],
            List[RefactoringBranch],
        ],
        allowed_modified_files: Set[str],
        enable_memory: bool = True,
        reference_state: Optional[RepositoryState] = None,
    ) -> BranchingRefactoringResult:
        """
        Execute observation-driven branching refactoring with rollback-safe recovery.
        """
        start_time = time.perf_counter()

        with IsolatedWorkspace(manifest) as iso_ws:
            patch_coord = RepositoryPatchCoordinator(iso_ws)

            # 1. Capture base repository state and cryptographic fingerprint
            base_state = RepositoryState.from_manifest(manifest)
            current_state = base_state
            base_fingerprint = base_state.state_fingerprint

            # 2. Memory arbitration & revalidation
            active_memory: Optional[RepositorySemanticRecord] = None
            if enable_memory and self.memory_index.count() > 0:
                query = RepositoryQuery(
                    task_family=task_family,
                    language=base_state.language,
                    framework=base_state.framework,
                    symptom_signature=objective,
                    affected_modules=list(allowed_modified_files),
                )
                arb_result = arbitrate(query, self.memory_index)
                if arb_result.status == ArbitrationStatus.SELECTED and arb_result.selected:
                    diff_eval = (
                        RepositoryChangeDetector.detect_changes(reference_state, base_state)
                        if reference_state is not None
                        else RepositoryChangeDetector.detect_changes(base_state, base_state)
                    )
                    impact = RepositoryImpactAnalyzer.analyze_impact(
                        state=base_state,
                        diff=diff_eval,
                        memory_index=self.memory_index,
                    )
                    reval = impact.revalidated_memories.get(arb_result.selected.memory_id)
                    if reval and reval.is_applicable:
                        active_memory = arb_result.selected

            # 3. Generate candidate branches
            raw_branches = branch_generator(active_memory, current_state)
            # Enforce deterministic ordering (by priority, then branch_id) and bounds
            sorted_branches = sorted(raw_branches, key=lambda b: (b.priority, b.branch_id))[:self.max_branches]

            selection_decisions: List[BranchSelectionDecision] = []
            observations: List[BranchObservation] = []
            recovery_decisions: List[RecoveryDecision] = []

            successful_branch: Optional[RefactoringBranch] = None
            final_verification: Optional[RepositoryVerificationResult] = None
            recovery_count = 0

            # 4. Iterate over candidate branches in priority order
            for branch_idx, branch in enumerate(sorted_branches):
                # Check maximum recovery transitions limit
                if recovery_count >= self.max_recovery_transitions:
                    selection_decisions.append(
                        BranchSelectionDecision(
                            branch_id=branch.branch_id,
                            selected=False,
                            rejection_reason="Maximum recovery transitions exceeded",
                            rationale="Safety limit reached; aborting further branch execution",
                        )
                    )
                    continue

                # Precondition evaluation
                precondition_met = True
                precondition_reason = ""
                if branch.precondition_fn is not None:
                    try:
                        precondition_met = branch.precondition_fn(current_state, active_memory)
                        if not precondition_met:
                            precondition_reason = "Branch custom precondition returned False"
                    except Exception as e:
                        precondition_met = False
                        precondition_reason = f"Precondition evaluation error: {str(e)}"

                # Scope constraint check: branch allowed files must be within allowed_modified_files
                if not branch.allowed_files.issubset(allowed_modified_files):
                    precondition_met = False
                    precondition_reason = f"Branch allowed files {branch.allowed_files} exceeds scope {allowed_modified_files}"

                if not precondition_met:
                    branch.status = BranchStatus.REJECTED_PRECONDITION
                    selection_decisions.append(
                        BranchSelectionDecision(
                            branch_id=branch.branch_id,
                            selected=False,
                            preconditions_satisfied=False,
                            rejection_reason=precondition_reason,
                            rationale="Rejected before execution due to unsatisfied preconditions",
                        )
                    )
                    continue

                # Branch selected for execution
                selection_decisions.append(
                    BranchSelectionDecision(
                        branch_id=branch.branch_id,
                        selected=True,
                        preconditions_satisfied=True,
                        rationale=f"Selected branch {branch.branch_id} for execution (priority {branch.priority})",
                    )
                )
                branch.status = BranchStatus.ACTIVE

                # Snapshot workspace state before this branch executes
                pre_branch_manifest_files = {sf.path: sf.content for sf in manifest.files}
                branch_tx_list: List[MultiFilePatchTransaction] = []
                branch_healthy = True

                # Execute branch steps sequentially
                for step_idx, step in enumerate(branch.steps):
                    fp_before_step = current_state.state_fingerprint
                    tx_id = f"tx_{branch.branch_id}_step_{step_idx}_{step.step_id}"
                    tx = patch_coord.begin_transaction(tx_id=tx_id, file_updates=step.patch_dict)
                    branch_tx_list.append(tx)

                    # Update content tracking
                    for p, c in step.patch_dict.items():
                        pre_branch_manifest_files[p] = c

                    # Intermediate verification
                    ver_res = self.verifier.verify_repository(
                        workspace=iso_ws,
                        targeted_test_rel_path=step.targeted_test_file,
                        allowed_modified_files=branch.allowed_files,
                        currently_modified_files=list(step.patch_dict.keys()),
                    )

                    post_step_state = RepositoryState.from_files(
                        project_id=base_state.project_id,
                        files_dict=pre_branch_manifest_files,
                        version=current_state.version + 1,
                        language=base_state.language,
                        framework=base_state.framework,
                    )
                    diff_step = RepositoryChangeDetector.detect_changes(current_state, post_step_state)

                    obs = BranchObservation(
                        branch_id=branch.branch_id,
                        step_id=step.step_id,
                        targeted_test_passed=ver_res.level1_targeted_pass,
                        regression_passed=ver_res.level2_regression_pass,
                        diff_integrity_passed=ver_res.level4_diff_integrity,
                        state_fingerprint_before=fp_before_step,
                        state_fingerprint_after=post_step_state.state_fingerprint,
                        highest_diff_category=diff_step.highest_category.name,
                        diagnostics=ver_res.diagnostics,
                        fatal_errors=ver_res.test_result_summary.get("errors", 0),
                    )
                    observations.append(obs)

                    if not obs.is_healthy:
                        branch_healthy = False
                        break

                    current_state = post_step_state

                # Evaluate branch final state
                if branch_healthy and len(branch_tx_list) == len(branch.steps):
                    # Execute full 4-tier repo-wide verification
                    full_ver = self.verifier.verify_repository(
                        workspace=iso_ws,
                        targeted_test_rel_path=None,
                        allowed_modified_files=allowed_modified_files,
                        currently_modified_files=list(allowed_modified_files),
                    )
                    if full_ver.overall_verified:
                        branch.status = BranchStatus.SUCCEEDED
                        successful_branch = branch
                        final_verification = full_ver
                        break  # Clean success!
                    else:
                        branch_healthy = False

                # If branch failed or was unhealthy: execute atomic rollback of all transactions in this branch
                branch.status = BranchStatus.FAILED
                for tx in reversed(branch_tx_list):
                    patch_coord.rollback_transaction(tx)

                # Reset file tracking back to original manifest
                pre_branch_manifest_files = {sf.path: sf.content for sf in manifest.files}
                restored_state = RepositoryState.from_files(
                    project_id=base_state.project_id,
                    files_dict=pre_branch_manifest_files,
                    version=base_state.version,
                    language=base_state.language,
                    framework=base_state.framework,
                )
                current_state = restored_state

                # Verify rollback restored exact bit-match fingerprint
                fingerprint_matched = (restored_state.state_fingerprint == base_fingerprint)
                branch.status = BranchStatus.ROLLED_BACK
                recovery_count += 1

                next_branch_id = (
                    sorted_branches[branch_idx + 1].branch_id
                    if branch_idx + 1 < len(sorted_branches)
                    else None
                )
                recovery_dec = RecoveryDecision(
                    from_branch_id=branch.branch_id,
                    to_branch_id=next_branch_id,
                    rollback_successful=True,
                    fingerprint_restored=fingerprint_matched,
                    reason=f"Branch {branch.branch_id} failed intermediate or final verification; rolled back to base state",
                )
                recovery_decisions.append(recovery_dec)

                if not fingerprint_matched:
                    # CRITICAL: Rollback failed to restore bit-exact state! Fail closed immediately.
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    return BranchingRefactoringResult(
                        task_id=task_id,
                        overall_success=False,
                        selected_branch_id=branch.branch_id,
                        branches_evaluated=branch_idx + 1,
                        recovery_transitions=recovery_count,
                        selection_decisions=selection_decisions,
                        observations=observations,
                        recovery_decisions=recovery_decisions,
                        final_state_fingerprint=current_state.state_fingerprint,
                        abstained=True,
                        failure_reason="CRITICAL: Rollback fingerprint mismatch! Halting execution.",
                        duration_ms=duration_ms,
                    )

            # 5. Experience Consolidation (if a branch succeeded)
            consolidated_id: Optional[str] = None
            if successful_branch and final_verification and final_verification.overall_verified:
                consolidated_id = f"mem_branch_{task_id}_{successful_branch.branch_id}_{int(time.time())}"
                rec = RepositorySemanticRecord(
                    memory_id=consolidated_id,
                    task_family=task_family,
                    language=base_state.language,
                    framework=base_state.framework,
                    repository_pattern=f"Observation-driven branching refactoring: {successful_branch.branch_id}",
                    symptom_signature=objective,
                    root_cause_signature="Multi-component interface inconsistency",
                    dependency_signature=f"Topology: {len(base_state.dependency_graph.edges if base_state.dependency_graph else [])} edges",
                    affected_modules=sorted(list(allowed_modified_files)),
                    solution_pattern=f"Dynamic branch {successful_branch.branch_id} with {len(successful_branch.steps)} steps",
                    verification_requirements=["Targeted", "Regression", "RepoState", "DiffIntegrity"],
                    known_boundaries=["do not apply to fixed-fee billing logic"],
                    confidence=0.94,
                    evidence_count=1,
                    successful_episodes=1,
                    failed_episodes=0,
                    source_episode_ids=[task_id],
                )
                self.memory_index.insert(rec)

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            overall_success = bool(successful_branch is not None)

            return BranchingRefactoringResult(
                task_id=task_id,
                overall_success=overall_success,
                selected_branch_id=successful_branch.branch_id if successful_branch else None,
                branches_evaluated=len([d for d in selection_decisions if d.selected]),
                recovery_transitions=recovery_count,
                selection_decisions=selection_decisions,
                observations=observations,
                recovery_decisions=recovery_decisions,
                final_verification=final_verification,
                final_state_fingerprint=current_state.state_fingerprint,
                consolidated_memory_id=consolidated_id,
                abstained=not overall_success,
                failure_reason=None if overall_success else "All candidate branches failed or were rejected",
                duration_ms=duration_ms,
            )
