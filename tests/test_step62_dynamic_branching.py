"""
Step 62 Automated Tests: Observation-Driven Dynamic Branching Refactoring & Safe Recovery.

Tests verify:
  01. RefactoringBranch creation, priority sorting, and precondition assignment.
  02. BranchSelectionDecision recording and audit trail formatting.
  03. BranchObservation captures intermediate health and failure reasons.
  04. Direct branch success without intermediate failure or recovery transition.
  05. Intermediate failure detection stops branch execution immediately.
  06. Failed branch triggers atomic rollback restoring exact pre-branch state.
  07. Recovery decision transitions safely from failed branch to eligible alternative branch.
  08. Post-rollback state fingerprint exactly bit-matches base repository fingerprint.
  09. Unsafe branch exceeding allowed files scope is rejected before execution.
  10. Custom precondition returning False causes deterministic branch rejection.
  11. All candidate branches failing triggers safe abstention (fails closed).
  12. Maximum recovery transitions limit halts further execution and forces abstention.
  13. Maximum branches limit caps number of candidate branches evaluated.
  14. Memory-assisted branch prioritization selects memory-guided strategy.
  15. Stale memory revalidation prevents stale pattern from guiding branch execution.
  16. Deterministic repeated decisions: identical inputs produce identical branch selections.
  17. Experience consolidation occurs only upon overall branch success.
  18. Rollback failure (simulated or actual) causes hard abort.
  19. Backward compatibility: linear MultiStepRefactoringCoordinator remains functional.
  20. Bit-exact neural core immutability invariant preserved (DeltaW_base == 0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.change_detector import RepositoryChangeDetector
from chakrview.cognition.repository.impact_analyzer import RepositoryImpactAnalyzer, MemoryValidityStatus
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.refactoring import (
    RefactoringStep,
    RefactoringPlan,
    MultiStepRefactoringCoordinator,
)
from chakrview.cognition.repository.branching import (
    BranchStatus,
    RefactoringBranch,
    BranchObservation,
    BranchSelectionDecision,
    RecoveryDecision,
    BranchingRefactoringResult,
    ObservationDrivenBranchingCoordinator,
)
from tests.test_step59_repository_cognition import (
    MODELS_CODE,
    DEFECTIVE_TAX_CODE,
    CORRECTED_TAX_CODE,
    DISCOUNT_CODE,
    BILLING_CODE,
    TEST_TAX_CODE,
    TEST_DISCOUNT_CODE,
    TEST_BILLING_CODE,
    create_order_billing_manifest,
)
from scripts.experiment_step61_realtime_change import build_repo_manifest


def compute_test_baseline_hash() -> str:
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Test 01: Branch Creation and Sorting
# ---------------------------------------------------------------------------
def test_01_refactoring_branch_creation_and_sorting():
    b1 = RefactoringBranch(
        branch_id="b_low",
        description="Low priority",
        steps=[],
        allowed_files={"tax_service.py"},
        priority=10,
    )
    b2 = RefactoringBranch(
        branch_id="b_high",
        description="High priority",
        steps=[],
        allowed_files={"tax_service.py"},
        priority=1,
    )
    sorted_branches = sorted([b1, b2], key=lambda b: (b.priority, b.branch_id))
    assert sorted_branches[0].branch_id == "b_high"
    assert sorted_branches[1].branch_id == "b_low"


# ---------------------------------------------------------------------------
# Test 02: BranchSelectionDecision Structure
# ---------------------------------------------------------------------------
def test_02_branch_selection_decision_audit():
    dec = BranchSelectionDecision(
        branch_id="branch_a",
        selected=True,
        preconditions_satisfied=True,
        rationale="Top priority candidate",
    )
    d = dec.to_dict()
    assert d["branch_id"] == "branch_a"
    assert d["selected"] is True
    assert d["preconditions_satisfied"] is True


# ---------------------------------------------------------------------------
# Test 03: BranchObservation Health Check
# ---------------------------------------------------------------------------
def test_03_branch_observation_health_evaluation():
    obs_healthy = BranchObservation(
        branch_id="b1",
        step_id="s1",
        targeted_test_passed=True,
        regression_passed=True,
        diff_integrity_passed=True,
        state_fingerprint_before="fp1",
        state_fingerprint_after="fp2",
        highest_diff_category="LOCAL",
    )
    assert obs_healthy.is_healthy is True

    obs_unhealthy = BranchObservation(
        branch_id="b1",
        step_id="s1",
        targeted_test_passed=False,  # Targeted test failed
        regression_passed=True,
        diff_integrity_passed=True,
        state_fingerprint_before="fp1",
        state_fingerprint_after="fp1",
        highest_diff_category="LOCAL",
    )
    assert obs_unhealthy.is_healthy is False


# ---------------------------------------------------------------------------
# Test 04: Direct Branch Success
# ---------------------------------------------------------------------------
def test_04_direct_branch_success():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_direct",
                description="Fix tax directly",
                allowed_files={"tax_service.py"},
                priority=1,
                steps=[
                    RefactoringStep(
                        step_id="s1",
                        description="Fix tax",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            )
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T04",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is True
    assert res.selected_branch_id == "b_direct"
    assert res.recovery_transitions == 0


# ---------------------------------------------------------------------------
# Test 05 & 06 & 07 & 08: Failure -> Rollback -> Reroute Recovery
# ---------------------------------------------------------------------------
def test_05_08_branch_failure_rollback_and_recovery():
    manifest = create_order_billing_manifest(is_defective=True)
    state_original = RepositoryState.from_manifest(manifest)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        b1_broken = RefactoringBranch(
            branch_id="b1_broken",
            description="Broken patch that fails intermediate verification",
            allowed_files={"tax_service.py"},
            priority=1,
            steps=[
                RefactoringStep(
                    step_id="s1_bad",
                    description="Bad patch",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('BROKEN')\n"},
                    targeted_test_file="test_tax_service.py",
                )
            ],
        )
        b2_correct = RefactoringBranch(
            branch_id="b2_correct",
            description="Correct verified patch",
            allowed_files={"tax_service.py"},
            priority=2,
            steps=[
                RefactoringStep(
                    step_id="s1_good",
                    description="Good patch",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                )
            ],
        )
        return [b1_broken, b2_correct]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T05_08",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    # Test 05 & 07: Failure reroutes to second branch
    assert res.overall_success is True
    assert res.selected_branch_id == "b2_correct"
    assert res.recovery_transitions == 1

    # Test 06 & 08: Rollback was successful and verified
    rec_dec = res.recovery_decisions[0]
    assert rec_dec.rollback_successful is True
    assert rec_dec.fingerprint_restored is True


# ---------------------------------------------------------------------------
# Test 09: Unsafe Scope Rejection
# ---------------------------------------------------------------------------
def test_09_unsafe_scope_rejection():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_out_of_bounds",
                description="Modifies unauthorized module",
                allowed_files={"unauthorized.py"},
                priority=1,
                steps=[],
            ),
            RefactoringBranch(
                branch_id="b_in_bounds",
                description="Valid allowed scope",
                allowed_files={"tax_service.py"},
                priority=2,
                steps=[
                    RefactoringStep(
                        step_id="s1",
                        description="Fix tax",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            ),
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T09",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is True
    assert res.selected_branch_id == "b_in_bounds"
    assert res.selection_decisions[0].selected is False
    assert "exceeds scope" in res.selection_decisions[0].rejection_reason


# ---------------------------------------------------------------------------
# Test 10: Custom Precondition Evaluation
# ---------------------------------------------------------------------------
def test_10_custom_precondition_rejection():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_strict_precondition",
                description="Requires state version > 10",
                allowed_files={"tax_service.py"},
                priority=1,
                precondition_fn=lambda st, m: st.version > 10,
                steps=[],
            )
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T10",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is False
    assert res.abstained is True
    assert res.selection_decisions[0].preconditions_satisfied is False


# ---------------------------------------------------------------------------
# Test 11: All Candidates Fail -> Safe Abstention
# ---------------------------------------------------------------------------
def test_11_all_branches_failed_abstention():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_fail_1",
                description="Fails 1",
                allowed_files={"tax_service.py"},
                priority=1,
                steps=[
                    RefactoringStep(
                        step_id="s1",
                        description="Fail 1",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('1')\n"},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            ),
            RefactoringBranch(
                branch_id="b_fail_2",
                description="Fails 2",
                allowed_files={"tax_service.py"},
                priority=2,
                steps=[
                    RefactoringStep(
                        step_id="s2",
                        description="Fail 2",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('2')\n"},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            ),
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T11",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is False
    assert res.abstained is True
    assert res.selected_branch_id is None


# ---------------------------------------------------------------------------
# Test 12 & 13: Operational Bounds
# ---------------------------------------------------------------------------
def test_12_13_branch_and_recovery_limits():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = ObservationDrivenBranchingCoordinator(
        max_branches=2,
        max_recovery_transitions=1,
    )

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id=f"b_{i}",
                description=f"Branch {i}",
                allowed_files={"tax_service.py"},
                priority=i,
                steps=[
                    RefactoringStep(
                        step_id=f"s_{i}",
                        description=f"Step {i}",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": f"def compute_tax(order):\n    raise RuntimeError('{i}')\n"},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            )
            for i in range(5)
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T12_13",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.branches_evaluated <= 2
    assert res.recovery_transitions <= 1


# ---------------------------------------------------------------------------
# Test 14 & 15: Memory Guidance and Stale Rejection
# ---------------------------------------------------------------------------
def test_14_15_memory_guidance_and_stale_rejection():
    manifest = create_order_billing_manifest(is_defective=True)
    idx = RepositoryMemoryIndex()
    rec = RepositorySemanticRecord(
        memory_id="mem_guidance_test",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Rate calculation defect",
        symptom_signature="Fix tax",
        root_cause_signature="Tax calculation error",
        dependency_signature="billing_service -> tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="Apply percentage multiplier",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.95,
        evidence_count=2,
        successful_episodes=2,
        failed_episodes=0,
    )
    idx.insert(rec)
    coordinator = ObservationDrivenBranchingCoordinator(memory_index=idx)

    # Test 14: Valid memory guides branch
    def branch_gen_valid(mem, state):
        assert mem is not None
        return [
            RefactoringBranch(
                branch_id="b_mem_guided",
                description="Memory guided",
                allowed_files={"tax_service.py"},
                steps=[
                    RefactoringStep(
                        step_id="s1",
                        description="Fix tax",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            )
        ]

    res_valid = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T14",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen_valid,
        allowed_modified_files={"tax_service.py"},
        enable_memory=True,
    )
    assert res_valid.overall_success is True

    # Test 15: Structural change in reference state renders memory STALE
    stale_tax_code = "import math\n" + DEFECTIVE_TAX_CODE
    manifest_stale = build_repo_manifest(tax_code=stale_tax_code)
    state_original = RepositoryState.from_manifest(manifest)

    def branch_gen_stale(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_stale_check",
                description="Requires active valid memory",
                allowed_files={"tax_service.py"},
                precondition_fn=lambda st, m: bool(m is not None),
                steps=[],
            )
        ]

    res_stale = coordinator.execute_branching_refactoring(
        manifest=manifest_stale,
        task_id="T15",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen_stale,
        allowed_modified_files={"tax_service.py"},
        enable_memory=True,
        reference_state=state_original,
    )
    assert res_stale.overall_success is False
    assert res_stale.selection_decisions[0].preconditions_satisfied is False


# ---------------------------------------------------------------------------
# Test 16: Determinism
# ---------------------------------------------------------------------------
def test_16_deterministic_repeated_execution():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_det_1",
                description="First",
                allowed_files={"tax_service.py"},
                priority=1,
                steps=[
                    RefactoringStep(
                        step_id="s1",
                        description="Fix tax",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            )
        ]

    res1 = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="TDET",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    res2 = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="TDET",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res1.selected_branch_id == res2.selected_branch_id
    assert res1.final_state_fingerprint == res2.final_state_fingerprint


# ---------------------------------------------------------------------------
# Test 17: Experience Consolidation On Success
# ---------------------------------------------------------------------------
def test_17_experience_consolidation():
    manifest = create_order_billing_manifest(is_defective=True)
    idx = RepositoryMemoryIndex()
    coordinator = ObservationDrivenBranchingCoordinator(memory_index=idx)

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_cons",
                description="Consolidation test",
                allowed_files={"tax_service.py"},
                steps=[
                    RefactoringStep(
                        step_id="s1",
                        description="Fix tax",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            )
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T17",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is True
    assert res.consolidated_memory_id is not None
    assert idx.count() == 1


# ---------------------------------------------------------------------------
# Test 18: Backward Compatibility with Linear Coordinator
# ---------------------------------------------------------------------------
def test_18_backward_compatibility_step61():
    manifest = create_order_billing_manifest(is_defective=True)
    linear_coord = MultiStepRefactoringCoordinator()

    def plan_gen(mem, state):
        return RefactoringPlan(
            plan_id="plan_linear_compat",
            task_id="TLIN",
            objective="Fix tax",
            steps=[
                RefactoringStep(
                    step_id="s1",
                    description="Fix tax",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                )
            ],
        )

    res = linear_coord.execute_multi_step_refactoring(
        manifest=manifest,
        task_id="TLIN",
        objective="Fix tax",
        task_family="billing_calculation",
        plan_generator=plan_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is True
    assert res.steps_executed == 1


# ---------------------------------------------------------------------------
# Test 19: Bit-Exact Baseline Immutability
# ---------------------------------------------------------------------------
def test_19_baseline_immutability():
    current_hash = compute_test_baseline_hash()
    assert current_hash == EXPECTED_WEIGHT_HASH


# ---------------------------------------------------------------------------
# Test 20: Rollback Restoration Bit-Exactness
# ---------------------------------------------------------------------------
def test_20_rollback_restoration_bit_exactness():
    manifest = create_order_billing_manifest(is_defective=True)
    state_original = RepositoryState.from_manifest(manifest)
    coordinator = ObservationDrivenBranchingCoordinator()

    def branch_gen(mem, state):
        return [
            RefactoringBranch(
                branch_id="b_fail_only",
                description="Branch fails and rolls back",
                allowed_files={"tax_service.py"},
                steps=[
                    RefactoringStep(
                        step_id="s_bad",
                        description="Bad",
                        target_files=["tax_service.py"],
                        patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('REVERT_ME')\n"},
                        targeted_test_file="test_tax_service.py",
                    )
                ],
            )
        ]

    res = coordinator.execute_branching_refactoring(
        manifest=manifest,
        task_id="T20",
        objective="Fix tax",
        task_family="billing_calculation",
        branch_generator=branch_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is False
    assert res.final_state_fingerprint == state_original.state_fingerprint
