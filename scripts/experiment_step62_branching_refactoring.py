"""
ChakrView Step 62 Master Experiment:
Observation-Driven Dynamic Branching Refactoring & Safe Recovery.

Executes all 9 required benchmark conditions (A through I):
Condition A: First strategy succeeds directly without branching recovery.
Condition B: First strategy fails intermediate verification -> rollback -> second strategy succeeds.
Condition C: Multiple strategies exist but unsafe branch (violates allowed files/preconditions) is rejected.
Condition D: All candidate strategies fail -> safe abstention (fails closed).
Condition E: Rollback fingerprint matches original state bit-exact.
Condition F: Memory-assisted strategy selection prioritizing verified patterns.
Condition G: Stale memory cannot drive execution.
Condition H: Deterministic repeated decision (exact same inputs produce identical decisions).
Condition I: Branch limits (max_branches, max_recovery_transitions) prevent uncontrolled execution.

Also verifies:
- Baseline neural core immutability (DeltaW_base == 0)
- Machine-readable evidence written to artifacts/step62/step62_branching_evidence.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.change_detector import RepositoryChangeDetector, ChangeCategory
from chakrview.cognition.repository.impact_analyzer import (
    RepositoryImpactAnalyzer,
    MemoryValidityStatus,
)
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.refactoring import RefactoringStep
from chakrview.cognition.repository.branching import (
    BranchStatus,
    RefactoringBranch,
    BranchObservation,
    BranchSelectionDecision,
    RecoveryDecision,
    BranchingRefactoringResult,
    ObservationDrivenBranchingCoordinator,
)
from scripts.experiment_step61_realtime_change import (
    MODELS_CODE_V1,
    DEFECTIVE_TAX_CODE,
    CORRECTED_TAX_CODE,
    DEFECTIVE_DISCOUNT_CODE,
    CORRECTED_DISCOUNT_CODE,
    BILLING_CODE,
    TEST_TAX_CODE,
    TEST_DISCOUNT_CODE,
    TEST_BILLING_CODE,
    build_repo_manifest,
)

ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step62"


def compute_baseline_hash() -> str:
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_experiment_step62() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 62: OBSERVATION-DRIVEN DYNAMIC BRANCHING & RECOVERY EXPERIMENT")
    print("=" * 80)

    # 1. Baseline Pre-Verification
    hash_pre = compute_baseline_hash()
    print(f"\n[1] Baseline Neural Core Pre-Verification:")
    print(f"    Expected: {EXPECTED_WEIGHT_HASH}")
    print(f"    Measured: {hash_pre}")
    assert hash_pre == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline mismatch before Step 62!"
    print("    STATUS: PASSED (Bit-Exact Immutable)")

    memory_index = RepositoryMemoryIndex()
    coordinator = ObservationDrivenBranchingCoordinator(memory_index=memory_index)

    results: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_weight_hash": hash_pre,
        "conditions": {},
        "metrics": {},
    }

    manifest_base = build_repo_manifest(tax_code=DEFECTIVE_TAX_CODE, discount_code=DEFECTIVE_DISCOUNT_CODE)

    # -----------------------------------------------------------------------
    # Condition A: First Strategy Succeeds Directly
    # -----------------------------------------------------------------------
    print("\n[2] Condition A: First Strategy Direct Success...")

    def branch_gen_a(mem: Any, state: Any) -> List[RefactoringBranch]:
        b1 = RefactoringBranch(
            branch_id="branch_direct_clean",
            description="Fix tax and discount in sequential order",
            allowed_files={"tax_service.py", "discount_engine.py"},
            priority=1,
            steps=[
                RefactoringStep(
                    step_id="step1_tax",
                    description="Fix tax calculation",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                ),
                RefactoringStep(
                    step_id="step2_discount",
                    description="Fix discount calculation",
                    target_files=["discount_engine.py"],
                    patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
                    targeted_test_file="test_discount_engine.py",
                ),
            ],
        )
        return [b1]

    res_a = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_01",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_a,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    print(f"    Condition A Result: Success={res_a.overall_success}, Selected={res_a.selected_branch_id}, Recoveries={res_a.recovery_transitions}")
    assert res_a.overall_success is True
    assert res_a.selected_branch_id == "branch_direct_clean"
    assert res_a.recovery_transitions == 0
    results["conditions"]["condition_a_direct_success"] = res_a.to_dict()

    # -----------------------------------------------------------------------
    # Condition B: First Strategy Fails -> Rollback -> Second Strategy Succeeds
    # -----------------------------------------------------------------------
    print("\n[3] Condition B: First Strategy Fails -> Rollback -> Second Strategy Recovery...")

    def branch_gen_b(mem: Any, state: Any) -> List[RefactoringBranch]:
        # Branch 1 injects broken tax code that raises an error
        b1_broken = RefactoringBranch(
            branch_id="branch_broken_speculative",
            description="Speculative aggressive tax patch that fails verification",
            allowed_files={"tax_service.py"},
            priority=1,
            steps=[
                RefactoringStep(
                    step_id="step1_fail",
                    description="Flawed tax patch",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('BRANCH_1_FAIL')\n"},
                    targeted_test_file="test_tax_service.py",
                )
            ],
        )
        # Branch 2 is the robust correct patch
        b2_robust = RefactoringBranch(
            branch_id="branch_robust_recovery",
            description="Verified rate-based tax and discount patch",
            allowed_files={"tax_service.py", "discount_engine.py"},
            priority=2,
            steps=[
                RefactoringStep(
                    step_id="step1_tax_ok",
                    description="Fix tax calculation",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                ),
                RefactoringStep(
                    step_id="step2_discount_ok",
                    description="Fix discount calculation",
                    target_files=["discount_engine.py"],
                    patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
                    targeted_test_file="test_discount_engine.py",
                ),
            ],
        )
        return [b1_broken, b2_robust]

    res_b = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_02",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_b,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    print(f"    Condition B Result: Success={res_b.overall_success}, Selected={res_b.selected_branch_id}, Recovery Transitions={res_b.recovery_transitions}")
    assert res_b.overall_success is True
    assert res_b.selected_branch_id == "branch_robust_recovery"
    assert res_b.recovery_transitions == 1
    assert res_b.recovery_decisions[0].rollback_successful is True
    assert res_b.recovery_decisions[0].fingerprint_restored is True
    results["conditions"]["condition_b_recovery_reroute"] = res_b.to_dict()

    # -----------------------------------------------------------------------
    # Condition C: Unsafe Branch Rejected by Preconditions
    # -----------------------------------------------------------------------
    print("\n[4] Condition C: Unsafe Branch Precondition Rejection...")

    def branch_gen_c(mem: Any, state: Any) -> List[RefactoringBranch]:
        b_unsafe = RefactoringBranch(
            branch_id="branch_scope_violator",
            description="Attempts modifying unauthorized files",
            allowed_files={"unauthorized_payment_gateway.py"},  # Exceeds allowed_modified_files
            priority=1,
            steps=[
                RefactoringStep(
                    step_id="step_bad",
                    description="Bad patch",
                    target_files=["unauthorized_payment_gateway.py"],
                    patch_dict={"unauthorized_payment_gateway.py": "# evil\n"},
                )
            ],
        )
        b_safe = RefactoringBranch(
            branch_id="branch_safe_fallback",
            description="Safe verified patch",
            allowed_files={"tax_service.py", "discount_engine.py"},
            priority=2,
            steps=[
                RefactoringStep(
                    step_id="step1_tax",
                    description="Fix tax",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                ),
                RefactoringStep(
                    step_id="step2_discount",
                    description="Fix discount",
                    target_files=["discount_engine.py"],
                    patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
                    targeted_test_file="test_discount_engine.py",
                ),
            ],
        )
        return [b_unsafe, b_safe]

    res_c = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_03",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_c,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    print(f"    Condition C Result: Success={res_c.overall_success}, Selected={res_c.selected_branch_id}")
    assert res_c.overall_success is True
    assert res_c.selected_branch_id == "branch_safe_fallback"
    assert res_c.selection_decisions[0].selected is False
    assert "exceeds scope" in (res_c.selection_decisions[0].rejection_reason or "")
    results["conditions"]["condition_c_unsafe_rejection"] = res_c.to_dict()

    # -----------------------------------------------------------------------
    # Condition D: All Strategies Fail -> Safe Abstention (Fails Closed)
    # -----------------------------------------------------------------------
    print("\n[5] Condition D: All Strategies Fail -> Safe Abstention...")

    def branch_gen_d(mem: Any, state: Any) -> List[RefactoringBranch]:
        b1 = RefactoringBranch(
            branch_id="branch_broken_1",
            description="Broken branch 1",
            allowed_files={"tax_service.py"},
            priority=1,
            steps=[
                RefactoringStep(
                    step_id="s1",
                    description="Broken 1",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('FAIL_1')\n"},
                    targeted_test_file="test_tax_service.py",
                )
            ],
        )
        b2 = RefactoringBranch(
            branch_id="branch_broken_2",
            description="Broken branch 2",
            allowed_files={"tax_service.py"},
            priority=2,
            steps=[
                RefactoringStep(
                    step_id="s2",
                    description="Broken 2",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "def compute_tax(order):\n    raise RuntimeError('FAIL_2')\n"},
                    targeted_test_file="test_tax_service.py",
                )
            ],
        )
        return [b1, b2]

    res_d = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_04",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_d,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    print(f"    Condition D Result: Overall Success={res_d.overall_success}, Abstained={res_d.abstained}, Recoveries={res_d.recovery_transitions}")
    assert res_d.overall_success is False
    assert res_d.abstained is True
    assert res_d.selected_branch_id is None
    results["conditions"]["condition_d_all_fail_abstain"] = res_d.to_dict()

    # -----------------------------------------------------------------------
    # Condition E: Rollback Fingerprint Restoration
    # -----------------------------------------------------------------------
    print("\n[6] Condition E: Rollback Fingerprint Exact Bit-Match Verification...")
    state_original = RepositoryState.from_manifest(manifest_base)
    # The final state fingerprint in Condition D after rolling back both broken branches must match state_original
    print(f"    Original Fingerprint: {state_original.state_fingerprint}")
    print(f"    Post-Rollback Final:  {res_d.final_state_fingerprint}")
    assert res_d.final_state_fingerprint == state_original.state_fingerprint
    results["conditions"]["condition_e_rollback_fingerprint_match"] = {
        "original_fingerprint": state_original.state_fingerprint,
        "restored_fingerprint": res_d.final_state_fingerprint,
        "is_exact_match": True,
    }

    # -----------------------------------------------------------------------
    # Condition F: Memory-Assisted Strategy Selection
    # -----------------------------------------------------------------------
    print("\n[7] Condition F: Memory-Assisted Strategy Selection...")
    # Insert high-confidence memory for branch_memory_preferred
    rec_memory = RepositorySemanticRecord(
        memory_id="mem_preferred_strategy",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Rate calculation defect",
        symptom_signature="Multi-file tax and discount refactoring",
        root_cause_signature="Tax calculation error",
        dependency_signature="billing_service -> tax_service",
        affected_modules=["tax_service.py", "discount_engine.py"],
        solution_pattern="Sequential tax then discount patch",
        verification_requirements=["Targeted", "Regression"],
        known_boundaries=[],
        confidence=0.96,
        evidence_count=5,
        successful_episodes=5,
        failed_episodes=0,
    )
    coordinator.memory_index.insert(rec_memory)

    def branch_gen_f(mem: Any, state: Any) -> List[RefactoringBranch]:
        # Branch 1 uses prior memory pattern
        is_mem_available = bool(mem is not None and mem.memory_id == "mem_preferred_strategy")
        b_mem = RefactoringBranch(
            branch_id="branch_memory_guided",
            description="Branch guided by validated prior semantic memory",
            allowed_files={"tax_service.py", "discount_engine.py"},
            priority=1 if is_mem_available else 99,
            memory_guidance_id=mem.memory_id if is_mem_available else None,
            steps=[
                RefactoringStep(
                    step_id="step1_tax",
                    description="Fix tax",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                ),
                RefactoringStep(
                    step_id="step2_discount",
                    description="Fix discount",
                    target_files=["discount_engine.py"],
                    patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
                    targeted_test_file="test_discount_engine.py",
                ),
            ],
        )
        return [b_mem]

    res_f = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_05",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_f,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=True,
    )
    print(f"    Condition F Result: Success={res_f.overall_success}, Selected={res_f.selected_branch_id}")
    assert res_f.overall_success is True
    assert res_f.selected_branch_id == "branch_memory_guided"
    results["conditions"]["condition_f_memory_assisted"] = res_f.to_dict()

    # -----------------------------------------------------------------------
    # Condition G: Stale Memory Cannot Drive Execution
    # -----------------------------------------------------------------------
    print("\n[8] Condition G: Stale Memory Rejection...")
    # Manifest with structural import mutation in tax_service -> makes memory STALE
    stale_tax_code = "import math\n" + DEFECTIVE_TAX_CODE
    manifest_stale = build_repo_manifest(tax_code=stale_tax_code)

    def branch_gen_g(mem: Any, state: Any) -> List[RefactoringBranch]:
        # Branch requiring active valid memory
        b_mem = RefactoringBranch(
            branch_id="branch_requiring_valid_memory",
            description="Requires active valid memory",
            allowed_files={"tax_service.py"},
            priority=1,
            precondition_fn=lambda st, m: bool(m is not None),  # Stale memory will be None
            steps=[
                RefactoringStep(
                    step_id="s_dummy",
                    description="Dummy",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                )
            ],
        )
        return [b_mem]

    # When tax_service was changed with import math, revalidation against original state marks it STALE -> active_memory becomes None
    res_g = coordinator.execute_branching_refactoring(
        manifest=manifest_stale,
        task_id="TASK_BRANCH_06",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_g,
        allowed_modified_files={"tax_service.py"},
        enable_memory=True,
        reference_state=state_original,
    )
    print(f"    Condition G Result: Precondition Rejected={res_g.selection_decisions[0].preconditions_satisfied is False}")
    assert res_g.overall_success is False
    assert res_g.selection_decisions[0].preconditions_satisfied is False
    results["conditions"]["condition_g_stale_memory_rejection"] = res_g.to_dict()

    # -----------------------------------------------------------------------
    # Condition H: Deterministic Repeated Decisions
    # -----------------------------------------------------------------------
    print("\n[9] Condition H: Deterministic Repeated Execution...")
    res_h1 = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_DET",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_b,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    res_h2 = coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_DET",
        objective="Multi-file tax and discount refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_b,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    print(f"    Run 1 Selected: {res_h1.selected_branch_id}, Run 2 Selected: {res_h2.selected_branch_id}")
    assert res_h1.selected_branch_id == res_h2.selected_branch_id
    assert res_h1.recovery_transitions == res_h2.recovery_transitions
    assert res_h1.final_state_fingerprint == res_h2.final_state_fingerprint
    results["conditions"]["condition_h_determinism"] = {
        "run1_branch": res_h1.selected_branch_id,
        "run2_branch": res_h2.selected_branch_id,
        "fingerprint1": res_h1.final_state_fingerprint,
        "fingerprint2": res_h2.final_state_fingerprint,
        "identical": (res_h1.final_state_fingerprint == res_h2.final_state_fingerprint),
    }

    # -----------------------------------------------------------------------
    # Condition I: Operational Bounds (Branch & Recovery Limits)
    # -----------------------------------------------------------------------
    print("\n[10] Condition I: Operational Bounds & Recovery Limits...")
    bounded_coordinator = ObservationDrivenBranchingCoordinator(
        max_branches=2,
        max_recovery_transitions=1,  # Only allows 1 recovery transition
    )

    def branch_gen_i(mem: Any, state: Any) -> List[RefactoringBranch]:
        branches = []
        for i in range(5):  # 5 broken branches
            branches.append(
                RefactoringBranch(
                    branch_id=f"broken_branch_{i}",
                    description=f"Broken branch {i}",
                    allowed_files={"tax_service.py"},
                    priority=i + 1,
                    steps=[
                        RefactoringStep(
                            step_id=f"step_{i}",
                            description=f"Broken step {i}",
                            target_files=["tax_service.py"],
                            patch_dict={"tax_service.py": f"def compute_tax(order):\n    raise RuntimeError('FAIL_{i}')\n"},
                            targeted_test_file="test_tax_service.py",
                        )
                    ],
                )
            )
        return branches

    res_i = bounded_coordinator.execute_branching_refactoring(
        manifest=manifest_base,
        task_id="TASK_BRANCH_LIMITS",
        objective="Multi-file tax refactoring",
        task_family="billing_calculation",
        branch_generator=branch_gen_i,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    print(f"    Condition I Result: Evaluated Branches={res_i.branches_evaluated}, Recovery Transitions={res_i.recovery_transitions}")
    assert res_i.branches_evaluated <= bounded_coordinator.max_branches
    assert res_i.recovery_transitions <= bounded_coordinator.max_recovery_transitions
    results["conditions"]["condition_i_operational_bounds"] = res_i.to_dict()

    # -----------------------------------------------------------------------
    # 11. Baseline Post-Verification
    # -----------------------------------------------------------------------
    hash_post = compute_baseline_hash()
    print(f"\n[11] Baseline Neural Core Post-Verification:")
    print(f"     Expected: {EXPECTED_WEIGHT_HASH}")
    print(f"     Measured: {hash_post}")
    assert hash_post == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline mutated during Step 62!"
    print("     STATUS: PASSED (Bit-Exact Immutable)")

    # -----------------------------------------------------------------------
    # 12. Compile Summary Metrics & Export Artifacts
    # -----------------------------------------------------------------------
    results["post_weight_hash"] = hash_post
    results["metrics"] = {
        "condition_a_direct_success": res_a.overall_success,
        "condition_b_recovery_success": res_b.overall_success,
        "condition_c_unsafe_rejection": (res_c.selection_decisions[0].selected is False),
        "condition_d_safe_abstention": res_d.abstained,
        "condition_e_rollback_exact_fingerprint": True,
        "condition_f_memory_guidance": res_f.overall_success,
        "condition_g_stale_memory_rejected": (res_g.selection_decisions[0].preconditions_satisfied is False),
        "condition_h_deterministic_reproducible": (res_h1.final_state_fingerprint == res_h2.final_state_fingerprint),
        "condition_i_bounds_enforced": (res_i.branches_evaluated <= bounded_coordinator.max_branches),
        "baseline_preserved": (hash_pre == hash_post == EXPECTED_WEIGHT_HASH),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = ARTIFACTS_DIR / "step62_branching_evidence.json"
    evidence_path.write_text(json.dumps(results, indent=2))
    print(f"\n[12] Machine-readable evidence written to {evidence_path}")

    return results


if __name__ == "__main__":
    run_experiment_step62()
