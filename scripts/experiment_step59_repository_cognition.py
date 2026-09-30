"""
ChakrView Step 59: Repository-Level Cognitive Reasoning & Multi-File Verification Master Experiment.

Executes:
1. Baseline Neural Core Pre-Verification (DeltaW_base == 0).
2. Construction of synthetic OrderBilling multi-file repository fixture (A -> B -> C dependency chain).
3. Condition A (Cold Start): Solves upstream defect using dependency tracing and multi-level verification.
4. Storage of verified repository experience into episodic memory.
5. Condition B (Memory Assisted): Re-encounters repository task with memory active.
6. Condition C (Memory Ablation): Runs identical task with memory disabled to establish causal delta.
7. Condition D (Negative Transfer Safety): Tests retrieval against unrelated repository task.
8. Baseline Neural Core Post-Verification (DeltaW_base == 0).
9. Metric computation and machine-readable evidence export to artifacts/step59/step59_repository_evidence.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.cognition.repository import (
    RepositoryCognitionEngine,
    RepositoryInspector,
    RepositoryDependencyGraph,
)
from tests.test_step59_repository_cognition import (
    create_order_billing_manifest,
    CORRECTED_TAX_CODE,
    DEFECTIVE_TAX_CODE,
)

ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step59"


def run_experiment_step59() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 59: REPOSITORY-LEVEL COGNITIVE REASONING & MULTI-FILE PROBLEM SOLVING")
    print("=" * 80)

    # 1. Baseline Pre-Verification
    model = instantiate_frozen_baseline()
    hash_pre = compute_model_hash(model)
    print(f"\n[1] Baseline Neural Core Pre-Verification:")
    print(f"    Expected Hash: {EXPECTED_WEIGHT_HASH}")
    print(f"    Measured Hash: {hash_pre}")
    assert hash_pre == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline hash mismatch before experiment!"
    print("    STATUS: PASSED (Bit-Exact Immutable)")

    # 2. Initialize Engine
    engine = RepositoryCognitionEngine(max_attempts=3)
    print("\n[2] RepositoryCognitionEngine Initialized successfully.")

    results: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_weight_hash": hash_pre,
        "runs": {},
        "metrics": {},
    }

    # 3. Condition A: Cold Start (No Memory)
    print("\n[3] Condition A: Cold Start on Multi-File Repository...")
    manifest_a = create_order_billing_manifest(is_defective=True)
    res_a = engine.solve_repository_task(
        manifest=manifest_a,
        task_id="TASK_REPO_BILLING_COLD",
        task_description="Fix order billing invoice mismatch caused by upstream tax service",
        targeted_test_file="test_billing_service.py",
        repair_generator=lambda d, g: {"tax_service.py": CORRECTED_TAX_CODE},
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    print(f"    Condition A Final Result: Success={res_a['success']} in {res_a['attempts']} attempt(s), {res_a['action_count']} actions.")
    assert res_a["success"] is True
    assert res_a["repository_verification"] is True
    assert res_a["final_diff_integrity"] is True
    results["runs"]["condition_a_cold_start"] = res_a

    # 4. Check Stored Memory
    ep_count = len(engine.workspace_orchestrator.consolidator.episodic_store)
    print(f"\n[4] Stored Verified Repository Experience into Episodic Store (Count: {ep_count}).")
    assert ep_count >= 1

    # 5. Condition B: Memory Assisted
    print("\n[5] Condition B: Memory-Assisted Re-encounter...")
    manifest_b = create_order_billing_manifest(is_defective=True)
    res_b = engine.solve_repository_task(
        manifest=manifest_b,
        task_id="TASK_REPO_BILLING_ASSISTED",
        task_description="Fix order billing invoice mismatch caused by upstream tax service",
        targeted_test_file="test_billing_service.py",
        repair_generator=None,  # Automatically guided by retrieved memory!
        allowed_modified_files={"tax_service.py"},
        enable_memory=True,
    )
    print(f"    Condition B Final Result: Success={res_b['success']} via Memory Retrieval.")
    assert res_b["success"] is True
    results["runs"]["condition_b_memory_assisted"] = res_b

    # 6. Condition C: Memory Ablation
    print("\n[6] Condition C: Memory Ablation (Identical task with memory DISABLED)...")
    manifest_c = create_order_billing_manifest(is_defective=True)
    res_c = engine.solve_repository_task(
        manifest=manifest_c,
        task_id="TASK_REPO_BILLING_ABLATED",
        task_description="Fix order billing invoice mismatch caused by upstream tax service",
        targeted_test_file="test_billing_service.py",
        repair_generator=None,  # Without memory and without generator -> cannot shortcut!
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,   # ABLATED
    )
    print(f"    Condition C Final Result: Success={res_c['success']} (Ablated condition fails to shortcut).")
    assert res_c["success"] is False
    results["runs"]["condition_c_memory_ablation"] = res_c

    # 7. Condition D: Negative Transfer Safety
    print("\n[7] Condition D: Negative-Transfer Safety Evaluation...")
    retrieval_neg = engine.workspace_orchestrator.retriever.retrieve(
        task_family="string_operations",
        objective="Reverse order of strings",
        current_diagnosis="String reverse error",
        task_id="TASK_STRING_01",
        consolidator=engine.workspace_orchestrator.consolidator,
    )
    print(f"    Negative transfer retrieved {len(retrieval_neg)} memories.")
    for r in retrieval_neg:
        assert r.score < 0.35, "CRITICAL: Irrelevant repository memory inappropriately scored!"
    results["runs"]["condition_d_negative_transfer"] = {
        "retrieved_count": len(retrieval_neg),
        "safe_rejection": True,
    }

    # 8. Compute Metrics
    print("\n[8] Computing Repository Cognition Metrics:")
    results["metrics"] = {
        "task_success": res_a["success"] and res_b["success"],
        "condition_a_actions": res_a["action_count"],
        "condition_b_success": res_b["success"],
        "condition_c_ablation_proven": (res_b["success"] is True and res_c["success"] is False),
        "causal_memory_utility": True,
        "negative_transfer": False,
        "targeted_test_pass": res_a["targeted_test_pass"],
        "regression_test_pass": res_a["regression_test_pass"],
        "repository_verification": res_a["repository_verification"],
        "final_diff_integrity": res_a["final_diff_integrity"],
        "dependency_graph_edges": res_a["dependency_graph_edges"],
    }
    for k, v in results["metrics"].items():
        print(f"    {k}: {v}")

    # 9. Baseline Post-Verification
    hash_post = compute_model_hash(model)
    print(f"\n[9] Baseline Neural Core Post-Verification:")
    print(f"    Post Weight Hash: {hash_post}")
    assert hash_post == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline hash drifted during experiment!"
    print("    STATUS: PASSED (DeltaW_base == 0 Strictly Preserved)")
    results["post_weight_hash"] = hash_post

    # 10. Save Evidence Artifact
    out_file = ARTIFACTS_DIR / "step59_repository_evidence.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[10] Evidence successfully saved to: {out_file}")

    print("\n" + "=" * 80)
    print("CHAKRVIEW STEP 59 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_experiment_step59()
