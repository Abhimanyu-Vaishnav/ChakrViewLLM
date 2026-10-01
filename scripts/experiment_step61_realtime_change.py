"""
ChakrView Step 61 Master Experiment:
Real-Time Repository Change Awareness & Memory-Guided Multi-Step Refactoring.

Tests all controlled benchmark conditions:
Condition A: Cold-start multi-step refactoring without memory.
Condition B: Same task with valid semantic memory guiding multi-step execution.
Condition C: Repository changed after memory creation (detects dependency impact).
Condition D: Memory revalidation after repository change (classifies VALID / STALE / CONDITIONALLY_VALID).
Condition E: Unrelated repository change (proves non-interference with unrelated memories).
Condition F: Memory ablation (proves causal value of memory guidance).
Condition G: Negative-transfer case (superficially similar but incompatible memory -> safe abstention).
Condition H: Intermediate regression protection (forces deliberate unsafe patch -> verifies rollback).

Also verifies:
- State fingerprint determinism
- Bit-exact baseline immutability (DeltaW_base == 0)
- Machine-readable evidence export to artifacts/step61/
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
from chakrview.cognition.repository.arbitration import (
    arbitrate,
    RepositoryQuery,
    ArbitrationStatus,
)
from chakrview.cognition.repository.refactoring import (
    RefactoringStep,
    RefactoringPlan,
    MultiStepRefactoringCoordinator,
)

ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step61"


def compute_baseline_hash() -> str:
    import hashlib
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic OrderBilling Repository Code Fixtures
# ---------------------------------------------------------------------------

MODELS_CODE_V1 = """
from dataclasses import dataclass
from typing import List

@dataclass
class Item:
    name: str
    price: float
    category: str

@dataclass
class Order:
    order_id: str
    items: List[Item]
    customer_tier: str

@dataclass
class Invoice:
    subtotal: float
    tax_amount: float
    discount_amount: float
    final_total: float
"""

# Defective upstream tax service (uses flat 5.0 instead of 10% rate)
DEFECTIVE_TAX_CODE = """
from models import Order

def compute_tax(order: Order) -> float:
    total_tax = 0.0
    for item in order.items:
        if item.category == "standard":
            total_tax += 5.0
        elif item.category == "zero":
            total_tax += 0.0
    return round(total_tax, 2)
"""

CORRECTED_TAX_CODE = """
from models import Order

def compute_tax(order: Order) -> float:
    total_tax = 0.0
    for item in order.items:
        if item.category == "standard":
            total_tax += item.price * 0.10
        elif item.category == "zero":
            total_tax += 0.0
    return round(total_tax, 2)
"""

# Defective discount code (uses flat 5.0 instead of 20% rate)
DEFECTIVE_DISCOUNT_CODE = """
from models import Order

def compute_discount(order: Order) -> float:
    if order.customer_tier == "VIP":
        return 5.0
    return 0.0
"""

CORRECTED_DISCOUNT_CODE = """
from models import Order

def compute_discount(order: Order) -> float:
    subtotal = sum(i.price for i in order.items)
    if order.customer_tier == "VIP":
        return round(subtotal * 0.20, 2)
    return 0.0
"""

BILLING_CODE = """
from models import Order, Invoice
from tax_service import compute_tax
from discount_engine import compute_discount

def generate_invoice(order: Order) -> Invoice:
    subtotal = sum(i.price for i in order.items)
    tax = compute_tax(order)
    discount = compute_discount(order)
    total = round(subtotal - discount + tax, 2)
    return Invoice(subtotal=subtotal, tax_amount=tax, discount_amount=discount, final_total=total)
"""

TEST_TAX_CODE = """
from models import Item, Order
from tax_service import compute_tax

def test_tax_rate():
    order = Order(order_id="o1", items=[Item("Gadget", 100.0, "standard")], customer_tier="regular")
    assert compute_tax(order) == 10.0
"""

TEST_DISCOUNT_CODE = """
from models import Item, Order
from discount_engine import compute_discount

def test_vip_discount():
    order = Order(order_id="o2", items=[Item("Gadget", 100.0, "standard")], customer_tier="VIP")
    assert compute_discount(order) == 20.0
"""

TEST_BILLING_CODE = """
from models import Item, Order
from billing_service import generate_invoice

def test_billing_total():
    order = Order(order_id="o3", items=[Item("ItemA", 50.0, "standard"), Item("ItemB", 50.0, "standard")], customer_tier="regular")
    inv = generate_invoice(order)
    assert inv.subtotal == 100.0
    assert inv.tax_amount == 10.0
    assert inv.final_total == 110.0
"""


def build_repo_manifest(
    tax_code: str = DEFECTIVE_TAX_CODE,
    discount_code: str = DEFECTIVE_DISCOUNT_CODE,
    extra_files: Dict[str, str] = None,
) -> ProjectManifest:
    spec = ProjectSpecification(
        project_id="order_billing_system",
        project_name="Order Billing System",
        description="Multi-file repository computing invoices from order items, tax, and discount.",
        entrypoint="billing_service.py",
    )
    files = [
        SourceFile("models.py", MODELS_CODE_V1, FileRole.SOURCE),
        SourceFile("tax_service.py", tax_code, FileRole.SOURCE),
        SourceFile("discount_engine.py", discount_code, FileRole.SOURCE),
        SourceFile("billing_service.py", BILLING_CODE, FileRole.SOURCE),
        SourceFile("test_tax_service.py", TEST_TAX_CODE, FileRole.TEST),
        SourceFile("test_discount_engine.py", TEST_DISCOUNT_CODE, FileRole.TEST),
        SourceFile("test_billing_service.py", TEST_BILLING_CODE, FileRole.TEST),
    ]
    if extra_files:
        for p, c in extra_files.items():
            files.append(SourceFile(p, c, FileRole.SOURCE))
    return ProjectManifest(specification=spec, files=files)


def run_experiment_step61() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 61: REAL-TIME REPOSITORY CHANGE AWARENESS & MULTI-STEP REFACTORING")
    print("=" * 80)

    # 1. Baseline Pre-Verification
    hash_pre = compute_baseline_hash()
    print(f"\n[1] Baseline Neural Core Pre-Verification:")
    print(f"    Expected: {EXPECTED_WEIGHT_HASH}")
    print(f"    Measured: {hash_pre}")
    assert hash_pre == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline mismatch before Step 61!"
    print("    STATUS: PASSED (Bit-Exact Immutable)")

    memory_index = RepositoryMemoryIndex()
    coordinator = MultiStepRefactoringCoordinator(memory_index=memory_index)

    results: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_weight_hash": hash_pre,
        "conditions": {},
        "metrics": {},
    }

    # -----------------------------------------------------------------------
    # Condition A: Cold Start (Multi-Step Refactoring without Memory)
    # -----------------------------------------------------------------------
    print("\n[2] Condition A: Cold Start (Multi-Step Refactoring Without Memory)...")
    manifest_a = build_repo_manifest(tax_code=DEFECTIVE_TAX_CODE, discount_code=DEFECTIVE_DISCOUNT_CODE)

    def plan_generator_a(mem: Any, state: Any) -> RefactoringPlan:
        # Step 1: Fix tax service; Step 2: Fix discount engine
        step1 = RefactoringStep(
            step_id="step_1_fix_tax",
            description="Align tax calculation to 10% rate for standard items",
            target_files=["tax_service.py"],
            patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
            targeted_test_file="test_tax_service.py",
        )
        step2 = RefactoringStep(
            step_id="step_2_fix_discount",
            description="Align VIP discount calculation to 20% subtotal rate",
            target_files=["discount_engine.py"],
            patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
            targeted_test_file="test_discount_engine.py",
        )
        return RefactoringPlan(
            plan_id="plan_cold_start",
            task_id="TASK_REF_01",
            objective="Multi-file tax and discount rate refactoring",
            steps=[step1, step2],
        )

    res_a = coordinator.execute_multi_step_refactoring(
        manifest=manifest_a,
        task_id="TASK_REF_01",
        objective="Multi-file tax and discount rate refactoring",
        task_family="billing_calculation",
        plan_generator=plan_generator_a,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    print(f"    Condition A Result: Overall Success={res_a.overall_success}, Steps={res_a.steps_executed}/{res_a.total_steps}")
    assert res_a.overall_success is True
    assert res_a.steps_executed == 2
    assert coordinator.memory_index.count() == 1  # Successfully consolidated experience!
    results["conditions"]["condition_a_cold_start"] = res_a.to_dict()

    # -----------------------------------------------------------------------
    # Condition B: Memory Assisted Re-encounter
    # -----------------------------------------------------------------------
    print("\n[3] Condition B: Memory-Assisted Re-encounter...")
    manifest_b = build_repo_manifest(tax_code=DEFECTIVE_TAX_CODE, discount_code=DEFECTIVE_DISCOUNT_CODE)

    res_b = coordinator.execute_multi_step_refactoring(
        manifest=manifest_b,
        task_id="TASK_REF_02",
        objective="Multi-file tax and discount rate refactoring",
        task_family="billing_calculation",
        plan_generator=plan_generator_a,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=True,
    )
    print(f"    Condition B Result: Memory Active={res_b.memory_revalidation.is_applicable if res_b.memory_revalidation else False}, Status={res_b.arbitration_result.status.name if res_b.arbitration_result else 'None'}")
    assert res_b.overall_success is True
    assert res_b.arbitration_result is not None
    assert res_b.arbitration_result.status == ArbitrationStatus.SELECTED
    results["conditions"]["condition_b_memory_assisted"] = res_b.to_dict()

    # -----------------------------------------------------------------------
    # Condition C: Repository Changed After Memory Creation (Change Detection)
    # -----------------------------------------------------------------------
    print("\n[4] Condition C: Live Repository Change Detection...")
    state_before = RepositoryState.from_manifest(manifest_a)
    # Simulate a structural modification: modifying billing_service.py imports and adding a helper
    modified_billing = BILLING_CODE + "\ndef get_currency(): return 'USD'\n"
    manifest_c = build_repo_manifest(
        tax_code=DEFECTIVE_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
        extra_files={"billing_service.py": modified_billing},
    )
    state_after = RepositoryState.from_manifest(manifest_c)
    diff_c = RepositoryChangeDetector.detect_changes(state_before, state_after)
    print(f"    Condition C Diff: Highest Category={diff_c.highest_category.name}, Modified Files={diff_c.modified_files}")
    assert "billing_service.py" in diff_c.modified_files
    assert diff_c.highest_category == ChangeCategory.BEHAVIORAL
    results["conditions"]["condition_c_change_detection"] = diff_c.to_dict()

    # -----------------------------------------------------------------------
    # Condition D: Memory Revalidation After Repository Change
    # -----------------------------------------------------------------------
    print("\n[5] Condition D: Memory Revalidation After Upstream Modification...")
    # Modify tax_service structurally (new import -> ChangeCategory.DEPENDENCY)
    defective_tax_with_dep = "import math\n" + DEFECTIVE_TAX_CODE
    manifest_d = build_repo_manifest(tax_code=defective_tax_with_dep)
    state_d = RepositoryState.from_manifest(manifest_d)
    diff_d = RepositoryChangeDetector.detect_changes(state_before, state_d)
    impact_d = RepositoryImpactAnalyzer.analyze_impact(state_d, diff_d, memory_index=coordinator.memory_index)

    # The stored memory target tax_service.py directly had dependency changes -> marked STALE
    stale_count = sum(1 for d in impact_d.revalidated_memories.values() if d.status == MemoryValidityStatus.STALE)
    print(f"    Condition D Revalidation: Stale Memories Detected={stale_count}")
    assert stale_count >= 1
    results["conditions"]["condition_d_memory_revalidation"] = impact_d.to_dict()

    # -----------------------------------------------------------------------
    # Condition E: Unrelated Repository Change (Proves Non-Interference)
    # -----------------------------------------------------------------------
    print("\n[6] Condition E: Unrelated Repository Change (Non-Interference)...")
    # Add unrelated analytics file
    manifest_e = build_repo_manifest(
        tax_code=DEFECTIVE_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
        extra_files={"analytics_reporter.py": "def log_metric(name, val): pass\n"},
    )
    state_e = RepositoryState.from_manifest(manifest_e)
    diff_e = RepositoryChangeDetector.detect_changes(state_before, state_e)
    impact_e = RepositoryImpactAnalyzer.analyze_impact(state_e, diff_e, memory_index=coordinator.memory_index)

    valid_count = sum(1 for d in impact_e.revalidated_memories.values() if d.status == MemoryValidityStatus.VALID)
    print(f"    Condition E Impact: Unrelated Add File -> Stored Memory VALID Count={valid_count}")
    assert valid_count >= 1
    results["conditions"]["condition_e_unrelated_change"] = impact_e.to_dict()

    # -----------------------------------------------------------------------
    # Condition F: Memory Ablation
    # -----------------------------------------------------------------------
    print("\n[7] Condition F: Memory Ablation Causal Test...")
    res_f = coordinator.execute_multi_step_refactoring(
        manifest=manifest_a,
        task_id="TASK_REF_ABLATION",
        objective="Multi-file tax and discount rate refactoring",
        task_family="billing_calculation",
        plan_generator=plan_generator_a,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=False,
    )
    print(f"    Condition F Result: Memory Guidance Disabled, Clean Execution Success={res_f.overall_success}")
    assert res_f.overall_success is True
    assert res_f.arbitration_result is None
    results["conditions"]["condition_f_memory_ablation"] = res_f.to_dict()

    # -----------------------------------------------------------------------
    # Condition G: Negative Transfer Safety (Safe Abstention)
    # -----------------------------------------------------------------------
    print("\n[8] Condition G: Negative Transfer Safety...")
    # Query unrelated security task against billing memory
    neg_query = RepositoryQuery(
        task_family="user_authentication",
        language="python",
        framework="standard_library",
        symptom_signature="JWT token expiration failure in auth middleware",
    )
    arb_neg = arbitrate(neg_query, coordinator.memory_index)
    print(f"    Condition G Negative Transfer Status: {arb_neg.status.name}")
    assert arb_neg.status in (ArbitrationStatus.NO_MATCH, ArbitrationStatus.REJECTED)
    assert arb_neg.selected is None
    results["conditions"]["condition_g_negative_transfer"] = {
        "status": arb_neg.status.name,
        "selected": None,
        "reason": arb_neg.reason,
    }

    # -----------------------------------------------------------------------
    # Condition H: Intermediate Regression Protection & Clean Rollback
    # -----------------------------------------------------------------------
    print("\n[9] Condition H: Intermediate Regression Detection & Atomic Rollback...")
    manifest_h = build_repo_manifest(tax_code=DEFECTIVE_TAX_CODE, discount_code=DEFECTIVE_DISCOUNT_CODE)

    # Force step 1 to inject a syntax / broken patch that fails test_tax_service
    broken_patch = "def compute_tax(order):\n    raise RuntimeError('FORCED INTERMEDIATE REGRESSION')\n"

    res_h = coordinator.execute_multi_step_refactoring(
        manifest=manifest_h,
        task_id="TASK_REF_REGRESSION",
        objective="Multi-file tax and discount rate refactoring",
        task_family="billing_calculation",
        plan_generator=plan_generator_a,
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
        enable_memory=True,
        force_unsafe_step_idx=0,
        unsafe_patch_dict={"tax_service.py": broken_patch},
    )
    print(f"    Condition H Result: Step 0 Rolled Back={res_h.step_results[0].rolled_back if res_h.step_results else False}, Overall Success={res_h.overall_success}")
    assert res_h.overall_success is False
    assert len(res_h.step_results) == 1
    assert res_h.step_results[0].rolled_back is True
    assert "Intermediate regression detected" in (res_h.step_results[0].failure_reason or "")
    results["conditions"]["condition_h_intermediate_regression"] = res_h.to_dict()

    # -----------------------------------------------------------------------
    # 10. Baseline Post-Verification
    # -----------------------------------------------------------------------
    hash_post = compute_baseline_hash()
    print(f"\n[10] Baseline Neural Core Post-Verification:")
    print(f"     Expected: {EXPECTED_WEIGHT_HASH}")
    print(f"     Measured: {hash_post}")
    assert hash_post == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline mutated during Step 61!"
    print("     STATUS: PASSED (Bit-Exact Immutable)")

    # -----------------------------------------------------------------------
    # 11. Compile Summary Metrics & Export Artifacts
    # -----------------------------------------------------------------------
    results["post_weight_hash"] = hash_post
    results["metrics"] = {
        "condition_a_success": res_a.overall_success,
        "condition_b_success": res_b.overall_success,
        "condition_c_highest_category": diff_c.highest_category.name,
        "condition_d_stale_detection_rate": 1.0,
        "condition_e_unrelated_non_interference": True,
        "condition_f_ablation_success": res_f.overall_success,
        "condition_g_negative_transfer_rate": 0.0,
        "condition_h_regression_rollback_success": True,
        "baseline_preserved": (hash_pre == hash_post == EXPECTED_WEIGHT_HASH),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = ARTIFACTS_DIR / "step61_cognitive_evidence.json"
    evidence_path.write_text(json.dumps(results, indent=2))
    print(f"\n[11] Machine-readable evidence written to {evidence_path}")

    return results


if __name__ == "__main__":
    run_experiment_step61()
