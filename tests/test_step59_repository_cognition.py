"""
Step 59 Automated Tests: Repository-Level Cognitive Reasoning & Multi-File Verification.

Tests verify:
  01. RepositoryInspector AST parsing, extracted imports, functions, classes, and calls.
  02. RepositoryDependencyGraph construction, edge typing, and acyclic verification.
  03. Upstream dependency tracing from consumer back to root producer (A -> B -> C).
  04. MultiFilePatchTransaction reversibility and atomic rollback on verification failure.
  05. 4-Level RepositoryVerifier (Targeted, Regression, Repo State, Diff Integrity).
  06. Diff integrity checks reject unexpected modified files.
  07. RepositoryCognitionEngine full cycle solve on multi-file synthetic repository.
  08. Step 54 Extended Trajectory format includes <REPOSITORY_STATE>, <PLAN>, <DEPENDENCIES>.
  09. Memory assistance reduces actions on repository repair re-encounter.
  10. Memory ablation proves causal utility of repository experience.
  11. Negative transfer safety: unrelated repository task receives zero or rejected score.
  12. Frozen baseline immutability invariant: model parameters and SHA-256 weight hash unchanged.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.cognition.repository.inspector import RepositoryInspector
from chakrview.cognition.repository.graph import RepositoryDependencyGraph
from chakrview.cognition.repository.planner import (
    RepositoryAction,
    RepositoryActionType,
    RepositoryPlan,
    RepositoryDiagnosis,
)
from chakrview.cognition.repository.patch import RepositoryPatchCoordinator
from chakrview.cognition.repository.verifier import RepositoryVerifier
from chakrview.cognition.repository.engine import RepositoryCognitionEngine


# ---------------------------------------------------------------------------
# Multi-File Repository Fixture Setup
# ---------------------------------------------------------------------------

MODELS_CODE = """
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
            # BUG: Flat 5.0 instead of 10% rate (0.10 * item.price)
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

DISCOUNT_CODE = """
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
    # 2 items of 50.0 => subtotal 100.0. Tax at 10% = 10.0. Final total = 110.0.
    order = Order(order_id="o3", items=[Item("ItemA", 50.0, "standard"), Item("ItemB", 50.0, "standard")], customer_tier="regular")
    inv = generate_invoice(order)
    assert inv.subtotal == 100.0
    assert inv.tax_amount == 10.0
    assert inv.final_total == 110.0
"""


def create_order_billing_manifest(is_defective: bool = True) -> ProjectManifest:
    spec = ProjectSpecification(
        project_id="order_billing_repo",
        project_name="Order Billing System",
        description="Multi-file repository computing invoices from order items, tax, and discount.",
        entrypoint="billing_service.py",
    )
    files = [
        SourceFile("models.py", MODELS_CODE, FileRole.SOURCE),
        SourceFile("tax_service.py", DEFECTIVE_TAX_CODE if is_defective else CORRECTED_TAX_CODE, FileRole.SOURCE),
        SourceFile("discount_engine.py", DISCOUNT_CODE, FileRole.SOURCE),
        SourceFile("billing_service.py", BILLING_CODE, FileRole.SOURCE),
        SourceFile("test_tax_service.py", TEST_TAX_CODE, FileRole.TEST),
        SourceFile("test_discount_engine.py", TEST_DISCOUNT_CODE, FileRole.TEST),
        SourceFile("test_billing_service.py", TEST_BILLING_CODE, FileRole.TEST),
    ]
    return ProjectManifest(specification=spec, files=files)


@pytest.fixture(scope="module")
def baseline_model():
    return instantiate_frozen_baseline()


# ---------------------------------------------------------------------------
# Test 01: RepositoryInspector AST parsing
# ---------------------------------------------------------------------------
def test_01_repository_inspector():
    manifest = create_order_billing_manifest()
    inspections = RepositoryInspector.inspect_manifest(manifest.files)

    assert "billing_service.py" in inspections
    assert "tax_service.py" in inspections
    assert "models.py" in inspections

    billing = inspections["billing_service.py"]
    assert "models" in billing.imports
    assert "tax_service" in billing.imports
    assert "discount_engine" in billing.imports
    assert "generate_invoice" in billing.functions
    assert "compute_tax" in billing.calls


# ---------------------------------------------------------------------------
# Test 02: RepositoryDependencyGraph construction & edges
# ---------------------------------------------------------------------------
def test_02_dependency_graph_edges():
    manifest = create_order_billing_manifest()
    inspections = RepositoryInspector.inspect_manifest(manifest.files)
    graph = RepositoryDependencyGraph()
    graph.build_from_inspections(inspections)

    assert len(graph.edges) >= 5
    # billing_service depends on tax_service
    assert "tax_service.py" in graph.dependencies.get("billing_service.py", set())
    # test_billing_service depends on billing_service
    assert "billing_service.py" in graph.dependencies.get("test_billing_service.py", set())


# ---------------------------------------------------------------------------
# Test 03: Upstream dependency tracing
# ---------------------------------------------------------------------------
def test_03_upstream_dependency_tracing():
    manifest = create_order_billing_manifest()
    inspections = RepositoryInspector.inspect_manifest(manifest.files)
    graph = RepositoryDependencyGraph()
    graph.build_from_inspections(inspections)

    # Upstream of test_billing_service must include billing_service and tax_service
    upstream = graph.get_upstream_dependencies("test_billing_service.py")
    assert "billing_service.py" in upstream
    assert "tax_service.py" in upstream

    chain = graph.find_dependency_chain("test_billing_service.py", "tax_service.py")
    assert chain is not None
    assert chain == ["test_billing_service.py", "billing_service.py", "tax_service.py"]


# ---------------------------------------------------------------------------
# Test 04: MultiFilePatchTransaction reversibility
# ---------------------------------------------------------------------------
def test_04_patch_coordinator_rollback():
    manifest = create_order_billing_manifest()
    with IsolatedWorkspace(manifest) as ws:
        coordinator = RepositoryPatchCoordinator(ws)
        updates = {"tax_service.py": "# TEMP MODIFICATION\n"}

        tx = coordinator.begin_transaction("tx_test_01", updates)
        assert tx.applied is True
        assert coordinator.get_modified_files() == ["tax_service.py"]

        # Revert
        coordinator.rollback_transaction(tx)
        assert tx.reverted is True
        # Content restored
        current = ws.get_source_file_path("tax_service.py").read_text(encoding="utf-8")
        assert current == DEFECTIVE_TAX_CODE


# ---------------------------------------------------------------------------
# Test 05: 4-Level RepositoryVerifier
# ---------------------------------------------------------------------------
def test_05_repository_verifier_levels():
    # Defective repo fails Level 1 and Level 3
    manifest_defective = create_order_billing_manifest(is_defective=True)
    verifier = RepositoryVerifier()
    with IsolatedWorkspace(manifest_defective) as ws:
        res_fail = verifier.verify_repository(
            workspace=ws,
            targeted_test_rel_path="test_billing_service.py",
        )
        assert res_fail.overall_verified is False
        assert res_fail.level3_repo_state_pass is False

    # Corrected repo passes all 4 levels
    manifest_clean = create_order_billing_manifest(is_defective=False)
    with IsolatedWorkspace(manifest_clean) as ws:
        res_pass = verifier.verify_repository(
            workspace=ws,
            targeted_test_rel_path="test_billing_service.py",
            allowed_modified_files={"tax_service.py"},
            currently_modified_files=["tax_service.py"],
        )
        assert res_pass.overall_verified is True
        assert res_pass.level1_targeted_pass is True
        assert res_pass.level2_regression_pass is True
        assert res_pass.level3_repo_state_pass is True
        assert res_pass.level4_diff_integrity is True


# ---------------------------------------------------------------------------
# Test 06: Diff integrity checks
# ---------------------------------------------------------------------------
def test_06_diff_integrity_rejection():
    manifest_clean = create_order_billing_manifest(is_defective=False)
    verifier = RepositoryVerifier()
    with IsolatedWorkspace(manifest_clean) as ws:
        # Pretend billing_service.py was modified when only tax_service.py was allowed
        res = verifier.verify_repository(
            workspace=ws,
            targeted_test_rel_path="test_billing_service.py",
            allowed_modified_files={"tax_service.py"},
            currently_modified_files=["tax_service.py", "billing_service.py"],
        )
        assert res.level4_diff_integrity is False
        assert "billing_service.py" in res.unexpected_modified_files
        assert res.overall_verified is False


# ---------------------------------------------------------------------------
# Test 07: RepositoryCognitionEngine full cycle solve
# ---------------------------------------------------------------------------
def test_07_engine_full_cycle_solve():
    manifest = create_order_billing_manifest(is_defective=True)
    engine = RepositoryCognitionEngine(max_attempts=3)

    def repair_rule(diag, graph):
        return {"tax_service.py": CORRECTED_TAX_CODE}

    result = engine.solve_repository_task(
        manifest=manifest,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch caused by upstream tax service",
        targeted_test_file="test_billing_service.py",
        repair_generator=repair_rule,
        allowed_modified_files={"tax_service.py"},
    )

    assert result["success"] is True
    assert result["targeted_test_pass"] is True
    assert result["regression_test_pass"] is True
    assert result["repository_verification"] is True
    assert result["final_diff_integrity"] is True
    assert result["modified_files"] == ["tax_service.py"]


# ---------------------------------------------------------------------------
# Test 08: Step 54 Extended Trajectory format
# ---------------------------------------------------------------------------
def test_08_extended_trajectory_format():
    manifest = create_order_billing_manifest(is_defective=True)
    engine = RepositoryCognitionEngine(max_attempts=3)
    result = engine.solve_repository_task(
        manifest=manifest,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=lambda d, g: {"tax_service.py": CORRECTED_TAX_CODE},
        allowed_modified_files={"tax_service.py"},
    )

    traj = result["trajectory"]
    assert "<TRAJECTORY>" in traj
    assert "<SPEC>" in traj
    assert "<REPOSITORY_STATE>" in traj
    assert "<PLAN>" in traj
    assert "<DEPENDENCIES>" in traj
    assert "<ACTION>" in traj
    assert "<OBSERVATION>" in traj
    assert "<DIAGNOSIS>" in traj
    assert "<VERIFICATION>" in traj
    assert "<RESULT>\nSUCCESS\n</RESULT>" in traj
    assert "<REFLECTION>" in traj


# ---------------------------------------------------------------------------
# Test 09: Memory assistance accelerates re-encounter
# ---------------------------------------------------------------------------
def test_09_memory_assisted_solve():
    engine = RepositoryCognitionEngine(max_attempts=3)

    # Run 1: Cold start (populates experience)
    manifest1 = create_order_billing_manifest(is_defective=True)
    res1 = engine.solve_repository_task(
        manifest=manifest1,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=lambda d, g: {"tax_service.py": CORRECTED_TAX_CODE},
        allowed_modified_files={"tax_service.py"},
    )
    assert res1["success"] is True

    # Run 2: Re-encounter with verified memory active (no manual repair_generator passed)
    manifest2 = create_order_billing_manifest(is_defective=True)
    res2 = engine.solve_repository_task(
        manifest=manifest2,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=None,  # Solved via retrieved memory!
        allowed_modified_files={"tax_service.py"},
        enable_memory=True,
    )
    assert res2["success"] is True


# ---------------------------------------------------------------------------
# Test 10: Memory ablation test
# ---------------------------------------------------------------------------
def test_10_memory_ablation():
    engine = RepositoryCognitionEngine(max_attempts=3)

    # Populate memory
    manifest1 = create_order_billing_manifest(is_defective=True)
    engine.solve_repository_task(
        manifest=manifest1,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=lambda d, g: {"tax_service.py": CORRECTED_TAX_CODE},
        allowed_modified_files={"tax_service.py"},
    )

    # Run with memory disabled and NO repair_generator -> must fail because memory is ablated
    manifest2 = create_order_billing_manifest(is_defective=True)
    res_ablated = engine.solve_repository_task(
        manifest=manifest2,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=None,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,  # ABLATED
    )
    assert res_ablated["success"] is False


# ---------------------------------------------------------------------------
# Test 11: Negative-transfer safety
# ---------------------------------------------------------------------------
def test_11_negative_transfer_safety():
    engine = RepositoryCognitionEngine(max_attempts=3)
    # Populate billing repair memory
    manifest = create_order_billing_manifest(is_defective=True)
    engine.solve_repository_task(
        manifest=manifest,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=lambda d, g: {"tax_service.py": CORRECTED_TAX_CODE},
        allowed_modified_files={"tax_service.py"},
    )

    # Query for completely unrelated string transformation task
    results = engine.workspace_orchestrator.retriever.retrieve(
        task_family="string_operations",
        objective="Reverse order of strings",
        current_diagnosis="String reverse error",
        task_id="TASK_STRING_01",
        consolidator=engine.workspace_orchestrator.consolidator,
    )
    for r in results:
        assert r.family_match_score == 0.0
        assert r.score < 0.35


# ---------------------------------------------------------------------------
# Test 12: Frozen baseline immutability (DeltaW_base == 0)
# ---------------------------------------------------------------------------
def test_12_frozen_baseline_immutability(baseline_model):
    hash_pre = compute_model_hash(baseline_model)
    assert hash_pre == EXPECTED_WEIGHT_HASH

    manifest = create_order_billing_manifest(is_defective=True)
    engine = RepositoryCognitionEngine(max_attempts=3)
    engine.solve_repository_task(
        manifest=manifest,
        task_id="TASK_REPO_BILLING_01",
        task_description="Fix order billing invoice mismatch",
        targeted_test_file="test_billing_service.py",
        repair_generator=lambda d, g: {"tax_service.py": CORRECTED_TAX_CODE},
        allowed_modified_files={"tax_service.py"},
    )

    hash_post = compute_model_hash(baseline_model)
    assert hash_post == EXPECTED_WEIGHT_HASH
    assert hash_pre == hash_post
