"""
Step 61 Automated Tests: Real-Time Repository Change Awareness & Memory-Guided Multi-Step Refactoring.

Tests verify:
  01. RepositoryState deterministic construction and cryptographic fingerprint repeatability.
  02. Two identical repository states produce bit-exact identical state fingerprints.
  03. RepositoryChangeDetector classifies cosmetic modifications as COSMETIC.
  04. RepositoryChangeDetector classifies local function changes as LOCAL.
  05. RepositoryChangeDetector classifies import additions/removals as DEPENDENCY.
  06. RepositoryChangeDetector classifies signature additions/removals as BEHAVIORAL.
  07. RepositoryChangeDetector classifies file additions/removals as ARCHITECTURAL.
  08. RepositoryChangeDetector classifies test-only changes as TEST_ONLY.
  09. RepositoryImpactAnalyzer traces transitive downstream dependent modules.
  10. RepositoryImpactAnalyzer identifies all affected tests for modified modules.
  11. Memory revalidation returns VALID when repository changes are unrelated.
  12. Memory revalidation returns STALE when target modules undergo structural dependency changes.
  13. Memory revalidation returns INVALID when known boundary conditions are violated.
  14. Memory revalidation returns INVALID when candidate memory has been superseded.
  15. MultiStepRefactoringCoordinator executes ordered multi-step refactoring successfully.
  16. Intermediate regression detection catches unsafe intermediate patches and triggers atomic rollback.
  17. Rolled back intermediate transactions restore exact pre-step workspace content.
  18. Post-refactoring experience consolidation updates semantic repository memory.
  19. Safe abstention on negative-transfer cross-domain repository tasks.
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
from chakrview.cognition.repository.state import RepositoryState, FileState
from chakrview.cognition.repository.change_detector import (
    RepositoryChangeDetector,
    RepositoryDiff,
    ChangeCategory,
)
from chakrview.cognition.repository.impact_analyzer import (
    RepositoryImpactAnalyzer,
    MemoryValidityStatus,
    MemoryRevalidationDecision,
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
# Test 01 & 02: Deterministic State Fingerprinting
# ---------------------------------------------------------------------------
def test_01_state_fingerprint_determinism():
    manifest1 = create_order_billing_manifest(is_defective=True)
    manifest2 = create_order_billing_manifest(is_defective=True)

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_manifest(manifest2)

    assert state1.state_fingerprint != ""
    assert state1.state_fingerprint == state2.state_fingerprint


def test_02_state_fingerprint_changes_on_mutation():
    manifest1 = create_order_billing_manifest(is_defective=True)
    manifest2 = create_order_billing_manifest(is_defective=False)

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_manifest(manifest2)

    assert state1.state_fingerprint != state2.state_fingerprint


# ---------------------------------------------------------------------------
# Test 03-08: Change Detection Categories
# ---------------------------------------------------------------------------
def test_03_change_detection_cosmetic():
    manifest1 = create_order_billing_manifest(is_defective=True)
    # Add trailing comment / whitespace without altering AST
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    files_dict["tax_service.py"] = DEFECTIVE_TAX_CODE + "\n# Just a comment\n"

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)

    diff = RepositoryChangeDetector.detect_changes(state1, state2)
    assert "tax_service.py" in diff.modified_files
    assert diff.file_changes["tax_service.py"].category == ChangeCategory.COSMETIC


def test_04_change_detection_local():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    # Modifying internal calculation without signature changes
    files_dict["tax_service.py"] = CORRECTED_TAX_CODE

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)

    diff = RepositoryChangeDetector.detect_changes(state1, state2)
    assert diff.file_changes["tax_service.py"].category == ChangeCategory.LOCAL


def test_05_change_detection_dependency():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    # Add new import
    files_dict["tax_service.py"] = "import math\n" + DEFECTIVE_TAX_CODE

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)

    diff = RepositoryChangeDetector.detect_changes(state1, state2)
    assert diff.file_changes["tax_service.py"].category == ChangeCategory.DEPENDENCY
    assert diff.has_dependency_changes is True


def test_06_change_detection_behavioral():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    # Add new function signature
    files_dict["tax_service.py"] = DEFECTIVE_TAX_CODE + "\ndef get_tax_jurisdiction(): return 'CA'\n"

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)

    diff = RepositoryChangeDetector.detect_changes(state1, state2)
    assert diff.file_changes["tax_service.py"].category == ChangeCategory.BEHAVIORAL


def test_07_change_detection_architectural():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    # Add a brand new module file
    files_dict["currency_exchange.py"] = "def convert(val): return val * 1.1\n"

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)

    diff = RepositoryChangeDetector.detect_changes(state1, state2)
    assert "currency_exchange.py" in diff.added_files
    assert diff.has_architectural_changes is True
    assert diff.highest_category == ChangeCategory.ARCHITECTURAL


def test_08_change_detection_test_only():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    # Add test in test file
    files_dict["test_tax_service.py"] = TEST_TAX_CODE + "\ndef test_zero_tax(): pass\n"

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)

    diff = RepositoryChangeDetector.detect_changes(state1, state2)
    assert "test_tax_service.py" in diff.modified_files
    assert diff.file_changes["test_tax_service.py"].category == ChangeCategory.TEST_ONLY


# ---------------------------------------------------------------------------
# Test 09 & 10: Impact Analysis
# ---------------------------------------------------------------------------
def test_09_impact_analysis_downstream_propagation():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    files_dict["tax_service.py"] = CORRECTED_TAX_CODE

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)
    diff = RepositoryChangeDetector.detect_changes(state1, state2)

    report = RepositoryImpactAnalyzer.analyze_impact(state1, diff)
    assert "tax_service.py" in report.directly_modified_modules
    # billing_service depends on tax_service
    assert "billing_service.py" in report.transitive_affected_modules


def test_10_impact_analysis_affected_tests():
    manifest1 = create_order_billing_manifest(is_defective=True)
    files_dict = {sf.path: sf.content for sf in manifest1.files}
    files_dict["tax_service.py"] = CORRECTED_TAX_CODE

    state1 = RepositoryState.from_manifest(manifest1)
    state2 = RepositoryState.from_files(project_id="test_repo", files_dict=files_dict)
    diff = RepositoryChangeDetector.detect_changes(state1, state2)

    report = RepositoryImpactAnalyzer.analyze_impact(state1, diff)
    # Both test_tax_service and test_billing_service must be flagged as affected tests
    assert "test_tax_service.py" in report.affected_tests
    assert "test_billing_service.py" in report.affected_tests


# ---------------------------------------------------------------------------
# Test 11-14: Semantic Memory Revalidation Decisions
# ---------------------------------------------------------------------------
def test_11_memory_revalidation_valid_on_unrelated_changes():
    manifest = create_order_billing_manifest()
    state = RepositoryState.from_manifest(manifest)

    # Memory for tax service
    rec = RepositorySemanticRecord(
        memory_id="mem_tax_1",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Rate calculation defect",
        symptom_signature="Tax calculation error",
        root_cause_signature="Flat fee instead of rate",
        dependency_signature="tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="Apply percentage multiplier",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.9,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    idx = RepositoryMemoryIndex()
    idx.insert(rec)

    # Diff modifying unrelated discount_engine.py
    diff = RepositoryDiff(
        from_fingerprint="fp1",
        to_fingerprint="fp2",
        modified_files=["discount_engine.py"],
        affected_modules={"discount_engine.py"},
    )
    report = RepositoryImpactAnalyzer.analyze_impact(state, diff, memory_index=idx)
    assert report.revalidated_memories["mem_tax_1"].status == MemoryValidityStatus.VALID
    assert report.revalidated_memories["mem_tax_1"].is_applicable is True


def test_12_memory_revalidation_stale_on_structural_dependency_change():
    manifest = create_order_billing_manifest()
    state = RepositoryState.from_manifest(manifest)

    rec = RepositorySemanticRecord(
        memory_id="mem_tax_2",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Rate calculation defect",
        symptom_signature="Tax calculation error",
        root_cause_signature="Flat fee instead of rate",
        dependency_signature="tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="Apply percentage multiplier",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.9,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    idx = RepositoryMemoryIndex()
    idx.insert(rec)

    # Diff modifying tax_service with DEPENDENCY change
    diff = RepositoryDiff(
        from_fingerprint="fp1",
        to_fingerprint="fp2",
        modified_files=["tax_service.py"],
        highest_category=ChangeCategory.DEPENDENCY,
        has_dependency_changes=True,
        file_changes={
            "tax_service.py": None,  # Mocked below
        },
        affected_modules={"tax_service.py"},
    )
    from chakrview.cognition.repository.change_detector import FileChange
    diff.file_changes["tax_service.py"] = FileChange(
        rel_path="tax_service.py",
        change_type="MODIFIED",
        category=ChangeCategory.DEPENDENCY,
        is_test=False,
    )

    report = RepositoryImpactAnalyzer.analyze_impact(state, diff, memory_index=idx)
    assert report.revalidated_memories["mem_tax_2"].status == MemoryValidityStatus.STALE
    assert report.revalidated_memories["mem_tax_2"].is_applicable is False


def test_13_memory_revalidation_boundary_violation():
    manifest = create_order_billing_manifest()
    state = RepositoryState.from_manifest(manifest)

    rec = RepositorySemanticRecord(
        memory_id="mem_tax_3",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Rate calculation defect",
        symptom_signature="Tax calculation error",
        root_cause_signature="Flat fee instead of rate",
        dependency_signature="tax_service",
        affected_modules=["fixed_fee_billing.py"],
        solution_pattern="Apply percentage multiplier",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=["fixed_fee_billing"],
        confidence=0.9,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    idx = RepositoryMemoryIndex()
    idx.insert(rec)

    diff = RepositoryDiff(
        from_fingerprint="fp1",
        to_fingerprint="fp2",
        modified_files=["fixed_fee_billing.py"],
        affected_modules={"fixed_fee_billing.py"},
    )
    report = RepositoryImpactAnalyzer.analyze_impact(state, diff, memory_index=idx)
    assert report.revalidated_memories["mem_tax_3"].status == MemoryValidityStatus.INVALID
    assert report.revalidated_memories["mem_tax_3"].is_applicable is False


def test_14_memory_revalidation_superseded_memory_invalid():
    manifest = create_order_billing_manifest()
    state = RepositoryState.from_manifest(manifest)

    rec = RepositorySemanticRecord(
        memory_id="mem_tax_4",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Old pattern",
        symptom_signature="Old symptom",
        root_cause_signature="Old cause",
        dependency_signature="tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="Old solution",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.9,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
        active_version=False,
        superseded_by="mem_tax_4_v2",
    )
    idx = RepositoryMemoryIndex()
    idx.insert(rec)

    diff = RepositoryDiff(from_fingerprint="fp1", to_fingerprint="fp2")
    report = RepositoryImpactAnalyzer.analyze_impact(state, diff, memory_index=idx)
    assert report.revalidated_memories["mem_tax_4"].status == MemoryValidityStatus.INVALID


# ---------------------------------------------------------------------------
# Test 15-18: Multi-Step Refactoring & Intermediate Regression Protection
# ---------------------------------------------------------------------------
def test_15_multi_step_refactoring_execution():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = MultiStepRefactoringCoordinator()

    def plan_gen(mem, state):
        return RefactoringPlan(
            plan_id="plan_test_15",
            task_id="T15",
            objective="Multi-file repair",
            steps=[
                RefactoringStep(
                    step_id="step1",
                    description="Fix tax",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                ),
            ],
        )

    res = coordinator.execute_multi_step_refactoring(
        manifest=manifest,
        task_id="T15",
        objective="Multi-file repair",
        task_family="billing_calculation",
        plan_generator=plan_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is True
    assert res.steps_executed == 1


def test_16_intermediate_regression_rollback():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = MultiStepRefactoringCoordinator()

    def plan_gen(mem, state):
        return RefactoringPlan(
            plan_id="plan_test_16",
            task_id="T16",
            objective="Multi-file repair",
            steps=[
                RefactoringStep(
                    step_id="step1",
                    description="Inject broken tax",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "def compute_tax(order): raise RuntimeError('FAIL')\n"},
                    targeted_test_file="test_tax_service.py",
                ),
            ],
        )

    res = coordinator.execute_multi_step_refactoring(
        manifest=manifest,
        task_id="T16",
        objective="Multi-file repair",
        task_family="billing_calculation",
        plan_generator=plan_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is False
    assert res.step_results[0].rolled_back is True
    assert "Intermediate regression detected" in res.step_results[0].failure_reason


def test_17_rolled_back_workspace_integrity():
    manifest = create_order_billing_manifest(is_defective=True)
    coordinator = MultiStepRefactoringCoordinator()

    def plan_gen(mem, state):
        return RefactoringPlan(
            plan_id="plan_test_17",
            task_id="T17",
            objective="Multi-file repair",
            steps=[
                RefactoringStep(
                    step_id="step1",
                    description="Broken patch",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "# SYNTAX ERROR ))))(((\n"},
                    targeted_test_file="test_tax_service.py",
                ),
            ],
        )

    res = coordinator.execute_multi_step_refactoring(
        manifest=manifest,
        task_id="T17",
        objective="Multi-file repair",
        task_family="billing_calculation",
        plan_generator=plan_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is False
    assert res.step_results[0].state_fingerprint_after == res.step_results[0].state_fingerprint_before


def test_18_experience_consolidation_after_refactoring():
    manifest = create_order_billing_manifest(is_defective=True)
    idx = RepositoryMemoryIndex()
    coordinator = MultiStepRefactoringCoordinator(memory_index=idx)

    def plan_gen(mem, state):
        return RefactoringPlan(
            plan_id="plan_test_18",
            task_id="T18",
            objective="Multi-file repair",
            steps=[
                RefactoringStep(
                    step_id="step1",
                    description="Fix tax",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
                    targeted_test_file="test_tax_service.py",
                ),
            ],
        )

    res = coordinator.execute_multi_step_refactoring(
        manifest=manifest,
        task_id="T18",
        objective="Multi-file repair",
        task_family="billing_calculation",
        plan_generator=plan_gen,
        allowed_modified_files={"tax_service.py"},
        enable_memory=False,
    )
    assert res.overall_success is True
    assert res.consolidated_memory_id is not None
    assert idx.count() == 1


# ---------------------------------------------------------------------------
# Test 19: Safe Abstention on Negative Transfer
# ---------------------------------------------------------------------------
def test_19_negative_transfer_abstention():
    idx = RepositoryMemoryIndex()
    rec = RepositorySemanticRecord(
        memory_id="mem_billing_1",
        task_family="billing_calculation",
        language="python",
        framework="standard_library",
        repository_pattern="Billing invoice repair",
        symptom_signature="Invoice total mismatch",
        root_cause_signature="Tax calculation error",
        dependency_signature="billing_service -> tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="Apply 10% rate",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.95,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    idx.insert(rec)

    # Cross-domain query on security authentication
    query = RepositoryQuery(
        task_family="authentication_security",
        language="python",
        framework="django",
        symptom_signature="CSRF token validation failure in session cookie",
    )
    arb = arbitrate(query, idx)
    assert arb.status in (ArbitrationStatus.NO_MATCH, ArbitrationStatus.REJECTED)
    assert arb.selected is None


# ---------------------------------------------------------------------------
# Test 20: Bit-Exact Baseline Immutability
# ---------------------------------------------------------------------------
def test_20_baseline_immutability():
    current_hash = compute_test_baseline_hash()
    assert current_hash == EXPECTED_WEIGHT_HASH
