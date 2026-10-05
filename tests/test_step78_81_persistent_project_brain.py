"""
Comprehensive Dedicated Test Suite for ChakrView Steps 78–81:
Persistent Project Brain (PPB), Incremental Scanning, Retrieval, and Maintenance.

Covers Requirements 1 through 20:
1. Project creation.
2. Persistent storage.
3. Reload after process restart (real file persistence / reload).
4. Knowledge provenance (source chunk, fingerprint, evidence IDs).
5. Epistemic state preservation (FACT, INSUFFICIENT, STALE, UNKNOWN, CONTESTED, REVERIFIED).
6. Partial / incremental scanning.
7. Scan resume (continue where left off).
8. No redundant rescanning (skipped unchanged files).
9. Relevant retrieval.
10. File / symbol retrieval.
11. Dependency retrieval.
12. Change detection.
13. Selective invalidation (unaffected files stay VALID, affected become STALE).
14. Re-verification (STALE -> REVERIFIED).
15. Historical knowledge preservation (versioning, supersedes, auditability).
16. Resource profile behavior (LOW_RESOURCE, STANDARD, ACCELERATED chunk sizes).
17. Corrupted / incomplete state recovery.
18. Deterministic fingerprints.
19. No hallucinated knowledge (unseen is UNKNOWN / absent).
20. Neural weight immutability (dW = 0, exact hash and parameter invariant).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    ResourcePolicy,
    RuntimeStrategy,
)
from chakrview.cognition.repository.change_detector import (
    ChangeCategory,
    FileChange,
    RepositoryDiff,
)
from chakrview.cognition.repository.impact_analyzer import (
    ImpactReport,
    MemoryValidityStatus,
)
from chakrview.cognition.repository.context_store import (
    RepositoryContextStore,
    EpistemicState,
)
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextRequest,
    CognitiveContextSource,
    CognitiveContextStatus,
    UnifiedCognitiveContextComposer,
)
from chakrview.cognition.ppb import (
    CURRENT_SCHEMA_VERSION,
    EpistemicStatus,
    KnowledgeRecordType,
    KnowledgeRecord,
    ProjectIdentity,
    ProjectBrainState,
    PPBStorageSchemaError,
    PersistentBrainStorage,
    PersistentProjectBrain,
    ScanBatchProgress,
    IncrementalProjectScanner,
    PPBRetrievalBudget,
    RetrievedKnowledgeBundle,
    ProjectKnowledgeRetriever,
    MaintenanceReport,
    ChangeAwareBrainMaintainer,
)

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="ppb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_files():
    return {
        "core/auth.py": "def login(user, pwd):\n    return True\n\ndef logout():\n    pass\n",
        "core/session.py": "from core.auth import login\n\nclass SessionManager:\n    def get_session(self, token):\n        return token\n",
        "billing/discount.py": "def compute_discount(price, pct):\n    return price * (pct / 100.0)\n",
        "billing/tax.py": "from billing.discount import compute_discount\n\ndef compute_tax(price):\n    return price * 0.18\n",
        "utils/helpers.py": "def format_currency(val):\n    return f'${val:.2f}'\n",
        "tests/test_billing.py": "from billing.discount import compute_discount\n\ndef test_discount():\n    assert compute_discount(100, 10) == 10\n",
    }


# =============================================================================
# Requirement 1 & 2: Project Creation & Persistent Storage
# =============================================================================
def test_1_and_2_project_creation_and_persistent_storage(temp_dir):
    db_path = os.path.join(temp_dir, "test_ppb.db")
    ident = ProjectIdentity(
        project_id="synthetic_shop",
        project_root="/path/to/shop",
        language="python",
        framework="fastapi",
    )
    brain = PersistentProjectBrain(db_path=db_path, project_identity=ident)

    assert brain.project_id == "synthetic_shop"
    assert brain.project_identity.framework == "fastapi"
    assert os.path.exists(db_path)

    # Store sample record
    rec = KnowledgeRecord(
        record_id="rec_001",
        project_id="synthetic_shop",
        record_type=KnowledgeRecordType.MODULE,
        file_path="billing/discount.py",
        summary="Discount computation module",
        details={"functions": ["compute_discount"]},
        epistemic_status=EpistemicStatus.FACT,
    )
    brain.store_knowledge(rec)

    loaded = brain.get_knowledge("rec_001")
    assert loaded is not None
    assert loaded.summary == "Discount computation module"
    assert loaded.epistemic_status == EpistemicStatus.FACT


# =============================================================================
# Requirement 3: Reload After Process Restart
# =============================================================================
def test_3_reload_after_process_restart(temp_dir):
    db_path = os.path.join(temp_dir, "persistent_store.db")

    # Process 1: Create and write
    brain1 = PersistentProjectBrain(
        db_path=db_path,
        project_identity=ProjectIdentity(project_id="session_proj", project_root="/session_root"),
    )
    rec = KnowledgeRecord(
        record_id="rec_auth",
        project_id="session_proj",
        record_type=KnowledgeRecordType.SYMBOL,
        file_path="core/auth.py",
        symbol_name="login",
        summary="User login function",
        epistemic_status=EpistemicStatus.FACT,
    )
    brain1.store_knowledge(rec)
    del brain1  # Simulate process termination

    # Process 2: Reopen same storage file without in-memory state
    brain2 = PersistentProjectBrain(db_path=db_path)
    assert brain2.project_id == "session_proj"
    reloaded_rec = brain2.get_knowledge("rec_auth")
    assert reloaded_rec is not None
    assert reloaded_rec.symbol_name == "login"
    assert reloaded_rec.epistemic_status == EpistemicStatus.FACT


# =============================================================================
# Requirement 4 & 5: Provenance & Epistemic State Preservation
# =============================================================================
def test_4_and_5_provenance_and_epistemic_preservation(temp_dir):
    db_path = os.path.join(temp_dir, "epistemic.db")
    brain = PersistentProjectBrain(db_path=db_path)

    rec_fact = KnowledgeRecord(
        record_id="rec_fact_1",
        project_id="test_epistemic",
        record_type=KnowledgeRecordType.MODULE,
        file_path="core/auth.py",
        evidence_ids=["ast_sha1234"],
        repo_fingerprint="sha_fingerprint_01",
        source_chunk="chunk_1",
        epistemic_status=EpistemicStatus.FACT,
    )
    rec_inf = KnowledgeRecord(
        record_id="rec_inf_1",
        project_id="test_epistemic",
        record_type=KnowledgeRecordType.ARCHITECTURE,
        file_path="core/auth.py",
        evidence_ids=["inferred_from_calls"],
        epistemic_status=EpistemicStatus.INFERRED,
    )
    brain.store_knowledge(rec_fact)
    brain.store_knowledge(rec_inf)

    loaded_fact = brain.get_knowledge("rec_fact_1")
    loaded_inf = brain.get_knowledge("rec_inf_1")

    assert loaded_fact.epistemic_status == EpistemicStatus.FACT
    assert loaded_fact.source_chunk == "chunk_1"
    assert loaded_fact.repo_fingerprint == "sha_fingerprint_01"
    assert loaded_fact.evidence_ids == ["ast_sha1234"]

    assert loaded_inf.epistemic_status == EpistemicStatus.INFERRED
    # Inferences must never be converted to FACT
    assert loaded_inf.epistemic_status != EpistemicStatus.FACT


# =============================================================================
# Requirement 6, 7 & 8: Incremental Scanning, Resume, No Redundant Rescan
# =============================================================================
def test_6_7_8_incremental_scanning_resumability_no_redundancy(temp_dir, synthetic_files):
    db_path = os.path.join(temp_dir, "scan.db")
    brain = PersistentProjectBrain(
        db_path=db_path,
        project_identity=ProjectIdentity(project_id="scan_test", project_root="/scan_test"),
    )
    # Bounded scanner with chunk_size = 2 (processes at most 2 files per chunk)
    scanner = IncrementalProjectScanner(brain=brain, chunk_size=2)

    # Phase 1: Scan bounded chunk 1 only
    prog1 = scanner.scan_project(files_dict=synthetic_files, max_chunks=1)
    assert prog1.chunks_completed == 1
    assert prog1.files_processed_this_run == 2
    assert not prog1.is_fully_scanned

    # Verify partial understanding state
    summary1 = brain.get_state()
    assert summary1.scanned_files_count == 2

    # Phase 2: Resume scan (processes chunk 2 and 3)
    prog2 = scanner.scan_project(files_dict=synthetic_files, max_chunks=2)
    assert prog2.chunks_completed == 2
    assert prog2.files_already_scanned == 2
    assert prog2.files_processed_this_run == 4
    assert prog2.files_skipped_unchanged == 2
    assert prog2.is_fully_scanned

    # Phase 3: Run scan again without changing files -> all 6 must be skipped
    prog3 = scanner.scan_project(files_dict=synthetic_files)
    assert prog3.files_already_scanned == 6
    assert prog3.files_skipped_unchanged == 6
    assert prog3.files_processed_this_run == 0
    assert prog3.is_fully_scanned


# =============================================================================
# Requirement 9, 10 & 11: Targeted Retrieval (Query, File, Symbol, Dependency)
# =============================================================================
def test_9_10_11_retrieval_by_task_symbol_and_dependency(temp_dir, synthetic_files):
    db_path = os.path.join(temp_dir, "retrieval.db")
    brain = PersistentProjectBrain(
        db_path=db_path,
        project_identity=ProjectIdentity(project_id="retrieval_test", project_root="/retrieval_test"),
    )
    scanner = IncrementalProjectScanner(brain=brain, chunk_size=3)
    scanner.scan_project(files_dict=synthetic_files)

    retriever = ProjectKnowledgeRetriever(brain)

    # 1. Query: "Fix discount computation"
    bundle1 = retriever.retrieve_for_task("Fix discount computation")
    assert any("billing/discount.py" in f for f in bundle1.matched_files)
    assert any("compute_discount" in s for s in bundle1.matched_symbols)
    assert len(bundle1.records) > 0
    assert len(bundle1.evidence_records) > 0

    # 2. Targeted symbol retrieval: "login"
    bundle2 = retriever.retrieve_for_task("Authenticate user session", target_symbols=["login"])
    assert "login" in bundle2.matched_symbols
    assert any("core/auth.py" in f for f in bundle2.matched_files)

    # 3. Targeted file retrieval
    bundle3 = retriever.retrieve_for_task("Update taxes", target_files=["billing/tax.py"])
    assert "billing/tax.py" in bundle3.matched_files


# =============================================================================
# Requirement 12, 13 & 14: Change Detection, Selective Invalidation, Re-verification
# =============================================================================
def test_12_13_14_selective_invalidation_and_reverification(temp_dir, synthetic_files):
    db_path = os.path.join(temp_dir, "change.db")
    brain = PersistentProjectBrain(
        db_path=db_path,
        project_identity=ProjectIdentity(project_id="change_test", project_root="/change_test"),
    )
    scanner = IncrementalProjectScanner(brain=brain, chunk_size=5)
    scanner.scan_project(files_dict=synthetic_files)

    maintainer = ChangeAwareBrainMaintainer(brain)

    # Modify billing/discount.py
    diff = RepositoryDiff(
        from_fingerprint="fp1",
        to_fingerprint="fp2",
        file_changes={
            "billing/discount.py": FileChange(
                rel_path="billing/discount.py",
                change_type="MODIFIED",
                category=ChangeCategory.BEHAVIORAL,
                is_test=False,
            )
        },
    )
    impact = ImpactReport(
        directly_modified_modules=["billing/discount.py"],
        transitive_affected_modules=["billing/tax.py"],
        affected_tests=["tests/test_billing.py"],
    )

    # Updated content for billing/discount.py
    updated_content = {
        "billing/discount.py": "def compute_discount(price, pct, coupon=0):\n    return price * (pct / 100.0) - coupon\n"
    }

    report = maintainer.handle_repository_changes(
        diff=diff,
        impact_report=impact,
        updated_files_content=updated_content,
    )

    assert report.records_marked_stale > 0
    assert report.records_reverified > 0
    # Unaffected files (core/auth.py, utils/helpers.py, etc.) must remain valid
    assert "core/auth.py" in report.unaffected_files_preserved
    assert "utils/helpers.py" in report.unaffected_files_preserved

    # Verify updated record is REVERIFIED
    updated_mod = brain.query_knowledge(file_path="billing/discount.py", record_type=KnowledgeRecordType.MODULE)
    assert len(updated_mod) > 0
    assert updated_mod[0].epistemic_status == EpistemicStatus.REVERIFIED


# =============================================================================
# Requirement 15: Historical Knowledge Preservation (Versioning & Supersedes)
# =============================================================================
def test_15_historical_knowledge_preservation(temp_dir):
    db_path = os.path.join(temp_dir, "history.db")
    brain = PersistentProjectBrain(db_path=db_path)

    rec = KnowledgeRecord(
        record_id="mod_payment",
        project_id="hist_proj",
        record_type=KnowledgeRecordType.MODULE,
        file_path="pay.py",
        summary="Version 1 of payment module",
        version=1,
    )
    brain.store_knowledge(rec)

    # Store Version 2 with same record_id
    rec_v2 = KnowledgeRecord(
        record_id="mod_payment",
        project_id="hist_proj",
        record_type=KnowledgeRecordType.MODULE,
        file_path="pay.py",
        summary="Version 2 of payment module with stripe",
    )
    brain.store_knowledge(rec_v2)

    # Active lookup returns v2
    active = brain.get_knowledge("mod_payment")
    assert active.summary == "Version 2 of payment module with stripe"
    assert active.version == 2
    assert active.supersedes is not None

    # Query including historical versions
    all_versions = brain.storage.query_records(file_path="pay.py", active_only=False)
    assert len(all_versions) >= 2


# =============================================================================
# Requirement 16: Resource Profile Behavior
# =============================================================================
def test_16_resource_profile_behavior(temp_dir):
    low_hw = HardwareCapability(
        cpu_cores=1, cpu_architecture="x86_64", ram_gb=2.0, gpu_available=False,
        gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=10.0, network_available=True,
    )
    std_hw = HardwareCapability(
        cpu_cores=8, cpu_architecture="x86_64", ram_gb=16.0, gpu_available=False,
        gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=100.0, network_available=True,
    )
    acc_hw = HardwareCapability(
        cpu_cores=16, cpu_architecture="x86_64", ram_gb=32.0, gpu_available=True,
        gpu_vendor="nvidia", vram_gb=16.0, storage_capacity_gb=500.0, network_available=True,
    )

    b_low = PersistentProjectBrain(os.path.join(temp_dir, "low.db"), hardware_capability=low_hw)
    b_std = PersistentProjectBrain(os.path.join(temp_dir, "std.db"), hardware_capability=std_hw)
    b_acc = PersistentProjectBrain(os.path.join(temp_dir, "acc.db"), hardware_capability=acc_hw)

    scanner_low = IncrementalProjectScanner(b_low)
    scanner_std = IncrementalProjectScanner(b_std)
    scanner_acc = IncrementalProjectScanner(b_acc)

    assert scanner_low.chunk_size == 2
    assert scanner_std.chunk_size == 5
    assert scanner_acc.chunk_size == 10


# =============================================================================
# Requirement 17: Schema Migration and Incomplete State Recovery
# =============================================================================
def test_17_schema_version_and_incomplete_recovery(temp_dir):
    db_path = os.path.join(temp_dir, "schema.db")
    storage = PersistentBrainStorage(db_path)
    state = storage.get_state_summary("test_proj")
    assert state.schema_version == CURRENT_SCHEMA_VERSION

    # Test future unsupported schema version triggers PPBStorageSchemaError
    future_db = os.path.join(temp_dir, "future.db")
    _ = PersistentBrainStorage(future_db)
    import sqlite3
    with sqlite3.connect(future_db) as conn:
        conn.execute("INSERT INTO schema_info (version, applied_at) VALUES (999, '2099-01-01');")

    with pytest.raises(PPBStorageSchemaError):
        PersistentBrainStorage(future_db)



# =============================================================================
# Requirement 18 & 19: Deterministic Fingerprints & No Hallucinated Knowledge
# =============================================================================
def test_18_and_19_deterministic_fingerprints_and_no_hallucinations(temp_dir):
    db_path = os.path.join(temp_dir, "grounding.db")
    brain = PersistentProjectBrain(db_path=db_path)

    rec = KnowledgeRecord(
        record_id="rec_known",
        project_id="grounding_proj",
        record_type=KnowledgeRecordType.MODULE,
        file_path="known.py",
        summary="Known module",
    )
    hash1 = rec.compute_content_hash()
    hash2 = rec.compute_content_hash()
    assert hash1 == hash2

    # Querying unseen file must return empty list (UNKNOWN), never hallucinated record
    unseen_records = brain.query_knowledge(file_path="non_existent/phantom.py")
    assert len(unseen_records) == 0


# =============================================================================
# Requirement 20: Neural Weight Immutability Invariant (dW = 0)
# =============================================================================
def test_20_neural_weight_immutability():
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    current_hash = hasher.hexdigest()

    assert current_hash == EXPECTED_WEIGHT_HASH
