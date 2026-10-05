"""
Demonstration & Experiment Protocol for Steps 78–81: Persistent Project Brain (PPB).

Demonstrates the 11 Required Phases:
- PHASE 1: Scan only part of the project (bounded scan).
- PHASE 2: Persist knowledge.
- PHASE 3: Terminate / restart process (simulate complete process restart).
- PHASE 4: Retrieve knowledge without rescanning previous files.
- PHASE 5: Continue scanning (resumes remaining unscanned files).
- PHASE 6: Perform a targeted task using retrieved knowledge.
- PHASE 7: Apply a safe patch.
- PHASE 8: Update Project Brain (record task learning and patch update).
- PHASE 9: Change one previously understood file.
- PHASE 10: Demonstrate selective invalidation / re-analysis (only affected file modified, rest unchanged).
- PHASE 11: Ask a second task involving an unrelated module, demonstrating that unrelated knowledge
           remains usable without full-project rescanning.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import pytest

from chakrview.cognition.repository.change_detector import (
    ChangeCategory,
    FileChange,
    RepositoryDiff,
)
from chakrview.cognition.repository.impact_analyzer import ImpactReport
from chakrview.cognition.ppb import (
    EpistemicStatus,
    KnowledgeRecordType,
    KnowledgeRecord,
    ProjectIdentity,
    PersistentProjectBrain,
    IncrementalProjectScanner,
    ProjectKnowledgeRetriever,
    ChangeAwareBrainMaintainer,
)


def run_experiment_11_phases():
    temp_dir = tempfile.mkdtemp(prefix="ppb_experiment_")
    db_path = os.path.join(temp_dir, "experiment_ppb.db")

    synthetic_project = {
        "auth/service.py": "def verify_credentials(u, p):\n    return True\n",
        "auth/tokens.py": "from auth.service import verify_credentials\ndef generate_jwt(user_id):\n    return f'token_{user_id}'\n",
        "billing/discount.py": "def calculate_discount(price, rate):\n    return price * rate\n",
        "billing/invoice.py": "from billing.discount import calculate_discount\ndef create_invoice(items, discount_rate):\n    return 100.0\n",
        "shipping/carrier.py": "def quote_shipping(weight):\n    return weight * 2.5\n",
        "inventory/stock.py": "def check_stock(item_id):\n    return 42\n",
        "analytics/tracker.py": "def log_event(event_name):\n    pass\n",
    }

    try:
        # ---------------------------------------------------------------------
        # PHASE 1 & 2: Scan only part of the project & Persist knowledge
        # ---------------------------------------------------------------------
        ident = ProjectIdentity(project_id="synthetic_corp", project_root="/virtual_corp")
        brain_p1 = PersistentProjectBrain(db_path=db_path, project_identity=ident)
        scanner_p1 = IncrementalProjectScanner(brain=brain_p1, chunk_size=2)

        # Scan bounded: exactly 1 chunk (2 files)
        res_p1 = scanner_p1.scan_project(files_dict=synthetic_project, max_chunks=1)
        assert res_p1.files_processed_this_run == 2
        assert not res_p1.is_fully_scanned
        initial_scanned_count = brain_p1.get_state().scanned_files_count
        assert initial_scanned_count == 2

        # ---------------------------------------------------------------------
        # PHASE 3: Terminate / restart process
        # ---------------------------------------------------------------------
        del scanner_p1
        del brain_p1  # Object destroyed, simulating process termination

        # ---------------------------------------------------------------------
        # PHASE 4: Retrieve knowledge without rescanning previous files
        # ---------------------------------------------------------------------
        brain_p2 = PersistentProjectBrain(db_path=db_path)
        assert brain_p2.project_id == "synthetic_corp"
        assert brain_p2.get_state().scanned_files_count == 2

        retriever_p2 = ProjectKnowledgeRetriever(brain_p2)
        # Search for something in the first scanned files ("auth")
        auth_bundle = retriever_p2.retrieve_for_task("Authenticate credentials")
        assert len(auth_bundle.records) > 0
        assert any("auth/" in f for f in auth_bundle.matched_files)

        # ---------------------------------------------------------------------
        # PHASE 5: Continue scanning (resumes remaining unscanned files)
        # ---------------------------------------------------------------------
        scanner_p2 = IncrementalProjectScanner(brain=brain_p2, chunk_size=3)
        res_p2 = scanner_p2.scan_project(files_dict=synthetic_project)
        # 2 files were already scanned, so remaining 5 files are processed
        assert res_p2.files_already_scanned == 2
        assert res_p2.files_skipped_unchanged == 2
        assert res_p2.files_processed_this_run == 5
        assert res_p2.is_fully_scanned

        # ---------------------------------------------------------------------
        # PHASE 6: Perform a targeted task using retrieved knowledge
        # ---------------------------------------------------------------------
        # Task: "Fix discount calculation"
        retriever_p3 = ProjectKnowledgeRetriever(brain_p2)
        discount_bundle = retriever_p3.retrieve_for_task("Fix discount calculation")
        assert any("billing/discount.py" in f for f in discount_bundle.matched_files)
        assert any("calculate_discount" in s for s in discount_bundle.matched_symbols)

        # ---------------------------------------------------------------------
        # PHASE 7 & 8: Apply a safe patch & Update Project Brain
        # ---------------------------------------------------------------------
        maintainer = ChangeAwareBrainMaintainer(brain_p2)
        task_record = maintainer.record_completed_task(
            task_id="TASK-801",
            task_description="Fix discount calculation rate bug",
            affected_files=["billing/discount.py"],
            solution_summary="Clamped rate between 0.0 and 1.0",
            verification_passed=True,
            root_cause="Rate exceeded 100% boundary",
        )
        assert task_record.epistemic_status == EpistemicStatus.FACT

        # ---------------------------------------------------------------------
        # PHASE 9 & 10: Change one file & Demonstrate selective invalidation / re-analysis
        # ---------------------------------------------------------------------
        diff = RepositoryDiff(
            from_fingerprint="fp_old",
            to_fingerprint="fp_new",
            file_changes={
                "billing/discount.py": FileChange(
                    rel_path="billing/discount.py",
                    change_type="MODIFIED",
                    category=ChangeCategory.LOCAL,
                    is_test=False,
                )
            },
        )
        impact = ImpactReport(
            directly_modified_modules=["billing/discount.py"],
            transitive_affected_modules=["billing/invoice.py"],
        )
        updated_content = {
            "billing/discount.py": "def calculate_discount(price, rate):\n    # Clamped\n    return price * min(max(rate, 0.0), 1.0)\n"
        }
        report = maintainer.handle_repository_changes(
            diff=diff,
            impact_report=impact,
            updated_files_content=updated_content,
        )
        # Billing discount is reverified
        assert report.records_reverified > 0
        # Other modules (shipping, inventory, analytics, auth) must remain unaffected!
        assert "shipping/carrier.py" in report.unaffected_files_preserved
        assert "inventory/stock.py" in report.unaffected_files_preserved
        assert "analytics/tracker.py" in report.unaffected_files_preserved

        # ---------------------------------------------------------------------
        # PHASE 11: Ask a second task involving an unrelated module
        # ---------------------------------------------------------------------
        # Task: "Quote shipping weight cost"
        shipping_bundle = retriever_p3.retrieve_for_task("Quote shipping weight cost")
        assert any("shipping/carrier.py" in f for f in shipping_bundle.matched_files)
        assert any("quote_shipping" in s for s in shipping_bundle.matched_symbols)
        # Unrelated records are VALID facts
        for r in shipping_bundle.records:
            if "shipping" in r.file_path:
                assert r.epistemic_status == EpistemicStatus.FACT

        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_11_phase_demonstration_experiment():
    success = run_experiment_11_phases()
    assert success is True
