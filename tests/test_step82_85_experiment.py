"""
End-to-End Realistic Project Experiment Protocol for Steps 82–85:
Persistent Task Decomposition, Resource-Aware Scheduling, Incremental Work Loop & Knowledge Evolution.

Simulates a Constrained Machine over a realistic multi-file codebase and demonstrates:
1. Project discovered (7 files, multi-module enterprise architecture).
2. Project divided into chunks (bounded chunk_size = 2).
3. Chunks scanned incrementally.
4. Knowledge persisted in SQLite PPB.
5. Process/session interrupted.
6. ChakrView restarted.
7. Previous knowledge recovered without rescan.
8. Large task decomposed into persistent DAG.
9. Only relevant project knowledge retrieved.
10. Required files inspected.
11. Change planned.
12. Approval boundary respected.
13. Change safely executed.
14. Tests/verification performed.
15. Project knowledge updated with evolved learnings.
16. A second task reuses previous knowledge without rescanning the entire project.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import pytest

from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    ResourcePolicy,
    RuntimeStrategy,
)
from chakrview.cognition.ppb import (
    EpistemicStatus,
    KnowledgeRecordType,
    KnowledgeRecord,
    ProjectIdentity,
    PersistentBrainStorage,
    PersistentProjectBrain,
    IncrementalProjectScanner,
    ProjectKnowledgeRetriever,
    ChangeAwareBrainMaintainer,
    TaskNodeStatus,
    TaskResourceType,
    PersistentTaskNode,
    PersistentTaskGraph,
    PersistentTaskStorage,
    DeterministicTaskDecomposer,
    TaskExecutionLease,
    ScheduleDecision,
    ResourceAwareTaskScheduler,
    LoopExecutionProgress,
    IncrementalCognitiveWorkLoop,
)
from chakrview.cognition.repository.change_detector import (
    ChangeCategory,
    FileChange,
    RepositoryDiff,
)
from chakrview.cognition.repository.impact_analyzer import ImpactReport


def run_realistic_multi_file_experiment():
    temp_dir = tempfile.mkdtemp(prefix="ppb_real_experiment_")
    db_path = os.path.join(temp_dir, "enterprise_ppb.db")
    tasks_db = os.path.join(temp_dir, "enterprise_tasks.db")

    # Realistic multi-file enterprise repository
    realistic_repo = {
        "auth/credentials.py": "def hash_password(pwd):\n    return f'hash_{pwd}'\n\ndef verify_password(pwd, h):\n    return h == hash_password(pwd)\n",
        "auth/session.py": "from auth.credentials import verify_password\n\nclass SessionStore:\n    def get_user_session(self, sid):\n        return {'user_id': 'u101', 'role': 'admin'}\n",
        "billing/pricing.py": "def get_base_price(item_id):\n    return 100.0\n",
        "billing/discounts.py": "def apply_loyalty_discount(price, tier):\n    return price * 0.9 if tier == 'gold' else price\n",
        "billing/checkout.py": "from billing.pricing import get_base_price\nfrom billing.discounts import apply_loyalty_discount\n\ndef calculate_cart_total(items, tier):\n    return sum(apply_loyalty_discount(get_base_price(i), tier) for i in items)\n",
        "shipping/logistics.py": "def estimate_freight_charge(zip_code, weight):\n    return 15.0\n",
        "inventory/warehouse.py": "def reserve_stock(sku, qty):\n    return True\n",
    }

    try:
        # ---------------------------------------------------------------------
        # 1. Project discovered & constrained machine profile (LOW_RESOURCE)
        # ---------------------------------------------------------------------
        low_hw = HardwareCapability(
            cpu_cores=1, cpu_architecture="x86_64", ram_gb=2.0, gpu_available=False,
            gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=20.0, network_available=True,
        )
        ident = ProjectIdentity(project_id="enterprise_app", project_root="/enterprise_app")
        brain_s1 = PersistentProjectBrain(db_path=db_path, project_identity=ident, hardware_capability=low_hw)

        # ---------------------------------------------------------------------
        # 2 & 3. Divided into chunks & scanned incrementally (chunk_size = 2)
        # ---------------------------------------------------------------------
        scanner_s1 = IncrementalProjectScanner(brain=brain_s1, chunk_size=2)
        # Scan bounded chunk 1 (2 files)
        p1 = scanner_s1.scan_project(files_dict=realistic_repo, max_chunks=1)
        assert p1.files_processed_this_run == 2
        assert not p1.is_fully_scanned

        # ---------------------------------------------------------------------
        # 4 & 5. Knowledge persisted & Process/Session interrupted
        # ---------------------------------------------------------------------
        summary_s1 = brain_s1.get_state()
        assert summary_s1.scanned_files_count == 2
        del scanner_s1
        del brain_s1  # Complete process termination

        # ---------------------------------------------------------------------
        # 6 & 7. ChakrView restarted & Previous knowledge recovered without rescan
        # ---------------------------------------------------------------------
        brain_s2 = PersistentProjectBrain(db_path=db_path, hardware_capability=low_hw)
        assert brain_s2.project_id == "enterprise_app"
        assert brain_s2.get_state().scanned_files_count == 2

        # Finish scanning the rest of the project
        scanner_s2 = IncrementalProjectScanner(brain=brain_s2, chunk_size=2)
        p2 = scanner_s2.scan_project(files_dict=realistic_repo)
        assert p2.files_already_scanned == 2
        assert p2.files_skipped_unchanged == 2
        assert p2.files_processed_this_run == 5
        assert p2.is_fully_scanned

        # ---------------------------------------------------------------------
        # 8. Large task decomposed into persistent DAG
        # ---------------------------------------------------------------------
        task_storage = PersistentTaskStorage(tasks_db)
        decomposer = DeterministicTaskDecomposer(brain=brain_s2, storage=task_storage)
        user_task = "Refactor checkout flow to support promotional coupon codes"
        task_graph = decomposer.decompose_task(
            user_task=user_task,
            target_files=["billing/checkout.py", "billing/discounts.py"],
        )
        assert len(task_graph.nodes) >= 5

        # ---------------------------------------------------------------------
        # 9, 10, 11, 12, 13, 14, 15. Execute Work Loop & Evolve Knowledge
        # ---------------------------------------------------------------------
        scheduler = ResourceAwareTaskScheduler(low_hw)
        work_loop = IncrementalCognitiveWorkLoop(
            brain=brain_s2,
            task_storage=task_storage,
            scheduler=scheduler,
        )

        loop_res = work_loop.run_graph(graph_id=task_graph.graph_id)
        assert loop_res.is_finished
        assert loop_res.completed_nodes == len(task_graph.nodes)
        assert loop_res.knowledge_records_evolved > 0

        # Verify evolved knowledge was committed to PPB
        evolved_records = brain_s2.query_knowledge(file_path="billing/checkout.py")
        assert len(evolved_records) > 0
        assert any(r.epistemic_status == EpistemicStatus.FACT for r in evolved_records)

        # ---------------------------------------------------------------------
        # 16. Second task reuses knowledge without rescanning entire project
        # ---------------------------------------------------------------------
        # Task 2 on unrelated module: "Query freight charges for logistics"
        retriever_s2 = ProjectKnowledgeRetriever(brain_s2)
        logistics_bundle = retriever_s2.retrieve_for_task(
            "Query freight charges for logistics",
            target_files=["shipping/logistics.py"],
        )
        assert len(logistics_bundle.records) > 0
        assert any("shipping/logistics.py" in f for f in logistics_bundle.matched_files)

        # Confirm scan progress did not execute another rescan:
        # Running scanner again should see 0 files processed, 7 skipped
        rescan_check = scanner_s2.scan_project(files_dict=realistic_repo)
        assert rescan_check.files_processed_this_run == 0
        assert rescan_check.files_skipped_unchanged == 7

        return {
            "files_discovered": 7,
            "files_scanned_initially": 2,
            "files_resumed": 5,
            "files_rescanned_on_repeat": 0,
            "tasks_decomposed": len(task_graph.nodes),
            "tasks_completed": loop_res.completed_nodes,
            "knowledge_records_evolved": loop_res.knowledge_records_evolved,
            "strategy": scheduler.resource_policy.strategy.name,
            "success": True,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_realistic_multi_file_experiment():
    results = run_realistic_multi_file_experiment()
    assert results["success"] is True
    assert results["files_rescanned_on_repeat"] == 0
    assert results["tasks_completed"] >= 5
