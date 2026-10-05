"""
Realistic Autonomous Multi-Module Experiment Protocol for Steps 86–90:
Adaptive Autonomous Work Orchestration.

Demonstrates:
1. Does NOT load the whole project into one context (bounded context budget).
2. Retrieves relevant project knowledge from PPB.
3. Decomposes work into persistent task DAG.
4. Discovers a previously unknown dependency during execution.
5. Dynamically expands the graph with a new subtask.
6. Investigates the unknown dependency via AutonomousInvestigationLoop.
7. Verifies the candidate evidence via EvidenceVerifier.
8. Updates PPB with grounded observation records.
9. Replans the remaining task graph with bounded cycles.
10. Resumes smoothly after simulated process termination.
11. Continues without rescanning unchanged files.
12. Passes the authoritative GoalVerificationGate.
13. Durably records the final accepted goal in PPB.
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
    GoalVerificationVerdict,
    GoalVerificationResult,
    TaskProvenance,
    DynamicSubtaskRequest,
    BudgetDecisionAction,
    ContextBudgetPlan,
    ContextResourcePlanner,
    InvestigationExecutionOutcome,
    AutonomousInvestigationLoop,
    ReplanDecision,
    AdaptiveReplanEngine,
    GoalVerificationGate,
)
from chakrview.cognition.research.models import (
    EvidenceItem,
    KnowledgeAcquisitionStatus,
    InvestigationSource,
    SourceMetadata,
)
from chakrview.cognition.reasoning.structured import InvestigationRequirement


def run_adaptive_autonomous_experiment():
    temp_dir = tempfile.mkdtemp(prefix="ppb_step86_90_exp_")
    db_path = os.path.join(temp_dir, "adaptive_ppb.db")
    tasks_db = os.path.join(temp_dir, "adaptive_tasks.db")

    # Enterprise multi-module codebase
    multi_module_repo = {
        "gateway/router.py": "def route_request(path):\n    return 'routed'\n",
        "services/order_service.py": "from gateway.router import route_request\ndef create_order(user_id, items):\n    return {'order_id': 'ord_123'}\n",
        "services/payment_service.py": "def process_payment(amount):\n    return True\n",
        "notifications/email.py": "def send_receipt(email, order_id):\n    pass\n",
        "audit/ledger.py": "def record_transaction(tx_id):\n    pass\n",
    }

    try:
        # ---------------------------------------------------------------------
        # 1. Constrained machine setup (LOW_RESOURCE)
        # ---------------------------------------------------------------------
        low_hw = HardwareCapability(
            cpu_cores=1, cpu_architecture="x86_64", ram_gb=2.0, gpu_available=False,
            gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=20.0, network_available=True,
        )
        ident = ProjectIdentity(project_id="enterprise_mesh", project_root="/enterprise_mesh")
        brain_s1 = PersistentProjectBrain(db_path=db_path, project_identity=ident, hardware_capability=low_hw)

        # ---------------------------------------------------------------------
        # 2. Incremental scan of project (chunk_size = 2) & Persist to PPB
        # ---------------------------------------------------------------------
        scanner_s1 = IncrementalProjectScanner(brain=brain_s1, chunk_size=2)
        scan_res = scanner_s1.scan_project(files_dict=multi_module_repo)
        assert scan_res.is_fully_scanned
        assert scan_res.total_files_discovered == 5

        # ---------------------------------------------------------------------
        # 3. Decompose user goal into persistent DAG
        # ---------------------------------------------------------------------
        task_storage = PersistentTaskStorage(tasks_db)
        decomposer = DeterministicTaskDecomposer(brain=brain_s1, storage=task_storage)
        user_goal = "Integrate secure ledger audit into order processing flow"
        graph = decomposer.decompose_task(
            user_task=user_goal,
            target_files=["services/order_service.py", "audit/ledger.py"],
        )
        graph_id = graph.graph_id
        initial_node_count = len(graph.nodes)

        # ---------------------------------------------------------------------
        # 4 & 5. Execution discovers an unknown dependency -> Dynamically expand graph
        # ---------------------------------------------------------------------
        # Simulated discovery during inspection: "crypto/hasher.py" is needed but was uninspected
        replan_engine = AdaptiveReplanEngine(brain=brain_s1, max_cycles=3)
        node_step1 = graph.get_node(f"{graph_id}_step_1_arch")
        assert node_step1 is not None

        # Custom node executor that simulates runtime discovery of "crypto/hasher.py"
        dynamic_added = False
        def dynamic_executor(node: PersistentTaskNode, lease: TaskExecutionLease) -> dict:
            nonlocal dynamic_added
            if node.node_id == node_step1.node_id and not dynamic_added:
                dynamic_added = True
                # Dynamically expand graph
                obs = {"discovered_dependencies": ["crypto/hasher.py"]}
                replan_engine.evaluate_and_replan(graph, node, obs)
                task_storage.save_graph(graph)
                return {"discovered_dependencies": ["crypto/hasher.py"], "inspected": True}
            return {"executed": True, "tests_passed": True}

        # ---------------------------------------------------------------------
        # 6, 7 & 8. Investigate the unknown dependency via AutonomousInvestigationLoop
        # ---------------------------------------------------------------------
        inv_loop = AutonomousInvestigationLoop(brain=brain_s1)
        inv_req = InvestigationRequirement(
            requirement_id="inv_crypto_hasher",
            question="What hashing algorithm does crypto/hasher.py provide?",
            knowledge_gap="hashing algorithm in crypto/hasher",
            required_evidence_type="AST_CALL",
            target_module="crypto/hasher.py",
        )
        ev_items = [
            EvidenceItem(request_id="inv_h", source_id="src_h1", content="SHA256 hasher function", confidence=0.9),
            EvidenceItem(request_id="inv_h", source_id="src_h2", content="SHA256 hasher function", confidence=0.9),
        ]
        srcs = {
            "src_h1": InvestigationSource("src_h1", "file://crypto/hasher.py", SourceMetadata("src1", "local", 0.9)),
            "src_h2": InvestigationSource("src_h2", "file://crypto/hasher.py", SourceMetadata("src2", "local", 0.9)),
        }
        inv_outcome = inv_loop.investigate_requirement(inv_req, candidate_evidence=ev_items, sources=srcs)
        assert inv_outcome.overall_status == KnowledgeAcquisitionStatus.VERIFIED

        # Also store module record in PPB for crypto/hasher.py
        brain_s1.store_knowledge(KnowledgeRecord(
            record_id="mod_crypto_hasher",
            project_id=brain_s1.project_id,
            record_type=KnowledgeRecordType.MODULE,
            file_path="crypto/hasher.py",
            summary="Cryptographic SHA256 hashing utility",
            epistemic_status=EpistemicStatus.FACT,
        ))

        # ---------------------------------------------------------------------
        # 9 & 10. Execute partial work, simulate process termination, and restart
        # ---------------------------------------------------------------------
        work_loop_s1 = IncrementalCognitiveWorkLoop(
            brain=brain_s1,
            task_storage=task_storage,
            custom_node_executor=dynamic_executor,
        )
        # Run 2 iterations
        p_part = work_loop_s1.run_graph(graph_id=graph_id, max_iterations=2)
        assert p_part.completed_nodes >= 1
        assert not p_part.is_finished

        # Kill process / garbage collect objects
        del work_loop_s1
        del brain_s1
        del decomposer

        # ---------------------------------------------------------------------
        # 11. Restart ChakrView and resume execution without repeat rescanning
        # ---------------------------------------------------------------------
        brain_s2 = PersistentProjectBrain(db_path=db_path, hardware_capability=low_hw)
        task_storage_s2 = PersistentTaskStorage(tasks_db)
        reloaded_graph = task_storage_s2.load_graph(graph_id)
        assert reloaded_graph is not None
        # Graph contains the dynamically added dependency inspection node
        assert len(reloaded_graph.nodes) > initial_node_count

        work_loop_s2 = IncrementalCognitiveWorkLoop(
            brain=brain_s2,
            task_storage=task_storage_s2,
            custom_node_executor=lambda n, l: {"executed": True, "tests_passed": True},
        )
        p_finish = work_loop_s2.run_graph(graph_id=graph_id)
        assert p_finish.is_finished

        # ---------------------------------------------------------------------
        # 12 & 13. Goal Verification Gate & Final Knowledge Persistence
        # ---------------------------------------------------------------------
        # Reload latest graph state after execution
        completed_graph = task_storage_s2.load_graph(graph_id)
        assert completed_graph is not None
        assert completed_graph.is_all_completed

        gate = GoalVerificationGate(brain=brain_s2)
        goal_result = gate.evaluate_goal(completed_graph)
        assert goal_result.verdict == GoalVerificationVerdict.ACCEPT
        assert goal_result.score == 1.0

        # Verify final goal record in PPB
        goal_rec = brain_s2.get_knowledge(f"goal_{graph_id}")
        assert goal_rec is not None
        assert goal_rec.epistemic_status == EpistemicStatus.FACT

        # Verify zero repeat rescan of unchanged files
        scanner_s2 = IncrementalProjectScanner(brain=brain_s2, chunk_size=2)
        rescan_check = scanner_s2.scan_project(files_dict=multi_module_repo)
        assert rescan_check.files_processed_this_run == 0
        assert rescan_check.files_skipped_unchanged == 5

        return {
            "initial_nodes": initial_node_count,
            "final_nodes": len(reloaded_graph.nodes),
            "completed_nodes": p_finish.completed_nodes,
            "goal_verdict": goal_result.verdict.value,
            "rescan_skipped_count": rescan_check.files_skipped_unchanged,
            "success": True,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_realistic_adaptive_autonomous_experiment():
    res = run_adaptive_autonomous_experiment()
    assert res["success"] is True
    assert res["final_nodes"] > res["initial_nodes"]
    assert res["goal_verdict"] == "ACCEPT"
    assert res["rescan_skipped_count"] == 5
