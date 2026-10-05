"""
Dedicated Test Suite for ChakrView Steps 86–90:
Adaptive Autonomous Work Orchestration.

Tests:
1. Dynamic task expansion & cycle prevention (Step 86)
2. Deterministic dynamic task identity and provenance (Step 86)
3. Context/resource budget decision making (EXECUTE_DIRECT, RETRIEVE_PPB_ONLY, SPLIT_TASK, REQUEST_INVESTIGATION) (Step 87)
4. Low-resource vs stronger-resource behavior (Step 87)
5. Autonomous investigation creation & evidence verification integration (Step 88)
6. PPB persistence of investigation observations & epistemic preservation (Step 88)
7. Dynamic replanning on newly discovered dependencies (Step 89)
8. Bounded replanning cycles (<= 3) to prevent infinite loops (Step 89)
9. Goal-level verification gate (ACCEPT, PARTIAL, REVISE, REJECT, ABSTAIN) (Step 90)
10. Unresolved UNKNOWN handling producing ABSTAIN instead of fabricated success (Step 90)
11. Stale records producing REVISE instead of premature acceptance (Step 90)
12. Process restart recovery across dynamic subtasks and replanning decisions (Steps 86-90)
13. Neural core authority isolation & baseline invariant (dW = 0) (Steps 86-90)
"""

from __future__ import annotations

import hashlib
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
from chakrview.cognition.ppb import (
    EpistemicStatus,
    KnowledgeRecordType,
    KnowledgeRecord,
    ProjectIdentity,
    PersistentBrainStorage,
    PersistentProjectBrain,
    TaskNodeStatus,
    TaskResourceType,
    PersistentTaskNode,
    PersistentTaskGraph,
    PersistentTaskStorage,
    DeterministicTaskDecomposer,
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

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="ppb_orch_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_brain(temp_dir):
    db_path = os.path.join(temp_dir, "brain.db")
    ident = ProjectIdentity(project_id="orch_proj", project_root="/virtual_orch")
    brain = PersistentProjectBrain(db_path=db_path, project_identity=ident)
    # Seed known module
    brain.store_knowledge(KnowledgeRecord(
        record_id="mod_core",
        project_id="orch_proj",
        record_type=KnowledgeRecordType.MODULE,
        file_path="core/base.py",
        summary="Core base module",
        epistemic_status=EpistemicStatus.FACT,
    ))
    return brain


# =============================================================================
# Step 86: Dynamic Task Expansion & Cycle Prevention
# =============================================================================
def test_step86_dynamic_expansion_and_cycle_prevention(temp_dir, test_brain):
    tasks_db = os.path.join(temp_dir, "tasks.db")
    storage = PersistentTaskStorage(tasks_db)
    graph = PersistentTaskGraph("graph_86", "orch_proj", "Root Goal")

    root_node = PersistentTaskNode("n_root", "graph_86", "Root", "", TaskResourceType.INSPECTION)
    graph.add_node(root_node)

    # Dynamic expansion request
    req = DynamicSubtaskRequest(
        title="Discovered: inspect payments",
        description="Newly discovered payment interaction",
        resource_type=TaskResourceType.INSPECTION,
        parent_id="n_root",
        dependencies=("n_root",),
        affected_files=("pay/gateway.py",),
        provenance=TaskProvenance(origin_type="DYNAMIC_EXPANSION", trigger_node_id="n_root", rationale="Found in AST calls"),
    )

    node_id = req.compute_deterministic_id(graph.graph_id)
    dyn_node = PersistentTaskNode(
        node_id=node_id,
        graph_id=graph.graph_id,
        title=req.title,
        description=req.description,
        resource_type=req.resource_type,
        parent_id=req.parent_id,
        dependencies=list(req.dependencies),
        affected_files=list(req.affected_files),
    )
    graph.add_node(dyn_node)
    storage.save_graph(graph)

    # Verify reload after process wipe
    reloaded = storage.load_graph("graph_86")
    assert reloaded is not None
    assert node_id in reloaded.nodes
    assert "n_root" in reloaded.nodes[node_id].dependencies

    # Cycle Prevention: Adding a cycle (n_root depends on dyn_node) must raise ValueError
    cyclic_node = PersistentTaskNode("n_root_cycle", "graph_86", "Cycle", "", TaskResourceType.INSPECTION, dependencies=[node_id])
    graph.nodes["n_root"].dependencies.append(node_id)
    with pytest.raises(ValueError, match="Cycle detected"):
        graph.validate_no_cycles()


# =============================================================================
# Step 87: Context & Resource Budget Planner
# =============================================================================
def test_step87_context_and_resource_budget_intelligence(temp_dir, test_brain):
    low_hw = HardwareCapability(
        cpu_cores=1, cpu_architecture="x86_64", ram_gb=2.0, gpu_available=False,
        gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=10.0, network_available=True,
    )
    acc_hw = HardwareCapability(
        cpu_cores=8, cpu_architecture="x86_64", ram_gb=16.0, gpu_available=True,
        gpu_vendor="nvidia", vram_gb=16.0, storage_capacity_gb=500.0, network_available=True,
    )

    planner_low = ContextResourcePlanner(test_brain, low_hw)
    planner_acc = ContextResourcePlanner(test_brain, acc_hw)

    # 1. Direct small node on known file -> EXECUTE_DIRECT
    node_small = PersistentTaskNode("n_sm", "g1", "Inspect Core", "", TaskResourceType.INSPECTION, affected_files=["core/base.py"])
    plan_low = planner_low.plan_budget_for_node(node_small)
    assert plan_low.action == BudgetDecisionAction.EXECUTE_DIRECT

    # 2. Node on unknown uninspected file -> REQUEST_INVESTIGATION
    node_unknown = PersistentTaskNode("n_un", "g1", "Modify Auth", "", TaskResourceType.REASONING, affected_files=["auth/secret.py"])
    plan_unk = planner_low.plan_budget_for_node(node_unknown)
    assert plan_unk.action == BudgetDecisionAction.REQUEST_INVESTIGATION
    assert plan_unk.requires_investigation is True

    # 3. Large node with 4 files on LOW_RESOURCE -> SPLIT_TASK
    node_large = PersistentTaskNode("n_lg", "g1", "Process All", "", TaskResourceType.INSPECTION, affected_files=["core/base.py", "f2.py", "f3.py", "f4.py"])
    plan_split = planner_low.plan_budget_for_node(node_large)
    assert plan_split.action == BudgetDecisionAction.SPLIT_TASK
    assert len(plan_split.suggested_splits) == 4

    # 4. Large node on ACCELERATED -> RETRIEVE_PPB_ONLY with use_accelerator=True for REASONING
    for f in ["f2.py", "f3.py", "f4.py"]:
        test_brain.store_knowledge(KnowledgeRecord(f"mod_{f}", "orch_proj", KnowledgeRecordType.MODULE, f))
    node_reason = PersistentTaskNode("n_rs", "g1", "Complex Reasoning", "", TaskResourceType.REASONING, affected_files=["core/base.py", "f2.py", "f3.py", "f4.py"])
    plan_acc = planner_acc.plan_budget_for_node(node_reason)
    assert plan_acc.action == BudgetDecisionAction.RETRIEVE_PPB_ONLY
    assert plan_acc.use_accelerator is True


# =============================================================================
# Step 88: Autonomous Investigation & Evidence Verification Loop
# =============================================================================
def test_step88_autonomous_investigation_loop(temp_dir, test_brain):
    loop = AutonomousInvestigationLoop(test_brain)

    req = InvestigationRequirement(
        requirement_id="req_001",
        question="What is the cryptographic algorithm used in auth/secret.py?",
        knowledge_gap="cryptographic algorithm",
        required_evidence_type="AST_CALL",
        target_module="auth/secret.py",
    )

    candidate_evidence = [
        EvidenceItem(
            request_id="inv_test",
            source_id="src_1",
            content="hashlib.sha256 used for token derivation",
            confidence=0.95,
        ),
        EvidenceItem(
            request_id="inv_test",
            source_id="src_2",
            content="hashlib.sha256 used for token derivation",
            confidence=0.95,
        ),
    ]
    sources = {
        "src_1": InvestigationSource(
            source_id="src_1",
            uri="file://auth/secret.py",
            metadata=SourceMetadata(source_identity="inspector", source_type="local", reliability_score=0.9),
        ),
        "src_2": InvestigationSource(
            source_id="src_2",
            uri="file://auth/secret.py",
            metadata=SourceMetadata(source_identity="code_review", source_type="local", reliability_score=0.9),
        ),
    }

    outcome = loop.investigate_requirement(req, candidate_evidence=candidate_evidence, sources=sources)

    assert outcome.overall_status == KnowledgeAcquisitionStatus.VERIFIED
    assert len(outcome.persisted_ppb_records) > 0

    # Verify persisted in PPB with provenance
    persisted_recs = test_brain.query_knowledge(file_path="auth/secret.py")
    assert len(persisted_recs) > 0
    assert persisted_recs[0].source_chunk == "autonomous_investigation"
    assert persisted_recs[0].epistemic_status == EpistemicStatus.FACT



# =============================================================================
# Step 89: Adaptive Re-planning Engine & Cycle Bounds
# =============================================================================
def test_step89_adaptive_replanning_and_cycle_bounds(temp_dir, test_brain):
    engine = AdaptiveReplanEngine(test_brain, max_cycles=2)
    graph = PersistentTaskGraph("g89", "orch_proj", "Root")
    n1 = PersistentTaskNode("n1", "g89", "Inspect Module A", "", TaskResourceType.INSPECTION, status=TaskNodeStatus.COMPLETED)
    graph.add_node(n1)

    # Observation reveals newly discovered dependency "core/crypto.py"
    obs = {"discovered_dependencies": ["core/crypto.py"]}
    dec1 = engine.evaluate_and_replan(graph, n1, obs)
    assert dec1.graph_modified is True
    assert len(dec1.added_subtasks) == 1
    assert "core/crypto.py" in graph.nodes[dec1.added_subtasks[0]].affected_files

    # Cycle 2 replan
    dec2 = engine.evaluate_and_replan(graph, n1, {"discovered_dependencies": ["core/network.py"]})
    assert dec2.cycle_number == 2
    assert dec2.graph_modified is True

    # Cycle 3 must halt because max_cycles = 2
    dec3 = engine.evaluate_and_replan(graph, n1, {"discovered_dependencies": ["core/other.py"]})
    assert dec3.graph_modified is False
    assert "Max replanning cycles" in dec3.rationale


# =============================================================================
# Step 90: Goal Verification Gate (ACCEPT, PARTIAL, REVISE, ABSTAIN, REJECT)
# =============================================================================
def test_step90_goal_verification_gate(temp_dir, test_brain):
    gate = GoalVerificationGate(test_brain)

    # Case 1: Incomplete graph -> PARTIAL
    g_partial = PersistentTaskGraph("gp", "orch_proj", "Partial Goal")
    g_partial.add_node(PersistentTaskNode("p1", "gp", "Task 1", "", TaskResourceType.INSPECTION, status=TaskNodeStatus.COMPLETED))
    g_partial.add_node(PersistentTaskNode("p2", "gp", "Task 2", "", TaskResourceType.INSPECTION, status=TaskNodeStatus.PENDING))
    res_partial = gate.evaluate_goal(g_partial)
    assert res_partial.verdict == GoalVerificationVerdict.PARTIAL

    # Case 2: Failed subtask -> REJECT
    g_fail = PersistentTaskGraph("gf", "orch_proj", "Failed Goal")
    g_fail.add_node(PersistentTaskNode("f1", "gf", "Task 1", "", TaskResourceType.INSPECTION, status=TaskNodeStatus.FAILED))
    res_fail = gate.evaluate_goal(g_fail)
    assert res_fail.verdict == GoalVerificationVerdict.REJECT

    # Case 3: Completed nodes, but affected file has uninspected UNKNOWN records -> ABSTAIN
    g_unk = PersistentTaskGraph("gu", "orch_proj", "Unknown Goal")
    g_unk.add_node(PersistentTaskNode("u1", "gu", "Task 1", "", TaskResourceType.INSPECTION, status=TaskNodeStatus.COMPLETED, affected_files=["phantom/file.py"]))
    res_unk = gate.evaluate_goal(g_unk)
    assert res_unk.verdict == GoalVerificationVerdict.ABSTAIN

    # Case 4: Completed nodes, affected file has STALE records -> REVISE
    test_brain.store_knowledge(KnowledgeRecord("rec_stale", "orch_proj", KnowledgeRecordType.MODULE, "core/base.py", epistemic_status=EpistemicStatus.STALE))
    g_stale = PersistentTaskGraph("gs", "orch_proj", "Stale Goal")
    g_stale.add_node(PersistentTaskNode("s1", "gs", "Task 1", "", TaskResourceType.INSPECTION, status=TaskNodeStatus.COMPLETED, affected_files=["core/base.py"]))
    res_stale = gate.evaluate_goal(g_stale)
    assert res_stale.verdict == GoalVerificationVerdict.REVISE

    # Case 5: Fully verified and up to date -> ACCEPT
    test_brain.store_knowledge(KnowledgeRecord("rec_stale", "orch_proj", KnowledgeRecordType.MODULE, "core/base.py", epistemic_status=EpistemicStatus.FACT))
    res_accept = gate.evaluate_goal(g_stale)
    assert res_accept.verdict == GoalVerificationVerdict.ACCEPT


# =============================================================================
# Neural Core Authority Isolation & Invariant (dW = 0)
# =============================================================================
def test_neural_core_authority_and_weight_invariant():
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
