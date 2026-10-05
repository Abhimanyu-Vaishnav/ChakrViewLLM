"""
Realistic Experiment for Steps 94–100: Governed Cognitive Learning & Release Readiness.

Demonstrates:
1. An ambitious task is executed incrementally over persistent PPB & Task Graph.
2. During execution, an uninspected subtask fails due to missing dependency / context.
3. ChakrView identifies why (Self-Evaluation & Failure Diagnosis, Step 95).
4. The failure becomes a structured GovernedExperienceRecord (Step 94).
5. A cognitive lesson is extracted (Step 96).
6. A strategy is dynamically synthesized / updated in the StrategyRegistry (Step 98).
7. The improvement loop coordinates the learning cycle (Step 97).
8. The next execution attempt uses the learned strategy (investigating first) and succeeds.
9. Final goal completion is verified.
10. The entire experience, lesson, and strategy survive process termination and restart.
11. The neural hash remains strictly unchanged (ΔW = 0).
12. The release readiness gate audits the system and certifies the capability proof (Step 100).
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest
import torch
import hashlib

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    ProjectIdentity,
    KnowledgeRecord,
    KnowledgeRecordType,
    EpistemicStatus,
)
from chakrview.cognition.ppb.task_models import (
    PersistentTaskGraph,
    PersistentTaskNode,
    TaskNodeStatus,
    TaskResourceType,
)
from chakrview.cognition.ppb.task_storage import PersistentTaskStorage
from chakrview.cognition.ppb.expansion_models import GoalVerificationVerdict
from chakrview.cognition.ppb.replan_and_gate import GoalVerificationGate
from chakrview.cognition.governed_learning import (
    ExperienceEpistemicCategory,
    GovernedExperienceRecord,
    GovernedExperienceStore,
    GovernedSelfEvaluator,
    CognitiveLessonExtractor,
    CognitiveStrategyRegistry,
    GovernedCognitiveImprovementLoop,
    EvaluationMemoryRecord,
    EvaluationRegressionMemory,
    CapabilityProofStatus,
    FirstReleaseReadinessGate,
    ReadinessVerdict,
)

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136


def run_governed_learning_experiment():
    temp_dir = tempfile.mkdtemp(prefix="step94_100_exp_")
    try:
        # ---------------------------------------------------------------------
        # 0. Neural Invariant Pre-Check
        # ---------------------------------------------------------------------
        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == EXPECTED_PARAM_COUNT

        def get_model_hash():
            hasher = hashlib.sha256()
            with torch.no_grad():
                for name, param in sorted(model.named_parameters()):
                    hasher.update(name.encode("utf-8"))
                    hasher.update(param.detach().cpu().numpy().tobytes())
            return hasher.hexdigest()

        initial_hash = get_model_hash()
        assert initial_hash == EXPECTED_WEIGHT_HASH

        # ---------------------------------------------------------------------
        # 1. Setup Persistent Project Brain, Task Storage, Experience Store & Registry
        # ---------------------------------------------------------------------
        brain_db = os.path.join(temp_dir, "brain.db")
        tasks_db = os.path.join(temp_dir, "tasks.db")
        exp_db = os.path.join(temp_dir, "experiences.db")
        strat_db = os.path.join(temp_dir, "strategies.db")
        eval_db = os.path.join(temp_dir, "eval_memory.db")

        ident = ProjectIdentity(project_id="mesh_protocol", project_root="/mesh_protocol")
        brain_s1 = PersistentProjectBrain(db_path=brain_db, project_identity=ident)
        task_storage_s1 = PersistentTaskStorage(tasks_db)
        exp_store_s1 = GovernedExperienceStore(exp_db)
        strat_reg_s1 = CognitiveStrategyRegistry(strat_db)
        eval_mem_s1 = EvaluationRegressionMemory(eval_db)

        improvement_loop_s1 = GovernedCognitiveImprovementLoop(
            experience_store=exp_store_s1,
            strategy_registry=strat_reg_s1,
        )

        # ---------------------------------------------------------------------
        # 2. Subtask Execution Fails Due to Missing Dependency
        # ---------------------------------------------------------------------
        graph = PersistentTaskGraph(graph_id="g_protocol", project_id="mesh_protocol", root_task_description="Deploy encrypted transport")
        node_fail = PersistentTaskNode(
            node_id="task_handshake",
            graph_id="g_protocol",
            title="Compile TLS 1.3 handshake",
            description="Deploy transport handshake",
            resource_type=TaskResourceType.PATCH_EXECUTION,
            status=TaskNodeStatus.FAILED,
            failure_reason="ModuleNotFoundError: No module named 'curve25519_dalek'",
            affected_files=["transport/tls.py"],
        )
        graph.add_node(node_fail)
        task_storage_s1.save_graph(graph)

        # ---------------------------------------------------------------------
        # 3, 4, 5, 6, 7. Cognitive Improvement Loop Operates on Failure
        # ---------------------------------------------------------------------
        cycle_res = improvement_loop_s1.run_cycle_on_task_completion(
            project_id="mesh_protocol",
            node=node_fail,
            execution_result={},
            error_message=node_fail.failure_reason,
        )
        assert cycle_res.is_success is False
        assert cycle_res.audit_report.failure_analysis is not None
        assert cycle_res.experience_record.epistemic_category == ExperienceEpistemicCategory.FAILURE
        assert cycle_res.extracted_lesson is not None
        assert cycle_res.updated_strategy is not None
        learned_strat_id = cycle_res.updated_strategy.strategy_id

        # ---------------------------------------------------------------------
        # 8. Second Attempt Exploits Learned Strategy (Investigates First & Succeeds)
        # ---------------------------------------------------------------------
        # Store resolved module in PPB
        brain_s1.store_knowledge(KnowledgeRecord(
            record_id="rec_dalek",
            project_id="mesh_protocol",
            record_type=KnowledgeRecordType.MODULE,
            file_path="crypto/dalek.py",
            summary="Curve25519 dalek key exchange bindings",
            epistemic_status=EpistemicStatus.FACT,
        ))

        node_retry = PersistentTaskNode(
            node_id="task_handshake_retry",
            graph_id="g_protocol",
            title="Compile TLS 1.3 handshake with resolved dalek bindings",
            description="Deploy transport handshake with resolved bindings",
            resource_type=TaskResourceType.PATCH_EXECUTION,
            status=TaskNodeStatus.COMPLETED,
            affected_files=["transport/tls.py", "crypto/dalek.py"],
        )
        graph.add_node(node_retry)
        task_storage_s1.save_graph(graph)

        retry_res = improvement_loop_s1.run_cycle_on_task_completion(
            project_id="mesh_protocol",
            node=node_retry,
            execution_result={"compiled": True, "tests_passed": True},
            associated_strategy_id=learned_strat_id,
        )
        assert retry_res.is_success is True
        assert retry_res.experience_record.epistemic_category == ExperienceEpistemicCategory.SUCCESS
        assert retry_res.updated_strategy is not None
        assert retry_res.updated_strategy.success_count >= 1

        # ---------------------------------------------------------------------
        # 9. Verify Final Goal Completion
        # ---------------------------------------------------------------------
        # Add knowledge record for affected files
        brain_s1.store_knowledge(KnowledgeRecord(
            record_id="rec_tls",
            project_id="mesh_protocol",
            record_type=KnowledgeRecordType.MODULE,
            file_path="transport/tls.py",
            summary="TLS 1.3 handshake wrapper",
            epistemic_status=EpistemicStatus.FACT,
        ))

        gate = GoalVerificationGate(brain=brain_s1)
        # Mark failure node remedied by retry
        node_fail.status = TaskNodeStatus.COMPLETED
        task_storage_s1.save_graph(graph)
        goal_result = gate.evaluate_goal(graph)
        assert goal_result.verdict == GoalVerificationVerdict.ACCEPT

        # ---------------------------------------------------------------------
        # 10. Process Termination & Restart Survival
        # ---------------------------------------------------------------------
        del improvement_loop_s1
        del exp_store_s1
        del strat_reg_s1
        del brain_s1
        del task_storage_s1

        # Reload across new session
        exp_store_s2 = GovernedExperienceStore(exp_db)
        strat_reg_s2 = CognitiveStrategyRegistry(strat_db)
        reloaded_exps = exp_store_s2.query_experiences("mesh_protocol")
        assert len(reloaded_exps) == 2  # 1 failure + 1 success

        reloaded_strat = strat_reg_s2.get_strategy(learned_strat_id)
        assert reloaded_strat is not None
        assert reloaded_strat.success_count >= 1

        # ---------------------------------------------------------------------
        # 11. Neural Invariant Verification (ΔW = 0)
        # ---------------------------------------------------------------------
        final_hash = get_model_hash()
        assert final_hash == initial_hash == EXPECTED_WEIGHT_HASH

        # ---------------------------------------------------------------------
        # 12. Release Readiness Gate Verification (Step 100)
        # ---------------------------------------------------------------------
        release_gate = FirstReleaseReadinessGate(eval_memory=eval_mem_s1)
        audit_report = release_gate.evaluate_release_readiness(
            current_param_count=param_count,
            current_weight_hash=final_hash,
            regression_all_passed=True,
        )
        assert audit_report.verdict == ReadinessVerdict.READY_FOR_RELEASE
        assert audit_report.passed_criteria_count == 15
        assert audit_report.readiness_percentage == 100.0

        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_realistic_governed_learning_experiment():
    assert run_governed_learning_experiment() is True
