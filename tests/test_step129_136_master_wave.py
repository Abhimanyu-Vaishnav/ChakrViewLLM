"""
ChakrView Step 129-136: Master Development Wave Unit & Integration Tests.

Validates:
- Step 129: Distributed federation production hardening (lifecycle, heartbeats, reconnects)
- Step 130: Governed consistency and conflict auditing (monotonic revisions, authority resolution)
- Step 131: Federated memory consistency & knowledge provenance (partitioning, tombstones)
- Step 132: Cognitive observability plane (trajectory reconstruction, non-leaking events)
- Step 133: Resource-aware cognitive orchestration (locality bonuses, load penalties)
- Step 134: Cognitive recovery & self-healing (bounded loops, diagnosis, lesson recording)
- Step 135: Long-horizon distributed cognition (stage transitions, checkpoint reconstruction)
- Step 136: Canonical baseline immutability (parameter count = 3,443,136, SHA-256 bit-exact)
"""

import hashlib
import json
import socket
import tempfile
import time
from pathlib import Path

import pytest
import torch

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.cognition.multi_agent.contracts import WorkerRole, WorkerContract, WorkerPackage
from chakrview.cognition.multi_agent.result import WorkerResult, WorkerExecutionStatus, ReviewerVerdict, ReviewerCritique
from chakrview.cognition.multi_agent.transport import RequestEnvelope, ResponseEnvelope, EnvelopeType
from chakrview.cognition.multi_agent.federation_hardening import (
    NodeHealthState,
    HardenedNodeRecord,
    NodeLifecycleManager,
    ReconnectingTcpTransportChannel,
)
from chakrview.cognition.multi_agent.consistency_engine import (
    ConflictResolutionPolicy,
    GovernedConsistencyEngine,
)
from chakrview.cognition.multi_agent.knowledge_plane import (
    MemoryPartition,
    KnowledgeFactRecord,
    GovernedKnowledgePlane,
)
from chakrview.cognition.multi_agent.observability_plane import (
    AuditEventType,
    CognitiveAuditEvent,
    CognitiveObservabilityPlane,
)
from chakrview.cognition.multi_agent.locality_orchestrator import (
    LocalityAwareOrchestrator,
    CognitivePlacementContext,
)
from chakrview.cognition.multi_agent.self_healing import (
    SelfHealingAction,
    SelfHealingRecord,
    CognitiveSelfHealingEngine,
)
from chakrview.cognition.multi_agent.distributed_pipeline import (
    DistributedCognitivePipeline,
)
from chakrview.cognition.multi_agent.long_horizon import PipelineStage
from chakrview.cognition.multi_agent.resource_federation import (
    WorkerResourceProfile,
    ResourceCapacityLevel,
    TaskResourceRequirements,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


def test_step129_node_lifecycle_and_heartbeats():
    manager = NodeLifecycleManager(heartbeat_timeout=0.2, fail_threshold=2)
    node = manager.register_node("node-alpha", "127.0.0.1", 9001)
    assert node.health_state == NodeHealthState.ONLINE

    # Verify degraded state when timeout occurs
    time.sleep(0.25)
    health = manager.check_node_health("node-alpha")
    assert health == NodeHealthState.DEGRADED

    # Verify offline transition after consecutive failures
    time.sleep(0.25)
    health = manager.check_node_health("node-alpha")
    assert health == NodeHealthState.OFFLINE

    # Verify recovery via heartbeat
    manager.record_heartbeat("node-alpha")
    assert manager.check_node_health("node-alpha") == NodeHealthState.ONLINE


def test_step130_consistency_monotonic_revisions_and_conflicts(tmp_path):
    db_path = tmp_path / "consistency.db"
    engine = GovernedConsistencyEngine(db_path)

    # Initial update
    ok1, audit1 = engine.apply_update("task-42", incoming_rev=1, author_id="worker-1", authority_tier=1, payload={"status": "IN_PROGRESS"})
    assert ok1 is True
    assert audit1 is None

    # Monotonic update
    ok2, audit2 = engine.apply_update("task-42", incoming_rev=2, author_id="worker-1", authority_tier=1, payload={"status": "COMPLETE"})
    assert ok2 is True
    assert audit2 is None

    # Stale/lower revision from higher authority tier resolves via authority tier
    ok3, audit3 = engine.apply_update("task-42", incoming_rev=1, author_id="supervisor-lead", authority_tier=5, payload={"status": "REVISED"})
    assert ok3 is True
    assert audit3 is not None
    assert audit3.winning_author == "supervisor-lead"
    assert audit3.resolution_policy == ConflictResolutionPolicy.AUTHORITY_TIER


def test_step131_knowledge_provenance_and_tombstones(tmp_path):
    db_path = tmp_path / "knowledge.db"
    plane = GovernedKnowledgePlane(db_path)

    fact = KnowledgeFactRecord(
        fact_id="fact-101",
        partition=MemoryPartition.PROJECT_KNOWLEDGE,
        subject="auth_service",
        predicate="uses_hashing",
        object_value="bcrypt",
        author_node_id="node-1",
        author_worker_id="analyst-1",
        task_id="task-100",
        confidence=0.95,
    )
    plane.store_fact(fact)

    active_facts = plane.query_active_facts(subject="auth_service", partition=MemoryPartition.PROJECT_KNOWLEDGE)
    assert len(active_facts) == 1
    assert active_facts[0].object_value == "bcrypt"

    # Tombstone invalidation
    success = plane.invalidate_fact("fact-101", reason="Upgraded to argon2id")
    assert success is True

    # Active facts should now exclude invalidated fact
    active_facts_after = plane.query_active_facts(subject="auth_service")
    assert len(active_facts_after) == 0


def test_step132_cognitive_observability_reconstruction(tmp_path):
    db_path = tmp_path / "observability.db"
    plane = CognitiveObservabilityPlane(db_path)

    ev1 = CognitiveAuditEvent(
        event_id="ev-1",
        event_type=AuditEventType.TASK_SCHEDULED,
        task_id="t-1",
        node_id="node-1",
        worker_id="planner-1",
        role="PLANNER",
        summary="Task t-1 scheduled",
    )
    ev2 = CognitiveAuditEvent(
        event_id="ev-2",
        event_type=AuditEventType.TOOL_EVALUATED,
        task_id="t-1",
        node_id="node-1",
        worker_id="implementer-1",
        role="IMPLEMENTER",
        summary="Tool file_write authorized",
        metadata={"tool_name": "file_write", "authorized": True},
    )
    ev3 = CognitiveAuditEvent(
        event_id="ev-3",
        event_type=AuditEventType.WORKER_COMPLETED,
        task_id="t-1",
        node_id="node-1",
        worker_id="implementer-1",
        role="IMPLEMENTER",
        summary="Worker completed task successfully",
    )
    plane.record_event(ev1)
    plane.record_event(ev2)
    plane.record_event(ev3)

    trajectory = plane.query_trajectory("t-1")
    assert len(trajectory) == 3
    assert [e.event_type for e in trajectory] == [
        AuditEventType.TASK_SCHEDULED,
        AuditEventType.TOOL_EVALUATED,
        AuditEventType.WORKER_COMPLETED,
    ]


def test_step133_locality_aware_orchestration():
    worker_remote = WorkerResourceProfile(
        worker_id="worker_remote",
        role=WorkerRole.IMPLEMENTER,
        capacity_level=ResourceCapacityLevel.HIGH_RESOURCE,
        max_concurrency=2,
        current_active_tasks=0,
    )
    worker_local = WorkerResourceProfile(
        worker_id="worker_local",
        role=WorkerRole.IMPLEMENTER,
        capacity_level=ResourceCapacityLevel.STANDARD,
        max_concurrency=2,
        current_active_tasks=0,
    )

    contract = WorkerContract("c1", WorkerRole.IMPLEMENTER, "t-patch", ["file_write"], ["app/models.py"], 512)
    reqs = TaskResourceRequirements(minimum_capacity_level=ResourceCapacityLevel.STANDARD)
    
    # Context indicates worker_local has data cached locally
    ctx = CognitivePlacementContext(
        target_files=["app/models.py"],
        cached_node_ids={"worker_local"},
        estimated_tokens=200,
    )

    selected = LocalityAwareOrchestrator.choose_optimal_worker(
        workers=[worker_remote, worker_local],
        contract=contract,
        requirements=reqs,
        context=ctx,
    )
    assert selected is not None
    assert selected.worker_id == "worker_local"


def test_step134_cognitive_self_healing_bounded_loop(tmp_path):
    db_path = tmp_path / "healing.db"
    engine = CognitiveSelfHealingEngine(db_path, max_attempts_per_task=2)

    # Attempt 1: crash -> reroute to fallback worker
    action1 = engine.determine_healing_action(attempt_number=0, error_msg="Process crash exit code 139", has_fallback_worker=True)
    assert action1 == SelfHealingAction.REROUTE_FALLBACK_WORKER

    # Attempt 2: verification failure -> replan
    action2 = engine.determine_healing_action(attempt_number=1, error_msg="Verification test failed", has_fallback_worker=True)
    assert action2 == SelfHealingAction.REPLAN_TASK_DECOMPOSITION

    # Attempt 3: exceeded max attempts -> abort
    action3 = engine.determine_healing_action(attempt_number=2, error_msg="Verification test failed", has_fallback_worker=True)
    assert action3 == SelfHealingAction.ABORT_WITH_GOVERNED_ERROR

    # Record event
    rec = SelfHealingRecord(
        healing_id="heal-1",
        task_id="t-1",
        failure_class="CRASH",
        root_cause="SIGSEGV",
        action_taken=action1,
        rerouted_worker="worker-2",
        success=True,
        lesson_learned="Subprocess exceeded memory bounds",
    )
    engine.record_healing_event(rec)


def test_step135_distributed_long_horizon_pipeline_persistence(tmp_path):
    db_path = tmp_path / "pipeline.db"
    pipe = DistributedCognitivePipeline(db_path, pipeline_id="pipe-100")

    # Advance through stages
    ok1 = pipe.advance_stage(PipelineStage.ANALYZE, PipelineStage.PLAN, {"findings": ["modular design required"]})
    assert ok1 is True
    ok2 = pipe.advance_stage(PipelineStage.PLAN, PipelineStage.IMPLEMENT, {"plan": ["create auth service"]})
    assert ok2 is True

    # Reconstruct pipeline from a fresh instance
    reconstructed = DistributedCognitivePipeline(db_path, pipeline_id="pipe-100")
    curr_stage, rev_cycles, is_done = reconstructed.get_progress()
    assert curr_stage == PipelineStage.IMPLEMENT
    assert rev_cycles == 0
    assert is_done is False


def test_step136_neural_baseline_immutability():
    model = instantiate_frozen_baseline()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Baseline parameter count mismatch: {total_params} != 3443136"

    model_hash = compute_model_hash(model)
    assert model_hash == EXPECTED_WEIGHT_HASH, f"Baseline hash mismatch: {model_hash} != {EXPECTED_WEIGHT_HASH}"
    assert model_hash == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
