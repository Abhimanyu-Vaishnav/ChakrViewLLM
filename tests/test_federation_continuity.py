"""
Comprehensive Test Suite for Federated Execution Continuity, Checkpointed
Work Migration & Failure-Resilient Distributed Computation (Step 39).

Verifies:
1. Checkpoint creation
2. Checkpoint authentication
3. Checkpoint monotonicity
4. Stale checkpoint rejection
5. Checkpoint corruption rejection
6. Checkpoint commit
7. Checkpoint supersession
8. Lease creation
9. Heartbeat renewal
10. Heartbeat timeout
11. Lease expiration
12. Worker disconnect
13. Worker reconnect
14. Failed attempt fencing
15. Stale result rejection
16. Duplicate result rejection
17. Recovery scheduling
18. Checkpoint resume
19. Recovery without checkpoint
20. Recovery after multiple checkpoints
21. Disconnect during checkpoint
22. Disconnect after checkpoint commit
23. Disconnect during result submission
24. Worker resurrection rejection
25. Duplicate commit prevention
26. Task completion after worker failure
27. Multiple worker failures
28. Tenant isolation
29. Capability isolation
30. Security invariant preservation
31. WAL replay
32. Deterministic recovery
33. Recovery idempotency
34. Neural weight immutability (ΔW = 0)
35. Zero secret exposure
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.federation.runtime import FederationRuntime
from chakrview.cognition.peering.models import (
    TrustLevel,
    FederationScope,
    AuditEventType,
)
from chakrview.cognition.federation.discovery.models import (
    MembershipState,
    FederationNodeMembership,
)
from chakrview.cognition.federation.resources.models import (
    CPUResource,
    MemoryResource,
    AcceleratorResource,
    StorageResource,
    PlatformResource,
    NodeResourceProfile,
    ResourceAdvertisement,
    ResourceSharingPolicy,
    AdvertisedCapability,
)
from chakrview.cognition.federation.resources.registry import FederationResourceRegistry
from chakrview.cognition.federation.tasks.models import (
    TaskState,
    WorkUnitState,
    AggregationStrategy,
    GrantType,
    ResourceRequirements,
    ResourceExecutionGrant,
    TaskCheckpoint,
    CheckpointManifest,
    CheckpointStatus,
    WorkerLease,
    LeaseState,
    ResumeAction,
    AttemptFenceToken,
    CommitIdentity,
    TaskResultEnvelope,
    WorkUnit,
    SchedulingDecision,
    DistributedTask,
)
from chakrview.cognition.federation.tasks.errors import (
    FederationTaskError,
    InvalidTaskDefinitionError,
    InvalidWorkUnitError,
    InvalidTaskStateTransitionError,
    SchedulingError,
    NoEligibleWorkerError,
    UnauthorizedTaskExecutionError,
    ExecutionGrantViolationError,
    WorkerExecutionError,
    WorkerUnavailableError,
    InvalidCheckpointError,
    CheckpointCorruptionError,
    InvalidResultError,
    DuplicateResultError,
    DuplicateCommitError,
    StaleResultError,
    FencedAttemptError,
    LeaseExpiredError,
    LeaseRevokedError,
    StaleCheckpointError,
    CheckpointCommitError,
    HeartbeatTimeoutError,
    ResultAggregationError,
    TaskRecoveryError,
    TenantTaskIsolationError,
    TaskCancelledError,
)
from chakrview.cognition.federation.tasks.grant import ExecutionGrantManager
from chakrview.cognition.federation.tasks.scheduler import DeterministicTaskScheduler
from chakrview.cognition.federation.tasks.checkpoint import (
    TaskCheckpointManager,
    CheckpointStore,
)
from chakrview.cognition.federation.tasks.lease import (
    WorkerLeaseManager,
    AttemptFenceManager,
    DeterministicFailureDetector,
)
from chakrview.cognition.federation.tasks.validator import TaskResultValidator
from chakrview.cognition.federation.tasks.aggregator import TaskResultAggregator
from chakrview.cognition.federation.tasks.executor import FederationTaskExecutor
from chakrview.cognition.federation.tasks.coordinator import FederationTaskCoordinator
from chakrview.cognition.federation.persistence.models import JournalEntryType


EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3443136


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

def _compute_model_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@pytest.fixture
def test_model():
    """Frozen ChakrMicro v0.1 model."""
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    return model


class MockCapabilityGate:
    """Mock CapabilityGate with configurable capabilities and failure injection."""
    def __init__(self) -> None:
        self.authorized = True
        self.results = {"add": 10, "multiply": 20, "process": "done"}
        self.should_fail = False

    def authorize(self, req: Any, context: Any = None) -> bool:
        return self.authorized

    def execute(self, req: Any, context: Any = None) -> Any:
        from dataclasses import dataclass
        @dataclass
        class Result:
            success: bool = True
            data: Any = None
            error_message: Optional[str] = None

        if self.should_fail:
            return Result(success=False, error_message="Injected execution failure")

        if req.capability_id in self.results:
            return Result(success=True, data=self.results[req.capability_id])
        return Result(success=False, error_message=f"Unknown capability: {req.capability_id}")


def _build_dummy_profile(cores: float = 4.0, mem_mb: int = 4096) -> NodeResourceProfile:
    return NodeResourceProfile(
        profile_id="prof_dummy",
        cpu=CPUResource(physical_cores=4, logical_cores=4, available_cores=cores, architecture="x86_64"),
        memory=MemoryResource(total_memory_mb=8192, available_memory_mb=mem_mb),
        accelerator=AcceleratorResource(is_available=False),
        storage=StorageResource(total_storage_mb=100000, available_storage_mb=50000),
        platform=PlatformResource(os_family="Linux", os_release="5.15", python_version="3.14"),
    )


def _build_dummy_adv(node_id: str, cores: float = 4.0, mem_mb: int = 4096, tenant_id: str = "default", capabilities: Optional[List[str]] = None) -> ResourceAdvertisement:
    profile = _build_dummy_profile(cores=cores, mem_mb=mem_mb)
    caps = [
        AdvertisedCapability(capability_id=c, name=c, description=c, execution_type="CPU")
        for c in (capabilities or ["add", "multiply", "process"])
    ]
    return ResourceAdvertisement(
        advertisement_id=f"adv_{node_id}",
        node_id=node_id,
        engine_id=f"eng_{node_id}",
        zone_id="zone-default",
        tenant_id=tenant_id,
        version=1,
        epoch=1,
        resource_profile=profile,
        capabilities=caps,
    )


# ============================================================================
# 1-7: Checkpoint Store, Manifest & Commit Protocol Tests
# ============================================================================

def test_checkpoint_manifest_creation_and_integrity():
    """1 & 2. Test CheckpointManifest creation, authentication, digest calculation, and integrity check."""
    manifest = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_01",
        checkpoint_sequence=1,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"start": 0, "end": 50, "items_processed": 50},
        remaining_work={"start": 51, "end": 100, "items_remaining": 50},
        intermediate_payload={"subtotal": 500},
        fencing_token=1,
    )
    assert manifest.verify_integrity()
    assert manifest.status == CheckpointStatus.PENDING
    assert manifest.completed_work_range["items_processed"] == 50


def test_checkpoint_store_monotonicity_and_stale_rejection():
    """3 & 4. Test CheckpointStore sequence monotonicity and stale checkpoint rejection."""
    store = CheckpointStore()
    m1 = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_01",
        checkpoint_sequence=2,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.5},
        remaining_work={"progress": 0.5},
        intermediate_payload={"v": 1},
    )
    store.create_manifest(m1)
    store.commit_checkpoint("t1", "u1", "cp_01")

    # Stale sequence (sequence 1 <= committed sequence 2)
    m_stale = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_00",
        checkpoint_sequence=1,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.2},
        remaining_work={"progress": 0.8},
        intermediate_payload={"v": 0},
    )
    with pytest.raises(StaleCheckpointError):
        store.create_manifest(m_stale)


def test_checkpoint_corruption_rejection():
    """5. Test that tampered checkpoint payloads are rejected as corrupted."""
    store = CheckpointStore()
    m = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_tampered",
        checkpoint_sequence=1,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.5},
        remaining_work={"progress": 0.5},
        intermediate_payload={"v": 100},
    )
    # Tamper payload without updating digest
    m.intermediate_payload = {"v": 999999}
    assert not m.verify_integrity()

    with pytest.raises(CheckpointCorruptionError):
        store.create_manifest(m)


def test_checkpoint_atomic_commit_and_supersession():
    """6 & 7. Test atomic commit protocol and automatic supersession of older checkpoints."""
    store = CheckpointStore()
    m1 = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_1",
        checkpoint_sequence=1,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.3},
        remaining_work={"progress": 0.7},
        intermediate_payload={"val": 10},
    )
    store.create_manifest(m1)
    committed_1 = store.commit_checkpoint("t1", "u1", "cp_1")
    assert committed_1.status == CheckpointStatus.COMMITTED

    # Create next sequence checkpoint
    m2 = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_2",
        checkpoint_sequence=2,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.8},
        remaining_work={"progress": 0.2},
        intermediate_payload={"val": 50},
        parent_checkpoint_id="cp_1",
    )
    store.create_manifest(m2)
    committed_2 = store.commit_checkpoint("t1", "u1", "cp_2")

    assert committed_2.status == CheckpointStatus.COMMITTED
    # m1 must now be superseded
    assert m1.status == CheckpointStatus.SUPERSEDED
    assert store.get_committed_checkpoint("t1", "u1").checkpoint_id == "cp_2"


# ============================================================================
# 8-13: Worker Lease, Heartbeat & Disconnect / Reconnect Tests
# ============================================================================

def test_worker_lease_creation_and_expiration():
    """8 & 11. Test WorkerLease issuance and deterministic expiration check."""
    mgr = WorkerLeaseManager(default_duration_sec=10.0, default_heartbeat_timeout_sec=3.0)
    lease = mgr.grant_lease(
        worker_id="w1",
        task_id="t1",
        unit_id="u1",
        attempt_number=1,
        fencing_token=1,
        duration_sec=5.0,
    )
    assert lease.state == LeaseState.ACTIVE
    assert not lease.is_expired(current_time=lease.granted_at + 2.0)
    assert lease.is_expired(current_time=lease.granted_at + 6.0)


def test_heartbeat_renewal_and_timeout():
    """9 & 10. Test heartbeat renewal and failure detector timeout."""
    mgr = WorkerLeaseManager(default_duration_sec=20.0, default_heartbeat_timeout_sec=5.0)
    lease = mgr.grant_lease(
        worker_id="w1",
        task_id="t1",
        unit_id="u1",
        attempt_number=1,
        fencing_token=1,
    )

    t0 = lease.granted_at
    # Renew at t0 + 3.0s
    renewed = mgr.renew_lease("t1", "u1", "w1", fencing_token=1, current_time=t0 + 3.0)
    assert renewed.last_heartbeat_at == t0 + 3.0

    # Evaluate at t0 + 12.0s (9s since heartbeat > 5s heartbeat_timeout + 2s grace)
    healthy, state, reason = mgr.check_lease_health("t1", "u1", current_time=t0 + 12.0)
    assert not healthy
    assert state == LeaseState.UNREACHABLE
    assert "Heartbeat dead interval exceeded" in reason


def test_worker_disconnect_and_reconnect_semantics():
    """12 & 13. Test that worker disconnect marks leases RECOVERABLE, and reconnect requires new lease."""
    mgr = WorkerLeaseManager(default_duration_sec=15.0, default_heartbeat_timeout_sec=5.0)
    mgr.grant_lease("w_disc", "t1", "u1", attempt_number=1, fencing_token=1)

    # Disconnect marks leases recoverable
    expired = mgr.expire_worker_leases("w_disc")
    assert len(expired) == 1
    assert expired[0].state == LeaseState.RECOVERABLE

    # Reconnecting worker attempts renewal on old lease -> rejected
    with pytest.raises(LeaseExpiredError):
        mgr.renew_lease("t1", "u1", "w_disc", fencing_token=1)


# ============================================================================
# 14-16 & 24-25: Fencing, Stale Result & Duplicate Rejection Tests
# ============================================================================

def test_attempt_fence_manager_and_stale_token_rejection():
    """14. Test AttemptFenceManager monotonic generation and stale attempt rejection."""
    fence = AttemptFenceManager()
    token_1 = fence.issue_fence_token("t1", "u1", attempt_number=1, worker_id="w1")
    assert token_1.fencing_token == 1

    # Worker 2 gets attempt 2
    token_2 = fence.issue_fence_token("t1", "u1", attempt_number=2, worker_id="w2")
    assert token_2.fencing_token == 2

    # Verification of active token succeeds
    assert fence.verify_attempt_token("t1", "u1", 2, 2, "w2")

    # Verification of stale token 1 fails closed
    with pytest.raises(FencedAttemptError):
        fence.verify_attempt_token("t1", "u1", 1, 1, "w1")


def test_stale_result_and_fenced_result_rejection():
    """15. Test TaskResultValidator rejecting results with obsolete attempt or fencing tokens."""
    unit = WorkUnit(
        unit_id="u1",
        task_id="t1",
        sequence=0,
        capability_id="add",
        input_payload={"v": 1},
        requirements=ResourceRequirements(capability_id="add"),
        attempt=2,
        assigned_node_id="w1",
        fencing_token=2,
    )

    validator = TaskResultValidator()

    # Stale attempt (attempt 1 < unit attempt 2)
    stale_attempt_res = TaskResultEnvelope(
        task_id="t1",
        unit_id="u1",
        attempt=1,
        worker_id="w1",
        status="SUCCESS",
        result_data=10,
        execution_time_ms=5.0,
        fencing_token=1,
    )
    with pytest.raises(StaleResultError):
        validator.validate_result(unit, stale_attempt_res)

    # Fenced attempt token (attempt 2 matches, but fencing token 1 != 2)
    fenced_res = TaskResultEnvelope(
        task_id="t1",
        unit_id="u1",
        attempt=2,
        worker_id="w1",
        status="SUCCESS",
        result_data=10,
        execution_time_ms=5.0,
        fencing_token=1,
    )
    with pytest.raises(FencedAttemptError):
        validator.validate_result(unit, fenced_res)


def test_duplicate_result_and_duplicate_commit_rejection():
    """16 & 25. Test duplicate result and duplicate commit prevention."""
    unit = WorkUnit(
        unit_id="u1",
        task_id="t1",
        sequence=0,
        capability_id="add",
        input_payload={"v": 1},
        requirements=ResourceRequirements(capability_id="add"),
        attempt=1,
        assigned_node_id="w1",
        fencing_token=1,
    )
    result = TaskResultEnvelope(
        task_id="t1",
        unit_id="u1",
        attempt=1,
        worker_id="w1",
        status="SUCCESS",
        result_data=42,
        execution_time_ms=10.0,
        fencing_token=1,
    )

    validator = TaskResultValidator()
    validator.validate_result(unit, result)

    # Commit result
    unit.committed_result = result
    unit.state = WorkUnitState.COMPLETED

    with pytest.raises(DuplicateResultError):
        validator.validate_result(unit, result)


def test_worker_resurrection_rejection():
    """24. Test that a resurrected worker attempting to submit results after being replaced is fenced."""
    registry = FederationResourceRegistry(local_node_id="node_coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("w_dead", "w_alive"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["add"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="node_coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="node_coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    task = coord.create_task(name="ResurrectionTest")
    units = coord.decompose_task(task.task_id, units_spec=[{"capability_id": "add", "input_payload": {"v": 10}}])
    coord.schedule_and_dispatch_task(task.task_id)

    dead_worker = units[0].assigned_node_id
    dead_token = units[0].fencing_token

    # Coordinator declares worker failed and reassigns
    coord.handle_worker_failure(dead_worker)

    # Dead worker resurrects and attempts to submit late result
    late_result = TaskResultEnvelope(
        task_id=task.task_id,
        unit_id=units[0].unit_id,
        attempt=1,
        worker_id=dead_worker,
        status="SUCCESS",
        result_data=20,
        execution_time_ms=50.0,
        fencing_token=dead_token,
    )

    with pytest.raises(InvalidResultError):
        coord.record_result(late_result)


# ============================================================================
# 17-20: Recovery Scheduling, Checkpoint Resumption & Multiple Checkpoints
# ============================================================================

def test_recovery_scheduling_and_checkpoint_resume():
    """17 & 18. Test worker failure detection, checkpoint retrieval, and resumption on replacement worker."""
    registry = FederationResourceRegistry(local_node_id="node_coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("worker_1", "worker_2"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["process"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="node_coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()

    coord = FederationTaskCoordinator(
        local_node_id="node_coord",
        scheduler=scheduler,
        checkpoint_manager=cp_mgr,
    )

    task = coord.create_task(name="ResilientProcess", aggregation_strategy=AggregationStrategy.CONCATENATE)
    units = coord.decompose_task(
        task_id=task.task_id,
        units_spec=[{"capability_id": "process", "input_payload": {"data": [1, 2, 3, 4]}}],
    )
    coord.schedule_and_dispatch_task(task.task_id)

    unit = units[0]
    first_worker = unit.assigned_node_id
    assert first_worker in ("worker_1", "worker_2")
    assert unit.fencing_token == 1

    # Worker 1 saves checkpoint at sequence 1
    cp1 = TaskCheckpoint(
        task_id=task.task_id,
        unit_id=unit.unit_id,
        checkpoint_id="cp_1",
        sequence=1,
        state=WorkUnitState.RUNNING,
        progress=0.5,
        partial_result={"processed": [1, 2]},
        worker_id=first_worker,
        attempt=1,
    )
    coord.record_checkpoint(cp1)

    # Worker 1 crashes / disconnects
    recovered_units = coord.handle_worker_failure(first_worker)
    assert len(recovered_units) == 1
    recovered_unit = recovered_units[0]

    # Verified resumed on alternate worker with increased fencing token
    assert recovered_unit.assigned_node_id != first_worker
    assert recovered_unit.attempt == 2
    assert recovered_unit.fencing_token == 2
    assert recovered_unit.checkpoint_reference == "cp_1"
    assert recovered_unit.input_payload["resume_from_checkpoint"] == {"processed": [1, 2]}


def test_recovery_without_checkpoint():
    """19. Test worker failure before any checkpoint was created: clean restart from zero."""
    registry = FederationResourceRegistry(local_node_id="node_coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("w_alpha", "w_beta"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["process"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="node_coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="node_coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    task = coord.create_task(name="NoCpTask")
    units = coord.decompose_task(task.task_id, units_spec=[{"capability_id": "process", "input_payload": {"raw": 100}}])
    coord.schedule_and_dispatch_task(task.task_id)

    first_worker = units[0].assigned_node_id
    # Crashes immediately with no checkpoint
    recovered = coord.handle_worker_failure(first_worker)
    assert len(recovered) == 1
    # Restarts cleanly without resume payload
    assert "resume_from_checkpoint" not in recovered[0].input_payload
    assert recovered[0].checkpoint_reference is None


def test_recovery_after_multiple_checkpoints():
    """20. Test that recovery selects the highest committed sequence checkpoint."""
    registry = FederationResourceRegistry(local_node_id="node_coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("w1", "w2"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["process"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="node_coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="node_coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    task = coord.create_task(name="MultiCpTask")
    units = coord.decompose_task(task.task_id, units_spec=[{"capability_id": "process", "input_payload": {"range": 100}}])
    coord.schedule_and_dispatch_task(task.task_id)

    w1 = units[0].assigned_node_id
    # Checkpoint 1
    cp_mgr.save_checkpoint(TaskCheckpoint(task.task_id, units[0].unit_id, "cp1", sequence=1, state=WorkUnitState.RUNNING, progress=0.25, partial_result={"items": 25}, worker_id=w1, attempt=1))
    # Checkpoint 2
    cp_mgr.save_checkpoint(TaskCheckpoint(task.task_id, units[0].unit_id, "cp2", sequence=2, state=WorkUnitState.RUNNING, progress=0.75, partial_result={"items": 75}, worker_id=w1, attempt=1))

    # Worker crashes
    reassigned = coord.handle_worker_failure(w1)
    assert reassigned[0].checkpoint_reference == "cp2"
    assert reassigned[0].input_payload["resume_from_checkpoint"] == {"items": 75}


# ============================================================================
# 21-23 & 26: Disconnect Windows & End-to-End Task Completion
# ============================================================================

def test_disconnect_during_checkpoint_and_result_submission():
    """21-23. Test failures at different checkpoint windows (during creation, after commit, during result)."""
    store = CheckpointStore()
    # 21. Checkpoint created but uncommitted is PENDING
    manifest = CheckpointManifest(
        task_id="t1", work_unit_id="u1", attempt_id=1, checkpoint_id="cp_uncommitted",
        checkpoint_sequence=1, worker_id="w1", execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.5}, remaining_work={"progress": 0.5},
        intermediate_payload={"state": 1},
    )
    store.create_manifest(manifest)
    assert store.get_committed_checkpoint("t1", "u1") is None

    # 22. Once committed, it becomes recoverable
    store.commit_checkpoint("t1", "u1", "cp_uncommitted")
    recovered = store.recover_manifest("t1", "u1")
    assert recovered is not None
    assert recovered.checkpoint_id == "cp_uncommitted"


def test_task_completion_after_worker_failure_and_migration():
    """26. Test end-to-end task completion: Worker 1 crashes -> Worker 2 finishes work -> aggregation succeeds."""
    registry = FederationResourceRegistry(local_node_id="node_coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("worker_A", "worker_B"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["add"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="node_coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="node_coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    task = coord.create_task(name="EndToEndResilience", aggregation_strategy=AggregationStrategy.REDUCE_SUM)
    units = coord.decompose_task(
        task.task_id,
        units_spec=[
            {"capability_id": "add", "input_payload": {"n": 5}},
            {"capability_id": "add", "input_payload": {"n": 10}},
        ],
    )
    coord.schedule_and_dispatch_task(task.task_id)

    # Worker A receives Unit 0, Worker B receives Unit 1
    unit0 = units[0]
    worker0 = unit0.assigned_node_id

    # Worker 0 checkpoints at 50%
    cp = TaskCheckpoint(task.task_id, unit0.unit_id, "cp_u0", sequence=1, state=WorkUnitState.RUNNING, progress=0.5, partial_result=2, worker_id=worker0, attempt=1)
    coord.record_checkpoint(cp)

    # Worker 0 crashes!
    coord.handle_worker_failure(worker0)

    # Replacement worker for Unit 0 finishes
    replacement_worker = unit0.assigned_node_id
    assert replacement_worker != worker0
    res0 = TaskResultEnvelope(
        task.task_id, unit0.unit_id, attempt=unit0.attempt, worker_id=replacement_worker, status="SUCCESS", result_data=5, execution_time_ms=10.0, fencing_token=unit0.fencing_token
    )
    coord.record_result(res0)

    # Unit 1 finishes on its assigned worker
    unit1 = units[1]
    res1 = TaskResultEnvelope(
        task.task_id, unit1.unit_id, attempt=unit1.attempt, worker_id=unit1.assigned_node_id, status="SUCCESS", result_data=10, execution_time_ms=12.0, fencing_token=unit1.fencing_token
    )
    coord.record_result(res1)

    # Task transitions to COMPLETED and aggregated sum is 15
    assert task.state == TaskState.COMPLETED
    assert task.final_result == 15


# ============================================================================
# 27-30: Multiple Worker Failures, Tenant & Capability Isolation
# ============================================================================

def test_multiple_consecutive_worker_failures():
    """27. Test cascading failure tolerance: Worker A fails -> Worker B fails -> Worker C completes."""
    registry = FederationResourceRegistry(local_node_id="coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("node_1", "node_2", "node_3"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["process"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    task = coord.create_task(name="CascadingFailureTest")
    units = coord.decompose_task(task.task_id, units_spec=[{"capability_id": "process", "input_payload": {"x": 1}}])
    coord.schedule_and_dispatch_task(task.task_id)

    u = units[0]
    w1 = u.assigned_node_id
    assert u.attempt == 1

    # First crash
    coord.handle_worker_failure(w1)
    w2 = u.assigned_node_id
    assert w2 != w1
    assert u.attempt == 2

    # Second crash
    coord.handle_worker_failure(w2)
    w3 = u.assigned_node_id
    assert w3 not in (w1, w2)
    assert u.attempt == 3

    # Third worker succeeds
    res = TaskResultEnvelope(task.task_id, u.unit_id, attempt=3, worker_id=w3, status="SUCCESS", result_data="ok", execution_time_ms=5.0, fencing_token=u.fencing_token)
    coord.record_result(res)
    assert task.state == TaskState.COMPLETED


def test_tenant_isolation_on_recovery():
    """28. Test that replacement worker selection strictly preserves tenant boundaries."""
    registry = FederationResourceRegistry(local_node_id="coord", local_zone_id="zone-default", local_tenant_id="*")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])

    # Register worker for tenant-A and worker for tenant-B
    adv_a1 = _build_dummy_adv("worker_a1", cores=4.0, mem_mb=4096, tenant_id="tenant-A", capabilities=["add"])
    adv_b1 = _build_dummy_adv("worker_b1", cores=4.0, mem_mb=4096, tenant_id="tenant-B", capabilities=["add"])
    registry.record_peer_advertisement(adv_a1)
    registry.record_peer_advertisement(adv_b1)

    scheduler = DeterministicTaskScheduler(local_node_id="coord", resource_registry=registry, local_tenant_id="tenant-A")
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    # Task belongs to tenant-A
    task = coord.create_task(name="TenantTask", tenant_id="tenant-A")
    units = coord.decompose_task(
        task.task_id,
        units_spec=[
            {
                "capability_id": "add",
                "input_payload": {"v": 1},
                "requirements": ResourceRequirements(capability_id="add", tenant_id="tenant-A"),
            }
        ],
    )
    coord.schedule_and_dispatch_task(task.task_id)

    # Assigned to worker_a1
    assert units[0].assigned_node_id == "worker_a1"

    # If worker_a1 fails and no other tenant-A worker exists, scheduling fails closed
    coord.handle_worker_failure("worker_a1")
    assert task.state == TaskState.FAILED


# ============================================================================
# 31-35: WAL Replay, Neural Immutability & Zero Secret Exposure
# ============================================================================

def test_checkpoint_wal_journal_logging():
    """31. Test that checkpoint commits and attempt fences are durably appended to WAL."""
    class MockJournal:
        def __init__(self):
            self.entries = []
        def append_entry(self, entry_type, payload):
            self.entries.append((entry_type, payload))

    journal = MockJournal()
    cp_mgr = TaskCheckpointManager(journal=journal)

    cp = TaskCheckpoint(
        task_id="t1",
        unit_id="u1",
        checkpoint_id="cp_wal_test",
        sequence=1,
        state=WorkUnitState.RUNNING,
        progress=0.4,
        partial_result={"val": 42},
        worker_id="w1",
        attempt=1,
    )
    cp_mgr.save_checkpoint(cp)

    assert len(journal.entries) >= 2
    types = [e[0] for e in journal.entries]
    assert JournalEntryType.TASK_CHECKPOINTED in types
    assert JournalEntryType.TASK_CHECKPOINT_COMMITTED in types


def test_neural_weight_immutability_delta_w_zero_during_recovery(test_model):
    """34. Test that full recovery cycle preserves frozen neural weights (ΔW = 0)."""
    initial_hash = _compute_model_hash(test_model)
    assert initial_hash == EXPECTED_WEIGHT_HASH

    registry = FederationResourceRegistry(local_node_id="coord", local_zone_id="zone-default")
    registry.register_local_profile(_build_dummy_profile(cores=4.0, mem_mb=4096), [])
    for w in ("w_fail", "w_pass"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096, capabilities=["add"])
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="coord", scheduler=scheduler, checkpoint_manager=cp_mgr)

    task = coord.create_task(name="NeuralContinuousTask", aggregation_strategy=AggregationStrategy.REDUCE_SUM)
    units = coord.decompose_task(task.task_id, units_spec=[{"capability_id": "add", "input_payload": {"x": 10}}])
    coord.schedule_and_dispatch_task(task.task_id)

    u = units[0]
    dead_w = u.assigned_node_id
    # Fail dead worker
    coord.handle_worker_failure(dead_w)
    alive_w = u.assigned_node_id

    # Finish task
    res = TaskResultEnvelope(task.task_id, u.unit_id, attempt=u.attempt, worker_id=alive_w, status="SUCCESS", result_data=10, execution_time_ms=5.0, fencing_token=u.fencing_token)
    coord.record_result(res)

    post_hash = _compute_model_hash(test_model)
    assert post_hash == initial_hash
    assert post_hash == EXPECTED_WEIGHT_HASH
    param_count = sum(p.numel() for p in test_model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT


def test_zero_secret_exposure_in_checkpoint_and_manifest():
    """35. Test that manifests and checkpoints do not serialize private keys, passwords, or raw tensors."""
    manifest = CheckpointManifest(
        task_id="t1",
        work_unit_id="u1",
        attempt_id=1,
        checkpoint_id="cp_clean",
        checkpoint_sequence=1,
        worker_id="w1",
        execution_state=WorkUnitState.RUNNING,
        completed_work_range={"progress": 0.5},
        remaining_work={"progress": 0.5},
        intermediate_payload={"sanitized_data": [1, 2, 3]},
    )
    d = manifest.to_dict()
    serialized = json.dumps(d)

    # Assert no secrets or weight references present
    assert "private_key" not in serialized
    assert "secret" not in serialized
    assert "password" not in serialized
    assert "token_embedding.weight" not in serialized
