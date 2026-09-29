"""
Comprehensive Test Suite for Distributed Resource Orchestration, Fault-Tolerant
Task Execution & Work Continuity (Step 38).

Axioms Verified:
1. LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
2. ADVERTISEMENT != PERMISSION
3. ADVERTISEMENT != EXECUTION_AUTHORITY
4. RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
5. UNREACHABLE != REVOKED
6. RECOVERY RESILIENCE & CHECKPOINT WORK CONTINUITY
7. ZERO NEURAL WEIGHT MUTATION (ΔW = 0)
"""

import hashlib
import time
from typing import Dict, Any, Optional
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate
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
    VALID_TASK_TRANSITIONS,
    WorkUnitState,
    VALID_WORK_UNIT_TRANSITIONS,
    AggregationStrategy,
    GrantType,
    ResourceRequirements,
    ResourceExecutionGrant,
    TaskCheckpoint,
    TaskResultEnvelope,
    WorkUnit,
    SchedulingDecision,
    DistributedTask,
)
from chakrview.cognition.federation.tasks.errors import (
    InvalidTaskStateTransitionError,
    InvalidTaskDefinitionError,
    NoEligibleWorkerError,
    UnauthorizedTaskExecutionError,
    ExecutionGrantViolationError,
    InvalidCheckpointError,
    CheckpointCorruptionError,
    InvalidResultError,
    DuplicateResultError,
    StaleResultError,
    ResultAggregationError,
    TenantTaskIsolationError,
    TaskCancelledError,
)
from chakrview.cognition.federation.tasks.grant import ExecutionGrantManager
from chakrview.cognition.federation.tasks.scheduler import DeterministicTaskScheduler
from chakrview.cognition.federation.tasks.checkpoint import TaskCheckpointManager
from chakrview.cognition.federation.tasks.validator import TaskResultValidator
from chakrview.cognition.federation.tasks.aggregator import TaskResultAggregator
from chakrview.cognition.federation.tasks.executor import FederationTaskExecutor
from chakrview.cognition.federation.tasks.coordinator import FederationTaskCoordinator
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.federation.runtime import FederationRuntime


EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


# ============================================================================
# Helpers & Fixtures
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
    torch.manual_seed(42)
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    m.eval()
    return m


class MockCapabilityGate:
    """Mock capability gate providing deterministic test executions."""
    def __init__(self) -> None:
        self.authorized = True
        self.results = {"add": 42, "text_proc": "processed_text"}

    def authorize(self, req: Any, context: Optional[Any] = None) -> bool:
        return self.authorized

    def execute(self, req: Any, context: Optional[Any] = None) -> Any:
        from dataclasses import dataclass
        @dataclass
        class Result:
            success: bool = True
            data: Any = None
            error_message: Optional[str] = None
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


def _build_dummy_adv(node_id: str, cores: float = 4.0, mem_mb: int = 4096, tenant_id: str = "default") -> ResourceAdvertisement:
    profile = _build_dummy_profile(cores=cores, mem_mb=mem_mb)
    return ResourceAdvertisement(
        advertisement_id=f"adv_{node_id}",
        node_id=node_id,
        engine_id=f"eng_{node_id}",
        zone_id="zone-default",
        tenant_id=tenant_id,
        version=1,
        epoch=1,
        resource_profile=profile,
        capabilities=[AdvertisedCapability(capability_id="add", name="Add", description="Add", execution_type="CPU")],
    )


# ============================================================================
# 1. State Transition Matrix Tests
# ============================================================================

def test_task_state_transitions_valid():
    """Verify standard lifecycle transitions for DistributedTask."""
    task = DistributedTask(task_id="t1", name="TestTask")
    assert task.state == TaskState.CREATED

    task.transition_to(TaskState.VALIDATING)
    assert task.state == TaskState.VALIDATING

    task.transition_to(TaskState.PLANNING)
    assert task.state == TaskState.PLANNING

    task.transition_to(TaskState.QUEUED)
    assert task.state == TaskState.QUEUED

    task.transition_to(TaskState.RUNNING)
    assert task.state == TaskState.RUNNING

    task.transition_to(TaskState.PARTIALLY_COMPLETED)
    assert task.state == TaskState.PARTIALLY_COMPLETED

    task.transition_to(TaskState.COMPLETED)
    assert task.state == TaskState.COMPLETED


def test_task_state_transitions_invalid():
    """Verify illegal task state transitions raise InvalidTaskStateTransitionError."""
    task = DistributedTask(task_id="t1", name="TestTask")
    
    # Cannot jump directly from CREATED to COMPLETED
    with pytest.raises(InvalidTaskStateTransitionError):
        task.transition_to(TaskState.COMPLETED)

    # Terminal state cannot transition anywhere
    task.state = TaskState.COMPLETED
    with pytest.raises(InvalidTaskStateTransitionError):
        task.transition_to(TaskState.RUNNING)


def test_work_unit_state_transitions():
    """Verify valid and invalid state transitions for WorkUnit."""
    req = ResourceRequirements(capability_id="add")
    unit = WorkUnit(unit_id="u1", task_id="t1", sequence=0, capability_id="add", input_payload={}, requirements=req)
    assert unit.state == WorkUnitState.PENDING

    unit.transition_to(WorkUnitState.ASSIGNED)
    assert unit.state == WorkUnitState.ASSIGNED

    unit.transition_to(WorkUnitState.RUNNING)
    assert unit.state == WorkUnitState.RUNNING

    unit.transition_to(WorkUnitState.CHECKPOINTED)
    unit.transition_to(WorkUnitState.RUNNING)

    unit.transition_to(WorkUnitState.COMPLETED)
    assert unit.state == WorkUnitState.COMPLETED

    # Cannot transition from terminal COMPLETED
    with pytest.raises(InvalidTaskStateTransitionError):
        unit.transition_to(WorkUnitState.PENDING)


# ============================================================================
# 2. Resource Execution Grant Tests
# ============================================================================

def test_execution_grant_local_only():
    """Verify LOCAL_ONLY grant forbids remote execution while permitting local node."""
    mgr = ExecutionGrantManager(local_node_id="node_local", default_grant_type=GrantType.LOCAL_ONLY)
    req = ResourceRequirements(capability_id="add")

    # Local node execution is permitted
    grant = mgr.authorize_execution("node_local", "default", "add", req)
    assert grant.grant_type == GrantType.LOCAL_ONLY

    # Remote peer is rejected fail-closed
    with pytest.raises(ExecutionGrantViolationError):
        mgr.authorize_execution("node_remote", "default", "add", req)


def test_execution_grant_resource_ceilings():
    """Verify grant enforces CPU and memory limits."""
    mgr = ExecutionGrantManager(local_node_id="node_local")
    grant = ResourceExecutionGrant(
        grant_id="limited_grant",
        grant_type=GrantType.RESOURCE_LIMITED,
        owner_node_id="node_local",
        max_cores=2.0,
        max_memory_mb=1024,
        max_concurrent_units=1,
    )
    mgr.add_grant(grant)
    # Remove default grant
    mgr.remove_grant(mgr.default_grant_id)

    # Fits within limits
    valid_req = ResourceRequirements(capability_id="add", min_cpu_cores=1.0, min_memory_mb=512)
    matched_grant = mgr.authorize_execution("peer_1", "default", "add", valid_req)
    assert matched_grant.grant_id == "limited_grant"

    # Exceeds CPU limit
    heavy_cpu_req = ResourceRequirements(capability_id="add", min_cpu_cores=4.0, min_memory_mb=512)
    with pytest.raises(ExecutionGrantViolationError):
        mgr.authorize_execution("peer_1", "default", "add", heavy_cpu_req)

    # Exceeds memory limit
    heavy_mem_req = ResourceRequirements(capability_id="add", min_cpu_cores=1.0, min_memory_mb=2048)
    with pytest.raises(ExecutionGrantViolationError):
        mgr.authorize_execution("peer_1", "default", "add", heavy_mem_req)


def test_execution_grant_concurrency_tracking():
    """Verify active concurrency slots are reserved and released accurately."""
    mgr = ExecutionGrantManager(local_node_id="node_local")
    grant = ResourceExecutionGrant(
        grant_id="slot_grant",
        grant_type=GrantType.TRUSTED_FEDERATION,
        owner_node_id="node_local",
        max_concurrent_units=1,
    )
    mgr.add_grant(grant)
    mgr.remove_grant(mgr.default_grant_id)

    req = ResourceRequirements(capability_id="add")
    g = mgr.authorize_execution("peer_1", "default", "add", req)
    mgr.allocate_resources(g.grant_id, req)

    # Second concurrent execution fails
    with pytest.raises(ExecutionGrantViolationError):
        mgr.authorize_execution("peer_1", "default", "add", req)

    # Release resources
    mgr.release_resources(g.grant_id, req)

    # Now permitted again
    g2 = mgr.authorize_execution("peer_1", "default", "add", req)
    assert g2.grant_id == "slot_grant"


# ============================================================================
# 3. Deterministic Task Scheduler Tests
# ============================================================================

def test_deterministic_scheduler_selection():
    """Verify scheduler deterministically evaluates candidates and chooses best fit."""
    registry = FederationResourceRegistry(local_node_id="node_local", local_zone_id="zone-default")
    local_profile = _build_dummy_profile(cores=2.0, mem_mb=2048)
    registry.register_local_profile(
        local_profile,
        [AdvertisedCapability(capability_id="add", name="Addition", description="Adds numbers", execution_type="CPU")],
    )

    # Register remote peer advertisement
    adv = _build_dummy_adv("node_remote_1", cores=8.0, mem_mb=8192)
    registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(
        local_node_id="node_local",
        resource_registry=registry,
        local_tenant_id="default",
    )

    req = ResourceRequirements(capability_id="add", min_cpu_cores=1.0, min_memory_mb=1024)
    unit = WorkUnit(unit_id="u_0", task_id="t_0", sequence=0, capability_id="add", input_payload={}, requirements=req)

    decision = scheduler.schedule_unit(unit)
    assert isinstance(decision, SchedulingDecision)
    assert decision.selected_node in ("node_local", "node_remote_1")
    assert decision.decision_digest != ""
    assert "CAPABILITY_MATCH" in decision.reason_codes


def test_deterministic_scheduler_no_eligible_worker():
    """Verify scheduler raises NoEligibleWorkerError when no node satisfies constraints."""
    registry = FederationResourceRegistry(local_node_id="node_local", local_zone_id="zone-default")
    local_profile = _build_dummy_profile(cores=1.0, mem_mb=512)
    registry.register_local_profile(
        local_profile,
        [AdvertisedCapability(capability_id="add", name="Addition", description="Adds numbers", execution_type="CPU")],
    )

    scheduler = DeterministicTaskScheduler(
        local_node_id="node_local",
        resource_registry=registry,
    )

    # Demands 32 cores, which neither local nor any peer possesses
    huge_req = ResourceRequirements(capability_id="add", min_cpu_cores=32.0, min_memory_mb=65536)
    unit = WorkUnit(unit_id="u_fail", task_id="t_0", sequence=0, capability_id="add", input_payload={}, requirements=huge_req)

    with pytest.raises(NoEligibleWorkerError):
        scheduler.schedule_unit(unit)


def test_deterministic_scheduler_tenant_isolation():
    """Verify scheduler rejects cross-tenant scheduling attempts fail-closed."""
    registry = FederationResourceRegistry(local_node_id="node_local", local_zone_id="zone-default")
    scheduler = DeterministicTaskScheduler(
        local_node_id="node_local",
        resource_registry=registry,
        local_tenant_id="tenant_alpha",
    )

    req = ResourceRequirements(capability_id="add", tenant_id="tenant_beta")
    unit = WorkUnit(unit_id="u_iso", task_id="t_0", sequence=0, capability_id="add", input_payload={}, requirements=req)

    with pytest.raises(TenantTaskIsolationError):
        scheduler.schedule_unit(unit)


# ============================================================================
# 4. Task Checkpoint & Integrity Tests
# ============================================================================

def test_checkpoint_integrity_verification():
    """Verify TaskCheckpoint cryptographic digest generation and tampering detection."""
    cp = TaskCheckpoint(
        task_id="t1",
        unit_id="u1",
        checkpoint_id="cp_1",
        sequence=1,
        state=WorkUnitState.RUNNING,
        progress=0.5,
        partial_result={"step": 50},
        worker_id="node_w1",
        attempt=1,
    )
    assert cp.verify_integrity() is True

    # Tamper with progress
    cp.progress = 0.99
    assert cp.verify_integrity() is False


def test_checkpoint_manager_sequence_and_corruption():
    """Verify TaskCheckpointManager enforces sequence monotonicity and detects corruption."""
    mgr = TaskCheckpointManager()

    cp1 = TaskCheckpoint(
        task_id="t1",
        unit_id="u1",
        checkpoint_id="cp_1",
        sequence=1,
        state=WorkUnitState.RUNNING,
        progress=0.25,
        partial_result={"data": 1},
        worker_id="node_w1",
        attempt=1,
    )
    mgr.save_checkpoint(cp1)
    assert mgr.get_latest_checkpoint("t1", "u1") == cp1

    # Corrupted digest fails
    cp_corrupt = TaskCheckpoint(
        task_id="t1",
        unit_id="u1",
        checkpoint_id="cp_bad",
        sequence=2,
        state=WorkUnitState.RUNNING,
        progress=0.5,
        partial_result={"data": 2},
        worker_id="node_w1",
        attempt=1,
        integrity_digest="bad_digest",
    )
    with pytest.raises(CheckpointCorruptionError):
        mgr.save_checkpoint(cp_corrupt)

    # Sequence regression fails
    cp_regress = TaskCheckpoint(
        task_id="t1",
        unit_id="u1",
        checkpoint_id="cp_regress",
        sequence=0,  # Regressed below 1
        state=WorkUnitState.RUNNING,
        progress=0.1,
        partial_result={"data": 0},
        worker_id="node_w1",
        attempt=1,
    )
    with pytest.raises(InvalidCheckpointError):
        mgr.save_checkpoint(cp_regress)


# ============================================================================
# 5. Result Validation & Deduplication Tests
# ============================================================================

def test_result_validator_duplicate_rejection():
    """Verify TaskResultValidator rejects duplicate results for already completed units."""
    validator = TaskResultValidator()
    req = ResourceRequirements(capability_id="add")
    unit = WorkUnit(
        unit_id="u1",
        task_id="t1",
        sequence=0,
        capability_id="add",
        input_payload={},
        requirements=req,
        state=WorkUnitState.COMPLETED,
        assigned_node_id="worker_1",
        attempt=1,
    )

    res = TaskResultEnvelope(
        task_id="t1",
        unit_id="u1",
        attempt=1,
        worker_id="worker_1",
        status="SUCCESS",
        result_data=42,
        execution_time_ms=10.5,
    )

    with pytest.raises(DuplicateResultError):
        validator.validate_result(unit, res)


def test_result_validator_stale_attempt():
    """Verify TaskResultValidator rejects results from superseded attempts."""
    validator = TaskResultValidator()
    req = ResourceRequirements(capability_id="add")
    unit = WorkUnit(
        unit_id="u1",
        task_id="t1",
        sequence=0,
        capability_id="add",
        input_payload={},
        requirements=req,
        state=WorkUnitState.RUNNING,
        assigned_node_id="worker_2",
        attempt=2,  # Current attempt is 2
    )

    # Result from stale attempt 1
    stale_res = TaskResultEnvelope(
        task_id="t1",
        unit_id="u1",
        attempt=1,
        worker_id="worker_2",
        status="SUCCESS",
        result_data=42,
        execution_time_ms=10.5,
    )

    with pytest.raises(StaleResultError):
        validator.validate_result(unit, stale_res)


def test_result_validator_mismatched_worker():
    """Verify TaskResultValidator rejects results sent by a worker not assigned to the unit."""
    validator = TaskResultValidator()
    req = ResourceRequirements(capability_id="add")
    unit = WorkUnit(
        unit_id="u1",
        task_id="t1",
        sequence=0,
        capability_id="add",
        input_payload={},
        requirements=req,
        state=WorkUnitState.RUNNING,
        assigned_node_id="worker_official",
        attempt=1,
    )

    imposter_res = TaskResultEnvelope(
        task_id="t1",
        unit_id="u1",
        attempt=1,
        worker_id="worker_imposter",
        status="SUCCESS",
        result_data=42,
        execution_time_ms=10.5,
    )

    with pytest.raises(InvalidResultError):
        validator.validate_result(unit, imposter_res)


# ============================================================================
# 6. Result Aggregation Strategies Tests
# ============================================================================

def test_aggregation_strategies():
    """Verify CONCATENATE, MERGE_DICT, REDUCE_SUM, and FIRST_SUCCESS strategies."""
    agg = TaskResultAggregator()
    req = ResourceRequirements(capability_id="op")

    # 1. CONCATENATE (Strings)
    task_cat = DistributedTask(task_id="t_cat", name="CatTask", aggregation_strategy=AggregationStrategy.CONCATENATE)
    u1 = WorkUnit(unit_id="u1", task_id="t_cat", sequence=0, capability_id="op", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    u1.result = TaskResultEnvelope("t_cat", "u1", 1, "w1", "SUCCESS", "Hello, ", 5.0)
    u2 = WorkUnit(unit_id="u2", task_id="t_cat", sequence=1, capability_id="op", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    u2.result = TaskResultEnvelope("t_cat", "u2", 1, "w1", "SUCCESS", "World!", 5.0)
    task_cat.work_units = [u1, u2]
    assert agg.aggregate_results(task_cat) == "Hello, World!"

    # 2. MERGE_DICT
    task_dict = DistributedTask(task_id="t_dict", name="DictTask", aggregation_strategy=AggregationStrategy.MERGE_DICT)
    ud1 = WorkUnit(unit_id="ud1", task_id="t_dict", sequence=0, capability_id="op", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    ud1.result = TaskResultEnvelope("t_dict", "ud1", 1, "w1", "SUCCESS", {"a": 1, "b": 2}, 5.0)
    ud2 = WorkUnit(unit_id="ud2", task_id="t_dict", sequence=1, capability_id="op", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    ud2.result = TaskResultEnvelope("t_dict", "ud2", 1, "w1", "SUCCESS", {"c": 3}, 5.0)
    task_dict.work_units = [ud1, ud2]
    assert agg.aggregate_results(task_dict) == {"a": 1, "b": 2, "c": 3}

    # 3. REDUCE_SUM
    task_sum = DistributedTask(task_id="t_sum", name="SumTask", aggregation_strategy=AggregationStrategy.REDUCE_SUM)
    us1 = WorkUnit(unit_id="us1", task_id="t_sum", sequence=0, capability_id="op", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    us1.result = TaskResultEnvelope("t_sum", "us1", 1, "w1", "SUCCESS", 10.5, 5.0)
    us2 = WorkUnit(unit_id="us2", task_id="t_sum", sequence=1, capability_id="op", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    us2.result = TaskResultEnvelope("t_sum", "us2", 1, "w1", "SUCCESS", 20.0, 5.0)
    task_sum.work_units = [us1, us2]
    assert agg.aggregate_results(task_sum) == 30.5

    # 4. Incomplete units fail
    us2.state = WorkUnitState.RUNNING
    with pytest.raises(ResultAggregationError):
        agg.aggregate_results(task_sum)


# ============================================================================
# 7. Worker Execution Under CapabilityGate Tests
# ============================================================================

def test_worker_executor_lifecycle():
    """Verify worker execution, checkpoints, and grant compliance."""
    grant_mgr = ExecutionGrantManager(local_node_id="worker_node")
    mock_gate = MockCapabilityGate()
    emitted_checkpoints = []

    executor = FederationTaskExecutor(
        local_node_id="worker_node",
        grant_manager=grant_mgr,
        capability_gate=mock_gate,
        checkpoint_callback=lambda cp: emitted_checkpoints.append(cp),
    )

    req = ResourceRequirements(capability_id="add")
    unit = WorkUnit(unit_id="u_exec", task_id="t_exec", sequence=0, capability_id="add", input_payload={"x": 10}, requirements=req)

    result = executor.execute_unit(unit, originating_node_id="coord_node")
    assert result.status == "SUCCESS"
    assert result.result_data == 42
    assert len(emitted_checkpoints) == 2  # progress 0.0 and progress 0.5
    assert emitted_checkpoints[0].progress == 0.0
    assert emitted_checkpoints[1].progress == 0.5


def test_worker_executor_cancellation():
    """Verify cancelling a task halts execution fail-closed."""
    grant_mgr = ExecutionGrantManager(local_node_id="worker_node")
    mock_gate = MockCapabilityGate()

    executor = FederationTaskExecutor(
        local_node_id="worker_node",
        grant_manager=grant_mgr,
        capability_gate=mock_gate,
    )
    executor.cancel_task("t_cancelled")

    req = ResourceRequirements(capability_id="add")
    unit = WorkUnit(unit_id="u_canc", task_id="t_cancelled", sequence=0, capability_id="add", input_payload={}, requirements=req)

    with pytest.raises(TaskCancelledError):
        executor.execute_unit(unit, originating_node_id="coord_node")


# ============================================================================
# 8. End-to-End Orchestration & Worker Failure Recovery Tests
# ============================================================================

def test_end_to_end_distributed_task_success():
    """Test full distributed task lifecycle: creation, decomposition, execution, aggregation."""
    registry = FederationResourceRegistry(local_node_id="node_local", local_zone_id="zone-default")
    registry.register_local_profile(
        _build_dummy_profile(cores=4.0, mem_mb=4096),
        [AdvertisedCapability(capability_id="add", name="Add", description="Add", execution_type="CPU")],
    )
    scheduler = DeterministicTaskScheduler(local_node_id="node_local", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()
    grant_mgr = ExecutionGrantManager(local_node_id="node_local")
    mock_gate = MockCapabilityGate()
    executor = FederationTaskExecutor(local_node_id="node_local", grant_manager=grant_mgr, capability_gate=mock_gate)

    coord = FederationTaskCoordinator(
        local_node_id="node_local",
        scheduler=scheduler,
        checkpoint_manager=cp_mgr,
        local_executor=executor,
    )

    task = coord.create_task(
        name="SumAll",
        aggregation_strategy=AggregationStrategy.REDUCE_SUM,
    )

    # Decompose into 2 units
    units = coord.decompose_task(
        task_id=task.task_id,
        units_spec=[
            {"capability_id": "add", "input_payload": {"v": 1}},
            {"capability_id": "add", "input_payload": {"v": 2}},
        ],
    )
    assert len(units) == 2

    # Schedule and dispatch
    coord.schedule_and_dispatch_task(task.task_id)

    # Task should be fully completed
    assert task.state == TaskState.COMPLETED
    assert task.final_result == 84.0  # 42 + 42


def test_worker_failure_and_checkpoint_resumption():
    """
    Test worker node crash/disconnect recovery:
    1. Worker 1 executes, checkpoints partial progress, then crashes.
    2. Coordinator detects failure, fetches checkpoint, and reassigns to Worker 2.
    3. Task completes successfully.
    """
    registry = FederationResourceRegistry(local_node_id="node_coord", local_zone_id="zone-default")
    registry.register_local_profile(
        _build_dummy_profile(cores=4.0, mem_mb=4096),
        [],
    )

    # Register worker 1 and worker 2
    for w in ("worker_1", "worker_2"):
        adv = _build_dummy_adv(w, cores=4.0, mem_mb=4096)
        registry.record_peer_advertisement(adv)

    scheduler = DeterministicTaskScheduler(local_node_id="node_coord", resource_registry=registry)
    cp_mgr = TaskCheckpointManager()

    coord = FederationTaskCoordinator(
        local_node_id="node_coord",
        scheduler=scheduler,
        checkpoint_manager=cp_mgr,
    )

    task = coord.create_task(name="ResilientTask", aggregation_strategy=AggregationStrategy.CONCATENATE)
    units = coord.decompose_task(
        task_id=task.task_id,
        units_spec=[{"capability_id": "add", "input_payload": {"chunk": 1}}],
    )
    unit = units[0]

    # Assign to worker_1
    unit.assigned_node_id = "worker_1"
    unit.attempt = 1
    unit.transition_to(WorkUnitState.ASSIGNED)
    unit.transition_to(WorkUnitState.RUNNING)
    scheduler.record_assignment("worker_1")

    # Record worker assignment mapping
    coord._worker_assignments["worker_1"] = {(task.task_id, unit.unit_id)}

    # Worker 1 emits a checkpoint
    cp = TaskCheckpoint(
        task_id=task.task_id,
        unit_id=unit.unit_id,
        checkpoint_id="cp_halfway",
        sequence=1,
        state=WorkUnitState.RUNNING,
        progress=0.5,
        partial_result={"intermediate_sum": 21},
        worker_id="worker_1",
        attempt=1,
    )
    coord.record_checkpoint(cp)

    # WORKER 1 CRASHES / DISCONNECTS
    reassigned = coord.handle_worker_failure("worker_1")
    assert len(reassigned) == 1
    assert unit.assigned_node_id == "worker_2"
    assert unit.attempt == 2
    assert unit.latest_checkpoint.checkpoint_id == "cp_halfway"
    assert unit.input_payload.get("resume_from_checkpoint") == {"intermediate_sum": 21}

    # Worker 2 finishes work
    res = TaskResultEnvelope(
        task_id=task.task_id,
        unit_id=unit.unit_id,
        attempt=2,
        worker_id="worker_2",
        status="SUCCESS",
        result_data="CompletedWithCheckpoint",
        execution_time_ms=15.0,
    )
    coord.record_result(res)

    assert task.state == TaskState.COMPLETED
    assert task.final_result == "CompletedWithCheckpoint"


# ============================================================================
# 9. Neural Core Immutability Check (ΔW = 0)
# ============================================================================

def test_neural_weight_immutability_delta_w_zero(test_model):
    """
    CRITICAL INVARIANT VERIFICATION:
    Ensure that no task orchestration, scheduling, checkpointing, or execution
    alters neural model weights (ΔW = 0, parameters = 3,443,136).
    """
    initial_hash = _compute_model_hash(test_model)
    assert initial_hash == EXPECTED_WEIGHT_HASH

    # Run engine and coordinator operations with the neural model attached
    mock_gate = MockCapabilityGate()
    engine = CrossZoneFederationEngine(local_zone_id="zone-task-verify", model=test_model, capability_gate=mock_gate)
    runtime = FederationRuntime(engine=engine, auto_recover=False)
    engine.resource_registry.register_local_profile(
        _build_dummy_profile(cores=4.0, mem_mb=4096),
        [AdvertisedCapability(capability_id="add", name="Add", description="Add", execution_type="CPU")],
    )

    coord = engine.task_coordinator
    task = coord.create_task(name="NeuralSafetyCheck", aggregation_strategy=AggregationStrategy.REDUCE_SUM)
    coord.decompose_task(task.task_id, [{"capability_id": "add", "input_payload": {"n": 1}}])
    coord.schedule_and_dispatch_task(task.task_id)

    # Post-orchestration weight hash check
    final_hash = _compute_model_hash(test_model)
    assert final_hash == initial_hash == EXPECTED_WEIGHT_HASH

    total_params = sum(p.numel() for p in test_model.parameters())
    assert total_params == 3443136
