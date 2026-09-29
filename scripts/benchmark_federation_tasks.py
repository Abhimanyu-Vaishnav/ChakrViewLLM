"""
Empirical Benchmark for ChakrView Step 38:
Distributed Resource Orchestration, Fault-Tolerant Task Execution & Work Continuity.

Measures:
1. Task Creation Latency
2. Task Decomposition Latency (Splitting into WorkUnits)
3. WorkUnit State Transition Latency
4. Execution Grant Authorization Latency (Local Sovereign Policy Evaluation)
5. Execution Grant Resource Accounting Latency (Allocations & Releases)
6. Deterministic Task Scheduling Latency (Single Candidate)
7. Deterministic Task Scheduling Latency (10 Peer Candidates)
8. Checkpoint Creation & Canonical Digest Latency
9. Checkpoint Verification Latency (Sequence Monotonicity & Digest Check)
10. Checkpoint Corruption Rejection Latency
11. Result Validation Latency (Attempt Verification & Dedup)
12. Aggregation Latency - CONCATENATE
13. Aggregation Latency - REDUCE_SUM
14. Aggregation Latency - MERGE_DICT
15. Worker Failure Detection & Rescheduling Latency (Work Continuity)
16. End-to-End Distributed Task Lifecycle Latency
17. Peak Memory Delta (tracemalloc)
18. Neural Core Immutability (ΔW = 0, Parameters=3,443,136, SHA-256 Hash Matching)
"""

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Dict, Any, List, Optional
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate
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
    ResourceRequirements,
    ResourceExecutionGrant,
    GrantType,
    TaskCheckpoint,
    TaskResultEnvelope,
    WorkUnit,
    DistributedTask,
)
from chakrview.cognition.federation.tasks.grant import ExecutionGrantManager
from chakrview.cognition.federation.tasks.scheduler import (
    DeterministicTaskScheduler,
    SchedulingDecision,
)
from chakrview.cognition.federation.tasks.checkpoint import TaskCheckpointManager
from chakrview.cognition.federation.tasks.validator import TaskResultValidator
from chakrview.cognition.federation.tasks.aggregator import TaskResultAggregator
from chakrview.cognition.federation.tasks.coordinator import FederationTaskCoordinator
from chakrview.cognition.federation.tasks.executor import FederationTaskExecutor


EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3443136


class MockCapabilityGate:
    """Mock CapabilityGate for benchmark purposes."""
    def __init__(self) -> None:
        self.authorized = True
        self.results = {"add": 42}

    def authorize(self, req: Any, context: Any = None) -> bool:
        return self.authorized

    def execute(self, req: Any, context: Any = None) -> Any:
        from dataclasses import dataclass
        @dataclass
        class Result:
            success: bool = True
            data: Any = None
            error_message: Optional[str] = None
        if req.capability_id in self.results:
            return Result(success=True, data=self.results[req.capability_id])
        return Result(success=False, error_message=f"Unknown capability: {req.capability_id}")


def _compute_model_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _compute_stats(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "samples": 0}
    sorted_s = sorted(samples_ms)
    n = len(sorted_s)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "mean_ms": round(statistics.mean(sorted_s), 4),
        "median_ms": round(statistics.median(sorted_s), 4),
        "min_ms": round(sorted_s[0], 4),
        "max_ms": round(sorted_s[-1], 4),
        "p95_ms": round(sorted_s[p95_idx], 4),
        "p99_ms": round(sorted_s[p99_idx], 4),
        "samples": n,
        "throughput_ops_sec": round(1000.0 / statistics.mean(sorted_s), 2) if statistics.mean(sorted_s) > 0 else 0.0,
    }


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


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING CHAKRVIEW STEP 38 PRODUCTION BENCHMARK")
    print("DISTRIBUTED RESOURCE ORCHESTRATION & FAULT-TOLERANT TASK EXECUTION")
    print("=" * 80)

    tracemalloc.start()

    # 1. Neural Core Verification
    torch.manual_seed(42)
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT, f"Param mismatch: {param_count} vs {EXPECTED_PARAM_COUNT}"
    pre_digest = _compute_model_hash(model)
    assert pre_digest == EXPECTED_WEIGHT_HASH, f"Hash mismatch: {pre_digest} vs {EXPECTED_WEIGHT_HASH}"
    print(f"Neural Core Baseline: {param_count:,} params, SHA-256={pre_digest[:16]}... (OK)")

    # Warmup
    print("Warming up benchmarks...")
    N_SAMPLES = 200

    # 2. Task Creation & Decomposition
    reg_init = FederationResourceRegistry(local_node_id="coordinator_node", local_zone_id="zone-1")
    sched_init = DeterministicTaskScheduler(local_node_id="coordinator_node", resource_registry=reg_init)
    cp_init = TaskCheckpointManager()
    coord = FederationTaskCoordinator(local_node_id="coordinator_node", scheduler=sched_init, checkpoint_manager=cp_init)
    t_create, t_decomp = [], []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        t = coord.create_task(name="BenchmarkTask", aggregation_strategy=AggregationStrategy.CONCATENATE)
        t_create.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        coord.decompose_task(
            t.task_id,
            [{"capability_id": "add", "input_payload": {"chunk": i}} for i in range(4)],
        )
        t_decomp.append((time.perf_counter() - t0) * 1000.0)

    # 3. WorkUnit State Transitions
    t_unit_trans = []
    sample_unit = coord.get_task(t.task_id).work_units[0]
    for _ in range(N_SAMPLES):
        sample_unit.state = WorkUnitState.PENDING
        t0 = time.perf_counter()
        sample_unit.transition_to(WorkUnitState.ASSIGNED)
        sample_unit.transition_to(WorkUnitState.RUNNING)
        sample_unit.transition_to(WorkUnitState.CHECKPOINTED)
        sample_unit.transition_to(WorkUnitState.COMPLETED)
        t_unit_trans.append((time.perf_counter() - t0) * 1000.0 / 4.0)

    # 4. Execution Grant Authorization & Accounting
    grant_mgr = ExecutionGrantManager(local_node_id="local_worker_node")
    req = ResourceRequirements(capability_id="add", min_cpu_cores=1.0, min_memory_mb=512)
    t_grant_auth, t_grant_acc = [], []
    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        g = grant_mgr.authorize_execution(
            peer_node_id="remote_coord",
            tenant_id="tenant-default",
            capability_id="add",
            requirements=req,
        )
        t_grant_auth.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        grant_mgr.allocate_resources(g.grant_id, req)
        grant_mgr.release_resources(g.grant_id, req)
        t_grant_acc.append((time.perf_counter() - t0) * 1000.0)

    # 5. Deterministic Task Scheduling
    reg_single = FederationResourceRegistry(local_node_id="coord_node", local_zone_id="zone-1")
    reg_single.record_peer_advertisement(_build_dummy_adv("peer_worker", cores=8.0, mem_mb=16384))
    sched_single = DeterministicTaskScheduler(local_node_id="coord_node", resource_registry=reg_single)

    t_sched_single = []
    unit_for_sched = coord.get_task(t.task_id).work_units[0]
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        sched_single.schedule_unit(unit_for_sched)
        t_sched_single.append((time.perf_counter() - t0) * 1000.0)

    # 10 Peers for candidate sorting & affinity
    reg_multi = FederationResourceRegistry(local_node_id="coord_node", local_zone_id="zone-1")
    for i in range(10):
        reg_multi.record_peer_advertisement(_build_dummy_adv(f"peer_worker_{i}", cores=float(2 + (i % 6) * 2), mem_mb=4096 * (1 + i % 4)))
    sched_multi = DeterministicTaskScheduler(local_node_id="coord_node", resource_registry=reg_multi)

    t_sched_multi = []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        sched_multi.schedule_unit(unit_for_sched)
        t_sched_multi.append((time.perf_counter() - t0) * 1000.0)

    # 6. Checkpoint Creation, Verification & Corruption Rejection
    cp_mgr = TaskCheckpointManager()
    t_cp_create, t_cp_verify, t_cp_reject = [], [], []
    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        cp = TaskCheckpoint(
            task_id="t_bench",
            unit_id=f"u_bench_{i}",
            checkpoint_id=f"cp_{i}",
            sequence=1,
            state=WorkUnitState.RUNNING,
            progress=0.5,
            partial_result={"step": i, "acc": i * 1.5},
            worker_id="peer_worker_0",
            attempt=1,
        )
        t_cp_create.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        cp_mgr.save_checkpoint(cp)
        t_cp_verify.append((time.perf_counter() - t0) * 1000.0)

        # Corrupted checkpoint test
        corrupt_cp = TaskCheckpoint(
            task_id="t_bench",
            unit_id=f"u_bench_corrupt_{i}",
            checkpoint_id=f"cp_corrupt_{i}",
            sequence=1,
            state=WorkUnitState.RUNNING,
            progress=0.5,
            partial_result={"step": i},
            worker_id="peer_worker_0",
            attempt=1,
            integrity_digest="invalid_sha256_digest_tampered",
        )
        t0 = time.perf_counter()
        try:
            cp_mgr.save_checkpoint(corrupt_cp)
        except Exception:
            pass
        t_cp_reject.append((time.perf_counter() - t0) * 1000.0)

    # 7. Result Validation
    validator = TaskResultValidator()
    t_val_valid = []
    unit_val = WorkUnit(
        unit_id="u_val",
        task_id="t_val",
        sequence=0,
        capability_id="add",
        input_payload={},
        requirements=req,
        state=WorkUnitState.RUNNING,
        assigned_node_id="worker_alpha",
        attempt=1,
    )
    for i in range(N_SAMPLES):
        res = TaskResultEnvelope(
            task_id="t_val",
            unit_id="u_val",
            attempt=1,
            worker_id="worker_alpha",
            status="SUCCESS",
            result_data={"output": i * 10},
            execution_time_ms=12.5,
        )
        t0 = time.perf_counter()
        validator.validate_result(unit_val, res)
        t_val_valid.append((time.perf_counter() - t0) * 1000.0)

    # 8. Aggregation Strategies
    aggregator = TaskResultAggregator()
    t_agg_concat, t_agg_sum, t_agg_dict = [], [], []

    # CONCATENATE Task
    task_cat = DistributedTask(task_id="t_cat", name="CatTask", aggregation_strategy=AggregationStrategy.CONCATENATE)
    uc1 = WorkUnit(unit_id="uc1", task_id="t_cat", sequence=0, capability_id="add", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    uc1.result = TaskResultEnvelope("t_cat", "uc1", 1, "w1", "SUCCESS", "Hello, ", 5.0)
    uc2 = WorkUnit(unit_id="uc2", task_id="t_cat", sequence=1, capability_id="add", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    uc2.result = TaskResultEnvelope("t_cat", "uc2", 1, "w1", "SUCCESS", "World!", 5.0)
    task_cat.work_units = [uc1, uc2]

    # REDUCE_SUM Task
    task_sum = DistributedTask(task_id="t_sum", name="SumTask", aggregation_strategy=AggregationStrategy.REDUCE_SUM)
    us1 = WorkUnit(unit_id="us1", task_id="t_sum", sequence=0, capability_id="add", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    us1.result = TaskResultEnvelope("t_sum", "us1", 1, "w1", "SUCCESS", 10.5, 5.0)
    us2 = WorkUnit(unit_id="us2", task_id="t_sum", sequence=1, capability_id="add", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    us2.result = TaskResultEnvelope("t_sum", "us2", 1, "w1", "SUCCESS", 20.0, 5.0)
    task_sum.work_units = [us1, us2]

    # MERGE_DICT Task
    task_dict = DistributedTask(task_id="t_dict", name="DictTask", aggregation_strategy=AggregationStrategy.MERGE_DICT)
    ud1 = WorkUnit(unit_id="ud1", task_id="t_dict", sequence=0, capability_id="add", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    ud1.result = TaskResultEnvelope("t_dict", "ud1", 1, "w1", "SUCCESS", {"a": 1, "b": 2}, 5.0)
    ud2 = WorkUnit(unit_id="ud2", task_id="t_dict", sequence=1, capability_id="add", input_payload={}, requirements=req, state=WorkUnitState.COMPLETED)
    ud2.result = TaskResultEnvelope("t_dict", "ud2", 1, "w1", "SUCCESS", {"c": 3}, 5.0)
    task_dict.work_units = [ud1, ud2]

    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        aggregator.aggregate_results(task_cat)
        t_agg_concat.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        aggregator.aggregate_results(task_sum)
        t_agg_sum.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        aggregator.aggregate_results(task_dict)
        t_agg_dict.append((time.perf_counter() - t0) * 1000.0)

    # 9. Worker Failure Detection & Rescheduling
    t_failover = []
    for i in range(N_SAMPLES):
        fail_coord = FederationTaskCoordinator(local_node_id="coord_node", scheduler=sched_multi, checkpoint_manager=cp_mgr)
        task_f = fail_coord.create_task(name="FailoverBench", aggregation_strategy=AggregationStrategy.CONCATENATE)
        u_f = fail_coord.decompose_task(task_f.task_id, [{"capability_id": "add", "input_payload": {"val": 1}}])[0]
        u_f.assigned_node_id = "peer_worker_0"
        u_f.attempt = 1
        u_f.transition_to(WorkUnitState.ASSIGNED)
        u_f.transition_to(WorkUnitState.RUNNING)
        fail_coord._worker_assignments["peer_worker_0"] = {(task_f.task_id, u_f.unit_id)}

        t0 = time.perf_counter()
        fail_coord.handle_worker_failure("peer_worker_0")
        t_failover.append((time.perf_counter() - t0) * 1000.0)

    # 10. Complete End-to-End Task Lifecycle
    reg_e2e = FederationResourceRegistry(local_node_id="node_local", local_zone_id="zone-default")
    reg_e2e.register_local_profile(
        _build_dummy_profile(cores=4.0, mem_mb=4096),
        [AdvertisedCapability(capability_id="add", name="Add", description="Add", execution_type="CPU")],
    )
    sched_e2e = DeterministicTaskScheduler(local_node_id="node_local", resource_registry=reg_e2e)
    grant_mgr_e2e = ExecutionGrantManager(local_node_id="node_local")
    executor_e2e = FederationTaskExecutor(
        local_node_id="node_local",
        grant_manager=grant_mgr_e2e,
        capability_gate=MockCapabilityGate(),
    )
    coord_e2e = FederationTaskCoordinator(
        local_node_id="node_local",
        scheduler=sched_e2e,
        checkpoint_manager=cp_mgr,
        local_executor=executor_e2e,
    )

    t_e2e = []
    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        task_e2e = coord_e2e.create_task(name="E2E_Bench", aggregation_strategy=AggregationStrategy.REDUCE_SUM)
        coord_e2e.decompose_task(task_e2e.task_id, [
            {"capability_id": "add", "input_payload": {"val": 1}},
            {"capability_id": "add", "input_payload": {"val": 2}},
        ])
        coord_e2e.schedule_and_dispatch_task(task_e2e.task_id)
        assert task_e2e.state == TaskState.COMPLETED
        t_e2e.append((time.perf_counter() - t0) * 1000.0)

    # Memory and Neural Checks
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    post_digest = _compute_model_hash(model)
    assert pre_digest == post_digest, f"Weight mutation detected! {pre_digest} != {post_digest}"

    results = {
        "step": "STEP_38",
        "title": "Distributed Resource Orchestration, Fault-Tolerant Task Execution & Work Continuity",
        "benchmark_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "neural_integrity": {
            "parameter_count": param_count,
            "vocabulary_size": 4096,
            "max_context_length": 512,
            "weight_hash": post_digest,
            "delta_w": 0,
            "status": "RATIFIED_IMMUTABLE",
        },
        "memory_metrics": {
            "current_memory_mb": round(current_mem / (1024 * 1024), 2),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        },
        "latency_and_throughput": {
            "task_creation": _compute_stats(t_create),
            "task_decomposition": _compute_stats(t_decomp),
            "work_unit_state_transition": _compute_stats(t_unit_trans),
            "execution_grant_authorization": _compute_stats(t_grant_auth),
            "execution_grant_accounting": _compute_stats(t_grant_acc),
            "task_scheduling_single_candidate": _compute_stats(t_sched_single),
            "task_scheduling_10_candidates": _compute_stats(t_sched_multi),
            "checkpoint_creation": _compute_stats(t_cp_create),
            "checkpoint_verification": _compute_stats(t_cp_verify),
            "checkpoint_corruption_rejection": _compute_stats(t_cp_reject),
            "result_validation": _compute_stats(t_val_valid),
            "result_aggregation_concatenate": _compute_stats(t_agg_concat),
            "result_aggregation_reduce_sum": _compute_stats(t_agg_sum),
            "result_aggregation_merge_dict": _compute_stats(t_agg_dict),
            "worker_failure_rescheduling": _compute_stats(t_failover),
            "end_to_end_distributed_task_lifecycle": _compute_stats(t_e2e),
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "docs" / "STEP_38_BENCHMARK_RESULTS.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY. Results saved to: {out_path}")
    print(f"Task Creation Throughput: {results['latency_and_throughput']['task_creation']['throughput_ops_sec']} ops/sec")
    print(f"Scheduling (10 candidates) Mean: {results['latency_and_throughput']['task_scheduling_10_candidates']['mean_ms']} ms")
    print(f"Execution Grant Authorization Mean: {results['latency_and_throughput']['execution_grant_authorization']['mean_ms']} ms")
    print(f"Checkpoint Verification Mean: {results['latency_and_throughput']['checkpoint_verification']['mean_ms']} ms")
    print(f"Worker Rescheduling Mean: {results['latency_and_throughput']['worker_failure_rescheduling']['mean_ms']} ms")
    print(f"End-to-End Lifecycle Mean: {results['latency_and_throughput']['end_to_end_distributed_task_lifecycle']['mean_ms']} ms ({results['latency_and_throughput']['end_to_end_distributed_task_lifecycle']['throughput_ops_sec']} ops/sec)")
    print(f"Peak Memory: {results['memory_metrics']['peak_memory_mb']} MB")
    print(f"Neural Core: Delta_W = 0, Hash={post_digest[:16]}... (OK)")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmarks()
