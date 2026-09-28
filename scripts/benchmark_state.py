"""
Empirical Performance Benchmark for Step 18: Cognitive Identity & System State.

Measures:
1. State manager instantiation latency
2. State update latency (task updates, epistemic assertions, uncertainties)
3. Snapshot creation latency
4. Snapshot comparison latency (delta calculation)
5. JSON serialization latency (to_json)
6. JSON deserialization latency (from_json)
7. Rollback transition latency (non-destructive state restoration with audit)
8. Assertion lookup and query latency
9. Scale benchmark: 10, 100, 1,000 assertions and tasks

Outputs structured JSON to docs/STEP_18_BENCHMARK_RESULTS.json.
"""

import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List

from chakrview.state.identity import get_current_system_identity
from chakrview.state.epistemic import KnowledgeState, EpistemicStatus
from chakrview.state.task_state import TaskState, TaskPhase
from chakrview.state.manager import CognitiveStateManager
from chakrview.state.snapshot import CognitiveStateSnapshot, compare_snapshots


def benchmark_state_subsystem() -> Dict[str, Any]:
    print("=" * 70)
    print("CHAKRVIEW STEP 18: COGNITIVE IDENTITY & SYSTEM STATE BENCHMARK")
    print("=" * 70)

    # 1. State Manager Creation Overhead
    t0 = time.perf_counter()
    n_create_iters = 5000
    for _ in range(n_create_iters):
        _ = CognitiveStateManager(owner_id="benchmark_user", session_id="bench_sess")
    t_create = (time.perf_counter() - t0) / n_create_iters
    create_ops_sec = int(1.0 / t_create) if t_create > 0 else 0
    print(f"[1] State Manager Creation Latency: {t_create * 1e6:.2f} µs/op ({create_ops_sec:,} ops/sec)")

    mgr = CognitiveStateManager(owner_id="benchmark_user", session_id="bench_sess")

    # 2. Knowledge Assertion Overhead
    t0 = time.perf_counter()
    n_assert_iters = 10000
    for i in range(n_assert_iters):
        mgr.assert_knowledge(
            subject=f"entity_{i % 100}",
            predicate="metric_val",
            value=float(i),
            status=EpistemicStatus.KNOWN,
            confidence=0.95,
        )
    t_assert = (time.perf_counter() - t0) / n_assert_iters
    assert_ops_sec = int(1.0 / t_assert) if t_assert > 0 else 0
    print(f"[2] Knowledge Assertion Latency: {t_assert * 1e6:.2f} µs/op ({assert_ops_sec:,} ops/sec)")

    # 3. Task Update Overhead
    task = TaskState(task_id="bench_task_01", goal="Perform benchmarking task")
    t0 = time.perf_counter()
    n_task_iters = 10000
    for i in range(n_task_iters):
        task.transition_phase(TaskPhase.EXECUTING if i % 2 == 0 else TaskPhase.OBSERVING)
        mgr.update_task(task)
    t_task = (time.perf_counter() - t0) / n_task_iters
    task_ops_sec = int(1.0 / t_task) if t_task > 0 else 0
    print(f"[3] Task State Update Latency: {t_task * 1e6:.2f} µs/op ({task_ops_sec:,} ops/sec)")

    # 4. Uncertainty Record Overhead
    t0 = time.perf_counter()
    n_unc_iters = 10000
    for i in range(n_unc_iters):
        mgr.record_uncertainty(
            key=f"unc_key_{i % 50}",
            confidence=0.8,
            reason="Sample measurement variance",
            source="bench_sensor",
        )
    t_unc = (time.perf_counter() - t0) / n_unc_iters
    unc_ops_sec = int(1.0 / t_unc) if t_unc > 0 else 0
    print(f"[4] Uncertainty Recording Latency: {t_unc * 1e6:.2f} µs/op ({unc_ops_sec:,} ops/sec)")

    # 5. Snapshot Creation Latency (Realistic Session State: 100 assertions, 10 tasks)
    snap_mgr = CognitiveStateManager(owner_id="snap_user", session_id="snap_sess")
    for i in range(100):
        snap_mgr.assert_knowledge(f"entity_{i}", "status", "active")
    for i in range(10):
        snap_mgr.update_task(TaskState(task_id=f"task_{i}", goal=f"Goal {i}"))
    for i in range(10):
        snap_mgr.record_uncertainty(f"unc_{i}", confidence=0.85, reason="Observation variance", source="sensor")

    t0 = time.perf_counter()
    n_snap_iters = 50
    snaps = []
    for i in range(n_snap_iters):
        s = snap_mgr.snapshot(description=f"Bench snapshot {i}")
        snaps.append(s)
    t_snap = (time.perf_counter() - t0) / n_snap_iters
    snap_ops_sec = int(1.0 / t_snap) if t_snap > 0 else 0
    print(f"[5] Snapshot Creation Latency: {t_snap * 1000.0:.3f} ms/snapshot ({snap_ops_sec:,} snaps/sec)")

    snap_a = snaps[0]
    snap_b = snaps[-1]

    # 6. Snapshot Comparison Latency
    t0 = time.perf_counter()
    n_cmp_iters = 1000
    for _ in range(n_cmp_iters):
        _ = compare_snapshots(snap_a, snap_b)
    t_cmp = (time.perf_counter() - t0) / n_cmp_iters
    cmp_ops_sec = int(1.0 / t_cmp) if t_cmp > 0 else 0
    print(f"[6] Snapshot Comparison Latency: {t_cmp * 1e6:.2f} µs/compare ({cmp_ops_sec:,} compares/sec)")

    # 7. JSON Serialization Latency
    t0 = time.perf_counter()
    n_ser_iters = 50
    for _ in range(n_ser_iters):
        json_str = snap_b.to_json()
    t_ser = (time.perf_counter() - t0) / n_ser_iters
    ser_ops_sec = int(1.0 / t_ser) if t_ser > 0 else 0
    print(f"[7] Snapshot JSON Serialization: {t_ser * 1000.0:.3f} ms/serialize ({ser_ops_sec:,} snaps/sec, size={len(json_str)/1024:.1f} KB)")

    # 8. JSON Deserialization Latency
    t0 = time.perf_counter()
    n_deser_iters = 50
    for _ in range(n_deser_iters):
        _ = CognitiveStateSnapshot.from_json(json_str)
    t_deser = (time.perf_counter() - t0) / n_deser_iters
    deser_ops_sec = int(1.0 / t_deser) if t_deser > 0 else 0
    print(f"[8] Snapshot JSON Deserialization: {t_deser * 1000.0:.3f} ms/deserialize ({deser_ops_sec:,} snaps/sec)")

    # 9. Rollback Transition Latency
    t0 = time.perf_counter()
    n_rb_iters = 50
    for i in range(n_rb_iters):
        _ = snap_mgr.rollback(snap_a.snapshot_id, reason="Bench rollback")
    t_rb = (time.perf_counter() - t0) / n_rb_iters
    rb_ops_sec = int(1.0 / t_rb) if t_rb > 0 else 0
    print(f"[9] State Rollback Latency: {t_rb * 1000.0:.3f} ms/rollback ({rb_ops_sec:,} rollbacks/sec)")

    # 10. Query & Lookup Latency
    t0 = time.perf_counter()
    n_queries = 10000
    for i in range(n_queries):
        _ = mgr.knowledge.query(subject=f"entity_{i % 100}")
    t_query = (time.perf_counter() - t0) / n_queries
    query_ops_sec = int(1.0 / t_query) if t_query > 0 else 0
    print(f"[10] Knowledge Query Latency: {t_query * 1e6:.2f} µs/query ({query_ops_sec:,} queries/sec)")

    # 11. Scale Benchmark (10, 100, 1,000 active entities)
    print("\n[11] State Scale Benchmarks:")
    scale_bench = {}
    for n_entities in [10, 100, 1000]:
        scaled_mgr = CognitiveStateManager(owner_id="scale_user")
        for i in range(n_entities):
            scaled_mgr.assert_knowledge(
                subject=f"scale_entity_{i}",
                predicate="status",
                value="ACTIVE",
                status=EpistemicStatus.KNOWN,
            )
            scaled_mgr.update_task(TaskState(task_id=f"scale_task_{i}", goal="Test goal"))

        t0 = time.perf_counter()
        scaled_snap = scaled_mgr.snapshot(description=f"Scale N={n_entities}")
        t_scale_snap = time.perf_counter() - t0

        t0 = time.perf_counter()
        scaled_json = scaled_snap.to_json()
        t_scale_ser = time.perf_counter() - t0

        scale_bench[f"scale_{n_entities}"] = {
            "entities": n_entities,
            "snapshot_latency_ms": round(t_scale_snap * 1000.0, 3),
            "serialization_latency_ms": round(t_scale_ser * 1000.0, 3),
            "payload_size_kb": round(len(scaled_json) / 1024.0, 2),
        }
        print(f"  - Scale N={n_entities}: Snapshot = {t_scale_snap * 1000.0:.3f} ms, JSON = {t_scale_ser * 1000.0:.3f} ms, Size = {len(scaled_json)/1024.0:.1f} KB")

    report = {
        "timestamp": time.time(),
        "platform": {
            "system": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "instantiation_benchmark": {
            "latency_us": round(t_create * 1e6, 2),
            "ops_per_sec": create_ops_sec,
        },
        "assertion_benchmark": {
            "latency_us": round(t_assert * 1e6, 2),
            "ops_per_sec": assert_ops_sec,
        },
        "task_update_benchmark": {
            "latency_us": round(t_task * 1e6, 2),
            "ops_per_sec": task_ops_sec,
        },
        "uncertainty_benchmark": {
            "latency_us": round(t_unc * 1e6, 2),
            "ops_per_sec": unc_ops_sec,
        },
        "snapshot_creation_benchmark": {
            "latency_ms": round(t_snap * 1000.0, 3),
            "snapshots_per_sec": snap_ops_sec,
        },
        "snapshot_compare_benchmark": {
            "latency_us": round(t_cmp * 1e6, 2),
            "compares_per_sec": cmp_ops_sec,
        },
        "serialization_benchmark": {
            "latency_ms": round(t_ser * 1000.0, 3),
            "ops_per_sec": ser_ops_sec,
        },
        "deserialization_benchmark": {
            "latency_ms": round(t_deser * 1000.0, 3),
            "ops_per_sec": deser_ops_sec,
        },
        "rollback_benchmark": {
            "latency_ms": round(t_rb * 1000.0, 3),
            "rollbacks_per_sec": rb_ops_sec,
        },
        "query_benchmark": {
            "latency_us": round(t_query * 1e6, 2),
            "queries_per_sec": query_ops_sec,
        },
        "scale_benchmarks": scale_bench,
    }

    out_path = os.path.join("docs", "STEP_18_BENCHMARK_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark results successfully written to {out_path}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    benchmark_state_subsystem()
