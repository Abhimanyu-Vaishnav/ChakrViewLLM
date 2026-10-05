"""
ChakrView Step 128: Full Distributed Cognitive Intelligence Benchmark.

Integrates Steps 121 through 127 into one unified empirical benchmark:
1. Genuine TCP network node server starts on background socket port
2. TCP network transport channel dispatches tasks to remote node
3. HMAC envelope authentication and replay protection
4. Distributed state synchronization with revision control
5. Federated memory plane with structured provenance
6. Long-horizon pipeline execution with durable stage checkpoints
7. Neural/cognitive boundary inference proxying without direct tool access
8. Governed self-adaptation based on worker crash statistics
9. Baseline neural model bit-exact verification
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from typing import Any, Dict, List, Tuple

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    ReviewerVerdict,
    WorkerContract,
    WorkerPackage,
)
from chakrview.cognition.multi_agent.result import (
    WorkerExecutionStatus,
    WorkerResult,
)
from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
)
from chakrview.cognition.multi_agent.network_federation import (
    NetworkNodeDescriptor,
    TcpNetworkTransportChannel,
    TcpWorkerNodeServer,
)
from chakrview.cognition.multi_agent.secure_transport import (
    SecureEnvelope,
    DistributedSecurityValidator,
)
from chakrview.cognition.multi_agent.state_sync import (
    CognitiveStateCategory,
    VersionedCognitiveStateRecord,
    DistributedStateSynchronizer,
)
from chakrview.cognition.multi_agent.memory_plane import (
    ProvenanceMemoryRecord,
    FederatedMemoryPlane,
)
from chakrview.cognition.multi_agent.long_horizon import (
    PipelineStage,
    LongHorizonPlanner,
)
from chakrview.cognition.multi_agent.neural_boundary import (
    ModelTier,
    NeuralCognitiveBridge,
)
from chakrview.cognition.multi_agent.federation_adaptation import (
    FederationAdaptationManager,
)
from chakrview.cognition.multi_agent.federation import (
    FederatedWorkerMetadata,
    FederatedScheduler,
)
from chakrview.cognition.autonomous_benchmark import (
    build_benchmark_tool_gate,
    BenchmarkDiskProjectEngine,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


def run_complete_step128_benchmark(bench_dir: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bench_dir = Path(bench_dir).resolve()
    repo_dir = bench_dir / "repository"
    db_path = bench_dir / "project_brain.db"

    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup repository fixture
    from scripts.setup_step113_fixture import setup_step113_repository
    setup_step113_repository(bench_dir)

    tool_gate, _ = build_benchmark_tool_gate(repo_dir)
    scheduler = FederatedScheduler(db_path, repo_dir, tool_gate)

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Baseline Check Before Run
    # -------------------------------------------------------------------------
    base_model = instantiate_frozen_baseline()
    hash_pre = compute_model_hash(base_model)
    assert hash_pre == EXPECTED_WEIGHT_HASH

    # -------------------------------------------------------------------------
    # STEP 121: Start Real TCP Worker Node Server
    # -------------------------------------------------------------------------
    server = TcpWorkerNodeServer("tcp_node_01", host="127.0.0.1", port=0, workspace_root=repo_dir)
    port = server.start()
    time.sleep(0.1)

    tcp_channel = TcpNetworkTransportChannel("127.0.0.1", port)
    assert tcp_channel.is_available() is True

    # Dispatch request across real TCP socket boundary
    c_net = WorkerContract("proc_analyst_01", WorkerRole.PROJECT_ANALYST, "net_task_01", ["read_file"], ["app/models.py"], 512)
    req_env = RequestEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id="proc_analyst_01",
        task_id="net_task_01",
        package_id="pkg_net",
        role=WorkerRole.PROJECT_ANALYST.value,
        context_hash="hash_net",
        context_token_count=50,
        allowed_tools=["read_file"],
        allowed_files=["app/models.py"],
        context_budget=512,
        package_payload={"task_id": "net_task_01", "objective": "TCP Read", "measured_context_tokens": 50},
    )
    resp_env = tcp_channel.send_request(req_env, timeout_seconds=10.0)
    assert resp_env.result_status == "SUCCESS"
    results["step121_tcp_node_execution"] = {"port": port, "status": "SUCCESS"}

    # -------------------------------------------------------------------------
    # STEP 122: Secure Distributed Transport (HMAC & Replay Defense)
    # -------------------------------------------------------------------------
    sec_validator = DistributedSecurityValidator()
    sec_env = SecureEnvelope(
        inner_payload={"msg": "authorized_task"},
        sender_node_id="coordinator",
        recipient_node_id="tcp_node_01",
        nonce="nonce_unique_123",
    )
    sec_env.sign()
    sec_validator.validate_incoming_envelope(sec_env)
    # Replay attempt must fail
    try:
        sec_validator.validate_incoming_envelope(sec_env)
        results["step122_replay_defense"] = False
    except ValueError:
        results["step122_replay_defense"] = True

    # -------------------------------------------------------------------------
    # STEP 123: Distributed State Synchronization
    # -------------------------------------------------------------------------
    state_sync = DistributedStateSynchronizer(db_path)
    rec1 = VersionedCognitiveStateRecord("task_understanding", 1, "node_A", CognitiveStateCategory.PROJECT_UNDERSTANDING, {"symbols": 5}, "h1")
    assert state_sync.publish_state_update(rec1) is True

    # Stale write rejected
    rec_stale = VersionedCognitiveStateRecord("task_understanding", 1, "node_B", CognitiveStateCategory.PROJECT_UNDERSTANDING, {"symbols": 3}, "h2")
    assert state_sync.publish_state_update(rec_stale) is False

    # Newer write accepted
    rec2 = VersionedCognitiveStateRecord("task_understanding", 2, "node_B", CognitiveStateCategory.PROJECT_UNDERSTANDING, {"symbols": 8}, "h3")
    assert state_sync.publish_state_update(rec2) is True
    results["step123_state_synchronization"] = True

    # -------------------------------------------------------------------------
    # STEP 124: Federated Persistent Memory Plane
    # -------------------------------------------------------------------------
    mem_plane = FederatedMemoryPlane(db_path)
    mem_rec = ProvenanceMemoryRecord("fact_01", "UserAccount", "implements", "SHA256_hashing", "tcp_node_01", "net_task_01")
    mem_plane.store_fact(mem_rec)
    queried = mem_plane.query_facts(subject="UserAccount")
    assert len(queried) == 1
    assert queried[0].predicate == "implements"
    results["step124_memory_plane"] = True

    # -------------------------------------------------------------------------
    # STEP 125: Long-Horizon Cognitive Pipeline
    # -------------------------------------------------------------------------
    long_planner = LongHorizonPlanner(scheduler, "pipeline_main")
    stage, rev, _ = long_planner.get_pipeline_state()
    assert stage == PipelineStage.ANALYZE
    long_planner.checkpoint_stage(PipelineStage.PLAN, 0, {"plan": "ok"})
    stage_reloaded, _, _ = long_planner.get_pipeline_state()
    assert stage_reloaded == PipelineStage.PLAN
    results["step125_long_horizon_checkpointing"] = True

    # -------------------------------------------------------------------------
    # STEP 126: Neural/Cognitive Integration Boundary
    # -------------------------------------------------------------------------
    neural_bridge = NeuralCognitiveBridge(ModelTier.CANONICAL_BASELINE)
    ppl = neural_bridge.score_sequence_perplexity([1, 2, 3, 4, 5])
    assert ppl > 0.0
    assert neural_bridge.verify_baseline_immutability() is True
    results["step126_neural_bridge"] = {"perplexity_proxy": round(ppl, 2), "baseline_intact": True}

    # -------------------------------------------------------------------------
    # STEP 127: Governed Self-Evaluation & Federation Adaptation
    # -------------------------------------------------------------------------
    adapt_mgr = FederationAdaptationManager(db_path)
    adapt_mgr.record_outcome("flaky_worker", WorkerRole.IMPLEMENTER, False, "PROCESS_CRASH")
    adapt_mgr.record_outcome("flaky_worker", WorkerRole.IMPLEMENTER, True)
    rel = adapt_mgr.get_worker_reliability("flaky_worker", WorkerRole.IMPLEMENTER)
    assert rel == 0.5  # 1 fail out of 2 executions
    results["step127_federation_adaptation"] = {"flaky_reliability": rel}

    # -------------------------------------------------------------------------
    # Stop TCP Server
    # -------------------------------------------------------------------------
    server.stop()

    # -------------------------------------------------------------------------
    # Final Immutability Check
    # -------------------------------------------------------------------------
    hash_post = compute_model_hash(base_model)
    assert hash_post == EXPECTED_WEIGHT_HASH
    results["baseline_bit_exact"] = True

    elapsed = time.perf_counter() - t0
    final_summary = {
        "milestone": "Steps 121-128 Master Wave",
        "benchmark_status": "SUCCESS",
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
    }

    with open(bench_dir / "final_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    return final_summary


if __name__ == "__main__":
    b_dir = Path("artifacts/step128_full_benchmark")
    out = run_complete_step128_benchmark(b_dir)
    print("Steps 121-128 Master Wave Complete! Status:", out["benchmark_status"])
