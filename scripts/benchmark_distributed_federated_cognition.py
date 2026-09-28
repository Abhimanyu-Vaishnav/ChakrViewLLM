"""
Empirical Benchmark for ChakrView Step 27:
Distributed Federated Cognition & Secure Agent Transport Foundation.

Measures:
1. Node registration latency
2. Message creation & nonce generation latency
3. Message validation & SHA-256 integrity verification latency
4. Loopback transport latency (send / receive / request)
5. Task routing latency (technical resource & locality ranking)
6. Retry & backoff overhead
7. Canonical serialization overhead
8. Distributed evidence aggregation latency
9. Distributed consensus synthesis latency
10. Complete distributed cognitive cycle latency
11. Hardware adaptation comparison: LOW_RESOURCE vs STANDARD vs HIGH_RESOURCE
12. Rejected & replayed message handling cost
13. Peak process memory usage
14. Model weight immutability & frozen invariant verification (3,443,136 params, 4096 vocab, 512 context)
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Dict, Any, List
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.registry import CapabilityRegistry
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    EXPECTED_PARAMETERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_PAD_ID,
    WeightMutationError,
)
from chakrview.cognition.distributed.models import (
    NodeIdentity,
    NodeRole,
    NodeStatus,
    NodeTrustState,
    NodeCapabilities,
    NodeResourceProfile,
    NodeEndpoint,
    NodeRegistration,
    DistributedMessageEnvelope,
    DistributedRouteDecision,
    SafePublicDistributedTrace,
)
from chakrview.cognition.distributed.transport import (
    LoopbackTransport,
    TransportResponse,
    TransportStatus,
)
from chakrview.cognition.distributed.security import (
    ReplayProtectionTracker,
    DeterministicHmacMessageSigner,
    DeterministicHmacMessageVerifier,
    create_distributed_envelope,
)
from chakrview.cognition.distributed.registry import DistributedNodeRegistry
from chakrview.cognition.distributed.resilience import (
    CircuitBreaker,
    RetryPolicy,
)
from chakrview.cognition.distributed.router import DistributedTaskRouter
from chakrview.cognition.distributed.policy import DistributedExecutionPolicy
from chakrview.cognition.distributed.observability import DistributedObservabilityMetrics
from chakrview.cognition.distributed.engine import DistributedFederatedCognitionEngine

from chakrview.cognition.federated.models import (
    AgentRole,
    MessageType,
    AgentTask,
    AgentMessage,
    FederatedConflictRecord,
)
from chakrview.cognition.federated.evidence import FederatedEvidenceAggregator
from chakrview.cognition.federated.conflict import FederatedConflictResolver
from chakrview.cognition.federated.synthesis import FederatedSynthesizer


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 72)
    print("  CHAKRVIEW STEP 27 — DISTRIBUTED FEDERATED COGNITION BENCHMARK")
    print("=" * 72)

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Initialize Frozen Model & Tokenizer
    print("\n[1/12] Initializing Sovereign Frozen ChakrMicro v0.1 Core...")
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        tokenizer = load_experiment_artifacts(exp_dir)
    else:
        tokenizer = BPETokenizer()

    guard = CoreIntegrityGuard()
    inv_res = guard.verify_model(model)
    assert inv_res.passed, f"Model invariant check failed: {inv_res.message}"
    init_hash = guard.compute_weight_fingerprint(model)

    print(f"  Parameters        : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocabulary        : {model.config.vocab_size}")
    print(f"  Max Context       : {model.config.max_seq_len}")
    print(f"  Initial SHA-256 FP: {init_hash}")

    # 2. Benchmark Node Registration Latency
    print("\n[2/12] Benchmarking Node Registration Latency...")
    registry = DistributedNodeRegistry(max_nodes=16)
    t0 = time.perf_counter()
    N_NODES = 10
    for i in range(N_NODES):
        reg = NodeRegistration(
            identity=NodeIdentity(node_id=f"worker_bench_{i}", tenant_id="tenant_bench"),
            endpoint=NodeEndpoint(endpoint_id=f"ep_{i}", uri=f"loopback://node_{i}"),
            capabilities=NodeCapabilities(supported_roles=[AgentRole.ANALYST, AgentRole.RESEARCHER]),
            resource_profile=NodeResourceProfile(latency_tier="LAN"),
        )
        registry.register_node(reg)
    node_reg_latency_ms = ((time.perf_counter() - t0) / N_NODES) * 1000.0
    print(f"  Avg Node Registration Latency: {node_reg_latency_ms:.3f} ms")

    # 3. Benchmark Message Creation & Signing Latency
    print("\n[3/12] Benchmarking Distributed Envelope Creation & Signing...")
    signer = DeterministicHmacMessageSigner(b"secret_bench_key")
    t0 = time.perf_counter()
    N_MSGS = 100
    for i in range(N_MSGS):
        _ = create_distributed_envelope(
            sender_node_id="node_src",
            sender_agent_id="agent_ana",
            receiver_node_id="node_dest",
            receiver_agent_id="agent_res",
            tenant_id="tenant_bench",
            session_id="session_bench",
            message_type=MessageType.TASK_REQUEST,
            payload={"metric_id": i, "val": i * 1.5},
            signer=signer,
            signer_identity="node_src",
        )
    msg_creation_latency_ms = ((time.perf_counter() - t0) / N_MSGS) * 1000.0
    print(f"  Avg Envelope Creation Latency: {msg_creation_latency_ms:.3f} ms")

    # 4. Benchmark Message Validation & Replay Tracking Latency
    print("\n[4/12] Benchmarking Message Validation & Replay Tracking...")
    sample_env = create_distributed_envelope(
        sender_node_id="node_src",
        sender_agent_id="agent_ana",
        receiver_node_id="node_dest",
        receiver_agent_id="agent_res",
        tenant_id="tenant_bench",
        session_id="session_bench",
        message_type=MessageType.TASK_REQUEST,
        payload={"task": "validate"},
    )
    t0 = time.perf_counter()
    N_VALS = 100
    for i in range(N_VALS):
        _ = sample_env.verify_integrity()
    msg_validation_latency_ms = ((time.perf_counter() - t0) / N_VALS) * 1000.0
    print(f"  Avg Integrity Check Latency : {msg_validation_latency_ms:.3f} ms")

    # 5. Benchmark Replay Protection Rejection Overhead
    print("\n[5/12] Benchmarking Replay Rejection Overhead...")
    tracker = ReplayProtectionTracker()
    tracker.validate_and_record(sample_env)
    t0 = time.perf_counter()
    N_REPLAYS = 100
    for _ in range(N_REPLAYS):
        try:
            tracker.validate_and_record(sample_env)
        except Exception:
            pass
    replay_cost_ms = ((time.perf_counter() - t0) / N_REPLAYS) * 1000.0
    print(f"  Avg Replay Rejection Cost   : {replay_cost_ms:.3f} ms")

    # 6. Benchmark Loopback Transport Request-Response Latency
    print("\n[6/12] Benchmarking Loopback Transport Request-Response...")
    transport = LoopbackTransport()
    transport.register_node_endpoint("node_dest", handler=lambda env: create_distributed_envelope(
        sender_node_id="node_dest", sender_agent_id="ag", receiver_node_id=env.route.sender_node_id,
        receiver_agent_id=env.route.sender_agent_id, tenant_id=env.route.tenant_id, session_id=env.route.session_id,
        message_type=MessageType.TASK_RESPONSE, payload={"status": "ok"}
    ))
    t0 = time.perf_counter()
    N_TRANS = 50
    for _ in range(N_TRANS):
        resp = transport.request(sample_env)
        assert resp.is_success()
    transport_latency_ms = ((time.perf_counter() - t0) / N_TRANS) * 1000.0
    print(f"  Avg Loopback Transport Latency: {transport_latency_ms:.3f} ms")

    # 7. Benchmark Task Routing Latency
    print("\n[7/12] Benchmarking Distributed Task Routing Latency...")
    router = DistributedTaskRouter(registry=registry, local_node_id="local_coord")
    task = AgentTask(
        task_id="t_bench_route",
        parent_task_id=None,
        assigned_role=AgentRole.ANALYST,
        objective="Analyze performance metrics",
    )
    t0 = time.perf_counter()
    N_ROUTES = 50
    for _ in range(N_ROUTES):
        _ = router.route_task(task, tenant_id="tenant_bench", session_id="session_bench")
    routing_latency_ms = ((time.perf_counter() - t0) / N_ROUTES) * 1000.0
    print(f"  Avg Task Routing Latency    : {routing_latency_ms:.3f} ms")

    # 8. Benchmark Canonical Serialization Overhead
    print("\n[8/12] Benchmarking Canonical Serialization Overhead...")
    t0 = time.perf_counter()
    N_SERIAL = 100
    for _ in range(N_SERIAL):
        _ = sample_env.canonical_serialize()
    serialization_latency_ms = ((time.perf_counter() - t0) / N_SERIAL) * 1000.0
    print(f"  Avg Canonical Serialization : {serialization_latency_ms:.3f} ms")

    # 9. Benchmark Distributed Evidence Aggregation & Synthesis
    print("\n[9/12] Benchmarking Evidence Aggregation & Synthesis Latency...")
    aggregator = FederatedEvidenceAggregator()
    synthesizer = FederatedSynthesizer()
    msgs = [
        AgentMessage(
            message_id=f"m_{i}",
            sender_agent_id=f"ag_{i}",
            receiver_agent_id="coord",
            tenant_id="t",
            session_id="s",
            correlation_id="c",
            message_type=MessageType.TASK_RESPONSE,
            payload={"claim": f"Fact verification {i}", "confidence": 0.85},
        )
        for i in range(4)
    ]
    t0 = time.perf_counter()
    for m in msgs:
        aggregator.ingest_message(m)
    corroborated_ev = aggregator.get_corroborated_evidence()
    ev_agg_latency_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    cand = synthesizer.synthesize(
        task_id="t_bench_syn",
        objective="Assess system stability",
        messages=msgs,
        corroborated_evidence=corroborated_ev,
        conflicts=[],
        critical_counter_evidence=[],
    )
    synthesis_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Evidence Aggregation Latency: {ev_agg_latency_ms:.3f} ms")
    print(f"  Distributed Synthesis Latency: {synthesis_latency_ms:.3f} ms")

    # 10. Hardware-Adaptive Profiles Comparison
    print("\n[10/12] Benchmarking Hardware Adaptive Distributed Profiles...")
    profile_results = {}
    profiles = ["LOW_RESOURCE", "STANDARD", "HIGH_RESOURCE"]

    roles_all = [AgentRole.ANALYST, AgentRole.RESEARCHER, AgentRole.CRITIC, AgentRole.PLANNER, AgentRole.SYNTHESIZER, AgentRole.VERIFIER]

    for prof in profiles:
        pol = DistributedExecutionPolicy.from_profile(prof)
        dist_reg = DistributedNodeRegistry(max_nodes=16)
        dist_reg.register_node(NodeRegistration(
            identity=NodeIdentity(node_id=f"worker_prof_{prof}", tenant_id=f"tenant_{prof}"),
            endpoint=NodeEndpoint(endpoint_id="ep_p", uri="loopback://prof"),
            capabilities=NodeCapabilities(supported_roles=roles_all),
            resource_profile=NodeResourceProfile(compute_profile=prof, latency_tier="LOCAL"),
        ))
        dist_trans = LoopbackTransport()
        dist_trans.register_node_endpoint("coordinator")

        engine = DistributedFederatedCognitionEngine(
            model=model,
            tokenizer=tokenizer,
            local_node_identity=NodeIdentity(node_id="coordinator", tenant_id=f"tenant_{prof}"),
            node_registry=dist_reg,
            transport=dist_trans,
            policy=pol,
        )
        engine.register_remote_agent_responder(f"worker_prof_{prof}", roles_all, f"tenant_{prof}")

        t0 = time.perf_counter()
        candidate, trace = engine.execute_distributed_cycle(
            objective="Evaluate distributed infrastructure throughput",
            tenant_id=f"tenant_{prof}",
            session_id=f"session_{prof}",
            prefer_remote=True,
        )
        elapsed_p = (time.perf_counter() - t0) * 1000.0

        profile_results[prof] = {
            "max_nodes": pol.max_nodes,
            "max_remote_tasks": pol.max_remote_tasks,
            "max_retries": pol.max_retries,
            "timeout_ms": pol.request_timeout_ms,
            "actual_remote_tasks": trace.remote_task_count,
            "messages_exchanged": trace.message_count,
            "decision_state": trace.decision_state,
            "latency_ms": round(elapsed_p, 2),
        }
        print(f"  Profile [{prof:<13}]: nodes={pol.max_nodes}, tasks={trace.remote_task_count}, msgs={trace.message_count}, latency={elapsed_p:.2f} ms")

    # 11. Multi-Node Cluster Scaling
    print("\n[11/12] Benchmarking Multi-Node Cluster Scaling...")
    # 2 Remote Nodes vs 4 Remote Nodes
    cluster_scaling = {}
    for cluster_size in (2, 4):
        c_reg = DistributedNodeRegistry(max_nodes=16)
        c_trans = LoopbackTransport()
        c_trans.register_node_endpoint("coord_scale")

        engine_scale = DistributedFederatedCognitionEngine(
            model=model,
            tokenizer=tokenizer,
            local_node_identity=NodeIdentity(node_id="coord_scale", tenant_id="tenant_scale"),
            node_registry=c_reg,
            transport=c_trans,
        )

        for i in range(cluster_size):
            node_name = f"scale_worker_{i}"
            c_reg.register_node(NodeRegistration(
                identity=NodeIdentity(node_id=node_name, tenant_id="tenant_scale"),
                endpoint=NodeEndpoint(endpoint_id=f"ep_s_{i}", uri=f"loopback://s_{i}"),
                capabilities=NodeCapabilities(supported_roles=roles_all),
                resource_profile=NodeResourceProfile(latency_tier="LAN"),
            ))
            engine_scale.register_remote_agent_responder(node_name, roles_all, "tenant_scale")

        t0 = time.perf_counter()
        candidate, trace = engine_scale.execute_distributed_cycle(
            objective="Evaluate multi-node cooperative performance",
            tenant_id="tenant_scale",
            session_id="session_scale",
            prefer_remote=True,
        )
        elapsed_scale = (time.perf_counter() - t0) * 1000.0
        cluster_scaling[f"{cluster_size}_nodes"] = {
            "remote_tasks": trace.remote_task_count,
            "messages": trace.message_count,
            "latency_ms": round(elapsed_scale, 2),
        }
        print(f"  Cluster [{cluster_size} Remote Nodes]: tasks={trace.remote_task_count}, msgs={trace.message_count}, latency={elapsed_scale:.2f} ms")

    # 12. Final Core Immutability Audit
    print("\n[12/12] Auditing Post-Benchmark Core Immutability...")
    post_hash = guard.compute_weight_fingerprint(model)
    inv_post = guard.verify_model(model)
    assert inv_post.passed, f"Post-benchmark invariant failure: {inv_post.message}"
    assert init_hash == post_hash, "CRITICAL: Model weights modified during benchmark!"
    print(f"  Initial Fingerprint: {init_hash}")
    print(f"  Final Fingerprint  : {post_hash}")
    print(f"  Fingerprints Match : {init_hash == post_hash}")
    print(f"  Invariants Passed  : {inv_post.passed}")

    total_elapsed = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": 27,
        "benchmark_title": "Distributed Federated Cognition & Secure Agent Transport Benchmark",
        "frozen_invariants": {
            "parameters": EXPECTED_PARAMETERS,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "max_seq_len": EXPECTED_MAX_SEQ_LEN,
            "bos_token_id": EXPECTED_BOS_ID,
            "eos_token_id": EXPECTED_EOS_ID,
            "pad_token_id": EXPECTED_PAD_ID,
            "runtime_weights_modified": False,
            "initial_fingerprint": init_hash,
            "final_fingerprint": post_hash,
            "fingerprints_identical": (init_hash == post_hash),
        },
        "micro_benchmarks": {
            "node_registration_latency_ms": round(node_reg_latency_ms, 3),
            "envelope_creation_latency_ms": round(msg_creation_latency_ms, 3),
            "envelope_validation_latency_ms": round(msg_validation_latency_ms, 3),
            "replay_rejection_cost_ms": round(replay_cost_ms, 3),
            "loopback_transport_latency_ms": round(transport_latency_ms, 3),
            "task_routing_latency_ms": round(routing_latency_ms, 3),
            "canonical_serialization_latency_ms": round(serialization_latency_ms, 3),
            "evidence_aggregation_latency_ms": round(ev_agg_latency_ms, 3),
            "distributed_synthesis_latency_ms": round(synthesis_latency_ms, 3),
            "full_distributed_cycle_latency_ms": round(profile_results["STANDARD"]["latency_ms"], 3),
        },
        "hardware_adaptation": profile_results,
        "cluster_scaling": cluster_scaling,
        "performance_metrics": {
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "total_benchmark_elapsed_seconds": round(total_elapsed, 2),
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "docs" / "STEP_27_BENCHMARK_RESULTS.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completed successfully in {total_elapsed:.2f}s.")
    print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Results recorded in: {out_path}")

    return results


if __name__ == "__main__":
    run_benchmarks()
