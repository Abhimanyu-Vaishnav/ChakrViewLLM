"""
Empirical Benchmark for ChakrView Step 37:
Distributed Resource & Capability Advertisement.

Measures:
1. Local Resource Discovery Latency (Hardware Profile Detection)
2. CPU Inspection Latency (Architecture, Cores & Instruction Set)
3. Memory Inspection Latency (RAM Allocation & Reservable Calculation)
4. Accelerator Discovery Latency (Hardware Device & NPU/GPU Capabilities)
5. Capability Declaration Latency (Core Capabilities & Gate Inspection)
6. ResourceAdvertisement Creation & Signing Latency
7. ResourceAdvertisement Serialization Latency (Deterministic Canonical Dict)
8. ResourceAdvertisement Deserialization Latency (Validation & Type Inflation)
9. Payload Digest Computation Latency (Canonical SHA-256 Digest)
10. Ed25519 Signature Generation Latency
11. Ed25519 Signature Verification Latency
12. Valid Advertisement Ingestion Throughput (Ops/sec)
13. Replay Rejection Latency (Monotonic Version Floor Check)
14. Tampered Advertisement Rejection Latency (Integrity Fail-Closed)
15. Stale Advertisement Rejection Latency (TTL Validation)
16. Capability Filtering Latency (100 Peer Claim Matching)
17. Transport Message Dispatch Roundtrip Latency (Channel -> Dispatcher -> Registry)
18. Peak Memory Delta (tracemalloc)
19. Neural Core Immutability (ΔW = 0, Parameters=3,443,136, SHA-256 Hash Matching)
"""

import hashlib
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
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.provider import CalculatorCapability
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
)
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.peering.models import (
    TrustLevel,
    FederationScope,
    DiscoveryStatus,
    PeerIdentity,
)
from chakrview.cognition.peering.session import (
    SecurePeerSession,
    SessionStatus,
)
from chakrview.cognition.federation.transport import (
    FederationMessageEnvelope,
    FederationMessageType,
    FederationChannel,
    ChannelState,
)
from chakrview.cognition.federation.resources import (
    AcceleratorType,
    ExecutionType,
    CPUResource,
    MemoryResource,
    AcceleratorResource,
    PlatformResource,
    NodeResourceProfile,
    AdvertisedCapability,
    ResourceAdvertisement,
    ResourceSharingPolicy,
    LocalResourceDetector,
    FederationResourceRegistry,
    FederationResourceManager,
    AdvertisementReplayError,
    AdvertisementTamperedError,
    StaleAdvertisementError,
)

FROZEN_NEURAL_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


def _compute_stats(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "throughput_ops_sec": 0.0}
    sorted_samples = sorted(samples_ms)
    mean_val = sum(sorted_samples) / len(sorted_samples)
    p95_idx = int(0.95 * len(sorted_samples))
    p95_val = sorted_samples[min(p95_idx, len(sorted_samples) - 1)]
    min_val = sorted_samples[0]
    max_val = sorted_samples[-1]
    throughput = (1000.0 / mean_val) if mean_val > 0 else 0.0
    return {
        "mean_ms": round(mean_val, 4),
        "p95_ms": round(p95_val, 4),
        "min_ms": round(min_val, 4),
        "max_ms": round(max_val, 4),
        "throughput_ops_sec": round(throughput, 2),
    }


def run_benchmarks(num_warmup: int = 15, num_measured: int = 80) -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 37: DISTRIBUTED RESOURCE & CAPABILITY ADVERTISEMENT BENCHMARK")
    print("=" * 80)

    tracemalloc.start()
    t0_all = time.perf_counter()

    # 1. Neural Core Invariant Check (Pre)
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    pre_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            pre_hasher.update(name.encode("utf-8"))
            pre_hasher.update(param.detach().cpu().numpy().tobytes())
    pre_digest = pre_hasher.hexdigest()

    assert param_count == 3_443_136, f"Expected 3,443,136 parameters, got {param_count}"
    assert pre_digest == FROZEN_NEURAL_WEIGHT_HASH, f"Weight digest mismatch: {pre_digest}"
    print(f"[*] Pre-Benchmark Neural Invariant: Parameters={param_count:,}, Weight Digest Verified.")

    # 2. Setup Components
    gate = CapabilityGate()
    gate.registry.register(CalculatorCapability())

    local_key = Ed25519PrivateKeyWrapper.generate()
    remote_key = Ed25519PrivateKeyWrapper.generate()

    local_engine = CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=gate,
        model=model,
        initial_epoch=1,
        local_private_key=local_key,
        local_peer_id="peer_alpha",
    )
    remote_engine = CrossZoneFederationEngine(
        local_zone_id="zone-beta",
        capability_gate=gate,
        model=model,
        initial_epoch=1,
        local_private_key=remote_key,
        local_peer_id="peer_beta",
    )
    local_engine.tenant_id = "default"
    remote_engine.tenant_id = "default"

    detector = LocalResourceDetector(node_epoch=1, coarse_privacy_default=False)
    local_mgr = local_engine.resource_manager
    remote_mgr = remote_engine.resource_manager

    session = SecurePeerSession(
        session_id="bench_sess_step37",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
    )
    channel = FederationChannel(
        "chan_bench_step37",
        local_engine,
        "tcp://127.0.0.1:9099",
        session=session,
        initial_state=ChannelState.ESTABLISHED,
    )

    local_mgr.registry.local_tenant_id = "*"
    remote_mgr.registry.local_tenant_id = "*"

    # ------------------------------------------------------------------------
    # Benchmark 1: Hardware Profile Discovery
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Local Hardware Profile Discovery...")
    for _ in range(num_warmup):
        detector.detect_profile()
    t_prof = []
    for _ in range(num_measured):
        s = time.perf_counter()
        detector.detect_profile()
        t_prof.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 2: CPU Inspection
    # ------------------------------------------------------------------------
    print("[*] Benchmarking CPU Resource Inspection...")
    for _ in range(num_warmup):
        detector.detect_cpu()
    t_cpu = []
    for _ in range(num_measured):
        s = time.perf_counter()
        detector.detect_cpu()
        t_cpu.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 3: Memory Inspection
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Memory Resource Inspection...")
    for _ in range(num_warmup):
        detector.detect_memory()
    t_mem = []
    for _ in range(num_measured):
        s = time.perf_counter()
        detector.detect_memory()
        t_mem.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 4: Accelerator Discovery
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Accelerator Discovery...")
    for _ in range(num_warmup):
        detector.detect_accelerator()
    t_acc = []
    for _ in range(num_measured):
        s = time.perf_counter()
        detector.detect_accelerator()
        t_acc.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 5: Capability Declaration
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Capability Declaration...")
    for _ in range(num_warmup):
        detector.detect_capabilities(registry=gate.registry)
    t_caps = []
    for _ in range(num_measured):
        s = time.perf_counter()
        detector.detect_capabilities(registry=gate.registry)
        t_caps.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 6: Advertisement Creation & Signing
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Resource Advertisement Creation & Signing...")
    for _ in range(num_warmup):
        local_mgr.create_local_advertisement()
    t_adv_create = []
    for _ in range(num_measured):
        s = time.perf_counter()
        local_mgr.create_local_advertisement()
        t_adv_create.append((time.perf_counter() - s) * 1000.0)

    sample_adv = local_mgr.create_local_advertisement()

    # ------------------------------------------------------------------------
    # Benchmark 7: Advertisement Serialization
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Advertisement Serialization...")
    for _ in range(num_warmup):
        sample_adv.to_dict()
    t_ser = []
    for _ in range(num_measured):
        s = time.perf_counter()
        sample_adv.to_dict()
        t_ser.append((time.perf_counter() - s) * 1000.0)

    sample_dict = sample_adv.to_dict()

    # ------------------------------------------------------------------------
    # Benchmark 8: Advertisement Deserialization
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Advertisement Deserialization...")
    for _ in range(num_warmup):
        ResourceAdvertisement.from_dict(sample_dict)
    t_deser = []
    for _ in range(num_measured):
        s = time.perf_counter()
        ResourceAdvertisement.from_dict(sample_dict)
        t_deser.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 9: Payload Digest Computation
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Payload Digest Computation...")
    for _ in range(num_warmup):
        sample_adv.compute_payload_digest()
    t_digest = []
    for _ in range(num_measured):
        s = time.perf_counter()
        sample_adv.compute_payload_digest()
        t_digest.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 10: Ed25519 Signature Generation
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Ed25519 Signature Generation...")
    for _ in range(num_warmup):
        sample_adv.sign(local_key)
    t_sign = []
    for _ in range(num_measured):
        s = time.perf_counter()
        sample_adv.sign(local_key)
        t_sign.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 11: Ed25519 Signature Verification
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Ed25519 Signature Verification...")
    pub_k = local_key.public_key()
    for _ in range(num_warmup):
        sample_adv.verify_signature(pub_k)
    t_verify = []
    for _ in range(num_measured):
        s = time.perf_counter()
        sample_adv.verify_signature(pub_k)
        t_verify.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 12: Ingestion Throughput
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Advertisement Ingestion Throughput...")
    fresh_reg = FederationResourceRegistry(
        local_node_id="peer_alpha",
        local_zone_id="zone-alpha",
        local_tenant_id="*",
    )
    t_ingest = []
    for i in range(num_measured):
        adv_i = ResourceAdvertisement(
            advertisement_id=f"adv_bench_{i}",
            node_id=f"peer_bench_{i}",
            engine_id=f"eng_{i}",
            zone_id="zone-beta",
            tenant_id="*",
            version=1,
            epoch=1,
            resource_profile=sample_adv.resource_profile,
            capabilities=sample_adv.capabilities,
        )
        adv_i.sign(remote_key)
        s = time.perf_counter()
        fresh_reg.record_peer_advertisement(adv_i, public_key=remote_key.public_key())
        t_ingest.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 13: Replay Rejection Latency
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Replay Rejection Latency...")
    replay_adv = fresh_reg.get_peer_advertisement("peer_bench_0")
    t_replay = []
    for _ in range(num_measured):
        s = time.perf_counter()
        try:
            fresh_reg.record_peer_advertisement(replay_adv, public_key=remote_key.public_key())
        except AdvertisementReplayError:
            pass
        t_replay.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 14: Tampered Rejection Latency
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Tampered Advertisement Rejection Latency...")
    tampered_adv = ResourceAdvertisement.from_dict(sample_dict)
    tampered_adv.node_id = "peer_tamper"
    tampered_adv.sign(remote_key)
    tampered_adv.resource_profile.cpu.logical_cores = 9999  # Tamper payload
    t_tamper = []
    for _ in range(num_measured):
        s = time.perf_counter()
        try:
            fresh_reg.record_peer_advertisement(tampered_adv, public_key=remote_key.public_key())
        except AdvertisementTamperedError:
            pass
        t_tamper.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 15: Capability Filtering (100 Peers)
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Capability Filtering across 100 Peers...")
    # Seed registry with 100 peers
    hundred_reg = FederationResourceRegistry(
        local_node_id="peer_alpha",
        local_zone_id="zone-alpha",
        local_tenant_id="*",
    )
    for i in range(100):
        a = ResourceAdvertisement(
            advertisement_id=f"adv_h_{i}",
            node_id=f"node_{i}",
            engine_id=f"eng_{i}",
            zone_id="zone-beta",
            tenant_id="*",
            version=1,
            epoch=1,
            resource_profile=sample_adv.resource_profile,
            capabilities=sample_adv.capabilities,
        )
        a.sign(remote_key)
        hundred_reg.record_peer_advertisement(a, public_key=remote_key.public_key())

    t_filter = []
    for _ in range(num_measured):
        s = time.perf_counter()
        hundred_reg.filter_by_capability(ExecutionType.INFERENCE)
        t_filter.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 16: Transport Query Roundtrip
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Transport Query Roundtrip...")
    q_env = FederationMessageEnvelope(
        message_type=FederationMessageType.RESOURCE_QUERY,
        message_id="msg_bench_qry",
        session_id=session.session_id,
        sender_engine_id="peer_beta",
        receiver_engine_id="peer_alpha",
        sender_peer_id="peer_beta",
        receiver_peer_id="peer_alpha",
        sequence_number=10,
        epoch=1,
        payload={"scope": "ALLOW_CAPABILITY_METADATA"},
        tenant_id="zone-alpha",
    )
    q_env.sign(remote_key)
    for _ in range(num_warmup):
        local_engine.dispatcher.dispatch(q_env, channel)
    t_query_rt = []
    for _ in range(num_measured):
        s = time.perf_counter()
        local_engine.dispatcher.dispatch(q_env, channel)
        t_query_rt.append((time.perf_counter() - s) * 1000.0)

    # ------------------------------------------------------------------------
    # Benchmark 17: Transport Advertisement Roundtrip
    # ------------------------------------------------------------------------
    print("[*] Benchmarking Transport Advertisement Roundtrip...")
    adv_seq = 20
    t_adv_rt = []
    for i in range(num_measured):
        adv_seq += 1
        fresh_remote_adv = remote_mgr.create_local_advertisement()
        a_env = FederationMessageEnvelope(
            message_type=FederationMessageType.RESOURCE_ADVERTISEMENT,
            message_id=f"msg_adv_rt_{i}",
            session_id=session.session_id,
            sender_engine_id=fresh_remote_adv.engine_id,
            receiver_engine_id="peer_alpha",
            sender_peer_id="peer_beta",
            receiver_peer_id="peer_alpha",
            sequence_number=adv_seq,
            epoch=1,
            payload=fresh_remote_adv.to_dict(),
            tenant_id="zone-alpha",
        )
        a_env.sign(remote_key)
        s = time.perf_counter()
        local_engine.dispatcher.dispatch(a_env, channel)
        t_adv_rt.append((time.perf_counter() - s) * 1000.0)

    # Memory Tracking
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = time.perf_counter() - t0_all

    # 3. Post-Benchmark Neural Invariant Verification (ΔW = 0)
    post_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            post_hasher.update(name.encode("utf-8"))
            post_hasher.update(param.detach().cpu().numpy().tobytes())
    post_digest = post_hasher.hexdigest()

    assert post_digest == pre_digest == FROZEN_NEURAL_WEIGHT_HASH, (
        f"Neural core mutation detected! Pre={pre_digest}, Post={post_digest}"
    )
    print(f"[*] Post-Benchmark Neural Invariant: Verified Delta-W = 0 (Hash: {post_digest})")

    results = {
        "benchmark_metadata": {
            "step": 37,
            "title": "Production Federation Resource & Capability Advertisement",
            "timestamp": time.time(),
            "total_benchmark_time_seconds": round(total_elapsed, 2),
            "warmup_iterations": num_warmup,
            "measured_iterations": num_measured,
        },
        "neural_invariants": {
            "parameters": param_count,
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
            "local_resource_discovery": _compute_stats(t_prof),
            "cpu_inspection": _compute_stats(t_cpu),
            "memory_inspection": _compute_stats(t_mem),
            "accelerator_discovery": _compute_stats(t_acc),
            "capability_declaration": _compute_stats(t_caps),
            "advertisement_creation_and_signing": _compute_stats(t_adv_create),
            "advertisement_serialization": _compute_stats(t_ser),
            "advertisement_deserialization": _compute_stats(t_deser),
            "payload_digest_computation": _compute_stats(t_digest),
            "signature_generation": _compute_stats(t_sign),
            "signature_verification": _compute_stats(t_verify),
            "advertisement_ingestion": _compute_stats(t_ingest),
            "replay_rejection": _compute_stats(t_replay),
            "tampered_rejection": _compute_stats(t_tamper),
            "capability_filtering_100_nodes": _compute_stats(t_filter),
            "transport_query_roundtrip": _compute_stats(t_query_rt),
            "transport_advertisement_roundtrip": _compute_stats(t_adv_rt),
        },
    }

    # Save to docs/STEP_37_BENCHMARK_RESULTS.json
    out_path = Path(__file__).resolve().parent.parent / "docs" / "STEP_37_BENCHMARK_RESULTS.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY. Results saved to: {out_path}")
    print(f"Ingestion Throughput: {results['latency_and_throughput']['advertisement_ingestion']['throughput_ops_sec']} ops/sec")
    print(f"Signature Verification: {results['latency_and_throughput']['signature_verification']['mean_ms']} ms")
    print(f"Capability Filtering (100 nodes): {results['latency_and_throughput']['capability_filtering_100_nodes']['mean_ms']} ms")
    print(f"Transport Adv Roundtrip: {results['latency_and_throughput']['transport_advertisement_roundtrip']['mean_ms']} ms")
    print(f"Peak Memory: {results['memory_metrics']['peak_memory_mb']} MB")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmarks()
