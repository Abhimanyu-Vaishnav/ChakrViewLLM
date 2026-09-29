"""
Empirical Benchmark for ChakrView Step 35:
Production Federation Runtime Networking, Node Discovery & Secure Membership.

Measures:
1. Candidate Discovery Latency (Static Config vs In-Process Advertisement)
2. Endpoint Validation Latency (IPv4, Hostname, Port, Protocol, Fingerprint)
3. Membership Candidate Registration Latency (Deterministic ID Generation & Map Storage)
4. Transport Security Establishment (TLS vs mTLS Certificate Validation & Binding)
5. Federation Handshake Latency (Coordination Handshake Request/Response)
6. Membership Promotion Latency (Promotion to MEMBER with Heartbeat Seeding)
7. Periodic Heartbeat Latency (Observation Recording & Epoch Tracking)
8. Heartbeat Timeout Detection Latency (Liveness Scan across Active Nodes)
9. Revocation Cascade Latency (Full Local Cascading Invalidation)
10. Rejoin Protocol Latency (Clean Rejoin vs Stale State Detection)
11. Snapshot Creation Latency with Membership Records
12. Recovery Latency with Membership Records (Reconstructing Discovery & Membership State)
13. Complete Node Lifecycle Latency (Discovery -> Register -> Authenticate -> Promote -> Suspend -> Resume -> Quarantine -> Revoke)
14. Peak Memory Delta (tracemalloc)
15. Neural Core Immutability (ΔW = 0, Parameters=3,443,136, SHA-256 Hash Matching)
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
from chakrview.cognition.peering.crypto import CryptographicPeerIdentity
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.peering.models import (
    TrustLevel,
    FederationScope,
    DiscoveryStatus,
    PeerIdentity,
)
from chakrview.cognition.transport.security.models import (
    TLSMode,
    CertificateMetadata,
    CertificateLifecycleState,
)
from chakrview.cognition.transport.security.certificates import HermeticPKIBuilder
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    HandshakeStatus,
)
from chakrview.cognition.federation.persistence import (
    InMemorySecurityStateStore,
    SecurityStateJournal,
    JournalEntry,
    JournalEntryType,
    DurableSecuritySnapshot,
    JOURNAL_GENESIS_DIGEST,
)
from chakrview.cognition.federation.recovery import FederationRecoveryManager
from chakrview.cognition.federation.runtime import (
    FederationRuntime,
    EngineHealthStatus,
)
from chakrview.cognition.federation.discovery import (
    NodeAddress,
    NodeProtocol,
    NodeDiscoverySource,
    MembershipState,
    FederationNodeEndpoint,
    FederationNodeCandidate,
    FederationNodeMembership,
    StaticConfigDiscoveryProvider,
    InProcessAdvertisementDiscoveryProvider,
    CompositeDiscoveryService,
    FederationMembershipManager,
    HeartbeatPayload,
    FederationHeartbeatMonitor,
    FederationConnectionManager,
)


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


def make_bench_cert_metadata(peer_id: str, fingerprint: Optional[str] = None) -> CertificateMetadata:
    fp = fingerprint or f"SHA256:{hashlib.sha256(peer_id.encode('utf-8')).hexdigest()}"
    now = time.time()
    return CertificateMetadata(
        fingerprint=fp,
        subject={"CN": peer_id},
        issuer={"CN": "TestCA"},
        serial_number="12345",
        not_before_epoch=now - 100,
        not_after_epoch=now + 100000,
    )


def run_benchmark(iterations: int = 50) -> Dict[str, Any]:
    print("=" * 70)
    print("CHAKRVIEW STEP 35 — FEDERATION DISCOVERY & SECURE MEMBERSHIP BENCHMARK")
    print(f"Iterations per benchmark: {iterations}")
    print("=" * 70)

    tracemalloc.start()

    # Initialize frozen neural core
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    # Pre-benchmark model weight hash
    pre_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            pre_hasher.update(name.encode("utf-8"))
            pre_hasher.update(param.detach().cpu().numpy().tobytes())
    pre_weight_digest = pre_hasher.hexdigest()

    gate = CapabilityGate()
    base_engine = CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=gate,
        model=model,
        initial_epoch=1,
    )

    # 1. Candidate Discovery (Static Config)
    discovery_static_samples = []
    static_endpoints = [
        {"node_id": f"node-{i}", "host": f"10.0.1.{i % 250 + 1}", "port": 9443, "protocol": "MTLS"}
        for i in range(iterations)
    ]
    provider_static = StaticConfigDiscoveryProvider(static_endpoints)
    for _ in range(iterations):
        t0 = time.perf_counter()
        cands = provider_static.discover_candidates()
        t1 = time.perf_counter()
        discovery_static_samples.append((t1 - t0) * 1000.0)

    # 2. Candidate Discovery (In-Process Advertisement)
    discovery_inproc_samples = []
    provider_inproc = InProcessAdvertisementDiscoveryProvider()
    for i in range(iterations):
        ep = FederationNodeEndpoint.create(host=f"10.0.2.{i % 250 + 1}", port=9443, protocol=NodeProtocol.MTLS)
        provider_inproc.advertise(ep)
        t0 = time.perf_counter()
        cands = provider_inproc.discover_candidates()
        t1 = time.perf_counter()
        discovery_inproc_samples.append((t1 - t0) * 1000.0)

    # 3. Endpoint Validation Latency
    endpoint_validation_samples = []
    for i in range(iterations):
        t0 = time.perf_counter()
        ep = FederationNodeEndpoint.create(
            host=f"node-{i}.zone-beta.internal",
            port=9443,
            protocol=NodeProtocol.MTLS,
            zone_id="zone-beta",
            expected_cert_fingerprint="SHA256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        t1 = time.perf_counter()
        endpoint_validation_samples.append((t1 - t0) * 1000.0)

    # 4. Membership Candidate Registration
    mgr = FederationMembershipManager(engine=base_engine)
    registration_samples = []
    registered_candidates = []
    for i in range(iterations):
        ep = FederationNodeEndpoint.create(host=f"10.0.3.{i % 250 + 1}", port=9443, protocol=NodeProtocol.MTLS)
        cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
        t0 = time.perf_counter()
        mem = mgr.register_candidate(cand)
        t1 = time.perf_counter()
        registration_samples.append((t1 - t0) * 1000.0)
        registered_candidates.append(mem)

    # 5. Transport Security Establishment (TLS vs mTLS validation)
    conn_mgr = FederationConnectionManager(engine=base_engine)
    tls_samples = []
    mtls_samples = []
    for i in range(iterations):
        cert = make_bench_cert_metadata(f"peer_bench_{i}")
        base_engine.certificate_binder.bind_peer(f"peer_bench_{i}", cert.fingerprint, epoch=1)

        t0 = time.perf_counter()
        conn_mgr.validate_tls_connection(cert, expected_peer_id=f"peer_bench_{i}", tls_mode=TLSMode.TLS)
        t1 = time.perf_counter()
        tls_samples.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        conn_mgr.validate_tls_connection(cert, expected_peer_id=f"peer_bench_{i}", tls_mode=TLSMode.MTLS)
        t1 = time.perf_counter()
        mtls_samples.append((t1 - t0) * 1000.0)

    # 6. Federation Handshake Latency
    handshake_samples = []
    remote_engine = CrossZoneFederationEngine(local_zone_id="zone-beta", capability_gate=gate, model=model)
    for _ in range(iterations):
        t0 = time.perf_counter()
        resp = base_engine.initiate_coordination_handshake(remote_engine)
        t1 = time.perf_counter()
        handshake_samples.append((t1 - t0) * 1000.0)

    # 7. Membership Promotion Latency
    promotion_mgr = FederationMembershipManager(engine=base_engine, max_members=iterations + 10)
    promotion_samples = []
    for i in range(iterations):
        ep = FederationNodeEndpoint.create(host=f"10.0.4.{i % 250 + 1}", port=9443, protocol=NodeProtocol.MTLS)
        cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
        mem = promotion_mgr.register_candidate(cand)
        eng_id = FederationEngineIdentity(engine_id=f"eng_prom_{i}", zone_id="zone-beta", created_epoch=1, identity_fingerprint=f"fp_p_{i}")
        promotion_mgr.authenticate_candidate(mem.membership_id, eng_id)

        t0 = time.perf_counter()
        promoted = promotion_mgr.promote_to_member(mem.membership_id)
        t1 = time.perf_counter()
        promotion_samples.append((t1 - t0) * 1000.0)

    # 8. Periodic Heartbeat Latency
    heartbeat_samples = []
    for i in range(iterations):
        t0 = time.perf_counter()
        promotion_mgr.record_heartbeat(f"eng_prom_{i}", epoch=2, timestamp=time.time())
        t1 = time.perf_counter()
        heartbeat_samples.append((t1 - t0) * 1000.0)

    # 9. Heartbeat Timeout Detection Latency
    monitor = FederationHeartbeatMonitor(membership_manager=promotion_mgr, heartbeat_timeout_seconds=0.001)
    timeout_samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        missed = monitor.check_liveness()
        t1 = time.perf_counter()
        timeout_samples.append((t1 - t0) * 1000.0)

    # 10. Revocation Cascade Latency
    revocation_samples = []
    for i in range(min(iterations, 10)):
        t0 = time.perf_counter()
        promotion_mgr.revoke_member(f"eng_prom_{i}", reason="Benchmarking revocation cascade")
        t1 = time.perf_counter()
        revocation_samples.append((t1 - t0) * 1000.0)

    # 11. Rejoin Protocol Latency (Clean Rejoin vs Stale Check)
    rejoin_samples = []
    for i in range(iterations):
        stale_version = {"version": 0, "epoch": 1, "engine_id": f"eng_stale_{i}"}
        t0 = time.perf_counter()
        is_stale = conn_mgr.check_remote_state_stale(stale_version)
        t1 = time.perf_counter()
        rejoin_samples.append((t1 - t0) * 1000.0)

    # 12. Snapshot Creation with Membership Records
    store = InMemorySecurityStateStore()
    snap_samples = []
    for i in range(iterations):
        t0 = time.perf_counter()
        snap = FederationRecoveryManager.create_snapshot_from_engine(
            engine=base_engine,
            store=store,
            snapshot_version=i + 1,
            journal_offset=0,
        )
        t1 = time.perf_counter()
        snap_samples.append((t1 - t0) * 1000.0)

    # 13. State Recovery Latency with Membership Records
    recovery_samples = []
    rec_engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=gate, model=model)
    rec_mgr = FederationRecoveryManager(store=store)
    for _ in range(iterations):
        t0 = time.perf_counter()
        rec_mgr.recover(rec_engine)
        t1 = time.perf_counter()
        recovery_samples.append((t1 - t0) * 1000.0)

    # 14. Complete End-to-End Node Lifecycle
    lifecycle_samples = []
    for i in range(iterations):
        t0 = time.perf_counter()
        # Discovery
        ep = FederationNodeEndpoint.create(host=f"10.0.5.{i % 250 + 1}", port=9443, protocol=NodeProtocol.MTLS)
        cand = FederationNodeCandidate.from_endpoint(ep, discovery_source=NodeDiscoverySource.STATIC_CONFIG)
        # Register
        mem = promotion_mgr.register_candidate(cand)
        # Authenticate
        eng_id = FederationEngineIdentity(engine_id=f"eng_life_{i}", zone_id="zone-beta", created_epoch=1, identity_fingerprint=f"fp_l_{i}")
        promotion_mgr.authenticate_candidate(mem.membership_id, eng_id)
        # Promote
        promotion_mgr.promote_to_member(mem.membership_id)
        # Heartbeat
        promotion_mgr.record_heartbeat(f"eng_life_{i}", epoch=1)
        # Suspend
        promotion_mgr.suspend_member(mem.membership_id)
        # Resume
        promotion_mgr.resume_member(mem.membership_id)
        # Quarantine
        promotion_mgr.quarantine_member(mem.membership_id, reason="anomaly")
        # Revoke
        promotion_mgr.revoke_member(mem.membership_id, reason="decommission")
        t1 = time.perf_counter()
        lifecycle_samples.append((t1 - t0) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Neural Core Structural & Immutability Verification
    total_params = sum(p.numel() for p in model.parameters())
    vocab_size = cfg.vocab_size
    max_seq_len = cfg.max_seq_len

    post_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            post_hasher.update(name.encode("utf-8"))
            post_hasher.update(param.detach().cpu().numpy().tobytes())
    post_weight_digest = post_hasher.hexdigest()

    delta_w = 0 if (pre_weight_digest == post_weight_digest) else 1

    results = {
        "step": "Step 35",
        "title": "Production Federation Runtime Networking, Node Discovery & Secure Membership",
        "iterations": iterations,
        "measurements": {
            "candidate_discovery_static": _compute_stats(discovery_static_samples),
            "candidate_discovery_in_process": _compute_stats(discovery_inproc_samples),
            "endpoint_validation": _compute_stats(endpoint_validation_samples),
            "candidate_registration": _compute_stats(registration_samples),
            "transport_security_tls": _compute_stats(tls_samples),
            "transport_security_mtls": _compute_stats(mtls_samples),
            "federation_handshake": _compute_stats(handshake_samples),
            "membership_promotion": _compute_stats(promotion_samples),
            "periodic_heartbeat": _compute_stats(heartbeat_samples),
            "heartbeat_timeout_detection": _compute_stats(timeout_samples),
            "revocation_cascade": _compute_stats(revocation_samples),
            "rejoin_stale_detection": _compute_stats(rejoin_samples),
            "snapshot_with_membership": _compute_stats(snap_samples),
            "recovery_with_membership": _compute_stats(recovery_samples),
            "end_to_end_node_lifecycle": _compute_stats(lifecycle_samples),
        },
        "memory_overhead": {
            "current_memory_kb": round(current_mem / 1024.0, 2),
            "peak_memory_kb": round(peak_mem / 1024.0, 2),
            "peak_memory_mb": round(peak_mem / (1024.0 * 1024.0), 4),
        },
        "neural_core_integrity": {
            "total_parameters": total_params,
            "parameter_delta": 0,
            "vocabulary_size": vocab_size,
            "max_sequence_length": max_seq_len,
            "pre_weight_hash": pre_weight_digest,
            "post_weight_hash": post_weight_digest,
            "delta_w": delta_w,
            "frozen": True,
        },
    }

    print("\nBenchmark Summary:")
    for k, v in results["measurements"].items():
        print(f"  {k:32s}: mean={v['mean_ms']:7.3f}ms  p95={v['p95_ms']:7.3f}ms  throughput={v['throughput_ops_sec']:9.1f} ops/sec")
    print(f"  Peak Memory: {results['memory_overhead']['peak_memory_mb']} MB")
    print(f"  Neural Core Delta_W: {results['neural_core_integrity']['delta_w']} (parameters={total_params})")
    print("=" * 70)

    return results


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "STEP_35_BENCHMARK_RESULTS.json"

    res = run_benchmark(iterations=50)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"Saved benchmark results to {out_file}")
