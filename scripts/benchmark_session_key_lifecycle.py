"""
Empirical Benchmark for ChakrView Step 32:
Secure Federation Session & Key Lifecycle Hardening.

Measures:
1. Session Creation Latency
2. Cryptographic Authentication & Activation Latency
3. Session Freshness & Renewal Latency
4. Session Termination Latency
5. Peer Ed25519 Key Rotation Latency
6. TLS Certificate Rotation & Binding Latency
7. Message Replay & Sequence Verification Latency
8. Revocation Cascade Latency
9. Complete Secure Federation Lifecycle Latency
10. Peak Memory Overhead (tracemalloc)
11. Neural Core Immutability (ΔW = 0) and Structural Invariants
"""

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
from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    RiskClassification,
)
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
)
from chakrview.cognition.peering.session import (
    SecurePeerSession,
    SessionStatus,
    SessionKeyState,
    SessionKeyMetadata,
)
from chakrview.cognition.peering.models import (
    PeerIdentity,
    TrustGrant,
    TrustLevel,
    FederationScope,
)
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.transport.models import WireEnvelope, MessageType
from chakrview.cognition.transport.security import (
    HermeticPKIBuilder,
    extract_certificate_metadata,
)


class BenchmarkEchoCapability(Capability):
    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id="bench.echo",
            name="Benchmark Echo",
            description="Safe benchmark capability",
            risk_level=RiskClassification.READ_ONLY,
            required_permissions=[],
        )

    def execute(self, request: CapabilityRequest, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult(
            request_id=request.request_id,
            capability_id="bench.echo",
            success=True,
            output={"echo": request.parameters.get("val", 0)},
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


def run_benchmark(iterations: int = 200) -> Dict[str, Any]:
    print("=" * 72)
    print("  CHAKRVIEW STEP 32: SECURE FEDERATION SESSION & KEY LIFECYCLE BENCHMARK")
    print("=" * 72)

    tracemalloc.start()

    # 1. Setup Neural Core & Invariant Verification
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    integrity_pre = CoreIntegrityGuard.verify_model(model)
    assert integrity_pre.passed is True
    pre_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)

    gate = CapabilityGate()
    gate.registry.register(BenchmarkEchoCapability())

    policy = CrossZoneFederationPolicy(
        allowed_zones={"zone-benchmark-remote"},
        allowed_scopes={FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION},
    )

    engine = CrossZoneFederationEngine(
        local_zone_id="zone-benchmark-local",
        capability_gate=gate,
        policy=policy,
        model=model,
        initial_epoch=1,
    )

    # Setup remote peer
    remote_key = Ed25519PrivateKeyWrapper.generate()
    remote_pub = remote_key.public_key()
    remote_crypto = CryptographicPeerIdentity.create(
        zone_id="zone-benchmark-remote",
        organization_id="org-benchmark",
        public_key=remote_pub,
        peer_id="peer-bench-01",
        created_epoch=1,
        ttl_epochs=1000,
    )
    engine.register_cryptographic_peer(remote_crypto)

    grant = TrustGrant(
        grant_id="grant_bench_001",
        issuer_zone_id="zone-benchmark-local",
        subject_peer_id="peer-bench-01",
        subject_zone_id="zone-benchmark-remote",
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION],
        issued_epoch=1,
        expires_epoch=5000,
    )
    engine.registry.update_trust_grant("peer-bench-01", grant)

    # 1. Session Creation Latency
    session_create_samples = []
    created_sessions = []
    for i in range(iterations):
        t0 = time.perf_counter()
        sess = SecurePeerSession(
            session_id=f"sess_bench_{i}",
            local_peer_id=engine.local_peer_id,
            remote_peer_id="peer-bench-01",
            local_zone_id=engine.local_zone_id,
            remote_zone_id="zone-benchmark-remote",
            created_epoch=engine.current_epoch,
            expires_at_epoch=engine.current_epoch + 100,
        )
        t1 = time.perf_counter()
        session_create_samples.append((t1 - t0) * 1000.0)
        created_sessions.append(sess)

    # 2. Authentication & Activation Latency
    auth_samples = []
    for sess in created_sessions:
        t0 = time.perf_counter()
        sess.mark_authenticated(remote_pub, authenticated_epoch=engine.current_epoch)
        t1 = time.perf_counter()
        auth_samples.append((t1 - t0) * 1000.0)

    # Register one session in engine for wire execution
    active_sess = created_sessions[0]
    engine.sessions[active_sess.session_id] = active_sess

    # 3. Session Renewal Latency
    renewal_samples = []
    for _ in range(iterations):
        # Create dedicated renewing session with room for renewal
        r_sess = SecurePeerSession(
            session_id=f"sess_renew_{_}",
            local_peer_id=engine.local_peer_id,
            remote_peer_id="peer-bench-01",
            local_zone_id=engine.local_zone_id,
            remote_zone_id="zone-benchmark-remote",
            created_epoch=engine.current_epoch,
            expires_at_epoch=engine.current_epoch + 50,
        )
        r_sess.mark_authenticated(remote_pub, authenticated_epoch=engine.current_epoch)
        engine.sessions[r_sess.session_id] = r_sess

        t0 = time.perf_counter()
        engine.renew_session(r_sess.session_id, extension_epochs=20)
        t1 = time.perf_counter()
        renewal_samples.append((t1 - t0) * 1000.0)

    # 4. Session Termination Latency
    term_samples = []
    for i in range(iterations):
        t_sess = SecurePeerSession(
            session_id=f"sess_term_{i}",
            local_peer_id=engine.local_peer_id,
            remote_peer_id="peer-bench-01",
            local_zone_id=engine.local_zone_id,
            remote_zone_id="zone-benchmark-remote",
            created_epoch=engine.current_epoch,
            expires_at_epoch=engine.current_epoch + 50,
        )
        t_sess.mark_authenticated(remote_pub, authenticated_epoch=engine.current_epoch)
        engine.sessions[t_sess.session_id] = t_sess

        t0 = time.perf_counter()
        engine.terminate_session(t_sess.session_id, reason="Benchmark termination")
        t1 = time.perf_counter()
        term_samples.append((t1 - t0) * 1000.0)

    # 5. Peer Ed25519 Key Rotation Latency
    key_rot_samples = []
    curr_priv = remote_key
    curr_pub = remote_pub
    for _ in range(iterations):
        new_priv = Ed25519PrivateKeyWrapper.generate()
        new_pub = new_priv.public_key()
        proof_payload = f"ROTATE_KEY:{curr_pub.fingerprint}:{new_pub.fingerprint}:{engine.current_epoch}".encode("utf-8")
        proof_sig = curr_priv.sign_hex(proof_payload)

        t0 = time.perf_counter()
        engine.rotate_peer_key("peer-bench-01", new_pub, proof_sig)
        t1 = time.perf_counter()
        key_rot_samples.append((t1 - t0) * 1000.0)
        curr_priv = new_priv
        curr_pub = new_pub

    # 6. Certificate Rotation & Binding Latency
    ca_pem, ca_key_pem, ca_cert, ca_key = HermeticPKIBuilder.create_ca(common_name="Bench CA")
    cert_rot_samples = []
    for i in range(min(iterations, 50)):  # Cert generation is heavier
        _, _, s_cert, _ = HermeticPKIBuilder.create_server_cert(ca_cert, ca_key, common_name="beta.example.com", san_dns=["beta.example.com"])
        meta = extract_certificate_metadata(s_cert)
        t0 = time.perf_counter()
        engine.rotate_peer_certificate("peer-bench-01", meta, expected_common_name="beta.example.com")
        t1 = time.perf_counter()
        cert_rot_samples.append((t1 - t0) * 1000.0)

    # 7. Replay & Sequence Monotonicity Verification Latency
    replay_samples = []
    for i in range(iterations):
        t0 = time.perf_counter()
        fresh = active_sess.record_and_check_message_id(f"msg_fresh_{i}")
        seq_ok = active_sess.record_and_check_sequence(i + 1)
        t1 = time.perf_counter()
        replay_samples.append((t1 - t0) * 1000.0)
        assert fresh is True
        assert seq_ok is True

    # 8. Revocation Cascade Latency
    cascade_samples = []
    for i in range(min(iterations, 12)):
        # Setup temporary peer
        t_key = Ed25519PrivateKeyWrapper.generate()
        t_pub = t_key.public_key()
        t_peer_id = f"peer_temp_{i}"
        t_crypto = CryptographicPeerIdentity.create(
            zone_id="zone-benchmark-remote",
            organization_id="org-benchmark",
            public_key=t_pub,
            peer_id=t_peer_id,
            created_epoch=1,
            ttl_epochs=1000,
        )
        engine.register_cryptographic_peer(t_crypto)
        s1 = engine.create_secure_session(t_peer_id, ttl_epochs=50)
        s2 = engine.create_secure_session(t_peer_id, ttl_epochs=50)

        t0 = time.perf_counter()
        engine.revoke_peer(t_peer_id, reason="Cascade benchmark test", revoked_by="bench")
        t1 = time.perf_counter()
        cascade_samples.append((t1 - t0) * 1000.0)

    # 9. Complete Secure Federation Wire Lifecycle Latency
    lifecycle_samples = []
    for i in range(iterations):
        env = WireEnvelope(
            protocol_version="31.0",
            message_type=MessageType.CAPABILITY_REQUEST,
            message_id=f"msg_e2e_{i}",
            session_id=active_sess.session_id,
            sender_peer_id="peer-bench-01",
            receiver_peer_id=engine.local_peer_id,
            created_epoch=engine.current_epoch,
            expires_epoch=engine.current_epoch + 10,
            payload={
                "requested_scope": FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value,
                "data": {"task": "echo", "val": i},
                "capability_request": {"capability_id": "bench.echo", "parameters": {"val": i}},
                "sequence_number": 1000 + i,
            },
        ).sign(curr_priv)

        t0 = time.perf_counter()
        resp = engine.authorize_and_execute_wire_envelope(env)
        t1 = time.perf_counter()
        lifecycle_samples.append((t1 - t0) * 1000.0)
        assert resp.payload.get("authorized") is True

    # Memory Tracking
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(peak_mem / (1024 * 1024), 3)

    # Neural Core Immutability Check
    integrity_post = CoreIntegrityGuard.verify_model(model)
    post_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)
    delta_w = 0 if pre_hash == post_hash else 1

    results = {
        "step": 32,
        "title": "Secure Federation Session & Key Lifecycle Hardening Benchmark",
        "iterations": iterations,
        "session_creation": _compute_stats(session_create_samples),
        "authentication_activation": _compute_stats(auth_samples),
        "session_renewal": _compute_stats(renewal_samples),
        "session_termination": _compute_stats(term_samples),
        "key_rotation": _compute_stats(key_rot_samples),
        "certificate_rotation": _compute_stats(cert_rot_samples),
        "replay_sequence_verification": _compute_stats(replay_samples),
        "revocation_cascade": _compute_stats(cascade_samples),
        "complete_secure_lifecycle": _compute_stats(lifecycle_samples),
        "memory_overhead": {
            "current_mem_mb": round(current_mem / (1024 * 1024), 3),
            "peak_mem_mb": peak_mem_mb,
        },
        "neural_core_invariants": {
            "parameters": integrity_post.parameter_count,
            "vocab_size": integrity_post.vocab_size,
            "max_seq_len": integrity_post.max_seq_len,
            "pre_hash": pre_hash,
            "post_hash": post_hash,
            "delta_w": delta_w,
            "verified": integrity_post.passed and (delta_w == 0),
        },
    }

    # Print summary table
    print("\nBenchmark Results Summary:")
    print(f"{'Operation':<35} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Throughput (ops/s)':<18}")
    print("-" * 80)
    for op, stats in [
        ("Session Creation", results["session_creation"]),
        ("Auth & Activation", results["authentication_activation"]),
        ("Session Renewal", results["session_renewal"]),
        ("Session Termination", results["session_termination"]),
        ("Peer Key Rotation", results["key_rotation"]),
        ("Certificate Rotation", results["certificate_rotation"]),
        ("Replay & Sequence Check", results["replay_sequence_verification"]),
        ("Revocation Cascade", results["revocation_cascade"]),
        ("Complete Wire Lifecycle", results["complete_secure_lifecycle"]),
    ]:
        print(f"{op:<35} | {stats['mean_ms']:<10.4f} | {stats['p95_ms']:<10.4f} | {stats['throughput_ops_sec']:<18.2f}")
    print("-" * 80)
    print(f"Peak Memory Overhead: {peak_mem_mb} MB")
    print(f"Neural Core Immutability (Delta_W = 0): {'VERIFIED' if delta_w == 0 else 'FAILED'}")
    print(f"Parameters: {integrity_post.parameter_count:,} (Expected: 3,443,136)")
    print(f"Pre-Hash:  {pre_hash}")
    print(f"Post-Hash: {post_hash}")
    print("=" * 72)

    out_file = Path(__file__).resolve().parent.parent / "docs" / "STEP_32_BENCHMARK_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_file}")

    return results


if __name__ == "__main__":
    run_benchmark(iterations=200)
