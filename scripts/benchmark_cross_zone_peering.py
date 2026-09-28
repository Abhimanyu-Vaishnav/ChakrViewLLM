"""
Empirical Benchmark for ChakrView Step 29:
Cross-Zone Peering & Trust Negotiation.

Measures:
1. Peer identity creation and canonical fingerprinting latency
2. Discovery latency
3. Attestation verification latency
4. Deterministic trust negotiation latency
5. Policy evaluation latency
6. Federation establishment latency
7. Cross-zone request authorization latency (via CapabilityGate)
8. Revocation latency
9. Complete lifecycle latency
10. Peak memory overhead
11. Model weight immutability & frozen invariant verification (3,443,136 params, 4096 vocab, 512 context)
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
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    RiskClassification,
)
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    EXPECTED_PARAMETERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_PAD_ID,
)
from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerAttestation,
    PeerDeclaration,
    FederationScope,
    TrustLevel,
)
from chakrview.cognition.peering.identity import PeerIdentityProvider
from chakrview.cognition.peering.attestation import PeerAttestationVerifier
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.trust import TrustModel
from chakrview.cognition.peering.discovery import PeerDiscoveryManager
from chakrview.cognition.peering.negotiation import TrustNegotiator
from chakrview.cognition.peering.engine import CrossZoneFederationEngine


class BenchmarkEchoCapability(Capability):
    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id="bench_echo",
            name="bench_echo",
            description="Benchmarking echo capability.",
            risk_level=RiskClassification.READ_ONLY,
            required_permissions=["bench.echo"],
        )

    def execute(self, request, context=None) -> CapabilityResult:
        return CapabilityResult(
            request_id=request.request_id,
            capability_id="bench_echo",
            success=True,
            output={"echo": request.parameters.get("val", "")},
        )


def setup_benchmark_environment():
    """Build deterministic test model and capability gate."""
    config = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        max_seq_len=512,
        hidden_dim=512,
    )
    model = ChakrMicro(config)
    model.eval()

    cap_registry = CapabilityRegistry()
    cap_registry.register(BenchmarkEchoCapability())
    gate = CapabilityGate(registry=cap_registry)

    return model, gate


def run_peering_benchmarks(iterations: int = 50) -> Dict[str, Any]:
    print(f"[*] Starting Step 29 Cross-Zone Peering Benchmarks ({iterations} iterations)...")
    tracemalloc.start()

    model, gate = setup_benchmark_environment()
    pre_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)

    policy = CrossZoneFederationPolicy(
        allowed_scopes={
            FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            FederationScope.ALLOW_VERIFICATION,
            FederationScope.ALLOW_MODEL_METADATA,
        }
    )

    # 1. Identity Creation Latency
    t0 = time.perf_counter()
    for i in range(iterations):
        _ = PeerIdentityProvider.create_identity(
            peer_id=f"peer_bench_{i % 10}",
            zone_id="zone_bench",
            organization_id="org_bench",
            capability_profile={"token_ceiling": 512},
            supported_features=["evidence_exchange"],
        )
    t1 = time.perf_counter()
    identity_creation_ms = ((t1 - t0) / iterations) * 1000.0

    # 2. Discovery Latency
    disc_mgr = PeerDiscoveryManager(policy=policy)
    test_identity = PeerIdentityProvider.create_identity("peer_sample", "zone_sample", "org_sample")
    test_attestation = PeerAttestationVerifier.create_attestation("att_s", "peer_sample", "zone_sample")
    test_decl = PeerDeclaration("decl_s", test_identity, test_attestation, {}, [FederationScope.ALLOW_EVIDENCE_EXCHANGE], {}, 1)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = disc_mgr.discover(test_decl, current_epoch=1)
    t1 = time.perf_counter()
    discovery_ms = ((t1 - t0) / iterations) * 1000.0

    # 3. Attestation Verification Latency
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = PeerAttestationVerifier.verify_attestation(test_attestation, "peer_sample", "zone_sample")
    t1 = time.perf_counter()
    attestation_ms = ((t1 - t0) / iterations) * 1000.0

    # 4. Trust Negotiation Latency
    negotiator = TrustNegotiator(local_zone_id="zone_local", policy=policy)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = negotiator.negotiate(test_decl, current_epoch=1)
    t1 = time.perf_counter()
    trust_negotiation_ms = ((t1 - t0) / iterations) * 1000.0

    # 5. Policy Evaluation Latency
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = policy.evaluate_scope_request(test_identity, FederationScope.ALLOW_EVIDENCE_EXCHANGE)
    t1 = time.perf_counter()
    policy_evaluation_ms = ((t1 - t0) / iterations) * 1000.0

    # 6. Federation Establishment Latency
    engine = CrossZoneFederationEngine(local_zone_id="zone_local", policy=policy, capability_gate=gate, model=model)
    t0 = time.perf_counter()
    for i in range(iterations):
        p_id = f"peer_est_{i % 5}"
        p_ident = PeerIdentityProvider.create_identity(p_id, "zone_est", "org_est")
        p_att = PeerAttestationVerifier.create_attestation(f"att_{i}", p_id, "zone_est")
        p_decl = PeerDeclaration(f"decl_{i}", p_ident, p_att, {}, [FederationScope.ALLOW_EVIDENCE_EXCHANGE], {}, 1)
        engine.discover_peer(p_decl)
        _ = engine.negotiate_trust(p_decl)
    t1 = time.perf_counter()
    federation_establishment_ms = ((t1 - t0) / iterations) * 1000.0

    # 7. Request Authorization Latency (Gated)
    cap_req = CapabilityRequest(capability_id="bench_echo", parameters={"val": "ok"}, caller_id="peer_sample")
    cap_ctx = CapabilityContext(granted_permissions={"bench.echo"})
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = engine.authorize_cross_zone_request(
            peer_id="peer_est_0",
            peer_zone_id="zone_est",
            peer_tenant_id="tenant_01",
            target_tenant_id="tenant_01",
            session_id="sess_01",
            requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            payload={"task": "ping"},
            capability_request=cap_req,
            capability_context=cap_ctx,
        )
    t1 = time.perf_counter()
    request_authorization_ms = ((t1 - t0) / iterations) * 1000.0

    # 8. Revocation Latency
    t0 = time.perf_counter()
    for i in range(iterations):
        _ = engine.revoke_peer("peer_est_0", reason="Bench revocation", revoked_by="bench")
    t1 = time.perf_counter()
    revocation_ms = ((t1 - t0) / iterations) * 1000.0

    # 9. Complete Lifecycle Latency
    t0 = time.perf_counter()
    for i in range(iterations):
        life_engine = CrossZoneFederationEngine(local_zone_id="zone_local", policy=policy, capability_gate=gate)
        life_id = f"peer_life_{i}"
        life_ident = PeerIdentityProvider.create_identity(life_id, "zone_life", "org_life")
        life_att = PeerAttestationVerifier.create_attestation(f"att_l_{i}", life_id, "zone_life")
        life_decl = PeerDeclaration(f"decl_l_{i}", life_ident, life_att, {}, [FederationScope.ALLOW_EVIDENCE_EXCHANGE], {}, 1)

        life_engine.discover_peer(life_decl)
        life_engine.attest_peer(life_id, life_att)
        life_agr = life_engine.negotiate_trust(life_decl)
        life_engine.authorize_cross_zone_request(
            peer_id=life_id,
            peer_zone_id="zone_life",
            peer_tenant_id="tenant_life",
            target_tenant_id="tenant_life",
            session_id="sess_life",
            requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            payload={"task": "bench_test"},
            capability_request=cap_req,
            capability_context=cap_ctx,
        )
        life_engine.revoke_peer(life_id, reason="Completed lifecycle", revoked_by="admin")
    t1 = time.perf_counter()
    complete_lifecycle_ms = ((t1 - t0) / iterations) * 1000.0

    # Memory Tracking
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = round(peak_mem / (1024 * 1024), 2)

    # Invariants Verification
    post_hash = CoreIntegrityGuard.compute_weight_fingerprint(model)
    weight_mutation = (pre_hash != post_hash)

    total_params = sum(p.numel() for p in model.parameters())

    results = {
        "step": "Step 29 — Cross-Zone Peering & Trust Negotiation",
        "iterations": iterations,
        "latencies_ms": {
            "identity_creation_ms": round(identity_creation_ms, 3),
            "discovery_ms": round(discovery_ms, 3),
            "attestation_ms": round(attestation_ms, 3),
            "trust_negotiation_ms": round(trust_negotiation_ms, 3),
            "policy_evaluation_ms": round(policy_evaluation_ms, 3),
            "federation_establishment_ms": round(federation_establishment_ms, 3),
            "request_authorization_ms": round(request_authorization_ms, 3),
            "revocation_ms": round(revocation_ms, 3),
            "complete_lifecycle_ms": round(complete_lifecycle_ms, 3),
        },
        "memory": {
            "peak_memory_mb": peak_memory_mb,
        },
        "invariants": {
            "chakrmicro_parameters": total_params,
            "vocabulary_size": model.config.vocab_size,
            "context_window": model.config.max_seq_len,
            "model_pre_hash": pre_hash,
            "model_post_hash": post_hash,
            "weight_mutation_detected": weight_mutation,
        },
    }

    return results


if __name__ == "__main__":
    results = run_peering_benchmarks(iterations=50)
    print("\n--- Step 29 Cross-Zone Peering Benchmark Results ---")
    print(json.dumps(results, indent=2))

    output_path = Path(__file__).resolve().parent.parent / "docs" / "STEP_29_BENCHMARK_RESULTS.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results successfully written to: {output_path}")
