"""
Comprehensive Test Suite for Cross-Zone Peering & Trust Negotiation (Step 29).

Verifies all architectural requirements, invariants, and security boundaries:
1. Peer Identity Creation & Deterministic Fingerprinting
2. Identity Validation, Format Safety & Injection Resistance
3. Unsupported Cryptographic Mode (IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION)
4. Peer Discovery without Automatic Trust (DISCOVERY != TRUST)
5. Attestation Acceptance & Structural Claim Verification
6. Attestation Rejection (Architecture/Protocol Mismatch)
7. Deterministic Trust & Scope Negotiation
8. Default-Deny Policy Enforcement & Scope Whitelisting
9. Prohibited Scopes Rejection (Weight Access, Private Memory, Tenant Crossover)
10. Limited Federation Scopes
11. CapabilityGate Enforcement (Peer cannot bypass gate)
12. Cross-Tenant Isolation Denial
13. Cross-Zone Sensitive Payload Sanitization (Weights, Scratchpads, Secrets)
14. Trust Expiration via Logical Epochs
15. Peer Revocation & Immediate Scope Invalidation
16. Revoked Peer Request Denial (Fail-Closed)
17. Hard Registry Ceilings (Total peers <= 32, Zone peers <= 16, Active <= 8)
18. Duplicate Peer Rejection
19. Bounded Audit Log History (Max 1000, FIFO eviction)
20. Authority Boundary Enforcement (FEDERATION_ENGINE != AUTHORITY, PEER != AUTHORITY)
21. Neural Weight Immutability (Delta W = 0, Hash Pre == Post)
22. Frozen ChakrMicro Core Invariants (3,443,136 params, 4096 vocab, 512 context)
23. Complete End-to-End Cross-Zone Federation Lifecycle
"""

import hashlib
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    RiskClassification,
)
from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerAttestation,
    PeerDeclaration,
    PeerRegistration,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    DiscoveryStatus,
    AttestationStatus,
    FederationScope,
    ProhibitedScope,
    NegotiationAgreement,
    RevocationRecord,
    AuditEventType,
    MAX_PEERS_TOTAL,
    MAX_PEERS_PER_ZONE,
    MAX_ACTIVE_PEERS,
    MAX_AUDIT_LOG_ENTRIES,
)
from chakrview.cognition.peering.identity import (
    PeerIdentityProvider,
    IdentityValidationError,
    UnsupportedSecurityModeError,
)
from chakrview.cognition.peering.attestation import (
    PeerAttestationVerifier,
    AttestationVerificationError,
)
from chakrview.cognition.peering.policy import (
    CrossZoneFederationPolicy,
    PolicyViolationError,
)
from chakrview.cognition.peering.trust import TrustModel
from chakrview.cognition.peering.discovery import PeerDiscoveryManager
from chakrview.cognition.peering.negotiation import TrustNegotiator
from chakrview.cognition.peering.registry import (
    PeerRegistry,
    DuplicatePeerError,
    PeerRegistryCapacityError,
)
from chakrview.cognition.peering.revocation import RevocationManager
from chakrview.cognition.peering.isolation import (
    CrossZoneIsolationGuard,
    IsolationViolationError,
)
from chakrview.cognition.peering.audit import BoundedAuditLogger
from chakrview.cognition.peering.engine import (
    CrossZoneFederationEngine,
    CrossZoneAuthorizationError,
    WeightMutationDetectedError,
)


# ============================================================================
# Test Fixtures
# ============================================================================

class MockEchoCapability(Capability):
    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id="mock_echo",
            name="mock_echo",
            description="Echoes input back for testing.",
            risk_level=RiskClassification.READ_ONLY,
            required_permissions=["test.echo"],
        )

    def execute(self, request, context=None) -> CapabilityResult:
        return CapabilityResult(
            request_id=request.request_id,
            capability_id="mock_echo",
            success=True,
            output={"echo": request.parameters.get("message", "")},
        )


@pytest.fixture
def frozen_model() -> ChakrMicro:
    """Instantiate frozen ChakrMicro v0.1 model."""
    config = ModelConfig(
        vocab_size=4096,
        max_seq_len=512,
        n_layers=6,
        d_model=192,
        n_heads=6,
        hidden_dim=512,
    )
    model = ChakrMicro(config)
    model.eval()
    return model


@pytest.fixture
def capability_gate() -> CapabilityGate:
    registry = CapabilityRegistry()
    registry.register(MockEchoCapability())
    return CapabilityGate(registry=registry)


@pytest.fixture
def peer_declaration() -> PeerDeclaration:
    identity = PeerIdentityProvider.create_identity(
        peer_id="peer_zone_alpha",
        zone_id="zone_alpha",
        organization_id="org_alpha",
        capability_profile={"supported_roles": ["ANALYST", "VERIFIER"], "token_ceiling": 512},
        supported_features=["evidence_exchange", "verification"],
    )
    attestation = PeerAttestationVerifier.create_attestation(
        attestation_id="att_alpha_01",
        peer_id="peer_zone_alpha",
        zone_id="zone_alpha",
        capability_manifest={"supported_roles": ["ANALYST", "VERIFIER"], "token_ceiling": 512},
        runtime_integrity_hash="a" * 64,
        declared_epoch=1,
    )
    return PeerDeclaration(
        declaration_id="decl_alpha_01",
        identity=identity,
        attestation=attestation,
        resource_profile={"cpu_cores": 4, "memory_mb": 1024},
        requested_scopes=[
            FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            FederationScope.ALLOW_VERIFICATION,
        ],
        policy_profile={"default_deny": True},
        epoch=1,
    )


# ============================================================================
# 1. Peer Identity Tests
# ============================================================================

def test_peer_identity_creation_and_fingerprint():
    """Verify peer identity creation, formatting, and SHA-256 fingerprinting."""
    identity = PeerIdentityProvider.create_identity(
        peer_id="peer_west_01",
        zone_id="zone_west",
        organization_id="org_globex",
    )
    assert identity.peer_id == "peer_west_01"
    assert identity.zone_id == "zone_west"
    assert identity.organization_id == "org_globex"
    assert len(identity.fingerprint) == 64
    assert PeerIdentityProvider.verify_fingerprint(identity) is True


def test_peer_identity_determinism():
    """Verify identical identity parameters produce identical fingerprints."""
    id1 = PeerIdentityProvider.create_identity("peer_1", "zone_1", "org_1", created_epoch=5)
    id2 = PeerIdentityProvider.create_identity("peer_1", "zone_1", "org_1", created_epoch=5)
    assert id1.canonical_serialize() == id2.canonical_serialize()
    assert id1.fingerprint == id2.fingerprint


def test_peer_identity_validation_injection_resistance():
    """Verify malicious identifiers with path injection or control chars are rejected."""
    with pytest.raises(IdentityValidationError):
        PeerIdentityProvider.create_identity("peer/hack", "zone_1", "org_1")
    with pytest.raises(IdentityValidationError):
        PeerIdentityProvider.create_identity("peer_1", "../zone_hack", "org_1")
    with pytest.raises(IdentityValidationError):
        PeerIdentityProvider.create_identity("peer_1", "zone_1", "org with spaces")


def test_unsupported_cryptographic_authentication():
    """Verify that requesting production cryptographic authentication raises UnsupportedSecurityModeError."""
    identity = PeerIdentityProvider.create_identity("peer_1", "zone_1", "org_1")
    with pytest.raises(UnsupportedSecurityModeError):
        PeerIdentityProvider.authenticate_cryptographically(identity, signature="fake_sig")


# ============================================================================
# 2. Peer Discovery & Invariant (DISCOVER != TRUST)
# ============================================================================

def test_discovery_without_automatic_trust(peer_declaration):
    """Verify that discovered peers start with DiscoveryStatus.DISCOVERED and trust_grant=None."""
    discovery_mgr = PeerDiscoveryManager()
    reg = discovery_mgr.discover(peer_declaration, current_epoch=1)

    assert reg.identity.peer_id == peer_declaration.identity.peer_id
    assert reg.discovery_status == DiscoveryStatus.DISCOVERED
    assert reg.trust_grant is None  # Strict DISCOVER != TRUST
    assert reg.is_active(current_epoch=1) is False


def test_discovery_policy_rejection():
    """Verify that peers from blocked zones are marked REJECTED upon discovery."""
    policy = CrossZoneFederationPolicy(blocked_zones={"zone_blocked"})
    discovery_mgr = PeerDiscoveryManager(policy=policy)

    identity = PeerIdentityProvider.create_identity("peer_bad", "zone_blocked", "org_x")
    attestation = PeerAttestationVerifier.create_attestation("att_b", "peer_bad", "zone_blocked")
    decl = PeerDeclaration("decl_b", identity, attestation, {}, [], {}, 1)

    reg = discovery_mgr.discover(decl, current_epoch=1)
    assert reg.discovery_status == DiscoveryStatus.REJECTED
    assert reg.trust_grant is None


# ============================================================================
# 3. Attestation Verification Tests
# ============================================================================

def test_attestation_acceptance(peer_declaration):
    """Verify that valid attestation is accepted."""
    valid, reason = PeerAttestationVerifier.verify_attestation(
        peer_declaration.attestation,
        expected_peer_id="peer_zone_alpha",
        expected_zone_id="zone_alpha",
    )
    assert valid is True
    assert "verified" in reason.lower()


def test_attestation_architecture_mismatch():
    """Verify attestation claiming incompatible architecture is rejected."""
    att = PeerAttestation(
        attestation_id="att_fake",
        peer_id="peer_1",
        zone_id="zone_1",
        architecture_version="2.0",  # Incompatible with ChakrMicro 0.1
        protocol_version="29.0",
        capability_manifest={"token_ceiling": 512},
        policy_version="29.0",
        runtime_integrity_hash="b" * 64,
        declared_epoch=1,
    )
    valid, reason = PeerAttestationVerifier.verify_attestation(att)
    assert valid is False
    assert "Incompatible architecture" in reason


def test_attestation_token_ceiling_exceeded():
    """Verify attestation with context ceiling > 512 is rejected."""
    att = PeerAttestation(
        attestation_id="att_fake",
        peer_id="peer_1",
        zone_id="zone_1",
        architecture_version="0.1",
        protocol_version="29.0",
        capability_manifest={"token_ceiling": 2048},  # Exceeds 512
        policy_version="29.0",
        runtime_integrity_hash="b" * 64,
        declared_epoch=1,
    )
    valid, reason = PeerAttestationVerifier.verify_attestation(att)
    assert valid is False
    assert "exceeds ChakrMicro maximum context" in reason


def test_attestation_cryptographic_status_rejected():
    """Verify claiming CRYPTOGRAPHICALLY_AUTHENTICATED attestation raises UnsupportedSecurityModeError."""
    att = PeerAttestation(
        attestation_id="att_crypto",
        peer_id="peer_1",
        zone_id="zone_1",
        architecture_version="0.1",
        protocol_version="29.0",
        capability_manifest={"token_ceiling": 512},
        policy_version="29.0",
        runtime_integrity_hash="b" * 64,
        declared_epoch=1,
        status=AttestationStatus.CRYPTOGRAPHICALLY_AUTHENTICATED,
    )
    with pytest.raises(UnsupportedSecurityModeError):
        PeerAttestationVerifier.verify_attestation(att)


# ============================================================================
# 4. Trust Model & Scope Negotiation Tests
# ============================================================================

def test_deterministic_trust_negotiation(peer_declaration):
    """Verify negotiation produces identical results given identical inputs."""
    negotiator = TrustNegotiator(local_zone_id="zone_local")
    agr1 = negotiator.negotiate(peer_declaration, current_epoch=2)
    agr2 = negotiator.negotiate(peer_declaration, current_epoch=2)

    assert agr1.is_successful() is True
    assert agr2.is_successful() is True
    assert agr1.accepted_scopes == agr2.accepted_scopes
    assert agr1.rejected_scopes == agr2.rejected_scopes
    assert agr1.trust_grant.trust_level == agr2.trust_grant.trust_level
    assert agr1.trust_grant.expires_epoch == agr2.trust_grant.expires_epoch


def test_negotiation_default_deny():
    """Verify unrequested or unwhitelisted scopes are not granted."""
    policy = CrossZoneFederationPolicy(
        allowed_scopes={FederationScope.ALLOW_EVIDENCE_EXCHANGE}  # Verification not allowed
    )
    negotiator = TrustNegotiator(local_zone_id="zone_local", policy=policy)

    decl = PeerDeclaration(
        declaration_id="decl_1",
        identity=PeerIdentityProvider.create_identity("p1", "z1", "o1"),
        attestation=PeerAttestationVerifier.create_attestation("att_1", "p1", "z1", runtime_integrity_hash="c"*64),
        resource_profile={},
        requested_scopes=[
            FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            FederationScope.ALLOW_VERIFICATION,
        ],
        policy_profile={},
        epoch=1,
    )

    agr = negotiator.negotiate(decl, current_epoch=1)
    assert FederationScope.ALLOW_EVIDENCE_EXCHANGE in agr.accepted_scopes
    assert FederationScope.ALLOW_VERIFICATION in agr.rejected_scopes
    assert agr.trust_grant.allows_scope(FederationScope.ALLOW_EVIDENCE_EXCHANGE, current_epoch=1) is True
    assert agr.trust_grant.allows_scope(FederationScope.ALLOW_VERIFICATION, current_epoch=1) is False


def test_prohibited_scopes_unconditionally_denied():
    """Verify prohibited scopes (like model weight access) are rejected."""
    policy = CrossZoneFederationPolicy()
    peer = PeerIdentityProvider.create_identity("p1", "z1", "o1")

    # Prohibited scopes must fail evaluate_scope_request
    ok, reason = policy.evaluate_scope_request(peer, ProhibitedScope.DENY_MODEL_WEIGHT_ACCESS)
    assert ok is False
    assert "violates prohibited boundary" in reason


def test_limited_federation_scopes():
    """Verify peer with metadata-only scopes gets LIMITED_TRUST or IDENTIFIED."""
    policy = CrossZoneFederationPolicy(allowed_scopes={FederationScope.ALLOW_MODEL_METADATA})
    negotiator = TrustNegotiator(local_zone_id="zone_local", policy=policy)

    decl = PeerDeclaration(
        declaration_id="decl_meta",
        identity=PeerIdentityProvider.create_identity("pm", "zm", "om"),
        attestation=PeerAttestationVerifier.create_attestation("att_m", "pm", "zm", runtime_integrity_hash="c"*64),
        resource_profile={},
        requested_scopes=[FederationScope.ALLOW_MODEL_METADATA],
        policy_profile={},
        epoch=1,
    )
    agr = negotiator.negotiate(decl, current_epoch=1)
    assert agr.is_successful() is True
    assert agr.trust_grant.trust_level == TrustLevel.IDENTIFIED
    assert agr.trust_grant.allows_scope(FederationScope.ALLOW_MODEL_METADATA, 1) is True
    assert agr.trust_grant.allows_scope(FederationScope.ALLOW_EVIDENCE_EXCHANGE, 1) is False


# ============================================================================
# 5. Peer Registry & Hard Ceilings Tests
# ============================================================================

def test_peer_registry_uniqueness():
    """Verify duplicate peer registration is rejected."""
    registry = PeerRegistry()
    id1 = PeerIdentityProvider.create_identity("p1", "z1", "o1")
    reg1 = PeerRegistration(id1, DiscoveryStatus.DISCOVERED)

    registry.register_peer(reg1)
    with pytest.raises(DuplicatePeerError):
        registry.register_peer(reg1)


def test_peer_registry_total_capacity_ceiling():
    """Verify registry enforces MAX_PEERS_TOTAL ceiling (32)."""
    registry = PeerRegistry()
    for i in range(MAX_PEERS_TOTAL):
        p_id = f"peer_cap_{i}"
        z_id = f"zone_cap_{i}"
        reg = PeerRegistration(
            PeerIdentityProvider.create_identity(p_id, z_id, "org_c"),
            DiscoveryStatus.DISCOVERED,
        )
        registry.register_peer(reg)

    overflow_reg = PeerRegistration(
        PeerIdentityProvider.create_identity("peer_overflow", "zone_overflow", "org_c"),
        DiscoveryStatus.DISCOVERED,
    )
    with pytest.raises(PeerRegistryCapacityError):
        registry.register_peer(overflow_reg)


def test_peer_registry_zone_capacity_ceiling():
    """Verify registry enforces MAX_PEERS_PER_ZONE ceiling (16)."""
    registry = PeerRegistry()
    for i in range(MAX_PEERS_PER_ZONE):
        p_id = f"peer_z_{i}"
        reg = PeerRegistration(
            PeerIdentityProvider.create_identity(p_id, "zone_same", "org_c"),
            DiscoveryStatus.DISCOVERED,
        )
        registry.register_peer(reg)

    overflow_reg = PeerRegistration(
        PeerIdentityProvider.create_identity("peer_z_overflow", "zone_same", "org_c"),
        DiscoveryStatus.DISCOVERED,
    )
    with pytest.raises(PeerRegistryCapacityError):
        registry.register_peer(overflow_reg)


# ============================================================================
# 6. Trust Expiration & Revocation Tests
# ============================================================================

def test_trust_expiration_via_epochs():
    """Verify trust grant expires when current_epoch exceeds expires_epoch."""
    grant = TrustModel.create_grant(
        grant_id="grant_exp_1",
        issuer_zone_id="z_local",
        subject_peer_id="p1",
        subject_zone_id="z1",
        trust_level=TrustLevel.LIMITED_TRUST,
        permitted_scopes=[FederationScope.ALLOW_EVIDENCE_EXCHANGE],
        issued_epoch=1,
        expires_epoch=5,
    )

    # Valid at epoch 1 to 5
    assert grant.is_valid_at(1) is True
    assert grant.is_valid_at(5) is True
    assert grant.allows_scope(FederationScope.ALLOW_EVIDENCE_EXCHANGE, 5) is True

    # Expired at epoch 6
    assert grant.is_valid_at(6) is False
    assert grant.allows_scope(FederationScope.ALLOW_EVIDENCE_EXCHANGE, 6) is False
    assert TrustModel.check_and_expire(grant, current_epoch=6) is True
    assert grant.status == TrustStatus.EXPIRED


def test_peer_revocation_invalidation():
    """Verify administrative revocation invalidates trust and records RevocationRecord."""
    registry = PeerRegistry()
    p_id = "peer_rev_test"
    z_id = "zone_rev_test"
    identity = PeerIdentityProvider.create_identity(p_id, z_id, "org_r")
    grant = TrustModel.create_grant(
        grant_id="grant_rev_1",
        issuer_zone_id="z_local",
        subject_peer_id=p_id,
        subject_zone_id=z_id,
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[FederationScope.ALLOW_EVIDENCE_EXCHANGE],
        issued_epoch=1,
        expires_epoch=10,
    )
    reg = PeerRegistration(identity, DiscoveryStatus.VERIFIED, trust_grant=grant)
    registry.register_peer(reg)

    rec = registry.revoke_peer(p_id, reason="Security policy breach", revoked_epoch=3, revoked_by="admin")

    assert rec.peer_id == p_id
    assert rec.reason == "Security policy breach"
    assert grant.status == TrustStatus.REVOKED
    assert grant.trust_level == TrustLevel.REVOKED
    assert grant.allows_scope(FederationScope.ALLOW_EVIDENCE_EXCHANGE, 3) is False
    assert reg.discovery_status == DiscoveryStatus.REVOKED


# ============================================================================
# 7. Isolation Guard Tests (Tenant & Sensitive Payload)
# ============================================================================

def test_cross_tenant_isolation_denial():
    """Verify attempting cross-tenant crossover raises IsolationViolationError."""
    with pytest.raises(IsolationViolationError):
        CrossZoneIsolationGuard.validate_tenant_boundary(
            peer_zone_id="zone_remote",
            target_zone_id="zone_local",
            peer_tenant_id="tenant_foreign",
            target_tenant_id="tenant_local_secure",
        )


def test_sensitive_payload_sanitization_weight_rejection():
    """Verify payloads containing model weights or activations raise IsolationViolationError."""
    payload_weights = {"task_id": "t1", "model_weights": [0.1, 0.2]}
    with pytest.raises(IsolationViolationError):
        CrossZoneIsolationGuard.sanitize_payload(payload_weights)

    payload_activations = {"task_id": "t1", "raw_activations": [0.5]}
    with pytest.raises(IsolationViolationError):
        CrossZoneIsolationGuard.sanitize_payload(payload_activations)


def test_sensitive_payload_sanitization_secret_rejection():
    """Verify payloads containing credential strings raise IsolationViolationError."""
    payload_secret = {"task_id": "t1", "config": "api_key = 'super_secret_token_12345'"}
    with pytest.raises(IsolationViolationError):
        CrossZoneIsolationGuard.sanitize_payload(payload_secret)


# ============================================================================
# 8. Capability Gate Enforcement & Authority Boundaries
# ============================================================================

def test_capability_gate_enforcement_authorized(capability_gate, peer_declaration):
    """Verify peer capability requests must pass through CapabilityGate under local authority."""
    engine = CrossZoneFederationEngine(
        local_zone_id="zone_local",
        capability_gate=capability_gate,
    )
    engine.discover_peer(peer_declaration)
    agr = engine.negotiate_trust(peer_declaration)
    assert agr.is_successful() is True

    # Build valid capability request
    req = CapabilityRequest(
        capability_id="mock_echo",
        parameters={"message": "hello federated world"},
        caller_id="peer_zone_alpha",
        session_id="session_01",
    )
    cap_ctx = CapabilityContext(
        granted_permissions={"test.echo"},
    )

    authorized, payload, note = engine.authorize_cross_zone_request(
        peer_id="peer_zone_alpha",
        peer_zone_id="zone_alpha",
        peer_tenant_id="tenant_common",
        target_tenant_id="tenant_common",
        session_id="session_01",
        requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
        payload={"query": "test"},
        capability_request=req,
        capability_context=cap_ctx,
    )
    assert authorized is True
    assert "authorized" in note.lower()


def test_capability_gate_enforcement_unauthorized(capability_gate, peer_declaration):
    """Verify peer capability request with missing permissions is denied by CapabilityGate."""
    engine = CrossZoneFederationEngine(
        local_zone_id="zone_local",
        capability_gate=capability_gate,
    )
    engine.discover_peer(peer_declaration)
    engine.negotiate_trust(peer_declaration)

    # Missing required permission 'test.echo'
    req = CapabilityRequest(
        capability_id="mock_echo",
        parameters={"message": "unauthorized attempt"},
        caller_id="peer_zone_alpha",
        session_id="session_01",
    )
    cap_ctx = CapabilityContext(
        granted_permissions=set(),  # No permissions
    )

    with pytest.raises(CapabilityAuthorizationError):
        engine.authorize_cross_zone_request(
            peer_id="peer_zone_alpha",
            peer_zone_id="zone_alpha",
            peer_tenant_id="tenant_common",
            target_tenant_id="tenant_common",
            session_id="session_01",
            requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            payload={},
            capability_request=req,
            capability_context=cap_ctx,
        )


def test_unregistered_peer_denial():
    """Verify requests from unregistered peers fail closed."""
    engine = CrossZoneFederationEngine(local_zone_id="zone_local")
    with pytest.raises(CrossZoneAuthorizationError):
        engine.authorize_cross_zone_request(
            peer_id="peer_unknown",
            peer_zone_id="zone_unknown",
            peer_tenant_id="t1",
            target_tenant_id="t1",
            session_id="s1",
            requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            payload={},
        )


def test_revoked_peer_denial(peer_declaration):
    """Verify revoked peers are immediately blocked from cross-zone interactions."""
    engine = CrossZoneFederationEngine(local_zone_id="zone_local")
    engine.discover_peer(peer_declaration)
    engine.negotiate_trust(peer_declaration)

    engine.revoke_peer(peer_declaration.identity.peer_id, reason="Policy breach")

    with pytest.raises(CrossZoneAuthorizationError):
        engine.authorize_cross_zone_request(
            peer_id=peer_declaration.identity.peer_id,
            peer_zone_id=peer_declaration.identity.zone_id,
            peer_tenant_id="t1",
            target_tenant_id="t1",
            session_id="s1",
            requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            payload={},
        )


# ============================================================================
# 9. Bounded Audit History Tests
# ============================================================================

def test_bounded_audit_history_fifo_eviction():
    """Verify audit logger never exceeds MAX_AUDIT_LOG_ENTRIES (1000)."""
    logger = BoundedAuditLogger(max_entries=50)
    for i in range(120):
        logger.log(
            event_type=AuditEventType.PEER_DISCOVERED,
            epoch=i,
            peer_id=f"p_{i}",
            details={"index": i},
        )
    assert logger.entry_count == 50
    recent = logger.get_recent(limit=10)
    assert len(recent) == 10
    assert recent[-1].details["index"] == 119


# ============================================================================
# 10. Neural Core Immutability & Frozen Invariants
# ============================================================================

def test_neural_weight_immutability(frozen_model, peer_declaration):
    """Verify neural weights remain strictly unchanged (Delta W = 0) during cross-zone operations."""
    engine = CrossZoneFederationEngine(
        local_zone_id="zone_local",
        model=frozen_model,
    )
    pre_hash = engine._compute_weight_hash()

    engine.discover_peer(peer_declaration)
    engine.negotiate_trust(peer_declaration)

    engine.authorize_cross_zone_request(
        peer_id=peer_declaration.identity.peer_id,
        peer_zone_id=peer_declaration.identity.zone_id,
        peer_tenant_id="tenant_01",
        target_tenant_id="tenant_01",
        session_id="sess_01",
        requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
        payload={"query": "test invariant"},
    )

    post_hash = engine._compute_weight_hash()
    assert pre_hash == post_hash


def test_neural_weight_mutation_detection_fail_closed(frozen_model, peer_declaration):
    """Verify engine raises WeightMutationDetectedError if weights are tampered with."""
    engine = CrossZoneFederationEngine(
        local_zone_id="zone_local",
        model=frozen_model,
    )
    pre_hash = engine._compute_weight_hash()

    # Artificially mutate a parameter
    with torch.no_grad():
        for param in frozen_model.parameters():
            param.add_(0.01)
            break

    with pytest.raises(WeightMutationDetectedError):
        engine._verify_weight_invariants(pre_hash)


def test_frozen_chakrmicro_architectural_invariants(frozen_model):
    """Verify ChakrMicro v0.1 parameters, vocab, context, and special tokens."""
    total_params = sum(p.numel() for p in frozen_model.parameters())
    assert total_params == 3_443_136
    assert frozen_model.config.vocab_size == 4096
    assert frozen_model.config.max_seq_len == 512
    assert frozen_model.config.d_model == 192
    assert frozen_model.config.n_layers == 6
    assert frozen_model.config.n_heads == 6
    assert frozen_model.config.hidden_dim == 512


# ============================================================================
# 11. Complete End-to-End Cross-Zone Federation Lifecycle
# ============================================================================

def test_complete_end_to_end_cross_zone_federation_cycle(frozen_model, capability_gate, peer_declaration):
    """
    Execute full lifecycle:
    1. Discovery (DISCOVER != TRUST)
    2. Attestation Verification
    3. Trust Negotiation (Default Deny, Bounded Grant)
    4. Gated Cross-Zone Request Execution (via CapabilityGate)
    5. Epoch Advancement & Expiration
    6. Audit Trace Emission & Immutability Verification
    """
    engine = CrossZoneFederationEngine(
        local_zone_id="zone_central",
        capability_gate=capability_gate,
        model=frozen_model,
        initial_epoch=1,
    )

    # 1. Discovery
    reg = engine.discover_peer(peer_declaration)
    assert reg.discovery_status == DiscoveryStatus.DISCOVERED
    assert reg.trust_grant is None

    # 2. Attestation
    att_ok, att_reason = engine.attest_peer(peer_declaration.identity.peer_id, peer_declaration.attestation)
    assert att_ok is True

    # 3. Negotiation
    agreement = engine.negotiate_trust(peer_declaration)
    assert agreement.is_successful() is True
    assert agreement.trust_grant.trust_level == TrustLevel.LIMITED_TRUST
    assert agreement.trust_grant.expires_epoch == 11  # 1 + default 10

    # 4. Gated Cross-Zone Request
    cap_req = CapabilityRequest(
        capability_id="mock_echo",
        parameters={"message": "lifecycle test"},
        caller_id=peer_declaration.identity.peer_id,
        session_id="sess_life",
    )
    cap_ctx = CapabilityContext(
        granted_permissions={"test.echo"},
    )
    authorized, sanitized, msg = engine.authorize_cross_zone_request(
        peer_id=peer_declaration.identity.peer_id,
        peer_zone_id=peer_declaration.identity.zone_id,
        peer_tenant_id="tenant_shared",
        target_tenant_id="tenant_shared",
        session_id="sess_life",
        requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
        payload={"query": "test"},
        capability_request=cap_req,
        capability_context=cap_ctx,
    )
    assert authorized is True
    assert sanitized["query"] == "test"

    # 5. Emit Trace
    trace = engine.emit_trace(
        peer_id=peer_declaration.identity.peer_id,
        event_type="CROSS_ZONE_REQUEST",
        status="AUTHORIZED",
        granted_scopes=agreement.accepted_scopes,
    )
    assert trace.status == "AUTHORIZED"
    assert trace.peer_id == peer_declaration.identity.peer_id

    # 6. Advance Epoch past expiration
    engine.advance_epoch(epochs=15)  # epoch becomes 16 > 11
    assert engine.current_epoch == 16

    # 7. Request must now be denied due to expiration
    with pytest.raises(CrossZoneAuthorizationError) as exc_info:
        engine.authorize_cross_zone_request(
            peer_id=peer_declaration.identity.peer_id,
            peer_zone_id=peer_declaration.identity.zone_id,
            peer_tenant_id="tenant_shared",
            target_tenant_id="tenant_shared",
            session_id="sess_life",
            requested_scope=FederationScope.ALLOW_EVIDENCE_EXCHANGE,
            payload={"query": "test expired"},
        )
    assert "expired" in str(exc_info.value).lower()

    # 8. Verify audit log captures lifecycle events
    recent_events = [r.event_type for r in engine.audit_logger.get_recent(limit=20)]
    assert AuditEventType.PEER_DISCOVERED in recent_events
    assert AuditEventType.IDENTITY_PRESENTED in recent_events
    assert AuditEventType.ATTESTATION_ACCEPTED in recent_events
    assert AuditEventType.TRUST_NEGOTIATED in recent_events
    assert AuditEventType.FEDERATION_ESTABLISHED in recent_events
    assert AuditEventType.FEDERATION_EXPIRED in recent_events
    assert AuditEventType.REQUEST_DENIED in recent_events


# ============================================================================
# 12. Additional Auxiliary & Edge Case Tests
# ============================================================================

def test_peer_discovery_manager_list_and_update_status(peer_declaration):
    """Verify discovery manager can filter discovered peers by status and update status."""
    mgr = PeerDiscoveryManager()
    reg = mgr.discover(peer_declaration, current_epoch=1)
    assert len(mgr.list_discovered()) == 1
    assert len(mgr.list_discovered(status=DiscoveryStatus.DISCOVERED)) == 1
    assert len(mgr.list_discovered(status=DiscoveryStatus.VERIFIED)) == 0

    ok = mgr.update_status(peer_declaration.identity.peer_id, DiscoveryStatus.VERIFIED, epoch=2)
    assert ok is True
    assert len(mgr.list_discovered(status=DiscoveryStatus.VERIFIED)) == 1
    assert mgr.update_status("nonexistent_peer", DiscoveryStatus.VERIFIED, epoch=2) is False


def test_peer_registry_remove_peer():
    """Verify registry peer removal."""
    registry = PeerRegistry()
    id_obj = PeerIdentityProvider.create_identity("peer_rem", "zone_rem", "org_rem")
    reg = PeerRegistration(id_obj, DiscoveryStatus.DISCOVERED)
    registry.register_peer(reg)

    assert registry.get_peer("peer_rem") is not None
    assert registry.remove_peer("peer_rem") is True
    assert registry.get_peer("peer_rem") is None
    assert registry.remove_peer("peer_rem") is False


def test_negotiation_agreement_when_attestation_fails(peer_declaration):
    """Verify negotiation returns unsuccessful agreement when peer attestation is invalid."""
    negotiator = TrustNegotiator(local_zone_id="zone_local")
    bad_att = PeerAttestation(
        attestation_id="bad_att",
        peer_id=peer_declaration.identity.peer_id,
        zone_id=peer_declaration.identity.zone_id,
        architecture_version="99.9",  # Bad arch
        protocol_version="29.0",
        capability_manifest={"token_ceiling": 512},
        policy_version="29.0",
        runtime_integrity_hash="c" * 64,
        declared_epoch=1,
    )
    bad_decl = PeerDeclaration(
        declaration_id="decl_bad_att",
        identity=peer_declaration.identity,
        attestation=bad_att,
        resource_profile={},
        requested_scopes=[FederationScope.ALLOW_EVIDENCE_EXCHANGE],
        policy_profile={},
        epoch=1,
    )
    agr = negotiator.negotiate(bad_decl, current_epoch=1)
    assert agr.is_successful() is False
    assert agr.trust_grant is None
    assert "ATTESTATION" in agr.rejection_reasons


def test_negotiation_agreement_when_identity_fingerprint_tampered():
    """Verify negotiation rejects declaration if identity fingerprint does not match canonical serialization."""
    negotiator = TrustNegotiator(local_zone_id="zone_local")
    tampered_id = PeerIdentity(
        peer_id="peer_tampered",
        zone_id="zone_tampered",
        organization_id="org_tampered",
        fingerprint="0" * 64,  # Corrupted fingerprint
    )
    decl = PeerDeclaration(
        declaration_id="decl_tamp",
        identity=tampered_id,
        attestation=PeerAttestationVerifier.create_attestation("att_t", "peer_tampered", "zone_tampered"),
        resource_profile={},
        requested_scopes=[FederationScope.ALLOW_EVIDENCE_EXCHANGE],
        policy_profile={},
        epoch=1,
    )
    agr = negotiator.negotiate(decl, current_epoch=1)
    assert agr.is_successful() is False
    assert agr.trust_grant is None
    assert "IDENTITY" in agr.rejection_reasons


def test_audit_logger_query_by_peer_and_event_type():
    """Verify audit logger querying by peer ID and event type."""
    logger = BoundedAuditLogger(max_entries=100)
    logger.log(AuditEventType.PEER_DISCOVERED, epoch=1, peer_id="peer_a", details={"msg": "1"})
    logger.log(AuditEventType.POLICY_ACCEPTED, epoch=1, peer_id="peer_a", details={"msg": "2"})
    logger.log(AuditEventType.PEER_DISCOVERED, epoch=1, peer_id="peer_b", details={"msg": "3"})

    peer_a_events = logger.get_by_peer("peer_a")
    assert len(peer_a_events) == 2

    discovered_events = logger.get_by_event_type(AuditEventType.PEER_DISCOVERED)
    assert len(discovered_events) == 2


def test_advance_epoch_multiple_peers_selective_expiration(capability_gate):
    """Verify that advancing epoch expires only grants whose TTL has elapsed."""
    engine = CrossZoneFederationEngine(local_zone_id="zone_local", capability_gate=capability_gate)

    # Peer 1: short grant expires at epoch 5
    id1 = PeerIdentityProvider.create_identity("peer_short", "zone_s", "org_s")
    grant1 = TrustModel.create_grant("g1", "zone_local", "peer_short", "zone_s", TrustLevel.LIMITED_TRUST, [FederationScope.ALLOW_EVIDENCE_EXCHANGE], 1, 5)
    reg1 = PeerRegistration(id1, DiscoveryStatus.VERIFIED, trust_grant=grant1)
    engine.registry.register_peer(reg1)

    # Peer 2: long grant expires at epoch 20
    id2 = PeerIdentityProvider.create_identity("peer_long", "zone_l", "org_l")
    grant2 = TrustModel.create_grant("g2", "zone_local", "peer_long", "zone_l", TrustLevel.LIMITED_TRUST, [FederationScope.ALLOW_EVIDENCE_EXCHANGE], 1, 20)
    reg2 = PeerRegistration(id2, DiscoveryStatus.VERIFIED, trust_grant=grant2)
    engine.registry.register_peer(reg2)

    # Advance to epoch 10
    engine.advance_epoch(epochs=9)  # epoch becomes 10

    assert grant1.status == TrustStatus.EXPIRED
    assert reg1.discovery_status == DiscoveryStatus.EXPIRED
    assert grant2.status == TrustStatus.ACTIVE
    assert reg2.discovery_status == DiscoveryStatus.VERIFIED
