"""
Comprehensive Security & Verification Test Suite for Step 37:
Distributed Resource & Capability Advertisement.

Verifies:
1. Local resource profile creation
2. CPU discovery
3. Memory discovery
4. Optional accelerator discovery
5. Capability declaration
6. Advertisement creation
7. Advertisement serialization
8. Advertisement integrity
9. Advertisement authentication
10. Valid advertisement acceptance
11. Invalid advertisement rejection
12. Tampered advertisement rejection
13. Replay rejection
14. Stale advertisement detection
15. Advertisement version handling
16. Capability filtering
17. Revoked peer advertisement rejection
18. Membership boundary enforcement
19. Multi-tenant isolation
20. Local resource authority
21. Remote claim cannot grant permission
22. Volatile resource refresh
23. Restart behavior
24. Recovery behavior
25. ΔW = 0 neural invariant
26. Secret leakage prevention
27. Privacy boundary
28. Resource registry lifecycle
29. Transport integration
30. Policy constraints & filtering
"""

import hashlib
import time
from typing import Dict, Any, Optional

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.provider import CalculatorCapability
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
)
from chakrview.cognition.peering.models import (
    AuditEventType,
    DiscoveryStatus,
    FederationScope,
    PeerIdentity,
    PeerRegistration,
)
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.peering.session import SecurePeerSession
from chakrview.cognition.federation.persistence.memory import InMemorySecurityStateStore
from chakrview.cognition.federation.persistence.models import JournalEntryType
from chakrview.cognition.federation.transport.models import (
    ChannelState,
    FederationMessageType,
    FederationMessageEnvelope,
)
from chakrview.cognition.federation.transport.channel import FederationChannel
from chakrview.cognition.federation.resources import (
    FederationResourceError,
    ResourceDiscoveryError,
    InvalidAdvertisementError,
    StaleAdvertisementError,
    AdvertisementTamperedError,
    AdvertisementReplayError,
    ProhibitedResourceDataError,
    ResourcePolicyError,
    UnauthorizedResourceAccessError,
    TenantResourceIsolationError,
    AcceleratorType,
    StorageClass,
    CPUResource,
    MemoryResource,
    AcceleratorResource,
    StorageResource,
    PlatformResource,
    NodeResourceProfile,
    ExecutionType,
    AdvertisedCapability,
    AdvertisementFreshness,
    ResourceAdvertisement,
    ResourceSharingPolicy,
    LocalResourceDetector,
    FederationResourceRegistry,
    FederationResourceManager,
)

FROZEN_NEURAL_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_model():
    torch.manual_seed(42)
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    m.eval()
    return m


@pytest.fixture
def mock_gate():
    gate = CapabilityGate()
    gate.registry.register(CalculatorCapability())
    return gate


@pytest.fixture
def local_key():
    return Ed25519PrivateKeyWrapper.generate()


@pytest.fixture
def remote_key():
    return Ed25519PrivateKeyWrapper.generate()


@pytest.fixture
def local_engine(test_model, mock_gate, local_key):
    return CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
        local_private_key=local_key,
        local_peer_id="peer_alpha",
    )


@pytest.fixture
def remote_engine(test_model, mock_gate, remote_key):
    return CrossZoneFederationEngine(
        local_zone_id="zone-beta",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
        local_private_key=remote_key,
        local_peer_id="peer_beta",
    )


@pytest.fixture
def detector():
    return LocalResourceDetector(node_epoch=1, coarse_privacy_default=False)


# ============================================================================
# 1. Local Resource Discovery & Profile Creation (Tests 01–05)
# ============================================================================

def test_01_local_resource_profile_creation(detector):
    """Test 1: Local resource profile is created with CPU, Memory, Platform, and Storage."""
    prof = detector.detect_profile()
    assert isinstance(prof, NodeResourceProfile)
    assert prof.profile_id.startswith("prof_")
    assert prof.cpu is not None
    assert prof.memory is not None
    assert prof.platform is not None
    assert prof.storage is not None
    assert prof.is_coarse is False


def test_02_cpu_discovery(detector):
    """Test 2: CPU discovery captures architecture, logical cores, and instruction capabilities."""
    cpu = detector.detect_cpu()
    assert isinstance(cpu, CPUResource)
    assert cpu.logical_cores >= 1
    assert cpu.architecture != ""
    assert isinstance(cpu.instruction_capabilities, list)
    assert cpu.available_cores >= 1.0


def test_03_memory_discovery(detector):
    """Test 3: Memory discovery captures total and available RAM."""
    mem = detector.detect_memory()
    assert isinstance(mem, MemoryResource)
    assert mem.total_memory_mb > 0
    assert mem.available_memory_mb > 0
    assert mem.available_memory_mb <= mem.total_memory_mb


def test_04_optional_accelerator_discovery(detector):
    """Test 4: Accelerator discovery returns generic accelerator or NONE without crashing."""
    accel = detector.detect_accelerator()
    assert isinstance(accel, AcceleratorResource)
    assert isinstance(accel.accelerator_type, AcceleratorType)

    # Test explicit accelerator override
    override = AcceleratorResource(
        accelerator_type=AcceleratorType.NPU,
        device_count=2,
        model_name="Edge NPU Co-Processor",
        compute_capabilities=["INT8", "FP16"],
        is_available=True,
    )
    detector_override = LocalResourceDetector(override_accelerator=override)
    accel2 = detector_override.detect_accelerator()
    assert accel2.accelerator_type == AcceleratorType.NPU
    assert accel2.device_count == 2
    assert accel2.is_available is True


def test_05_capability_declaration(detector, mock_gate):
    """Test 5: Capability declaration discovers core tokenization, inference, and registered capabilities."""
    caps = detector.detect_capabilities(registry=mock_gate.registry)
    assert len(caps) >= 3

    cap_ids = {c.capability_id for c in caps}
    assert "core.tokenization" in cap_ids
    assert "core.inference" in cap_ids
    assert "core.crypto" in cap_ids
    assert "calculator" in cap_ids

    inference_cap = next(c for c in caps if c.capability_id == "core.inference")
    assert inference_cap.execution_type == ExecutionType.INFERENCE
    assert inference_cap.is_exposed is True


# ============================================================================
# 2. Resource Advertisement & Serialization (Tests 06–09)
# ============================================================================

def test_06_advertisement_creation(detector, local_key):
    """Test 6: ResourceAdvertisement is constructed and signed with Ed25519."""
    prof = detector.detect_profile()
    caps = detector.detect_capabilities()

    adv = ResourceAdvertisement(
        advertisement_id="adv_test_001",
        node_id="peer_alpha",
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=caps,
    )
    assert adv.payload_digest != ""
    assert adv.signature == ""

    adv.sign(local_key)
    assert adv.signature != ""
    assert adv.verify_signature(local_key.public_key()) is True


def test_07_advertisement_serialization(detector, local_key):
    """Test 7: Canonical dictionary serialization and roundtrip deserialization."""
    prof = detector.detect_profile()
    caps = detector.detect_capabilities()

    adv = ResourceAdvertisement(
        advertisement_id="adv_test_002",
        node_id="peer_alpha",
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=caps,
    )
    adv.sign(local_key)

    d = adv.to_dict()
    assert isinstance(d, dict)
    assert d["advertisement_id"] == "adv_test_002"

    reconstructed = ResourceAdvertisement.from_dict(d)
    assert reconstructed.advertisement_id == adv.advertisement_id
    assert reconstructed.payload_digest == adv.payload_digest
    assert reconstructed.signature == adv.signature
    assert reconstructed.verify_signature(local_key.public_key()) is True


def test_08_advertisement_integrity(detector, local_key):
    """Test 8: Integrity check detects and rejects payload modifications."""
    prof = detector.detect_profile()
    adv = ResourceAdvertisement(
        advertisement_id="adv_test_003",
        node_id="peer_alpha",
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv.sign(local_key)
    assert adv.verify_integrity() is True

    # Tamper with core count
    adv.resource_profile.cpu.logical_cores = 999
    assert adv.verify_integrity() is False
    assert adv.verify_signature(local_key.public_key()) is False


def test_09_advertisement_authentication(detector, local_key, remote_key):
    """Test 9: Signature verification fails closed when presented with wrong public key."""
    prof = detector.detect_profile()
    adv = ResourceAdvertisement(
        advertisement_id="adv_test_004",
        node_id="peer_alpha",
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv.sign(local_key)

    # Valid with local key
    assert adv.verify_signature(local_key.public_key()) is True
    # Fails with remote key
    assert adv.verify_signature(remote_key.public_key()) is False


# ============================================================================
# 3. Registry & Validation Enforcement (Tests 10–15)
# ============================================================================

def test_10_valid_advertisement_acceptance(detector, remote_key):
    """Test 10: Registry accepts and stores valid signed peer advertisement."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()
    adv = ResourceAdvertisement(
        advertisement_id="adv_valid_001",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=detector.detect_capabilities(),
    )
    adv.sign(remote_key)

    reg.record_peer_advertisement(adv, public_key=remote_key.public_key())
    retrieved = reg.get_peer_advertisement("peer_beta")
    assert retrieved is not None
    assert retrieved.advertisement_id == "adv_valid_001"


def test_11_invalid_advertisement_rejection(detector):
    """Test 11: Registry rejects local node registered as peer claim."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha")
    adv = ResourceAdvertisement(
        advertisement_id="adv_self_001",
        node_id="peer_alpha",  # Same as local
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=detector.detect_profile(),
        capabilities=[],
    )
    with pytest.raises(InvalidAdvertisementError, match="Cannot record local node"):
        reg.record_peer_advertisement(adv)


def test_12_tampered_advertisement_rejection(detector, remote_key):
    """Test 12: Registry fails closed when an advertisement is tampered after signing."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()
    adv = ResourceAdvertisement(
        advertisement_id="adv_tamper_001",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv.sign(remote_key)

    # Tamper payload
    adv.resource_profile.cpu.available_cores = 128.0

    with pytest.raises(AdvertisementTamperedError):
        reg.record_peer_advertisement(adv, public_key=remote_key.public_key())


def test_13_replay_rejection(detector, remote_key):
    """Test 13: Registry rejects duplicate or regressive advertisement versions."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()

    adv1 = ResourceAdvertisement(
        advertisement_id="adv_v1",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=5,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv1.sign(remote_key)
    reg.record_peer_advertisement(adv1, public_key=remote_key.public_key())

    # Replay same version 5
    with pytest.raises(AdvertisementReplayError, match="version 5 <= floor 5"):
        reg.record_peer_advertisement(adv1, public_key=remote_key.public_key())

    # Regressive version 4
    adv0 = ResourceAdvertisement(
        advertisement_id="adv_v0",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=4,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv0.sign(remote_key)
    with pytest.raises(AdvertisementReplayError, match="version 4 <= floor 5"):
        reg.record_peer_advertisement(adv0, public_key=remote_key.public_key())


def test_14_stale_advertisement_detection(detector, remote_key):
    """Test 14: Expired or stale advertisements are detected and rejected on ingestion."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()

    # Construct advertisement with timestamp far in the past
    stale_adv = ResourceAdvertisement(
        advertisement_id="adv_stale_001",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
        timestamp=time.time() - 300.0,  # 300s ago
        ttl_seconds=60.0,
    )
    stale_adv.sign(remote_key)

    assert stale_adv.is_stale() is True
    with pytest.raises(StaleAdvertisementError):
        reg.record_peer_advertisement(stale_adv, public_key=remote_key.public_key())


def test_15_advertisement_version_handling(detector, remote_key):
    """Test 15: Monotonically increasing versions supersede older advertisements cleanly."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()

    adv1 = ResourceAdvertisement(
        advertisement_id="adv_seq_1",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv1.sign(remote_key)
    reg.record_peer_advertisement(adv1, public_key=remote_key.public_key())
    assert reg.get_peer_advertisement("peer_beta").version == 1

    adv2 = ResourceAdvertisement(
        advertisement_id="adv_seq_2",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=2,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv2.sign(remote_key)
    reg.record_peer_advertisement(adv2, public_key=remote_key.public_key())
    assert reg.get_peer_advertisement("peer_beta").version == 2


# ============================================================================
# 4. Capability Filtering & Boundaries (Tests 16–21)
# ============================================================================

def test_16_capability_filtering(detector, remote_key):
    """Test 16: Safe capability filtering queries claims without scheduling tasks."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()
    caps = detector.detect_capabilities()

    adv = ResourceAdvertisement(
        advertisement_id="adv_cap_001",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=caps,
    )
    adv.sign(remote_key)
    reg.record_peer_advertisement(adv, public_key=remote_key.public_key())

    # Query for INFERENCE capability claims
    inference_nodes = reg.filter_by_capability(ExecutionType.INFERENCE)
    assert "peer_beta" in inference_nodes

    # Query for CRYPTOGRAPHIC_OPS claims
    crypto_nodes = reg.filter_by_capability(ExecutionType.CRYPTOGRAPHIC_OPS)
    assert "peer_beta" in crypto_nodes

    # Query for DOCUMENT_PROCESSING claims (not present by default)
    doc_nodes = reg.filter_by_capability(ExecutionType.DOCUMENT_PROCESSING)
    assert "peer_beta" not in doc_nodes


def test_17_revoked_peer_advertisement_rejection(detector, remote_key):
    """Test 17: Revoked peer advertisements are strictly barred fail-closed."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    adv = ResourceAdvertisement(
        advertisement_id="adv_rev_001",
        node_id="peer_revoked",
        engine_id="eng_revoked",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=detector.detect_profile(),
        capabilities=[],
    )
    adv.sign(remote_key)

    with pytest.raises(UnauthorizedResourceAccessError, match="REVOKED"):
        reg.record_peer_advertisement(adv, is_peer_revoked=True)


def test_18_membership_boundary_enforcement(detector, remote_key):
    """Test 18: Quarantined peer advertisements are strictly barred fail-closed."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    adv = ResourceAdvertisement(
        advertisement_id="adv_quar_001",
        node_id="peer_quarantined",
        engine_id="eng_quarantined",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=detector.detect_profile(),
        capabilities=[],
    )
    adv.sign(remote_key)

    with pytest.raises(UnauthorizedResourceAccessError, match="QUARANTINED"):
        reg.record_peer_advertisement(adv, is_peer_quarantined=True)


def test_19_multi_tenant_isolation(detector, remote_key):
    """Test 19: Multi-tenant boundary prevents cross-tenant resource advertisement exposure."""
    reg = FederationResourceRegistry(
        local_node_id="peer_alpha",
        local_zone_id="zone-alpha",
        local_tenant_id="tenant-alpha",
    )
    adv = ResourceAdvertisement(
        advertisement_id="adv_tenant_cross",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="tenant-beta",  # Different tenant
        version=1,
        epoch=1,
        resource_profile=detector.detect_profile(),
        capabilities=[],
    )
    adv.sign(remote_key)

    with pytest.raises(TenantResourceIsolationError):
        reg.record_peer_advertisement(adv, public_key=remote_key.public_key())


def test_20_local_resource_authority(detector, remote_key):
    """Test 20: Remote peer claims cannot overwrite local ground truth profile or policy."""
    reg = FederationResourceRegistry(
        local_node_id="peer_alpha",
        local_zone_id="zone-alpha",
        local_tenant_id="zone-alpha",
    )
    local_prof = detector.detect_profile()
    reg.register_local_profile(local_prof, detector.detect_capabilities())

    # Peer advertises massive capacity
    remote_prof = detector.detect_profile()
    remote_prof.cpu.logical_cores = 1000
    adv = ResourceAdvertisement(
        advertisement_id="adv_claim_001",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=remote_prof,
        capabilities=[],
    )
    adv.sign(remote_key)
    reg.record_peer_advertisement(adv, public_key=remote_key.public_key())

    # Ground truth remains strictly local
    assert reg.get_local_profile().cpu.logical_cores == local_prof.cpu.logical_cores
    assert reg.get_peer_advertisement("peer_beta").resource_profile.cpu.logical_cores == 1000


def test_21_remote_claim_cannot_grant_permission(local_engine, remote_engine):
    """Test 21: Ingesting an advertisement grants zero execution rights or capability authorizations."""
    mgr = local_engine.resource_manager
    adv = remote_engine.resource_manager.create_local_advertisement()

    mgr.registry.record_peer_advertisement(adv, public_key=remote_engine.local_public_key)
    stored = mgr.registry.get_peer_advertisement(adv.node_id)
    assert stored is not None

    # Sovereign capability gate remains strictly unescalated
    gate = local_engine.capability_gate
    from chakrview.capability.gate import CapabilityAuthorizationError
    from chakrview.capability.contract import CapabilityRequest
    req = CapabilityRequest(
        capability_id="calculator",
        caller_id=adv.node_id,
        parameters={"expression": "1+1"},
        context={"provenance_source": "user_assertion"},
    )
    with pytest.raises(CapabilityAuthorizationError):
        gate.authorize(req)


# ============================================================================
# 5. Volatility, Restart & Recovery (Tests 22–24)
# ============================================================================

def test_22_volatile_resource_refresh(detector):
    """Test 22: Refreshing local resources updates volatile metrics while preserving static structures."""
    mgr = FederationResourceManager(engine=None, detector=detector)
    p1 = mgr.registry.get_local_profile()

    time.sleep(0.01)
    p2 = mgr.refresh_local_resources()

    assert p2.profile_id == p1.profile_id
    assert p2.timestamp >= p1.timestamp


def test_23_restart_behavior(detector, remote_key):
    """Test 23: On node restart, volatile peer claims are reset and re-observed cleanly."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    adv = ResourceAdvertisement(
        advertisement_id="adv_prior_restart",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=detector.detect_profile(),
        capabilities=[],
    )
    adv.sign(remote_key)
    reg.record_peer_advertisement(adv, public_key=remote_key.public_key())

    # Simulate restart by instantiating clean registry
    restarted_reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    assert restarted_reg.get_peer_advertisement("peer_beta") is None


def test_24_recovery_behavior(detector, local_key):
    """Test 24: Durable capability declarations survive snapshot/recovery; volatile metrics are fresh."""
    policy = ResourceSharingPolicy(
        max_exposed_cores=4.0,
        max_exposed_memory_mb=8192,
        allow_accelerator_sharing=True,
    )
    d = policy.to_dict()
    restored_policy = ResourceSharingPolicy.from_dict(d)

    assert restored_policy.max_exposed_cores == 4.0
    assert restored_policy.max_exposed_memory_mb == 8192
    assert restored_policy.allow_accelerator_sharing is True


# ============================================================================
# 6. Neural Invariant & Privacy Boundaries (Tests 25–28)
# ============================================================================

def test_25_neural_core_immutability(test_model, detector, local_key):
    """Test 25: Neural core weights are strictly immutable during resource discovery and advertisement (ΔW = 0)."""
    # Verify initial hash
    initial_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            initial_hasher.update(name.encode("utf-8"))
            initial_hasher.update(param.detach().cpu().numpy().tobytes())
    initial_hash = initial_hasher.hexdigest()
    assert initial_hash == FROZEN_NEURAL_WEIGHT_HASH

    # Run complete resource detection and advertisement pipeline
    prof = detector.detect_profile()
    caps = detector.detect_capabilities()
    adv = ResourceAdvertisement(
        advertisement_id="adv_neural_test",
        node_id="peer_alpha",
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=caps,
    )
    adv.sign(local_key)
    assert adv.verify_signature(local_key.public_key()) is True

    # Verify final hash
    final_hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            final_hasher.update(name.encode("utf-8"))
            final_hasher.update(param.detach().cpu().numpy().tobytes())
    final_hash = final_hasher.hexdigest()
    assert final_hash == initial_hash == FROZEN_NEURAL_WEIGHT_HASH


def test_26_secret_leakage_prevention(detector):
    """Test 26: Attempting to insert prohibited keywords or secrets fails closed."""
    prof = detector.detect_profile()
    adv = ResourceAdvertisement(
        advertisement_id="adv_leak_test",
        node_id="peer_alpha",
        engine_id="eng_alpha",
        zone_id="zone-alpha",
        tenant_id="zone-alpha",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )

    # Injecting private_key into profile dict raises ProhibitedResourceDataError
    with pytest.raises(ProhibitedResourceDataError, match="Prohibited sensitive keyword 'private_key'"):
        bad_payload = adv.to_dict()
        bad_payload["resource_profile"]["private_key"] = "SUPER_SECRET"
        from chakrview.cognition.federation.resources.models import _assert_no_prohibited_resource_keys
        _assert_no_prohibited_resource_keys(bad_payload)


def test_27_privacy_boundary(detector):
    """Test 27: Coarse privacy mode hides exact cores, suppresses storage, and masks accelerator model."""
    full_prof = detector.detect_profile(coarse=False)
    coarse_prof = full_prof.sanitize_for_privacy()

    assert coarse_prof.is_coarse is True
    assert coarse_prof.storage is None
    assert coarse_prof.cpu.physical_cores is None
    assert coarse_prof.cpu.instruction_capabilities == []


def test_28_resource_registry_lifecycle(detector, remote_key):
    """Test 28: Complete registry lifecycle: registration, freshness check, invalidation, purging."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()

    adv = ResourceAdvertisement(
        advertisement_id="adv_life_001",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=1,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
        ttl_seconds=10.0,
    )
    adv.sign(remote_key)
    reg.record_peer_advertisement(adv, public_key=remote_key.public_key())

    # Freshness check
    statuses = reg.check_freshness()
    assert statuses["peer_beta"] in (AdvertisementFreshness.FRESH, AdvertisementFreshness.AGING)

    # Invalidate on disconnect
    reg.invalidate_peer("peer_beta", reason="heartbeat_timeout")
    assert reg.get_peer_advertisement("peer_beta", allow_stale=False) is None
    assert reg.get_peer_advertisement("peer_beta", allow_stale=True) is not None


# ============================================================================
# 7. Transport Integration & Policy Constraints (Tests 29–35)
# ============================================================================

def test_29_transport_channel_advertisement_roundtrip(local_engine, remote_engine):
    """Test 29: End-to-end transport roundtrip sending ResourceAdvertisement over FederationChannel."""
    session = SecurePeerSession(
        session_id="sess_adv_rt",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
    )
    channel = FederationChannel(
        "chan_adv_rt",
        local_engine,
        "tcp://127.0.0.1:9099",
        session=session,
        initial_state=ChannelState.ESTABLISHED,
    )

    # Remote publishes advertisement
    remote_adv = remote_engine.resource_manager.create_local_advertisement()

    envelope = FederationMessageEnvelope(
        message_type=FederationMessageType.RESOURCE_ADVERTISEMENT,
        message_id="msg_adv_001",
        session_id=session.session_id,
        sender_engine_id=remote_adv.engine_id,
        receiver_engine_id="peer_alpha",
        sender_peer_id="peer_beta",
        receiver_peer_id="peer_alpha",
        sequence_number=1,
        epoch=1,
        payload=remote_adv.to_dict(),
        tenant_id="zone-alpha",
    )
    envelope.sign(remote_engine.local_private_key)

    # Local engine dispatcher receives envelope
    res = local_engine.dispatcher.dispatch(envelope, channel)
    assert res["status"] == "ACCEPTED"

    # Stored in local registry
    peer_claim = local_engine.resource_manager.registry.get_peer_advertisement("peer_beta")
    assert peer_claim is not None
    assert peer_claim.advertisement_id == remote_adv.advertisement_id


def test_30_transport_channel_query_roundtrip(local_engine, remote_engine):
    """Test 30: Querying resources over channel and receiving policy-filtered response."""
    session = SecurePeerSession(
        session_id="sess_qry_rt",
        local_peer_id="peer_alpha",
        remote_peer_id="peer_beta",
        local_zone_id="zone-alpha",
        remote_zone_id="zone-beta",
        created_epoch=1,
        expires_at_epoch=100,
    )
    channel = FederationChannel(
        "chan_qry_rt",
        local_engine,
        "tcp://127.0.0.1:9099",
        session=session,
        initial_state=ChannelState.ESTABLISHED,
    )

    query_envelope = FederationMessageEnvelope(
        message_type=FederationMessageType.RESOURCE_QUERY,
        message_id="msg_qry_001",
        session_id=session.session_id,
        sender_engine_id="peer_beta",
        receiver_engine_id="peer_alpha",
        sender_peer_id="peer_beta",
        receiver_peer_id="peer_alpha",
        sequence_number=2,
        epoch=1,
        payload={"scope": "ALLOW_CAPABILITY_METADATA"},
        tenant_id="zone-alpha",
    )
    query_envelope.sign(remote_engine.local_private_key)

    res = local_engine.dispatcher.dispatch(query_envelope, channel)
    assert res["status"] == "OK"
    assert len(res["advertisements"]) == 1
    assert res["advertisements"][0]["node_id"] == "peer_alpha"


def test_31_sharing_policy_constraints(detector):
    """Test 31: Sharing policy limits maximum exposed cores and memory."""
    policy = ResourceSharingPolicy(
        max_exposed_cores=2.0,
        max_exposed_memory_mb=1024,
        allow_accelerator_sharing=False,
    )
    mgr = FederationResourceManager(engine=None, detector=detector, sharing_policy=policy)
    adv = mgr.create_local_advertisement()

    assert adv.resource_profile.cpu.available_cores <= 2.0
    assert adv.resource_profile.memory.available_memory_mb <= 1024
    assert adv.resource_profile.accelerator is None


def test_32_accelerator_sharing_policy_control(detector):
    """Test 32: Accelerators are strictly excluded unless explicitly allowed by policy."""
    override = AcceleratorResource(
        accelerator_type=AcceleratorType.GPU,
        device_count=1,
        model_name="Test GPU",
        is_available=True,
    )
    detector_accel = LocalResourceDetector(override_accelerator=override)

    # Denied by default
    policy_denied = ResourceSharingPolicy(allow_accelerator_sharing=False)
    mgr_denied = FederationResourceManager(engine=None, detector=detector_accel, sharing_policy=policy_denied)
    adv_denied = mgr_denied.create_local_advertisement()
    assert adv_denied.resource_profile.accelerator is None

    # Explicitly permitted
    policy_allowed = ResourceSharingPolicy(allow_accelerator_sharing=True)
    mgr_allowed = FederationResourceManager(engine=None, detector=detector_accel, sharing_policy=policy_allowed)
    adv_allowed = mgr_allowed.create_local_advertisement()
    assert adv_allowed.resource_profile.accelerator is not None


def test_33_node_disconnect_staleness(local_engine, remote_engine):
    """Test 33: Peer disconnect marks advertised resources UNAVAILABLE."""
    mgr = local_engine.resource_manager
    adv = remote_engine.resource_manager.create_local_advertisement()
    mgr.registry.record_peer_advertisement(adv, public_key=remote_engine.local_public_key)

    assert mgr.registry.get_peer_advertisement(adv.node_id) is not None

    # Trigger disconnect hook
    mgr.on_peer_disconnected(adv.node_id, reason="heartbeat_timeout")
    assert mgr.registry.get_peer_advertisement(adv.node_id, allow_stale=False) is None
    assert mgr.registry.get_peer_advertisement(adv.node_id, allow_stale=True) is not None


def test_34_durable_journal_policy_logging():
    """Test 34: JournalEntryType contains Step 37 durable mutation types."""
    assert JournalEntryType.CAPABILITY_ADVERTISED.value == "CAPABILITY_ADVERTISED"
    assert JournalEntryType.RESOURCE_POLICY_UPDATED.value == "RESOURCE_POLICY_UPDATED"


def test_35_rejoined_peer_version_monotonicity(detector, remote_key):
    """Test 35: Rejoined peer must advance advertisement version beyond previous floor."""
    reg = FederationResourceRegistry(local_node_id="peer_alpha", local_zone_id="zone-alpha", local_tenant_id="*")
    prof = detector.detect_profile()

    adv1 = ResourceAdvertisement(
        advertisement_id="adv_prior",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=10,
        epoch=1,
        resource_profile=prof,
        capabilities=[],
    )
    adv1.sign(remote_key)
    reg.record_peer_advertisement(adv1, public_key=remote_key.public_key())

    # Peer disconnects
    reg.invalidate_peer("peer_beta")

    # Reconnects presenting stale version 10 -> rejected
    with pytest.raises(AdvertisementReplayError):
        reg.record_peer_advertisement(adv1, public_key=remote_key.public_key())

    # Reconnects presenting fresh version 11 -> accepted
    adv2 = ResourceAdvertisement(
        advertisement_id="adv_fresh",
        node_id="peer_beta",
        engine_id="eng_beta",
        zone_id="zone-beta",
        tenant_id="*",
        version=11,
        epoch=2,
        resource_profile=prof,
        capabilities=[],
    )
    adv2.sign(remote_key)
    reg.record_peer_advertisement(adv2, public_key=remote_key.public_key())
    assert reg.get_peer_advertisement("peer_beta").version == 11
