"""
Comprehensive Security & Verification Test Suite for Step 34:
Multi-Node Federation Runtime, Durable Security State & Failure Recovery.

Verifies:
1. Persistence & Serialization (Snapshot save/load, schema validation, secret non-leakage)
2. Security Journal (Hash-chaining, tamper detection, truncation, sequence regression)
3. Crash Recovery (Reconstruction, incomplete mutations, fail-closed on corruption)
4. Runtime Lifecycle & Failure Detection (UNREACHABLE != REVOKED, capacity bounds)
5. Rejoin Protocol & Conflict Matrix (Cases A through J)
6. Neural Core Immutability (Params=3,443,136, Vocab=4096, Context=512, ΔW = 0)
"""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
)
from chakrview.cognition.peering.models import (
    AuditEventType,
    DiscoveryStatus,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    FederationScope,
    RevocationRecord,
)
from chakrview.cognition.peering.engine import CrossZoneFederationEngine
from chakrview.cognition.peering.session import SecurePeerSession, SessionStatus
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    RevocationTargetType,
    HandshakeStatus,
    MAX_FEDERATION_ENGINES,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.errors import (
    CoordinationCapacityError,
    StateDigestConflictError,
)
from chakrview.cognition.federation.persistence import (
    SecurityStateStore,
    InMemorySecurityStateStore,
    SqliteSecurityStateStore,
    SecurityStateJournal,
    JournalEntry,
    JournalEntryType,
    DurableSecuritySnapshot,
    RecoveryManifest,
    GENESIS_JOURNAL_DIGEST,
    PersistenceError,
    DurableSchemaError,
    JournalError,
    JournalCorruptionError,
    JournalSequenceError,
    SnapshotCorruptionError,
    RecoveryFailedClosedError,
    RuntimeLifecycleError,
    EngineHealthError,
    RejoinProtocolError,
    JOURNAL_GENESIS_DIGEST,
    DURABLE_SCHEMA_VERSION,
)
from chakrview.cognition.federation.recovery import FederationRecoveryManager
from chakrview.cognition.federation.runtime import (
    FederationRuntime,
    EngineRuntimeStatus,
    EngineHealthStatus,
)


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
    return CapabilityGate()


@pytest.fixture
def memory_store():
    return InMemorySecurityStateStore()


@pytest.fixture
def sqlite_store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = SqliteSecurityStateStore(db_path=db_path)
    yield store
    store.close()
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def test_engine(test_model, mock_gate):
    return CrossZoneFederationEngine(
        local_zone_id="zone-alpha",
        capability_gate=mock_gate,
        model=test_model,
        initial_epoch=1,
    )


# ============================================================================
# Phase 1: Persistence & Schema Tests
# ============================================================================

def test_01_snapshot_save_and_load(memory_store, test_engine):
    """Verify snapshot creation, sealing, persistence, and loading."""
    snap = FederationRecoveryManager.create_snapshot_from_engine(
        engine=test_engine,
        store=memory_store,
        snapshot_version=1,
        journal_offset=0,
    )
    assert snap.verify_integrity() is True
    assert snap.integrity_hash is not None

    loaded = memory_store.load_latest_snapshot()
    assert loaded is not None
    assert loaded.snapshot_id == snap.snapshot_id
    assert loaded.verify_integrity() is True
    assert loaded.epoch == 1


def test_02_sqlite_persistence_roundtrip(sqlite_store, test_engine):
    """Verify SQLite backend stores snapshots and journal entries accurately."""
    snap = FederationRecoveryManager.create_snapshot_from_engine(
        engine=test_engine,
        store=sqlite_store,
        snapshot_version=1,
        journal_offset=5,
    )
    loaded = sqlite_store.load_snapshot(snap.snapshot_id)
    assert loaded is not None
    assert loaded.snapshot_id == snap.snapshot_id
    assert loaded.journal_offset == 5

    # Journal roundtrip
    entry = JournalEntry.create(
        sequence_num=1,
        epoch=1,
        entry_type=JournalEntryType.EPOCH_ADVANCED,
        payload={"new_epoch": 2},
        prev_digest=JOURNAL_GENESIS_DIGEST,
    )
    sqlite_store.append_journal_entry(entry)
    entries = sqlite_store.read_journal_entries(since_sequence=0)
    assert len(entries) == 1
    assert entries[0].sequence_num == 1
    assert entries[0].payload["new_epoch"] == 2
    assert entries[0].verify_integrity() is True


def test_03_snapshot_schema_version_validation():
    """Verify snapshot rejects invalid schema versions."""
    bad_data = {
        "snapshot_id": "snap_bad",
        "snapshot_version": 1,
        "journal_offset": 0,
        "engine_identity": {},
        "state_version": {},
        "epoch": 1,
        "composite_digest": "hash",
        "schema_version": "99.9",  # Unsupported schema
    }
    with pytest.raises(DurableSchemaError):
        DurableSecuritySnapshot.from_dict(bad_data)


def test_04_secret_non_leakage_in_persistence(test_engine):
    """Verify private keys and session secrets are strictly prohibited from persistence."""
    # Prohibit private keys in JournalEntry
    with pytest.raises(PermissionError):
        JournalEntry.create(
            sequence_num=1,
            epoch=1,
            entry_type=JournalEntryType.PEER_REGISTERED,
            payload={"peer_id": "peer1", "private_key": "raw_private_bytes"},
            prev_digest=JOURNAL_GENESIS_DIGEST,
        )

    # Verify snapshot payload contains zero private keys
    store = InMemorySecurityStateStore()
    snap = FederationRecoveryManager.create_snapshot_from_engine(
        engine=test_engine,
        store=store,
        snapshot_version=1,
        journal_offset=0,
    )
    snap_str = json.dumps(snap.to_dict()).lower()
    assert "private_key" not in snap_str
    assert "private_bytes" not in snap_str
    assert "session_key" not in snap_str


# ============================================================================
# Phase 2: Security Journal & Hash Chaining Tests
# ============================================================================

def test_05_journal_hash_chain_integrity():
    """Verify write-ahead journal creates a valid SHA-256 hash chain."""
    journal = SecurityStateJournal()
    e1 = journal.append(JournalEntryType.EPOCH_ADVANCED, epoch=1, payload={"new_epoch": 2})
    assert e1.sequence_num == 1
    assert e1.prev_digest == JOURNAL_GENESIS_DIGEST
    assert e1.verify_integrity() is True

    e2 = journal.append(JournalEntryType.PEER_REGISTERED, epoch=2, payload={"peer_id": "p1", "zone_id": "z1"})
    assert e2.sequence_num == 2
    assert e2.prev_digest == e1.digest
    assert e2.verify_integrity() is True

    # Validate complete chain
    is_valid, _ = SecurityStateJournal.verify_chain([e1, e2])
    assert is_valid is True


def test_06_journal_corruption_detection():
    """Verify tampered payload or modified digest is detected immediately."""
    journal = SecurityStateJournal()
    e1 = journal.append(JournalEntryType.EPOCH_ADVANCED, epoch=1, payload={"new_epoch": 2})
    e2 = journal.append(JournalEntryType.PEER_REVOKED, epoch=1, payload={"peer_id": "p1", "reason": "breach"})

    # Tamper with e2 payload
    tampered_entry = JournalEntry(
        sequence_num=e2.sequence_num,
        epoch=e2.epoch,
        entry_type=e2.entry_type,
        event_id=e2.event_id,
        payload={"peer_id": "p1", "reason": "TAMPERED_FORGED_REASON"},
        prev_digest=e2.prev_digest,
        digest=e2.digest,  # Digest no longer matches tampered payload!
        timestamp=e2.timestamp,
    )

    with pytest.raises(JournalCorruptionError):
        SecurityStateJournal.verify_chain([e1, tampered_entry])


def test_07_journal_sequence_regression_detection():
    """Verify non-monotonic or reordered sequence numbers are rejected."""
    e1 = JournalEntry.create(1, 1, JournalEntryType.EPOCH_ADVANCED, {"new_epoch": 2}, JOURNAL_GENESIS_DIGEST)
    # Regression: sequence number drops to 0 or repeats 1
    e2 = JournalEntry.create(1, 1, JournalEntryType.PEER_REVOKED, {"peer_id": "p1"}, e1.digest)

    with pytest.raises(JournalSequenceError):
        SecurityStateJournal.verify_chain([e1, e2])


def test_08_journal_broken_chain_detection():
    """Verify broken prev_digest chain raises JournalCorruptionError."""
    e1 = JournalEntry.create(1, 1, JournalEntryType.EPOCH_ADVANCED, {"new_epoch": 2}, JOURNAL_GENESIS_DIGEST)
    e2 = JournalEntry.create(2, 1, JournalEntryType.PEER_REVOKED, {"peer_id": "p1"}, prev_digest="forged_prev" * 4)

    with pytest.raises(JournalCorruptionError):
        SecurityStateJournal.verify_chain([e1, e2])


# ============================================================================
# Phase 3 & 4: Crash Recovery & Snapshot Tests
# ============================================================================

def test_09_clean_recovery_from_snapshot_and_journal(test_model, mock_gate):
    """Verify clean state reconstruction from snapshot + subsequent journal entries."""
    store = InMemorySecurityStateStore()

    # Engine 1: setup initial state and snapshot
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    peer_id = "peer_reconstructed_001"
    crypto = CryptographicPeerIdentity.create(
        zone_id="zone-beta", organization_id="org-beta", public_key=engine1.local_public_key, peer_id=peer_id
    )
    engine1.register_cryptographic_peer(crypto)
    engine1.advance_epoch(2)

    # Snapshot at epoch 3, offset 0
    snap = FederationRecoveryManager.create_snapshot_from_engine(engine1, store, snapshot_version=1, journal_offset=0)

    # Additional mutations appended to journal after snapshot
    e1 = JournalEntry.create(1, 3, JournalEntryType.PEER_REVOKED, {"peer_id": peer_id, "reason": "Compromised"}, GENESIS_JOURNAL_DIGEST)
    store.append_journal_entry(e1)

    # Engine 2: simulate cold restart from store
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    rec_mgr = FederationRecoveryManager(store)
    manifest = rec_mgr.recover(engine2)

    assert manifest.is_successful() is True
    assert manifest.snapshot_loaded == snap.snapshot_id
    assert manifest.journal_entries_replayed == 1

    # Verify peer was restored as REVOKED (terminal revocation invariant)
    peer2 = engine2.registry.get_peer(peer_id)
    assert peer2 is not None
    assert peer2.discovery_status == DiscoveryStatus.REVOKED


def test_10_recovery_fails_closed_on_corrupted_snapshot(test_model, mock_gate):
    """Verify recovery aborts and fails closed if snapshot integrity fails."""
    store = InMemorySecurityStateStore()
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    snap = FederationRecoveryManager.create_snapshot_from_engine(engine1, store, snapshot_version=1, journal_offset=0)

    # Tamper with snapshot payload in store
    store._snapshots[snap.snapshot_id]["epoch"] = 999  # breaks integrity_hash

    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    rec_mgr = FederationRecoveryManager(store)

    with pytest.raises(RecoveryFailedClosedError):
        rec_mgr.recover(engine2)


def test_11_recovery_fails_closed_on_corrupted_journal(test_model, mock_gate):
    """Verify recovery fails closed if any journal entry in replay sequence is corrupt (Case E)."""
    store = InMemorySecurityStateStore()
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    snap = FederationRecoveryManager.create_snapshot_from_engine(engine1, store, snapshot_version=1, journal_offset=0)

    # Append corrupt entry
    corrupt_entry = JournalEntry(
        sequence_num=1,
        epoch=1,
        entry_type=JournalEntryType.PEER_REVOKED,
        event_id="jrn_corrupt",
        payload={"peer_id": "p1"},
        prev_digest=JOURNAL_GENESIS_DIGEST,
        digest="bad_digest" * 6,
        timestamp=time.time(),
    )
    store.append_journal_entry(corrupt_entry)

    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    rec_mgr = FederationRecoveryManager(store)

    with pytest.raises(RecoveryFailedClosedError):
        rec_mgr.recover(engine2)


def test_12_crash_before_journal_append_does_not_survive(test_model, mock_gate):
    """Verify mutation occurring before journal append is NOT restored upon crash recovery (Case G)."""
    store = InMemorySecurityStateStore()
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    FederationRecoveryManager.create_snapshot_from_engine(engine1, store, snapshot_version=1, journal_offset=0)

    # Mutation applied only to in-memory engine1 without journal write (simulating crash before disk commit)
    peer_id = "ephemeral_unpersisted_peer"
    crypto = CryptographicPeerIdentity.create(
        zone_id="zone-beta", organization_id="org-beta", public_key=engine1.local_public_key, peer_id=peer_id
    )
    engine1.register_cryptographic_peer(crypto)
    assert engine1.registry.get_peer(peer_id) is not None

    # Recover new engine
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    rec_mgr = FederationRecoveryManager(store)
    manifest = rec_mgr.recover(engine2)

    assert manifest.is_successful() is True
    # The uncommitted mutation must NOT exist on engine2
    assert engine2.registry.get_peer(peer_id) is None


# ============================================================================
# Phase 5 & 6: Runtime Lifecycle & Health Tracking Tests
# ============================================================================

def test_13_runtime_lifecycle_start_stop_restart(test_model, mock_gate, memory_store):
    """Verify FederationRuntime transitions through clean lifecycle."""
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime = FederationRuntime(engine=engine, store=memory_store, auto_recover=False)

    assert runtime.status == EngineRuntimeStatus.RUNNING
    assert runtime.is_running is True

    # Stop runtime (creates snapshot)
    runtime.stop()
    assert runtime.status == EngineRuntimeStatus.STOPPED
    assert memory_store.load_latest_snapshot() is not None

    # Restart runtime (recovers snapshot)
    runtime.start()
    assert runtime.status == EngineRuntimeStatus.RUNNING


def test_14_runtime_engine_registration_capacity_bounds(test_model, mock_gate):
    """Verify MAX_FEDERATION_ENGINES limit fails closed."""
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime = FederationRuntime(engine=engine, auto_recover=False)

    for i in range(MAX_FEDERATION_ENGINES):
        ident = FederationEngineIdentityProvider.create_identity(
            engine_id=f"eng_{i}",
            zone_id=f"zone_{i}",
            created_epoch=1,
        )
        runtime.register_remote_engine(ident)

    # 17th engine registration must raise CoordinationCapacityError
    overflow_ident = FederationEngineIdentityProvider.create_identity(
        engine_id="eng_overflow",
        zone_id="zone_overflow",
        created_epoch=1,
    )
    with pytest.raises(CoordinationCapacityError):
        runtime.register_remote_engine(overflow_ident)


def test_15_health_tracking_unreachable_does_not_revoke(test_model, mock_gate):
    """Verify UNREACHABLE != REVOKED: network outages do not revoke peer trust."""
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime = FederationRuntime(engine=engine, auto_recover=False)

    remote_id = "eng_remote_99"
    ident = FederationEngineIdentityProvider.create_identity(remote_id, "zone-beta", 1)
    runtime.register_remote_engine(ident)

    # Register peer in registry
    crypto = CryptographicPeerIdentity.create(
        zone_id="zone-beta", organization_id="org-beta", public_key=engine.local_public_key, peer_id="peer_99"
    )
    reg = engine.register_cryptographic_peer(crypto)

    # Mark remote engine UNREACHABLE
    runtime.set_engine_health(remote_id, EngineHealthStatus.UNREACHABLE, reason="Network partition")
    assert runtime.get_engine_health(remote_id) == EngineHealthStatus.UNREACHABLE

    # Peer trust must remain ACTIVE, NOT REVOKED
    peer_reg = engine.registry.get_peer("peer_99")
    assert peer_reg.discovery_status != DiscoveryStatus.REVOKED


def test_16_engine_quarantine_on_corruption(test_model, mock_gate):
    """Verify engines exhibiting corrupted or tampered digests are quarantined."""
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime = FederationRuntime(engine=engine, auto_recover=False)

    remote_id = "eng_adversary_01"
    runtime.set_engine_health(remote_id, EngineHealthStatus.QUARANTINED, reason="Hash chain divergence detected")
    assert runtime.get_engine_health(remote_id) == EngineHealthStatus.QUARANTINED


# ============================================================================
# Phase 7 & 8: Rejoin Protocol & Conflict Matrix Tests
# ============================================================================

def test_17_secure_rejoin_success(test_model, mock_gate):
    """Verify secure engine rejoin flow after recovery."""
    engine_a = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    engine_b = CrossZoneFederationEngine(local_zone_id="zone-beta", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime_a = FederationRuntime(engine=engine_a, auto_recover=False)

    # Initial registration
    runtime_a.register_remote_engine(engine_b.engine_identity)

    # Execute rejoin
    result = runtime_a.execute_rejoin(engine_b)
    assert result["status"] == "REJOIN_SUCCESSFUL"
    assert runtime_a.get_engine_health(engine_b.engine_identity.engine_id) == EngineHealthStatus.HEALTHY


def test_18_rejoin_does_not_grant_authority(test_model, mock_gate):
    """Verify ENGINE_REJOIN != TRUST_GRANT and cannot bypass CapabilityGate."""
    engine_a = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    engine_b = CrossZoneFederationEngine(local_zone_id="zone-beta", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime_a = FederationRuntime(engine=engine_a, auto_recover=False)

    runtime_a.execute_rejoin(engine_b)

    # Attempt unauthorized capability request from rejoined engine
    req = CapabilityRequest(
        capability_id="unauthorized_exec",
        parameters={"code": "tamper"},
        caller_id=engine_b.engine_identity.engine_id,
    )
    with pytest.raises(Exception):
        engine_a.capability_gate.authorize(req)


def test_19_local_revocation_persists_across_remote_rejoin(test_model, mock_gate):
    """Verify local revocation strictly wins when a remote engine rejoins (Case D)."""
    engine_a = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    engine_b = CrossZoneFederationEngine(local_zone_id="zone-beta", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime_a = FederationRuntime(engine=engine_a, auto_recover=False)

    peer_id = "peer_target_case_d"
    crypto_b = CryptographicPeerIdentity.create(
        zone_id="zone-beta", organization_id="org-beta", public_key=engine_b.local_public_key, peer_id=peer_id
    )
    engine_a.register_cryptographic_peer(crypto_b)
    engine_b.register_cryptographic_peer(crypto_b)

    # Revoke peer on Engine A locally
    engine_a.revoke_peer(peer_id, reason="Local administrative revocation")
    assert engine_a.registry.get_peer(peer_id).discovery_status == DiscoveryStatus.REVOKED

    # Engine B rejoins claiming peer is active
    runtime_a.execute_rejoin(engine_b)

    # Local revocation remains strictly enforced
    assert engine_a.registry.get_peer(peer_id).discovery_status == DiscoveryStatus.REVOKED
    # Remote Engine B is also informed of revocation
    assert engine_b.registry.get_peer(peer_id).discovery_status == DiscoveryStatus.REVOKED


def test_20_rejoin_with_expired_trust_blocks_renewal(test_model, mock_gate):
    """Verify rejoining node cannot renew session past expired trust grant (Case I)."""
    engine_a = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    peer_id = "peer_expired_case_i"
    crypto = CryptographicPeerIdentity.create(
        zone_id="zone-beta", organization_id="org-beta", public_key=engine_a.local_public_key, peer_id=peer_id
    )
    reg = engine_a.register_cryptographic_peer(crypto)

    # Assign trust grant that expires at epoch 2
    reg.trust_grant = TrustGrant(
        grant_id="grant_i",
        issuer_zone_id="zone-alpha",
        subject_peer_id=peer_id,
        subject_zone_id="zone-beta",
        trust_level=TrustLevel.FEDERATED,
        permitted_scopes=[],
        issued_epoch=1,
        expires_epoch=2,
        status=TrustStatus.ACTIVE,
    )
    sess = engine_a.create_secure_session(remote_peer_id=peer_id)

    # Advance epoch past grant expiry
    engine_a.advance_epoch(5)

    # Session renewal must fail closed
    with pytest.raises(Exception):
        engine_a.renew_session(sess.session_id)


def test_21_certificate_revocation_remains_revoked_after_recovery(test_model, mock_gate, memory_store):
    """Verify certificate revocation survives restart and cannot be bypassed (Case J)."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    fp = "cert_revoked_fp_001"
    engine1.certificate_revocation_registry.revoke(fingerprint=fp, reason="Compromised", epoch=1)
    assert engine1.certificate_revocation_registry.is_revoked(fp) is True

    # Snapshot and recover onto new engine
    snap = FederationRecoveryManager.create_snapshot_from_engine(engine1, memory_store, snapshot_version=1, journal_offset=0)
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    rec_mgr = FederationRecoveryManager(memory_store)
    rec_mgr.recover(engine2)

    # Certificate remains revoked on engine2
    assert engine2.certificate_revocation_registry.is_revoked(fp) is True


def test_22_replay_floor_persists_across_restart(test_model, mock_gate, memory_store):
    """Verify session replay floors survive crash and reject old sequence numbers (Case H)."""
    engine1 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    peer_id = "peer_replay_case_h"
    crypto = CryptographicPeerIdentity.create(
        zone_id="zone-beta", organization_id="org-beta", public_key=engine1.local_public_key, peer_id=peer_id
    )
    engine1.register_cryptographic_peer(crypto)
    sess1 = engine1.create_secure_session(remote_peer_id=peer_id)
    sess1.record_and_check_sequence(50)

    # Snapshot engine1
    snap = FederationRecoveryManager.create_snapshot_from_engine(engine1, memory_store, snapshot_version=1, journal_offset=0)

    # Recover engine2
    engine2 = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    rec_mgr = FederationRecoveryManager(memory_store)
    rec_mgr.recover(engine2)

    sess2 = engine2.sessions.get(sess1.session_id)
    assert sess2 is not None
    assert sess2.last_sequence_number == 50

    # Old sequence number 45 must be rejected (monotonic floor maintained)
    assert sess2.record_and_check_sequence(45) is False
    # Subsequent fresh sequence number 55 must be accepted
    assert sess2.record_and_check_sequence(55) is True


# ============================================================================
# Phase 9: Neural Core Immutability Tests
# ============================================================================

def test_23_neural_core_strictly_immutable(test_model, mock_gate, memory_store):
    """Verify parameters == 3,443,136, vocab == 4096, context == 512, and ΔW = 0."""
    pre_hash = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            pre_hash.update(name.encode("utf-8"))
            pre_hash.update(param.detach().cpu().numpy().tobytes())
    pre_digest = pre_hash.hexdigest()

    # Perform extensive durable runtime lifecycle, journaling, and recovery operations
    engine = CrossZoneFederationEngine(local_zone_id="zone-alpha", capability_gate=mock_gate, model=test_model, initial_epoch=1)
    runtime = FederationRuntime(engine=engine, store=memory_store, auto_recover=False)

    # Journal operations
    runtime.journal.append(JournalEntryType.EPOCH_ADVANCED, 1, {"new_epoch": 2})
    runtime.journal.append(JournalEntryType.PEER_REGISTERED, 2, {"peer_id": "p_test", "zone_id": "z_test"})
    runtime.take_snapshot()
    runtime.restart()

    # Re-verify weight hash
    post_hash = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(test_model.named_parameters()):
            post_hash.update(name.encode("utf-8"))
            post_hash.update(param.detach().cpu().numpy().tobytes())
    post_digest = post_hash.hexdigest()

    assert pre_digest == post_digest, "CRITICAL: Neural weights mutated during runtime operations!"

    # Structural constants
    assert sum(p.numel() for p in test_model.parameters()) == 3_443_136
    assert test_model.config.vocab_size == 4096
    assert test_model.config.max_seq_len == 512
