"""
Strongly Typed Data Models for Distributed Federation Coordination (Step 33).

Defines foundational primitives for multi-engine federation coordination:
- Engine Identity (distinguished from Peer, Tenant, Session, Key, and Cert identity)
- Monotonic Security State Versions
- Reproducible, Canonical Security State Digests (Replay, Trust, Revocation, Peer, Composite)
- Federation Handshake Requests & Responses
- Bounded Replay, Trust, and Revocation Synchronization Messages

CRITICAL ARCHITECTURAL AXIOMS:
1. ENGINE_IDENTITY != AUTHORITY & REMOTE_ENGINE != LOCAL_AUTHORITY:
   An engine identity proves host participation in coordination; it confers
   ZERO local execution authority, capability permissions, or trust grants.
2. FEDERATION_COORDINATION != AUTHORITY_TRANSFER:
   Coordination synchronizes security state (replays, revocations, digests);
   it never transfers capability authorization or modifies local policy.
3. CANONICAL DETERMINISM & ZERO SECRET LEAKAGE:
   All digests are SHA-256 over deterministic, canonical JSON with sorted keys.
   Private keys, session keys, model weights, and scratchpad activations are
   NEVER included in digests, serialization, representations, or audit records.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple


# ============================================================================
# Hard Architectural Ceilings (Centrally Enforced)
# ============================================================================

MAX_FEDERATION_ENGINES = 16
MAX_SYNC_MESSAGE_IDS = 100
MAX_REVOCATION_SYNC_BATCH = 100
MAX_STATE_HISTORY = 100
FEDERATION_PROTOCOL_VERSION = "33.0"


# ============================================================================
# Enums
# ============================================================================

class HandshakeStatus(str, Enum):
    """Lifecycle status of a federation coordination handshake."""
    INITIATED = "INITIATED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INCOMPATIBLE_PROTOCOL = "INCOMPATIBLE_PROTOCOL"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    DIGEST_CONFLICT = "DIGEST_CONFLICT"
    FAILED = "FAILED"


class RevocationTargetType(str, Enum):
    """Target classification of a propagated revocation event."""
    PEER = "PEER"
    SESSION = "SESSION"
    KEY = "KEY"
    CERTIFICATE = "CERTIFICATE"
    TRUST_GRANT = "TRUST_GRANT"


# ============================================================================
# Phase 1: Engine Identity
# ============================================================================

@dataclass(frozen=True)
class FederationEngineIdentity:
    """
    Immutable identity for a local or remote federation coordination engine.

    CRITICAL ARCHITECTURAL AXIOMS:
    1. ENGINE_IDENTITY != PEER_IDENTITY:
       Identifies the coordinating runtime instance/node, not the peer entity.
    2. ENGINE_IDENTITY != AUTHORITY:
       A remote engine never gains authority over local decisions or capabilities.
    3. REMOTE_ENGINE != LOCAL_AUTHORITY:
       All local actions remain governed by local policy and CapabilityGate.
    """
    engine_id: str
    zone_id: str
    created_epoch: int = 1
    identity_fingerprint: str = ""
    protocol_version: str = FEDERATION_PROTOCOL_VERSION
    architecture_version: str = "0.1"
    public_key_fingerprint: Optional[str] = None

    def canonical_serialize(self) -> bytes:
        """Produce deterministic canonical JSON representation for fingerprinting."""
        payload = {
            "architecture_version": self.architecture_version,
            "created_epoch": self.created_epoch,
            "engine_id": self.engine_id,
            "protocol_version": self.protocol_version,
            "public_key_fingerprint": self.public_key_fingerprint or "",
            "zone_id": self.zone_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_fingerprint(self) -> str:
        """Compute SHA-256 digest of canonical serialization."""
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Safe non-secret dictionary export."""
        return {
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "created_epoch": self.created_epoch,
            "identity_fingerprint": self.identity_fingerprint or self.compute_fingerprint(),
            "protocol_version": self.protocol_version,
            "architecture_version": self.architecture_version,
            "public_key_fingerprint": self.public_key_fingerprint,
        }


# ============================================================================
# Phase 2 & 4: Versioned Security State
# ============================================================================

@dataclass(frozen=True)
class SecurityStateVersion:
    """
    Monotonically increasing version tracking for engine security state.
    Requires version >= 1 and epoch >= 1.
    """
    version: int
    epoch: int
    engine_id: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"Security state version must be >= 1, got {self.version}.")
        if self.epoch < 1:
            raise ValueError(f"Security state epoch must be >= 1, got {self.epoch}.")
        if not self.engine_id:
            raise ValueError("Engine ID cannot be empty.")

    def is_newer_than(self, other: "SecurityStateVersion") -> bool:
        """Check if this version is strictly newer than other."""
        if self.version > other.version:
            return True
        if self.version == other.version and self.epoch > other.epoch:
            return True
        return False

    def is_stale_compared_to(self, other: "SecurityStateVersion") -> bool:
        """Check if this version is strictly older than other."""
        if self.version < other.version:
            return True
        if self.version == other.version and self.epoch < other.epoch:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "epoch": self.epoch,
            "engine_id": self.engine_id,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Phase 2 & 7: Deterministic State Digests
# ============================================================================

@dataclass(frozen=True)
class ReplayStateDigest:
    """
    Deterministic digest of session replay protection state.
    Contains no session secrets or raw message payloads.
    """
    engine_id: str
    session_id: str
    epoch: int
    highest_sequence_number: int
    message_count: int
    message_id_digest: str
    state_version: int

    def canonical_serialize(self) -> bytes:
        payload = {
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "highest_sequence_number": self.highest_sequence_number,
            "message_count": self.message_count,
            "message_id_digest": self.message_id_digest,
            "session_id": self.session_id,
            "state_version": self.state_version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "highest_sequence_number": self.highest_sequence_number,
            "message_count": self.message_count,
            "message_id_digest": self.message_id_digest,
            "state_version": self.state_version,
            "digest": self.compute_digest(),
        }


@dataclass(frozen=True)
class TrustStateDigest:
    """
    Deterministic digest of active/revoked trust grants known to the engine.
    """
    engine_id: str
    zone_id: str
    epoch: int
    active_grants_count: int
    revoked_grants_count: int
    grant_ids_hash: str
    state_version: int

    def canonical_serialize(self) -> bytes:
        payload = {
            "active_grants_count": self.active_grants_count,
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "grant_ids_hash": self.grant_ids_hash,
            "revoked_grants_count": self.revoked_grants_count,
            "state_version": self.state_version,
            "zone_id": self.zone_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "epoch": self.epoch,
            "active_grants_count": self.active_grants_count,
            "revoked_grants_count": self.revoked_grants_count,
            "grant_ids_hash": self.grant_ids_hash,
            "state_version": self.state_version,
            "digest": self.compute_digest(),
        }


@dataclass(frozen=True)
class RevocationStateDigest:
    """
    Deterministic digest of cumulative revocations (peer, key, session, cert, grant).
    """
    engine_id: str
    epoch: int
    revocation_count: int
    revocation_root_hash: str
    state_version: int

    def canonical_serialize(self) -> bytes:
        payload = {
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "revocation_count": self.revocation_count,
            "revocation_root_hash": self.revocation_root_hash,
            "state_version": self.state_version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "revocation_count": self.revocation_count,
            "revocation_root_hash": self.revocation_root_hash,
            "state_version": self.state_version,
            "digest": self.compute_digest(),
        }


@dataclass(frozen=True)
class PeerStateDigest:
    """
    Deterministic digest of registered and active peers in the engine registry.
    """
    engine_id: str
    zone_id: str
    epoch: int
    registered_peers_count: int
    active_peers_count: int
    peer_ids_hash: str
    state_version: int

    def canonical_serialize(self) -> bytes:
        payload = {
            "active_peers_count": self.active_peers_count,
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "peer_ids_hash": self.peer_ids_hash,
            "registered_peers_count": self.registered_peers_count,
            "state_version": self.state_version,
            "zone_id": self.zone_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "epoch": self.epoch,
            "registered_peers_count": self.registered_peers_count,
            "active_peers_count": self.active_peers_count,
            "peer_ids_hash": self.peer_ids_hash,
            "state_version": self.state_version,
            "digest": self.compute_digest(),
        }


@dataclass(frozen=True)
class FederationSecurityStateDigest:
    """
    Master composite digest over all security state facets of an engine.
    Ensures complete, cryptographic reproducibility of distributed security state.
    """
    engine_id: str
    zone_id: str
    epoch: int
    state_version: int
    peer_digest: str
    replay_digest: str
    trust_digest: str
    revocation_digest: str

    def canonical_serialize(self) -> bytes:
        payload = {
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "peer_digest": self.peer_digest,
            "replay_digest": self.replay_digest,
            "revocation_digest": self.revocation_digest,
            "state_version": self.state_version,
            "trust_digest": self.trust_digest,
            "zone_id": self.zone_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_composite_digest(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "epoch": self.epoch,
            "state_version": self.state_version,
            "peer_digest": self.peer_digest,
            "replay_digest": self.replay_digest,
            "trust_digest": self.trust_digest,
            "revocation_digest": self.revocation_digest,
            "composite_digest": self.compute_composite_digest(),
        }


# ============================================================================
# Phase 8: Federation State Handshake Models
# ============================================================================

@dataclass(frozen=True)
class FederationHandshakeRequest:
    """
    Bounded coordination handshake request emitted by an engine.

    CRITICAL AXIOMS:
    - FEDERATION_HANDSHAKE != TRUST_GRANT
    - FEDERATION_HANDSHAKE != AUTHORIZATION
    """
    sender_identity: FederationEngineIdentity
    protocol_version: str
    epoch: int
    state_version: int
    composite_digest: str
    replay_digest: str
    trust_digest: str
    revocation_digest: str
    nonce: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_identity": self.sender_identity.to_dict(),
            "protocol_version": self.protocol_version,
            "epoch": self.epoch,
            "state_version": self.state_version,
            "composite_digest": self.composite_digest,
            "replay_digest": self.replay_digest,
            "trust_digest": self.trust_digest,
            "revocation_digest": self.revocation_digest,
            "nonce": self.nonce,
        }


@dataclass(frozen=True)
class FederationHandshakeResponse:
    """
    Bounded coordination handshake response emitted by the receiving engine.
    """
    responder_identity: FederationEngineIdentity
    status: HandshakeStatus
    protocol_version: str
    epoch: int
    state_version: int
    sync_required: bool
    details: Dict[str, Any] = field(default_factory=dict)
    nonce: str = ""

    def is_accepted(self) -> bool:
        return self.status == HandshakeStatus.ACCEPTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "responder_identity": self.responder_identity.to_dict(),
            "status": self.status.value,
            "protocol_version": self.protocol_version,
            "epoch": self.epoch,
            "state_version": self.state_version,
            "sync_required": self.sync_required,
            "details": dict(self.details),
            "nonce": self.nonce,
        }


# ============================================================================
# Phase 3: Bounded Replay Synchronization
# ============================================================================

@dataclass(frozen=True)
class ReplaySyncMessage:
    """
    Advisory replay state information exchanged between engines.
    Local replay protection remains unconditionally authoritative.
    """
    engine_id: str
    session_id: str
    epoch: int
    highest_sequence_number: int
    bounded_message_id_digest: str
    recent_message_ids: List[str]
    state_version: int

    def __post_init__(self) -> None:
        if len(self.recent_message_ids) > MAX_SYNC_MESSAGE_IDS:
            raise ValueError(
                f"Recent message IDs count ({len(self.recent_message_ids)}) "
                f"exceeds ceiling ({MAX_SYNC_MESSAGE_IDS})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "highest_sequence_number": self.highest_sequence_number,
            "bounded_message_id_digest": self.bounded_message_id_digest,
            "recent_message_ids": list(self.recent_message_ids),
            "state_version": self.state_version,
        }


# ============================================================================
# Phase 5: Trust State Consistency
# ============================================================================

@dataclass(frozen=True)
class TrustSyncRecord:
    """
    Information report regarding a peer's trust status in a remote engine.

    CRITICAL AXIOMS:
    - REMOTE_TRUST_CLAIM != LOCAL_TRUST_AUTHORIZATION
    - A remote engine CANNOT grant itself or another peer local capabilities.
    """
    subject_peer_id: str
    subject_zone_id: str
    trust_level: str
    status: str
    expires_epoch: int
    permitted_scopes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_peer_id": self.subject_peer_id,
            "subject_zone_id": self.subject_zone_id,
            "trust_level": self.trust_level,
            "status": self.status,
            "expires_epoch": self.expires_epoch,
            "permitted_scopes": list(self.permitted_scopes),
        }


@dataclass(frozen=True)
class TrustSyncMessage:
    """
    Summary message conveying remote trust state claims.
    """
    engine_id: str
    zone_id: str
    epoch: int
    state_version: int
    trust_records: List[TrustSyncRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "epoch": self.epoch,
            "state_version": self.state_version,
            "trust_records": [r.to_dict() for r in self.trust_records],
        }


# ============================================================================
# Phase 6: Revocation Propagation
# ============================================================================

@dataclass(frozen=True)
class RevocationSyncRecord:
    """
    Deterministic revocation claim for propagation across engines.

    CRITICAL INVARIANT:
    REVOKED -> NEVER ACTIVE AGAIN.
    """
    revocation_id: str
    target_type: RevocationTargetType
    target_id: str
    reason: str
    revoked_epoch: int
    revoked_by: str
    timestamp: float = field(default_factory=time.time)

    def canonical_serialize(self) -> bytes:
        payload = {
            "reason": self.reason,
            "revocation_id": self.revocation_id,
            "revoked_by": self.revoked_by,
            "revoked_epoch": self.revoked_epoch,
            "target_id": self.target_id,
            "target_type": self.target_type.value,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revocation_id": self.revocation_id,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "reason": self.reason,
            "revoked_epoch": self.revoked_epoch,
            "revoked_by": self.revoked_by,
            "timestamp": self.timestamp,
            "record_hash": self.compute_hash(),
        }


@dataclass(frozen=True)
class RevocationSyncMessage:
    """
    Batch of revocation sync records for propagation.
    """
    engine_id: str
    epoch: int
    state_version: int
    revocations: List[RevocationSyncRecord]

    def __post_init__(self) -> None:
        if len(self.revocations) > MAX_REVOCATION_SYNC_BATCH:
            raise ValueError(
                f"Revocation sync batch size ({len(self.revocations)}) "
                f"exceeds ceiling ({MAX_REVOCATION_SYNC_BATCH})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "epoch": self.epoch,
            "state_version": self.state_version,
            "revocations": [r.to_dict() for r in self.revocations],
        }
