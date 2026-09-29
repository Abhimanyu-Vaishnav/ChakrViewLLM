"""
Strongly Typed Data Models for Cross-Zone Peering & Trust Negotiation (Step 29).

Defines foundational primitives for multi-zone cognitive federation:
- Peer Identity & Administrative Domain Metadata
- Peer Discovery States & Attestation Claims
- Bounded Trust Levels, Grants, and Statuses
- Federation Scopes, Policy Rules & Default-Deny Boundaries
- Trust Negotiation Declarations and Agreements
- Trust Revocation Records & Expiration Tracking
- Bounded Audit Records & Sanitized Telemetry Traces

CRITICAL ARCHITECTURAL AXIOMS:
1. CROSS-ZONE FEDERATION != AUTHORITY TRANSFER:
   A peer may provide evidence, verification, or compute, but never gains local authority.
2. LOCAL_AUTHORITY > PEER_AUTHORITY & PEER_TRUST != PEER_AUTHORITY:
   Trust is a bounded, scoped, and revocable capability grant, never blanket permission.
3. IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION:
   Step 29 provides deterministic local identity representation. Production PKI
   and Ed25519 authentication remain deferred.
4. STRICT TENANT & ZONE ISOLATION:
   No cross-tenant crossover; no access to model weights, activations, logits,
   private sessions, or private memory.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set


# ============================================================================
# Hard Architectural Ceilings (Centrally Enforced)
# ============================================================================

MAX_PEERS_TOTAL = 32
MAX_PEERS_PER_ZONE = 16
MAX_ACTIVE_PEERS = 8
MAX_DELEGATION_DEPTH = 2
MAX_CONCURRENT_PEER_TASKS = 8
DEFAULT_TRUST_TTL_EPOCHS = 10
MAX_AUDIT_LOG_ENTRIES = 1000
MAX_METADATA_BYTES = 65536


# ============================================================================
# Enums: Discovery, Trust, Attestation, Scopes & Audit
# ============================================================================

class DiscoveryStatus(str, Enum):
    """Lifecycle state of peer discovery."""
    DISCOVERED = "DISCOVERED"       # Peer announced/observed, zero verification
    UNVERIFIED = "UNVERIFIED"       # Identity presented, attestation pending
    VERIFIED = "VERIFIED"           # Attestation verified by local policy
    REJECTED = "REJECTED"           # Incompatible protocol or rejected by policy
    REVOKED = "REVOKED"             # Trust or registration administratively revoked
    EXPIRED = "EXPIRED"             # Trust grant or registration has elapsed


class TrustLevel(str, Enum):
    """
    Explicit bounded trust hierarchy.
    Trust is NEVER a boolean.
    """
    NONE = "NONE"                   # Default unverified standing; zero trust
    IDENTIFIED = "IDENTIFIED"       # Valid identity structure; no execution rights
    ATTESTED = "ATTESTED"           # Architecture/runtime claims validated
    LIMITED_TRUST = "LIMITED_TRUST" # Read-only capability & evidence exchange
    FEDERATED = "FEDERATED"         # Bounded task delegation permitted under policy
    REVOKED = "REVOKED"             # Explicitly revoked; all requests blocked


class TrustStatus(str, Enum):
    """Operational status of a TrustGrant."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    SUSPENDED = "SUSPENDED"


class AttestationStatus(str, Enum):
    """Validation standing of an attestation claim."""
    DECLARED = "DECLARED"           # Raw claim submitted by peer
    VERIFIED = "VERIFIED"           # Claim structurally and architecturally verified
    CRYPTOGRAPHICALLY_AUTHENTICATED = "CRYPTOGRAPHICALLY_AUTHENTICATED"  # Deferred


class FederationScope(str, Enum):
    """Explicit permitted federation permissions."""
    ALLOW_EVIDENCE_EXCHANGE = "ALLOW_EVIDENCE_EXCHANGE"
    ALLOW_VERIFICATION = "ALLOW_VERIFICATION"
    ALLOW_COGNITIVE_TASK_DELEGATION = "ALLOW_COGNITIVE_TASK_DELEGATION"
    ALLOW_CAPABILITY_METADATA = "ALLOW_CAPABILITY_METADATA"
    ALLOW_MODEL_METADATA = "ALLOW_MODEL_METADATA"
    ALLOW_MEMORY_METADATA = "ALLOW_MEMORY_METADATA"


class ProhibitedScope(str, Enum):
    """Strict architectural prohibitions that can NEVER be granted."""
    DENY_MODEL_WEIGHT_ACCESS = "DENY_MODEL_WEIGHT_ACCESS"
    DENY_PRIVATE_MEMORY = "DENY_PRIVATE_MEMORY"
    DENY_TENANT_CROSSOVER = "DENY_TENANT_CROSSOVER"
    DENY_AUTHORITY_TRANSFER = "DENY_AUTHORITY_TRANSFER"


class AuditEventType(str, Enum):
    """Audit taxonomy for cross-zone federation operations."""
    PEER_DISCOVERED = "PEER_DISCOVERED"
    IDENTITY_PRESENTED = "IDENTITY_PRESENTED"
    ATTESTATION_RECEIVED = "ATTESTATION_RECEIVED"
    ATTESTATION_ACCEPTED = "ATTESTATION_ACCEPTED"
    ATTESTATION_REJECTED = "ATTESTATION_REJECTED"
    TRUST_NEGOTIATED = "TRUST_NEGOTIATED"
    POLICY_ACCEPTED = "POLICY_ACCEPTED"
    POLICY_REJECTED = "POLICY_REJECTED"
    FEDERATION_ESTABLISHED = "FEDERATION_ESTABLISHED"
    FEDERATION_REVOKED = "FEDERATION_REVOKED"
    FEDERATION_EXPIRED = "FEDERATION_EXPIRED"
    REQUEST_DENIED = "REQUEST_DENIED"
    ISOLATION_VIOLATION = "ISOLATION_VIOLATION"
    PEER_AUTHENTICATED = "PEER_AUTHENTICATED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_ACTIVATED = "SESSION_ACTIVATED"
    SESSION_RENEWED = "SESSION_RENEWED"
    SESSION_RENEWAL_FAILED = "SESSION_RENEWAL_FAILED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_TERMINATED = "SESSION_TERMINATED"
    SESSION_REVOKED = "SESSION_REVOKED"
    SESSION_FAILED = "SESSION_FAILED"
    KEY_ROTATED = "KEY_ROTATED"
    KEY_ROTATION_STARTED = "KEY_ROTATION_STARTED"
    KEY_ROTATION_COMPLETED = "KEY_ROTATION_COMPLETED"
    KEY_ROTATION_FAILED = "KEY_ROTATION_FAILED"
    KEY_REVOKED = "KEY_REVOKED"
    REPLAY_ATTACK_DETECTED = "REPLAY_ATTACK_DETECTED"
    TLS_HANDSHAKE_COMPLETED = "TLS_HANDSHAKE_COMPLETED"
    TLS_HANDSHAKE_FAILED = "TLS_HANDSHAKE_FAILED"
    CERTIFICATE_VALIDATED = "CERTIFICATE_VALIDATED"
    CERTIFICATE_VALIDATION_FAILED = "CERTIFICATE_VALIDATION_FAILED"
    CERTIFICATE_EXPIRED = "CERTIFICATE_EXPIRED"
    CERTIFICATE_REVOKED = "CERTIFICATE_REVOKED"
    CERTIFICATE_ROTATED = "CERTIFICATE_ROTATED"
    CERTIFICATE_ROTATION_STARTED = "CERTIFICATE_ROTATION_STARTED"
    CERTIFICATE_ROTATION_COMPLETED = "CERTIFICATE_ROTATION_COMPLETED"
    CERTIFICATE_ROTATION_FAILED = "CERTIFICATE_ROTATION_FAILED"
    PEER_BINDING_VERIFIED = "PEER_BINDING_VERIFIED"
    PEER_BINDING_FAILED = "PEER_BINDING_FAILED"
    INSECURE_DOWNGRADE_ATTEMPTED = "INSECURE_DOWNGRADE_ATTEMPTED"
    REVOCATION_CASCADE_TRIGGERED = "REVOCATION_CASCADE_TRIGGERED"
    REVOCATION_CASCADE_COMPLETED = "REVOCATION_CASCADE_COMPLETED"
    STALE_AUTHORIZATION_DENIED = "STALE_AUTHORIZATION_DENIED"

    # Step 33 Distributed Federation Coordination Audit Events
    ENGINE_REGISTERED = "ENGINE_REGISTERED"
    ENGINE_AUTHENTICATED = "ENGINE_AUTHENTICATED"
    FEDERATION_HANDSHAKE_STARTED = "FEDERATION_HANDSHAKE_STARTED"
    FEDERATION_HANDSHAKE_COMPLETED = "FEDERATION_HANDSHAKE_COMPLETED"
    FEDERATION_HANDSHAKE_FAILED = "FEDERATION_HANDSHAKE_FAILED"
    STATE_SYNC_STARTED = "STATE_SYNC_STARTED"
    STATE_SYNC_COMPLETED = "STATE_SYNC_COMPLETED"
    STATE_SYNC_FAILED = "STATE_SYNC_FAILED"
    REPLAY_STATE_SYNCED = "REPLAY_STATE_SYNCED"
    TRUST_STATE_SYNCED = "TRUST_STATE_SYNCED"
    REVOCATION_STATE_SYNCED = "REVOCATION_STATE_SYNCED"
    STATE_VERSION_CONFLICT = "STATE_VERSION_CONFLICT"
    STATE_DIGEST_CONFLICT = "STATE_DIGEST_CONFLICT"
    REMOTE_STATE_REJECTED = "REMOTE_STATE_REJECTED"
    REVOCATION_PROPAGATED = "REVOCATION_PROPAGATED"
    REVOCATION_DUPLICATE_IGNORED = "REVOCATION_DUPLICATE_IGNORED"
    STALE_STATE_REJECTED = "STALE_STATE_REJECTED"

    # Step 34 Durable Security State & Failure Recovery Audit Events
    SNAPSHOT_CREATED = "SNAPSHOT_CREATED"
    SNAPSHOT_RESTORED = "SNAPSHOT_RESTORED"
    SNAPSHOT_FAILED = "SNAPSHOT_FAILED"
    JOURNAL_APPENDED = "JOURNAL_APPENDED"
    JOURNAL_REPLAYED = "JOURNAL_REPLAYED"
    JOURNAL_CORRUPTION_DETECTED = "JOURNAL_CORRUPTION_DETECTED"
    JOURNAL_TRUNCATION_DETECTED = "JOURNAL_TRUNCATION_DETECTED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    RECOVERY_FAILED_CLOSED = "RECOVERY_FAILED_CLOSED"
    ENGINE_REJOIN_STARTED = "ENGINE_REJOIN_STARTED"
    ENGINE_REJOIN_COMPLETED = "ENGINE_REJOIN_COMPLETED"
    ENGINE_REJOIN_FAILED = "ENGINE_REJOIN_FAILED"
    ENGINE_HEALTH_CHANGED = "ENGINE_HEALTH_CHANGED"
    ENGINE_QUARANTINED = "ENGINE_QUARANTINED"

    # Step 35 Federation Discovery & Secure Membership Audit Events
    NODE_DISCOVERED = "NODE_DISCOVERED"
    NODE_AUTHENTICATION_STARTED = "NODE_AUTHENTICATION_STARTED"
    NODE_AUTHENTICATED = "NODE_AUTHENTICATED"
    NODE_AUTHENTICATION_FAILED = "NODE_AUTHENTICATION_FAILED"
    NODE_MEMBERSHIP_GRANTED = "NODE_MEMBERSHIP_GRANTED"
    NODE_MEMBERSHIP_SUSPENDED = "NODE_MEMBERSHIP_SUSPENDED"
    NODE_QUARANTINED = "NODE_QUARANTINED"
    NODE_REVOKED = "NODE_REVOKED"
    NODE_TERMINATED = "NODE_TERMINATED"
    NODE_REMOVED = "NODE_REMOVED"
    HEARTBEAT_SENT = "HEARTBEAT_SENT"
    HEARTBEAT_RECEIVED = "HEARTBEAT_RECEIVED"
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"
    HEARTBEAT_FAILED = "HEARTBEAT_FAILED"
    MEMBERSHIP_STATE_TRANSITION = "MEMBERSHIP_STATE_TRANSITION"

    # Step 36 Federation Message Transport Audit Events
    CONNECTION_ATTEMPTED = "CONNECTION_ATTEMPTED"
    CONNECTION_ESTABLISHED = "CONNECTION_ESTABLISHED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    FRAME_REJECTED = "FRAME_REJECTED"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    MESSAGE_REJECTED = "MESSAGE_REJECTED"
    REPLAY_REJECTED = "REPLAY_REJECTED"
    SEQUENCE_REJECTED = "SEQUENCE_REJECTED"
    MESSAGE_DISPATCHED = "MESSAGE_DISPATCHED"
    CONNECTION_DEGRADED = "CONNECTION_DEGRADED"
    CONNECTION_CLOSED = "CONNECTION_CLOSED"
    RECONNECT_ATTEMPTED = "RECONNECT_ATTEMPTED"
    RECONNECT_SUCCEEDED = "RECONNECT_SUCCEEDED"
    RECONNECT_FAILED = "RECONNECT_FAILED"

    # Step 37 Distributed Resource & Capability Advertisement Audit Events
    RESOURCE_PROFILE_DETECTED = "RESOURCE_PROFILE_DETECTED"
    RESOURCE_ADVERTISEMENT_PUBLISHED = "RESOURCE_ADVERTISEMENT_PUBLISHED"
    RESOURCE_ADVERTISEMENT_RECEIVED = "RESOURCE_ADVERTISEMENT_RECEIVED"
    RESOURCE_ADVERTISEMENT_ACCEPTED = "RESOURCE_ADVERTISEMENT_ACCEPTED"
    RESOURCE_ADVERTISEMENT_REJECTED = "RESOURCE_ADVERTISEMENT_REJECTED"
    RESOURCE_ADVERTISEMENT_STALE = "RESOURCE_ADVERTISEMENT_STALE"
    RESOURCE_QUERY_DISPATCHED = "RESOURCE_QUERY_DISPATCHED"
    RESOURCE_QUERY_RECEIVED = "RESOURCE_QUERY_RECEIVED"

    # Step 38 Distributed Task Orchestration Audit Events
    TASK_CREATED = "TASK_CREATED"
    TASK_DECOMPOSED = "TASK_DECOMPOSED"
    TASK_SCHEDULED = "TASK_SCHEDULED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    TASK_ASSIGNMENT_ACCEPTED = "TASK_ASSIGNMENT_ACCEPTED"
    TASK_ASSIGNMENT_REJECTED = "TASK_ASSIGNMENT_REJECTED"
    TASK_STARTED = "TASK_STARTED"
    TASK_CHECKPOINTED = "TASK_CHECKPOINTED"
    TASK_PROGRESS = "TASK_PROGRESS"
    TASK_WORKER_FAILED = "TASK_WORKER_FAILED"
    TASK_REASSIGNED = "TASK_REASSIGNED"
    TASK_RESULT_RECEIVED = "TASK_RESULT_RECEIVED"
    TASK_RESULT_VALIDATED = "TASK_RESULT_VALIDATED"
    TASK_RESULT_REJECTED = "TASK_RESULT_REJECTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_RECOVERED = "TASK_RECOVERED"

    # Step 39 Federated Execution Continuity & Work Resilience Audit Events
    TASK_CHECKPOINT_COMMITTED = "TASK_CHECKPOINT_COMMITTED"
    TASK_CHECKPOINT_SUPERSEDED = "TASK_CHECKPOINT_SUPERSEDED"
    TASK_LEASE_GRANTED = "TASK_LEASE_GRANTED"
    TASK_LEASE_RENEWED = "TASK_LEASE_RENEWED"
    TASK_LEASE_EXPIRED = "TASK_LEASE_EXPIRED"
    TASK_ATTEMPT_FENCED = "TASK_ATTEMPT_FENCED"
    TASK_WORK_MIGRATED = "TASK_WORK_MIGRATED"
    TASK_RECOVERY_INITIATED = "TASK_RECOVERY_INITIATED"
    TASK_RECOVERY_COMPLETED = "TASK_RECOVERY_COMPLETED"
    TASK_RECOVERY_FAILED = "TASK_RECOVERY_FAILED"




# ============================================================================
# Strongly Typed Models
# ============================================================================

@dataclass(frozen=True)
class PeerIdentity:
    """
    Deterministic identity for a remote or peered cognitive zone.
    CRITICAL: IDENTITY != CRYPTOGRAPHIC_AUTHENTICATION.
    """
    peer_id: str
    zone_id: str
    organization_id: str
    protocol_version: str = "29.0"
    architecture_version: str = "0.1"
    capability_profile: Dict[str, Any] = field(default_factory=dict)
    supported_features: List[str] = field(default_factory=list)
    created_epoch: int = 1
    fingerprint: str = ""

    def canonical_serialize(self) -> bytes:
        """Produce canonical, deterministic JSON representation for fingerprinting."""
        payload = {
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "organization_id": self.organization_id,
            "protocol_version": self.protocol_version,
            "architecture_version": self.architecture_version,
            "capability_profile": self.capability_profile,
            "supported_features": sorted(self.supported_features),
            "created_epoch": self.created_epoch,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_fingerprint(self) -> str:
        """Compute SHA-256 digest of canonical serialization."""
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "organization_id": self.organization_id,
            "protocol_version": self.protocol_version,
            "architecture_version": self.architecture_version,
            "capability_profile": self.capability_profile,
            "supported_features": list(self.supported_features),
            "created_epoch": self.created_epoch,
            "fingerprint": self.fingerprint or self.compute_fingerprint(),
        }


@dataclass(frozen=True)
class PeerAttestation:
    """
    Structured deterministic attestation claim submitted by a peer.
    Does NOT constitute production cryptographic proof.
    """
    attestation_id: str
    peer_id: str
    zone_id: str
    architecture_version: str
    protocol_version: str
    capability_manifest: Dict[str, Any]
    policy_version: str
    runtime_integrity_hash: str
    declared_epoch: int
    status: AttestationStatus = AttestationStatus.DECLARED
    verification_notes: str = ""

    def canonical_serialize(self) -> bytes:
        payload = {
            "attestation_id": self.attestation_id,
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "architecture_version": self.architecture_version,
            "protocol_version": self.protocol_version,
            "capability_manifest": self.capability_manifest,
            "policy_version": self.policy_version,
            "runtime_integrity_hash": self.runtime_integrity_hash,
            "declared_epoch": self.declared_epoch,
            "status": self.status.value,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attestation_id": self.attestation_id,
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "architecture_version": self.architecture_version,
            "protocol_version": self.protocol_version,
            "capability_manifest": self.capability_manifest,
            "policy_version": self.policy_version,
            "runtime_integrity_hash": self.runtime_integrity_hash,
            "declared_epoch": self.declared_epoch,
            "status": self.status.value,
            "verification_notes": self.verification_notes,
            "digest": self.compute_digest(),
        }


@dataclass(frozen=True)
class PeerDeclaration:
    """Initial declaration bundle presented during peer discovery and negotiation."""
    declaration_id: str
    identity: PeerIdentity
    attestation: PeerAttestation
    resource_profile: Dict[str, Any]
    requested_scopes: List[FederationScope]
    policy_profile: Dict[str, Any]
    epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declaration_id": self.declaration_id,
            "identity": self.identity.to_dict(),
            "attestation": self.attestation.to_dict(),
            "resource_profile": self.resource_profile,
            "requested_scopes": [s.value for s in self.requested_scopes],
            "policy_profile": self.policy_profile,
            "epoch": self.epoch,
        }


@dataclass
class TrustGrant:
    """
    Explicit, bounded, and expiring trust grant issued to a peer.
    Trust is NEVER permanent and NEVER a boolean.
    """
    grant_id: str
    issuer_zone_id: str
    subject_peer_id: str
    subject_zone_id: str
    trust_level: TrustLevel
    permitted_scopes: List[FederationScope]
    issued_epoch: int
    expires_epoch: int
    policy_constraints: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    status: TrustStatus = TrustStatus.ACTIVE
    provenance: str = "local_policy_engine"

    def is_valid_at(self, current_epoch: int) -> bool:
        """Verify if the grant is active and not expired."""
        if self.status != TrustStatus.ACTIVE:
            return False
        if current_epoch > self.expires_epoch:
            return False
        if current_epoch < self.issued_epoch:
            return False
        return True

    def is_expired(self, current_epoch: int) -> bool:
        """Check if trust grant has expired."""
        return not self.is_valid_at(current_epoch)

    def allows_scope(self, scope: FederationScope, current_epoch: int) -> bool:
        """Verify if the requested scope is permitted under this grant at epoch."""
        if not self.is_valid_at(current_epoch):
            return False
        if self.trust_level in (TrustLevel.NONE, TrustLevel.REVOKED):
            return False
        return scope in self.permitted_scopes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "issuer_zone_id": self.issuer_zone_id,
            "subject_peer_id": self.subject_peer_id,
            "subject_zone_id": self.subject_zone_id,
            "trust_level": self.trust_level.value,
            "permitted_scopes": [s.value for s in self.permitted_scopes],
            "issued_epoch": self.issued_epoch,
            "expires_epoch": self.expires_epoch,
            "policy_constraints": self.policy_constraints,
            "reason": self.reason,
            "status": self.status.value,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class NegotiationAgreement:
    """Deterministic result of a trust negotiation transaction."""
    agreement_id: str
    local_zone_id: str
    peer_id: str
    peer_zone_id: str
    trust_grant: Optional[TrustGrant]
    accepted_scopes: List[FederationScope]
    rejected_scopes: List[FederationScope]
    rejection_reasons: Dict[str, str]
    epoch: int

    def is_successful(self) -> bool:
        return self.trust_grant is not None and self.trust_grant.status == TrustStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreement_id": self.agreement_id,
            "local_zone_id": self.local_zone_id,
            "peer_id": self.peer_id,
            "peer_zone_id": self.peer_zone_id,
            "trust_grant": self.trust_grant.to_dict() if self.trust_grant else None,
            "accepted_scopes": [s.value for s in self.accepted_scopes],
            "rejected_scopes": [s.value for s in self.rejected_scopes],
            "rejection_reasons": self.rejection_reasons,
            "epoch": self.epoch,
            "is_successful": self.is_successful(),
        }


@dataclass(frozen=True)
class RevocationRecord:
    """Auditable record of peer or grant revocation."""
    record_id: str
    peer_id: str
    zone_id: str
    revoked_by: str
    reason: str
    revoked_epoch: int
    affected_grant_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "revoked_by": self.revoked_by,
            "reason": self.reason,
            "revoked_epoch": self.revoked_epoch,
            "affected_grant_ids": list(self.affected_grant_ids),
        }


@dataclass
class PeerRegistration:
    """Complete registry entry for a discovered or peered entity."""
    identity: PeerIdentity
    discovery_status: DiscoveryStatus
    trust_grant: Optional[TrustGrant] = None
    latest_attestation: Optional[PeerAttestation] = None
    registered_epoch: int = 1
    last_seen_epoch: int = 1
    revocation_record: Optional[RevocationRecord] = None
    cryptographic_identity: Optional[Any] = None

    def is_active(self, current_epoch: int) -> bool:
        if self.discovery_status in (DiscoveryStatus.REJECTED, DiscoveryStatus.REVOKED, DiscoveryStatus.EXPIRED):
            return False
        if not self.trust_grant:
            return False
        return self.trust_grant.is_valid_at(current_epoch)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "discovery_status": self.discovery_status.value,
            "trust_grant": self.trust_grant.to_dict() if self.trust_grant else None,
            "latest_attestation": self.latest_attestation.to_dict() if self.latest_attestation else None,
            "registered_epoch": self.registered_epoch,
            "last_seen_epoch": self.last_seen_epoch,
            "revocation_record": self.revocation_record.to_dict() if self.revocation_record else None,
            "has_cryptographic_identity": self.cryptographic_identity is not None,
        }


@dataclass(frozen=True)
class AuditRecord:
    """Bounded, sanitized audit log entry."""
    event_id: str
    event_type: AuditEventType
    epoch: int
    peer_id: Optional[str] = None
    zone_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "epoch": self.epoch,
            "peer_id": self.peer_id,
            "zone_id": self.zone_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SafePublicPeeringTrace:
    """Sanitized public telemetry trace of a peering operation."""
    operation_id: str
    local_zone_id: str
    peer_id: str
    peer_zone_id: str
    event_type: str
    status: str
    epoch: int
    granted_scopes: List[str] = field(default_factory=list)
    denied_scopes: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "local_zone_id": self.local_zone_id,
            "peer_id": self.peer_id,
            "peer_zone_id": self.peer_zone_id,
            "event_type": self.event_type,
            "status": self.status,
            "epoch": self.epoch,
            "granted_scopes": self.granted_scopes,
            "denied_scopes": self.denied_scopes,
            "notes": self.notes,
        }
