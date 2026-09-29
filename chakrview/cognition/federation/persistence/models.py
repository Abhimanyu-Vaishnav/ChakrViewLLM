"""
Durable Security State & Journal Models for ChakrView (Step 34).

Strict Invariants:
- NEVER persist private keys, session secrets, or neural model weights.
- All serialization uses deterministic, canonical JSON (sort_keys=True, compact separators).
- Cryptographic hash chaining ensures tamper-evident journal logs.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.federation.persistence.errors import (
    DurableSchemaError,
    JournalCorruptionError,
    SnapshotCorruptionError,
)

DURABLE_SCHEMA_VERSION = "34.0"
JOURNAL_GENESIS_DIGEST = "0" * 64
GENESIS_JOURNAL_DIGEST = JOURNAL_GENESIS_DIGEST
MAX_JOURNAL_PAYLOAD_BYTES = 65536     # 64 KB ceiling per entry
MAX_SNAPSHOT_BYTES = 1048576          # 1 MB ceiling per snapshot


class JournalEntryType(str, Enum):
    """Taxonomy of security mutations requiring durable write-ahead journaling."""
    SNAPSHOT_RECORD = "SNAPSHOT_RECORD"
    PEER_REGISTERED = "PEER_REGISTERED"
    PEER_REVOKED = "PEER_REVOKED"
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_ACTIVATED = "SESSION_ACTIVATED"
    SESSION_TERMINATED = "SESSION_TERMINATED"
    SESSION_REVOKED = "SESSION_REVOKED"
    KEY_ROTATED = "KEY_ROTATED"
    CERTIFICATE_REVOKED = "CERTIFICATE_REVOKED"
    TRUST_GRANTED = "TRUST_GRANTED"
    TRUST_REVOKED = "TRUST_REVOKED"
    REPLAY_FLOOR_ADVANCED = "REPLAY_FLOOR_ADVANCED"
    EPOCH_ADVANCED = "EPOCH_ADVANCED"

    # Step 35 Node Membership Journal Entries
    NODE_DISCOVERED = "NODE_DISCOVERED"
    NODE_AUTHENTICATED = "NODE_AUTHENTICATED"
    NODE_MEMBERSHIP_GRANTED = "NODE_MEMBERSHIP_GRANTED"
    NODE_MEMBERSHIP_SUSPENDED = "NODE_MEMBERSHIP_SUSPENDED"
    NODE_QUARANTINED = "NODE_QUARANTINED"
    NODE_REVOKED = "NODE_REVOKED"
    NODE_TERMINATED = "NODE_TERMINATED"
    NODE_REMOVED = "NODE_REMOVED"

    # Step 36 Federation Transport Journal Entries
    CONNECTION_ESTABLISHED = "CONNECTION_ESTABLISHED"
    CONNECTION_CLOSED = "CONNECTION_CLOSED"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    MESSAGE_DISPATCHED = "MESSAGE_DISPATCHED"
    MESSAGE_REJECTED = "MESSAGE_REJECTED"
    REPLAY_REJECTED = "REPLAY_REJECTED"
    CHANNEL_REVOKED = "CHANNEL_REVOKED"
    CHANNEL_QUARANTINED = "CHANNEL_QUARANTINED"

    # Step 37 Distributed Resource & Capability Journal Entries
    CAPABILITY_ADVERTISED = "CAPABILITY_ADVERTISED"
    RESOURCE_POLICY_UPDATED = "RESOURCE_POLICY_UPDATED"

    # Step 38 Distributed Task Orchestration Journal Entries
    TASK_CREATED = "TASK_CREATED"
    TASK_PLANNED = "TASK_PLANNED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    TASK_ACCEPTED = "TASK_ACCEPTED"
    TASK_STARTED = "TASK_STARTED"
    TASK_CHECKPOINTED = "TASK_CHECKPOINTED"
    TASK_PROGRESS = "TASK_PROGRESS"
    TASK_REASSIGNED = "TASK_REASSIGNED"
    TASK_RESULT_RECEIVED = "TASK_RESULT_RECEIVED"
    TASK_RESULT_ACCEPTED = "TASK_RESULT_ACCEPTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_RECOVERED = "TASK_RECOVERED"



@dataclass(frozen=True)
class JournalEntry:
    """
    Append-only cryptographically chained journal record.
    Chain formula: digest_N = SHA-256(canonical_payload_N || prev_digest_{N-1})
    """
    sequence_num: int
    epoch: int
    entry_type: JournalEntryType
    event_id: str
    payload: Dict[str, Any]
    prev_digest: str
    digest: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Guard against private key / secret leaks
        payload_str = json.dumps(self.payload).lower()
        if "private_key" in payload_str or "private_bytes" in payload_str or "session_key" in payload_str:
            raise PermissionError(
                "Security violation: Plaintext private keys or session secrets "
                "must NEVER be persisted in the security journal."
            )

    @classmethod
    def create(
        cls,
        sequence_num: int,
        epoch: int,
        entry_type: JournalEntryType,
        payload: Dict[str, Any],
        prev_digest: str,
        timestamp: Optional[float] = None,
    ) -> "JournalEntry":
        """Factory creating entry with deterministic SHA-256 digest computation."""
        event_id = f"jrn_{epoch}_{sequence_num}_{entry_type.value.lower()}"
        ts = timestamp if timestamp is not None else time.time()
        canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        
        hasher = hashlib.sha256()
        hasher.update(canonical_payload.encode("utf-8"))
        hasher.update(prev_digest.encode("utf-8"))
        hasher.update(f":{sequence_num}:{epoch}:{entry_type.value}:{event_id}".encode("utf-8"))
        computed_digest = hasher.hexdigest()

        return cls(
            sequence_num=sequence_num,
            epoch=epoch,
            entry_type=entry_type,
            event_id=event_id,
            payload=dict(payload),
            prev_digest=prev_digest,
            digest=computed_digest,
            timestamp=ts,
        )

    def verify_integrity(self, expected_prev_digest: Optional[str] = None) -> bool:
        """
        Verify that this entry's digest matches its payload and previous digest.
        """
        if expected_prev_digest is not None and self.prev_digest != expected_prev_digest:
            return False

        canonical_payload = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        hasher = hashlib.sha256()
        hasher.update(canonical_payload.encode("utf-8"))
        hasher.update(self.prev_digest.encode("utf-8"))
        hasher.update(f":{self.sequence_num}:{self.epoch}:{self.entry_type.value}:{self.event_id}".encode("utf-8"))
        return hasher.hexdigest() == self.digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_num": self.sequence_num,
            "epoch": self.epoch,
            "entry_type": self.entry_type.value,
            "event_id": self.event_id,
            "payload": self.payload,
            "prev_digest": self.prev_digest,
            "digest": self.digest,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalEntry":
        try:
            return cls(
                sequence_num=int(data["sequence_num"]),
                epoch=int(data["epoch"]),
                entry_type=JournalEntryType(data["entry_type"]),
                event_id=str(data["event_id"]),
                payload=dict(data["payload"]),
                prev_digest=str(data["prev_digest"]),
                digest=str(data["digest"]),
                timestamp=float(data.get("timestamp", time.time())),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise DurableSchemaError(f"Malformed journal entry data: {e}")


@dataclass
class DurableSecuritySnapshot:
    """
    Point-in-time normalized snapshot of federation security state.
    Allows bounded recovery time by replaying only journal entries after journal_offset.
    """
    snapshot_id: str
    snapshot_version: int
    journal_offset: int
    engine_identity: Dict[str, Any]
    state_version: Dict[str, Any]
    epoch: int
    peers: List[Dict[str, Any]]
    sessions: List[Dict[str, Any]]
    revocations: List[Dict[str, Any]]
    trust_grants: List[Dict[str, Any]]
    certificate_revocations: List[str]
    replay_floors: Dict[str, int]
    composite_digest: str
    memberships: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = DURABLE_SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    integrity_hash: Optional[str] = None

    def compute_integrity_hash(self) -> str:
        """Compute SHA-256 over canonical JSON of all snapshot payload fields."""
        payload = {
            "snapshot_id": self.snapshot_id,
            "snapshot_version": self.snapshot_version,
            "journal_offset": self.journal_offset,
            "engine_identity": self.engine_identity,
            "state_version": self.state_version,
            "epoch": self.epoch,
            "peers": self.peers,
            "sessions": self.sessions,
            "revocations": self.revocations,
            "trust_grants": self.trust_grants,
            "certificate_revocations": sorted(self.certificate_revocations),
            "replay_floors": dict(sorted(self.replay_floors.items())),
            "composite_digest": self.composite_digest,
            "memberships": self.memberships,
            "schema_version": self.schema_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> "DurableSecuritySnapshot":
        """Compute and set integrity hash."""
        self.integrity_hash = self.compute_integrity_hash()
        return self

    def verify_integrity(self) -> bool:
        """Verify snapshot payload has not suffered bit rot, corruption, or tampering."""
        if not self.integrity_hash:
            return False
        return self.compute_integrity_hash() == self.integrity_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_version": self.snapshot_version,
            "journal_offset": self.journal_offset,
            "engine_identity": self.engine_identity,
            "state_version": self.state_version,
            "epoch": self.epoch,
            "peers": self.peers,
            "sessions": self.sessions,
            "revocations": self.revocations,
            "trust_grants": self.trust_grants,
            "certificate_revocations": self.certificate_revocations,
            "replay_floors": self.replay_floors,
            "composite_digest": self.composite_digest,
            "memberships": self.memberships,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "integrity_hash": self.integrity_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DurableSecuritySnapshot":
        schema = data.get("schema_version")
        if schema != DURABLE_SCHEMA_VERSION:
            raise DurableSchemaError(
                f"Incompatible snapshot schema version: expected '{DURABLE_SCHEMA_VERSION}', got '{schema}'"
            )

        snap = cls(
            snapshot_id=str(data["snapshot_id"]),
            snapshot_version=int(data["snapshot_version"]),
            journal_offset=int(data["journal_offset"]),
            engine_identity=dict(data["engine_identity"]),
            state_version=dict(data["state_version"]),
            epoch=int(data["epoch"]),
            peers=list(data.get("peers", [])),
            sessions=list(data.get("sessions", [])),
            revocations=list(data.get("revocations", [])),
            trust_grants=list(data.get("trust_grants", [])),
            certificate_revocations=list(data.get("certificate_revocations", [])),
            replay_floors=dict(data.get("replay_floors", {})),
            composite_digest=str(data["composite_digest"]),
            memberships=list(data.get("memberships", [])),
            schema_version=str(schema),
            created_at=float(data.get("created_at", time.time())),
            integrity_hash=data.get("integrity_hash"),
        )
        if not snap.verify_integrity():
            raise SnapshotCorruptionError(
                f"Snapshot '{snap.snapshot_id}' failed SHA-256 integrity verification."
            )
        return snap


@dataclass(frozen=True)
class RecoveryManifest:
    """Audit document recording the outcome of a recovery procedure."""
    engine_id: str
    snapshot_loaded: Optional[str]
    journal_entries_replayed: int
    final_version: int
    final_epoch: int
    final_digest: str
    status: str
    failure_reason: Optional[str] = None
    recovered_at: float = field(default_factory=time.time)

    def is_successful(self) -> bool:
        return self.status == "SUCCESS"
