"""
Data Models for Federation Networking, Node Discovery & Secure Membership (Step 35).

Defines strongly typed, bounded representations for node endpoints, discovery candidates,
and membership lifecycle states with strict transition enforcement.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
import time
from typing import Dict, Any, Optional, Set, List, Tuple

from chakrview.cognition.transport.security.models import TLSMode
from chakrview.cognition.federation.models import FederationEngineIdentity
from chakrview.cognition.federation.discovery.errors import (
    EndpointValidationError,
    MembershipStateTransitionError,
    MalformedEndpointError,
    UnsupportedProtocolError,
)

MAX_MEMBERSHIP_NODES = 16
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_MISSED_HEARTBEATS = 3


class NodeProtocol(str, Enum):
    """Transport protocol used for node communication."""
    TCP = "TCP"
    TLS = "TLS"
    MTLS = "MTLS"


class NodeDiscoverySource(str, Enum):
    """Controlled source mechanism that discovered a node candidate."""
    STATIC_CONFIG = "STATIC_CONFIG"
    CONFIG_FILE = "CONFIG_FILE"
    IN_PROCESS_ADVERTISEMENT = "IN_PROCESS_ADVERTISEMENT"


class MembershipState(str, Enum):
    """
    Explicit, bounded lifecycle states for federation node membership.
    
    States:
    - DISCOVERED: Node discovered as a candidate; zero trust, unauthenticated.
    - PENDING_AUTHENTICATION: Transport established, cryptographic authentication in progress.
    - AUTHENTICATED: Identity, certificate, and handshake validated; pending membership grant.
    - MEMBER: Fully enrolled federation member permitted bounded communication.
    - SUSPENDED: Temporarily inactive or unreachable; trust not revoked.
    - QUARANTINED: Security or anomaly violation detected; traffic strictly blocked.
    - REVOKED: Terminal revocation; node is permanently barred.
    - TERMINATED: Administratively shut down or decommissioned.
    """
    DISCOVERED = "DISCOVERED"
    PENDING_AUTHENTICATION = "PENDING_AUTHENTICATION"
    AUTHENTICATED = "AUTHENTICATED"
    MEMBER = "MEMBER"
    SUSPENDED = "SUSPENDED"
    QUARANTINED = "QUARANTINED"
    REVOKED = "REVOKED"
    TERMINATED = "TERMINATED"


# Strict state transition matrix
VALID_MEMBERSHIP_TRANSITIONS: Dict[MembershipState, Set[MembershipState]] = {
    MembershipState.DISCOVERED: {
        MembershipState.PENDING_AUTHENTICATION,
        MembershipState.QUARANTINED,
        MembershipState.REVOKED,
        MembershipState.TERMINATED,
    },
    MembershipState.PENDING_AUTHENTICATION: {
        MembershipState.AUTHENTICATED,
        MembershipState.QUARANTINED,
        MembershipState.REVOKED,
        MembershipState.TERMINATED,
    },
    MembershipState.AUTHENTICATED: {
        MembershipState.MEMBER,
        MembershipState.SUSPENDED,
        MembershipState.QUARANTINED,
        MembershipState.REVOKED,
        MembershipState.TERMINATED,
    },
    MembershipState.MEMBER: {
        MembershipState.SUSPENDED,
        MembershipState.QUARANTINED,
        MembershipState.REVOKED,
        MembershipState.TERMINATED,
    },
    MembershipState.SUSPENDED: {
        MembershipState.PENDING_AUTHENTICATION,
        MembershipState.MEMBER,
        MembershipState.QUARANTINED,
        MembershipState.REVOKED,
        MembershipState.TERMINATED,
    },
    MembershipState.QUARANTINED: {
        MembershipState.PENDING_AUTHENTICATION,  # Explicit re-verification if policy permits
        MembershipState.REVOKED,
        MembershipState.TERMINATED,
    },
    MembershipState.REVOKED: set(),      # Absorbing terminal state; cannot transition
    MembershipState.TERMINATED: set(),   # Absorbing terminal state
}


@dataclass(frozen=True)
class NodeAddress:
    """Network address consisting of a validated hostname/IP and port."""
    host: str
    port: int

    HOST_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,255}$")

    def __post_init__(self) -> None:
        if not self.host or not self.HOST_PATTERN.match(self.host):
            raise EndpointValidationError(f"Invalid host address '{self.host}'.")
        if not (1 <= self.port <= 65535):
            raise EndpointValidationError(f"Port must be in range 1-65535, got {self.port}.")

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class FederationNodeEndpoint:
    """
    Explicit, non-secret network endpoint descriptor for a federation node.
    """
    endpoint_id: str
    address: NodeAddress
    protocol: NodeProtocol
    zone_id: str
    tls_mode: TLSMode = TLSMode.MTLS
    expected_cert_fingerprint: Optional[str] = None
    expected_san: Optional[str] = None
    expected_engine_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        host: str,
        port: int,
        protocol: NodeProtocol = NodeProtocol.MTLS,
        zone_id: str = "zone-default",
        tls_mode: TLSMode = TLSMode.MTLS,
        expected_cert_fingerprint: Optional[str] = None,
        expected_san: Optional[str] = None,
        expected_engine_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FederationNodeEndpoint":
        """
        Derive an immutable endpoint with deterministic SHA-256 endpoint_id.
        """
        addr = NodeAddress(host=host, port=port)
        canonical_key = f"{protocol.value}:{addr}:{zone_id}:{tls_mode.value}:{expected_cert_fingerprint or ''}"
        endpoint_id = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:32]

        return cls(
            endpoint_id=f"ep_{endpoint_id}",
            address=addr,
            protocol=protocol,
            zone_id=zone_id,
            tls_mode=tls_mode,
            expected_cert_fingerprint=expected_cert_fingerprint,
            expected_san=expected_san,
            expected_engine_id=expected_engine_id,
            metadata=metadata or {},
        )

    def to_transport_uri(self) -> str:
        scheme = "mtls" if self.protocol == NodeProtocol.MTLS else ("tls" if self.protocol == NodeProtocol.TLS else "tcp")
        return f"{scheme}://{self.address.host}:{self.address.port}"

    @classmethod
    def from_transport_uri(cls, uri: str, zone_id: str = "zone-default") -> "FederationNodeEndpoint":
        """Parse canonical transport URI into an explicit node endpoint."""
        if "://" not in uri:
            raise MalformedEndpointError(f"Malformed URI, missing scheme delimiter '://': {uri}")
        scheme, rest = uri.split("://", 1)
        scheme_lower = scheme.lower()
        if scheme_lower == "mtls":
            proto = NodeProtocol.MTLS
        elif scheme_lower == "tls":
            proto = NodeProtocol.TLS
        elif scheme_lower == "tcp":
            proto = NodeProtocol.TCP
        else:
            raise UnsupportedProtocolError(f"Protocol scheme '{scheme}' is unsupported.")
        if ":" not in rest:
            raise MalformedEndpointError(f"Malformed URI, missing port: {uri}")
        host, port_str = rest.split(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            raise MalformedEndpointError(f"Invalid port in URI: {port_str}")
        return cls.create(host=host, port=port, protocol=proto, zone_id=zone_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "host": self.address.host,
            "port": self.address.port,
            "protocol": self.protocol.value,
            "zone_id": self.zone_id,
            "tls_mode": self.tls_mode.value,
            "expected_cert_fingerprint": self.expected_cert_fingerprint,
            "expected_san": self.expected_san,
            "expected_engine_id": self.expected_engine_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederationNodeEndpoint":
        return cls(
            endpoint_id=data["endpoint_id"],
            address=NodeAddress(host=data["host"], port=int(data["port"])),
            protocol=NodeProtocol(data["protocol"]),
            zone_id=data["zone_id"],
            tls_mode=TLSMode(data.get("tls_mode", TLSMode.MTLS.value)),
            expected_cert_fingerprint=data.get("expected_cert_fingerprint"),
            expected_san=data.get("expected_san"),
            expected_engine_id=data.get("expected_engine_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class FederationNodeCandidate:
    """
    Unauthenticated discovered node awaiting verification and authentication.
    CRITICAL: A candidate holds ZERO trust and ZERO authority.
    """
    candidate_id: str
    endpoint: FederationNodeEndpoint
    discovery_source: NodeDiscoverySource
    discovered_epoch: int
    discovered_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        endpoint: FederationNodeEndpoint,
        discovery_source: NodeDiscoverySource,
        discovered_epoch: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FederationNodeCandidate":
        canonical_str = f"{endpoint.endpoint_id}:{discovery_source.value}:{discovered_epoch}"
        cand_id = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return cls(
            candidate_id=cand_id,
            endpoint=endpoint,
            discovery_source=discovery_source,
            discovered_epoch=discovered_epoch,
            metadata=metadata or {},
        )

    @classmethod
    def from_endpoint(
        cls,
        endpoint: FederationNodeEndpoint,
        discovery_source: NodeDiscoverySource = NodeDiscoverySource.STATIC_CONFIG,
        discovered_epoch: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FederationNodeCandidate":
        return cls.create(
            endpoint=endpoint,
            discovery_source=discovery_source,
            discovered_epoch=discovered_epoch,
            metadata=metadata,
        )

    @property
    def state(self) -> MembershipState:
        return MembershipState.DISCOVERED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "endpoint": self.endpoint.to_dict(),
            "discovery_source": self.discovery_source.value,
            "discovered_epoch": self.discovered_epoch,
            "discovered_at": self.discovered_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederationNodeCandidate":
        return cls(
            candidate_id=data["candidate_id"],
            endpoint=FederationNodeEndpoint.from_dict(data["endpoint"]),
            discovery_source=NodeDiscoverySource(data["discovery_source"]),
            discovered_epoch=int(data["discovered_epoch"]),
            discovered_at=float(data.get("discovered_at", time.time())),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FederationNodeMembership:
    """
    Active or managed membership record for a node within the federation runtime.
    """
    membership_id: str
    candidate_id: str
    endpoint: FederationNodeEndpoint
    zone_id: str
    state: MembershipState
    node_id: Optional[str] = None
    engine_identity: Optional[FederationEngineIdentity] = None
    peer_id: Optional[str] = None
    enrolled_epoch: int = 1
    last_heartbeat_epoch: int = 1
    last_heartbeat_timestamp: float = field(default_factory=time.time)
    missed_heartbeats: int = 0
    quarantine_reason: Optional[str] = None
    quarantine_evidence: Dict[str, Any] = field(default_factory=dict)
    revocation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        candidate: FederationNodeCandidate,
        enrolled_epoch: int = 1,
        node_id: Optional[str] = None,
    ) -> "FederationNodeMembership":
        mem_id = f"mem_{candidate.endpoint.endpoint_id[3:]}"
        return cls(
            membership_id=mem_id,
            candidate_id=candidate.candidate_id,
            endpoint=candidate.endpoint,
            zone_id=candidate.endpoint.zone_id,
            state=MembershipState.DISCOVERED,
            node_id=node_id,
            enrolled_epoch=enrolled_epoch,
            last_heartbeat_epoch=enrolled_epoch,
            last_heartbeat_timestamp=time.time(),
        )

    def transition_to(self, target_state: MembershipState, reason: Optional[str] = None) -> None:
        """
        Enforce valid state machine transitions.
        Terminal states (REVOKED, TERMINATED) can never transition back to active.
        """
        if self.state == target_state:
            return

        valid_targets = VALID_MEMBERSHIP_TRANSITIONS.get(self.state, set())
        if target_state not in valid_targets:
            raise MembershipStateTransitionError(
                f"Illegal membership transition from '{self.state.value}' to '{target_state.value}' "
                f"(membership_id='{self.membership_id}', reason='{reason}')."
            )

        self.state = target_state
        if target_state == MembershipState.QUARANTINED:
            self.quarantine_reason = reason or "Security anomaly detected"
        elif target_state == MembershipState.REVOKED:
            self.revocation_reason = reason or "Administrative revocation"

    def is_active_member(self) -> bool:
        return self.state == MembershipState.MEMBER

    def is_quarantined(self) -> bool:
        return self.state == MembershipState.QUARANTINED

    def is_revoked(self) -> bool:
        return self.state == MembershipState.REVOKED

    def record_heartbeat(self, epoch: int, timestamp: Optional[float] = None) -> None:
        self.last_heartbeat_epoch = epoch
        self.last_heartbeat_timestamp = timestamp or time.time()
        self.missed_heartbeats = 0

    def to_dict(self) -> Dict[str, Any]:
        engine_dict = None
        if self.engine_identity:
            engine_dict = {
                "engine_id": self.engine_identity.engine_id,
                "zone_id": self.engine_identity.zone_id,
                "created_epoch": self.engine_identity.created_epoch,
                "identity_fingerprint": self.engine_identity.identity_fingerprint,
                "protocol_version": self.engine_identity.protocol_version,
                "architecture_version": self.engine_identity.architecture_version,
                "public_key_fingerprint": self.engine_identity.public_key_fingerprint,
            }
        return {
            "membership_id": self.membership_id,
            "candidate_id": self.candidate_id,
            "endpoint": self.endpoint.to_dict(),
            "zone_id": self.zone_id,
            "state": self.state.value,
            "node_id": self.node_id,
            "engine_identity": engine_dict,
            "peer_id": self.peer_id,
            "enrolled_epoch": self.enrolled_epoch,
            "last_heartbeat_epoch": self.last_heartbeat_epoch,
            "last_heartbeat_timestamp": self.last_heartbeat_timestamp,
            "missed_heartbeats": self.missed_heartbeats,
            "quarantine_reason": self.quarantine_reason,
            "quarantine_evidence": self.quarantine_evidence,
            "revocation_reason": self.revocation_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederationNodeMembership":
        engine_id_obj = None
        if data.get("engine_identity"):
            ed = data["engine_identity"]
            engine_id_obj = FederationEngineIdentity(
                engine_id=ed["engine_id"],
                zone_id=ed["zone_id"],
                created_epoch=int(ed["created_epoch"]),
                identity_fingerprint=ed["identity_fingerprint"],
                protocol_version=ed.get("protocol_version", "33.0"),
                architecture_version=ed.get("architecture_version", "0.1"),
                public_key_fingerprint=ed.get("public_key_fingerprint"),
            )
        return cls(
            membership_id=data["membership_id"],
            candidate_id=data["candidate_id"],
            endpoint=FederationNodeEndpoint.from_dict(data["endpoint"]),
            zone_id=data["zone_id"],
            state=MembershipState(data["state"]),
            node_id=data.get("node_id"),
            engine_identity=engine_id_obj,
            peer_id=data.get("peer_id"),
            enrolled_epoch=int(data.get("enrolled_epoch", 1)),
            last_heartbeat_epoch=int(data.get("last_heartbeat_epoch", 1)),
            last_heartbeat_timestamp=float(data.get("last_heartbeat_timestamp", time.time())),
            missed_heartbeats=int(data.get("missed_heartbeats", 0)),
            quarantine_reason=data.get("quarantine_reason"),
            quarantine_evidence=data.get("quarantine_evidence", {}),
            revocation_reason=data.get("revocation_reason"),
            metadata=data.get("metadata", {}),
        )
