"""
Strongly Typed Data Models for Distributed Federated Cognition (Step 27).

Defines foundational primitives for secure agent transport and distributed federation:
- Node Identity, Roles, Status, Capabilities, Endpoints & Health
- Distributed Message Envelope, Headers, Routing & Integrity
- Deterministic Serialization, SHA-256 Hashing & Nonces
- Distributed Routing Decisions & Sanitized Public Telemetry

CRITICAL ARCHITECTURAL AXIOMS:
1. NODE != AUTHORITY & AGENT != AUTHORITY
   Nodes and remote agents are compute hosts and logical roles; they never
   hold authority to execute capabilities or mutate weights.
2. STRICT TENANT & SESSION ISOLATION:
   All node registries, message routes, and task dispatches strictly enforce
   tenant and session boundary verification.
3. FAIL-CLOSED IMMUTABILITY:
   Distributed coordination operates around the single frozen ChakrMicro v0.1 core.
   Runtime weight mutation is strictly impossible.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set
import uuid

from chakrview.cognition.federated.models import (
    AgentRole,
    AgentCapability,
    MessageType,
    MessagePriority,
    ConflictState,
    FederatedConflictRecord,
    FederatedSynthesisCandidate,
)
from chakrview.cognition.unified.models import DecisionState

# Hard Architectural Ceilings (Centrally Enforced)
MAX_FEDERATION_NODES = 16
MAX_AGENTS_PER_NODE = 8
MAX_FEDERATION_DEPTH = 4
MAX_REMOTE_TASKS_PER_CYCLE = 32
MAX_MESSAGE_HOPS = 4
DEFAULT_MESSAGE_TTL_SECONDS = 30.0


# ============================================================================
# 1. Distributed Node Enums & Primitives
# ============================================================================

class NodeRole(str, Enum):
    """Architectural role of a federated node in the cluster."""
    PRIMARY = "PRIMARY"      # Coordinates the local federation and task decomposition
    WORKER = "WORKER"        # Hosts cognitive agent roles for subtask execution
    ARBITER = "ARBITER"      # Participates in conflict arbitration and consensus verification
    OBSERVER = "OBSERVER"    # Read-only audit telemetry and observability sink
    RELAY = "RELAY"          # Message forwarding and boundary routing


class NodeStatus(str, Enum):
    """Operational lifecycle state of a distributed node."""
    REGISTERED = "REGISTERED"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    QUARANTINED = "QUARANTINED"
    REVOKED = "REVOKED"


class NodeTrustState(str, Enum):
    """Cryptographic and administrative trust standing of a node."""
    TRUSTED = "TRUSTED"
    PROBATION = "PROBATION"
    UNTRUSTED = "UNTRUSTED"
    REVOKED = "REVOKED"


class TransportStatus(str, Enum):
    """Delivery status of a transport transaction."""
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    REJECTED = "REJECTED"


# ============================================================================
# 2. Node Identity, Capabilities & Health
# ============================================================================

@dataclass(frozen=True)
class NodeIdentity:
    """
    Deterministic identity for a distributed ChakrView computing node.
    Independent from logical agent identities.
    """
    node_id: str
    tenant_id: str
    cluster_id: str = "chakrview_cluster_local"
    role: NodeRole = NodeRole.WORKER
    version: str = "27.0.0"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tenant_id": self.tenant_id,
            "cluster_id": self.cluster_id,
            "role": self.role.value,
            "version": self.version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class NodeEndpoint:
    """Network or IPC endpoint abstraction for node transport."""
    endpoint_id: str
    uri: str
    transport_type: str = "loopback"
    protocol_version: str = "27.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "uri": self.uri,
            "transport_type": self.transport_type,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True)
class NodeCapabilities:
    """Logical roles and resource capacities hosted by a node."""
    supported_roles: List[AgentRole] = field(default_factory=list)
    max_concurrency: int = 4
    memory_budget_mb: float = 256.0
    token_ceiling: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported_roles": [r.value for r in self.supported_roles],
            "max_concurrency": self.max_concurrency,
            "memory_budget_mb": self.memory_budget_mb,
            "token_ceiling": self.token_ceiling,
        }


@dataclass(frozen=True)
class NodeResourceProfile:
    """Hardware and compute capacity profile for resource-aware routing."""
    cpu_cores: int = 2
    memory_mb: float = 512.0
    latency_tier: str = "LOCAL"            # "LOCAL", "LAN", "WAN"
    compute_profile: str = "STANDARD"      # "LOW_RESOURCE", "STANDARD", "HIGH_RESOURCE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "latency_tier": self.latency_tier,
            "compute_profile": self.compute_profile,
        }


@dataclass
class NodeHealth:
    """Mutable health and availability telemetry of a registered node."""
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class NodeRegistration:
    """Complete registration record for a node in the DistributedNodeRegistry."""
    identity: NodeIdentity
    endpoint: NodeEndpoint
    capabilities: NodeCapabilities
    resource_profile: NodeResourceProfile
    health: NodeHealth = field(default_factory=NodeHealth)
    trust_state: NodeTrustState = NodeTrustState.TRUSTED
    registered_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def is_eligible_for_tasks(self) -> bool:
        """Determines if the node can receive tasks under current health/trust state."""
        return (
            self.health.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)
            and self.trust_state in (NodeTrustState.TRUSTED, NodeTrustState.PROBATION)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "endpoint": self.endpoint.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "resource_profile": self.resource_profile.to_dict(),
            "health": self.health.to_dict(),
            "trust_state": self.trust_state.value,
            "registered_at": self.registered_at,
            "expires_at": self.expires_at,
        }


# ============================================================================
# 3. Secure Distributed Message Envelope
# ============================================================================

@dataclass(frozen=True)
class MessageHeader:
    """Header metadata for routing, tracing, and replay protection."""
    message_id: str
    correlation_id: str
    causation_id: Optional[str] = None
    message_type: MessageType = MessageType.TASK_REQUEST
    protocol_version: str = "27.0"
    timestamp: float = field(default_factory=time.time)
    logical_clock: int = 0
    ttl_seconds: float = DEFAULT_MESSAGE_TTL_SECONDS
    hop_count: int = 0
    max_hops: int = MAX_MESSAGE_HOPS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "message_type": self.message_type.value,
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
            "logical_clock": self.logical_clock,
            "ttl_seconds": self.ttl_seconds,
            "hop_count": self.hop_count,
            "max_hops": self.max_hops,
        }


@dataclass(frozen=True)
class MessageRoute:
    """Network-level routing addressing with strict tenant/session isolation."""
    sender_node_id: str
    sender_agent_id: str
    receiver_node_id: str
    receiver_agent_id: str
    tenant_id: str
    session_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_node_id": self.sender_node_id,
            "sender_agent_id": self.sender_agent_id,
            "receiver_node_id": self.receiver_node_id,
            "receiver_agent_id": self.receiver_agent_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class MessageIntegrity:
    """Cryptographic tamper-evidence, nonces, and signature verification."""
    fingerprint: str = ""
    nonce: str = ""
    signature: Optional[str] = None
    signer_identity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "nonce": self.nonce,
            "signature": self.signature,
            "signer_identity": self.signer_identity,
        }


@dataclass(frozen=True)
class DistributedMessageEnvelope:
    """
    Cryptographically verifiable message container for distributed transport.
    """
    envelope_id: str
    header: MessageHeader
    route: MessageRoute
    integrity: MessageIntegrity
    payload: Dict[str, Any] = field(default_factory=dict)

    def canonical_serialize(self) -> bytes:
        """Produce canonical, deterministic JSON-encoded bytes for signing/hashing."""
        canonical_dict = {
            "envelope_id": self.envelope_id,
            "header": self.header.to_dict(),
            "route": self.route.to_dict(),
            "payload": self.payload,
            "nonce": self.integrity.nonce,
        }
        return json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    def compute_integrity_hash(self) -> str:
        """Compute canonical SHA-256 fingerprint over envelope contents."""
        return hashlib.sha256(self.canonical_serialize()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that fingerprint matches the canonically serialized envelope payload."""
        if not self.integrity.fingerprint:
            return False
        return self.integrity.fingerprint == self.compute_integrity_hash()

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if message exceeds TTL."""
        now = current_time if current_time is not None else time.time()
        return (now - self.header.timestamp) > self.header.ttl_seconds

    def has_exceeded_hops(self) -> bool:
        """Check if hop count exceeded max allowable hops."""
        return self.header.hop_count > self.header.max_hops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "header": self.header.to_dict(),
            "route": self.route.to_dict(),
            "integrity": self.integrity.to_dict(),
            "payload": self.payload,
        }


# ============================================================================
# 4. Distributed Task Routing & Audit Telemetry
# ============================================================================

@dataclass(frozen=True)
class DistributedRouteDecision:
    """Deterministic routing decision allocating a task to a local or remote node."""
    task_id: str
    assigned_node_id: str
    assigned_agent_id: str
    assigned_role: AgentRole
    is_remote: bool
    estimated_latency_ms: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "assigned_node_id": self.assigned_node_id,
            "assigned_agent_id": self.assigned_agent_id,
            "assigned_role": self.assigned_role.value,
            "is_remote": self.is_remote,
            "estimated_latency_ms": self.estimated_latency_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SafePublicDistributedTrace:
    """
    Sanitized public telemetry for distributed federated execution.
    Exposes high-level routing, nodes, latencies, and consensus states without
    leaking internal chain-of-thought, weights, logits, or secret tokens.
    """
    trace_id: str
    task_id: str
    federation_id: str
    participating_node_ids: List[str]
    participating_agent_roles: List[str]
    routing_decisions: List[Dict[str, Any]]
    message_count: int
    remote_task_count: int
    retry_count: int
    failure_count: int
    conflict_count: int
    synthesis_state: str
    decision_state: str
    latency_ms: float
    final_status: str
    trace_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "federation_id": self.federation_id,
            "participating_node_ids": list(self.participating_node_ids),
            "participating_agent_roles": list(self.participating_agent_roles),
            "routing_decisions": list(self.routing_decisions),
            "message_count": self.message_count,
            "remote_task_count": self.remote_task_count,
            "retry_count": self.retry_count,
            "failure_count": self.failure_count,
            "conflict_count": self.conflict_count,
            "synthesis_state": self.synthesis_state,
            "decision_state": self.decision_state,
            "latency_ms": self.latency_ms,
            "final_status": self.final_status,
            "trace_fingerprint": self.trace_fingerprint,
        }
