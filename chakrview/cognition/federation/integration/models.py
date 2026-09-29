"""
Models and Configurations for End-to-End Federated Node Integration (Step 41).
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Set, Any


class NodeLifecycleState(str, Enum):
    """Lifecycle states of an integrated federated node."""
    UNINITIALIZED = "UNINITIALIZED"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass
class FederatedNodeConfig:
    """Configuration profile for bootstrapping a unified federated node."""
    node_id: str
    zone_id: str = "zone-default"
    tenant_id: str = "default"
    bind_host: str = "127.0.0.1"
    port: int = 9100
    auto_recover: bool = True
    bft_mode: bool = True
    consensus_validators: List[str] = field(default_factory=list)
    lease_duration_sec: float = 15.0
    heartbeat_timeout_sec: float = 5.0
    proposal_timeout_sec: float = 3.0
    max_concurrent_tasks: int = 4
    enable_persistence: bool = True


@dataclass
class FederatedNodeStatus:
    """Operational status snapshot of a federated node."""
    node_id: str
    lifecycle_state: NodeLifecycleState
    zone_id: str
    tenant_id: str
    active_peers_count: int
    active_tasks_count: int
    consensus_height: int
    state_root_hash: str
    is_consensus_leader: bool
    quarantined_nodes_count: int
    revoked_nodes_count: int
    uptime_seconds: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "lifecycle_state": self.lifecycle_state.value,
            "zone_id": self.zone_id,
            "tenant_id": self.tenant_id,
            "active_peers_count": self.active_peers_count,
            "active_tasks_count": self.active_tasks_count,
            "consensus_height": self.consensus_height,
            "state_root_hash": self.state_root_hash,
            "is_consensus_leader": self.is_consensus_leader,
            "quarantined_nodes_count": self.quarantined_nodes_count,
            "revoked_nodes_count": self.revoked_nodes_count,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp,
        }
