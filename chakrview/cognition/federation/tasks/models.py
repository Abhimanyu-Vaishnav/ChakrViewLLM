"""
Domain Models for Distributed Resource Orchestration, Fault-Tolerant Task Execution
and Work Continuity (Step 38).

Axioms:
- LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
- ADVERTISEMENT != PERMISSION
- ADVERTISEMENT != EXECUTION_AUTHORITY
- RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
- UNREACHABLE != REVOKED
- AT-LEAST-ONCE EXECUTION + IDEMPOTENCY + CHECKPOINTING = SAFE COMPLETION
- ΔW = 0
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.federation.tasks.errors import (
    InvalidTaskStateTransitionError,
    InvalidTaskDefinitionError,
    InvalidWorkUnitError,
    TenantTaskIsolationError,
)

DEFAULT_TASK_TIMEOUT_SECONDS: float = 300.0
DEFAULT_UNIT_TIMEOUT_SECONDS: float = 60.0
MAX_WORK_UNITS_PER_TASK: int = 1000
MAX_TASK_PAYLOAD_BYTES: int = 1048576  # 1 MB ceiling


# ============================================================================
# State Enums & Transition Maps
# ============================================================================

class TaskState(str, Enum):
    """Explicit lifecycle states for a DistributedTask."""
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PLANNING = "PLANNING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


VALID_TASK_TRANSITIONS: Dict[TaskState, Set[TaskState]] = {
    TaskState.CREATED: {
        TaskState.VALIDATING,
        TaskState.CANCELLED,
    },
    TaskState.VALIDATING: {
        TaskState.PLANNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.PLANNING: {
        TaskState.QUEUED,
        TaskState.RUNNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.QUEUED: {
        TaskState.RUNNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.EXPIRED,
    },
    TaskState.RUNNING: {
        TaskState.CHECKPOINTING,
        TaskState.PARTIALLY_COMPLETED,
        TaskState.RECOVERING,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.EXPIRED,
    },
    TaskState.CHECKPOINTING: {
        TaskState.RUNNING,
        TaskState.PARTIALLY_COMPLETED,
        TaskState.RECOVERING,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.PARTIALLY_COMPLETED: {
        TaskState.RUNNING,
        TaskState.RECOVERING,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.RECOVERING: {
        TaskState.RUNNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.COMPLETED: set(),  # Terminal state
    TaskState.FAILED: set(),     # Terminal state
    TaskState.CANCELLED: set(),  # Terminal state
    TaskState.EXPIRED: set(),    # Terminal state
}


class WorkUnitState(str, Enum):
    """Explicit lifecycle states for a single WorkUnit within a task."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"
    ABANDONED = "ABANDONED"


VALID_WORK_UNIT_TRANSITIONS: Dict[WorkUnitState, Set[WorkUnitState]] = {
    WorkUnitState.PENDING: {
        WorkUnitState.ASSIGNED,
        WorkUnitState.FAILED,
        WorkUnitState.ABANDONED,
    },
    WorkUnitState.ASSIGNED: {
        WorkUnitState.RUNNING,
        WorkUnitState.RETRYABLE,
        WorkUnitState.FAILED,
        WorkUnitState.ABANDONED,
    },
    WorkUnitState.RUNNING: {
        WorkUnitState.CHECKPOINTED,
        WorkUnitState.COMPLETED,
        WorkUnitState.RETRYABLE,
        WorkUnitState.FAILED,
        WorkUnitState.ABANDONED,
    },
    WorkUnitState.CHECKPOINTED: {
        WorkUnitState.RUNNING,
        WorkUnitState.COMPLETED,
        WorkUnitState.RETRYABLE,
        WorkUnitState.FAILED,
        WorkUnitState.ABANDONED,
    },
    WorkUnitState.RETRYABLE: {
        WorkUnitState.PENDING,
        WorkUnitState.ASSIGNED,
        WorkUnitState.FAILED,
        WorkUnitState.ABANDONED,
    },
    WorkUnitState.COMPLETED: set(),  # Terminal state
    WorkUnitState.FAILED: set(),     # Terminal state
    WorkUnitState.ABANDONED: set(),  # Terminal state
}


class AggregationStrategy(str, Enum):
    """Taxonomy of deterministic result merging strategies."""
    CONCATENATE = "CONCATENATE"      # Lists/strings concatenated by sequence order
    MERGE_DICT = "MERGE_DICT"        # Dicts merged by key
    REDUCE_SUM = "REDUCE_SUM"        # Numeric values summed
    FIRST_SUCCESS = "FIRST_SUCCESS"  # First valid completed result accepted
    CUSTOM = "CUSTOM"                # Registered handler strategy


class GrantType(str, Enum):
    """Scope of user device authorization for federated compute execution."""
    LOCAL_ONLY = "LOCAL_ONLY"                  # No remote execution permitted under any circumstance
    USER_APPROVED = "USER_APPROVED"            # Requires explicit manual confirmation per task
    TRUSTED_FEDERATION = "TRUSTED_FEDERATION"  # Automatically allows members with FEDERATED trust
    TASK_SCOPED = "TASK_SCOPED"                # Valid only for a specific task ID
    TIME_LIMITED = "TIME_LIMITED"              # Valid only until a defined expiration timestamp
    RESOURCE_LIMITED = "RESOURCE_LIMITED"      # Enforces hard CPU / RAM / concurrency limits


# ============================================================================
# Resource Requirements & Execution Grants
# ============================================================================

@dataclass
class ResourceRequirements:
    """Computational and capability requirements for a task or work unit."""
    capability_id: str
    min_cpu_cores: float = 1.0
    min_memory_mb: int = 512
    requires_accelerator: bool = False
    timeout_seconds: float = DEFAULT_UNIT_TIMEOUT_SECONDS
    tenant_id: str = "default"
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "min_cpu_cores": self.min_cpu_cores,
            "min_memory_mb": self.min_memory_mb,
            "requires_accelerator": self.requires_accelerator,
            "timeout_seconds": self.timeout_seconds,
            "tenant_id": self.tenant_id,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceRequirements":
        return cls(**data)


@dataclass
class ResourceExecutionGrant:
    """
    Device-owner sovereign authorization defining whether, when, and how much compute
    a node will share for federated task execution.
    
    Axioms:
    ADVERTISEMENT != PERMISSION
    ADVERTISEMENT != EXECUTION_AUTHORITY
    """
    grant_id: str
    grant_type: GrantType
    owner_node_id: str
    allowed_peer_nodes: Set[str] = field(default_factory=lambda: {"*"})
    allowed_tenants: Set[str] = field(default_factory=lambda: {"*"})
    allowed_capability_ids: Set[str] = field(default_factory=lambda: {"*"})
    max_concurrent_units: int = 2
    max_memory_mb: int = 2048
    max_cores: float = 2.0
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    def is_authorized(
        self,
        peer_node_id: str,
        tenant_id: str,
        capability_id: str,
        req: Optional[ResourceRequirements] = None,
    ) -> Tuple[bool, str]:
        """
        Verify whether an execution request satisfies this sovereign grant.
        Returns (is_authorized, reason_string).
        """
        if self.grant_type == GrantType.LOCAL_ONLY:
            if peer_node_id != self.owner_node_id:
                return False, "Node policy is LOCAL_ONLY; remote execution forbidden"

        if self.expires_at is not None and time.time() > self.expires_at:
            return False, f"ResourceExecutionGrant {self.grant_id} expired at {self.expires_at}"

        # Peer node filter
        if "*" not in self.allowed_peer_nodes and peer_node_id not in self.allowed_peer_nodes:
            return False, f"Peer {peer_node_id} not in allowed_peer_nodes for grant {self.grant_id}"

        # Tenant filter
        if "*" not in self.allowed_tenants and tenant_id not in self.allowed_tenants:
            return False, f"Tenant {tenant_id} not in allowed_tenants for grant {self.grant_id}"

        # Capability filter
        if "*" not in self.allowed_capability_ids and capability_id not in self.allowed_capability_ids:
            return False, f"Capability {capability_id} not in allowed_capability_ids for grant {self.grant_id}"

        # Resource limits filter
        if req is not None:
            if req.min_cpu_cores > self.max_cores:
                return False, f"Requested CPU cores {req.min_cpu_cores} exceeds grant limit {self.max_cores}"
            if req.min_memory_mb > self.max_memory_mb:
                return False, f"Requested memory {req.min_memory_mb} MB exceeds grant limit {self.max_memory_mb} MB"

        return True, "Authorized"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "grant_type": self.grant_type.value,
            "owner_node_id": self.owner_node_id,
            "allowed_peer_nodes": sorted(list(self.allowed_peer_nodes)),
            "allowed_tenants": sorted(list(self.allowed_tenants)),
            "allowed_capability_ids": sorted(list(self.allowed_capability_ids)),
            "max_concurrent_units": self.max_concurrent_units,
            "max_memory_mb": self.max_memory_mb,
            "max_cores": self.max_cores,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceExecutionGrant":
        d = dict(data)
        d["grant_type"] = GrantType(d["grant_type"])
        d["allowed_peer_nodes"] = set(d.get("allowed_peer_nodes", ["*"]))
        d["allowed_tenants"] = set(d.get("allowed_tenants", ["*"]))
        d["allowed_capability_ids"] = set(d.get("allowed_capability_ids", ["*"]))
        return cls(**d)


# ============================================================================
# Checkpoint & Result Models
# ============================================================================

@dataclass
class TaskCheckpoint:
    """
    Durable intermediate progress record for a work unit.
    Allows work to be resumed on another worker without starting from scratch.
    """
    task_id: str
    unit_id: str
    checkpoint_id: str
    sequence: int
    state: WorkUnitState
    progress: float               # 0.0 to 1.0
    partial_result: Any
    worker_id: str
    attempt: int
    created_at: float = field(default_factory=time.time)
    integrity_digest: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_digest:
            self.integrity_digest = self.compute_digest()

    def compute_digest(self) -> str:
        """Compute tamper-evident SHA-256 digest of checkpoint state."""
        canonical_str = f"{self.task_id}:{self.unit_id}:{self.checkpoint_id}:{self.sequence}:{self.state.value}:{round(self.progress, 4)}:{self.worker_id}:{self.attempt}:{round(self.created_at, 2)}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that the checkpoint state matches its integrity digest."""
        return self.compute_digest() == self.integrity_digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "unit_id": self.unit_id,
            "checkpoint_id": self.checkpoint_id,
            "sequence": self.sequence,
            "state": self.state.value,
            "progress": self.progress,
            "partial_result": self.partial_result,
            "worker_id": self.worker_id,
            "attempt": self.attempt,
            "created_at": self.created_at,
            "integrity_digest": self.integrity_digest,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskCheckpoint":
        d = dict(data)
        d["state"] = WorkUnitState(d["state"])
        return cls(**d)


@dataclass
class TaskResultEnvelope:
    """
    Validated worker result container carrying unit output, execution metadata,
    and integrity proofs.
    """
    task_id: str
    unit_id: str
    attempt: int
    worker_id: str
    status: str                   # "SUCCESS" or "FAILED"
    result_data: Any
    execution_time_ms: float
    error_message: Optional[str] = None
    result_digest: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.result_digest:
            self.result_digest = self.compute_digest()

    def compute_digest(self) -> str:
        """Deterministic digest of the result content and provenance."""
        payload_repr = json.dumps(self.result_data, sort_keys=True, separators=(',', ':')) if self.result_data is not None else ""
        canonical_str = f"{self.task_id}:{self.unit_id}:{self.attempt}:{self.worker_id}:{self.status}:{payload_repr}:{self.error_message or ''}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        return self.compute_digest() == self.result_digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "unit_id": self.unit_id,
            "attempt": self.attempt,
            "worker_id": self.worker_id,
            "status": self.status,
            "result_data": self.result_data,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "result_digest": self.result_digest,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResultEnvelope":
        return cls(**data)


# ============================================================================
# WorkUnit & DistributedTask Models
# ============================================================================

@dataclass
class WorkUnit:
    """
    An independent or checkpointable slice of a distributed task.
    Identified deterministically, idempotent across retries.
    """
    unit_id: str
    task_id: str
    sequence: int
    capability_id: str
    input_payload: Dict[str, Any]
    requirements: ResourceRequirements
    state: WorkUnitState = WorkUnitState.PENDING
    attempt: int = 0
    assigned_node_id: Optional[str] = None
    checkpoint_reference: Optional[str] = None
    result_reference: Optional[str] = None
    latest_checkpoint: Optional[TaskCheckpoint] = None
    result: Optional[TaskResultEnvelope] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def transition_to(self, new_state: WorkUnitState) -> None:
        """Validate and apply a lifecycle transition."""
        allowed = VALID_WORK_UNIT_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTaskStateTransitionError(
                f"Illegal WorkUnit state transition from {self.state.value} to {new_state.value} "
                f"for unit {self.unit_id} (allowed: {[s.value for s in allowed]})"
            )
        self.state = new_state
        self.updated_at = time.time()

    def compute_input_digest(self) -> str:
        """Deterministic digest of unit input payload."""
        canonical_json = json.dumps(self.input_payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "task_id": self.task_id,
            "sequence": self.sequence,
            "capability_id": self.capability_id,
            "input_payload": self.input_payload,
            "requirements": self.requirements.to_dict(),
            "state": self.state.value,
            "attempt": self.attempt,
            "assigned_node_id": self.assigned_node_id,
            "checkpoint_reference": self.checkpoint_reference,
            "result_reference": self.result_reference,
            "latest_checkpoint": self.latest_checkpoint.to_dict() if self.latest_checkpoint else None,
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkUnit":
        d = dict(data)
        d["requirements"] = ResourceRequirements.from_dict(d["requirements"])
        d["state"] = WorkUnitState(d["state"])
        if d.get("latest_checkpoint"):
            d["latest_checkpoint"] = TaskCheckpoint.from_dict(d["latest_checkpoint"])
        if d.get("result"):
            d["result"] = TaskResultEnvelope.from_dict(d["result"])
        return cls(**d)


@dataclass
class SchedulingDecision:
    """
    Auditable, explainable, and reproducible record of a scheduler decision.
    """
    decision_id: str
    task_id: str
    unit_id: str
    candidate_nodes: List[str]
    selected_node: str
    reason_codes: List[str]
    policy_version: int = 1
    resource_snapshot_reference: str = ""
    decision_digest: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.decision_digest:
            canonical_str = f"{self.task_id}:{self.unit_id}:{','.join(self.candidate_nodes)}:{self.selected_node}:{','.join(self.reason_codes)}:{round(self.timestamp, 2)}"
            self.decision_digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "task_id": self.task_id,
            "unit_id": self.unit_id,
            "candidate_nodes": self.candidate_nodes,
            "selected_node": self.selected_node,
            "reason_codes": self.reason_codes,
            "policy_version": self.policy_version,
            "resource_snapshot_reference": self.resource_snapshot_reference,
            "decision_digest": self.decision_digest,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulingDecision":
        return cls(**data)


@dataclass
class DistributedTask:
    """
    A user-requested or system-generated distributed workload decomposable
    into discrete WorkUnits.
    """
    task_id: str
    name: str
    tenant_id: str = "default"
    description: str = ""
    created_by_node_id: str = ""
    priority: int = 5                        # 1 (lowest) to 10 (highest)
    deadline: Optional[float] = None
    state: TaskState = TaskState.CREATED
    requirements: Optional[ResourceRequirements] = None
    work_units: List[WorkUnit] = field(default_factory=list)
    aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE
    scheduling_decisions: List[SchedulingDecision] = field(default_factory=list)
    final_result: Optional[Any] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.deadline:
            self.deadline = self.created_at + DEFAULT_TASK_TIMEOUT_SECONDS

    def transition_to(self, new_state: TaskState) -> None:
        """Validate and apply a lifecycle transition."""
        allowed = VALID_TASK_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTaskStateTransitionError(
                f"Illegal Task state transition from {self.state.value} to {new_state.value} "
                f"for task {self.task_id} (allowed: {[s.value for s in allowed]})"
            )
        self.state = new_state
        self.updated_at = time.time()

    def get_unit(self, unit_id: str) -> Optional[WorkUnit]:
        """Look up a work unit by ID."""
        for u in self.work_units:
            if u.unit_id == unit_id:
                return u
        return None

    def are_all_units_completed(self) -> bool:
        """Check if all work units have completed successfully."""
        if not self.work_units:
            return False
        return all(u.state == WorkUnitState.COMPLETED for u in self.work_units)

    def get_unfinished_units(self) -> List[WorkUnit]:
        """Return all units that are not in COMPLETED or ABANDONED states."""
        return [u for u in self.work_units if u.state not in (WorkUnitState.COMPLETED, WorkUnitState.ABANDONED)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "tenant_id": self.tenant_id,
            "description": self.description,
            "created_by_node_id": self.created_by_node_id,
            "priority": self.priority,
            "deadline": self.deadline,
            "state": self.state.value,
            "requirements": self.requirements.to_dict() if self.requirements else None,
            "work_units": [u.to_dict() for u in self.work_units],
            "aggregation_strategy": self.aggregation_strategy.value,
            "scheduling_decisions": [d.to_dict() for d in self.scheduling_decisions],
            "final_result": self.final_result,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedTask":
        d = dict(data)
        d["state"] = TaskState(d["state"])
        d["aggregation_strategy"] = AggregationStrategy(d["aggregation_strategy"])
        if d.get("requirements"):
            d["requirements"] = ResourceRequirements.from_dict(d["requirements"])
        d["work_units"] = [WorkUnit.from_dict(u) for u in d.get("work_units", [])]
        d["scheduling_decisions"] = [SchedulingDecision.from_dict(sd) for sd in d.get("scheduling_decisions", [])]
        return cls(**d)


@dataclass
class DistributedExecutionPlan:
    """
    High-level distributed execution plan for heterogeneous multi-device tasks.
    Precursor abstraction for future multi-device neural inference and cognitive jobs.
    """
    plan_id: str
    task_id: str
    tenant_id: str
    participating_nodes: List[str]
    planned_units: List[Dict[str, Any]]
    total_estimated_cores: float
    total_estimated_memory_mb: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "tenant_id": self.tenant_id,
            "participating_nodes": self.participating_nodes,
            "planned_units": self.planned_units,
            "total_estimated_cores": self.total_estimated_cores,
            "total_estimated_memory_mb": self.total_estimated_memory_mb,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedExecutionPlan":
        return cls(**data)
