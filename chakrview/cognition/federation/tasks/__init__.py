"""
Distributed Resource Orchestration, Fault-Tolerant Task Execution & Work Continuity (Step 38).

Provides sovereign, decentralized task decomposition, deterministic resource scheduling,
worker execution under CapabilityGate, tamper-evident checkpointing, failure recovery,
and idempotent result aggregation.

Axioms:
- LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
- ADVERTISEMENT != PERMISSION
- ADVERTISEMENT != EXECUTION_AUTHORITY
- RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
- UNREACHABLE != REVOKED
- ΔW = 0
"""

from chakrview.cognition.federation.tasks.errors import (
    FederationTaskError,
    InvalidTaskDefinitionError,
    InvalidWorkUnitError,
    InvalidTaskStateTransitionError,
    SchedulingError,
    NoEligibleWorkerError,
    UnauthorizedTaskExecutionError,
    ExecutionGrantViolationError,
    WorkerExecutionError,
    WorkerUnavailableError,
    InvalidCheckpointError,
    CheckpointCorruptionError,
    InvalidResultError,
    DuplicateResultError,
    StaleResultError,
    ResultAggregationError,
    TaskRecoveryError,
    TenantTaskIsolationError,
    TaskCancelledError,
)
from chakrview.cognition.federation.tasks.models import (
    TaskState,
    VALID_TASK_TRANSITIONS,
    WorkUnitState,
    VALID_WORK_UNIT_TRANSITIONS,
    AggregationStrategy,
    GrantType,
    ResourceRequirements,
    ResourceExecutionGrant,
    TaskCheckpoint,
    TaskResultEnvelope,
    WorkUnit,
    SchedulingDecision,
    DistributedTask,
    DistributedExecutionPlan,
)
from chakrview.cognition.federation.tasks.grant import (
    ExecutionGrantManager,
)
from chakrview.cognition.federation.tasks.scheduler import (
    DeterministicTaskScheduler,
)
from chakrview.cognition.federation.tasks.checkpoint import (
    TaskCheckpointManager,
)
from chakrview.cognition.federation.tasks.validator import (
    TaskResultValidator,
)
from chakrview.cognition.federation.tasks.aggregator import (
    TaskResultAggregator,
)
from chakrview.cognition.federation.tasks.executor import (
    FederationTaskExecutor,
)
from chakrview.cognition.federation.tasks.coordinator import (
    FederationTaskCoordinator,
)

__all__ = [
    # Errors
    "FederationTaskError",
    "InvalidTaskDefinitionError",
    "InvalidWorkUnitError",
    "InvalidTaskStateTransitionError",
    "SchedulingError",
    "NoEligibleWorkerError",
    "UnauthorizedTaskExecutionError",
    "ExecutionGrantViolationError",
    "WorkerExecutionError",
    "WorkerUnavailableError",
    "InvalidCheckpointError",
    "CheckpointCorruptionError",
    "InvalidResultError",
    "DuplicateResultError",
    "StaleResultError",
    "ResultAggregationError",
    "TaskRecoveryError",
    "TenantTaskIsolationError",
    "TaskCancelledError",
    # Models
    "TaskState",
    "VALID_TASK_TRANSITIONS",
    "WorkUnitState",
    "VALID_WORK_UNIT_TRANSITIONS",
    "AggregationStrategy",
    "GrantType",
    "ResourceRequirements",
    "ResourceExecutionGrant",
    "TaskCheckpoint",
    "TaskResultEnvelope",
    "WorkUnit",
    "SchedulingDecision",
    "DistributedTask",
    "DistributedExecutionPlan",
    # Core Subsystems
    "ExecutionGrantManager",
    "DeterministicTaskScheduler",
    "TaskCheckpointManager",
    "TaskResultValidator",
    "TaskResultAggregator",
    "FederationTaskExecutor",
    "FederationTaskCoordinator",
]
