"""
Typed Exception Hierarchy for Distributed Task Orchestration & Execution (Step 38).

CRITICAL ARCHITECTURAL AXIOMS:
1. LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE:
   No remote worker can unilaterally claim completion, cancel a task, or modify policy.
2. FAIL-CLOSED EXECUTION:
   Any schema violation, unknown capability, authorization failure, or corrupted
   checkpoint halts execution and raises a typed FederationTaskError.
3. RECOVERY RESILIENCE:
   Failures and disconnects trigger controlled reassignment rather than cascading aborts.
"""

from typing import Optional


class FederationTaskError(Exception):
    """Base exception for all distributed task orchestration failures."""
    pass


class InvalidTaskDefinitionError(FederationTaskError):
    """Raised when a task definition fails validation, schema checks, or boundary rules."""
    pass


class InvalidWorkUnitError(FederationTaskError):
    """Raised when a work unit contains invalid inputs, invalid sequence, or illegal states."""
    pass


class InvalidTaskStateTransitionError(FederationTaskError):
    """Raised when an illegal lifecycle transition is attempted on a task or work unit."""
    pass


class SchedulingError(FederationTaskError):
    """Base exception for scheduler failures."""
    pass


class NoEligibleWorkerError(SchedulingError):
    """Raised when no federated node satisfies the resource, capability, and trust requirements."""
    pass


class UnauthorizedTaskExecutionError(FederationTaskError):
    """Raised when task execution is denied by sovereign local policy or missing grant."""
    pass


class ExecutionGrantViolationError(UnauthorizedTaskExecutionError):
    """Raised when a task exceeds resource ceilings, tenant scope, or temporal limits of a grant."""
    pass


class WorkerExecutionError(FederationTaskError):
    """Raised when worker-side capability execution fails."""
    pass


class WorkerUnavailableError(FederationTaskError):
    """Raised when a worker node is unreachable, disconnected, quarantined, or revoked."""
    pass


class InvalidCheckpointError(FederationTaskError):
    """Raised when a checkpoint fails validation, sequence check, or integrity verification."""
    pass


class CheckpointCorruptionError(InvalidCheckpointError):
    """Raised when a stored checkpoint exhibits payload tampering or digest mismatch."""
    pass


class InvalidResultError(FederationTaskError):
    """Raised when a worker result envelope fails validation or integrity checks."""
    pass


class DuplicateResultError(InvalidResultError):
    """Raised when a result for an already-completed unit is received (idempotency guard)."""
    pass


class StaleResultError(InvalidResultError):
    """Raised when a result is received from a superseded worker attempt or expired assignment."""
    pass


class ResultAggregationError(FederationTaskError):
    """Raised when partial results cannot be merged under the declared aggregation strategy."""
    pass


class TaskRecoveryError(FederationTaskError):
    """Raised when recovering task state from journal or checkpoint store fails."""
    pass


class TenantTaskIsolationError(FederationTaskError):
    """Raised when a cross-tenant execution attempt is detected and blocked."""
    pass


class TaskCancelledError(FederationTaskError):
    """Raised when an operation is attempted on a task that has been cancelled."""
    pass
