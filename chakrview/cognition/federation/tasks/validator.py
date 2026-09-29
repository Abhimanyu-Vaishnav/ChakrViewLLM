"""
Result Envelope Validation, Attempt Fencing & Commit Idempotency Enforcement (Steps 38 & 39).

CRITICAL AXIOMS:
- LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
- REMOTE RESULT IS NOT AUTOMATICALLY AUTHORITATIVE
- ATTEMPT FENCING: Results from obsolete, partitioned, or resurrected workers are rejected.
- IDEMPOTENT DEDUPLICATION & COMMIT GUARDS: Duplicate results and duplicate commits are blocked.
- LEASE DEADLINE ENFORCEMENT: A worker cannot commit results after its execution lease expires.
- INTEGRITY VERIFIED: All result payloads are cryptographically validated against their digests.
"""

from typing import Dict, Any, Optional

from chakrview.cognition.federation.tasks.models import (
    WorkUnit,
    WorkUnitState,
    TaskResultEnvelope,
)
from chakrview.cognition.federation.tasks.errors import (
    InvalidResultError,
    DuplicateResultError,
    DuplicateCommitError,
    StaleResultError,
    FencedAttemptError,
    LeaseExpiredError,
)


class TaskResultValidator:
    """
    Validates incoming worker results against work unit state, fencing tokens,
    lease deadlines, and security invariants.
    """

    @staticmethod
    def validate_result(unit: WorkUnit, result: TaskResultEnvelope) -> None:
        """
        Validate a TaskResultEnvelope for a given WorkUnit.
        
        Raises:
            InvalidResultError if envelope data or digest is invalid.
            DuplicateResultError if the unit is already COMPLETED.
            DuplicateCommitError if the unit has already committed a result.
            StaleResultError if the result attempt is older than the current unit attempt.
            FencedAttemptError if the result fencing token is superseded or invalid.
            LeaseExpiredError if the worker lease deadline has expired.
        """
        # 1. Identity match
        if result.task_id != unit.task_id:
            raise InvalidResultError(
                f"Result task_id '{result.task_id}' does not match unit task_id '{unit.task_id}'"
            )
        if result.unit_id != unit.unit_id:
            raise InvalidResultError(
                f"Result unit_id '{result.unit_id}' does not match unit '{unit.unit_id}'"
            )

        # 2. Worker assignment match
        if unit.assigned_node_id is not None and result.worker_id != unit.assigned_node_id:
            raise InvalidResultError(
                f"Result from worker '{result.worker_id}' rejected; unit {unit.unit_id} is assigned to '{unit.assigned_node_id}'"
            )

        # 3. Idempotency & Duplicate commit guards
        if unit.state == WorkUnitState.COMPLETED:
            raise DuplicateResultError(
                f"WorkUnit {unit.unit_id} has already completed. Duplicate result from {result.worker_id} rejected."
            )
        if getattr(unit, "committed_result", None) is not None:
            raise DuplicateCommitError(
                f"WorkUnit {unit.unit_id} has already committed a result. Duplicate commit blocked."
            )

        # 4. Attempt monotonicity: Stale attempt check
        if result.attempt < unit.attempt:
            raise StaleResultError(
                f"Stale result attempt {result.attempt} for unit {unit.unit_id} rejected; current attempt is {unit.attempt}"
            )

        # 5. Attempt fencing token verification (Step 39)
        if unit.fencing_token > 0 and result.fencing_token > 0:
            if result.fencing_token != unit.fencing_token:
                raise FencedAttemptError(
                    f"Fenced attempt for unit {unit.unit_id}: result fencing token {result.fencing_token} "
                    f"!= authoritative active token {unit.fencing_token}"
                )

        # 6. Worker lease deadline verification (Step 39)
        if unit.lease is not None:
            if unit.lease.is_expired():
                raise LeaseExpiredError(
                    f"Worker lease {unit.lease.lease_id} expired at {unit.lease.expires_at}. Result rejected."
                )

        # 7. Digest integrity verification
        if not result.verify_integrity():
            raise InvalidResultError(
                f"Result envelope for unit {unit.unit_id} failed cryptographic integrity verification. "
                f"Digest mismatch."
            )

        # 8. Basic sanity checks
        if result.execution_time_ms < 0:
            raise InvalidResultError(
                f"Invalid execution_time_ms: {result.execution_time_ms} (must be non-negative)"
            )
        if result.status not in ("SUCCESS", "FAILED"):
            raise InvalidResultError(
                f"Invalid result status '{result.status}'; must be 'SUCCESS' or 'FAILED'"
            )
