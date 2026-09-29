"""
Result Envelope Validation & Idempotency Enforcement (Step 38).

CRITICAL AXIOMS:
- LOCAL_TASK_AUTHORITY > REMOTE_WORKER_STATE
- REMOTE RESULT IS NOT AUTOMATICALLY AUTHORITATIVE
- IDEMPOTENT DEDUPLICATION: Duplicate results from racing or retried workers are rejected.
- STALE RESULTS DROPPED: Late results from superseded worker attempts are discarded.
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
    StaleResultError,
)


class TaskResultValidator:
    """
    Validates incoming worker results against work unit state and security invariants.
    """

    @staticmethod
    def validate_result(unit: WorkUnit, result: TaskResultEnvelope) -> None:
        """
        Validate a TaskResultEnvelope for a given WorkUnit.
        
        Raises:
            InvalidResultError if envelope data or digest is invalid.
            DuplicateResultError if the unit is already COMPLETED.
            StaleResultError if the result attempt is older than the current unit attempt.
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

        # 3. Idempotency guard: Duplicate check
        if unit.state == WorkUnitState.COMPLETED:
            raise DuplicateResultError(
                f"WorkUnit {unit.unit_id} has already completed. Duplicate result from {result.worker_id} rejected."
            )

        # 4. Attempt monotonicity: Stale attempt check
        if result.attempt < unit.attempt:
            raise StaleResultError(
                f"Stale result attempt {result.attempt} for unit {unit.unit_id} rejected; current attempt is {unit.attempt}"
            )

        # 5. Digest integrity verification
        if not result.verify_integrity():
            raise InvalidResultError(
                f"Result envelope for unit {unit.unit_id} failed cryptographic integrity verification. "
                f"Digest mismatch."
            )

        # 6. Basic sanity checks
        if result.execution_time_ms < 0:
            raise InvalidResultError(
                f"Invalid execution_time_ms: {result.execution_time_ms} (must be non-negative)"
            )
        if result.status not in ("SUCCESS", "FAILED"):
            raise InvalidResultError(
                f"Invalid result status '{result.status}'; must be 'SUCCESS' or 'FAILED'"
            )
