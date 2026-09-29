"""
Worker Execution Lease, Heartbeat Monitoring & Attempt Fencing (Step 39).

CRITICAL AXIOMS:
1. UNREACHABLE != REVOKED:
   A temporary transport disconnect or missed heartbeat marks a lease UNREACHABLE
   or RECOVERABLE, triggering work migration without revoking cryptographic identity.
2. ATTEMPT FENCING:
   Every work unit attempt receives a strictly monotonic fencing token.
   Late worker writes or racing results from prior attempts are rejected immediately.
3. BOUNDED AUTHORITY:
   Execution authority is strictly bounded by lease expiration.
   A partitioned or crashed worker cannot commit progress once its lease expires.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

from chakrview.cognition.federation.tasks.models import (
    WorkerLease,
    LeaseState,
    AttemptFenceToken,
)
from chakrview.cognition.federation.tasks.errors import (
    FencedAttemptError,
    LeaseExpiredError,
    LeaseRevokedError,
    HeartbeatTimeoutError,
)

logger = logging.getLogger(__name__)


class AttemptFenceManager:
    """
    Coordinates monotonic fencing tokens and attempt generations per work unit.
    Prevents split-brain executions, resurrection of failed workers, and late writes.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Key: (task_id, unit_id) -> active AttemptFenceToken
        self._active_tokens: Dict[Tuple[str, str], AttemptFenceToken] = {}
        # Key: (task_id, unit_id) -> current generation counter
        self._generation_counters: Dict[Tuple[str, str], int] = {}

    def issue_fence_token(
        self,
        task_id: str,
        unit_id: str,
        attempt_number: int,
        worker_id: str,
    ) -> AttemptFenceToken:
        """
        Issue a new strictly monotonic fencing token for an execution attempt.
        Invalidates any previously active token for this work unit.
        """
        with self._lock:
            key = (task_id, unit_id)
            current_gen = self._generation_counters.get(key, 0) + 1
            self._generation_counters[key] = current_gen

            token = AttemptFenceToken(
                task_id=task_id,
                unit_id=unit_id,
                attempt_number=attempt_number,
                generation=current_gen,
                fencing_token=current_gen,
                worker_id=worker_id,
                issued_at=time.time(),
            )
            self._active_tokens[key] = token
            logger.debug(
                "Issued fencing token %d for unit %s (attempt %d, worker %s)",
                token.fencing_token, unit_id, attempt_number, worker_id,
            )
            return token

    def verify_attempt_token(
        self,
        task_id: str,
        unit_id: str,
        attempt_number: int,
        fencing_token: int,
        worker_id: str,
    ) -> bool:
        """
        Validate whether the presented attempt and fencing token are current.
        
        Raises:
            FencedAttemptError if the token is obsolete, lower than current generation,
            or assigned to another worker.
        """
        with self._lock:
            key = (task_id, unit_id)
            active = self._active_tokens.get(key)
            if active is None:
                raise FencedAttemptError(
                    f"No active fencing token registered for work unit {unit_id} (task {task_id})"
                )

            if active.fencing_token != fencing_token:
                raise FencedAttemptError(
                    f"Fencing token regression for unit {unit_id}: received token {fencing_token} "
                    f"!= authoritative active token {active.fencing_token}"
                )

            if active.attempt_number != attempt_number:
                raise FencedAttemptError(
                    f"Attempt number mismatch for unit {unit_id}: received attempt {attempt_number} "
                    f"!= authoritative attempt {active.attempt_number}"
                )

            if active.worker_id != worker_id:
                raise FencedAttemptError(
                    f"Worker mismatch for unit {unit_id}: token belongs to {active.worker_id}, "
                    f"not sender {worker_id}"
                )

            return True

    def get_active_token(self, task_id: str, unit_id: str) -> Optional[AttemptFenceToken]:
        """Retrieve current active fencing token for a work unit."""
        with self._lock:
            return self._active_tokens.get((task_id, unit_id))

    def fence_work_unit(self, task_id: str, unit_id: str) -> None:
        """Revoke active token, fencing off current worker without issuing a new one."""
        with self._lock:
            self._active_tokens.pop((task_id, unit_id), None)


class DeterministicFailureDetector:
    """
    Deterministic failure detection evaluating heartbeat arrival, execution duration,
    and transport status. Eliminates arbitrary non-deterministic timeouts.
    """

    def __init__(
        self,
        heartbeat_timeout_sec: float = 5.0,
        lease_duration_sec: float = 15.0,
        grace_period_sec: float = 2.0,
    ) -> None:
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.lease_duration_sec = lease_duration_sec
        self.grace_period_sec = grace_period_sec

    def evaluate_lease(
        self,
        lease: WorkerLease,
        current_time: Optional[float] = None,
        is_transport_connected: bool = True,
    ) -> Tuple[bool, LeaseState, str]:
        """
        Evaluate health of a worker lease.
        
        Returns:
            (is_healthy, determined_state, diagnosis_reason)
        """
        now = current_time if current_time is not None else time.time()

        # Terminal revocation check
        if lease.state == LeaseState.REVOKED:
            return False, LeaseState.REVOKED, "Worker cryptographic identity is revoked"

        # Explicit transport disconnect check
        if not is_transport_connected:
            return False, LeaseState.UNREACHABLE, "Transport channel disconnected"

        # Hard expiration check
        if now >= lease.expires_at:
            return False, LeaseState.LEASE_EXPIRED, f"Lease deadline exceeded ({now - lease.expires_at:.2f}s late)"

        # Heartbeat lateness check
        elapsed_since_heartbeat = now - lease.last_heartbeat_at
        if elapsed_since_heartbeat >= (lease.heartbeat_timeout_sec + self.grace_period_sec):
            return False, LeaseState.UNREACHABLE, f"Heartbeat dead interval exceeded ({elapsed_since_heartbeat:.2f}s)"

        if elapsed_since_heartbeat >= lease.heartbeat_timeout_sec:
            return True, LeaseState.HEARTBEAT_LATE, f"Heartbeat delayed ({elapsed_since_heartbeat:.2f}s)"

        return True, LeaseState.ACTIVE, "Healthy"


class WorkerLeaseManager:
    """
    Coordinates creation, renewal, expiration, and lifecycle tracking of worker leases.
    """

    def __init__(
        self,
        default_duration_sec: float = 15.0,
        default_heartbeat_timeout_sec: float = 5.0,
        failure_detector: Optional[DeterministicFailureDetector] = None,
    ) -> None:
        self.default_duration_sec = default_duration_sec
        self.default_heartbeat_timeout_sec = default_heartbeat_timeout_sec
        self.failure_detector = failure_detector or DeterministicFailureDetector(
            heartbeat_timeout_sec=default_heartbeat_timeout_sec,
            lease_duration_sec=default_duration_sec,
        )
        self._lock = threading.RLock()
        # Key: (task_id, unit_id) -> active WorkerLease
        self._leases: Dict[Tuple[str, str], WorkerLease] = {}
        # Key: worker_id -> Set[(task_id, unit_id)]
        self._worker_leases: Dict[str, set] = {}

    def grant_lease(
        self,
        worker_id: str,
        task_id: str,
        unit_id: str,
        attempt_number: int,
        fencing_token: int,
        duration_sec: Optional[float] = None,
    ) -> WorkerLease:
        """Grant a new execution lease to an assigned worker."""
        with self._lock:
            duration = duration_sec if duration_sec is not None else self.default_duration_sec
            now = time.time()
            lease_id = f"lease_{task_id}_{unit_id}_{attempt_number}_{int(now)}"

            lease = WorkerLease(
                lease_id=lease_id,
                worker_id=worker_id,
                task_id=task_id,
                unit_id=unit_id,
                attempt_number=attempt_number,
                fencing_token=fencing_token,
                state=LeaseState.ACTIVE,
                granted_at=now,
                duration_sec=duration,
                expires_at=now + duration,
                last_heartbeat_at=now,
                heartbeat_timeout_sec=self.default_heartbeat_timeout_sec,
            )

            key = (task_id, unit_id)
            self._leases[key] = lease

            if worker_id not in self._worker_leases:
                self._worker_leases[worker_id] = set()
            self._worker_leases[worker_id].add(key)

            logger.debug("Granted lease %s to worker %s for unit %s (duration=%.1fs)", lease_id, worker_id, unit_id, duration)
            return lease

    def renew_lease(
        self,
        task_id: str,
        unit_id: str,
        worker_id: str,
        fencing_token: int,
        current_time: Optional[float] = None,
    ) -> WorkerLease:
        """
        Record heartbeat and renew an active lease.
        
        Raises:
            FencedAttemptError if fencing token does not match active lease.
            LeaseExpiredError if lease has already lapsed.
            LeaseRevokedError if lease has been revoked.
        """
        with self._lock:
            key = (task_id, unit_id)
            lease = self._leases.get(key)
            if lease is None:
                raise LeaseExpiredError(f"No lease registered for unit {unit_id} of task {task_id}")

            if lease.worker_id != worker_id:
                raise FencedAttemptError(f"Lease worker mismatch: assigned to {lease.worker_id}, not {worker_id}")

            if lease.fencing_token != fencing_token:
                raise FencedAttemptError(
                    f"Fenced attempt on lease renewal for unit {unit_id}: token {fencing_token} "
                    f"!= active {lease.fencing_token}"
                )

            if lease.state == LeaseState.REVOKED:
                raise LeaseRevokedError(f"Lease {lease.lease_id} is revoked")

            if lease.state in (LeaseState.LEASE_EXPIRED, LeaseState.RECOVERABLE, LeaseState.UNREACHABLE):
                raise LeaseExpiredError(f"Lease {lease.lease_id} is not active (state={lease.state.value})")

            now = current_time if current_time is not None else time.time()
            if lease.is_expired(now):
                lease.state = LeaseState.LEASE_EXPIRED
                raise LeaseExpiredError(f"Lease {lease.lease_id} expired at {lease.expires_at} (current {now})")

            lease.renew(current_time=now)
            return lease

    def check_lease_health(
        self,
        task_id: str,
        unit_id: str,
        current_time: Optional[float] = None,
        is_transport_connected: bool = True,
    ) -> Tuple[bool, LeaseState, str]:
        """Check health of an individual work unit lease."""
        with self._lock:
            lease = self._leases.get((task_id, unit_id))
            if lease is None:
                return False, LeaseState.LEASE_EXPIRED, "No lease found"
            is_healthy, state, reason = self.failure_detector.evaluate_lease(
                lease, current_time=current_time, is_transport_connected=is_transport_connected
            )
            lease.state = state
            return is_healthy, state, reason

    def get_lease(self, task_id: str, unit_id: str) -> Optional[WorkerLease]:
        """Retrieve active lease for a work unit."""
        with self._lock:
            return self._leases.get((task_id, unit_id))

    def revoke_worker_leases(self, worker_id: str) -> List[WorkerLease]:
        """Terminal revocation of all active leases for a revoked worker."""
        revoked: List[WorkerLease] = []
        with self._lock:
            keys = list(self._worker_leases.get(worker_id, set()))
            for key in keys:
                lease = self._leases.get(key)
                if lease:
                    lease.state = LeaseState.REVOKED
                    revoked.append(lease)
            self._worker_leases.pop(worker_id, None)
        return revoked

    def expire_worker_leases(self, worker_id: str) -> List[WorkerLease]:
        """Mark all active leases for an unreachable/failed worker as recoverable."""
        expired: List[WorkerLease] = []
        with self._lock:
            keys = list(self._worker_leases.get(worker_id, set()))
            for key in keys:
                lease = self._leases.get(key)
                if lease and lease.state != LeaseState.REVOKED:
                    lease.state = LeaseState.RECOVERABLE
                    expired.append(lease)
            self._worker_leases.pop(worker_id, None)
        return expired

    def prune_leases_for_task(self, task_id: str) -> None:
        """Clean up leases for finished tasks."""
        with self._lock:
            keys_to_remove = [k for k in self._leases if k[0] == task_id]
            for key in keys_to_remove:
                lease = self._leases.pop(key, None)
                if lease and lease.worker_id in self._worker_leases:
                    self._worker_leases[lease.worker_id].discard(key)
