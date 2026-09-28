"""
Distributed Resilience: Timeouts, Retries, and Circuit Breakers (Step 27).

Provides bounded failure handling, deterministic backoff calculation, and fault isolation:
- TimeoutPolicy & RetryPolicy
- CircuitBreaker (CLOSED, OPEN, HALF_OPEN states)
- FailureRecord & RecoveryState
- Zero retry storms, bounded ceilings

CRITICAL ARCHITECTURAL AXIOMS:
1. BOUNDED RETRIES & NO RETRY STORMS:
   Retries are hard-capped; exponential backoff prevents cascading node overload.
2. DETERMINISTIC TIMING:
   Backoff calculation is deterministic (no random jitter by default in tests) to
   ensure reproducible execution.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional


class CircuitBreakerState(str, Enum):
    """Operational state of a node circuit breaker."""
    CLOSED = "CLOSED"          # Normal operation; requests pass through
    OPEN = "OPEN"              # Tripped due to failures; requests fail fast
    HALF_OPEN = "HALF_OPEN"    # Probing recovery with limited test requests


@dataclass(frozen=True)
class TimeoutPolicy:
    """Bounded timeout configuration for distributed transactions."""
    request_timeout_ms: float = 1000.0
    total_cycle_timeout_ms: float = 10000.0
    max_timeout_ceiling_ms: float = 30000.0

    def get_effective_timeout_ms(self) -> float:
        return min(self.request_timeout_ms, self.max_timeout_ceiling_ms)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry and exponential backoff configuration."""
    max_retries: int = 2
    base_backoff_ms: float = 50.0
    backoff_multiplier: float = 2.0
    max_backoff_ms: float = 1000.0

    def calculate_backoff_ms(self, attempt: int) -> float:
        """Deterministic exponential backoff calculation."""
        if attempt <= 0:
            return 0.0
        backoff = self.base_backoff_ms * (self.backoff_multiplier ** (attempt - 1))
        return min(backoff, self.max_backoff_ms)


@dataclass
class FailureRecord:
    """Audit log entry for a remote node or task failure."""
    timestamp: float
    node_id: str
    task_id: str
    error_type: str
    error_message: str


class CircuitBreaker:
    """
    Per-node failure isolation guard. Trips open after consecutive failures,
    preventing wasted transport requests to failing or degraded nodes.
    """

    def __init__(
        self,
        node_id: str,
        failure_threshold: int = 3,
        recovery_timeout_ms: float = 3000.0,
        probe_success_threshold: int = 2,
    ) -> None:
        self.node_id = node_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.probe_success_threshold = probe_success_threshold

        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._last_state_change: float = time.time()
        self._failure_history: List[FailureRecord] = []

    @property
    def state(self) -> CircuitBreakerState:
        # Check if OPEN duration has expired -> transition to HALF_OPEN
        if self._state == CircuitBreakerState.OPEN:
            elapsed_ms = (time.time() - self._last_state_change) * 1000.0
            if elapsed_ms >= self.recovery_timeout_ms:
                self._state = CircuitBreakerState.HALF_OPEN
                self._last_state_change = time.time()
                self._consecutive_successes = 0
        return self._state

    def can_execute(self) -> bool:
        """Determines if requests are allowed through to the node."""
        st = self.state
        if st == CircuitBreakerState.CLOSED:
            return True
        if st == CircuitBreakerState.HALF_OPEN:
            return True  # Allows probing
        return False

    def record_success(self) -> None:
        """Register successful interaction."""
        st = self.state
        if st == CircuitBreakerState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.probe_success_threshold:
                self._state = CircuitBreakerState.CLOSED
                self._consecutive_failures = 0
                self._last_state_change = time.time()
        elif st == CircuitBreakerState.CLOSED:
            self._consecutive_failures = 0

    def record_failure(self, task_id: str, error_type: str, error_message: str) -> None:
        """Register interaction failure."""
        record = FailureRecord(
            timestamp=time.time(),
            node_id=self.node_id,
            task_id=task_id,
            error_type=error_type,
            error_message=error_message,
        )
        self._failure_history.append(record)
        self._consecutive_failures += 1

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Immediate re-trip to OPEN
            self._state = CircuitBreakerState.OPEN
            self._last_state_change = time.time()
        elif self._state == CircuitBreakerState.CLOSED:
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._last_state_change = time.time()

    def get_failure_records(self) -> List[FailureRecord]:
        return list(self._failure_history)
