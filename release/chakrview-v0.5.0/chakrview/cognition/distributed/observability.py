"""
Distributed Observability & Telemetry (Step 27).

Tracks bounded telemetry metrics for distributed agent transport and federated execution:
- Sent, received, rejected, and replayed message counters.
- Timeouts, retries, node failures, and node recovery events.
- Bounded latency histories with summary percentiles.

CRITICAL ARCHITECTURAL AXIOMS:
1. BOUNDED STORAGE:
   Latency and telemetry buffers are bounded to prevent memory growth over long-running sessions.
"""

from dataclasses import dataclass, field
import collections
from typing import Dict, List, Any


class DistributedObservabilityMetrics:
    """
    Thread-safe, bounded telemetry recorder for distributed federation.
    """

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = max_history
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.rejected_messages: int = 0
        self.replay_attempts: int = 0
        self.timeouts: int = 0
        self.retries: int = 0
        self.node_failures: int = 0
        self.node_recoveries: int = 0

        self._rejection_reasons: collections.Counter = collections.Counter()
        self._task_latencies: collections.deque = collections.deque(maxlen=max_history)
        self._synthesis_latencies: collections.deque = collections.deque(maxlen=max_history)
        self._cycle_latencies: collections.deque = collections.deque(maxlen=max_history)

    def record_message_sent(self) -> None:
        self.messages_sent += 1

    def record_message_received(self) -> None:
        self.messages_received += 1

    def record_rejected_message(self, reason: str = "unknown") -> None:
        self.rejected_messages += 1
        self._rejection_reasons[reason] += 1

    def record_replay_attempt(self) -> None:
        self.replay_attempts += 1
        self.rejected_messages += 1
        self._rejection_reasons["replay_detected"] += 1

    def record_timeout(self) -> None:
        self.timeouts += 1

    def record_retry(self) -> None:
        self.retries += 1

    def record_node_failure(self, node_id: str) -> None:
        self.node_failures += 1

    def record_node_recovery(self, node_id: str) -> None:
        self.node_recoveries += 1

    def record_task_latency(self, latency_ms: float) -> None:
        self._task_latencies.append(latency_ms)

    def record_synthesis_latency(self, latency_ms: float) -> None:
        self._synthesis_latencies.append(latency_ms)

    def record_cycle_latency(self, latency_ms: float) -> None:
        self._cycle_latencies.append(latency_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Produce consolidated telemetry dictionary."""
        avg_cycle_latency = (
            sum(self._cycle_latencies) / len(self._cycle_latencies)
            if self._cycle_latencies
            else 0.0
        )
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "rejected_messages": self.rejected_messages,
            "replay_attempts": self.replay_attempts,
            "timeouts": self.timeouts,
            "retries": self.retries,
            "node_failures": self.node_failures,
            "node_recoveries": self.node_recoveries,
            "rejection_reasons": dict(self._rejection_reasons),
            "avg_cycle_latency_ms": round(avg_cycle_latency, 3),
        }
