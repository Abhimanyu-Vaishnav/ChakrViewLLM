"""
Bounded Observability and Telemetry for Step 28 Cognitive Orchestration.

Tracks high-level operational metrics with bounded memory structures
to monitor adaptation, resource utilization, and decision states.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

from chakrview.cognition.orchestration.models import WorkloadClass
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.cognition.unified.models import DecisionState


@dataclass
class OrchestrationObservabilityMetrics:
    """
    Bounded telemetry tracker for Step 28 cognitive orchestration.
    """
    max_history: int = 1000

    # Counters
    tasks_orchestrated: int = 0
    verification_requests: int = 0
    verification_passes: int = 0
    verification_failures: int = 0
    uncertainty_count: int = 0
    safe_stop_count: int = 0
    agent_failures: int = 0
    node_failures: int = 0
    retries_executed: int = 0

    # Distribution tallies
    workload_counts: Dict[str, int] = field(default_factory=lambda: {w.value: 0 for w in WorkloadClass})
    profile_usage: Dict[str, int] = field(default_factory=lambda: {p.value: 0 for p in ResourceProfile})

    # Rolling window histories for averages
    _agent_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    _node_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    _round_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    _latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_orchestration(
        self,
        workload_class: WorkloadClass,
        profile: ResourceProfile,
        agent_count: int,
        node_count: int,
        rounds: int,
        latency_ms: float,
        decision_state: DecisionState,
        verification_invoked: bool = False,
        verification_passed: Optional[bool] = None,
        retries: int = 0,
        agent_failures: int = 0,
        node_failures: int = 0,
    ) -> None:
        """Record telemetry for a single orchestration cycle."""
        self.tasks_orchestrated += 1
        self.workload_counts[workload_class.value] = self.workload_counts.get(workload_class.value, 0) + 1
        self.profile_usage[profile.value] = self.profile_usage.get(profile.value, 0) + 1

        self._agent_counts.append(agent_count)
        self._node_counts.append(node_count)
        self._round_counts.append(rounds)
        self._latencies_ms.append(latency_ms)

        if verification_invoked:
            self.verification_requests += 1
            if verification_passed is True:
                self.verification_passes += 1
            elif verification_passed is False:
                self.verification_failures += 1

        if decision_state in (DecisionState.ANSWER_WITH_UNCERTAINTY, DecisionState.INSUFFICIENT_INFORMATION):
            self.uncertainty_count += 1
        elif decision_state == DecisionState.SAFE_STOP:
            self.safe_stop_count += 1

        self.retries_executed += retries
        self.agent_failures += agent_failures
        self.node_failures += node_failures

    @property
    def average_agent_count(self) -> float:
        return sum(self._agent_counts) / len(self._agent_counts) if self._agent_counts else 0.0

    @property
    def average_node_count(self) -> float:
        return sum(self._node_counts) / len(self._node_counts) if self._node_counts else 0.0

    @property
    def average_deliberation_rounds(self) -> float:
        return sum(self._round_counts) / len(self._round_counts) if self._round_counts else 0.0

    @property
    def average_latency_ms(self) -> float:
        return sum(self._latencies_ms) / len(self._latencies_ms) if self._latencies_ms else 0.0

    @property
    def uncertainty_rate(self) -> float:
        return self.uncertainty_count / self.tasks_orchestrated if self.tasks_orchestrated > 0 else 0.0

    @property
    def safe_stop_rate(self) -> float:
        return self.safe_stop_count / self.tasks_orchestrated if self.tasks_orchestrated > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks_orchestrated": self.tasks_orchestrated,
            "workload_counts": dict(self.workload_counts),
            "profile_usage": dict(self.profile_usage),
            "average_agent_count": round(self.average_agent_count, 2),
            "average_node_count": round(self.average_node_count, 2),
            "average_deliberation_rounds": round(self.average_deliberation_rounds, 2),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "uncertainty_rate": round(self.uncertainty_rate, 4),
            "safe_stop_rate": round(self.safe_stop_rate, 4),
            "verification_requests": self.verification_requests,
            "retries_executed": self.retries_executed,
            "agent_failures": self.agent_failures,
            "node_failures": self.node_failures,
        }
