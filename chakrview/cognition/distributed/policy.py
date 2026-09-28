"""
Distributed Execution Policy and Resource Adaptation (Step 27).

Maps system hardware profiles (LOW_RESOURCE, STANDARD, HIGH_RESOURCE) to distributed
execution constraints while strictly preserving core neural and safety invariants:
- Bounded node counts, task ceilings, and retry limits.
- Hard architectural ceilings universally enforced regardless of hardware tier.

CRITICAL ARCHITECTURAL AXIOMS:
1. IDENTICAL NEURAL BRAIN:
   Resource profiles scale remote concurrency, timeouts, and retry budgets.
   They NEVER alter ChakrMicro v0.1 parameters (3,443,136), vocab (4,096), or context (512).
"""

from dataclasses import dataclass
from typing import Dict, Any

from chakrview.cognition.distributed.models import (
    MAX_FEDERATION_NODES,
    MAX_REMOTE_TASKS_PER_CYCLE,
    MAX_FEDERATION_DEPTH,
    MAX_MESSAGE_HOPS,
)


@dataclass(frozen=True)
class DistributedExecutionPolicy:
    """
    Execution budget and resilience policy for distributed federated cognition.
    """
    profile_name: str = "STANDARD"
    max_nodes: int = 8
    max_remote_tasks: int = 16
    max_retries: int = 2
    request_timeout_ms: float = 1000.0
    total_cycle_timeout_ms: float = 10000.0
    max_concurrent_requests: int = 4
    max_message_hops: int = 4
    max_tracked_replays: int = 1024

    def __post_init__(self) -> None:
        # Enforce universal hard ceilings
        object.__setattr__(self, "max_nodes", min(self.max_nodes, MAX_FEDERATION_NODES))
        object.__setattr__(self, "max_remote_tasks", min(self.max_remote_tasks, MAX_REMOTE_TASKS_PER_CYCLE))
        object.__setattr__(self, "max_message_hops", min(self.max_message_hops, MAX_MESSAGE_HOPS))

    @classmethod
    def from_profile(cls, profile_name: str = "STANDARD") -> "DistributedExecutionPolicy":
        profile = profile_name.upper()
        if profile == "LOW_RESOURCE":
            return cls(
                profile_name="LOW_RESOURCE",
                max_nodes=4,
                max_remote_tasks=8,
                max_retries=1,
                request_timeout_ms=500.0,
                total_cycle_timeout_ms=5000.0,
                max_concurrent_requests=2,
                max_message_hops=2,
                max_tracked_replays=512,
            )
        elif profile == "HIGH_RESOURCE":
            return cls(
                profile_name="HIGH_RESOURCE",
                max_nodes=16,
                max_remote_tasks=32,
                max_retries=3,
                request_timeout_ms=2000.0,
                total_cycle_timeout_ms=15000.0,
                max_concurrent_requests=8,
                max_message_hops=4,
                max_tracked_replays=2048,
            )
        else:  # STANDARD
            return cls(
                profile_name="STANDARD",
                max_nodes=8,
                max_remote_tasks=16,
                max_retries=2,
                request_timeout_ms=1000.0,
                total_cycle_timeout_ms=10000.0,
                max_concurrent_requests=4,
                max_message_hops=4,
                max_tracked_replays=1024,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "max_nodes": self.max_nodes,
            "max_remote_tasks": self.max_remote_tasks,
            "max_retries": self.max_retries,
            "request_timeout_ms": self.request_timeout_ms,
            "total_cycle_timeout_ms": self.total_cycle_timeout_ms,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_message_hops": self.max_message_hops,
            "max_tracked_replays": self.max_tracked_replays,
        }
