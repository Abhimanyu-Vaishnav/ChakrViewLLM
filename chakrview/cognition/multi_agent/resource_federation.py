"""
ChakrView Step 115: Resource-Aware Worker Federation.

Extends worker metadata and scheduler with hardware resource profiling,
capacity scoring, and load-aware deterministic worker placement:
- ResourceCapacityLevel: LOW_RESOURCE, STANDARD, HIGH_RESOURCE
- WorkerResourceProfile: CPU count, memory budget, current load, context capacity
- ResourceAwareSelector: Deterministic placement matching task requirements to workers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.multi_agent.contracts import WorkerRole, WorkerContract
from chakrview.cognition.adaptation.profiles import ResourceProfile


class ResourceCapacityLevel(str, Enum):
    """Resource capacity classification for federated worker nodes."""
    LOW_RESOURCE = "LOW_RESOURCE"
    STANDARD = "STANDARD"
    HIGH_RESOURCE = "HIGH_RESOURCE"


@dataclass
class WorkerResourceProfile:
    """
    Bounded resource footprint and capacity profile of a federated worker node.
    """
    worker_id: str
    role: WorkerRole
    capacity_level: ResourceCapacityLevel = ResourceCapacityLevel.STANDARD
    cpu_cores: int = 2
    memory_budget_mb: int = 1024
    max_context_tokens: int = 512
    max_concurrency: int = 2
    current_active_tasks: int = 0
    is_healthy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        data["capacity_level"] = self.capacity_level.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerResourceProfile:
        data_copy = dict(data)
        if "role" in data_copy and isinstance(data_copy["role"], str):
            data_copy["role"] = WorkerRole(data_copy["role"])
        if "capacity_level" in data_copy and isinstance(data_copy["capacity_level"], str):
            data_copy["capacity_level"] = ResourceCapacityLevel(data_copy["capacity_level"])
        return cls(**data_copy)


@dataclass
class TaskResourceRequirements:
    """Resource demands declared by a task."""
    minimum_capacity_level: ResourceCapacityLevel = ResourceCapacityLevel.LOW_RESOURCE
    minimum_memory_mb: int = 256
    context_tokens_required: int = 128
    priority: int = 100  # Lower number = higher priority


class ResourceAwareWorkerSelector:
    """
    Deterministic placement engine selecting optimal workers based on
    task resource demands, worker health, load, and context capacity.
    """

    CAPACITY_ORDER = {
        ResourceCapacityLevel.LOW_RESOURCE: 1,
        ResourceCapacityLevel.STANDARD: 2,
        ResourceCapacityLevel.HIGH_RESOURCE: 3,
    }

    @classmethod
    def score_worker_eligibility(
        cls,
        worker: WorkerResourceProfile,
        contract: WorkerContract,
        requirements: TaskResourceRequirements,
    ) -> Tuple[bool, float, str]:
        """
        Calculates whether a worker is eligible, returns (eligible, score, reason).
        Score is higher for lower current load and optimal capacity match.
        """
        # Health check
        if not worker.is_healthy:
            return False, 0.0, "Worker is marked unhealthy"

        # Role capability match
        if worker.role != contract.role:
            return False, 0.0, f"Role mismatch: worker is {worker.role.value}, requires {contract.role.value}"

        # Concurrency limit check
        if worker.current_active_tasks >= worker.max_concurrency:
            return False, 0.0, f"Worker overloaded: {worker.current_active_tasks} active tasks >= limit {worker.max_concurrency}"

        # Capacity level check
        worker_cap_val = cls.CAPACITY_ORDER[worker.capacity_level]
        req_cap_val = cls.CAPACITY_ORDER[requirements.minimum_capacity_level]
        if worker_cap_val < req_cap_val:
            return False, 0.0, f"Insufficient capacity level: {worker.capacity_level.value} < {requirements.minimum_capacity_level.value}"

        # Memory check
        if worker.memory_budget_mb < requirements.minimum_memory_mb:
            return False, 0.0, f"Insufficient memory budget: {worker.memory_budget_mb}MB < {requirements.minimum_memory_mb}MB"

        # Context budget check
        if worker.max_context_tokens < contract.context_budget:
            return False, 0.0, f"Context capacity {worker.max_context_tokens} < contract budget {contract.context_budget}"

        # Score computation:
        # Base capacity match (prefer closest match to avoid wasting high-resource workers)
        capacity_delta = worker_cap_val - req_cap_val
        # Load penalty: each active task decreases score
        load_factor = (worker.max_concurrency - worker.current_active_tasks) / float(worker.max_concurrency)
        score = 100.0 + (load_factor * 50.0) - (capacity_delta * 10.0)
        return True, score, "Eligible"

    @classmethod
    def select_best_worker(
        cls,
        workers: List[WorkerResourceProfile],
        contract: WorkerContract,
        requirements: TaskResourceRequirements,
    ) -> Optional[WorkerResourceProfile]:
        """
        Selects the highest-scoring eligible worker deterministically.
        """
        eligible: List[Tuple[float, str, WorkerResourceProfile]] = []
        for w in workers:
            is_ok, score, reason = cls.score_worker_eligibility(w, contract, requirements)
            if is_ok:
                # Tie-breaker on worker_id string for determinism
                eligible.append((score, w.worker_id, w))

        if not eligible:
            return None

        # Sort descending by score, ascending by worker_id
        eligible.sort(key=lambda item: (-item[0], item[1]))
        return eligible[0][2]
