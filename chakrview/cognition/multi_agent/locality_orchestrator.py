"""
ChakrView Step 133: Resource-Aware Cognitive Orchestration & Locality Placement.

Advanced placement policies considering:
- Hardware capacity tiers (LOW, STANDARD, HIGH)
- Data locality (whether worker has cached AST symbols/project indices)
- Historical reliability weighting
- Avoids over-allocation and preserves CPU-first constraints
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.multi_agent.contracts import WorkerRole, WorkerContract
from chakrview.cognition.multi_agent.resource_federation import (
    ResourceCapacityLevel,
    WorkerResourceProfile,
    TaskResourceRequirements,
)


@dataclass
class CognitivePlacementContext:
    target_files: List[str] = field(default_factory=list)
    cached_node_ids: Set[str] = field(default_factory=set)
    estimated_tokens: int = 100
    is_critical_path: bool = False


class LocalityAwareOrchestrator:
    """
    Computes placement scores combining resource capacity, data locality,
    and historical reliability.
    """

    @classmethod
    def calculate_placement_score(
        cls,
        worker: WorkerResourceProfile,
        contract: WorkerContract,
        requirements: TaskResourceRequirements,
        context: CognitivePlacementContext,
        reliability_multiplier: float = 1.0,
    ) -> Tuple[bool, float, str]:
        if not worker.is_healthy:
            return False, 0.0, "Worker unhealthy"

        if worker.role != contract.role:
            return False, 0.0, "Role mismatch"

        if worker.current_active_tasks >= worker.max_concurrency:
            return False, 0.0, "Concurrency saturated"

        cap_order = {
            ResourceCapacityLevel.LOW_RESOURCE: 1,
            ResourceCapacityLevel.STANDARD: 2,
            ResourceCapacityLevel.HIGH_RESOURCE: 3,
        }
        w_val = cap_order[worker.capacity_level]
        r_val = cap_order[requirements.minimum_capacity_level]
        if w_val < r_val:
            return False, 0.0, "Insufficient capacity"

        # Locality bonus: 30 points if worker node already hosts cached files
        locality_bonus = 30.0 if worker.worker_id in context.cached_node_ids else 0.0
        # Load penalty
        load_penalty = (worker.current_active_tasks / float(worker.max_concurrency)) * 40.0
        # Capacity waste penalty (prefer exact tier match)
        waste_penalty = (w_val - r_val) * 15.0

        base_score = 100.0 + locality_bonus - load_penalty - waste_penalty
        final_score = base_score * reliability_multiplier
        return True, max(1.0, final_score), "Eligible"

    @classmethod
    def choose_optimal_worker(
        cls,
        workers: List[WorkerResourceProfile],
        contract: WorkerContract,
        requirements: TaskResourceRequirements,
        context: CognitivePlacementContext,
        reliability_map: Optional[Dict[str, float]] = None,
    ) -> Optional[WorkerResourceProfile]:
        scored: List[Tuple[float, str, WorkerResourceProfile]] = []
        rel_map = reliability_map or {}

        for w in workers:
            rel = rel_map.get(w.worker_id, 1.0)
            ok, score, _ = cls.calculate_placement_score(w, contract, requirements, context, rel)
            if ok:
                scored.append((score, w.worker_id, w))

        if not scored:
            return None

        # Sort descending by score, tie-break by worker_id ascending
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]
