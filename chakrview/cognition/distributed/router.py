"""
Distributed Task Router (Step 27).

Deterministically routes federated subtasks across available nodes in a cluster:
- Inspects required agent roles and tenant boundaries.
- Filters out unhealthy, quarantined, revoked, or circuit-broken nodes.
- Selects target nodes based on deterministic technical policies (locality, health, tie-breaking).
- No random selection; guarantees reproducible task allocation.

CRITICAL ARCHITECTURAL AXIOMS:
1. DETERMINISTIC TECHNICAL ROUTING:
   Task routing is strictly deterministic based on resource profiles and availability.
   No randomized or political node ranking.
2. STRICT TENANT & CIRCUIT BOUNDARIES:
   Nodes from foreign tenants or with open circuit breakers are never selected.
"""

from typing import Dict, List, Optional
import time

from chakrview.cognition.distributed.models import (
    DistributedRouteDecision,
    NodeRegistration,
    NodeRole,
)
from chakrview.cognition.distributed.registry import DistributedNodeRegistry
from chakrview.cognition.distributed.resilience import CircuitBreaker
from chakrview.cognition.federated.models import AgentTask, AgentRole


class RoutingError(Exception):
    """Base exception for task routing failure."""
    pass


class NoEligibleNodeError(RoutingError):
    """Raised when no healthy node supports the required task role."""
    pass


class DistributedTaskRouter:
    """
    Deterministic task allocation engine mapping AgentTasks to cluster nodes.
    """

    def __init__(
        self,
        registry: DistributedNodeRegistry,
        local_node_id: str,
        circuit_breakers: Optional[Dict[str, CircuitBreaker]] = None,
    ) -> None:
        self.registry = registry
        self.local_node_id = local_node_id
        self._circuit_breakers = circuit_breakers if circuit_breakers is not None else {}

    def get_or_create_circuit_breaker(self, node_id: str) -> CircuitBreaker:
        if node_id not in self._circuit_breakers:
            self._circuit_breakers[node_id] = CircuitBreaker(node_id=node_id)
        return self._circuit_breakers[node_id]

    def route_task(
        self,
        task: AgentTask,
        tenant_id: str,
        session_id: str,
        prefer_remote: bool = False,
    ) -> DistributedRouteDecision:
        """
        Deterministically selects the most appropriate node for an AgentTask.
        """
        # 1. Discover all candidate nodes supporting the role in this tenant
        candidates = self.registry.discover_nodes(
            tenant_id=tenant_id,
            agent_role=task.assigned_role,
            only_eligible=True,
        )

        # 2. Filter through circuit breakers
        available_candidates: List[NodeRegistration] = []
        for cand in candidates:
            cb = self.get_or_create_circuit_breaker(cand.identity.node_id)
            if cb.can_execute():
                available_candidates.append(cand)

        if not available_candidates:
            raise NoEligibleNodeError(
                f"No eligible node found for role '{task.assigned_role.value}' in tenant '{tenant_id}'."
            )

        # 3. Deterministic candidate ranking:
        # Priority order:
        # a) Preference match (prefer_remote vs local)
        # b) Error rate (ascending)
        # c) Latency tier (LOCAL < LAN < WAN)
        # d) Consecutive failures (ascending)
        # e) Alphabetical node_id (deterministic tie-break)
        def sort_key(cand: NodeRegistration):
            is_local = (cand.identity.node_id == self.local_node_id)
            pref_penalty = 1 if (prefer_remote and is_local) else 0
            tier_rank = {"LOCAL": 0, "LAN": 1, "WAN": 2}.get(cand.resource_profile.latency_tier, 3)
            return (
                pref_penalty,
                cand.health.error_rate,
                tier_rank,
                cand.health.consecutive_failures,
                cand.identity.node_id,
            )

        available_candidates.sort(key=sort_key)
        chosen_node = available_candidates[0]

        is_remote = (chosen_node.identity.node_id != self.local_node_id)
        assigned_agent_id = f"agent:{task.assigned_role.value.lower()}:{tenant_id}"
        est_latency = chosen_node.health.avg_latency_ms if is_remote else 0.5

        return DistributedRouteDecision(
            task_id=task.task_id,
            assigned_node_id=chosen_node.identity.node_id,
            assigned_agent_id=assigned_agent_id,
            assigned_role=task.assigned_role,
            is_remote=is_remote,
            estimated_latency_ms=est_latency,
            reason=f"Routed to node '{chosen_node.identity.node_id}' with latency tier '{chosen_node.resource_profile.latency_tier}'.",
        )
