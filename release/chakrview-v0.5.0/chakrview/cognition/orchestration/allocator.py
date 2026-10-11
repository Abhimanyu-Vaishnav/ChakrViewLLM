"""
Resource-Aware Allocator for Step 28 Cognitive Orchestration.

Maps TaskPlans, hardware resource profiles, and node registry telemetry
into deterministic ResourceAllocationDecisions.

CRITICAL INVARIANTS:
1. DETERMINISTIC: Identical cluster state, plan, and hardware profile yield identical allocation.
2. HARD CEILINGS: Never exceeds MAX_ORCHESTRATION_AGENTS (8), MAX_ORCHESTRATION_NODES (8),
   or MAX_DELIBERATION_ROUNDS (3).
3. TENANT & HEALTH ENFORCEMENT: Never allocates quarantined, revoked, or cross-tenant nodes.
"""

from typing import Dict, List, Optional, Any, Sequence
import uuid

from chakrview.cognition.orchestration.models import (
    TaskPlan,
    ResourceAllocationDecision,
    WorkloadClass,
    MAX_ORCHESTRATION_AGENTS,
    MAX_ORCHESTRATION_NODES,
    MAX_DELIBERATION_ROUNDS,
    MAX_ORCHESTRATION_RETRIES,
)
from chakrview.cognition.federated.models import AgentRole
from chakrview.cognition.distributed.models import (
    NodeRegistration,
    NodeStatus,
    NodeTrustState,
)
from chakrview.cognition.distributed.registry import DistributedNodeRegistry
from chakrview.cognition.adaptation.profiles import ResourceProfile


class ResourceAwareAllocator:
    """
    Computes deterministic ResourceAllocationDecisions tailored to hardware capacity
    and live node health.
    """

    def __init__(self, node_registry: Optional[DistributedNodeRegistry] = None) -> None:
        self.node_registry = node_registry

    def allocate(
        self,
        plan: TaskPlan,
        resource_profile: ResourceProfile = ResourceProfile.STANDARD,
        tenant_id: str = "default_tenant",
        local_node_id: Optional[str] = None,
    ) -> ResourceAllocationDecision:
        """
        Derive optimal, bounded resource budgets and participating nodes.
        """
        decision_id = f"alloc_{uuid.uuid4().hex[:8]}"

        # 1. Base capacities by hardware profile
        if resource_profile == ResourceProfile.LOW_RESOURCE:
            profile_agent_cap = 2
            profile_node_cap = 1
            budget_multiplier = 0.6
            retry_cap = 1
        elif resource_profile == ResourceProfile.HIGH_RESOURCE:
            profile_agent_cap = 8
            profile_node_cap = 4
            budget_multiplier = 1.4
            retry_cap = 2
        else:  # STANDARD
            profile_agent_cap = 4
            profile_node_cap = 2
            budget_multiplier = 1.0
            retry_cap = 2

        # 2. Determine agent allocation
        target_agent_count = min(
            plan.max_agent_count,
            profile_agent_cap,
            MAX_ORCHESTRATION_AGENTS,
        )
        # Ensure at least 1 agent is allocated
        target_agent_count = max(1, target_agent_count)

        # 3. Determine node allocation & select nodes deterministically
        target_node_count = min(
            plan.max_node_count,
            profile_node_cap,
            MAX_ORCHESTRATION_NODES,
        )
        target_node_count = max(1, target_node_count)

        selected_node_ids = self._select_nodes_deterministically(
            target_node_count=target_node_count,
            tenant_id=tenant_id,
            local_node_id=local_node_id,
        )

        # 4. Role distribution
        role_distribution: Dict[str, int] = {}
        # Allocate required roles first up to target_agent_count
        assigned_count = 0
        for r in plan.required_roles:
            if assigned_count < target_agent_count:
                role_distribution[r.value] = role_distribution.get(r.value, 0) + 1
                assigned_count += 1

        # If capacity remains, allocate optional roles
        for r in plan.optional_roles:
            if assigned_count < target_agent_count:
                role_distribution[r.value] = role_distribution.get(r.value, 0) + 1
                assigned_count += 1

        # If still room (e.g. HIGH_RESOURCE complex task), duplicate primary analyst/researcher
        if assigned_count < target_agent_count and AgentRole.ANALYST.value in role_distribution:
            role_distribution[AgentRole.ANALYST.value] += (target_agent_count - assigned_count)

        # 5. Budgets
        execution_budget_ms = int(plan.execution_budget_ms * budget_multiplier)
        deliberation_budget = min(plan.max_deliberation_rounds, MAX_DELIBERATION_ROUNDS)
        if resource_profile == ResourceProfile.LOW_RESOURCE:
            deliberation_budget = min(deliberation_budget, 1)

        verification_budget = 1 if plan.verification_required else 0
        retry_budget = min(retry_cap, MAX_ORCHESTRATION_RETRIES)

        explanation = (
            f"Allocated {sum(role_distribution.values())} agents across {len(selected_node_ids)} nodes "
            f"for workload {plan.workload_class.value} under {resource_profile.value} profile."
        )

        return ResourceAllocationDecision(
            decision_id=decision_id,
            task_id=plan.task_id,
            workload_class=plan.workload_class,
            resource_profile=resource_profile,
            allocated_agent_count=sum(role_distribution.values()),
            allocated_node_count=len(selected_node_ids),
            role_distribution=role_distribution,
            selected_node_ids=selected_node_ids,
            max_rounds=deliberation_budget,
            execution_budget_ms=execution_budget_ms,
            retry_budget=retry_budget,
            verification_budget=verification_budget,
            deliberation_budget=deliberation_budget,
            explanation=explanation,
        )

    def _select_nodes_deterministically(
        self,
        target_node_count: int,
        tenant_id: str,
        local_node_id: Optional[str],
    ) -> List[str]:
        """
        Select nodes deterministically based on health, locality, and active load.
        """
        if self.node_registry is None:
            # Standalone fallback: return local or default node ID
            return [local_node_id or "node_local_0"][:target_node_count]

        # Discover registered nodes for the tenant
        eligible_nodes: List[NodeRegistration] = []
        nodes_dict = getattr(self.node_registry, "_nodes", {}) or getattr(self.node_registry, "nodes", {})
        for reg in nodes_dict.values():
            if reg.identity.tenant_id != tenant_id:
                continue
            if reg.trust_state == NodeTrustState.REVOKED:
                continue
            if reg.health.status in (NodeStatus.QUARANTINED, NodeStatus.UNAVAILABLE, NodeStatus.REVOKED):
                continue
            eligible_nodes.append(reg)

        if not eligible_nodes:
            return [local_node_id or "node_local_0"][:target_node_count]

        # Deterministic ranking:
        # 1. Prefer local node (locality optimization)
        # 2. Prefer HEALTHY over DEGRADED
        # 3. Lowest error rate / failures
        # 4. Lowest average latency
        # 5. Lexicographical node_id for stable tie-breaking
        def sort_key(reg: NodeRegistration):
            is_local = 0 if (local_node_id and reg.identity.node_id == local_node_id) else 1
            health_order = 0 if reg.health.status == NodeStatus.HEALTHY else 1
            failures = reg.health.consecutive_failures
            latency = reg.health.avg_latency_ms
            return (is_local, health_order, failures, latency, reg.identity.node_id)

        sorted_nodes = sorted(eligible_nodes, key=sort_key)
        return [node.identity.node_id for node in sorted_nodes[:target_node_count]]
