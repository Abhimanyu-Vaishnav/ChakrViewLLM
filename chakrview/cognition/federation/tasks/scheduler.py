"""
Deterministic Task Scheduler for Distributed Workload Orchestration (Step 38).

CRITICAL AXIOMS:
- DETERMINISTIC & REPRODUCIBLE: Scheduling decisions must be auditable and explainable.
- LOCAL_AUTHORITY > PEER_AUTHORITY: Remote claims are filtered by sovereign trust & policy.
- LOW-RESOURCE DESIGN: Low-overhead heuristic scoring suitable for constrained hardware.
- FAIL-CLOSED: If no node satisfies trust, capability, and resource constraints,
  NoEligibleWorkerError is raised immediately.
"""

import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.peering.models import (
    TrustLevel,
    FederationScope,
)
from chakrview.cognition.federation.discovery.models import (
    MembershipState,
)
from chakrview.cognition.federation.resources.registry import (
    FederationResourceRegistry,
)
from chakrview.cognition.federation.resources.models import (
    AdvertisementFreshness,
)
from chakrview.cognition.federation.tasks.models import (
    WorkUnit,
    SchedulingDecision,
    ResourceRequirements,
)
from chakrview.cognition.federation.tasks.errors import (
    NoEligibleWorkerError,
    TenantTaskIsolationError,
)


class DeterministicTaskScheduler:
    """
    Deterministic scheduling engine matching WorkUnits to candidate federated nodes.
    """

    def __init__(
        self,
        local_node_id: str,
        resource_registry: FederationResourceRegistry,
        membership_manager: Optional[Any] = None,
        peering_engine: Optional[Any] = None,
        local_tenant_id: str = "default",
    ) -> None:
        self.local_node_id = local_node_id
        self.resource_registry = resource_registry
        self.membership_manager = membership_manager
        self.peering_engine = peering_engine
        self.local_tenant_id = local_tenant_id
        self._active_node_workload: Dict[str, int] = {}

    def record_assignment(self, node_id: str) -> None:
        """Track active unit assignments to balance load across nodes."""
        self._active_node_workload[node_id] = self._active_node_workload.get(node_id, 0) + 1

    def release_assignment(self, node_id: str) -> None:
        """Decrement active unit workload when a unit completes or fails."""
        if node_id in self._active_node_workload:
            self._active_node_workload[node_id] = max(0, self._active_node_workload[node_id] - 1)

    def schedule_unit(
        self,
        unit: WorkUnit,
        excluded_nodes: Optional[Set[str]] = None,
    ) -> SchedulingDecision:
        """
        Deterministically select the best candidate node for a work unit.
        
        Args:
            unit: The WorkUnit to be scheduled.
            excluded_nodes: Nodes to avoid (e.g. failed workers on previous attempts).
            
        Returns:
            An auditable SchedulingDecision.
            
        Raises:
            NoEligibleWorkerError if no candidate node qualifies.
            TenantTaskIsolationError if tenant mismatch occurs.
        """
        excluded = excluded_nodes or set()
        req = unit.requirements

        # Enforce tenant isolation
        if req.tenant_id != self.local_tenant_id and self.local_tenant_id != "*":
            raise TenantTaskIsolationError(
                f"WorkUnit tenant {req.tenant_id} does not match scheduler tenant {self.local_tenant_id}"
            )

        candidate_scores: Dict[str, Tuple[float, List[str]]] = {}

        # 1. Evaluate Local Node
        local_profile = self.resource_registry.get_local_profile()
        if self.local_node_id not in excluded:
            eligible, reason_codes, score = self._evaluate_node(
                node_id=self.local_node_id,
                profile=local_profile,
                req=req,
                is_local=True,
            )
            if eligible:
                candidate_scores[self.local_node_id] = (score, reason_codes)

        # 2. Evaluate Remote Federated Peers
        peer_advertisements = self.resource_registry.list_peer_advertisements(include_stale=False)
        for adv in peer_advertisements:
            node_id = adv.node_id
            if node_id in excluded:
                continue

            # Freshness check
            freshness = adv.evaluate_freshness()
            if freshness in (AdvertisementFreshness.STALE, AdvertisementFreshness.EXPIRED, AdvertisementFreshness.UNAVAILABLE):
                continue

            # Membership and Trust check
            if not self._check_membership_and_trust(node_id):
                continue

            eligible, reason_codes, score = self._evaluate_node(
                node_id=node_id,
                profile=adv.resource_profile,
                req=req,
                is_local=False,
                advertised_caps=adv.capabilities,
            )
            if eligible:
                candidate_scores[node_id] = (score, reason_codes)

        if not candidate_scores:
            raise NoEligibleWorkerError(
                f"No eligible worker found for unit {unit.unit_id} (capability: {req.capability_id}, "
                f"min_cores: {req.min_cpu_cores}, min_mem: {req.min_memory_mb} MB, "
                f"excluded: {sorted(list(excluded))})"
            )

        # 3. Deterministic Selection
        # Sort key: (-score, node_id) -> Highest score first, tie-break by node_id ascending
        sorted_candidates = sorted(
            candidate_scores.keys(),
            key=lambda nid: (-candidate_scores[nid][0], nid)
        )
        selected_node = sorted_candidates[0]
        selected_score, selected_reasons = candidate_scores[selected_node]

        decision_id = f"sched_{uuid.uuid4().hex[:12]}"
        decision = SchedulingDecision(
            decision_id=decision_id,
            task_id=unit.task_id,
            unit_id=unit.unit_id,
            candidate_nodes=sorted_candidates,
            selected_node=selected_node,
            reason_codes=selected_reasons,
            policy_version=1,
            resource_snapshot_reference=local_profile.profile_id if local_profile else "",
        )

        self.record_assignment(selected_node)
        return decision

    def _evaluate_node(
        self,
        node_id: str,
        profile: Any,
        req: ResourceRequirements,
        is_local: bool,
        advertised_caps: Optional[List[Any]] = None,
    ) -> Tuple[bool, List[str], float]:
        """
        Evaluate a single node's suitability and compute a deterministic score.
        Returns (is_eligible, reason_codes, score).
        """
        if profile is None:
            return False, ["PROFILE_MISSING"], 0.0

        reasons = []

        # Capability check
        if is_local:
            if req.capability_id != "*":
                has_cap = False
                if self.peering_engine and hasattr(self.peering_engine, "capability_registry"):
                    has_cap = self.peering_engine.capability_registry.has_capability(req.capability_id)
                elif self.resource_registry:
                    local_caps = self.resource_registry.get_local_capabilities(exposed_only=False)
                    has_cap = any(getattr(c, "capability_id", None) == req.capability_id for c in local_caps)
                else:
                    has_cap = True
                if not has_cap:
                    return False, ["CAPABILITY_UNSUPPORTED"], 0.0
        else:
            # Check advertised capabilities
            has_cap = False
            if req.capability_id == "*":
                has_cap = True
            elif advertised_caps:
                for cap in advertised_caps:
                    if getattr(cap, "capability_id", None) == req.capability_id:
                        has_cap = True
                        break
            if not has_cap:
                return False, ["CAPABILITY_UNSUPPORTED"], 0.0

        reasons.append("CAPABILITY_MATCH")

        # CPU core check
        avail_cores = profile.cpu.available_cores
        if avail_cores < req.min_cpu_cores:
            return False, [f"INSUFFICIENT_CORES ({avail_cores} < {req.min_cpu_cores})"], 0.0
        reasons.append(f"CORES_OK ({avail_cores:.1f})")

        # Memory check
        avail_mem = profile.memory.available_memory_mb
        if avail_mem < req.min_memory_mb:
            return False, [f"INSUFFICIENT_MEMORY ({avail_mem} < {req.min_memory_mb} MB)"], 0.0
        reasons.append(f"MEMORY_OK ({avail_mem} MB)")

        # Accelerator check
        if req.requires_accelerator:
            if not profile.accelerator or not profile.accelerator.is_available:
                return False, ["ACCELERATOR_REQUIRED_BUT_UNAVAILABLE"], 0.0
            reasons.append("ACCELERATOR_OK")

        # Score computation (Deterministic heuristic)
        # Base score from resources
        score = 0.0
        score += avail_cores * 2.0
        score += (avail_mem / 1024.0)

        # Workload balancing penalty
        current_load = self._active_node_workload.get(node_id, 0)
        score -= (current_load * 5.0)

        # Locality bonus: Local node preferred to avoid network serialization overhead
        if is_local:
            score += 50.0
            reasons.append("LOCALITY_BONUS")

        return True, reasons, score

    def _check_membership_and_trust(self, node_id: str) -> bool:
        """Verify peer has ACTIVE membership and FEDERATED trust delegation scope."""
        # 1. Check membership status
        if self.membership_manager:
            membership = self.membership_manager.get_membership(node_id)
            if not membership or membership.state != MembershipState.ACTIVE:
                return False

        # 2. Check peering trust and delegation scope
        if self.peering_engine and hasattr(self.peering_engine, "trust_store"):
            trust_store = self.peering_engine.trust_store
            grant = trust_store.get_grant(node_id) if hasattr(trust_store, "get_grant") else None
            if not grant:
                return False
            if grant.trust_level != TrustLevel.FEDERATED:
                return False
            # Check for ALLOW_COGNITIVE_TASK_DELEGATION scope
            if FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION not in grant.scopes:
                return False

        return True
