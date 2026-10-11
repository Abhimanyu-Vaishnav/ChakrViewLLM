"""
Sovereign Resource Execution Grant Manager for ChakrView (Step 38).

CRITICAL AXIOMS:
- LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
- ADVERTISEMENT != PERMISSION
- ADVERTISEMENT != EXECUTION_AUTHORITY
- RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
- DEFAULT-DENY: Without an explicit active grant, remote execution is rejected.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.federation.tasks.models import (
    ResourceExecutionGrant,
    GrantType,
    ResourceRequirements,
)
from chakrview.cognition.federation.tasks.errors import (
    UnauthorizedTaskExecutionError,
    ExecutionGrantViolationError,
)


class ExecutionGrantManager:
    """
    Manages local device execution grants and verifies whether incoming
    task requests from federated peers are permitted to execute on this node.
    """

    def __init__(self, local_node_id: str, default_grant_type: GrantType = GrantType.TRUSTED_FEDERATION) -> None:
        self.local_node_id = local_node_id
        self._lock = threading.RLock()
        self._grants: Dict[str, ResourceExecutionGrant] = {}
        self._active_unit_counts_by_grant: Dict[str, int] = {}
        self._allocated_cores_by_grant: Dict[str, float] = {}
        self._allocated_memory_by_grant: Dict[str, int] = {}

        # Initialize default grant
        self.default_grant_id = f"grant_default_{local_node_id}"
        self.add_grant(
            ResourceExecutionGrant(
                grant_id=self.default_grant_id,
                grant_type=default_grant_type,
                owner_node_id=local_node_id,
                allowed_peer_nodes={"*"},
                allowed_tenants={"*"},
                allowed_capability_ids={"*"},
                max_concurrent_units=4,
                max_memory_mb=4096,
                max_cores=4.0,
            )
        )

    def add_grant(self, grant: ResourceExecutionGrant) -> None:
        """Register or update an execution grant."""
        with self._lock:
            self._grants[grant.grant_id] = grant
            if grant.grant_id not in self._active_unit_counts_by_grant:
                self._active_unit_counts_by_grant[grant.grant_id] = 0
                self._allocated_cores_by_grant[grant.grant_id] = 0.0
                self._allocated_memory_by_grant[grant.grant_id] = 0

    def remove_grant(self, grant_id: str) -> None:
        """Revoke an execution grant immediately."""
        with self._lock:
            self._grants.pop(grant_id, None)
            self._active_unit_counts_by_grant.pop(grant_id, None)
            self._allocated_cores_by_grant.pop(grant_id, None)
            self._allocated_memory_by_grant.pop(grant_id, None)

    def get_grant(self, grant_id: str) -> Optional[ResourceExecutionGrant]:
        """Look up grant by ID."""
        with self._lock:
            return self._grants.get(grant_id)

    def list_grants(self) -> List[ResourceExecutionGrant]:
        """List all active execution grants."""
        with self._lock:
            return list(self._grants.values())

    def authorize_execution(
        self,
        peer_node_id: str,
        tenant_id: str,
        capability_id: str,
        requirements: ResourceRequirements,
    ) -> ResourceExecutionGrant:
        """
        Evaluate all active grants to authorize an execution request.
        Returns the matching ResourceExecutionGrant if permitted.
        Raises UnauthorizedTaskExecutionError or ExecutionGrantViolationError if denied.
        """
        with self._lock:
            # If no grants exist, reject immediately
            if not self._grants:
                raise UnauthorizedTaskExecutionError(
                    f"No execution grants configured on node {self.local_node_id}. Remote execution denied."
                )

            rejection_reasons = []

            for grant in self._grants.values():
                authorized, reason = grant.is_authorized(
                    peer_node_id=peer_node_id,
                    tenant_id=tenant_id,
                    capability_id=capability_id,
                    req=requirements,
                )
                if not authorized:
                    rejection_reasons.append(f"Grant {grant.grant_id}: {reason}")
                    continue

                # Check runtime quota allocations
                current_active = self._active_unit_counts_by_grant.get(grant.grant_id, 0)
                if current_active >= grant.max_concurrent_units:
                    rejection_reasons.append(
                        f"Grant {grant.grant_id}: max concurrent units reached ({current_active}/{grant.max_concurrent_units})"
                    )
                    continue

                current_cores = self._allocated_cores_by_grant.get(grant.grant_id, 0.0)
                if current_cores + requirements.min_cpu_cores > grant.max_cores:
                    rejection_reasons.append(
                        f"Grant {grant.grant_id}: CPU core limit would be exceeded ({current_cores + requirements.min_cpu_cores} > {grant.max_cores})"
                    )
                    continue

                current_mem = self._allocated_memory_by_grant.get(grant.grant_id, 0)
                if current_mem + requirements.min_memory_mb > grant.max_memory_mb:
                    rejection_reasons.append(
                        f"Grant {grant.grant_id}: Memory limit would be exceeded ({current_mem + requirements.min_memory_mb} > {grant.max_memory_mb} MB)"
                    )
                    continue

                # Match found and capacity verified
                return grant

            # If we reach here, all grants rejected or had insufficient quota
            raise ExecutionGrantViolationError(
                f"Execution request from peer {peer_node_id} for capability {capability_id} denied. "
                f"Reasons: {'; '.join(rejection_reasons)}"
            )

    def allocate_resources(self, grant_id: str, requirements: ResourceRequirements) -> None:
        """Reserve resources under a grant for an executing unit."""
        with self._lock:
            if grant_id in self._grants:
                self._active_unit_counts_by_grant[grant_id] = self._active_unit_counts_by_grant.get(grant_id, 0) + 1
                self._allocated_cores_by_grant[grant_id] = self._allocated_cores_by_grant.get(grant_id, 0.0) + requirements.min_cpu_cores
                self._allocated_memory_by_grant[grant_id] = self._allocated_memory_by_grant.get(grant_id, 0) + requirements.min_memory_mb

    def release_resources(self, grant_id: str, requirements: ResourceRequirements) -> None:
        """Release resources under a grant when a unit finishes, fails, or cancels."""
        with self._lock:
            if grant_id in self._grants:
                self._active_unit_counts_by_grant[grant_id] = max(0, self._active_unit_counts_by_grant.get(grant_id, 0) - 1)
                self._allocated_cores_by_grant[grant_id] = max(0.0, self._allocated_cores_by_grant.get(grant_id, 0.0) - requirements.min_cpu_cores)
                self._allocated_memory_by_grant[grant_id] = max(0, self._allocated_memory_by_grant.get(grant_id, 0) - requirements.min_memory_mb)
