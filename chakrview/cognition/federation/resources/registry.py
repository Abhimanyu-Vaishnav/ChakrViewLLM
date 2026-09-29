"""
Sovereign Federation Resource and Capability Registry (Step 37).

Maintains a strict boundary between:
1. LOCAL_RESOURCE_STATE: Local hardware profile and ground truth capabilities.
2. PEER_RESOURCE_ADVERTISEMENTS: Untrusted claims voluntarily submitted by remote peers.

Enforces:
- PEER_RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
- RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION
- RESOURCE_ADVERTISEMENT != TRUST
- RESOURCE_DISCOVERY != RESOURCE_AUTHORIZATION
- LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
- Zero ambient execution rights
"""

import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.peering.crypto import Ed25519PublicKeyWrapper
from chakrview.cognition.federation.resources.models import (
    NodeResourceProfile,
    AdvertisedCapability,
    ResourceAdvertisement,
    AdvertisementFreshness,
    ResourceSharingPolicy,
    ExecutionType,
)
from chakrview.cognition.federation.resources.errors import (
    AdvertisementTamperedError,
    AdvertisementReplayError,
    StaleAdvertisementError,
    TenantResourceIsolationError,
    UnauthorizedResourceAccessError,
    InvalidAdvertisementError,
)


class FederationResourceRegistry:
    """
    Thread-safe registry for local resource ground truth and remote peer claims.
    """

    def __init__(
        self,
        local_node_id: str,
        local_zone_id: str,
        local_tenant_id: str = "default",
        sharing_policy: Optional[ResourceSharingPolicy] = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.local_zone_id = local_zone_id
        self.local_tenant_id = local_tenant_id
        self.sharing_policy = sharing_policy or ResourceSharingPolicy()

        self._lock = threading.RLock()

        # Local Ground Truth
        self._local_profile: Optional[NodeResourceProfile] = None
        self._local_capabilities: Dict[str, AdvertisedCapability] = {}

        # Remote Peer Advertisements (node_id -> ResourceAdvertisement)
        self._peer_advertisements: Dict[str, ResourceAdvertisement] = {}
        # Tracking status (node_id -> AdvertisementFreshness)
        self._peer_freshness: Dict[str, AdvertisementFreshness] = {}
        # Known version floor per node to prevent replay attacks
        self._peer_version_floors: Dict[str, int] = {}
        # Revoked or quarantined nodes (permanently barred)
        self._barred_nodes: Set[str] = set()

    # ========================================================================
    # 1. Local Resource Ground Truth Management
    # ========================================================================

    def register_local_profile(
        self,
        profile: NodeResourceProfile,
        capabilities: List[AdvertisedCapability],
        policy: Optional[ResourceSharingPolicy] = None,
    ) -> None:
        """Register or update ground truth local hardware profile and capabilities."""
        with self._lock:
            self._local_profile = profile
            self._local_capabilities = {c.capability_id: c for c in capabilities}
            if policy is not None:
                self.sharing_policy = policy

    def get_local_profile(self) -> Optional[NodeResourceProfile]:
        """Retrieve local ground truth hardware profile."""
        with self._lock:
            return self._local_profile

    def get_local_capabilities(self, exposed_only: bool = True) -> List[AdvertisedCapability]:
        """Retrieve local computational capabilities."""
        with self._lock:
            if exposed_only:
                return [c for c in self._local_capabilities.values() if c.is_exposed]
            return list(self._local_capabilities.values())

    def update_sharing_policy(self, policy: ResourceSharingPolicy) -> None:
        """Update local resource sharing policy."""
        with self._lock:
            self.sharing_policy = policy

    def get_local_sharing_policy(self) -> ResourceSharingPolicy:
        """Retrieve current local resource sharing policy."""
        with self._lock:
            return self.sharing_policy

    # ========================================================================
    # 2. Remote Peer Advertisement Ingestion & Validation
    # ========================================================================

    def record_peer_advertisement(
        self,
        advertisement: ResourceAdvertisement,
        public_key: Optional[Ed25519PublicKeyWrapper] = None,
        is_peer_revoked: bool = False,
        is_peer_quarantined: bool = False,
    ) -> None:
        """
        Record and validate a remote peer's voluntary resource advertisement.
        Fails closed on signature mismatch, digest tampering, replay, or revoked sender.
        """
        with self._lock:
            node_id = advertisement.node_id

            # 1. Identity & Authority Checks
            if node_id == self.local_node_id:
                raise InvalidAdvertisementError("Cannot record local node as remote peer advertisement.")

            if node_id in self._barred_nodes or is_peer_revoked:
                self._barred_nodes.add(node_id)
                self._peer_advertisements.pop(node_id, None)
                self._peer_freshness[node_id] = AdvertisementFreshness.UNAVAILABLE
                raise UnauthorizedResourceAccessError(f"Peer '{node_id}' is REVOKED; resource claims barred.")

            if is_peer_quarantined:
                raise UnauthorizedResourceAccessError(f"Peer '{node_id}' is QUARANTINED; resource claims rejected.")

            # 2. Multi-Tenant Isolation Check
            if self.local_tenant_id != "*" and advertisement.tenant_id != "*":
                if self.local_tenant_id != advertisement.tenant_id:
                    raise TenantResourceIsolationError(
                        f"Cross-tenant resource advertisement rejected: local tenant '{self.local_tenant_id}' "
                        f"!= remote tenant '{advertisement.tenant_id}'."
                    )

            # 3. Cryptographic Integrity Verification
            if not advertisement.verify_integrity():
                raise AdvertisementTamperedError(
                    f"Advertisement '{advertisement.advertisement_id}' failed payload digest verification."
                )

            # 4. Cryptographic Signature Verification
            if public_key is not None:
                if not advertisement.verify_signature(public_key):
                    raise AdvertisementTamperedError(
                        f"Advertisement '{advertisement.advertisement_id}' failed Ed25519 signature verification."
                    )

            # 5. Monotonic Versioning & Replay Defense
            last_version = self._peer_version_floors.get(node_id, 0)
            if advertisement.version <= last_version:
                raise AdvertisementReplayError(
                    f"Replay detected for node '{node_id}': advertisement version {advertisement.version} "
                    f"<= floor {last_version}."
                )

            # 6. Freshness Check
            freshness = advertisement.evaluate_freshness()
            if freshness in (AdvertisementFreshness.STALE, AdvertisementFreshness.EXPIRED):
                raise StaleAdvertisementError(
                    f"Advertisement '{advertisement.advertisement_id}' from '{node_id}' is already {freshness.value}."
                )

            # Update state
            self._peer_version_floors[node_id] = advertisement.version
            self._peer_advertisements[node_id] = advertisement
            self._peer_freshness[node_id] = freshness

    def get_peer_advertisement(
        self,
        node_id: str,
        allow_stale: bool = False,
    ) -> Optional[ResourceAdvertisement]:
        """Retrieve peer advertisement by node_id if fresh or allow_stale is True."""
        with self._lock:
            adv = self._peer_advertisements.get(node_id)
            if not adv:
                return None
            if self._peer_freshness.get(node_id) == AdvertisementFreshness.UNAVAILABLE:
                freshness = AdvertisementFreshness.UNAVAILABLE
            else:
                freshness = adv.evaluate_freshness()
                self._peer_freshness[node_id] = freshness
            if not allow_stale and freshness in (AdvertisementFreshness.STALE, AdvertisementFreshness.EXPIRED, AdvertisementFreshness.UNAVAILABLE):
                return None
            return adv

    def list_peer_advertisements(
        self,
        include_stale: bool = False,
    ) -> List[ResourceAdvertisement]:
        """List all peer resource claims."""
        with self._lock:
            results = []
            for node_id, adv in self._peer_advertisements.items():
                if self._peer_freshness.get(node_id) == AdvertisementFreshness.UNAVAILABLE:
                    freshness = AdvertisementFreshness.UNAVAILABLE
                else:
                    freshness = adv.evaluate_freshness()
                    self._peer_freshness[node_id] = freshness
                if include_stale or freshness in (AdvertisementFreshness.FRESH, AdvertisementFreshness.AGING):
                    results.append(adv)
            return results

    # ========================================================================
    # 3. Capability Filtering (Claims Only — NOT Scheduling)
    # ========================================================================

    def filter_by_capability(
        self,
        execution_type: ExecutionType,
        min_cores: Optional[float] = None,
        min_memory_mb: Optional[int] = None,
        requires_accelerator: bool = False,
    ) -> List[str]:
        """
        Query which connected peers *claim* to support a given capability and capacity.
        
        CRITICAL ARCHITECTURAL BOUNDARY:
        This answers ONLY: 'Which nodes claim to support this capability?'
        This does NOT:
        - Schedule tasks
        - Lease resources
        - Grant execution permissions
        - Consume remote compute
        """
        with self._lock:
            matching_nodes: List[str] = []

            for node_id, adv in self._peer_advertisements.items():
                # Stale or unavailable claims are excluded
                freshness = adv.evaluate_freshness()
                if freshness not in (AdvertisementFreshness.FRESH, AdvertisementFreshness.AGING):
                    continue

                prof = adv.resource_profile

                # Hardware requirements check
                if min_cores is not None and prof.cpu.available_cores < min_cores:
                    continue
                if min_memory_mb is not None and prof.memory.available_memory_mb < min_memory_mb:
                    continue
                if requires_accelerator:
                    if not prof.accelerator or not prof.accelerator.is_available:
                        continue

                # Capability check
                has_capability = any(
                    c.execution_type == execution_type and c.is_exposed
                    for c in adv.capabilities
                )
                if has_capability:
                    matching_nodes.append(node_id)

            return matching_nodes

    # ========================================================================
    # 4. Freshness, Invalidation & Pruning
    # ========================================================================

    def check_freshness(self, current_time: Optional[float] = None) -> Dict[str, AdvertisementFreshness]:
        """Evaluate freshness status across all registered peer claims."""
        with self._lock:
            statuses: Dict[str, AdvertisementFreshness] = {}
            for node_id, adv in self._peer_advertisements.items():
                if self._peer_freshness.get(node_id) == AdvertisementFreshness.UNAVAILABLE:
                    statuses[node_id] = AdvertisementFreshness.UNAVAILABLE
                else:
                    freshness = adv.evaluate_freshness(current_time)
                    self._peer_freshness[node_id] = freshness
                    statuses[node_id] = freshness
            return statuses

    def invalidate_peer(self, node_id: str, reason: str = "disconnected") -> None:
        """Mark a peer's resource advertisement as UNAVAILABLE."""
        with self._lock:
            if node_id in self._peer_freshness:
                self._peer_freshness[node_id] = AdvertisementFreshness.UNAVAILABLE

    def bar_peer(self, node_id: str) -> None:
        """Permanently bar a revoked peer from submitting or retaining resource claims."""
        with self._lock:
            self._barred_nodes.add(node_id)
            self._peer_advertisements.pop(node_id, None)
            self._peer_freshness[node_id] = AdvertisementFreshness.UNAVAILABLE

    def purge_expired(self, current_time: Optional[float] = None) -> int:
        """Prune advertisements that have passed their expiration threshold."""
        with self._lock:
            expired_nodes = []
            for node_id, adv in self._peer_advertisements.items():
                freshness = adv.evaluate_freshness(current_time)
                if freshness == AdvertisementFreshness.EXPIRED:
                    expired_nodes.append(node_id)

            for node_id in expired_nodes:
                del self._peer_advertisements[node_id]
                self._peer_freshness[node_id] = AdvertisementFreshness.EXPIRED

            return len(expired_nodes)
