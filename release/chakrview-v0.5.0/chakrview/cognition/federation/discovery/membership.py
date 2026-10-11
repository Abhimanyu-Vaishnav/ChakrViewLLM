"""
Node Membership Manager for ChakrView Federation Runtime (Step 35).

Manages node candidate registration, authentication gates, member promotion,
suspension, quarantine, and revocation cascades.

CRITICAL ARCHITECTURAL AXIOMS:
- MEMBERSHIP != TRUST
- MEMBERSHIP != AUTHORIZATION
- CANDIDATE != MEMBER
- UNREACHABLE != REVOKED
- QUARANTINED != DELETED
- REVOKED -> Absorbing terminal state (cannot transition back to active)
- LOCAL_REVOCATION > REMOTE_ACTIVE_STATE
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from chakrview.cognition.peering.models import AuditEventType
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    MAX_FEDERATION_ENGINES,
    RevocationTargetType,
)
from chakrview.cognition.federation.discovery.models import (
    FederationNodeCandidate,
    FederationNodeMembership,
    MembershipState,
    MAX_MEMBERSHIP_NODES,
)
from chakrview.cognition.federation.discovery.errors import (
    MembershipError,
    MembershipCapacityError,
    MembershipStateTransitionError,
    QuarantineError,
    MembershipRevocationError,
    UnknownPeerError,
)
from chakrview.cognition.federation.persistence.models import JournalEntryType

logger = logging.getLogger(__name__)


class FederationMembershipManager:
    """
    Authoritative local manager for federation node membership states and lifecycle transitions.
    """

    def __init__(
        self,
        engine: Any,
        runtime: Optional[Any] = None,
        max_members: int = MAX_MEMBERSHIP_NODES,
    ) -> None:
        self.engine = engine
        self.runtime = runtime or getattr(engine, "runtime", None)
        self.max_members = max_members

        self._lock = threading.Lock()
        self._memberships: Dict[str, FederationNodeMembership] = {}
        self._node_to_membership: Dict[str, str] = {}
        self._endpoint_to_membership: Dict[str, str] = {}
        self._candidates: Dict[str, FederationNodeCandidate] = {}

    # ========================================================================
    # 1. Candidate Registration
    # ========================================================================

    def register_candidate(
        self,
        candidate: FederationNodeCandidate,
    ) -> FederationNodeMembership:
        """
        Record a newly discovered candidate node.
        Initial state is DISCOVERED (zero trust, zero execution permissions).
        """
        with self._lock:
            ep_id = candidate.endpoint.endpoint_id
            if ep_id in self._endpoint_to_membership:
                existing_id = self._endpoint_to_membership[ep_id]
                return self._memberships[existing_id]

            membership = FederationNodeMembership.create(
                candidate=candidate,
                enrolled_epoch=self.engine.current_epoch,
            )
            self._memberships[membership.membership_id] = membership
            self._endpoint_to_membership[ep_id] = membership.membership_id
            self._candidates[candidate.candidate_id] = candidate

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_DISCOVERED,
                epoch=self.engine.current_epoch,
                details={
                    "membership_id": membership.membership_id,
                    "candidate_id": candidate.candidate_id,
                    "endpoint": candidate.endpoint.to_transport_uri(),
                    "source": candidate.discovery_source.value,
                },
            )

        if self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.NODE_DISCOVERED,
                epoch=self.engine.current_epoch,
                payload={
                    "membership_id": membership.membership_id,
                    "candidate_id": candidate.candidate_id,
                    "endpoint": candidate.endpoint.to_dict(),
                },
            )

        return membership

    # ========================================================================
    # 2. Authentication Gate & Promotion
    # ========================================================================

    def start_authentication(self, membership_id: str) -> None:
        """Advance candidate from DISCOVERED to PENDING_AUTHENTICATION."""
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            membership.transition_to(MembershipState.PENDING_AUTHENTICATION, reason="Starting transport authentication")

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_AUTHENTICATION_STARTED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id},
            )

    def authenticate_candidate(
        self,
        membership_id: str,
        engine_identity: FederationEngineIdentity,
        peer_id: Optional[str] = None,
    ) -> None:
        """
        Mark node candidate as AUTHENTICATED following successful TLS/mTLS,
        peer certificate binding, and engine identity verification.
        """
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            if membership.state == MembershipState.DISCOVERED:
                membership.transition_to(MembershipState.PENDING_AUTHENTICATION, reason="Starting candidate authentication")
            membership.transition_to(MembershipState.AUTHENTICATED, reason="Identity and certificate validated")
            membership.engine_identity = engine_identity
            membership.node_id = engine_identity.engine_id
            membership.peer_id = peer_id
            self._node_to_membership[engine_identity.engine_id] = membership_id

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_AUTHENTICATED,
                epoch=self.engine.current_epoch,
                details={
                    "membership_id": membership_id,
                    "engine_id": engine_identity.engine_id,
                    "zone_id": engine_identity.zone_id,
                },
            )

        if self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.NODE_AUTHENTICATED,
                epoch=self.engine.current_epoch,
                payload={
                    "membership_id": membership_id,
                    "engine_id": engine_identity.engine_id,
                    "zone_id": engine_identity.zone_id,
                    "peer_id": peer_id,
                },
            )

    def promote_to_member(
        self,
        membership_id: str,
        reason: Optional[str] = "Handshake and digest validation completed",
    ) -> FederationNodeMembership:
        """
        Promote an AUTHENTICATED candidate to MEMBER standing.
        Enforces maximum capacity ceiling (MAX_MEMBERSHIP_NODES).
        """
        with self._lock:
            membership = self._get_membership_locked(membership_id)

            # Check capacity
            active_count = sum(1 for m in self._memberships.values() if m.is_active_member())
            if active_count >= self.max_members and not membership.is_active_member():
                raise MembershipCapacityError(
                    f"Cannot promote '{membership_id}': active member capacity ceiling ({self.max_members}) reached."
                )

            membership.transition_to(MembershipState.MEMBER, reason=reason)
            membership.record_heartbeat(self.engine.current_epoch)

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_MEMBERSHIP_GRANTED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id, "node_id": membership.node_id, "reason": reason},
            )

        if self.runtime:
            if hasattr(self.runtime, "set_engine_health") and membership.node_id:
                from chakrview.cognition.federation.runtime import EngineHealthStatus
                self.runtime.set_engine_health(
                    membership.node_id,
                    EngineHealthStatus.HEALTHY,
                    reason="Enrolled as active federation member",
                )
            if getattr(self.runtime, "journal", None):
                self.runtime.journal.append(
                    entry_type=JournalEntryType.NODE_MEMBERSHIP_GRANTED,
                    epoch=self.engine.current_epoch,
                    payload={"membership_id": membership_id, "node_id": membership.node_id},
                )

        return membership

    # ========================================================================
    # 3. Lifecycle Transitions: Suspend, Quarantine, Revoke, Terminate
    # ========================================================================

    def suspend_member(self, membership_id: str, reason: str = "Heartbeat timeout or network partition") -> None:
        """
        Transition member to SUSPENDED.
        CRITICAL: UNREACHABLE / SUSPENDED != REVOKED.
        """
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            membership.transition_to(MembershipState.SUSPENDED, reason=reason)

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_MEMBERSHIP_SUSPENDED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id, "reason": reason},
            )

        if self.runtime:
            if hasattr(self.runtime, "set_engine_health") and membership.node_id:
                from chakrview.cognition.federation.runtime import EngineHealthStatus
                self.runtime.set_engine_health(
                    membership.node_id,
                    EngineHealthStatus.UNREACHABLE,
                    reason=reason,
                )
            if getattr(self.runtime, "journal", None):
                self.runtime.journal.append(
                    entry_type=JournalEntryType.NODE_MEMBERSHIP_SUSPENDED,
                    epoch=self.engine.current_epoch,
                    payload={"membership_id": membership_id, "reason": reason},
                )

    def resume_member(self, membership_id: str, reason: str = "Resuming from suspension") -> None:
        """
        Transition member from SUSPENDED back to MEMBER standing.
        Re-enrolls node as active and records a heartbeat.
        """
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            membership.transition_to(MembershipState.MEMBER, reason=reason)
            membership.record_heartbeat(self.engine.current_epoch)

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_MEMBERSHIP_GRANTED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership.membership_id, "node_id": membership.node_id, "reason": reason},
            )

        if self.runtime:
            if hasattr(self.runtime, "set_engine_health") and membership.node_id:
                from chakrview.cognition.federation.runtime import EngineHealthStatus
                self.runtime.set_engine_health(
                    membership.node_id,
                    EngineHealthStatus.HEALTHY,
                    reason=reason,
                )
            if getattr(self.runtime, "journal", None):
                self.runtime.journal.append(
                    entry_type=JournalEntryType.NODE_MEMBERSHIP_GRANTED,
                    epoch=self.engine.current_epoch,
                    payload={"membership_id": membership.membership_id, "node_id": membership.node_id, "reason": reason},
                )

    def quarantine_member(
        self,
        membership_id: str,
        reason: str = "Security anomaly detected",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Transition node into QUARANTINED standing.
        All message transmission, capability execution, and state sync are strictly blocked.
        """
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            membership.transition_to(MembershipState.QUARANTINED, reason=reason)
            if evidence:
                membership.quarantine_evidence = dict(evidence)

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_QUARANTINED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id, "reason": reason},
            )
            audit.log(
                event_type=AuditEventType.ENGINE_QUARANTINED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id, "engine_id": membership.node_id, "reason": reason},
            )

        if self.runtime:
            if hasattr(self.runtime, "set_engine_health") and membership.node_id:
                from chakrview.cognition.federation.runtime import EngineHealthStatus
                self.runtime.set_engine_health(
                    membership.node_id,
                    EngineHealthStatus.QUARANTINED,
                    reason=reason,
                )
            if getattr(self.runtime, "journal", None):
                self.runtime.journal.append(
                    entry_type=JournalEntryType.NODE_QUARANTINED,
                    epoch=self.engine.current_epoch,
                    payload={"membership_id": membership_id, "reason": reason},
                )

    def revoke_member(
        self,
        membership_id: str,
        reason: str = "Administrative revocation",
        revoked_by: str = "local_authority",
    ) -> None:
        """
        Permanently revoke member node and trigger full local revocation cascade:
        MEMBERSHIP -> PEER IDENTITY -> SESSIONS -> SESSION KEYS -> CERT BINDINGS -> TRUST GRANTS.
        REVOKED is an absorbing terminal state.
        """
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            membership.transition_to(MembershipState.REVOKED, reason=reason)
            peer_id = membership.peer_id or membership.node_id

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_REVOKED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id, "reason": reason, "revoked_by": revoked_by},
            )

        # Trigger full cascade on engine if peer_id is registered
        if peer_id and hasattr(self.engine, "revoke_peer"):
            is_registered = True
            if hasattr(self.engine, "registry") and hasattr(self.engine.registry, "get_peer"):
                is_registered = self.engine.registry.get_peer(peer_id) is not None
            if is_registered:
                try:
                    self.engine.revoke_peer(peer_id=peer_id, reason=reason, revoked_by=revoked_by)
                except Exception as e:
                    logger.warning(f"Error during engine peer revocation cascade for {peer_id}: {e}")

        if self.runtime:
            if hasattr(self.runtime, "set_engine_health") and membership.node_id:
                from chakrview.cognition.federation.runtime import EngineHealthStatus
                self.runtime.set_engine_health(
                    membership.node_id,
                    EngineHealthStatus.TERMINATED,
                    reason=f"Revoked: {reason}",
                )
            if getattr(self.runtime, "journal", None):
                self.runtime.journal.append(
                    entry_type=JournalEntryType.NODE_REVOKED,
                    epoch=self.engine.current_epoch,
                    payload={"membership_id": membership_id, "reason": reason, "revoked_by": revoked_by},
                )

    def terminate_member(self, membership_id: str, reason: str = "Administrative termination") -> None:
        """Decommission node cleanly into TERMINATED state."""
        with self._lock:
            membership = self._get_membership_locked(membership_id)
            membership.transition_to(MembershipState.TERMINATED, reason=reason)

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.NODE_TERMINATED,
                epoch=self.engine.current_epoch,
                details={"membership_id": membership_id, "reason": reason},
            )

        if self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.NODE_TERMINATED,
                epoch=self.engine.current_epoch,
                payload={"membership_id": membership_id, "reason": reason},
            )

    # ========================================================================
    # 4. Lookup & Query APIs
    # ========================================================================

    def get_membership(self, membership_id: str) -> Optional[FederationNodeMembership]:
        with self._lock:
            return self._memberships.get(membership_id)

    def get_membership_by_node_id(self, node_id: str) -> Optional[FederationNodeMembership]:
        with self._lock:
            mem_id = self._node_to_membership.get(node_id)
            return self._memberships.get(mem_id) if mem_id else None

    def get_membership_by_endpoint(self, endpoint_id: str) -> Optional[FederationNodeMembership]:
        with self._lock:
            mem_id = self._endpoint_to_membership.get(endpoint_id)
            return self._memberships.get(mem_id) if mem_id else None

    def list_members(self, state_filter: Optional[MembershipState] = None) -> List[FederationNodeMembership]:
        with self._lock:
            if state_filter is None:
                return list(self._memberships.values())
            return [m for m in self._memberships.values() if m.state == state_filter]

    def list_memberships(self, state_filter: Optional[MembershipState] = None) -> List[FederationNodeMembership]:
        """Alias for list_members."""
        return self.list_members(state_filter=state_filter)

    def list_active_members(self) -> List[FederationNodeMembership]:
        return self.list_members(state_filter=MembershipState.MEMBER)

    def record_heartbeat(self, node_id: str, epoch: int, timestamp: Optional[float] = None) -> None:
        with self._lock:
            mem_id = self._node_to_membership.get(node_id) or node_id
            membership = self._memberships.get(mem_id)
            if not membership:
                raise UnknownPeerError(f"Cannot record heartbeat: node '{node_id}' is not a registered member.")
            membership.record_heartbeat(epoch=epoch, timestamp=timestamp)

    def _get_membership_locked(self, id_or_node_id: str) -> FederationNodeMembership:
        mem = self._memberships.get(id_or_node_id)
        if not mem:
            mem_id = self._node_to_membership.get(id_or_node_id)
            if mem_id:
                mem = self._memberships.get(mem_id)
        if not mem:
            raise MembershipError(f"Membership '{id_or_node_id}' does not exist.")
        return mem
