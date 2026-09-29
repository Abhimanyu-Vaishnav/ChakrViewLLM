"""
Distributed Federation Coordinator (Step 33).

Central coordinator orchestrating multi-engine consistency:
- Federation Engine Identity management
- Security State Versions and reproducible State Digests
- Bounded Replay State Synchronization
- Bounded Trust State Consistency (REMOTE_CLAIM != LOCAL_AUTHORITY)
- Monotonic, Idempotent Revocation Propagation (REVOKED -> NEVER ACTIVE AGAIN)
- Bounded Federation Coordination Handshakes
- Fail-Closed Conflict Detection and Audit Logging

CRITICAL ARCHITECTURAL AXIOMS:
1. LOCAL_AUTHORITY > PEER_AUTHORITY & REMOTE_ENGINE != LOCAL_AUTHORITY
2. FEDERATION_COORDINATION != AUTHORITY_TRANSFER
3. FEDERATION_HANDSHAKE != TRUST_GRANT & FEDERATION_HANDSHAKE != AUTHORIZATION
4. ZERO SECRET LEAKAGE: Private keys and session secrets are never shared.
5. ZERO NEURAL MUTATION: Model weights remain strictly frozen (ΔW = 0).
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import time

from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    FederationHandshakeRequest,
    FederationHandshakeResponse,
    HandshakeStatus,
    ReplaySyncMessage,
    TrustSyncMessage,
    RevocationSyncRecord,
    RevocationSyncMessage,
    RevocationTargetType,
    FederationSecurityStateDigest,
    MAX_FEDERATION_ENGINES,
    FEDERATION_PROTOCOL_VERSION,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.state import FederationStateManager
from chakrview.cognition.federation.replay_sync import ReplayStateSynchronizer
from chakrview.cognition.federation.trust_sync import TrustStateSynchronizer
from chakrview.cognition.federation.revocation_sync import RevocationStateSynchronizer
from chakrview.cognition.federation.handshake import FederationHandshakeManager
from chakrview.cognition.federation.errors import (
    FederationCoordinationError,
    EngineIdentityError,
    ProtocolMismatchError,
    HandshakeError,
    CoordinationCapacityError,
    StateDigestConflictError,
    StaleStateError,
)
from chakrview.cognition.peering.session import SecurePeerSession
from chakrview.cognition.peering.registry import PeerRegistry
from chakrview.cognition.peering.models import RevocationRecord, AuditEventType
from chakrview.cognition.peering.audit import BoundedAuditLogger


class DistributedFederationCoordinator:
    """
    Coordinates distributed multi-engine security state consistency, replay defense,
    and revocation propagation across federated zones.
    """

    def __init__(
        self,
        engine_id: str,
        zone_id: str,
        initial_epoch: int = 1,
        public_key_fingerprint: Optional[str] = None,
        audit_logger: Optional[BoundedAuditLogger] = None,
    ) -> None:
        self.engine_identity = FederationEngineIdentityProvider.create_identity(
            engine_id=engine_id,
            zone_id=zone_id,
            created_epoch=initial_epoch,
            public_key_fingerprint=public_key_fingerprint,
        )
        self.audit_logger = audit_logger or BoundedAuditLogger()

        self.state_manager = FederationStateManager(
            engine_identity=self.engine_identity,
            initial_epoch=initial_epoch,
        )
        self.replay_synchronizer = ReplayStateSynchronizer(engine_id=self.engine_identity.engine_id)
        self.trust_synchronizer = TrustStateSynchronizer(
            engine_id=self.engine_identity.engine_id,
            zone_id=self.engine_identity.zone_id,
        )
        self.revocation_synchronizer = RevocationStateSynchronizer(engine_id=self.engine_identity.engine_id)
        self.handshake_manager = FederationHandshakeManager(
            local_identity=self.engine_identity,
            state_manager=self.state_manager,
        )

        self._registered_engines: Dict[str, FederationEngineIdentity] = {}
        self.engine: Optional[Any] = None

    @property
    def current_epoch(self) -> int:
        return self.state_manager.current_epoch

    @property
    def current_version(self) -> SecurityStateVersion:
        return self.state_manager.current_version

    # ========================================================================
    # 1. Engine Registration (Phase 1)
    # ========================================================================

    def register_remote_engine(self, remote_identity: FederationEngineIdentity) -> None:
        """
        Register a known remote coordination engine.
        CRITICAL: Registration verifies identity structure; it confers ZERO authority.
        """
        is_valid, reason = FederationEngineIdentityProvider.validate_identity(remote_identity)
        if not is_valid:
            raise EngineIdentityError(f"Cannot register invalid engine identity: {reason}")

        if remote_identity.engine_id == self.engine_identity.engine_id:
            return

        if (
            remote_identity.engine_id not in self._registered_engines
            and len(self._registered_engines) >= MAX_FEDERATION_ENGINES
        ):
            raise CoordinationCapacityError(
                f"Cannot register engine '{remote_identity.engine_id}': "
                f"maximum engine capacity reached ({MAX_FEDERATION_ENGINES})."
            )

        self._registered_engines[remote_identity.engine_id] = remote_identity
        self.audit_logger.log(
            event_type=AuditEventType.ENGINE_REGISTERED,
            epoch=self.current_epoch,
            zone_id=remote_identity.zone_id,
            details={"registered_engine_id": remote_identity.engine_id},
        )

    def get_registered_engine(self, engine_id: str) -> Optional[FederationEngineIdentity]:
        return self._registered_engines.get(engine_id)

    def list_registered_engines(self) -> List[FederationEngineIdentity]:
        return list(self._registered_engines.values())

    # ========================================================================
    # 2. Handshake Protocol (Phase 8)
    # ========================================================================

    def initiate_handshake(
        self,
        remote_coordinator: "DistributedFederationCoordinator",
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocations: Optional[List[RevocationRecord]] = None,
    ) -> FederationHandshakeResponse:
        """
        Initiate a coordination handshake with a remote federation coordinator.
        """
        self.audit_logger.log(
            event_type=AuditEventType.FEDERATION_HANDSHAKE_STARTED,
            epoch=self.current_epoch,
            zone_id=remote_coordinator.engine_identity.zone_id,
            details={"target_engine_id": remote_coordinator.engine_identity.engine_id},
        )

        request = self.handshake_manager.create_handshake_request(
            session=session,
            registry=registry,
            revocations=revocations,
        )

        response = remote_coordinator.handle_handshake(
            request=request,
            session=session,
            registry=registry,
            revocations=revocations,
        )

        if response.is_accepted():
            self.register_remote_engine(remote_coordinator.engine_identity)
            self.audit_logger.log(
                event_type=AuditEventType.FEDERATION_HANDSHAKE_COMPLETED,
                epoch=self.current_epoch,
                zone_id=remote_coordinator.engine_identity.zone_id,
                details={
                    "target_engine_id": remote_coordinator.engine_identity.engine_id,
                    "sync_required": response.sync_required,
                },
            )
        else:
            self.audit_logger.log(
                event_type=AuditEventType.FEDERATION_HANDSHAKE_FAILED,
                epoch=self.current_epoch,
                zone_id=remote_coordinator.engine_identity.zone_id,
                details={
                    "target_engine_id": remote_coordinator.engine_identity.engine_id,
                    "status": response.status.value,
                    "reason": response.details.get("error", "Unknown"),
                },
            )

        return response

    def handle_handshake(
        self,
        request: FederationHandshakeRequest,
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocations: Optional[List[RevocationRecord]] = None,
    ) -> FederationHandshakeResponse:
        """
        Process an inbound handshake request from a remote coordinator.
        """
        response = self.handshake_manager.process_handshake_request(
            request=request,
            session=session,
            registry=registry,
            revocations=revocations,
        )

        if response.is_accepted():
            self.register_remote_engine(request.sender_identity)

        return response

    # ========================================================================
    # 3. Replay Synchronization (Phase 3)
    # ========================================================================

    def synchronize_replay(
        self,
        remote_coordinator: "DistributedFederationCoordinator",
        session: SecurePeerSession,
        remote_session: Optional[SecurePeerSession] = None,
    ) -> Dict[str, Any]:
        """
        Synchronize advisory replay state with a remote coordinator.
        Local replay defense remains unconditionally authoritative.
        """
        target_session = remote_session or session
        self.audit_logger.log(
            event_type=AuditEventType.STATE_SYNC_STARTED,
            epoch=self.current_epoch,
            session_id=session.session_id,
            details={"sync_type": "REPLAY", "target_engine_id": remote_coordinator.engine_identity.engine_id},
        )

        sync_msg = self.replay_synchronizer.generate_sync_message(
            session=session,
            epoch=self.current_epoch,
            state_version=self.current_version.version,
        )

        outcome = remote_coordinator.replay_synchronizer.ingest_sync_message(
            session=target_session,
            sync_message=sync_msg,
        )

        self.audit_logger.log(
            event_type=AuditEventType.REPLAY_STATE_SYNCED,
            epoch=self.current_epoch,
            session_id=session.session_id,
            details=outcome,
        )
        self.audit_logger.log(
            event_type=AuditEventType.STATE_SYNC_COMPLETED,
            epoch=self.current_epoch,
            session_id=session.session_id,
            details={"sync_type": "REPLAY"},
        )

        return outcome

    # ========================================================================
    # 4. Trust Synchronization (Phase 5)
    # ========================================================================

    def synchronize_trust(
        self,
        remote_coordinator: "DistributedFederationCoordinator",
        local_registry: PeerRegistry,
        remote_registry: PeerRegistry,
        local_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Synchronize trust state claims with a remote coordinator.
        CRITICAL: Remote claims NEVER elevate local trust or grant capabilities.
        """
        self.audit_logger.log(
            event_type=AuditEventType.STATE_SYNC_STARTED,
            epoch=self.current_epoch,
            details={"sync_type": "TRUST", "target_engine_id": remote_coordinator.engine_identity.engine_id},
        )

        trust_msg = self.trust_synchronizer.generate_sync_message(
            registry=local_registry,
            current_epoch=self.current_epoch,
            state_version=self.current_version.version,
        )

        outcome = remote_coordinator.trust_synchronizer.ingest_sync_message(
            registry=remote_registry,
            sync_message=trust_msg,
            local_policy=local_policy,
            current_epoch=remote_coordinator.current_epoch,
        )

        self.audit_logger.log(
            event_type=AuditEventType.TRUST_STATE_SYNCED,
            epoch=self.current_epoch,
            details=outcome,
        )
        self.audit_logger.log(
            event_type=AuditEventType.STATE_SYNC_COMPLETED,
            epoch=self.current_epoch,
            details={"sync_type": "TRUST"},
        )

        return outcome

    # ========================================================================
    # 5. Revocation Propagation (Phase 6)
    # ========================================================================

    def record_and_propagate_revocation(
        self,
        target_type: RevocationTargetType,
        target_id: str,
        reason: str,
        engine: Any,
        remote_coordinators: Optional[List["DistributedFederationCoordinator"]] = None,
    ) -> RevocationSyncRecord:
        """
        Record a local revocation and synchronously propagate to remote coordinators.
        Enforces: REVOKED -> NEVER ACTIVE AGAIN.
        """
        rec = self.revocation_synchronizer.record_local_revocation(
            target_type=target_type,
            target_id=target_id,
            reason=reason,
            revoked_epoch=self.current_epoch,
        )
        self.state_manager.increment_version()

        self.audit_logger.log(
            event_type=AuditEventType.REVOCATION_PROPAGATED,
            epoch=self.current_epoch,
            peer_id=target_id if target_type == RevocationTargetType.PEER else None,
            details={
                "revocation_id": rec.revocation_id,
                "target_type": target_type.value,
                "reason": reason,
            },
        )

        if remote_coordinators:
            msg = self.revocation_synchronizer.generate_sync_message(
                current_epoch=self.current_epoch,
                state_version=self.current_version.version,
            )
            for remote in remote_coordinators:
                target_engine = getattr(remote, "engine", remote)
                remote.revocation_synchronizer.ingest_sync_message(
                    engine=target_engine,
                    sync_message=msg,
                )
                remote.state_manager.increment_version()

        return rec

    # ========================================================================
    # 6. Cryptographic State Digest & Invariants (Phase 7)
    # ========================================================================

    def compute_state_digest(
        self,
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocations: Optional[List[RevocationRecord]] = None,
    ) -> FederationSecurityStateDigest:
        """
        Compute master composite security state digest across all state facets.
        """
        return self.state_manager.compute_state_digest(
            session=session,
            registry=registry,
            revocation_records=revocations,
        )

    def compare_state_digest(
        self,
        other_coordinator: "DistributedFederationCoordinator",
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocations: Optional[List[RevocationRecord]] = None,
    ) -> Tuple[bool, str]:
        """
        Compare local state digest with another coordinator.
        """
        local_digest = self.compute_state_digest(session, registry, revocations)
        other_digest = other_coordinator.compute_state_digest(session, registry, revocations)
        return self.state_manager.compare_state_digest(local_digest, other_digest)
