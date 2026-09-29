"""
Central Cross-Zone Federation Engine (Step 29).

CRITICAL ARCHITECTURAL AXIOMS:
1. CROSS-ZONE FEDERATION != AUTHORITY TRANSFER:
   Peered zones can exchange evidence, verify outputs, and delegate bounded tasks,
   but a peer NEVER gains local authority, tenant authority, or capability authority.
2. LOCAL_AUTHORITY > PEER_AUTHORITY & PEER_TRUST != PEER_AUTHORITY:
   Trust is bounded, expiring, and scoped. Local policies and CapabilityGate remain
   unconditionally authoritative.
3. FEDERATION_ENGINE != AUTHORITY:
   The engine coordinates discovery, attestation, negotiation, and gate enforcement;
   it cannot independently authorize capability execution.
4. ZERO WEIGHT MUTATION (ΔW = 0):
   Federation operates around the frozen ChakrMicro v0.1 core. Weight hashes are
   verified before and after every cross-zone transaction.
"""

import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import torch

from chakrview.cognition.peering.models import (
    PeerIdentity,
    PeerAttestation,
    PeerDeclaration,
    PeerRegistration,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    DiscoveryStatus,
    FederationScope,
    ProhibitedScope,
    NegotiationAgreement,
    RevocationRecord,
    AuditEventType,
    AuditRecord,
    SafePublicPeeringTrace,
)
from chakrview.cognition.peering.identity import PeerIdentityProvider, IdentityValidationError
from chakrview.cognition.peering.attestation import PeerAttestationVerifier
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy, PolicyViolationError
from chakrview.cognition.peering.trust import TrustModel
from chakrview.cognition.peering.discovery import PeerDiscoveryManager
from chakrview.cognition.peering.negotiation import TrustNegotiator
from chakrview.cognition.peering.registry import PeerRegistry, DuplicatePeerError, PeerRegistryCapacityError
from chakrview.cognition.peering.revocation import RevocationManager
from chakrview.cognition.peering.isolation import CrossZoneIsolationGuard, IsolationViolationError
from chakrview.cognition.peering.audit import BoundedAuditLogger
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest, CapabilityContext

# Step 30 Cryptographic & Transport Extensions
from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
    CryptographicPeerIdentity,
    KeyLifecycleState,
    SignatureVerificationError,
    KeyStateError,
)
from chakrview.cognition.peering.authentication import (
    AuthChallenge,
    AuthChallengeResponse,
    ChallengeResponseAuthenticator,
    PeerAuthenticationState,
    AuthenticationError,
    ReplayedChallengeError,
)
from chakrview.cognition.peering.session import (
    SecurePeerSession,
    SessionStatus,
    SessionTransitionError,
    SessionKeyState,
)
from chakrview.cognition.transport.models import (
    WireEnvelope,
    MessageType,
)
from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.errors import (
    TransportProtocolError,
    ReplayAttackError,
    WireSecurityError,
)
from chakrview.cognition.transport.security import (
    CertificateMetadata,
    PeerCertificateBinding,
    PeerCertificateBinder,
    CertificateRevocationRegistry,
    validate_certificate,
    validate_certificate_or_raise,
    parse_certificate_from_pem,
    parse_certificate_from_der,
    extract_certificate_metadata,
    CertificateValidationError,
    PeerBindingMismatchError,
)

# Step 33 Distributed Federation Coordination
from chakrview.cognition.federation.coordinator import DistributedFederationCoordinator
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    FederationSecurityStateDigest,
    RevocationTargetType,
    HandshakeStatus,
    FederationHandshakeResponse,
    RevocationSyncRecord,
)
from chakrview.cognition.federation.persistence.models import JournalEntryType


class CrossZoneAuthorizationError(PermissionError):
    """Raised when a cross-zone request fails trust, policy, or isolation checks."""
    pass


class WeightMutationDetectedError(RuntimeError):
    """Raised when runtime neural weights mutate during cross-zone operations."""
    pass


class CrossZoneFederationEngine:
    """
    Coordinates multi-zone peer discovery, attestation, trust negotiation,
    cryptographic authentication, capability gating, and bounded auditable execution.
    """

    def __init__(
        self,
        local_zone_id: str,
        policy: Optional[CrossZoneFederationPolicy] = None,
        capability_gate: Optional[CapabilityGate] = None,
        model: Optional[torch.nn.Module] = None,
        initial_epoch: int = 1,
        local_private_key: Optional[Ed25519PrivateKeyWrapper] = None,
        local_peer_id: Optional[str] = None,
        coordinator: Optional[DistributedFederationCoordinator] = None,
        runtime: Optional[Any] = None,
    ) -> None:
        self.local_zone_id = local_zone_id
        self.policy = policy or CrossZoneFederationPolicy()
        self.capability_gate = capability_gate
        self.model = model
        self.current_epoch = initial_epoch

        self.local_private_key = local_private_key or Ed25519PrivateKeyWrapper.generate()
        self.local_public_key = self.local_private_key.public_key()
        self.local_peer_id = local_peer_id or f"peer_local_{self.local_public_key.fingerprint[:12]}"

        self.revocation_manager = RevocationManager()
        self.registry = PeerRegistry(revocation_manager=self.revocation_manager)
        self.discovery_manager = PeerDiscoveryManager(policy=self.policy)
        self.negotiator = TrustNegotiator(local_zone_id=self.local_zone_id, policy=self.policy)
        self.audit_logger = BoundedAuditLogger()
        self.authenticator = ChallengeResponseAuthenticator()
        self.sessions: Dict[str, SecurePeerSession] = {}
        self.certificate_binder = PeerCertificateBinder()
        self.certificate_revocation_registry = CertificateRevocationRegistry()

        # Step 33 Distributed Federation Coordinator
        self.coordinator = coordinator or DistributedFederationCoordinator(
            engine_id=f"eng_{self.local_peer_id}",
            zone_id=self.local_zone_id,
            initial_epoch=initial_epoch,
            public_key_fingerprint=self.local_public_key.fingerprint,
            audit_logger=self.audit_logger,
        )
        self.coordinator.engine = self

        # Step 34 Federation Runtime
        self.runtime = runtime
        self._membership_manager: Optional[Any] = None
        if self.runtime and not hasattr(self.runtime, "engine"):
            self.runtime.engine = self

    @property
    def membership_manager(self) -> Any:
        if self._membership_manager is None:
            if self.runtime and hasattr(self.runtime, "membership_manager"):
                self._membership_manager = self.runtime.membership_manager
            else:
                from chakrview.cognition.federation.discovery.membership import FederationMembershipManager
                self._membership_manager = FederationMembershipManager(engine=self, runtime=self.runtime)
        return self._membership_manager

    @membership_manager.setter
    def membership_manager(self, mgr: Any) -> None:
        self._membership_manager = mgr

    def attach_runtime(self, runtime: Any) -> None:
        """Attach a FederationRuntime to this engine for durable write-ahead journaling."""
        self.runtime = runtime
        if runtime and getattr(runtime, "engine", None) != self:
            runtime.engine = self
        if runtime and hasattr(runtime, "membership_manager"):
            self._membership_manager = runtime.membership_manager

    @property
    def connection_manager(self) -> Any:
        if not hasattr(self, "_connection_manager") or self._connection_manager is None:
            from chakrview.cognition.federation.discovery.connection import FederationConnectionManager
            self._connection_manager = FederationConnectionManager(engine=self)
        return self._connection_manager

    @connection_manager.setter
    def connection_manager(self, mgr: Any) -> None:
        self._connection_manager = mgr

    @property
    def dispatcher(self) -> Any:
        if not hasattr(self, "_dispatcher") or self._dispatcher is None:
            from chakrview.cognition.federation.transport.dispatcher import FederationMessageDispatcher
            self._dispatcher = FederationMessageDispatcher(engine=self)
            if hasattr(self, "resource_manager"):
                _ = self.resource_manager
        return self._dispatcher

    @dispatcher.setter
    def dispatcher(self, d: Any) -> None:
        self._dispatcher = d

    @property
    def transport_client(self) -> Any:
        if not hasattr(self, "_transport_client") or self._transport_client is None:
            from chakrview.cognition.federation.transport.client import FederationTransportClient
            self._transport_client = FederationTransportClient(engine=self)
        return self._transport_client

    @transport_client.setter
    def transport_client(self, c: Any) -> None:
        self._transport_client = c

    @property
    def transport_server(self) -> Any:
        if not hasattr(self, "_transport_server") or self._transport_server is None:
            from chakrview.cognition.federation.transport.server import FederationTransportServer
            self._transport_server = FederationTransportServer(engine=self, dispatcher=self.dispatcher)
        return self._transport_server

    @transport_server.setter
    def transport_server(self, s: Any) -> None:
        self._transport_server = s

    @property
    def resource_manager(self) -> Any:
        if not hasattr(self, "_resource_manager") or self._resource_manager is None:
            from chakrview.cognition.federation.resources.manager import FederationResourceManager
            self._resource_manager = FederationResourceManager(engine=self)
        return self._resource_manager

    @resource_manager.setter
    def resource_manager(self, mgr: Any) -> None:
        self._resource_manager = mgr

    @property
    def resource_registry(self) -> Any:
        return self.resource_manager.registry

    @property
    def grant_manager(self) -> Any:
        if not hasattr(self, "_grant_manager") or self._grant_manager is None:
            from chakrview.cognition.federation.tasks.grant import ExecutionGrantManager
            self._grant_manager = ExecutionGrantManager(local_node_id=self.local_peer_id)
        return self._grant_manager

    @grant_manager.setter
    def grant_manager(self, mgr: Any) -> None:
        self._grant_manager = mgr

    @property
    def task_scheduler(self) -> Any:
        if not hasattr(self, "_task_scheduler") or self._task_scheduler is None:
            from chakrview.cognition.federation.tasks.scheduler import DeterministicTaskScheduler
            self._task_scheduler = DeterministicTaskScheduler(
                local_node_id=self.local_peer_id,
                resource_registry=self.resource_manager.registry,
                membership_manager=self.membership_manager,
                peering_engine=self,
            )
        return self._task_scheduler

    @task_scheduler.setter
    def task_scheduler(self, sched: Any) -> None:
        self._task_scheduler = sched

    @property
    def task_executor(self) -> Any:
        if not hasattr(self, "_task_executor") or self._task_executor is None:
            from chakrview.cognition.federation.tasks.executor import FederationTaskExecutor
            self._task_executor = FederationTaskExecutor(
                local_node_id=self.local_peer_id,
                grant_manager=self.grant_manager,
                capability_gate=self.capability_gate,
            )
        return self._task_executor

    @task_executor.setter
    def task_executor(self, exec_: Any) -> None:
        self._task_executor = exec_

    @property
    def task_coordinator(self) -> Any:
        if not hasattr(self, "_task_coordinator") or self._task_coordinator is None:
            from chakrview.cognition.federation.tasks.coordinator import FederationTaskCoordinator
            from chakrview.cognition.federation.tasks.checkpoint import TaskCheckpointManager
            journal = getattr(self.runtime, "journal", None) if hasattr(self, "runtime") and self.runtime else None
            checkpoint_mgr = TaskCheckpointManager(journal=journal)
            self._task_coordinator = FederationTaskCoordinator(
                local_node_id=self.local_peer_id,
                scheduler=self.task_scheduler,
                checkpoint_manager=checkpoint_mgr,
                local_executor=self.task_executor,
                transport_client=self.transport_client,
                journal=journal,
                audit_logger=self.audit_logger,
            )
        return self._task_coordinator

    @task_coordinator.setter
    def task_coordinator(self, tc: Any) -> None:
        self._task_coordinator = tc

    @property
    def consensus_engine(self) -> Any:
        if not hasattr(self, "_consensus_engine") or self._consensus_engine is None:
            from chakrview.cognition.federation.consensus.engine import FederatedConsensusEngine
            journal = getattr(self.runtime, "journal", None) if hasattr(self, "runtime") and self.runtime else None
            validators = [self.local_peer_id]
            if hasattr(self, "membership_manager") and self.membership_manager:
                members = self.membership_manager.list_active_members()
                if members:
                    validators = [m.node_id for m in members]
            self._consensus_engine = FederatedConsensusEngine(
                local_node_id=self.local_peer_id,
                validators=validators,
                capability_gate=self.capability_gate,
                journal=journal,
                audit_logger=self.audit_logger,
            )
        return self._consensus_engine

    @consensus_engine.setter
    def consensus_engine(self, engine: Any) -> None:
        self._consensus_engine = engine

    @property
    def engine_identity(self) -> FederationEngineIdentity:
        return self.coordinator.engine_identity

    @property
    def state_version(self) -> SecurityStateVersion:
        return self.coordinator.current_version

    # ========================================================================

    # 1. Neural Core Invariant Checks
    # ========================================================================

    def _compute_weight_hash(self) -> str:
        """Compute SHA-256 digest of all neural parameters."""
        if self.model is None:
            return "NO_MODEL"
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.model.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def _verify_weight_invariants(self, pre_hash: str) -> None:
        """Verify model weights remain strictly frozen."""
        if self.model is None or pre_hash == "NO_MODEL":
            return
        post_hash = self._compute_weight_hash()
        if pre_hash != post_hash:
            raise WeightMutationDetectedError(
                f"FATAL: Neural core weight mutation detected during cross-zone operation! "
                f"Pre-hash: {pre_hash}, Post-hash: {post_hash}."
            )

    # ========================================================================
    # 2. Logical Time & Epoch Management
    # ========================================================================

    def advance_epoch(self, epochs: int = 1) -> int:
        """Advance logical epoch and expire outdated peer grants."""
        if epochs < 1:
            raise ValueError("Epoch advancement must be >= 1.")
        self.current_epoch += epochs
        if hasattr(self, "coordinator"):
            self.coordinator.state_manager.advance_epoch(self.current_epoch)

        expired_peer_ids = self.registry.expire_peers(self.current_epoch)
        for peer_id in expired_peer_ids:
            self.audit_logger.log(
                event_type=AuditEventType.FEDERATION_EXPIRED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                zone_id=None,
                details={"reason": f"Trust grant expired at epoch {self.current_epoch}."},
            )

        if hasattr(self, "runtime") and self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.EPOCH_ADVANCED,
                epoch=self.current_epoch,
                payload={"new_epoch": self.current_epoch, "epochs_advanced": epochs},
            )

        return self.current_epoch

    # ========================================================================
    # 3. Peer Discovery (DISCOVER != TRUST)
    # ========================================================================

    def discover_peer(self, declaration: PeerDeclaration) -> PeerRegistration:
        """
        Announce/discover a peer.
        Enforces DISCOVER != TRUST: trust_grant is strictly None.
        """
        peer = declaration.identity

        # Register in discovery manager
        reg = self.discovery_manager.discover(declaration, current_epoch=self.current_epoch)

        # Register in central peer registry
        try:
            self.registry.register_peer(reg)
        except DuplicatePeerError:
            # Update existing registration in registry
            existing = self.registry.get_peer(peer.peer_id)
            if existing:
                existing.last_seen_epoch = self.current_epoch
                existing.latest_attestation = declaration.attestation
                reg = existing

        # Log audit events
        self.audit_logger.log(
            event_type=AuditEventType.PEER_DISCOVERED,
            epoch=self.current_epoch,
            peer_id=peer.peer_id,
            zone_id=peer.zone_id,
            details={"organization_id": peer.organization_id},
        )
        self.audit_logger.log(
            event_type=AuditEventType.IDENTITY_PRESENTED,
            epoch=self.current_epoch,
            peer_id=peer.peer_id,
            zone_id=peer.zone_id,
            details={"fingerprint": peer.fingerprint},
        )

        return reg

    # ========================================================================
    # 4. Peer Attestation
    # ========================================================================

    def attest_peer(self, peer_id: str, attestation: PeerAttestation) -> Tuple[bool, str]:
        """
        Process and verify a peer's architectural attestation claim.
        """
        reg = self.registry.get_peer(peer_id)
        if not reg:
            return False, f"Peer '{peer_id}' not found in registry."

        self.audit_logger.log(
            event_type=AuditEventType.ATTESTATION_RECEIVED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            zone_id=reg.identity.zone_id,
            details={"attestation_id": attestation.attestation_id},
        )

        valid, reason = PeerAttestationVerifier.verify_attestation(
            attestation=attestation,
            expected_peer_id=peer_id,
            expected_zone_id=reg.identity.zone_id,
        )

        if valid:
            reg.latest_attestation = attestation
            reg.discovery_status = DiscoveryStatus.VERIFIED
            self.audit_logger.log(
                event_type=AuditEventType.ATTESTATION_ACCEPTED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                zone_id=reg.identity.zone_id,
                details={"reason": reason},
            )
            return True, reason
        else:
            self.audit_logger.log(
                event_type=AuditEventType.ATTESTATION_REJECTED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                zone_id=reg.identity.zone_id,
                details={"reason": reason},
            )
            return False, reason

    # ========================================================================
    # 5. Trust & Scope Negotiation
    # ========================================================================

    def negotiate_trust(self, declaration: PeerDeclaration) -> NegotiationAgreement:
        """
        Execute deterministic trust negotiation and record resulting grants.
        """
        peer = declaration.identity

        # Execute negotiation protocol
        agreement = self.negotiator.negotiate(declaration, current_epoch=self.current_epoch)

        if agreement.is_successful() and agreement.trust_grant:
            # Update registry with active grant
            self.registry.update_trust_grant(peer.peer_id, agreement.trust_grant)

            self.audit_logger.log(
                event_type=AuditEventType.TRUST_NEGOTIATED,
                epoch=self.current_epoch,
                peer_id=peer.peer_id,
                zone_id=peer.zone_id,
                details={"trust_level": agreement.trust_grant.trust_level.value},
            )
            self.audit_logger.log(
                event_type=AuditEventType.POLICY_ACCEPTED,
                epoch=self.current_epoch,
                peer_id=peer.peer_id,
                zone_id=peer.zone_id,
                details={"accepted_scopes": [s.value for s in agreement.accepted_scopes]},
            )
            self.audit_logger.log(
                event_type=AuditEventType.FEDERATION_ESTABLISHED,
                epoch=self.current_epoch,
                peer_id=peer.peer_id,
                zone_id=peer.zone_id,
                details={"expires_epoch": agreement.trust_grant.expires_epoch},
            )
        else:
            self.audit_logger.log(
                event_type=AuditEventType.POLICY_REJECTED,
                epoch=self.current_epoch,
                peer_id=peer.peer_id,
                zone_id=peer.zone_id,
                details={"rejected_scopes": [s.value for s in agreement.rejected_scopes]},
            )

        return agreement

    # ========================================================================
    # 6. Gated Request Authorization
    # ========================================================================

    def authorize_cross_zone_request(
        self,
        peer_id: str,
        peer_zone_id: str,
        peer_tenant_id: str,
        target_tenant_id: str,
        session_id: str,
        requested_scope: FederationScope,
        payload: Dict[str, Any],
        capability_request: Optional[CapabilityRequest] = None,
        capability_context: Optional[CapabilityContext] = None,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Authorize and sanitize a cross-zone interaction request.

        Enforces:
        Peer Request -> Peer Identity -> Trust State -> Federation Policy -> CapabilityGate -> Execution.
        """
        pre_hash = self._compute_weight_hash()

        try:
            # 1. Tenant boundary isolation
            CrossZoneIsolationGuard.validate_tenant_boundary(
                peer_zone_id=peer_zone_id,
                target_zone_id=self.local_zone_id,
                peer_tenant_id=peer_tenant_id,
                target_tenant_id=target_tenant_id,
            )

            # 2. Payload sanitization (blocks weights, scratchpads, secrets)
            sanitized_payload = CrossZoneIsolationGuard.sanitize_payload(payload)

            # 3. Peer standing in registry
            reg = self.registry.get_peer(peer_id)
            if not reg:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=peer_id,
                    zone_id=peer_zone_id,
                    tenant_id=peer_tenant_id,
                    session_id=session_id,
                    details={"reason": "Peer not registered"},
                )
                raise CrossZoneAuthorizationError(f"Peer '{peer_id}' is not registered.")

            if reg.discovery_status == DiscoveryStatus.REVOKED or reg.revocation_record:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=peer_id,
                    zone_id=peer_zone_id,
                    tenant_id=peer_tenant_id,
                    session_id=session_id,
                    details={"reason": "Peer has been revoked"},
                )
                raise CrossZoneAuthorizationError(f"Peer '{peer_id}' has been revoked.")

            # 4. Trust grant scope check
            trusted, trust_reason = TrustModel.is_trusted_for_scope(
                grant=reg.trust_grant,
                scope=requested_scope,
                current_epoch=self.current_epoch,
            )
            if not trusted:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=peer_id,
                    zone_id=peer_zone_id,
                    tenant_id=peer_tenant_id,
                    session_id=session_id,
                    details={"reason": trust_reason, "scope": requested_scope.value},
                )
                raise CrossZoneAuthorizationError(f"Cross-zone request denied by trust grant: {trust_reason}")

            # 5. Local policy verification
            policy_ok, policy_reason = self.policy.evaluate_scope_request(reg.identity, requested_scope)
            if not policy_ok:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=peer_id,
                    zone_id=peer_zone_id,
                    tenant_id=peer_tenant_id,
                    session_id=session_id,
                    details={"reason": policy_reason, "scope": requested_scope.value},
                )
                raise CrossZoneAuthorizationError(f"Cross-zone request denied by policy: {policy_reason}")

            # 6. CapabilityGate mediation: Peer NEVER executes capability directly
            if capability_request is not None:
                if self.capability_gate is None:
                    raise CrossZoneAuthorizationError("Capability requested but CapabilityGate is not configured.")

                # CapabilityGate authorizes under local authority
                is_authorized = self.capability_gate.authorize(
                    request=capability_request,
                    context=capability_context,
                )
                if not is_authorized:
                    self.audit_logger.log(
                        event_type=AuditEventType.REQUEST_DENIED,
                        epoch=self.current_epoch,
                        peer_id=peer_id,
                        zone_id=peer_zone_id,
                        tenant_id=peer_tenant_id,
                        session_id=session_id,
                        details={"reason": "CapabilityGate denied authorization", "capability": capability_request.capability_name},
                    )
                    raise CapabilityAuthorizationError(
                        f"Capability '{capability_request.capability_name}' denied by CapabilityGate."
                    )

            # 7. Verify neural core immutability
            self._verify_weight_invariants(pre_hash)

            return True, sanitized_payload, f"Request authorized under scope {requested_scope.value}."

        except Exception:
            self._verify_weight_invariants(pre_hash)
            raise

    # ========================================================================
    # 7. Trust Revocation
    # ========================================================================

    def revoke_peer(
        self,
        peer_id: str,
        reason: str,
        revoked_by: str = "local_authority",
    ) -> RevocationRecord:
        """
        Administratively revoke peer trust and trigger full revocation cascade (Step 32):
        1. Log REVOCATION_CASCADE_TRIGGERED
        2. Invalidate trust grant and peer registration in registry
        3. Revoke peer's cryptographic identity
        4. Invalidate and revoke all active sessions for this peer
        5. Deactivate and unbind certificate bindings
        6. Log REVOCATION_CASCADE_COMPLETED and FEDERATION_REVOKED
        """
        self.audit_logger.log(
            event_type=AuditEventType.REVOCATION_CASCADE_TRIGGERED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            details={"reason": reason, "revoked_by": revoked_by},
        )

        record = self.registry.revoke_peer(
            peer_id=peer_id,
            reason=reason,
            revoked_epoch=self.current_epoch,
            revoked_by=revoked_by,
        )

        # Invalidate cryptographic identity if present
        reg = self.registry.get_peer(peer_id)
        if reg and reg.cryptographic_identity:
            reg.cryptographic_identity.revoke(
                reason=reason,
                revoked_epoch=self.current_epoch,
                revoked_by=revoked_by,
            )

        # Invalidate all active sessions for this peer
        revoked_session_count = 0
        for sid, sess in list(self.sessions.items()):
            if sess.remote_peer_id == peer_id:
                sess.revoke(reason=f"Peer revoked: {reason}")
                revoked_session_count += 1

        # Deactivate certificate binding
        self.certificate_binder.unbind_peer(peer_id)

        self.audit_logger.log(
            event_type=AuditEventType.REVOCATION_CASCADE_COMPLETED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            zone_id=record.zone_id,
            details={"revoked_sessions": revoked_session_count, "reason": reason},
        )

        self.audit_logger.log(
            event_type=AuditEventType.FEDERATION_REVOKED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            zone_id=record.zone_id,
            details={"reason": reason, "revoked_by": revoked_by},
        )

        # Record revocation in coordinator
        if hasattr(self, "coordinator"):
            self.coordinator.record_and_propagate_revocation(
                target_type=RevocationTargetType.PEER,
                target_id=peer_id,
                reason=reason,
                engine=self,
            )

        if hasattr(self, "runtime") and self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.PEER_REVOKED,
                epoch=self.current_epoch,
                payload={"peer_id": peer_id, "reason": reason, "revoked_by": revoked_by},
            )

        return record

    # ========================================================================
    # 8. Sanitized Public Trace Emission
    # ========================================================================

    def emit_trace(
        self,
        peer_id: str,
        event_type: str,
        status: str,
        granted_scopes: Optional[List[FederationScope]] = None,
        denied_scopes: Optional[List[FederationScope]] = None,
        notes: str = "",
    ) -> SafePublicPeeringTrace:
        """Construct sanitized public trace for telemetry and memory recording."""
        reg = self.registry.get_peer(peer_id)
        peer_zone = reg.identity.zone_id if reg else "UNKNOWN_ZONE"
        operation_id = f"trace_{peer_id}_{self.current_epoch}_{int(time.time() * 1000)}"

        return SafePublicPeeringTrace(
            operation_id=operation_id,
            local_zone_id=self.local_zone_id,
            peer_id=peer_id,
            peer_zone_id=peer_zone,
            event_type=event_type,
            status=status,
            epoch=self.current_epoch,
            granted_scopes=[s.value for s in (granted_scopes or [])],
            denied_scopes=[s.value for s in (denied_scopes or [])],
            notes=notes,
        )

    # ========================================================================
    # 9. Cryptographic Peer Identity & Lifecycle (Step 30)
    # ========================================================================

    def register_cryptographic_peer(
        self,
        crypto_identity: CryptographicPeerIdentity,
        supported_features: Optional[List[str]] = None,
        capability_profile: Optional[Dict[str, Any]] = None,
    ) -> PeerRegistration:
        """
        Register a peer backed by a validated Ed25519 cryptographic identity.
        """
        key_valid, key_reason = crypto_identity.is_valid(self.current_epoch)
        if not key_valid:
            raise CrossZoneAuthorizationError(f"Cannot register peer with invalid key: {key_reason}")

        # Construct base PeerIdentity
        base_identity = PeerIdentityProvider.create_identity(
            peer_id=crypto_identity.peer_id,
            zone_id=crypto_identity.zone_id,
            organization_id=crypto_identity.organization_id,
            capability_profile=capability_profile or {},
            supported_features=supported_features or [],
            protocol_version=crypto_identity.protocol_version,
            architecture_version=crypto_identity.architecture_version,
            created_epoch=self.current_epoch,
        )

        reg = PeerRegistration(
            identity=base_identity,
            discovery_status=DiscoveryStatus.VERIFIED,
            registered_epoch=self.current_epoch,
            last_seen_epoch=self.current_epoch,
            cryptographic_identity=crypto_identity,
        )
        self.registry.register_peer(reg)

        self.audit_logger.log(
            event_type=AuditEventType.IDENTITY_PRESENTED,
            epoch=self.current_epoch,
            peer_id=crypto_identity.peer_id,
            zone_id=crypto_identity.zone_id,
            details={
                "public_fingerprint": crypto_identity.public_key.fingerprint,
                "key_state": crypto_identity.key_state.value,
            },
        )

        if hasattr(self, "runtime") and self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.PEER_REGISTERED,
                epoch=self.current_epoch,
                payload={
                    "peer_id": crypto_identity.peer_id,
                    "zone_id": crypto_identity.zone_id,
                    "public_hex": crypto_identity.public_key.public_hex,
                },
            )

        return reg

    # ========================================================================
    # 10. Bounded Challenge-Response Peer Authentication (Step 30)
    # ========================================================================

    def issue_authentication_challenge(
        self,
        target_peer_id: str,
        session_id: Optional[str] = None,
        ttl_epochs: int = 2,
    ) -> AuthChallenge:
        """
        Issue a cryptographic challenge containing a secure random nonce to target peer.
        """
        reg = self.registry.get_peer(target_peer_id)
        if not reg:
            raise CrossZoneAuthorizationError(f"Target peer '{target_peer_id}' is not registered.")

        sid = session_id or f"sess_{secrets.token_hex(8)}"
        challenge = self.authenticator.issue_challenge(
            session_id=sid,
            challenger_peer_id=self.local_peer_id,
            target_peer_id=target_peer_id,
            current_epoch=self.current_epoch,
            ttl_epochs=ttl_epochs,
        )

        return challenge

    def verify_authentication_response(
        self,
        response: AuthChallengeResponse,
        transport_type: str = "loopback",
    ) -> Tuple[bool, str, Optional[SecurePeerSession]]:
        """
        Verify the signed challenge response. Upon success, transitions peer standing
        to CRYPTOGRAPHICALLY_AUTHENTICATED and establishes a SecurePeerSession.
        """
        reg = self.registry.get_peer(response.signer_peer_id)
        if not reg or not reg.cryptographic_identity:
            self.audit_logger.log(
                event_type=AuditEventType.AUTHENTICATION_FAILED,
                epoch=self.current_epoch,
                peer_id=response.signer_peer_id,
                details={"reason": "Peer or cryptographic identity not found"},
            )
            return False, "Peer or cryptographic identity not found in registry", None

        is_valid, reason = self.authenticator.verify_response(
            response=response,
            peer_identity=reg.cryptographic_identity,
            current_epoch=self.current_epoch,
        )

        if not is_valid:
            self.audit_logger.log(
                event_type=AuditEventType.AUTHENTICATION_FAILED,
                epoch=self.current_epoch,
                peer_id=response.signer_peer_id,
                details={"reason": reason},
            )
            return False, reason, None

        # Peer successfully proved private-key possession
        session = SecurePeerSession(
            session_id=response.session_id,
            local_peer_id=self.local_peer_id,
            remote_peer_id=response.signer_peer_id,
            local_zone_id=self.local_zone_id,
            remote_zone_id=reg.identity.zone_id,
            created_epoch=self.current_epoch,
            expires_at_epoch=self.current_epoch + 50,
            transport_type=transport_type,
        )
        session.mark_authenticated(
            remote_public_key=reg.cryptographic_identity.public_key,
            authenticated_epoch=self.current_epoch,
            trust_grant_id=reg.trust_grant.grant_id if reg.trust_grant else None,
        )

        self.sessions[session.session_id] = session

        self.audit_logger.log(
            event_type=AuditEventType.PEER_AUTHENTICATED,
            epoch=self.current_epoch,
            peer_id=response.signer_peer_id,
            zone_id=reg.identity.zone_id,
            session_id=session.session_id,
            details={"status": "CRYPTOGRAPHICALLY_AUTHENTICATED"},
        )

        return True, "Authentication successful", session

    def create_secure_session(
        self,
        remote_peer_id: str,
        transport_type: str = "loopback",
        ttl_epochs: int = 50,
    ) -> SecurePeerSession:
        """
        Manually create or initialize a SecurePeerSession for an authenticated peer.
        """
        reg = self.registry.get_peer(remote_peer_id)
        if not reg:
            raise CrossZoneAuthorizationError(f"Peer '{remote_peer_id}' not registered.")

        sid = f"sess_{secrets.token_hex(8)}"
        session = SecurePeerSession(
            session_id=sid,
            local_peer_id=self.local_peer_id,
            remote_peer_id=remote_peer_id,
            local_zone_id=self.local_zone_id,
            remote_zone_id=reg.identity.zone_id,
            created_epoch=self.current_epoch,
            expires_at_epoch=self.current_epoch + ttl_epochs,
            transport_type=transport_type,
        )
        if reg.cryptographic_identity:
            session.mark_authenticated(
                remote_public_key=reg.cryptographic_identity.public_key,
                authenticated_epoch=self.current_epoch,
                trust_grant_id=reg.trust_grant.grant_id if reg.trust_grant else None,
            )

        self.sessions[sid] = session

        if hasattr(self, "runtime") and self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.SESSION_CREATED,
                epoch=self.current_epoch,
                payload={
                    "session_id": sid,
                    "remote_peer_id": remote_peer_id,
                    "expires_at_epoch": session.expires_at_epoch,
                },
            )

        return session

    def get_session(self, session_id: str) -> Optional[SecurePeerSession]:
        """Retrieve active or cached secure peer session."""
        return self.sessions.get(session_id)

    def terminate_session(self, session_id: str, reason: str = "Administrative termination") -> None:
        """
        Explicitly terminate a secure peer session, invalidating session keys and context.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise CrossZoneAuthorizationError(f"Session '{session_id}' not found.")
        session.terminate(reason=reason)
        self.audit_logger.log(
            event_type=AuditEventType.SESSION_TERMINATED,
            epoch=self.current_epoch,
            peer_id=session.remote_peer_id,
            session_id=session.session_id,
            details={"reason": reason},
        )

        if hasattr(self, "runtime") and self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.SESSION_TERMINATED,
                epoch=self.current_epoch,
                payload={"session_id": session_id, "reason": reason},
            )

    def renew_session(
        self,
        session_id: str,
        extension_epochs: int = 25,
        renewal_proof_signature: Optional[str] = None,
    ) -> SecurePeerSession:
        """
        Harden session freshness: renew an active session without authority or trust escalation.
        Enforces:
        - Session must exist and be in RENEWING / ACTIVE state.
        - Peer identity must not be revoked or expired.
        - Cryptographic key must be valid.
        - Peer trust grant must be active and unexpired.
        - Session renewal boundary cannot exceed peer trust grant expiry epoch.
        - Bounded maximum renewals and bounded lifetime.
        - Renewal proof signature verified if provided.
        - Capability scope remains strictly immutable.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise CrossZoneAuthorizationError(f"Session '{session_id}' not found.")

        reg = self.registry.get_peer(session.remote_peer_id)
        if not reg or reg.discovery_status == DiscoveryStatus.REVOKED or reg.revocation_record:
            self.audit_logger.log(
                event_type=AuditEventType.SESSION_RENEWAL_FAILED,
                epoch=self.current_epoch,
                peer_id=session.remote_peer_id,
                session_id=session_id,
                details={"reason": "Remote peer is revoked or unregistered."},
            )
            raise CrossZoneAuthorizationError("Cannot renew session: peer is revoked or unregistered.")

        crypto_id = getattr(reg, "cryptographic_identity", None)
        if not crypto_id:
            raise CrossZoneAuthorizationError("Cannot renew session: peer has no cryptographic identity.")
        key_valid, key_reason = crypto_id.is_valid(self.current_epoch)
        if not key_valid:
            self.audit_logger.log(
                event_type=AuditEventType.SESSION_RENEWAL_FAILED,
                epoch=self.current_epoch,
                peer_id=session.remote_peer_id,
                session_id=session_id,
                details={"reason": f"Cryptographic key invalid: {key_reason}"},
            )
            raise CrossZoneAuthorizationError(f"Cannot renew session: peer key invalid: {key_reason}")

        # Trust grant must be active and unexpired (SESSION_RENEWAL != TRUST_RENEWAL)
        if not reg.trust_grant or reg.trust_grant.is_expired(self.current_epoch):
            self.audit_logger.log(
                event_type=AuditEventType.SESSION_RENEWAL_FAILED,
                epoch=self.current_epoch,
                peer_id=session.remote_peer_id,
                session_id=session_id,
                details={"reason": "Peer trust grant expired or missing."},
            )
            raise CrossZoneAuthorizationError("Cannot renew session: peer trust grant is expired or missing.")

        # Session cannot be renewed beyond trust grant expiration epoch
        if session.expires_at_epoch >= reg.trust_grant.expires_epoch:
            self.audit_logger.log(
                event_type=AuditEventType.SESSION_RENEWAL_FAILED,
                epoch=self.current_epoch,
                peer_id=session.remote_peer_id,
                session_id=session_id,
                details={"reason": "Session boundary already reached trust grant expiration epoch."},
            )
            raise CrossZoneAuthorizationError("Cannot renew session beyond trust grant expiration epoch.")

        effective_extension = min(extension_epochs, reg.trust_grant.expires_epoch - session.expires_at_epoch)

        # Optional renewal proof signature verification
        if renewal_proof_signature:
            proof_payload = f"RENEW_SESSION:{session.session_id}:{session.renewal_count + 1}:{self.current_epoch}".encode("utf-8")
            try:
                sig_bytes = bytes.fromhex(renewal_proof_signature.strip())
                if not crypto_id.public_key.verify(sig_bytes, proof_payload):
                    raise SignatureVerificationError("Renewal proof signature is invalid.")
            except Exception as e:
                self.audit_logger.log(
                    event_type=AuditEventType.SESSION_RENEWAL_FAILED,
                    epoch=self.current_epoch,
                    peer_id=session.remote_peer_id,
                    session_id=session_id,
                    details={"reason": f"Invalid renewal signature: {e}"},
                )
                raise SignatureVerificationError(f"Session renewal proof signature verification failed: {e}")

        # Transition through RENEWING to ACTIVE with bounded lifetime check
        try:
            session.renew(extension_epochs=effective_extension, current_epoch=self.current_epoch)
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.SESSION_RENEWAL_FAILED,
                epoch=self.current_epoch,
                peer_id=session.remote_peer_id,
                session_id=session_id,
                details={"reason": str(e)},
            )
            raise

        self.audit_logger.log(
            event_type=AuditEventType.SESSION_RENEWED,
            epoch=self.current_epoch,
            peer_id=session.remote_peer_id,
            session_id=session.session_id,
            details={
                "new_expires_at_epoch": session.expires_at_epoch,
                "renewal_count": session.renewal_count,
            },
        )
        return session

    def rotate_peer_key(
        self,
        peer_id: str,
        new_public_key: Ed25519PublicKeyWrapper,
        rotation_proof_signature: str,
    ) -> None:
        """
        Harden Ed25519 peer identity key rotation.
        Enforces:
        - KEY_ROTATION != TRUST_GRANT
        - KEY_ROTATION != CAPABILITY_ESCALATION
        - Fails closed on revoked peer or invalid proof.
        - Retires old key and updates active sessions.
        """
        self.audit_logger.log(
            event_type=AuditEventType.KEY_ROTATION_STARTED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            details={"new_fingerprint": new_public_key.fingerprint},
        )
        reg = self.registry.get_peer(peer_id)
        if not reg or reg.discovery_status == DiscoveryStatus.REVOKED or reg.revocation_record:
            self.audit_logger.log(
                event_type=AuditEventType.KEY_ROTATION_FAILED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                details={"reason": "Cannot rotate key: peer is revoked or unregistered."},
            )
            raise KeyStateError("Cannot rotate key: peer is revoked or unregistered.")

        crypto_id = getattr(reg, "cryptographic_identity", None)
        if not crypto_id:
            self.audit_logger.log(
                event_type=AuditEventType.KEY_ROTATION_FAILED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                details={"reason": "Peer has no cryptographic identity."},
            )
            raise KeyStateError("Cannot rotate key: peer has no cryptographic identity.")

        try:
            crypto_id.rotate_key(
                new_public_key=new_public_key,
                rotation_epoch=self.current_epoch,
                signature_from_old_key=rotation_proof_signature,
            )
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.KEY_ROTATION_FAILED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                details={"reason": str(e)},
            )
            raise

        # Update active sessions remote key binding
        for sess in self.sessions.values():
            if sess.remote_peer_id == peer_id:
                sess.remote_public_key = new_public_key
                if sess.session_key_metadata:
                    sess.session_key_metadata.key_state = SessionKeyState.ROTATING
                    sess.session_key_metadata.key_state = SessionKeyState.ACTIVE

        self.audit_logger.log(
            event_type=AuditEventType.KEY_ROTATION_COMPLETED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            details={"active_fingerprint": new_public_key.fingerprint},
        )

        if hasattr(self, "runtime") and self.runtime and getattr(self.runtime, "journal", None):
            self.runtime.journal.append(
                entry_type=JournalEntryType.KEY_ROTATED,
                epoch=self.current_epoch,
                payload={"peer_id": peer_id, "new_public_hex": new_public_key.public_hex},
            )

    def rotate_peer_certificate(
        self,
        peer_id: str,
        new_certificate_metadata: CertificateMetadata,
        expected_common_name: Optional[str] = None,
        expected_san: Optional[str] = None,
    ) -> PeerCertificateBinding:
        """
        Harden TLS certificate rotation and peer identity binding.
        Enforces:
        - CERTIFICATE_ROTATION != AUTHORIZATION
        - CERTIFICATE_ROTATION != TRUST_RENEWAL
        - Replacement certificate must pass validity checks.
        - Binds new certificate fingerprint to existing peer identity.
        """
        self.audit_logger.log(
            event_type=AuditEventType.CERTIFICATE_ROTATION_STARTED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            details={"new_fingerprint": new_certificate_metadata.fingerprint},
        )
        reg = self.registry.get_peer(peer_id)
        if not reg or reg.discovery_status == DiscoveryStatus.REVOKED or reg.revocation_record:
            self.audit_logger.log(
                event_type=AuditEventType.CERTIFICATE_ROTATION_FAILED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                details={"reason": "Cannot rotate certificate: peer is revoked or unregistered."},
            )
            raise CertificateValidationError("Cannot rotate certificate: peer is revoked or unregistered.")

        cert_valid, cert_reason = validate_certificate(
            new_certificate_metadata,
            revocation_registry=self.certificate_revocation_registry,
            current_time=time.time(),
        )
        if not cert_valid:
            self.audit_logger.log(
                event_type=AuditEventType.CERTIFICATE_ROTATION_FAILED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                details={"reason": cert_reason, "fingerprint": new_certificate_metadata.fingerprint},
            )
            raise CertificateValidationError(f"Certificate rotation rejected: {cert_reason}")

        binding = self.certificate_binder.bind_peer(
            peer_id=peer_id,
            certificate_fingerprint=new_certificate_metadata.fingerprint,
            expected_common_name=expected_common_name,
            expected_san=expected_san,
            epoch=self.current_epoch,
        )

        self.audit_logger.log(
            event_type=AuditEventType.CERTIFICATE_ROTATION_COMPLETED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            details={"fingerprint": new_certificate_metadata.fingerprint},
        )
        return binding

    # ========================================================================
    # 11. Wire-Level Request Authorization & Execution (Steps 30 & 31)
    # ========================================================================

    def bind_peer_certificate(
        self,
        peer_id: str,
        certificate_fingerprint: str,
        expected_common_name: Optional[str] = None,
        expected_san: Optional[str] = None,
    ) -> PeerCertificateBinding:
        """Explicitly bind a registered peer identity to an authorized TLS certificate fingerprint."""
        binding = self.certificate_binder.bind_peer(
            peer_id=peer_id,
            certificate_fingerprint=certificate_fingerprint,
            expected_common_name=expected_common_name,
            expected_san=expected_san,
            epoch=self.current_epoch,
        )
        self.audit_logger.log(
            event_type=AuditEventType.PEER_BINDING_VERIFIED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            details={"certificate_fingerprint": certificate_fingerprint},
        )
        return binding

    def authorize_and_execute_wire_envelope(
        self,
        envelope: WireEnvelope,
        capability_gate: Optional[CapabilityGate] = None,
        capability_context: Optional[CapabilityContext] = None,
        peer_certificate_metadata: Optional[CertificateMetadata] = None,
    ) -> WireEnvelope:
        """
        Authoritatively validate and execute an incoming signed WireEnvelope.
        Enforces:
        1. Envelope schema, size, and payload digest
        2. TLS transport certificate validity & peer identity binding (Step 31)
        3. Remote peer registration and cryptographic key validity
        4. Ed25519 signature verification against registered remote public key
        5. Session state, epoch expiration, and replay protection
        6. Tenant boundary isolation via CrossZoneIsolationGuard
        7. Scope authorization and CapabilityGate mediation
        8. Frozen neural core weight immutability (ΔW = 0)
        Returns a signed response WireEnvelope.
        """
        pre_hash = self._compute_weight_hash()

        try:
            # 1. Validate envelope framing & digest
            envelope.validate()

            # 1b. Validate TLS peer certificate & identity binding if present (Step 31)
            if peer_certificate_metadata is not None:
                cert_valid, cert_reason = validate_certificate(
                    peer_certificate_metadata,
                    revocation_registry=self.certificate_revocation_registry,
                    current_time=time.time(),
                )
                if not cert_valid:
                    self.audit_logger.log(
                        event_type=AuditEventType.CERTIFICATE_VALIDATION_FAILED,
                        epoch=self.current_epoch,
                        peer_id=envelope.sender_peer_id,
                        details={"reason": cert_reason, "fingerprint": peer_certificate_metadata.fingerprint},
                    )
                    raise CertificateValidationError(f"Incoming wire envelope rejected: invalid peer certificate: {cert_reason}")

                self.audit_logger.log(
                    event_type=AuditEventType.CERTIFICATE_VALIDATED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"fingerprint": peer_certificate_metadata.fingerprint},
                )

                # Verify peer identity binding
                bind_valid, bind_reason = self.certificate_binder.verify_binding(
                    envelope.sender_peer_id,
                    peer_certificate_metadata,
                )
                if not bind_valid:
                    self.audit_logger.log(
                        event_type=AuditEventType.PEER_BINDING_FAILED,
                        epoch=self.current_epoch,
                        peer_id=envelope.sender_peer_id,
                        details={"reason": bind_reason, "fingerprint": peer_certificate_metadata.fingerprint},
                    )
                    raise PeerBindingMismatchError(f"Incoming wire envelope rejected: {bind_reason}")

                self.audit_logger.log(
                    event_type=AuditEventType.PEER_BINDING_VERIFIED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"fingerprint": peer_certificate_metadata.fingerprint},
                )

            # 2. Check peer standing in registry
            reg = self.registry.get_peer(envelope.sender_peer_id)

            if not reg:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"reason": "Sender peer not registered"},
                )
                raise CrossZoneAuthorizationError(f"Peer '{envelope.sender_peer_id}' is not registered.")

            if reg.discovery_status == DiscoveryStatus.REVOKED or reg.revocation_record:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"reason": "Sender peer has been revoked"},
                )
                raise CrossZoneAuthorizationError(f"Peer '{envelope.sender_peer_id}' has been revoked.")

            # 3. Cryptographic identity & key validity check
            crypto_id = getattr(reg, "cryptographic_identity", None)
            if not crypto_id:
                raise CrossZoneAuthorizationError(
                    f"Peer '{envelope.sender_peer_id}' has no cryptographic identity registered."
                )

            key_valid, key_reason = crypto_id.is_valid(self.current_epoch)
            if not key_valid:
                self.audit_logger.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"reason": key_reason},
                )
                raise CrossZoneAuthorizationError(f"Peer cryptographic key invalid: {key_reason}")

            # Step 32: Check retired key signature rejection
            is_retired_sig = False
            if hasattr(crypto_id, "retired_keys"):
                for old_k in crypto_id.retired_keys:
                    if envelope.verify_signature(old_k):
                        is_retired_sig = True
                        break
            if is_retired_sig or (hasattr(crypto_id, "is_key_retired") and envelope.payload.get("key_fingerprint") and crypto_id.is_key_retired(envelope.payload.get("key_fingerprint"))):
                self.audit_logger.log(
                    event_type=AuditEventType.AUTHENTICATION_FAILED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"reason": "WireEnvelope signed by retired key rejected."},
                )
                raise SignatureVerificationError("WireEnvelope Ed25519 signature from retired key rejected.")

            # 4. Verify cryptographic signature against current active key
            if not envelope.verify_signature(crypto_id.public_key):
                self.audit_logger.log(
                    event_type=AuditEventType.AUTHENTICATION_FAILED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"reason": "Invalid Ed25519 signature on wire envelope"},
                )
                raise SignatureVerificationError("WireEnvelope Ed25519 signature verification failed.")

            # 5. Verify session & replay protection
            session = self.sessions.get(envelope.session_id)
            if not session:
                self.audit_logger.log(
                    event_type=AuditEventType.STALE_AUTHORIZATION_DENIED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    details={"reason": f"Session '{envelope.session_id}' not found."},
                )
                raise CrossZoneAuthorizationError(f"Session '{envelope.session_id}' not found.")

            sess_valid, sess_reason = session.is_active(self.current_epoch)
            if not sess_valid:
                self.audit_logger.log(
                    event_type=AuditEventType.STALE_AUTHORIZATION_DENIED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    session_id=session.session_id,
                    details={"reason": f"Session inactive: {sess_reason}"},
                )
                raise CrossZoneAuthorizationError(f"Session '{envelope.session_id}' inactive: {sess_reason}")

            if session.remote_peer_id != envelope.sender_peer_id or session.local_peer_id != envelope.receiver_peer_id:
                raise CrossZoneAuthorizationError("Envelope peer bindings do not match session.")

            # Sequence number monotonicity check if sequence_number present
            seq_num = envelope.payload.get("sequence_number")
            if seq_num is not None and not session.record_and_check_sequence(seq_num):
                self.audit_logger.log(
                    event_type=AuditEventType.REPLAY_ATTACK_DETECTED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    session_id=session.session_id,
                    details={"out_of_order_sequence": seq_num, "expected_min": session.last_seen_sequence_number + 1},
                )
                raise ReplayAttackError(f"Out-of-order or duplicate sequence number: {seq_num}")

            # Replay protection check
            if not session.record_and_check_message_id(envelope.message_id):
                self.audit_logger.log(
                    event_type=AuditEventType.REPLAY_ATTACK_DETECTED,
                    epoch=self.current_epoch,
                    peer_id=envelope.sender_peer_id,
                    session_id=session.session_id,
                    details={"replayed_message_id": envelope.message_id},
                )
                raise ReplayAttackError(f"Replay attack detected: Message ID '{envelope.message_id}' was already seen.")

            # Check message expiration
            if self.current_epoch > envelope.expires_epoch:
                raise CrossZoneAuthorizationError(
                    f"Envelope expired at epoch {envelope.expires_epoch} (current {self.current_epoch})."
                )

            # 6. Execute message payload based on message_type
            response_payload: Dict[str, Any] = {}
            resp_type = MessageType.CAPABILITY_RESPONSE

            if envelope.message_type == MessageType.HEARTBEAT:
                resp_type = MessageType.HEARTBEAT
                response_payload = {"status": "PONG", "current_epoch": self.current_epoch}

            elif envelope.message_type == MessageType.CAPABILITY_REQUEST:
                resp_type = MessageType.CAPABILITY_RESPONSE
                raw_scope = envelope.payload.get("requested_scope", FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION.value)
                requested_scope = FederationScope(raw_scope)
                target_tenant_id = envelope.payload.get("target_tenant_id", "default_tenant")
                peer_tenant_id = envelope.payload.get("peer_tenant_id", "default_tenant")
                peer_zone_id = envelope.payload.get("peer_zone_id", reg.identity.zone_id)
                cap_req_dict = envelope.payload.get("capability_request")

                cap_req = None
                if cap_req_dict:
                    cap_req = CapabilityRequest(
                        capability_id=cap_req_dict["capability_id"],
                        parameters=cap_req_dict.get("parameters", {}),
                        caller_id=envelope.sender_peer_id,
                    )

                # Authorize via existing CapabilityGate / isolation pipeline
                authorized, sanitized_result, rationale = self.authorize_cross_zone_request(
                    peer_id=envelope.sender_peer_id,
                    peer_zone_id=peer_zone_id,
                    peer_tenant_id=peer_tenant_id,
                    target_tenant_id=target_tenant_id,
                    session_id=envelope.session_id,
                    requested_scope=requested_scope,
                    payload=envelope.payload.get("data", {}),
                    capability_request=cap_req,
                    capability_context=capability_context,
                )

                response_payload = {
                    "authorized": authorized,
                    "result": sanitized_result,
                    "rationale": rationale,
                    "epoch": self.current_epoch,
                }

            elif envelope.message_type == MessageType.REVOCATION:
                resp_type = MessageType.REVOCATION
                reason = envelope.payload.get("reason", "Revocation requested over wire")
                rec = self.revoke_peer(envelope.sender_peer_id, reason=reason, revoked_by=f"peer:{envelope.sender_peer_id}")
                response_payload = {"revoked": True, "record": rec.to_dict()}

            else:
                resp_type = MessageType.ERROR_RESPONSE
                response_payload = {"error": f"Unsupported wire message type: {envelope.message_type.value}"}

            # 7. Construct and sign response envelope
            resp_envelope = WireEnvelope(
                protocol_version=envelope.protocol_version,
                message_type=resp_type,
                message_id=f"msg_{secrets.token_hex(8)}",
                session_id=envelope.session_id,
                sender_peer_id=self.local_peer_id,
                receiver_peer_id=envelope.sender_peer_id,
                created_epoch=self.current_epoch,
                expires_epoch=self.current_epoch + 10,
                payload=response_payload,
            )
            resp_envelope.sign(self.local_private_key)

            # 8. Verify neural core immutability
            self._verify_weight_invariants(pre_hash)

            return resp_envelope

        except Exception:
            self._verify_weight_invariants(pre_hash)
            raise

    # ========================================================================
    # 13. Step 33 Distributed Federation Coordination Helpers
    # ========================================================================

    def initiate_coordination_handshake(
        self,
        remote_engine: "CrossZoneFederationEngine",
    ) -> FederationHandshakeResponse:
        """Initiate coordination handshake with another CrossZoneFederationEngine."""
        revocations = list(self.revocation_manager._revocations.values()) if hasattr(self.revocation_manager, "_revocations") else []
        return self.coordinator.initiate_handshake(
            remote_coordinator=remote_engine.coordinator,
            registry=self.registry,
            revocations=revocations,
        )

    def synchronize_replay_with_engine(
        self,
        remote_engine: "CrossZoneFederationEngine",
        session_id: str,
    ) -> Dict[str, Any]:
        """Synchronize replay state for a session with a remote engine."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found locally.")
        session = self.sessions[session_id]
        remote_session = remote_engine.sessions.get(session_id)
        return self.coordinator.synchronize_replay(
            remote_coordinator=remote_engine.coordinator,
            session=session,
            remote_session=remote_session,
        )

    def synchronize_trust_with_engine(
        self,
        remote_engine: "CrossZoneFederationEngine",
    ) -> Dict[str, Any]:
        """Synchronize trust state claims with a remote engine."""
        return self.coordinator.synchronize_trust(
            remote_coordinator=remote_engine.coordinator,
            local_registry=self.registry,
            remote_registry=remote_engine.registry,
            local_policy=remote_engine.policy,
        )

    def propagate_revocation_to_engines(
        self,
        target_type: RevocationTargetType,
        target_id: str,
        reason: str,
        remote_engines: List["CrossZoneFederationEngine"],
    ) -> RevocationSyncRecord:
        """Propagate a revocation event to remote engines."""
        remote_coords = [re.coordinator for re in remote_engines]
        return self.coordinator.record_and_propagate_revocation(
            target_type=target_type,
            target_id=target_id,
            reason=reason,
            engine=self,
            remote_coordinators=remote_coords,
        )

    def compute_security_state_digest(
        self,
        session_id: Optional[str] = None,
    ) -> FederationSecurityStateDigest:
        """Compute composite security state digest for this engine."""
        session = self.sessions.get(session_id) if session_id else None
        revocations = list(self.revocation_manager._revocations.values()) if hasattr(self.revocation_manager, "_revocations") else []
        return self.coordinator.compute_state_digest(
            session=session,
            registry=self.registry,
            revocations=revocations,
        )


FederationEngine = CrossZoneFederationEngine
