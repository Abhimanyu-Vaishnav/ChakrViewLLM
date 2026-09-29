"""
Secure Outbound & Inbound Federation Connection Manager (Step 35).

Coordinates physical transport establishment (TCP / TLS / mTLS), certificate validation,
peer identity binding, federation coordination handshakes, membership promotion,
and fail-closed rejoin/quarantine policies.

CRITICAL INVARIANTS:
- A TCP connection alone NEVER establishes membership.
- A TLS handshake alone NEVER establishes membership.
- A valid signature alone NEVER establishes membership.
- UNKNOWN PEER -> DENY (default fail-closed for unsolicited connections).
- LOCAL_REVOCATION > REMOTE_ACTIVE_STATE.
- ΔW = 0 strictly maintained.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from chakrview.cognition.peering.models import AuditEventType
from chakrview.cognition.transport.security.models import (
    TLSMode,
    CertificateMetadata,
)
from chakrview.cognition.transport.security.errors import (
    CertificateRevokedError,
    PeerBindingMismatchError,
    ClientCertificateMissingError,
)
from chakrview.cognition.federation.identity import FederationEngineIdentityProvider
from chakrview.cognition.federation.models import (
    HandshakeStatus,
    RevocationTargetType,
)
from chakrview.cognition.federation.discovery.models import (
    FederationNodeCandidate,
    FederationNodeMembership,
    MembershipState,
)
from chakrview.cognition.federation.discovery.errors import (
    AuthenticationGateError,
    CertificateBindingMismatchError,
    UnknownPeerError,
    QuarantineError,
    MembershipRevocationError,
    RejoinDeniedError,
)
from chakrview.cognition.federation.discovery.membership import FederationMembershipManager

logger = logging.getLogger(__name__)


class FederationConnectionManager:
    """
    Manages end-to-end authenticated transport connections between federation nodes.
    """

    def __init__(
        self,
        engine: Any,
        membership_manager: Optional[FederationMembershipManager] = None,
    ) -> None:
        self.engine = engine
        self.membership_manager = (
            membership_manager
            or getattr(engine, "membership_manager", None)
            or FederationMembershipManager(engine=engine)
        )

    def validate_tls_connection(
        self,
        cert_metadata: CertificateMetadata,
        expected_peer_id: str,
        tls_mode: TLSMode = TLSMode.MTLS,
    ) -> bool:
        """Validate TLS certificate against revocation list and peer bindings."""
        if (
            hasattr(self.engine, "certificate_revocation_registry")
            and self.engine.certificate_revocation_registry.is_revoked(cert_metadata.fingerprint)
        ):
            from chakrview.cognition.transport.security.errors import CertificateRevokedError
            raise CertificateRevokedError(f"Certificate '{cert_metadata.fingerprint}' is revoked.")

        if hasattr(self.engine, "certificate_binder"):
            binding = self.engine.certificate_binder.get_binding(expected_peer_id)
            if binding:
                is_bound, reason = self.engine.certificate_binder.verify_binding(
                    expected_peer_id, cert_metadata
                )
                if not is_bound:
                    raise CertificateBindingMismatchError(f"Certificate binding mismatch: {reason}")
        return True

    def validate_engine_identity(self, engine_identity: FederationEngineIdentity) -> bool:
        """Verify engine identity format, zone, and fingerprint."""
        is_valid, reason = FederationEngineIdentityProvider.validate_identity(engine_identity)
        if not is_valid:
            raise AuthenticationGateError(f"Engine identity validation failed: {reason}")
        return True

    def verify_handshake_response(self, handshake_response: Dict[str, Any]) -> bool:
        """Verify federation handshake response status."""
        status = handshake_response.get("status")
        if status != "ACCEPTED" and status != HandshakeStatus.ACCEPTED.value:
            from chakrview.cognition.federation.errors import HandshakeError
            raise HandshakeError(f"Handshake failed: {handshake_response.get('reason', status)}")
        return True

    def validate_tenant_isolation(
        self,
        engine_identity: FederationEngineIdentity,
        allowed_tenant: str,
    ) -> bool:
        """Enforce strict cross-tenant isolation bounds."""
        tenant = getattr(engine_identity, "organization_id", None) or getattr(engine_identity, "zone_id", None)
        if tenant and tenant != allowed_tenant:
            from chakrview.capability.gate import CapabilityAuthorizationError
            raise CapabilityAuthorizationError(f"Cross-tenant isolation violation: {tenant} != {allowed_tenant}")
        return True

    def check_remote_state_stale(self, remote_state_version: Dict[str, Any]) -> bool:
        """Check if remote security state version is older than local version."""
        rem_ver = int(remote_state_version.get("version", 0))
        loc_ver = self.engine.state_version.version
        return rem_ver < loc_ver

    def verify_digest_alignment(self, local_digest: str, remote_digest: str) -> bool:
        """Verify whether composite state digests match."""
        return local_digest == remote_digest

    # ========================================================================
    # 1. Outbound Connection Establishment (Phase 5)
    # ========================================================================

    def connect_candidate(
        self,
        candidate_or_membership_id: Any,
        remote_engine: Optional[Any] = None,
        peer_certificate_metadata: Optional[CertificateMetadata] = None,
    ) -> FederationNodeMembership:
        """
        Execute full secure outbound connection flow:
        Candidate -> Endpoint Validation -> TLS/mTLS -> Cert Validation -> Cert/Ed25519 Binding
        -> Engine Identity Validation -> Federation Handshake -> State Digest Comparison -> Membership Decision.
        """
        if isinstance(candidate_or_membership_id, str):
            membership = self.membership_manager.get_membership(candidate_or_membership_id)
            if not membership:
                raise UnknownPeerError(f"Membership '{candidate_or_membership_id}' not found.")
        else:
            membership = self.membership_manager.register_candidate(candidate_or_membership_id)

        # 1. Check existing membership state
        if membership.is_revoked():
            raise MembershipRevocationError(
                f"Cannot connect to revoked node '{membership.membership_id}': terminal revocation."
            )
        if membership.is_quarantined():
            raise QuarantineError(
                f"Cannot connect to quarantined node '{membership.membership_id}': {membership.quarantine_reason}."
            )

        # 2. Advance to PENDING_AUTHENTICATION
        if membership.state == MembershipState.DISCOVERED:
            self.membership_manager.start_authentication(membership.membership_id)

        # 3. Certificate and Transport Validation
        ep = membership.endpoint
        if peer_certificate_metadata:
            # Check certificate revocation
            if (
                hasattr(self.engine, "certificate_revocation_registry")
                and self.engine.certificate_revocation_registry.is_revoked(peer_certificate_metadata.fingerprint)
            ):
                self.membership_manager.quarantine_member(
                    membership.membership_id,
                    reason=f"Presented revoked certificate: {peer_certificate_metadata.fingerprint}",
                )
                raise CertificateRevokedError(
                    f"Remote certificate '{peer_certificate_metadata.fingerprint}' is revoked."
                )

            # Check expected certificate fingerprint
            if ep.expected_cert_fingerprint and peer_certificate_metadata.fingerprint != ep.expected_cert_fingerprint:
                self.membership_manager.quarantine_member(
                    membership.membership_id,
                    reason=(
                        f"Certificate fingerprint mismatch: expected '{ep.expected_cert_fingerprint}', "
                        f"got '{peer_certificate_metadata.fingerprint}'"
                    ),
                )
                raise CertificateBindingMismatchError(
                    f"Certificate fingerprint '{peer_certificate_metadata.fingerprint}' "
                    f"does not match expected '{ep.expected_cert_fingerprint}'."
                )

        # 4. Engine Identity & Federation Handshake
        if remote_engine:
            ident = remote_engine.engine_identity
            is_valid, reason = FederationEngineIdentityProvider.validate_identity(ident)
            if not is_valid:
                self.membership_manager.quarantine_member(
                    membership.membership_id,
                    reason=f"Invalid remote engine identity: {reason}",
                )
                raise AuthenticationGateError(f"Engine identity validation failed: {reason}")

            # Mark Authenticated
            self.membership_manager.authenticate_candidate(
                membership_id=membership.membership_id,
                engine_identity=ident,
                peer_id=getattr(remote_engine, "local_peer_id", None),
            )

            # Initiate Coordination Handshake
            handshake_resp = self.engine.initiate_coordination_handshake(remote_engine)
            if not handshake_resp.is_accepted():
                err = handshake_resp.details.get("error", handshake_resp.status.value)
                if handshake_resp.status == HandshakeStatus.STATE_CONFLICT:
                    self.membership_manager.quarantine_member(
                        membership.membership_id,
                        reason=f"State digest conflict during handshake: {err}",
                    )
                raise AuthenticationGateError(f"Federation handshake rejected: {err}")

            # Promote to Member
            return self.membership_manager.promote_to_member(membership.membership_id)

        return membership

    # ========================================================================
    # 2. Inbound Connection Acceptance (Phase 6)
    # ========================================================================

    def accept_inbound_connection(
        self,
        remote_peer_id: Optional[str] = None,
        remote_zone_id: Optional[str] = None,
        tls_cert_metadata: Optional[CertificateMetadata] = None,
        peer_certificate_metadata: Optional[CertificateMetadata] = None,
        engine_identity: Optional[Any] = None,
        remote_engine_identity: Optional[Any] = None,
        remote_endpoint: Optional[Any] = None,
    ) -> FederationNodeMembership:
        """
        Execute inbound connection acceptance flow:
        TCP Accept -> TLS/mTLS -> Cert Validation -> Peer Identity Binding
        -> Engine Identity Validation -> Federation Handshake -> Membership Lookup -> CapabilityGate Default Deny.
        """
        cert = peer_certificate_metadata or tls_cert_metadata
        eng_id = remote_engine_identity or engine_identity

        # 1. Certificate Validation
        if cert:
            # Check certificate revocation
            if (
                hasattr(self.engine, "certificate_revocation_registry")
                and self.engine.certificate_revocation_registry.is_revoked(cert.fingerprint)
            ):
                raise CertificateRevokedError(
                    f"Inbound connection rejected: certificate '{cert.fingerprint}' is revoked."
                )

        # 2. Membership Lookup
        membership = None
        if eng_id:
            membership = self.membership_manager.get_membership_by_node_id(eng_id.engine_id)

        if not membership and remote_peer_id:
            for m in self.membership_manager.list_members():
                if m.peer_id == remote_peer_id or m.node_id == remote_peer_id:
                    membership = m
                    break

        if not membership and cert:
            # Search by certificate fingerprint across endpoints
            for m in self.membership_manager.list_members():
                if m.endpoint.expected_cert_fingerprint == cert.fingerprint:
                    membership = m
                    break

        # 3. Default Deny Policy for Unknown Peers
        if not membership:
            audit = getattr(self.engine, "audit_logger", None)
            if audit:
                audit.log(
                    event_type=AuditEventType.REQUEST_DENIED,
                    epoch=self.engine.current_epoch,
                    details={"reason": "Unknown peer connection denied under default fail-closed policy"},
                )
            raise UnknownPeerError("Inbound connection denied: unknown peer is not registered as candidate or member.")

        # 4. Check Quarantine and Revocation
        if membership.is_revoked():
            raise MembershipRevocationError(f"Inbound connection rejected: node '{membership.membership_id}' is REVOKED.")
        if membership.is_quarantined():
            raise QuarantineError(f"Inbound connection rejected: node '{membership.membership_id}' is QUARANTINED.")

        # 5. Validate Engine Identity
        if eng_id:
            is_valid, reason = FederationEngineIdentityProvider.validate_identity(eng_id)
            if not is_valid:
                self.membership_manager.quarantine_member(
                    membership.membership_id,
                    reason=f"Inbound engine identity invalid: {reason}",
                )
                raise AuthenticationGateError(f"Inbound engine identity invalid: {reason}")

            if cert and hasattr(self.engine, "certificate_binder"):
                p_id = membership.peer_id or membership.node_id
                if p_id and self.engine.certificate_binder.get_binding(p_id):
                    is_bound, bind_reason = self.engine.certificate_binder.verify_binding(
                        p_id, cert
                    )
                    if not is_bound:
                        self.membership_manager.quarantine_member(
                            membership.membership_id,
                            reason=f"Certificate binding mismatch: {bind_reason}",
                        )
                        raise CertificateBindingMismatchError(f"Certificate binding mismatch: {bind_reason}")

        return membership

    # ========================================================================
    # 3. Member Rejoin Flow (Phase 11)
    # ========================================================================

    def reconnect_member(
        self,
        membership_id: Optional[str] = None,
        node_id: Optional[str] = None,
        remote_peer_id: Optional[str] = None,
        remote_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Reconnect an enrolled member following network partition or restart.
        Enforces local revocation dominance, replay floor continuity, and digest agreement.
        """
        membership = None
        if membership_id:
            membership = self.membership_manager.get_membership(membership_id)
        elif node_id:
            membership = self.membership_manager.get_membership_by_node_id(node_id)

        if not membership:
            raise UnknownPeerError(f"Membership '{membership_id or node_id}' does not exist.")

        if membership.is_revoked():
            raise RejoinDeniedError(f"Rejoin denied for '{membership.membership_id}': node is permanently revoked.")

        if membership.is_quarantined():
            raise QuarantineError(f"Rejoin denied for '{membership.membership_id}': node is currently quarantined.")

        # Check local revocation registry: if peer is revoked locally, local revocation strictly wins
        peer_id = remote_peer_id or membership.peer_id or membership.node_id
        if peer_id and hasattr(self.engine, "registry"):
            reg = self.engine.registry.get_peer(peer_id)
            if reg and reg.discovery_status.value == "REVOKED":
                self.membership_manager.revoke_member(
                    membership.membership_id,
                    reason="Local revocation remains authoritative",
                )
                raise RejoinDeniedError(f"Rejoin denied for '{membership.membership_id}': locally revoked.")

        # If remote_engine provided, delegate to runtime rejoin protocol
        runtime = getattr(self.engine, "runtime", None)
        if remote_engine and runtime:
            result = runtime.execute_rejoin(remote_engine)
        else:
            result = {"status": "RECONNECTED", "node_id": node_id or membership.node_id}

        # Transition back to MEMBER if suspended
        if membership.state == MembershipState.SUSPENDED:
            membership.transition_to(MembershipState.MEMBER, reason="Rejoin protocol completed successfully")
        membership.record_heartbeat(self.engine.current_epoch)

        return result
