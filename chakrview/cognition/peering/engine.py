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
    validate_certificate,
    CertificateValidationError,
    PeerBindingMismatchError,
)



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

        expired_peer_ids = self.registry.expire_peers(self.current_epoch)
        for peer_id in expired_peer_ids:
            self.audit_logger.log(
                event_type=AuditEventType.FEDERATION_EXPIRED,
                epoch=self.current_epoch,
                peer_id=peer_id,
                zone_id=None,
                details={"reason": f"Trust grant expired at epoch {self.current_epoch}."},
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
        Administratively revoke peer trust and invalidate active grants immediately.
        """
        record = self.registry.revoke_peer(
            peer_id=peer_id,
            reason=reason,
            revoked_epoch=self.current_epoch,
            revoked_by=revoked_by,
        )

        self.audit_logger.log(
            event_type=AuditEventType.FEDERATION_REVOKED,
            epoch=self.current_epoch,
            peer_id=peer_id,
            zone_id=record.zone_id,
            details={"reason": reason, "revoked_by": revoked_by},
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
        return session

    def get_session(self, session_id: str) -> Optional[SecurePeerSession]:
        """Retrieve active or cached secure peer session."""
        return self.sessions.get(session_id)

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

            # 4. Verify cryptographic signature
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
                raise CrossZoneAuthorizationError(f"Session '{envelope.session_id}' not found.")

            sess_valid, sess_reason = session.is_active(self.current_epoch)
            if not sess_valid:
                raise CrossZoneAuthorizationError(f"Session '{envelope.session_id}' inactive: {sess_reason}")

            if session.remote_peer_id != envelope.sender_peer_id or session.local_peer_id != envelope.receiver_peer_id:
                raise CrossZoneAuthorizationError("Envelope peer bindings do not match session.")

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
