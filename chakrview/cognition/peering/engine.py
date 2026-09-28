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


class CrossZoneAuthorizationError(PermissionError):
    """Raised when a cross-zone request fails trust, policy, or isolation checks."""
    pass


class WeightMutationDetectedError(RuntimeError):
    """Raised when runtime neural weights mutate during cross-zone operations."""
    pass


class CrossZoneFederationEngine:
    """
    Coordinates multi-zone peer discovery, attestation, trust negotiation,
    capability gating, and bounded auditable execution.
    """

    def __init__(
        self,
        local_zone_id: str,
        policy: Optional[CrossZoneFederationPolicy] = None,
        capability_gate: Optional[CapabilityGate] = None,
        model: Optional[torch.nn.Module] = None,
        initial_epoch: int = 1,
    ) -> None:
        self.local_zone_id = local_zone_id
        self.policy = policy or CrossZoneFederationPolicy()
        self.capability_gate = capability_gate
        self.model = model
        self.current_epoch = initial_epoch

        self.revocation_manager = RevocationManager()
        self.registry = PeerRegistry(revocation_manager=self.revocation_manager)
        self.discovery_manager = PeerDiscoveryManager(policy=self.policy)
        self.negotiator = TrustNegotiator(local_zone_id=self.local_zone_id, policy=self.policy)
        self.audit_logger = BoundedAuditLogger()

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
