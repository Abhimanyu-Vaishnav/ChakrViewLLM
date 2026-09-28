"""
Deterministic Trust and Scope Negotiation for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOMS:
1. DETERMINISTIC NEGOTIATION:
   Identical local policy, peer declaration, and system state MUST produce
   identical negotiation agreements.
2. NEGOTIATED TRUST IS STRICTLY BOUNDED:
   Only explicitly requested and policy-permitted scopes are granted.
   Unrequested scopes are never added. Prohibited scopes are unconditionally rejected.
"""

from typing import Dict, List, Optional, Tuple, Set
import uuid

from chakrview.cognition.peering.models import (
    PeerDeclaration,
    NegotiationAgreement,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    FederationScope,
    DEFAULT_TRUST_TTL_EPOCHS,
)
from chakrview.cognition.peering.identity import PeerIdentityProvider
from chakrview.cognition.peering.attestation import PeerAttestationVerifier
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.trust import TrustModel


class TrustNegotiator:
    """
    Executes deterministic trust negotiation between a local zone and a peered zone.
    """

    def __init__(self, local_zone_id: str, policy: Optional[CrossZoneFederationPolicy] = None) -> None:
        self.local_zone_id = local_zone_id
        self.policy = policy or CrossZoneFederationPolicy()

    def negotiate(
        self,
        declaration: PeerDeclaration,
        current_epoch: int = 1,
    ) -> NegotiationAgreement:
        """
        Evaluate a peer's declaration against local policy and derive a bounded agreement.
        Guaranteed to be deterministic.
        """
        peer = declaration.identity
        agreement_id = f"agr_{peer.peer_id}_{current_epoch}"
        rejected_scopes: List[FederationScope] = []
        accepted_scopes: List[FederationScope] = []
        rejection_reasons: Dict[str, str] = {}

        # 1. Identity validation
        if not PeerIdentityProvider.verify_fingerprint(peer):
            rejection_reasons["IDENTITY"] = "Peer identity fingerprint verification failed."
            return NegotiationAgreement(
                agreement_id=agreement_id,
                local_zone_id=self.local_zone_id,
                peer_id=peer.peer_id,
                peer_zone_id=peer.zone_id,
                trust_grant=None,
                accepted_scopes=[],
                rejected_scopes=list(declaration.requested_scopes),
                rejection_reasons=rejection_reasons,
                epoch=current_epoch,
            )

        # 2. Peer admissibility under local policy
        admissible, admit_reason = self.policy.is_peer_admissible(peer)
        if not admissible:
            rejection_reasons["POLICY_ADMISSION"] = admit_reason
            return NegotiationAgreement(
                agreement_id=agreement_id,
                local_zone_id=self.local_zone_id,
                peer_id=peer.peer_id,
                peer_zone_id=peer.zone_id,
                trust_grant=None,
                accepted_scopes=[],
                rejected_scopes=list(declaration.requested_scopes),
                rejection_reasons=rejection_reasons,
                epoch=current_epoch,
            )

        # 3. Attestation verification
        att_valid, att_reason = PeerAttestationVerifier.verify_attestation(
            declaration.attestation,
            expected_peer_id=peer.peer_id,
            expected_zone_id=peer.zone_id,
        )
        if not att_valid:
            rejection_reasons["ATTESTATION"] = att_reason
            return NegotiationAgreement(
                agreement_id=agreement_id,
                local_zone_id=self.local_zone_id,
                peer_id=peer.peer_id,
                peer_zone_id=peer.zone_id,
                trust_grant=None,
                accepted_scopes=[],
                rejected_scopes=list(declaration.requested_scopes),
                rejection_reasons=rejection_reasons,
                epoch=current_epoch,
            )

        # 4. Scope-by-scope evaluation under default-deny policy
        for scope in declaration.requested_scopes:
            scope_ok, scope_reason = self.policy.evaluate_scope_request(peer, scope)
            if scope_ok:
                accepted_scopes.append(scope)
            else:
                rejected_scopes.append(scope)
                rejection_reasons[scope.value] = scope_reason

        # 5. Derive bounded trust level based on accepted scopes
        trust_grant: Optional[TrustGrant] = None
        if accepted_scopes:
            if FederationScope.ALLOW_COGNITIVE_TASK_DELEGATION in accepted_scopes:
                trust_level = TrustLevel.FEDERATED
            elif (
                FederationScope.ALLOW_EVIDENCE_EXCHANGE in accepted_scopes
                or FederationScope.ALLOW_VERIFICATION in accepted_scopes
            ):
                trust_level = TrustLevel.LIMITED_TRUST
            else:
                trust_level = TrustLevel.IDENTIFIED

            grant_id = f"grant_{peer.peer_id}_{current_epoch}"
            expires_epoch = current_epoch + self.policy.default_ttl_epochs

            trust_grant = TrustModel.create_grant(
                grant_id=grant_id,
                issuer_zone_id=self.local_zone_id,
                subject_peer_id=peer.peer_id,
                subject_zone_id=peer.zone_id,
                trust_level=trust_level,
                permitted_scopes=accepted_scopes,
                issued_epoch=current_epoch,
                expires_epoch=expires_epoch,
                policy_constraints={
                    "max_delegation_depth": self.policy.max_delegation_depth,
                    "default_deny": True,
                },
                reason=f"Negotiation successful: {len(accepted_scopes)} scopes accepted.",
                provenance=f"local_zone_{self.local_zone_id}",
            )

        return NegotiationAgreement(
            agreement_id=agreement_id,
            local_zone_id=self.local_zone_id,
            peer_id=peer.peer_id,
            peer_zone_id=peer.zone_id,
            trust_grant=trust_grant,
            accepted_scopes=accepted_scopes,
            rejected_scopes=rejected_scopes,
            rejection_reasons=rejection_reasons,
            epoch=current_epoch,
        )
