"""
Bounded Trust Model and Grant Management for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOMS:
1. PEER_TRUST != PEER_AUTHORITY:
   Trust in a peer grants only bounded interaction scopes. It NEVER grants local
   authority, capability bypass, or model mutation privileges.
2. TRUST IS NEVER A BOOLEAN:
   Trust is a strongly typed, scoped, expiring, and revocable capability grant.
"""

from typing import Dict, List, Optional, Tuple, Any

from chakrview.cognition.peering.models import (
    TrustLevel,
    TrustStatus,
    TrustGrant,
    FederationScope,
)


class TrustModel:
    """
    Manages bounded trust grants, scope verification, and lifecycle transitions.
    """

    TRUST_HIERARCHY = {
        TrustLevel.NONE: 0,
        TrustLevel.IDENTIFIED: 1,
        TrustLevel.ATTESTED: 2,
        TrustLevel.LIMITED_TRUST: 3,
        TrustLevel.FEDERATED: 4,
        TrustLevel.REVOKED: -1,
    }

    @classmethod
    def create_grant(
        cls,
        grant_id: str,
        issuer_zone_id: str,
        subject_peer_id: str,
        subject_zone_id: str,
        trust_level: TrustLevel,
        permitted_scopes: List[FederationScope],
        issued_epoch: int,
        expires_epoch: int,
        policy_constraints: Optional[Dict[str, Any]] = None,
        reason: str = "Policy negotiation completed",
        provenance: str = "local_trust_engine",
    ) -> TrustGrant:
        """Create a validated, bounded TrustGrant."""
        if expires_epoch <= issued_epoch:
            raise ValueError(f"expires_epoch ({expires_epoch}) must be strictly greater than issued_epoch ({issued_epoch}).")

        return TrustGrant(
            grant_id=grant_id,
            issuer_zone_id=issuer_zone_id,
            subject_peer_id=subject_peer_id,
            subject_zone_id=subject_zone_id,
            trust_level=trust_level,
            permitted_scopes=list(permitted_scopes),
            issued_epoch=issued_epoch,
            expires_epoch=expires_epoch,
            policy_constraints=dict(policy_constraints or {}),
            reason=reason,
            status=TrustStatus.ACTIVE,
            provenance=provenance,
        )

    @classmethod
    def is_trusted_for_scope(
        cls,
        grant: Optional[TrustGrant],
        scope: FederationScope,
        current_epoch: int,
    ) -> Tuple[bool, str]:
        """
        Verify whether an active trust grant permits the requested scope at current_epoch.
        Fails closed.
        """
        if grant is None:
            return False, "No trust grant present (NONE)."

        if grant.status == TrustStatus.REVOKED:
            return False, "Trust grant has been explicitly revoked."

        if grant.status == TrustStatus.EXPIRED or current_epoch > grant.expires_epoch:
            return False, f"Trust grant expired at epoch {grant.expires_epoch} (current: {current_epoch})."

        if grant.status != TrustStatus.ACTIVE:
            return False, f"Trust grant is in non-active status '{grant.status.value}'."

        if grant.trust_level in (TrustLevel.NONE, TrustLevel.REVOKED):
            return False, f"Trust level is '{grant.trust_level.value}', no permissions granted."

        if scope not in grant.permitted_scopes:
            return False, f"Scope '{scope.value}' is not included in permitted scopes."

        return True, f"Scope '{scope.value}' verified under active trust grant '{grant.grant_id}'."

    @classmethod
    def check_and_expire(cls, grant: TrustGrant, current_epoch: int) -> bool:
        """Transition grant status to EXPIRED if current_epoch exceeds expires_epoch."""
        if grant.status == TrustStatus.ACTIVE and current_epoch > grant.expires_epoch:
            grant.status = TrustStatus.EXPIRED
            return True
        return False
