"""
Trust Revocation and Lifecycle Invalidation for Cross-Zone Federation (Step 29).

CRITICAL ARCHITECTURAL AXIOM:
FAIL-CLOSED REVOCATION:
Upon trust revocation, all active scopes are immediately terminated.
In-flight or pending requests from the revoked peer fail closed without exception.
Revocation is deterministic, auditable, and non-negotiable.
"""

from typing import Dict, List, Optional, Set

from chakrview.cognition.peering.models import (
    RevocationRecord,
    TrustGrant,
    TrustStatus,
    TrustLevel,
)


class RevocationManager:
    """
    Coordinates peer trust revocation, grant invalidation, and fail-closed termination.
    """

    def __init__(self) -> None:
        self._revocations: Dict[str, RevocationRecord] = {}

    def revoke(
        self,
        peer_id: str,
        zone_id: str,
        reason: str,
        revoked_epoch: int,
        revoked_by: str = "administrative_policy",
        active_grant: Optional[TrustGrant] = None,
    ) -> RevocationRecord:
        """
        Execute fail-closed revocation for a peer.
        Invalidates active grant and records auditable RevocationRecord.
        """
        affected_grants: List[str] = []
        if active_grant:
            active_grant.status = TrustStatus.REVOKED
            active_grant.trust_level = TrustLevel.REVOKED
            affected_grants.append(active_grant.grant_id)

        record_id = f"rev_{peer_id}_{revoked_epoch}"
        record = RevocationRecord(
            record_id=record_id,
            peer_id=peer_id,
            zone_id=zone_id,
            revoked_by=revoked_by,
            reason=reason,
            revoked_epoch=revoked_epoch,
            affected_grant_ids=affected_grants,
        )

        self._revocations[peer_id] = record
        return record

    def is_revoked(self, peer_id: str) -> bool:
        """Check if peer is in revoked state."""
        return peer_id in self._revocations

    def get_revocation_record(self, peer_id: str) -> Optional[RevocationRecord]:
        return self._revocations.get(peer_id)

    def list_revocations(self) -> List[RevocationRecord]:
        return list(self._revocations.values())
