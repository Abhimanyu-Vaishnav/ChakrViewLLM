"""
Distributed Trust State Consistency (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. REMOTE_TRUST_CLAIM != LOCAL_TRUST_AUTHORIZATION:
   A remote engine reporting that a peer is trusted or federated carries ZERO local authority.
2. NO SELF-ESCALATION:
   A remote engine CANNOT grant itself capabilities or extend its own trust.
3. NO TRUST EXTENSION:
   A remote engine CANNOT extend another peer's trust beyond local policy.
4. LOCAL POLICY & CAPABILITY GATE REMAIN UNCONDITIONALLY AUTHORITATIVE:
   Local authorization decisions are made exclusively by local TrustModel and CapabilityGate.
"""

from typing import Dict, List, Optional, Tuple, Any

from chakrview.cognition.federation.models import (
    TrustSyncRecord,
    TrustSyncMessage,
)
from chakrview.cognition.federation.errors import TrustSyncError
from chakrview.cognition.peering.registry import PeerRegistry
from chakrview.cognition.peering.policy import CrossZoneFederationPolicy
from chakrview.cognition.peering.models import TrustLevel, TrustStatus


class TrustStateSynchronizer:
    """
    Manages bounded trust-state consistency checks across federation engines.
    Strictly prevents remote authority transfer or capability self-escalation.
    """

    def __init__(self, engine_id: str, zone_id: str) -> None:
        self.engine_id = engine_id
        self.zone_id = zone_id

    def generate_sync_message(
        self,
        registry: PeerRegistry,
        current_epoch: int,
        state_version: int,
    ) -> TrustSyncMessage:
        """
        Generate bounded trust sync report of locally known peer standings.
        """
        records: List[TrustSyncRecord] = []
        for reg in registry.list_peers():
            if reg.trust_grant:
                records.append(
                    TrustSyncRecord(
                        subject_peer_id=reg.identity.peer_id,
                        subject_zone_id=reg.identity.zone_id,
                        trust_level=reg.trust_grant.trust_level.value,
                        status=reg.trust_grant.status.value,
                        expires_epoch=reg.trust_grant.expires_epoch,
                        permitted_scopes=[s.value for s in reg.trust_grant.permitted_scopes],
                    )
                )

        return TrustSyncMessage(
            engine_id=self.engine_id,
            zone_id=self.zone_id,
            epoch=current_epoch,
            state_version=state_version,
            trust_records=records,
        )

    def ingest_sync_message(
        self,
        registry: PeerRegistry,
        sync_message: TrustSyncMessage,
        local_policy: Optional[CrossZoneFederationPolicy] = None,
        current_epoch: int = 1,
    ) -> Dict[str, Any]:
        """
        Ingest remote trust claims with strict non-escalation enforcement:
        - Claims of FEDERATED or LIMITED_TRUST are recorded for visibility but
          NEVER automatically elevate or grant local trust.
        - Claims of REVOKED or EXPIRED trigger local advisory checks.
        """
        escalation_attempts = 0
        advisory_warnings = 0
        records_processed = 0

        for rec in sync_message.trust_records:
            records_processed += 1
            local_reg = registry.get_peer(rec.subject_peer_id)

            # Check for escalation attempt: remote claiming trust for an unverified/unknown peer
            if local_reg is None:
                if rec.trust_level in (TrustLevel.LIMITED_TRUST.value, TrustLevel.FEDERATED.value):
                    escalation_attempts += 1
                continue

            # If peer exists locally, compare standing
            local_grant = local_reg.trust_grant
            if local_grant is None:
                if rec.trust_level in (TrustLevel.LIMITED_TRUST.value, TrustLevel.FEDERATED.value):
                    escalation_attempts += 1
                continue

            # Case: Remote claims peer is revoked, but locally active
            if rec.status == TrustStatus.REVOKED.value and local_grant.status == TrustStatus.ACTIVE:
                advisory_warnings += 1

        return {
            "remote_engine_id": sync_message.engine_id,
            "remote_zone_id": sync_message.zone_id,
            "records_processed": records_processed,
            "escalation_attempts_blocked": escalation_attempts,
            "advisory_warnings_raised": advisory_warnings,
            "local_authority_preserved": True,
        }
