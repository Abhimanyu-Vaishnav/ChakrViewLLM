"""
Distributed Revocation Propagation & Synchronization (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. REVOKED -> NEVER ACTIVE AGAIN:
   A revocation event is permanent and monotonic; a remote engine CANNOT un-revoke
   or restore an entity to active standing.
2. LOCAL REVOCATION WINS:
   If an entity is revoked locally but reported active remotely, the local revocation
   remains unconditionally authoritative.
3. IDEMPOTENT & DUPLICATE-SAFE:
   Re-processing previously applied revocations is a safe no-op.
4. SYNCHRONOUS CASCADE:
   Peer revocation automatically cascades to all associated sessions, keys,
   certificate bindings, and trust grants.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import time

from chakrview.cognition.federation.models import (
    RevocationSyncRecord,
    RevocationSyncMessage,
    RevocationTargetType,
    MAX_REVOCATION_SYNC_BATCH,
)
from chakrview.cognition.federation.errors import RevocationSyncError
from chakrview.cognition.peering.models import AuditEventType


class RevocationStateSynchronizer:
    """
    Manages bounded, monotonic, and idempotent revocation propagation across federation engines.
    """

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        # Bounded tracking of processed revocation hashes to guarantee idempotency
        self._processed_revocation_hashes: Set[str] = set()
        self._processed_revocation_ids: Set[str] = set()
        self._local_revocations: List[RevocationSyncRecord] = []

    def record_local_revocation(
        self,
        target_type: RevocationTargetType,
        target_id: str,
        reason: str,
        revoked_epoch: int,
        revoked_by: Optional[str] = None,
    ) -> RevocationSyncRecord:
        """
        Record a local revocation event for subsequent distributed propagation.
        """
        rev_id = f"rev_{target_type.value.lower()}_{target_id}_{revoked_epoch}"
        record = RevocationSyncRecord(
            revocation_id=rev_id,
            target_type=target_type,
            target_id=target_id,
            reason=reason,
            revoked_epoch=revoked_epoch,
            revoked_by=revoked_by or self.engine_id,
        )
        rec_hash = record.compute_hash()
        self._processed_revocation_hashes.add(rec_hash)
        self._processed_revocation_ids.add(rev_id)
        self._local_revocations.append(record)

        if len(self._local_revocations) > 200:
            self._local_revocations.pop(0)

        return record

    def generate_sync_message(
        self,
        current_epoch: int,
        state_version: int,
        limit: int = MAX_REVOCATION_SYNC_BATCH,
    ) -> RevocationSyncMessage:
        """
        Generate a batch of revocation sync records for propagation to remote engines.
        """
        batch = self._local_revocations[-limit:]
        return RevocationSyncMessage(
            engine_id=self.engine_id,
            epoch=current_epoch,
            state_version=state_version,
            revocations=batch,
        )

    def ingest_sync_message(
        self,
        engine: Any,
        sync_message: RevocationSyncMessage,
    ) -> Dict[str, Any]:
        """
        Ingest and execute incoming revocation records against the local federation engine.
        Enforces idempotency, monotonicity, and fail-closed cascades.
        """
        applied_count = 0
        duplicate_count = 0

        target_engine = getattr(engine, "engine", engine)

        for rec in sync_message.revocations:
            rec_hash = rec.compute_hash()
            if rec.revocation_id in self._processed_revocation_ids or rec_hash in self._processed_revocation_hashes:
                duplicate_count += 1
                if hasattr(target_engine, "audit_logger"):
                    target_engine.audit_logger.log(
                        event_type=AuditEventType.REVOCATION_DUPLICATE_IGNORED,
                        epoch=sync_message.epoch,
                        peer_id=rec.target_id if rec.target_type == RevocationTargetType.PEER else None,
                        details={
                            "revocation_id": rec.revocation_id,
                            "target_type": rec.target_type.value,
                            "remote_engine_id": sync_message.engine_id,
                        },
                    )
                continue

            # Mark processed immediately for idempotency
            self._processed_revocation_ids.add(rec.revocation_id)
            self._processed_revocation_hashes.add(rec_hash)
            self._local_revocations.append(rec)

            # Cascade revocation locally based on target type
            if rec.target_type == RevocationTargetType.PEER:
                if hasattr(target_engine, "registry") and target_engine.registry.get_peer(rec.target_id):
                    target_engine.revoke_peer(
                        peer_id=rec.target_id,
                        reason=f"Propagated from engine '{sync_message.engine_id}': {rec.reason}",
                    )
            elif rec.target_type == RevocationTargetType.SESSION:
                if hasattr(target_engine, "sessions") and rec.target_id in target_engine.sessions:
                    session = target_engine.sessions[rec.target_id]
                    session.revoke(reason=f"Propagated from engine '{sync_message.engine_id}': {rec.reason}")
            elif rec.target_type == RevocationTargetType.CERTIFICATE:
                if hasattr(target_engine, "certificate_revocation_registry"):
                    target_engine.certificate_revocation_registry.revoke(
                        fingerprint=rec.target_id,
                        reason=f"Propagated from engine '{sync_message.engine_id}': {rec.reason}",
                        epoch=rec.revoked_epoch,
                    )
            elif rec.target_type == RevocationTargetType.KEY:
                # Invalidate peer key if found
                if hasattr(target_engine, "registry"):
                    peer_reg = target_engine.registry.get_peer(rec.target_id)
                    if peer_reg and peer_reg.cryptographic_identity:
                        peer_reg.cryptographic_identity.revoke(
                            reason=f"Propagated from engine '{sync_message.engine_id}': {rec.reason}",
                            revoked_epoch=rec.revoked_epoch,
                            revoked_by=sync_message.engine_id,
                        )

            applied_count += 1
            if hasattr(target_engine, "audit_logger"):
                target_engine.audit_logger.log(
                    event_type=AuditEventType.REVOCATION_PROPAGATED,
                    epoch=sync_message.epoch,
                    peer_id=rec.target_id if rec.target_type == RevocationTargetType.PEER else None,
                    details={
                        "revocation_id": rec.revocation_id,
                        "target_type": rec.target_type.value,
                        "remote_engine_id": sync_message.engine_id,
                        "reason": rec.reason,
                    },
                )

        return {
            "remote_engine_id": sync_message.engine_id,
            "revocations_applied": applied_count,
            "duplicates_ignored": duplicate_count,
        }
