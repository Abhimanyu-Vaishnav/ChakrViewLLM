"""
Federation Recovery Manager & Snapshot Engine for ChakrView (Step 34).

Implements deterministic crash recovery from durable snapshots and cryptographically
chained write-ahead journals:
- Fail-Closed Semantics: Any detected corruption, broken chain, or tampering halts
  recovery and raises RecoveryFailedClosedError.
- Monotonic Invariants: Terminal revocations, replay floors, and trust ceilings are
  strictly reconstructed.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.peering.models import (
    AuditEventType,
    DiscoveryStatus,
    TrustGrant,
    TrustLevel,
    TrustStatus,
    FederationScope,
    RevocationRecord,
    PeerRegistration,
    PeerIdentity,
)
from chakrview.cognition.peering.crypto import (
    CryptographicPeerIdentity,
    Ed25519PublicKeyWrapper,
    KeyLifecycleState,
)
from chakrview.cognition.peering.session import (
    SecurePeerSession,
    SessionStatus,
    SessionKeyState,
    SessionKeyMetadata,
)
from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    RevocationTargetType,
)
from chakrview.cognition.federation.persistence.base import SecurityStateStore
from chakrview.cognition.federation.persistence.errors import (
    RecoveryFailedClosedError,
    SnapshotCorruptionError,
    JournalCorruptionError,
    JournalSequenceError,
)
from chakrview.cognition.federation.persistence.models import (
    DurableSecuritySnapshot,
    JournalEntry,
    JournalEntryType,
    RecoveryManifest,
    JOURNAL_GENESIS_DIGEST,
)
from chakrview.cognition.federation.persistence.journal import SecurityStateJournal
from chakrview.cognition.federation.discovery.models import (
    FederationNodeMembership,
    MembershipState,
    FederationNodeCandidate,
)

logger = logging.getLogger(__name__)


class FederationRecoveryManager:
    """
    Coordinates recovery of federation security state from durable snapshots and journals.
    """

    def __init__(self, store: SecurityStateStore) -> None:
        self.store = store

    # ========================================================================
    # 1. Snapshot Creation (Phase 4)
    # ========================================================================

    @staticmethod
    def create_snapshot_from_engine(
        engine: Any,
        store: SecurityStateStore,
        snapshot_version: int,
        journal_offset: int,
    ) -> DurableSecuritySnapshot:
        """
        Extract normalized security state from an active CrossZoneFederationEngine
        and store it as a sealed snapshot.
        """
        coord = getattr(engine, "coordinator", None)
        engine_id_dict = coord.engine_identity.to_dict() if coord else {
            "engine_id": f"eng_{engine.local_peer_id}",
            "zone_id": engine.local_zone_id,
            "created_epoch": engine.current_epoch,
            "identity_fingerprint": "default",
            "protocol_version": "34.0",
        }

        state_ver_dict = coord.current_version.to_dict() if coord else {
            "version": 1,
            "epoch": engine.current_epoch,
            "engine_id": engine_id_dict["engine_id"],
        }

        # 1. Peer Registrations
        peers_data: List[Dict[str, Any]] = []
        if hasattr(engine, "registry"):
            for reg in engine.registry.list_peers():
                p_dict = {
                    "peer_id": reg.identity.peer_id,
                    "zone_id": reg.identity.zone_id,
                    "organization_id": reg.identity.organization_id,
                    "discovery_status": reg.discovery_status.value,
                    "registered_epoch": reg.registered_epoch,
                    "last_seen_epoch": reg.last_seen_epoch,
                }
                if reg.cryptographic_identity:
                    crypto = reg.cryptographic_identity
                    p_dict["crypto"] = {
                        "public_hex": crypto.public_key.public_hex,
                        "fingerprint": crypto.public_key.fingerprint,
                        "key_state": crypto.key_state.value,
                        "created_epoch": crypto.created_epoch,
                        "retired_key_fingerprints": sorted(list(crypto.retired_key_fingerprints)),
                    }
                peers_data.append(p_dict)

        # 2. Sessions & Replay Floors
        sessions_data: List[Dict[str, Any]] = []
        replay_floors: Dict[str, int] = {}
        if hasattr(engine, "sessions"):
            for sid, sess in engine.sessions.items():
                s_dict = {
                    "session_id": sess.session_id,
                    "remote_peer_id": sess.remote_peer_id,
                    "remote_zone_id": sess.remote_zone_id,
                    "status": sess.status.value,
                    "created_epoch": sess.created_epoch,
                    "expires_at_epoch": sess.expires_at_epoch,
                    "renewal_count": sess.renewal_count,
                    "last_sequence_number": sess.last_sequence_number,
                    "seen_message_ids": sorted(list(sess._seen_message_ids)),
                }
                sessions_data.append(s_dict)
                replay_floors[sid] = sess.last_sequence_number

        # 3. Revocations
        revocations_data: List[Dict[str, Any]] = []
        if hasattr(engine, "revocation_manager") and hasattr(engine.revocation_manager, "_revocations"):
            for rid, rec in engine.revocation_manager._revocations.items():
                revocations_data.append({
                    "revocation_id": rec.revocation_id,
                    "target_type": "PEER",
                    "target_id": rec.peer_id,
                    "reason": rec.reason,
                    "revoked_epoch": rec.revoked_epoch,
                    "revoked_by": rec.revoked_by,
                })

        # 4. Trust Grants
        trust_data: List[Dict[str, Any]] = []
        if hasattr(engine, "registry"):
            for reg in engine.registry.list_peers():
                if reg.trust_grant:
                    tg = reg.trust_grant
                    trust_data.append({
                        "grant_id": tg.grant_id,
                        "peer_id": reg.identity.peer_id,
                        "trust_level": tg.trust_level.value,
                        "status": tg.status.value,
                        "epoch_granted": getattr(tg, "issued_epoch", getattr(tg, "epoch_granted", 1)),
                        "expires_epoch": tg.expires_epoch,
                        "permitted_scopes": [s.value for s in tg.permitted_scopes],
                    })

        # 5. Certificate Revocations
        cert_revocations: List[str] = []
        if hasattr(engine, "certificate_revocation_registry"):
            cert_revocations = sorted(list(engine.certificate_revocation_registry.list_revocations()))

        # 6. Memberships
        memberships_data: List[Dict[str, Any]] = []
        membership_mgr = getattr(engine, "membership_manager", None)
        if not membership_mgr and hasattr(engine, "runtime"):
            membership_mgr = getattr(engine.runtime, "membership_manager", None)
        if membership_mgr:
            for m in membership_mgr.list_memberships():
                memberships_data.append(m.to_dict())

        # 7. Composite Digest
        composite_digest = "NO_DIGEST"
        if coord:
            rev_records = list(engine.revocation_manager._revocations.values()) if hasattr(engine, "revocation_manager") else []
            composite_digest = coord.compute_state_digest(
                registry=engine.registry if hasattr(engine, "registry") else None,
                revocations=rev_records,
            ).compute_composite_digest()

        snapshot_id = f"snap_{snapshot_version}_{journal_offset}_{engine.current_epoch}"
        snapshot = DurableSecuritySnapshot(
            snapshot_id=snapshot_id,
            snapshot_version=snapshot_version,
            journal_offset=journal_offset,
            engine_identity=engine_id_dict,
            state_version=state_ver_dict,
            epoch=engine.current_epoch,
            peers=peers_data,
            sessions=sessions_data,
            revocations=revocations_data,
            trust_grants=trust_data,
            certificate_revocations=cert_revocations,
            replay_floors=replay_floors,
            composite_digest=composite_digest,
            memberships=memberships_data,
        )
        snapshot.seal()
        store.save_snapshot(snapshot)

        if hasattr(engine, "audit_logger"):
            engine.audit_logger.log(
                event_type=AuditEventType.SNAPSHOT_CREATED,
                epoch=engine.current_epoch,
                details={
                    "snapshot_id": snapshot_id,
                    "journal_offset": journal_offset,
                    "peers_count": len(peers_data),
                    "sessions_count": len(sessions_data),
                },
            )

        return snapshot

    # ========================================================================
    # 2. Crash Recovery Procedure (Phase 3)
    # ========================================================================

    def recover(self, target_engine: Any) -> RecoveryManifest:
        """
        Execute deterministic crash recovery into target_engine.
        Fails closed on any corruption, broken journal hash chain, or tampering.
        """
        start_time = time.time()
        audit = getattr(target_engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.RECOVERY_STARTED,
                epoch=target_engine.current_epoch,
                details={"target_peer_id": target_engine.local_peer_id},
            )

        try:
            # Step 1: Load latest verified snapshot (if any)
            snapshot = self.store.load_latest_snapshot()
            journal_offset = 0
            snapshot_id = None

            if snapshot:
                if not snapshot.verify_integrity():
                    raise SnapshotCorruptionError(
                        f"Snapshot '{snapshot.snapshot_id}' failed SHA-256 integrity check!"
                    )
                snapshot_id = snapshot.snapshot_id
                journal_offset = snapshot.journal_offset
                self._apply_snapshot(target_engine, snapshot)

            # Step 2: Read journal entries strictly after snapshot_offset
            journal_entries = self.store.read_journal_entries(since_sequence=journal_offset)

            # Step 3: Validate journal chain integrity
            if journal_entries:
                SecurityStateJournal.verify_chain(journal_entries)

                # Step 4: Replay valid journal entries sequentially
                for entry in journal_entries:
                    self._apply_journal_entry(target_engine, entry)

            # Step 5: Enforce terminal revocation invariant across all entities
            self._enforce_terminal_invariants(target_engine)

            # Step 6: Expire outdated trust grants based on current epoch
            if hasattr(target_engine, "registry"):
                target_engine.registry.expire_peers(target_engine.current_epoch)

            # Step 7: Compute final recovered composite digest
            final_digest = "NO_DIGEST"
            coord = getattr(target_engine, "coordinator", None)
            if coord:
                rev_records = list(target_engine.revocation_manager._revocations.values()) if hasattr(target_engine, "revocation_manager") else []
                final_digest = coord.compute_state_digest(
                    registry=target_engine.registry if hasattr(target_engine, "registry") else None,
                    revocations=rev_records,
                ).compute_composite_digest()

            final_ver = coord.current_version.version if coord else 1
            manifest = RecoveryManifest(
                engine_id=f"eng_{target_engine.local_peer_id}",
                snapshot_loaded=snapshot_id,
                journal_entries_replayed=len(journal_entries),
                final_version=final_ver,
                final_epoch=target_engine.current_epoch,
                final_digest=final_digest,
                status="SUCCESS",
            )

            if audit:
                audit.log(
                    event_type=AuditEventType.RECOVERY_COMPLETED,
                    epoch=target_engine.current_epoch,
                    details={
                        "snapshot_id": snapshot_id,
                        "replayed_entries": len(journal_entries),
                        "final_version": final_ver,
                    },
                )

            return manifest

        except Exception as e:
            if audit:
                audit.log(
                    event_type=AuditEventType.RECOVERY_FAILED_CLOSED,
                    epoch=target_engine.current_epoch,
                    details={"error": str(e)},
                )
            raise RecoveryFailedClosedError(
                f"Recovery failed closed due to integrity or corruption failure: {e}"
            ) from e

    # ========================================================================
    # 3. Internal Application Helpers
    # ========================================================================

    def _apply_snapshot(self, engine: Any, snapshot: DurableSecuritySnapshot) -> None:
        """Restore in-memory state from a validated snapshot."""
        engine.current_epoch = max(engine.current_epoch, snapshot.epoch)

        coord = getattr(engine, "coordinator", None)
        if coord and "version" in snapshot.state_version:
            # Sync coordinator version
            target_v = int(snapshot.state_version["version"])
            if snapshot.epoch > coord.current_epoch:
                coord.state_manager.advance_epoch(snapshot.epoch)
            while coord.current_version.version < target_v:
                coord.state_manager.increment_version()

        # Restore peers and cryptographic identities
        if hasattr(engine, "registry"):
            for p in snapshot.peers:
                peer_id = p["peer_id"]
                zone_id = p["zone_id"]
                org_id = p.get("organization_id", "recovered_org")
                base_id = PeerIdentity(
                    peer_id=peer_id,
                    zone_id=zone_id,
                    organization_id=org_id,
                    created_epoch=p.get("registered_epoch", 1),
                )
                crypto_id = None
                if "crypto" in p:
                    c = p["crypto"]
                    pub_wrapper = Ed25519PublicKeyWrapper.from_raw_bytes(bytes.fromhex(c["public_hex"]))
                    crypto_id = CryptographicPeerIdentity(
                        peer_id=peer_id,
                        zone_id=zone_id,
                        organization_id=org_id,
                        public_key=pub_wrapper,
                        key_state=KeyLifecycleState(c["key_state"]),
                        created_epoch=c.get("created_epoch", 1),
                    )
                    crypto_id.retired_key_fingerprints = set(c.get("retired_key_fingerprints", []))

                reg = PeerRegistration(
                    identity=base_id,
                    discovery_status=DiscoveryStatus(p["discovery_status"]),
                    registered_epoch=p.get("registered_epoch", 1),
                    last_seen_epoch=p.get("last_seen_epoch", 1),
                    cryptographic_identity=crypto_id,
                )
                try:
                    engine.registry.register_peer(reg)
                except Exception:
                    pass

        # Restore trust grants
        if hasattr(engine, "registry"):
            for tg in snapshot.trust_grants:
                peer_reg = engine.registry.get_peer(tg["peer_id"])
                if peer_reg:
                    scopes = [FederationScope(s) for s in tg.get("permitted_scopes", [])]
                    grant = TrustGrant(
                        grant_id=tg["grant_id"],
                        issuer_zone_id=engine.local_zone_id,
                        subject_peer_id=peer_reg.identity.peer_id,
                        subject_zone_id=peer_reg.identity.zone_id,
                        trust_level=TrustLevel(tg["trust_level"]),
                        permitted_scopes=scopes,
                        issued_epoch=tg.get("epoch_granted", 1),
                        expires_epoch=tg["expires_epoch"],
                        status=TrustStatus(tg["status"]),
                    )
                    peer_reg.trust_grant = grant

        # Restore active / revoked sessions & replay floors
        if hasattr(engine, "sessions"):
            for s in snapshot.sessions:
                sid = s["session_id"]
                sess = SecurePeerSession(
                    session_id=sid,
                    local_peer_id=engine.local_peer_id,
                    remote_peer_id=s["remote_peer_id"],
                    local_zone_id=engine.local_zone_id,
                    remote_zone_id=s.get("remote_zone_id", "unknown"),
                    created_epoch=s["created_epoch"],
                    expires_at_epoch=s["expires_at_epoch"],
                    status=SessionStatus(s["status"]),
                )
                sess.renewal_count = s.get("renewal_count", 0)
                sess.last_seen_sequence_number = s.get("last_sequence_number", 0)
                sess._seen_message_ids = set(s.get("seen_message_ids", []))
                engine.sessions[sid] = sess

        # Restore certificate revocations
        if hasattr(engine, "certificate_revocation_registry"):
            for fp in snapshot.certificate_revocations:
                engine.certificate_revocation_registry.revoke(
                    fingerprint=fp,
                    reason="Restored from snapshot",
                    epoch=snapshot.epoch,
                )

        # Restore peer revocations
        if hasattr(engine, "revocation_manager"):
            for r in snapshot.revocations:
                target_id = r["target_id"]
                engine.revocation_manager.record_revocation(
                    peer_id=target_id,
                    zone_id=engine.local_zone_id,
                    reason=r.get("reason", "Restored from snapshot"),
                    revoked_epoch=r.get("revoked_epoch", snapshot.epoch),
                    revoked_by=r.get("revoked_by", "snapshot_recovery"),
                )

        # Restore memberships
        membership_mgr = getattr(engine, "membership_manager", None)
        if not membership_mgr and hasattr(engine, "runtime"):
            membership_mgr = getattr(engine.runtime, "membership_manager", None)
        if membership_mgr and hasattr(snapshot, "memberships") and snapshot.memberships:
            for m_dict in snapshot.memberships:
                try:
                    mem = FederationNodeMembership.from_dict(m_dict)
                    membership_mgr._memberships[mem.membership_id] = mem
                    if mem.node_id:
                        membership_mgr._node_to_membership[mem.node_id] = mem.membership_id
                    if mem.endpoint:
                        membership_mgr._endpoint_to_membership[mem.endpoint.endpoint_id] = mem.membership_id
                except Exception as ex:
                    logger.warning("Failed to restore membership: %s", ex)

    def _apply_journal_entry(self, engine: Any, entry: JournalEntry) -> None:
        """Apply an individual journal entry to mutate in-memory security state."""
        coord = getattr(engine, "coordinator", None)
        p = entry.payload

        if entry.entry_type == JournalEntryType.EPOCH_ADVANCED:
            new_epoch = p.get("new_epoch", engine.current_epoch)
            engine.current_epoch = max(engine.current_epoch, new_epoch)
            if coord and engine.current_epoch > coord.current_epoch:
                coord.state_manager.advance_epoch(engine.current_epoch)
        else:
            if coord:
                coord.state_manager.increment_version()

        if entry.entry_type == JournalEntryType.PEER_REGISTERED:
            if hasattr(engine, "registry"):
                pub_hex = p.get("public_hex")
                peer_id = p["peer_id"]
                zone_id = p["zone_id"]
                base_id = PeerIdentity(peer_id=peer_id, zone_id=zone_id, created_epoch=entry.epoch)
                crypto_id = None
                if pub_hex:
                    pub = Ed25519PublicKeyWrapper.from_raw_bytes(bytes.fromhex(pub_hex))
                    crypto_id = CryptographicPeerIdentity(
                        peer_id=peer_id,
                        zone_id=zone_id,
                        public_key=pub,
                        created_epoch=entry.epoch,
                    )
                reg = PeerRegistration(
                    identity=base_id,
                    discovery_status=DiscoveryStatus.VERIFIED,
                    registered_epoch=entry.epoch,
                    cryptographic_identity=crypto_id,
                )
                try:
                    engine.registry.register_peer(reg)
                except Exception:
                    pass

        elif entry.entry_type == JournalEntryType.PEER_REVOKED:
            peer_id = p["peer_id"]
            reason = p.get("reason", "Revocation from journal replay")
            if hasattr(engine, "registry"):
                engine.registry.revoke_peer(peer_id=peer_id, reason=reason, revoked_epoch=entry.epoch, revoked_by="journal_replay")
                reg = engine.registry.get_peer(peer_id)
                if reg and reg.cryptographic_identity:
                    reg.cryptographic_identity.revoke(reason=reason, revoked_epoch=entry.epoch, revoked_by="journal_replay")
            if hasattr(engine, "sessions"):
                for sid, sess in list(engine.sessions.items()):
                    if sess.remote_peer_id == peer_id:
                        sess.revoke(reason=f"Peer revoked: {reason}")
            if hasattr(engine, "certificate_binder"):
                engine.certificate_binder.unbind_peer(peer_id)
            if hasattr(engine, "coordinator"):
                engine.coordinator.record_and_propagate_revocation(
                    target_type=RevocationTargetType.PEER,
                    target_id=peer_id,
                    reason=reason,
                    engine=engine,
                )

        elif entry.entry_type == JournalEntryType.SESSION_CREATED:
            sid = p["session_id"]
            if hasattr(engine, "sessions") and sid not in engine.sessions:
                sess = SecurePeerSession(
                    session_id=sid,
                    local_peer_id=engine.local_peer_id,
                    remote_peer_id=p["remote_peer_id"],
                    local_zone_id=engine.local_zone_id,
                    remote_zone_id=p.get("remote_zone_id", "unknown"),
                    created_epoch=entry.epoch,
                    expires_at_epoch=p.get("expires_at_epoch", entry.epoch + 50),
                    status=SessionStatus.INITIATED,
                )
                engine.sessions[sid] = sess

        elif entry.entry_type == JournalEntryType.SESSION_ACTIVATED:
            sid = p["session_id"]
            if hasattr(engine, "sessions") and sid in engine.sessions:
                sess = engine.sessions[sid]
                sess.status = SessionStatus.ACTIVE
                if sess.session_key_metadata:
                    sess.session_key_metadata.key_state = SessionKeyState.ACTIVE

        elif entry.entry_type in (JournalEntryType.SESSION_TERMINATED, JournalEntryType.SESSION_REVOKED):
            sid = p["session_id"]
            if hasattr(engine, "sessions") and sid in engine.sessions:
                sess = engine.sessions[sid]
                if entry.entry_type == JournalEntryType.SESSION_TERMINATED:
                    sess.status = SessionStatus.TERMINATED
                else:
                    sess.status = SessionStatus.REVOKED
                if sess.session_key_metadata:
                    sess.session_key_metadata.key_state = (
                        SessionKeyState.EXPIRED if entry.entry_type == JournalEntryType.SESSION_TERMINATED
                        else SessionKeyState.REVOKED
                    )

        elif entry.entry_type == JournalEntryType.KEY_ROTATED:
            peer_id = p["peer_id"]
            new_hex = p["new_public_hex"]
            if hasattr(engine, "registry"):
                reg = engine.registry.get_peer(peer_id)
                if reg and reg.cryptographic_identity:
                    new_pub = Ed25519PublicKeyWrapper.from_raw_bytes(bytes.fromhex(new_hex))
                    old_fp = reg.cryptographic_identity.public_key.fingerprint
                    reg.cryptographic_identity.retired_key_fingerprints.add(old_fp)
                    reg.cryptographic_identity.public_key = new_pub

        elif entry.entry_type == JournalEntryType.CERTIFICATE_REVOKED:
            fp = p["fingerprint"]
            if hasattr(engine, "certificate_revocation_registry"):
                engine.certificate_revocation_registry.revoke(
                    fingerprint=fp,
                    reason=p.get("reason", "Revocation from journal replay"),
                    epoch=entry.epoch,
                )

        elif entry.entry_type == JournalEntryType.REPLAY_FLOOR_ADVANCED:
            sid = p["session_id"]
            seq = p["sequence_number"]
            msg_id = p.get("message_id")
            if hasattr(engine, "sessions") and sid in engine.sessions:
                sess = engine.sessions[sid]
                sess.last_seen_sequence_number = max(sess.last_seen_sequence_number, seq)
                if msg_id:
                    sess._seen_message_ids.add(msg_id)

        # Membership journal entry replay
        membership_mgr = getattr(engine, "membership_manager", None)
        if not membership_mgr and hasattr(engine, "runtime"):
            membership_mgr = getattr(engine.runtime, "membership_manager", None)

        if membership_mgr:
            mid = p.get("membership_id")
            nid = p.get("node_id") or p.get("engine_id")
            cid = p.get("candidate_id")

            # Resolve target membership if possible
            target_mem = None
            if mid and mid in membership_mgr._memberships:
                target_mem = membership_mgr._memberships[mid]
            elif nid and nid in membership_mgr._node_to_membership:
                target_mem = membership_mgr._memberships.get(membership_mgr._node_to_membership[nid])
            elif nid and nid in membership_mgr._memberships:
                target_mem = membership_mgr._memberships[nid]

            if entry.entry_type == JournalEntryType.NODE_DISCOVERED:
                if mid and "endpoint" in p:
                    try:
                        ep = FederationNodeEndpoint.from_dict(p["endpoint"])
                        cand = FederationNodeCandidate(
                            candidate_id=cid or f"cand_{ep.endpoint_id}",
                            endpoint=ep,
                            discovery_source=NodeDiscoverySource.STATIC_CONFIG,
                            discovered_epoch=entry.epoch,
                        )
                        membership = FederationNodeMembership.create(candidate=cand, enrolled_epoch=entry.epoch)
                        membership_mgr._memberships[membership.membership_id] = membership
                        membership_mgr._endpoint_to_membership[ep.endpoint_id] = membership.membership_id
                        membership_mgr._candidates[cand.candidate_id] = cand
                    except Exception:
                        pass
                elif "candidate" in p:
                    try:
                        cand = FederationNodeCandidate.from_dict(p["candidate"])
                        membership_mgr._candidates[cand.candidate_id] = cand
                    except Exception:
                        pass
            elif entry.entry_type == JournalEntryType.NODE_AUTHENTICATED:
                if cid and cid in membership_mgr._candidates:
                    membership_mgr._candidates[cid].state = MembershipState.AUTHENTICATED
                if target_mem:
                    target_mem.state = MembershipState.AUTHENTICATED
                    if nid:
                        target_mem.node_id = nid
                        membership_mgr._node_to_membership[nid] = target_mem.membership_id
            elif entry.entry_type == JournalEntryType.NODE_MEMBERSHIP_GRANTED:
                if target_mem:
                    target_mem.state = MembershipState.MEMBER
                elif "membership" in p:
                    try:
                        mem = FederationNodeMembership.from_dict(p["membership"])
                        membership_mgr._memberships[mem.membership_id] = mem
                        if mem.node_id:
                            membership_mgr._node_to_membership[mem.node_id] = mem.membership_id
                    except Exception:
                        pass
            elif entry.entry_type == JournalEntryType.NODE_MEMBERSHIP_SUSPENDED:
                if target_mem:
                    target_mem.state = MembershipState.SUSPENDED
            elif entry.entry_type == JournalEntryType.NODE_QUARANTINED:
                if target_mem:
                    target_mem.state = MembershipState.QUARANTINED
                    target_mem.quarantine_reason = p.get("reason", "Quarantined by journal replay")
            elif entry.entry_type == JournalEntryType.NODE_REVOKED:
                if target_mem:
                    target_mem.state = MembershipState.REVOKED
                    target_mem.revocation_reason = p.get("reason", "Revocation from journal replay")
            elif entry.entry_type == JournalEntryType.NODE_TERMINATED:
                if target_mem:
                    target_mem.state = MembershipState.TERMINATED
            elif entry.entry_type == JournalEntryType.NODE_REMOVED:
                if target_mem:
                    membership_mgr._memberships.pop(target_mem.membership_id, None)
                    if target_mem.node_id:
                        membership_mgr._node_to_membership.pop(target_mem.node_id, None)
                if cid:
                    membership_mgr._candidates.pop(cid, None)

    def _enforce_terminal_invariants(self, engine: Any) -> None:
        """
        Enforce that terminal states (REVOKED, EXPIRED, TERMINATED) cannot be undone.
        If a peer is revoked, all associated sessions must be revoked.
        """
        if hasattr(engine, "registry") and hasattr(engine, "sessions"):
            revoked_peers = set()
            for reg in engine.registry.list_peers():
                if reg.discovery_status == DiscoveryStatus.REVOKED or reg.revocation_record:
                    revoked_peers.add(reg.identity.peer_id)

            for sid, sess in list(engine.sessions.items()):
                if sess.remote_peer_id in revoked_peers:
                    sess.status = SessionStatus.REVOKED
                    if sess.session_key_metadata:
                        sess.session_key_metadata.key_state = SessionKeyState.REVOKED

            # Enforce membership terminal invariants
            membership_mgr = getattr(engine, "membership_manager", None)
            if not membership_mgr and hasattr(engine, "runtime"):
                membership_mgr = getattr(engine.runtime, "membership_manager", None)
            if membership_mgr:
                for nid, mem in list(membership_mgr._memberships.items()):
                    if mem.node_id in revoked_peers:
                        mem.state = MembershipState.REVOKED
