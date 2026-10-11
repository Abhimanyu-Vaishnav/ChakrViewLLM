"""
Federation State Management, Versioning, and Cryptographic Digests (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. DETERMINISTIC REPRODUCIBILITY:
   All digests are SHA-256 over sorted, canonical JSON documents with compact separators.
2. ZERO SECRET INCLUSION:
   State digests NEVER include private keys, raw session secrets, token activations,
   internal object memory addresses, or model weights.
3. MONOTONIC VERSIONING:
   State versions strictly increase. Stale versions and divergent same-version states
   fail closed.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import json
import time

from chakrview.cognition.federation.models import (
    FederationEngineIdentity,
    SecurityStateVersion,
    ReplayStateDigest,
    TrustStateDigest,
    RevocationStateDigest,
    PeerStateDigest,
    FederationSecurityStateDigest,
)
from chakrview.cognition.federation.errors import (
    StateVersionError,
    StaleStateError,
    StateDigestConflictError,
)
from chakrview.cognition.peering.session import SecurePeerSession
from chakrview.cognition.peering.registry import PeerRegistry
from chakrview.cognition.peering.models import RevocationRecord, TrustStatus


class FederationStateManager:
    """
    Manages monotonic security state versions and reproducible cryptographic digests
    for distributed federation coordination.
    """

    def __init__(self, engine_identity: FederationEngineIdentity, initial_epoch: int = 1) -> None:
        self.engine_identity = engine_identity
        self.current_epoch = initial_epoch
        self._current_version = 1
        self._version_history: List[SecurityStateVersion] = []
        self._record_current_version()

    @property
    def current_version(self) -> SecurityStateVersion:
        return SecurityStateVersion(
            version=self._current_version,
            epoch=self.current_epoch,
            engine_id=self.engine_identity.engine_id,
        )

    def _record_current_version(self) -> None:
        rec = self.current_version
        self._version_history.append(rec)
        if len(self._version_history) > 100:
            self._version_history.pop(0)

    def increment_version(self, epoch: Optional[int] = None) -> SecurityStateVersion:
        """
        Monotonically increment state version when a security-relevant event occurs
        (e.g., peer registered, session authenticated, key/cert rotated, revocation).
        """
        if epoch is not None:
            if epoch < self.current_epoch:
                raise StateVersionError(
                    f"Epoch regression forbidden: cannot update to epoch {epoch} < {self.current_epoch}."
                )
            self.current_epoch = epoch

        self._current_version += 1
        self._record_current_version()
        return self.current_version

    def advance_epoch(self, new_epoch: int) -> SecurityStateVersion:
        """Advance logical epoch and increment state version."""
        if new_epoch <= self.current_epoch:
            raise StateVersionError(
                f"New epoch {new_epoch} must be strictly greater than current {self.current_epoch}."
            )
        self.current_epoch = new_epoch
        self._current_version += 1
        self._record_current_version()
        return self.current_version

    # ========================================================================
    # Deterministic Digest Computations (Phase 2 & Phase 7)
    # ========================================================================

    def compute_replay_digest(
        self,
        session: Optional[SecurePeerSession] = None,
        session_id: str = "global",
    ) -> ReplayStateDigest:
        """
        Compute deterministic digest of session replay protection state.
        Zero secret leakage: only message IDs and sequence counts are hashed.
        """
        highest_seq = 0
        msg_count = 0
        msg_ids_hash = "EMPTY"

        if session is not None:
            session_id = session.session_id
            highest_seq = session.last_seen_sequence_number
            seen_ids = sorted(list(session.seen_message_ids))
            msg_count = len(seen_ids)
            if seen_ids:
                canonical_msg_ids = json.dumps(seen_ids, separators=(",", ":"))
                msg_ids_hash = hashlib.sha256(canonical_msg_ids.encode("utf-8")).hexdigest()

        return ReplayStateDigest(
            engine_id=self.engine_identity.engine_id,
            session_id=session_id,
            epoch=self.current_epoch,
            highest_sequence_number=highest_seq,
            message_count=msg_count,
            message_id_digest=msg_ids_hash,
            state_version=self._current_version,
        )

    def compute_trust_digest(
        self,
        registry: Optional[PeerRegistry] = None,
    ) -> TrustStateDigest:
        """
        Compute deterministic digest of active and revoked trust grants.
        """
        active_count = 0
        revoked_count = 0
        grant_ids: List[str] = []

        if registry is not None:
            peers = registry.list_peers()
            for reg in peers:
                if reg.trust_grant:
                    grant_ids.append(reg.trust_grant.grant_id)
                    if reg.trust_grant.is_valid_at(self.current_epoch):
                        active_count += 1
                    else:
                        revoked_count += 1

        grant_ids.sort()
        grant_ids_hash = hashlib.sha256(
            json.dumps(grant_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if grant_ids else "NO_GRANTS"

        return TrustStateDigest(
            engine_id=self.engine_identity.engine_id,
            zone_id=self.engine_identity.zone_id,
            epoch=self.current_epoch,
            active_grants_count=active_count,
            revoked_grants_count=revoked_count,
            grant_ids_hash=grant_ids_hash,
            state_version=self._current_version,
        )

    def compute_revocation_digest(
        self,
        revocation_records: Optional[List[RevocationRecord]] = None,
    ) -> RevocationStateDigest:
        """
        Compute deterministic digest of all active revocation records.
        """
        rev_count = 0
        rev_hashes: List[str] = []

        if revocation_records:
            rev_count = len(revocation_records)
            for rec in sorted(revocation_records, key=lambda r: r.record_id):
                doc = {
                    "affected_grant_ids": sorted(rec.affected_grant_ids),
                    "peer_id": rec.peer_id,
                    "reason": rec.reason,
                    "record_id": rec.record_id,
                    "revoked_by": rec.revoked_by,
                    "revoked_epoch": rec.revoked_epoch,
                    "zone_id": rec.zone_id,
                }
                canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"))
                rev_hashes.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())

        root_hash = hashlib.sha256(
            ":".join(rev_hashes).encode("utf-8")
        ).hexdigest() if rev_hashes else "ZERO_REVOCATIONS"

        return RevocationStateDigest(
            engine_id=self.engine_identity.engine_id,
            epoch=self.current_epoch,
            revocation_count=rev_count,
            revocation_root_hash=root_hash,
            state_version=self._current_version,
        )

    def compute_peer_digest(
        self,
        registry: Optional[PeerRegistry] = None,
    ) -> PeerStateDigest:
        """
        Compute deterministic digest of registered and active peers.
        """
        reg_count = 0
        act_count = 0
        peer_ids: List[str] = []

        if registry is not None:
            peers = registry.list_peers()
            reg_count = len(peers)
            for p in peers:
                peer_ids.append(p.identity.peer_id)
                if p.is_active(self.current_epoch):
                    act_count += 1

        peer_ids.sort()
        peer_ids_hash = hashlib.sha256(
            json.dumps(peer_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if peer_ids else "NO_PEERS"

        return PeerStateDigest(
            engine_id=self.engine_identity.engine_id,
            zone_id=self.engine_identity.zone_id,
            epoch=self.current_epoch,
            registered_peers_count=reg_count,
            active_peers_count=act_count,
            peer_ids_hash=peer_ids_hash,
            state_version=self._current_version,
        )

    def compute_state_digest(
        self,
        session: Optional[SecurePeerSession] = None,
        registry: Optional[PeerRegistry] = None,
        revocation_records: Optional[List[RevocationRecord]] = None,
    ) -> FederationSecurityStateDigest:
        """
        Compute master composite security state digest across all four sub-states.
        """
        peer_digest = self.compute_peer_digest(registry).compute_digest()
        replay_digest = self.compute_replay_digest(session).compute_digest()
        trust_digest = self.compute_trust_digest(registry).compute_digest()
        revocation_digest = self.compute_revocation_digest(revocation_records).compute_digest()

        return FederationSecurityStateDigest(
            engine_id=self.engine_identity.engine_id,
            zone_id=self.engine_identity.zone_id,
            epoch=self.current_epoch,
            state_version=self._current_version,
            peer_digest=peer_digest,
            replay_digest=replay_digest,
            trust_digest=trust_digest,
            revocation_digest=revocation_digest,
        )

    # ========================================================================
    # Conflict Evaluation (Phase 4 & Phase 9)
    # ========================================================================

    def compare_state_digest(
        self,
        local_digest: FederationSecurityStateDigest,
        remote_digest: FederationSecurityStateDigest,
    ) -> Tuple[bool, str]:
        """
        Compare two composite state digests.
        Returns (is_identical, explanation).
        """
        local_hash = local_digest.compute_composite_digest()
        remote_hash = remote_digest.compute_composite_digest()

        if local_hash == remote_hash:
            return True, "State digests are cryptographically identical."

        # Find specific divergence
        mismatches: List[str] = []
        if local_digest.peer_digest != remote_digest.peer_digest:
            mismatches.append("peer_digest")
        if local_digest.replay_digest != remote_digest.replay_digest:
            mismatches.append("replay_digest")
        if local_digest.trust_digest != remote_digest.trust_digest:
            mismatches.append("trust_digest")
        if local_digest.revocation_digest != remote_digest.revocation_digest:
            mismatches.append("revocation_digest")

        return False, f"Digests diverge in: {', '.join(mismatches)}."

    def evaluate_remote_version(
        self,
        remote_version: SecurityStateVersion,
        remote_digest: Optional[str] = None,
        local_digest: Optional[str] = None,
    ) -> str:
        """
        Deterministic evaluation of remote version according to Phase 9 rules:
        - Case A: Remote is older -> "OLDER" (reject/ignore regression)
        - Case B: Remote is newer -> "NEWER" (eligible for validation and sync)
        - Case C: Same version, divergent digest -> raises StateDigestConflictError
        - Case D: Same version, matching digest -> "SYNCHRONIZED"
        """
        local_ver = self.current_version

        if remote_version.is_stale_compared_to(local_ver):
            return "OLDER"

        if remote_version.is_newer_than(local_ver):
            return "NEWER"

        # Versions match exactly
        if remote_digest and local_digest and remote_digest != local_digest:
            raise StateDigestConflictError(
                f"Same-version state divergence detected between local engine "
                f"'{local_ver.engine_id}' and remote engine '{remote_version.engine_id}' "
                f"at version {local_ver.version} (epoch {local_ver.epoch}). "
                f"Local digest: {local_digest[:16]}..., Remote digest: {remote_digest[:16]}..."
            )

        return "SYNCHRONIZED"
