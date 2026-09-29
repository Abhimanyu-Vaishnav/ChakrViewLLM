"""
Distributed Replay State Synchronization (Step 33).

CRITICAL ARCHITECTURAL AXIOMS:
1. LOCAL REPLAY PROTECTION IS UNCONDITIONALLY AUTHORITATIVE:
   Replay synchronization is consistency-oriented and advisory; it NEVER disables,
   replaces, or relaxes local session replay defenses.
2. MISSING SYNC != AUTHORIZATION:
   A missing or delayed replay sync message never authorizes traffic.
3. CONFLICTS FAIL CLOSED:
   Ambiguity or divergent sequence tracking fails closed where security is affected.
4. BOUNDED MEMORY:
   Memory growth is strictly capped; FIFO eviction prevents unbounded tracking.
5. ZERO SECRET LEAKAGE:
   Message payloads, private keys, and session keys are never synchronized.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import json

from chakrview.cognition.federation.models import (
    ReplaySyncMessage,
    MAX_SYNC_MESSAGE_IDS,
)
from chakrview.cognition.federation.errors import (
    ReplaySyncError,
    StaleStateError,
)
from chakrview.cognition.peering.session import SecurePeerSession


class ReplayStateSynchronizer:
    """
    Coordinates bounded exchange and consistency checks of replay state across engines.
    """

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        # Advisory tracking of highest sequence number observed across all engines per session
        self._highest_known_sequences: Dict[str, int] = {}

    def generate_sync_message(
        self,
        session: SecurePeerSession,
        epoch: int,
        state_version: int,
    ) -> ReplaySyncMessage:
        """
        Generate a bounded advisory replay sync message for an active session.
        """
        recent_ids = list(session.seen_message_ids)[-MAX_SYNC_MESSAGE_IDS:]
        recent_ids.sort()

        canonical_ids = json.dumps(recent_ids, separators=(",", ":"))
        digest = hashlib.sha256(canonical_ids.encode("utf-8")).hexdigest()

        return ReplaySyncMessage(
            engine_id=self.engine_id,
            session_id=session.session_id,
            epoch=epoch,
            highest_sequence_number=session.last_seen_sequence_number,
            bounded_message_id_digest=digest,
            recent_message_ids=recent_ids,
            state_version=state_version,
        )

    def ingest_sync_message(
        self,
        session: SecurePeerSession,
        sync_message: ReplaySyncMessage,
    ) -> Dict[str, Any]:
        """
        Ingest an advisory replay synchronization message from a remote engine.
        Returns telemetry summary of the synchronization action.
        """
        if sync_message.session_id != session.session_id:
            raise ReplaySyncError(
                f"Session ID mismatch: sync message for '{sync_message.session_id}' "
                f"presented to local session '{session.session_id}'."
            )

        # Validate digest integrity over received recent message IDs
        sorted_received_ids = sorted(sync_message.recent_message_ids)
        computed_digest = hashlib.sha256(
            json.dumps(sorted_received_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        if computed_digest != sync_message.bounded_message_id_digest:
            raise ReplaySyncError(
                f"Replay sync message digest mismatch: expected '{computed_digest}', "
                f"got '{sync_message.bounded_message_id_digest}'."
            )

        # Ingest message IDs into local bounded session replay cache
        new_ids_count = 0
        duplicate_ids_count = 0
        for msg_id in sync_message.recent_message_ids:
            if msg_id in session.seen_message_ids:
                duplicate_ids_count += 1
            else:
                # Add to local bounded cache
                session.record_and_check_message_id(msg_id)
                new_ids_count += 1

        # Evaluate sequence monotonicity
        local_seq = session.last_seen_sequence_number
        remote_seq = sync_message.highest_sequence_number
        seq_action = "UNCHANGED"

        if remote_seq > local_seq:
            # Advance local sequence floor to prevent replays of older messages
            session.last_seen_sequence_number = remote_seq
            self._highest_known_sequences[session.session_id] = remote_seq
            seq_action = "ADVANCED"
        elif remote_seq < local_seq:
            # Stale remote report; local protection remains higher
            seq_action = "REMOTE_STALE"

        return {
            "session_id": session.session_id,
            "remote_engine_id": sync_message.engine_id,
            "new_message_ids_absorbed": new_ids_count,
            "duplicate_message_ids_ignored": duplicate_ids_count,
            "local_sequence_before": local_seq,
            "local_sequence_after": session.last_seen_sequence_number,
            "sequence_action": seq_action,
        }
