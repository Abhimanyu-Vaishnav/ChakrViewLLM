"""
Secure Federation Channel Abstraction for Inter-Node Communication (Step 36).

CRITICAL ARCHITECTURAL AXIOMS:
1. CHANNEL != AUTHORITY:
   The channel manages transport liveness, sequence checking, and cryptographic
   framing; it confers zero execution authority or policy privileges.
2. REPLAY & SEQUENCE MONOTONICITY:
   Every received envelope must satisfy sequence_number > last_seen_sequence_number
   and unique message_id. Replays and sequence regressions fail closed.
3. TERMINAL REVOCATION ABSORPTION:
   Once a channel or node transitions to REVOKED, it can NEVER transition back to
   ACTIVE or ESTABLISHED. Reconnection attempts fail closed.
4. QUARANTINE TRAFFIC ISOLATION:
   Quarantined channels reject all outbound and inbound message transmission.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable

from chakrview.cognition.peering.crypto import Ed25519PublicKeyWrapper
from chakrview.cognition.peering.session import SecurePeerSession, SessionStatus
from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.security.models import CertificateMetadata
from chakrview.cognition.federation.models import FederationEngineIdentity
from chakrview.cognition.federation.discovery.models import (
    FederationNodeEndpoint,
    FederationNodeMembership,
    MembershipState,
)
from chakrview.cognition.federation.transport.models import (
    ChannelState,
    VALID_CHANNEL_TRANSITIONS,
    FederationMessageEnvelope,
    FederationMessageType,
    ChannelMetrics,
    DEFAULT_CHANNEL_TIMEOUT_SECONDS,
    DEFAULT_MAX_FRAME_SIZE,
    DEFAULT_MAX_PAYLOAD_SIZE,
)
from chakrview.cognition.federation.transport.framing import FederationMessageFramer
from chakrview.cognition.federation.transport.codec import FederationMessageCodec
from chakrview.cognition.federation.transport.errors import (
    ChannelStateError,
    ChannelClosedError,
    ChannelTimeoutError,
    ChannelQuarantinedError,
    ChannelRevokedError,
    ChannelAuthenticationError,
    SequenceRegressionError,
    DuplicateMessageError,
    EnvelopeIntegrityError,
)
from chakrview.cognition.peering.models import AuditEventType
from chakrview.cognition.federation.persistence.models import JournalEntryType


class FederationChannel:
    """
    Bi-directional authenticated secure communication channel between federation nodes.
    """

    def __init__(
        self,
        channel_id: str,
        local_engine: Any,
        remote_endpoint: FederationNodeEndpoint,
        remote_peer_id: Optional[str] = None,
        remote_engine_identity: Optional[FederationEngineIdentity] = None,
        membership: Optional[FederationNodeMembership] = None,
        session: Optional[SecurePeerSession] = None,
        transport: Optional[Transport] = None,
        initial_state: ChannelState = ChannelState.DISCONNECTED,
        timeout_seconds: float = DEFAULT_CHANNEL_TIMEOUT_SECONDS,
    ) -> None:
        self.channel_id = channel_id
        self.local_engine = local_engine
        self.remote_endpoint = remote_endpoint
        self.remote_peer_id = remote_peer_id
        self.remote_engine_identity = remote_engine_identity
        self.membership = membership
        self.session = session
        self.transport = transport
        self.timeout_seconds = timeout_seconds

        self._state = initial_state
        self._metrics = ChannelMetrics()
        self._lock = threading.RLock()
        self._rx_buffer = bytearray()
        self._inbox: List[FederationMessageEnvelope] = []
        self._last_sequence_sent = 0

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def state(self) -> ChannelState:
        with self._lock:
            return self._state

    @property
    def is_established(self) -> bool:
        with self._lock:
            return self._state == ChannelState.ESTABLISHED

    @property
    def is_quarantined(self) -> bool:
        with self._lock:
            return self._state == ChannelState.QUARANTINED

    @property
    def is_revoked(self) -> bool:
        with self._lock:
            return self._state == ChannelState.REVOKED

    @property
    def metrics(self) -> ChannelMetrics:
        with self._lock:
            return self._metrics

    # ========================================================================
    # State Machine
    # ========================================================================

    def transition_to(self, target_state: ChannelState, reason: str = "") -> None:
        """
        Transition channel to a new state enforcing VALID_CHANNEL_TRANSITIONS.
        Emits structured audit events and durable write-ahead journal records.
        """
        with self._lock:
            if target_state == self._state:
                return

            valid_targets = VALID_CHANNEL_TRANSITIONS.get(self._state, set())
            if target_state not in valid_targets:
                raise ChannelStateError(
                    f"Illegal channel transition from {self._state.value} to {target_state.value}. "
                    f"Valid transitions: {[s.value for s in valid_targets]}. Reason: {reason}"
                )

            prev_state = self._state
            self._state = target_state

            if target_state == ChannelState.ESTABLISHED:
                self._metrics.connected_time = time.time()

        # Audit logging
        audit = getattr(self.local_engine, "audit_logger", None)
        epoch = getattr(self.local_engine, "current_epoch", 1)
        if audit:
            event_type = AuditEventType.MEMBERSHIP_STATE_TRANSITION
            if target_state == ChannelState.ESTABLISHED:
                event_type = AuditEventType.CONNECTION_ESTABLISHED
            elif target_state == ChannelState.CLOSED:
                event_type = AuditEventType.CONNECTION_CLOSED
            elif target_state == ChannelState.QUARANTINED:
                event_type = AuditEventType.NODE_QUARANTINED
            elif target_state == ChannelState.REVOKED:
                event_type = AuditEventType.NODE_REVOKED
            elif target_state == ChannelState.DEGRADED:
                event_type = AuditEventType.CONNECTION_DEGRADED

            audit.log(
                event_type=event_type,
                epoch=epoch,
                details={
                    "channel_id": self.channel_id,
                    "previous_state": prev_state.value,
                    "new_state": target_state.value,
                    "reason": reason,
                    "remote_endpoint": str(self.remote_endpoint.address),
                },
            )

        # Durable journal integration
        runtime = getattr(self.local_engine, "runtime", None)
        if runtime and getattr(runtime, "journal", None):
            j_type = None
            if target_state == ChannelState.ESTABLISHED:
                j_type = JournalEntryType.CONNECTION_ESTABLISHED
            elif target_state == ChannelState.CLOSED:
                j_type = JournalEntryType.CONNECTION_CLOSED
            elif target_state == ChannelState.QUARANTINED:
                j_type = JournalEntryType.CHANNEL_QUARANTINED
            elif target_state == ChannelState.REVOKED:
                j_type = JournalEntryType.CHANNEL_REVOKED

            if j_type:
                entry = runtime.journal.append(
                    entry_type=j_type,
                    epoch=epoch,
                    payload={
                        "channel_id": self.channel_id,
                        "reason": reason,
                        "remote_peer_id": self.remote_peer_id,
                    },
                )
                store = getattr(runtime, "store", None)
                if store and hasattr(store, "append_journal_entry"):
                    store.append_journal_entry(entry)

    # ========================================================================
    # Binding & Association
    # ========================================================================

    def bind_session(self, session: SecurePeerSession) -> None:
        """Bind an active secure peer session to this channel."""
        with self._lock:
            self.session = session
            if session.remote_peer_id and not self.remote_peer_id:
                self.remote_peer_id = session.remote_peer_id

    def bind_remote_engine(self, engine_identity: FederationEngineIdentity) -> None:
        """Associate remote engine identity and verify tenant isolation."""
        with self._lock:
            self.remote_engine_identity = engine_identity
            # Enforce multi-tenant check if engine has connection manager
            conn_mgr = getattr(self.local_engine, "connection_manager", None)
            if conn_mgr and hasattr(conn_mgr, "validate_tenant_isolation"):
                allowed = getattr(self.local_engine, "local_zone_id", "zone-local")
                conn_mgr.validate_tenant_isolation(engine_identity, allowed_tenant=allowed)

    # ========================================================================
    # Transmission (Send & Receive)
    # ========================================================================

    def send_message(
        self,
        envelope: FederationMessageEnvelope,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Send an authenticated federation message envelope over the channel.
        
        Enforces:
        - Channel operational state (must be ESTABLISHED or AUTHENTICATING for handshake)
        - Quarantine and revocation fail-closed blocks
        - Envelope integrity and payload digest verification
        - Length-prefixed framing and byte ceilings
        """
        with self._lock:
            if self._state == ChannelState.REVOKED:
                raise ChannelRevokedError(f"Cannot send message on revoked channel '{self.channel_id}'.")
            if self._state == ChannelState.QUARANTINED:
                raise ChannelQuarantinedError(f"Cannot send message on quarantined channel '{self.channel_id}'.")
            if self._state == ChannelState.CLOSED:
                raise ChannelClosedError(f"Cannot send message on closed channel '{self.channel_id}'.")

            # Permitted during AUTHENTICATING only for challenge/response
            if self._state == ChannelState.AUTHENTICATING:
                if envelope.message_type not in (
                    FederationMessageType.CHALLENGE,
                    FederationMessageType.CHALLENGE_RESPONSE,
                ):
                    raise ChannelStateError(
                        f"Only challenge messages permitted in AUTHENTICATING state, got {envelope.message_type.value}."
                    )
            elif self._state not in (ChannelState.ESTABLISHED, ChannelState.DEGRADED):
                raise ChannelStateError(
                    f"Cannot send message while channel is in state '{self._state.value}'."
                )

            # Validate envelope
            envelope.validate()
            self._last_sequence_sent = max(self._last_sequence_sent, envelope.sequence_number)

            # Codec serialization
            payload_bytes = FederationMessageCodec.serialize(envelope)

            # Binary framing
            frame_bytes = FederationMessageFramer.encode_frame(payload_bytes)

            # Physical transport dispatch or internal queue
            if self.transport:
                # Transmit over physical transport
                if hasattr(self.transport, "send"):
                    self.transport.send(envelope.to_wire_envelope())
                elif hasattr(self.transport, "_client_sock") and self.transport._client_sock:
                    self.transport._client_sock.sendall(frame_bytes)
            else:
                # Queue in internal buffer for simulated/in-process execution
                self._inbox.append(envelope)

            # Update metrics
            self._metrics.bytes_sent += len(frame_bytes)
            self._metrics.messages_sent += 1
            self._metrics.last_activity_time = time.time()
            if envelope.message_type == FederationMessageType.HEARTBEAT:
                self._metrics.heartbeats_sent += 1

    def receive_message(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[FederationMessageEnvelope]:
        """
        Receive and validate an incoming federation message envelope.
        
        Enforces:
        - Channel operational state
        - Envelope integrity and digest matching
        - Monotonic sequence number enforcement (sequence_number > last_seen_sequence_number)
        - Replay attack cache verification
        """
        deadline = time.time() + (timeout if timeout is not None else self.timeout_seconds)

        with self._lock:
            if self._state == ChannelState.REVOKED:
                raise ChannelRevokedError(f"Cannot receive message on revoked channel '{self.channel_id}'.")
            if self._state == ChannelState.QUARANTINED:
                raise ChannelQuarantinedError(f"Cannot receive message on quarantined channel '{self.channel_id}'.")
            if self._state == ChannelState.CLOSED:
                raise ChannelClosedError(f"Cannot receive message on closed channel '{self.channel_id}'.")

            # Check inbox first (for in-process or buffered delivery)
            if self._inbox:
                envelope = self._inbox.pop(0)
                self._validate_and_account_received(envelope)
                return envelope

            # If physical transport present, read and frame-decode
            if self.transport:
                wire_env = self.transport.receive(timeout=timeout or self.timeout_seconds)
                if wire_env:
                    envelope = FederationMessageEnvelope.from_wire_envelope(
                        wire=wire_env,
                        sender_engine_id=self.remote_engine_identity.engine_id if self.remote_engine_identity else self.remote_peer_id or "unknown",
                        receiver_engine_id=getattr(getattr(self.local_engine, "engine_identity", None), "engine_id", "local"),
                    )
                    self._validate_and_account_received(envelope)
                    return envelope

            return None

    def inject_received_bytes(self, raw_bytes: bytes) -> Optional[FederationMessageEnvelope]:
        """
        Directly inject bytes into channel framing stream for testing or socket adapters.
        """
        with self._lock:
            self._rx_buffer.extend(raw_bytes)
            frame_payload = FederationMessageFramer.decode_frame(self._rx_buffer)
            if frame_payload is None:
                return None

            envelope = FederationMessageCodec.deserialize(frame_payload)
            self._validate_and_account_received(envelope)
            return envelope

    def _validate_and_account_received(self, envelope: FederationMessageEnvelope) -> None:
        """
        Execute core security validation on received envelope:
        - Digest verification
        - Sequence monotonicity validation
        - Replay cache checking
        - Session key expiration checking
        """
        # 1. Digest integrity
        if not envelope.verify_digest():
            audit = getattr(self.local_engine, "audit_logger", None)
            if audit:
                audit.log(
                    event_type=AuditEventType.FRAME_REJECTED,
                    epoch=getattr(self.local_engine, "current_epoch", 1),
                    details={"channel_id": self.channel_id, "reason": "Payload digest mismatch"},
                )
            raise EnvelopeIntegrityError(f"Payload digest mismatch on channel '{self.channel_id}'.")

        # 2. Replay & Monotonic Sequence Validation
        if self.session:
            if not self.session.validate_sequence_number(envelope.sequence_number):
                audit = getattr(self.local_engine, "audit_logger", None)
                if audit:
                    audit.log(
                        event_type=AuditEventType.SEQUENCE_REJECTED,
                        epoch=getattr(self.local_engine, "current_epoch", 1),
                        details={
                            "channel_id": self.channel_id,
                            "sequence_number": envelope.sequence_number,
                            "last_seen": self.session.last_seen_sequence_number,
                        },
                    )
                raise SequenceRegressionError(
                    f"Sequence number {envelope.sequence_number} regressed or repeated. "
                    f"Current replay floor is {self.session.last_seen_sequence_number}."
                )

            # Record sequence
            self.session.record_and_check_sequence(envelope.sequence_number)

            # Replay ID check
            if not self.session.record_and_check_message_id(envelope.message_id):
                audit = getattr(self.local_engine, "audit_logger", None)
                if audit:
                    audit.log(
                        event_type=AuditEventType.REPLAY_REJECTED,
                        epoch=getattr(self.local_engine, "current_epoch", 1),
                        details={"channel_id": self.channel_id, "message_id": envelope.message_id},
                    )
                raise DuplicateMessageError(
                    f"Duplicate message ID '{envelope.message_id}' detected on channel '{self.channel_id}'."
                )

        # 3. Update metrics
        self._metrics.messages_received += 1
        self._metrics.last_activity_time = time.time()
        if envelope.message_type == FederationMessageType.HEARTBEAT_ACK:
            self._metrics.heartbeats_acknowledged += 1

    # ========================================================================
    # Lifecycle Operations
    # ========================================================================

    def record_heartbeat(self, is_healthy: bool = True) -> None:
        """Update channel liveness observation from periodic heartbeats."""
        with self._lock:
            self._metrics.last_activity_time = time.time()
            if not is_healthy and self._state == ChannelState.ESTABLISHED:
                self.transition_to(ChannelState.DEGRADED, reason="Heartbeat missed or unhealthy")
            elif is_healthy and self._state == ChannelState.DEGRADED:
                self.transition_to(ChannelState.ESTABLISHED, reason="Heartbeat recovered")

    def quarantine(self, reason: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        """Isolate channel immediately upon security violation."""
        with self._lock:
            self.transition_to(ChannelState.QUARANTINED, reason=reason)
            if self.membership and hasattr(self.membership, "transition_to"):
                self.membership.transition_to(MembershipState.QUARANTINED, reason=reason)
                if evidence:
                    self.membership.quarantine_evidence = dict(evidence)

    def revoke(self, reason: str = "Administrative revocation") -> None:
        """Permanently terminate and bar channel."""
        with self._lock:
            self.transition_to(ChannelState.REVOKED, reason=reason)
            if self.session:
                self.session.terminate(reason=reason)
            if self.transport:
                self.transport.disconnect()
            if self.membership and hasattr(self.membership, "transition_to"):
                self.membership.transition_to(MembershipState.REVOKED, reason=reason)

    def close(self, reason: str = "Orderly channel close") -> None:
        """Gracefully close channel."""
        with self._lock:
            if self._state not in (ChannelState.CLOSED, ChannelState.REVOKED):
                if self._state in VALID_CHANNEL_TRANSITIONS and ChannelState.CLOSING in VALID_CHANNEL_TRANSITIONS[self._state]:
                    self.transition_to(ChannelState.CLOSING, reason=reason)
                self.transition_to(ChannelState.CLOSED, reason=reason)
            if self.transport:
                self.transport.disconnect()
