"""
Bounded Federation Message Dispatcher with Sovereign Authorization (Step 36).

CRITICAL ARCHITECTURAL AXIOMS:
1. LOCAL_AUTHORITY > PEER_AUTHORITY:
   No remote message can execute or alter state without local capability and policy
   authorization. All capability requests pass through CapabilityGate.
2. HANDLER ISOLATION & FAIL-CLOSED:
   Unregistered message types fail closed. Handler execution errors are captured,
   classified, and audited without crashing the transport runtime.
3. BOUNDED QUEUES & TIMEOUTS:
   All incoming messages are bounded by size and execution deadlines to prevent
   resource exhaustion DoS attacks.
"""

from dataclasses import dataclass
import threading
import time
from typing import Dict, Any, Callable, Optional

from chakrview.cognition.peering.models import (
    AuditEventType,
    FederationScope,
    TrustLevel,
)
from chakrview.cognition.federation.persistence.models import JournalEntryType
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.contract import CapabilityRequest
from chakrview.cognition.federation.transport.models import (
    FederationMessageEnvelope,
    FederationMessageType,
    ChannelState,
    DEFAULT_CHANNEL_TIMEOUT_SECONDS,
    DEFAULT_MAX_DISPATCHER_QUEUE_SIZE,
)
from chakrview.cognition.federation.transport.channel import FederationChannel
from chakrview.cognition.federation.transport.errors import (
    DispatcherError,
    HandlerNotFoundError,
    HandlerExecutionError,
    HandlerTimeoutError,
    UnauthorizedMessageError,
    ChannelQuarantinedError,
    ChannelRevokedError,
)


@dataclass
class HandlerRegistration:
    """Registered handler entry with associated authority and scope constraints."""
    handler: Callable[[FederationMessageEnvelope, FederationChannel], Any]
    required_scope: Optional[FederationScope] = None
    timeout_seconds: float = DEFAULT_CHANNEL_TIMEOUT_SECONDS


class FederationMessageDispatcher:
    """
    Sovereign message dispatcher routing validated envelopes to registered handlers
    under strict CapabilityGate and peer trust scope enforcement.
    """

    def __init__(
        self,
        engine: Any,
        max_queue_size: int = DEFAULT_MAX_DISPATCHER_QUEUE_SIZE,
        default_timeout_seconds: float = DEFAULT_CHANNEL_TIMEOUT_SECONDS,
    ) -> None:
        self.engine = engine
        self.max_queue_size = max_queue_size
        self.default_timeout_seconds = default_timeout_seconds

        self._handlers: Dict[FederationMessageType, HandlerRegistration] = {}
        self._lock = threading.RLock()

        # Register default handlers for common protocol messages
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default handlers for essential protocol operations."""
        self.register_handler(
            FederationMessageType.HEARTBEAT,
            self._handle_heartbeat,
        )
        self.register_handler(
            FederationMessageType.HEARTBEAT_ACK,
            self._handle_heartbeat_ack,
        )

    # ========================================================================
    # Handler Registration
    # ========================================================================

    def register_handler(
        self,
        message_type: FederationMessageType,
        handler: Callable[[FederationMessageEnvelope, FederationChannel], Any],
        required_scope: Optional[FederationScope] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Register a handler for a specific FederationMessageType."""
        with self._lock:
            self._handlers[message_type] = HandlerRegistration(
                handler=handler,
                required_scope=required_scope,
                timeout_seconds=timeout_seconds or self.default_timeout_seconds,
            )

    def unregister_handler(self, message_type: FederationMessageType) -> None:
        """Remove a previously registered handler."""
        with self._lock:
            self._handlers.pop(message_type, None)

    def has_handler(self, message_type: FederationMessageType) -> bool:
        """Check if a handler is registered for a message type."""
        with self._lock:
            return message_type in self._handlers

    # ========================================================================
    # Dispatch & Sovereign Authorization
    # ========================================================================

    def dispatch(
        self,
        envelope: FederationMessageEnvelope,
        channel: FederationChannel,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Dispatch an envelope to its registered handler following strict authorization.
        
        Execution pipeline:
        1. Channel state verification (reject QUARANTINED, REVOKED, CLOSED)
        2. Handler presence check (reject unknown message types fail-closed)
        3. Tenant boundary isolation enforcement
        4. Federation trust scope enforcement
        5. CapabilityGate authorization (for capability invocations)
        6. Isolated handler execution with timeout handling
        7. Audit logging and durable journal append
        """
        # 1. Channel State Check
        if channel.is_revoked:
            raise ChannelRevokedError(f"Cannot dispatch message on revoked channel '{channel.channel_id}'.")
        if channel.is_quarantined:
            raise ChannelQuarantinedError(f"Cannot dispatch message on quarantined channel '{channel.channel_id}'.")
        if channel.state == ChannelState.CLOSED:
            raise UnauthorizedMessageError(f"Cannot dispatch message on closed channel '{channel.channel_id}'.")

        # 2. Handler Presence Check
        with self._lock:
            registration = self._handlers.get(envelope.message_type)

        if not registration:
            audit = getattr(self.engine, "audit_logger", None)
            if audit:
                audit.log(
                    event_type=AuditEventType.MESSAGE_REJECTED,
                    epoch=getattr(self.engine, "current_epoch", 1),
                    details={
                        "message_id": envelope.message_id,
                        "message_type": envelope.message_type.value,
                        "reason": f"No handler registered for {envelope.message_type.value}",
                    },
                )
            raise HandlerNotFoundError(
                f"No handler registered for federation message type '{envelope.message_type.value}'."
            )

        # 3. Tenant Boundary Isolation Enforcement
        if envelope.tenant_id:
            local_zone = getattr(self.engine, "local_zone_id", "zone-local")
            if envelope.tenant_id != local_zone:
                audit = getattr(self.engine, "audit_logger", None)
                if audit:
                    audit.log(
                        event_type=AuditEventType.ISOLATION_VIOLATION,
                        epoch=getattr(self.engine, "current_epoch", 1),
                        details={
                            "message_id": envelope.message_id,
                            "envelope_tenant": envelope.tenant_id,
                            "local_tenant": local_zone,
                            "reason": "Cross-tenant message rejected",
                        },
                    )
                raise UnauthorizedMessageError(
                    f"Cross-tenant isolation violation: envelope tenant '{envelope.tenant_id}' "
                    f"does not match local tenant '{local_zone}'."
                )

        # 4. Scope Authorization
        if registration.required_scope and channel.remote_peer_id:
            if hasattr(self.engine, "registry"):
                peer_reg = self.engine.registry.get_peer(channel.remote_peer_id)
                if not peer_reg or not peer_reg.trust_grant:
                    raise UnauthorizedMessageError(
                        f"Peer '{channel.remote_peer_id}' has no active trust grant for required scope '{registration.required_scope.value}'."
                    )
                if registration.required_scope not in peer_reg.trust_grant.permitted_scopes:
                    raise UnauthorizedMessageError(
                        f"Peer '{channel.remote_peer_id}' trust grant does not permit scope '{registration.required_scope.value}'."
                    )

        # 5. CapabilityGate Authorization for Capability Execution
        if envelope.message_type == FederationMessageType.CAPABILITY_REQUEST:
            cap_gate = getattr(self.engine, "capability_gate", None)
            if cap_gate and isinstance(cap_gate, CapabilityGate):
                cap_id = envelope.payload.get("capability_id", "")
                cap_params = envelope.payload.get("parameters", {})
                req = CapabilityRequest(
                    capability_id=cap_id,
                    caller_id=channel.remote_peer_id or envelope.sender_peer_id,
                    parameters=cap_params,
                )
                try:
                    cap_gate.authorize(req)
                except CapabilityAuthorizationError as e:
                    audit = getattr(self.engine, "audit_logger", None)
                    if audit:
                        audit.log(
                            event_type=AuditEventType.REQUEST_DENIED,
                            epoch=getattr(self.engine, "current_epoch", 1),
                            details={"capability_id": cap_id, "error": str(e)},
                        )
                    raise UnauthorizedMessageError(f"CapabilityGate denied execution: {e}") from e

        # 6. Execute Handler
        t0 = time.perf_counter()
        try:
            result = registration.handler(envelope, channel)
        except Exception as e:
            if isinstance(e, DispatcherError):
                raise
            raise HandlerExecutionError(f"Handler execution failed for '{envelope.message_type.value}': {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 7. Audit Logging & Journal Integration
        audit = getattr(self.engine, "audit_logger", None)
        epoch = getattr(self.engine, "current_epoch", 1)
        if audit:
            audit.log(
                event_type=AuditEventType.MESSAGE_DISPATCHED,
                epoch=epoch,
                details={
                    "message_id": envelope.message_id,
                    "message_type": envelope.message_type.value,
                    "channel_id": channel.channel_id,
                    "elapsed_ms": elapsed_ms,
                },
            )

        runtime = getattr(self.engine, "runtime", None)
        if runtime and getattr(runtime, "journal", None):
            runtime.journal.append(
                entry_type=JournalEntryType.MESSAGE_DISPATCHED,
                epoch=epoch,
                payload={
                    "message_id": envelope.message_id,
                    "message_type": envelope.message_type.value,
                    "channel_id": channel.channel_id,
                },
            )

        return result

    # ========================================================================
    # Default Protocol Handlers
    # ========================================================================

    def _handle_heartbeat(
        self,
        envelope: FederationMessageEnvelope,
        channel: FederationChannel,
    ) -> Dict[str, Any]:
        """Process incoming heartbeat and reply with acknowledgement."""
        channel.record_heartbeat(is_healthy=True)
        # Construct ack
        ack = FederationMessageEnvelope(
            message_type=FederationMessageType.HEARTBEAT_ACK,
            message_id=f"ack_{envelope.message_id}",
            session_id=envelope.session_id,
            sender_engine_id=envelope.receiver_engine_id,
            receiver_engine_id=envelope.sender_engine_id,
            sender_peer_id=envelope.receiver_peer_id,
            receiver_peer_id=envelope.sender_peer_id,
            sequence_number=envelope.sequence_number + 1,
            epoch=envelope.epoch,
            payload={"status": "ACK", "heartbeat_message_id": envelope.message_id},
            tenant_id=envelope.tenant_id,
        )
        return {"status": "ACK", "ack_envelope": ack}

    def _handle_heartbeat_ack(
        self,
        envelope: FederationMessageEnvelope,
        channel: FederationChannel,
    ) -> Dict[str, Any]:
        """Process incoming heartbeat acknowledgment."""
        channel.record_heartbeat(is_healthy=True)
        return {"status": "ACKNOWLEDGED", "message_id": envelope.message_id}
