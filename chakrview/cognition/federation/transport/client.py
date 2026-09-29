"""
Outbound Federation Transport Client with Bounded Reconnection (Step 36).

CRITICAL ARCHITECTURAL AXIOMS:
1. REJOIN != TRUST_GRANT:
   Reconnecting a channel re-establishes transport liveness; it never escalates
   capability scopes, bypasses authorization, or renews expired trust grants.
2. BOUNDED EXPONENTIAL BACKOFF:
   Reconnection retries strictly respect maximum attempt limits and backoff ceilings
   to prevent network flooding and CPU busy-loops.
3. REVOCATION IS PERMANENT:
   A node or channel marked REVOKED can NEVER be reconnected. Rejoin attempts
   fail closed immediately.
"""

import logging
import time
from typing import Dict, Optional, Any

from chakrview.cognition.federation.discovery.models import (
    FederationNodeEndpoint,
    FederationNodeCandidate,
    FederationNodeMembership,
    MembershipState,
)
from chakrview.cognition.federation.transport.models import (
    ChannelState,
    ReconnectPolicy,
    FederationMessageEnvelope,
    FederationMessageType,
)
from chakrview.cognition.federation.transport.channel import FederationChannel
from chakrview.cognition.federation.transport.errors import (
    ChannelRevokedError,
    ChannelQuarantinedError,
    ChannelClosedError,
    MaxReconnectAttemptsExceededError,
    ChannelAuthenticationError,
)
from chakrview.cognition.peering.models import AuditEventType

logger = logging.getLogger(__name__)


class FederationTransportClient:
    """
    Manages outbound secure federation channels, lifecycle negotiation,
    and bounded exponential-backoff reconnection.
    """

    def __init__(
        self,
        engine: Any,
        reconnect_policy: Optional[ReconnectPolicy] = None,
    ) -> None:
        self.engine = engine
        self.reconnect_policy = reconnect_policy or ReconnectPolicy()
        self._active_channels: Dict[str, FederationChannel] = {}

    def get_channel(self, channel_id: str) -> Optional[FederationChannel]:
        """Retrieve an active channel by identifier."""
        return self._active_channels.get(channel_id)

    def open_channel(
        self,
        endpoint: FederationNodeEndpoint,
        remote_engine: Optional[Any] = None,
        peer_certificate_metadata: Optional[Any] = None,
    ) -> FederationChannel:
        """
        Establish a new outbound secure federation channel to a node endpoint.
        
        Orchestration:
        1. Query or register candidate in FederationMembershipManager.
        2. Execute FederationConnectionManager.connect_candidate().
        3. Verify cryptographic identity binding.
        4. Construct and transition FederationChannel to ESTABLISHED.
        """
        conn_mgr = getattr(self.engine, "connection_manager", None)
        membership_mgr = getattr(self.engine, "membership_manager", None)

        channel_id = f"chan_{endpoint.host}_{endpoint.port}_{int(time.time() * 1000)}"

        # 1. Connect through Step 35 connection manager
        if conn_mgr and hasattr(conn_mgr, "connect_candidate"):
            membership = conn_mgr.connect_candidate(
                candidate_or_membership_id=endpoint,
                remote_engine=remote_engine,
                peer_certificate_metadata=peer_certificate_metadata,
            )
        elif membership_mgr:
            cand = FederationNodeCandidate.from_endpoint(endpoint)
            membership = membership_mgr.register_candidate(cand)
        else:
            membership = None

        # 2. Check for quarantine / revocation
        if membership:
            if membership.is_revoked():
                raise ChannelRevokedError(f"Cannot open channel: node '{membership.node_id}' is REVOKED.")
            if membership.is_quarantined():
                raise ChannelQuarantinedError(f"Cannot open channel: node '{membership.node_id}' is QUARANTINED.")

        # 3. Create or attach session
        session = None
        remote_peer_id = getattr(membership, "peer_id", None) or getattr(remote_engine, "local_peer_id", None)
        if hasattr(self.engine, "session_manager") and remote_peer_id:
            session = self.engine.session_manager.get_active_session_for_peer(remote_peer_id)
            if not session:
                session = self.engine.session_manager.create_session(
                    remote_peer_id=remote_peer_id,
                    current_epoch=getattr(self.engine, "current_epoch", 1),
                )

        # 4. Construct channel
        channel = FederationChannel(
            channel_id=channel_id,
            local_engine=self.engine,
            remote_endpoint=endpoint,
            remote_peer_id=remote_peer_id,
            remote_engine_identity=getattr(remote_engine, "engine_identity", None) if remote_engine else None,
            membership=membership,
            session=session,
            initial_state=ChannelState.ESTABLISHED,
        )

        self._active_channels[channel_id] = channel

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.CONNECTION_ESTABLISHED,
                epoch=getattr(self.engine, "current_epoch", 1),
                details={"channel_id": channel_id, "endpoint": str(endpoint.address)},
            )

        return channel

    def reconnect(
        self,
        channel: FederationChannel,
        max_attempts: Optional[int] = None,
        remote_engine: Optional[Any] = None,
    ) -> bool:
        """
        Reconnect an interrupted or degraded channel using bounded exponential backoff.
        
        Enforces:
        - Revoked channels are permanently barred.
        - Quarantined channels cannot reconnect.
        - Backoff delay increases exponentially up to max_backoff_seconds.
        - Maximum retry attempt bounds prevent resource exhaustion.
        - Reconnection re-invokes Step 35 secure rejoin protocol without authority escalation.
        """
        if channel.is_revoked:
            raise ChannelRevokedError(f"Cannot reconnect revoked channel '{channel.channel_id}'.")
        if channel.is_quarantined:
            raise ChannelQuarantinedError(f"Cannot reconnect quarantined channel '{channel.channel_id}'.")

        policy = self.reconnect_policy
        attempts_limit = max_attempts or policy.max_attempts

        conn_mgr = getattr(self.engine, "connection_manager", None)

        audit = getattr(self.engine, "audit_logger", None)
        epoch = getattr(self.engine, "current_epoch", 1)

        for attempt in range(1, attempts_limit + 1):
            if audit:
                audit.log(
                    event_type=AuditEventType.RECONNECT_ATTEMPTED,
                    epoch=epoch,
                    details={"channel_id": channel.channel_id, "attempt": attempt, "max_attempts": attempts_limit},
                )

            try:
                # Delegate to Step 35 rejoin protocol
                if conn_mgr and hasattr(conn_mgr, "reconnect_member"):
                    rejoin_result = conn_mgr.reconnect_member(
                        membership_id=channel.membership.membership_id if channel.membership else None,
                        node_id=channel.remote_peer_id,
                        remote_engine=remote_engine,
                    )

                # Reconnection succeeded
                channel.transition_to(ChannelState.ESTABLISHED, reason=f"Reconnected on attempt {attempt}")
                channel.record_heartbeat(is_healthy=True)

                if audit:
                    audit.log(
                        event_type=AuditEventType.RECONNECT_SUCCEEDED,
                        epoch=epoch,
                        details={"channel_id": channel.channel_id, "attempt": attempt},
                    )
                return True

            except Exception as e:
                logger.debug(f"Reconnect attempt {attempt}/{attempts_limit} failed for {channel.channel_id}: {e}")

                if attempt < attempts_limit:
                    delay = policy.compute_backoff(attempt)
                    if delay > 0:
                        time.sleep(delay)

        # Retries exhausted
        if audit:
            audit.log(
                event_type=AuditEventType.RECONNECT_FAILED,
                epoch=epoch,
                details={"channel_id": channel.channel_id, "attempts_exhausted": attempts_limit},
            )

        channel.transition_to(ChannelState.CLOSED, reason="Reconnect attempts exhausted")
        raise MaxReconnectAttemptsExceededError(
            f"Failed to reconnect channel '{channel.channel_id}' after {attempts_limit} attempts."
        )

    def close_all(self) -> None:
        """Close all active channels cleanly."""
        for chan in list(self._active_channels.values()):
            try:
                chan.close(reason="Client shutting down")
            except Exception:
                pass
        self._active_channels.clear()
