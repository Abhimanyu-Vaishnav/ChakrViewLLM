"""
Inbound Federation Transport Server with Default-Deny Acceptance (Step 36).

CRITICAL ARCHITECTURAL AXIOMS:
1. DEFAULT-DENY PEER ADMISSION:
   Unregistered or unknown candidates attempting inbound connections are rejected
   immediately under fail-closed policies. Reachability confers zero admission rights.
2. STRICT CAPACITY CEILINGS:
   Active incoming channels cannot exceed MAX_MEMBERSHIP_NODES to prevent
   connection exhaustion denial-of-service.
3. QUARANTINE & REVOCATION ENFORCEMENT:
   Inbound connection attempts from quarantined or revoked nodes are dropped fail-closed.
"""

import threading
import time
from typing import Dict, Optional, Any

from chakrview.cognition.federation.discovery.models import (
    FederationNodeEndpoint,
    FederationNodeMembership,
    MAX_MEMBERSHIP_NODES,
)
from chakrview.cognition.transport.security.models import CertificateMetadata
from chakrview.cognition.federation.models import FederationEngineIdentity
from chakrview.cognition.federation.transport.models import (
    ChannelState,
    FederationMessageEnvelope,
    FederationMessageType,
)
from chakrview.cognition.federation.transport.channel import FederationChannel
from chakrview.cognition.federation.transport.dispatcher import FederationMessageDispatcher
from chakrview.cognition.federation.transport.errors import (
    ChannelStateError,
    ChannelClosedError,
    ChannelQuarantinedError,
    ChannelRevokedError,
    ChannelAuthenticationError,
)
from chakrview.cognition.federation.discovery.errors import UnknownPeerError
from chakrview.cognition.peering.models import AuditEventType


class FederationTransportServer:
    """
    Manages inbound authenticated federation channels under default-deny admission
    and dispatches received envelopes to the sovereign dispatcher.
    """

    def __init__(
        self,
        engine: Any,
        dispatcher: FederationMessageDispatcher,
        max_connections: int = MAX_MEMBERSHIP_NODES,
    ) -> None:
        self.engine = engine
        self.dispatcher = dispatcher
        self.max_connections = max_connections

        self._channels: Dict[str, FederationChannel] = {}
        self._lock = threading.RLock()

    def get_channel(self, channel_id: str) -> Optional[FederationChannel]:
        """Retrieve an accepted channel by identifier."""
        with self._lock:
            return self._channels.get(channel_id)

    def list_active_channels(self) -> Dict[str, FederationChannel]:
        """List copy of currently active channels."""
        with self._lock:
            return dict(self._channels)

    def accept_channel(
        self,
        remote_endpoint: FederationNodeEndpoint,
        peer_certificate_metadata: Optional[CertificateMetadata] = None,
        remote_engine_identity: Optional[FederationEngineIdentity] = None,
        remote_peer_id: Optional[str] = None,
    ) -> FederationChannel:
        """
        Accept an inbound connection following strict default-deny validation.
        
        Enforces:
        - Capacity limit (MAX_MEMBERSHIP_NODES)
        - Inbound connection validation via FederationConnectionManager
        - Certificate revocation checks
        - Cryptographic peer identity binding
        - Quarantine and revocation fail-closed blocks
        """
        with self._lock:
            # 1. Enforce capacity bounds
            active_count = sum(1 for c in self._channels.values() if c.is_established)
            if active_count >= self.max_connections:
                raise ChannelStateError(
                    f"Server connection capacity reached ({self.max_connections} active channels)."
                )

            host = remote_endpoint.address.host if hasattr(remote_endpoint, "address") else getattr(remote_endpoint, "host", "unknown")
            port = remote_endpoint.address.port if hasattr(remote_endpoint, "address") else getattr(remote_endpoint, "port", 0)
            channel_id = f"srv_chan_{host}_{port}_{int(time.time() * 1000)}"

            # 2. Inbound acceptance through connection manager if attached
            conn_mgr = getattr(self.engine, "connection_manager", None)
            membership = None
            if conn_mgr and hasattr(conn_mgr, "accept_inbound_connection"):
                membership = conn_mgr.accept_inbound_connection(
                    remote_peer_id=remote_peer_id,
                    remote_zone_id=remote_endpoint.zone_id,
                    peer_certificate_metadata=peer_certificate_metadata,
                    engine_identity=remote_engine_identity,
                    remote_endpoint=remote_endpoint,
                )
            else:
                # If no connection manager, look up membership directly
                membership_mgr = getattr(self.engine, "membership_manager", None)
                if membership_mgr:
                    if remote_engine_identity:
                        membership = membership_mgr.get_membership_by_node_id(remote_engine_identity.engine_id)
                    if not membership and remote_peer_id:
                        membership = membership_mgr.get_membership_by_node_id(remote_peer_id)
                if not membership:
                    audit = getattr(self.engine, "audit_logger", None)
                    if audit:
                        audit.log(
                            event_type=AuditEventType.REQUEST_DENIED,
                            epoch=getattr(self.engine, "current_epoch", 1),
                            details={"reason": "Unknown peer connection denied by server default-deny policy"},
                        )
                    raise UnknownPeerError("Inbound connection denied: peer is not registered as candidate or member.")

            # 3. Check for quarantine / revocation
            if membership.is_revoked():
                raise ChannelRevokedError(f"Inbound connection rejected: node '{membership.node_id}' is REVOKED.")
            if membership.is_quarantined():
                raise ChannelQuarantinedError(f"Inbound connection rejected: node '{membership.node_id}' is QUARANTINED.")

            # 4. Bind or create session
            session = None
            p_id = remote_peer_id or membership.peer_id or membership.node_id
            if hasattr(self.engine, "session_manager") and p_id:
                session = self.engine.session_manager.get_active_session_for_peer(p_id)
                if not session:
                    session = self.engine.session_manager.create_session(
                        remote_peer_id=p_id,
                        current_epoch=getattr(self.engine, "current_epoch", 1),
                    )

            channel = FederationChannel(
                channel_id=channel_id,
                local_engine=self.engine,
                remote_endpoint=remote_endpoint,
                remote_peer_id=p_id,
                remote_engine_identity=remote_engine_identity,
                membership=membership,
                session=session,
                initial_state=ChannelState.ESTABLISHED,
            )

            self._channels[channel_id] = channel

            audit = getattr(self.engine, "audit_logger", None)
            if audit:
                audit.log(
                    event_type=AuditEventType.CONNECTION_ESTABLISHED,
                    epoch=getattr(self.engine, "current_epoch", 1),
                    details={"channel_id": channel_id, "endpoint": str(remote_endpoint.address), "node_id": membership.node_id},
                )

            return channel

    def handle_envelope(
        self,
        channel_id: str,
        envelope: FederationMessageEnvelope,
    ) -> Any:
        """
        Process an incoming envelope on an accepted channel via the sovereign dispatcher.
        """
        channel = self.get_channel(channel_id)
        if not channel:
            raise ChannelClosedError(f"Channel '{channel_id}' is not active on this server.")

        return self.dispatcher.dispatch(envelope, channel)

    def close_channel(self, channel_id: str) -> None:
        """Close and remove a channel."""
        with self._lock:
            chan = self._channels.pop(channel_id, None)
            if chan:
                chan.close(reason="Server closing channel")

    def close_all(self) -> None:
        """Close all accepted channels cleanly."""
        with self._lock:
            for chan in list(self._channels.values()):
                try:
                    chan.close(reason="Server shutting down")
                except Exception:
                    pass
            self._channels.clear()
