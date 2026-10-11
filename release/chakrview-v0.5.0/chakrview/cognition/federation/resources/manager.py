"""
Federation Resource & Capability Advertisement Manager (Step 37).

Orchestrates:
1. Local hardware discovery and dynamic capacity detection.
2. Sovereign local sharing policy evaluation and privacy filtering.
3. Cryptographic signing and publishing of ResourceAdvertisement envelopes.
4. Inbound advertisement validation, replay defense, and registry storage.
5. Integration with Step 36 FederationChannel, Dispatcher, and Step 35 HeartbeatMonitor.

Axioms:
- RESOURCE_DISCOVERY != RESOURCE_AUTHORIZATION
- RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION
- LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
- Zero ambient execution rights
"""

import hashlib
import threading
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
)
from chakrview.cognition.peering.models import AuditEventType, DiscoveryStatus
from chakrview.cognition.federation.transport.models import (
    FederationMessageType,
    FederationMessageEnvelope,
)
from chakrview.cognition.federation.transport.channel import FederationChannel
from chakrview.cognition.federation.resources.models import (
    NodeResourceProfile,
    AdvertisedCapability,
    ResourceAdvertisement,
    ResourceSharingPolicy,
    ExecutionType,
    AdvertisementFreshness,
)
from chakrview.cognition.federation.resources.discovery import LocalResourceDetector
from chakrview.cognition.federation.resources.registry import FederationResourceRegistry
from chakrview.cognition.federation.resources.errors import (
    FederationResourceError,
    InvalidAdvertisementError,
    ResourcePolicyError,
    UnauthorizedResourceAccessError,
)


class FederationResourceManager:
    """
    High-level manager for distributed resource & capability advertisement.
    """

    def __init__(
        self,
        engine: Any,
        sharing_policy: Optional[ResourceSharingPolicy] = None,
        detector: Optional[LocalResourceDetector] = None,
        coarse_privacy_default: bool = True,
    ) -> None:
        self.engine = engine
        self.sharing_policy = sharing_policy or ResourceSharingPolicy(
            coarse_privacy_enabled=coarse_privacy_default
        )
        self.detector = detector or LocalResourceDetector(
            node_epoch=getattr(engine, "current_epoch", 1),
            coarse_privacy_default=coarse_privacy_default,
        )

        local_peer_id = getattr(engine, "local_peer_id", "local_node")
        local_zone_id = getattr(engine, "local_zone_id", "local_zone")
        local_tenant_id = getattr(engine, "tenant_id", "default")

        self.registry = FederationResourceRegistry(
            local_node_id=local_peer_id,
            local_zone_id=local_zone_id,
            local_tenant_id=local_tenant_id,
            sharing_policy=self.sharing_policy,
        )

        self._lock = threading.RLock()
        self._advertisement_version_counter: int = 1

        # Perform initial local resource detection
        self.refresh_local_resources()

        # Register message handlers with dispatcher if available
        self._register_dispatcher_handlers()

    # ========================================================================
    # 1. Local Resource Refresh & Registration
    # ========================================================================

    def refresh_local_resources(self) -> NodeResourceProfile:
        """
        Refresh local hardware observation and register ground truth profile.
        Volatile metrics are updated; static capabilities remain consistent.
        """
        with self._lock:
            # 1. Detect hardware profile
            profile = self.detector.detect_profile(coarse=self.sharing_policy.coarse_privacy_enabled)

            # 2. Detect local capabilities
            cap_registry = getattr(self.engine, "capability_gate", None)
            local_cap_reg = getattr(cap_registry, "registry", None) if cap_registry else None
            capabilities = self.detector.detect_capabilities(registry=local_cap_reg)

            # 3. Register in local resource registry
            self.registry.register_local_profile(
                profile=profile,
                capabilities=capabilities,
                policy=self.sharing_policy,
            )

            audit = getattr(self.engine, "audit_logger", None)
            if audit:
                audit.log(
                    event_type=AuditEventType.RESOURCE_PROFILE_DETECTED,
                    epoch=getattr(self.engine, "current_epoch", 1),
                    details={
                        "profile_id": profile.profile_id,
                        "cores": profile.cpu.logical_cores,
                        "total_mem_mb": profile.memory.total_memory_mb,
                        "capabilities_count": len(capabilities),
                    },
                )

            return profile

    # ========================================================================
    # 2. Advertisement Creation & Publishing
    # ========================================================================

    def create_local_advertisement(self) -> ResourceAdvertisement:
        """
        Construct and cryptographically sign a fresh ResourceAdvertisement for this node.
        Applies local ResourceSharingPolicy and privacy filtering.
        """
        with self._lock:
            profile = self.registry.get_local_profile()
            if not profile:
                profile = self.refresh_local_resources()

            capabilities = self.registry.get_local_capabilities(exposed_only=True)

            local_peer_id = getattr(self.engine, "local_peer_id", "local_node")
            local_engine_id = getattr(self.engine, "engine_identity", None)
            engine_id_str = local_engine_id.engine_id if local_engine_id else local_peer_id
            local_zone_id = getattr(self.engine, "local_zone_id", "local_zone")
            epoch = getattr(self.engine, "current_epoch", 1)

            version = self._advertisement_version_counter
            self._advertisement_version_counter += 1

            hasher = hashlib.sha256()
            hasher.update(f"{local_peer_id}:{epoch}:{version}:{time.time()}".encode("utf-8"))
            adv_id = f"adv_{hasher.hexdigest()[:16]}"

            local_tenant_id = getattr(self.engine, "tenant_id", "default")
            raw_adv = ResourceAdvertisement(
                advertisement_id=adv_id,
                node_id=local_peer_id,
                engine_id=engine_id_str,
                zone_id=local_zone_id,
                tenant_id=local_tenant_id,
                version=version,
                epoch=epoch,
                resource_profile=profile,
                capabilities=capabilities,
                timestamp=time.time(),
                ttl_seconds=self.sharing_policy.ttl_seconds,
            )

            # Apply local policy filter and privacy sanitization
            filtered_adv = self.sharing_policy.filter_advertisement(raw_adv)

            # Sign with engine private key
            local_key = getattr(self.engine, "local_private_key", None)
            if local_key:
                filtered_adv.sign(local_key)

            return filtered_adv

    def publish_advertisement(
        self,
        channel: Optional[FederationChannel] = None,
    ) -> ResourceAdvertisement:
        """
        Publish local resource advertisement over secure federation transport.
        If channel is provided, sends to specific peer; otherwise broadcasts across client channels.
        """
        adv = self.create_local_advertisement()

        # Wrap in Step 36 FederationMessageEnvelope
        envelope = FederationMessageEnvelope(
            message_type=FederationMessageType.RESOURCE_ADVERTISEMENT,
            message_id=f"msg_{adv.advertisement_id}",
            session_id=channel.session_id if channel and channel.session else "sess_broadcast",
            sender_engine_id=adv.engine_id,
            receiver_engine_id=channel.remote_peer_id if channel else "*",
            sender_peer_id=adv.node_id,
            receiver_peer_id=channel.remote_peer_id if channel else "*",
            sequence_number=1,
            epoch=adv.epoch,
            payload=adv.to_dict(),
            tenant_id=adv.tenant_id,
        )

        local_key = getattr(self.engine, "local_private_key", None)
        if local_key:
            envelope.sign(local_key)

        if channel:
            channel.send_message(envelope)

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.RESOURCE_ADVERTISEMENT_PUBLISHED,
                epoch=adv.epoch,
                details={
                    "advertisement_id": adv.advertisement_id,
                    "target": channel.channel_id if channel else "broadcast",
                    "version": adv.version,
                },
            )

        return adv

    # ========================================================================
    # 3. Inbound Dispatcher Handlers
    # ========================================================================

    def _register_dispatcher_handlers(self) -> None:
        """Register Step 37 message handlers with engine's FederationMessageDispatcher."""
        dispatcher = getattr(self.engine, "dispatcher", None)
        if not dispatcher or not hasattr(dispatcher, "register_handler"):
            return

        dispatcher.register_handler(
            FederationMessageType.RESOURCE_ADVERTISEMENT,
            self.handle_inbound_advertisement,
        )
        dispatcher.register_handler(
            FederationMessageType.RESOURCE_QUERY,
            self.handle_inbound_resource_query,
        )

    def handle_inbound_advertisement(
        self,
        envelope: FederationMessageEnvelope,
        channel: FederationChannel,
    ) -> Dict[str, Any]:
        """
        Process incoming RESOURCE_ADVERTISEMENT message.
        Validates envelope, checks sender revocation status, verifies signature,
        and registers peer claim in local registry.
        """
        audit = getattr(self.engine, "audit_logger", None)
        epoch = getattr(self.engine, "current_epoch", 1)

        if audit:
            audit.log(
                event_type=AuditEventType.RESOURCE_ADVERTISEMENT_RECEIVED,
                epoch=epoch,
                details={"sender": envelope.sender_peer_id, "message_id": envelope.message_id},
            )

        # Deserialize advertisement from payload
        try:
            adv = ResourceAdvertisement.from_dict(envelope.payload)
        except Exception as e:
            if audit:
                audit.log(
                    event_type=AuditEventType.RESOURCE_ADVERTISEMENT_REJECTED,
                    epoch=epoch,
                    details={"sender": envelope.sender_peer_id, "reason": f"Deserialization error: {e}"},
                )
            raise InvalidAdvertisementError(f"Malformed resource advertisement payload: {e}")

        # Check sender revocation & quarantine status in engine registry
        is_revoked = False
        is_quarantined = False

        peer_reg = getattr(self.engine, "registry", None)
        if peer_reg and hasattr(peer_reg, "get_peer"):
            peer_record = peer_reg.get_peer(adv.node_id)
            if peer_record:
                if peer_record.discovery_status == DiscoveryStatus.REVOKED:
                    is_revoked = True
                elif peer_record.discovery_status == DiscoveryStatus.QUARANTINED:
                    is_quarantined = True

        if channel:
            if channel.is_revoked:
                is_revoked = True
            elif channel.is_quarantined:
                is_quarantined = True

        # Fetch sender's public key
        pub_key: Optional[Ed25519PublicKeyWrapper] = None
        if peer_reg and hasattr(peer_reg, "get_peer"):
            p_rec = peer_reg.get_peer(adv.node_id)
            if p_rec and hasattr(p_rec, "public_key"):
                pub_key = p_rec.public_key

        try:
            self.registry.record_peer_advertisement(
                advertisement=adv,
                public_key=pub_key,
                is_peer_revoked=is_revoked,
                is_peer_quarantined=is_quarantined,
            )
        except Exception as err:
            if audit:
                audit.log(
                    event_type=AuditEventType.RESOURCE_ADVERTISEMENT_REJECTED,
                    epoch=epoch,
                    details={"sender": adv.node_id, "reason": str(err)},
                )
            raise

        if audit:
            audit.log(
                event_type=AuditEventType.RESOURCE_ADVERTISEMENT_ACCEPTED,
                epoch=epoch,
                details={
                    "advertisement_id": adv.advertisement_id,
                    "sender": adv.node_id,
                    "cores": adv.resource_profile.cpu.logical_cores,
                    "version": adv.version,
                },
            )

        return {"status": "ACCEPTED", "advertisement_id": adv.advertisement_id}

    def handle_inbound_resource_query(
        self,
        envelope: FederationMessageEnvelope,
        channel: FederationChannel,
    ) -> Dict[str, Any]:
        """
        Process incoming RESOURCE_QUERY message.
        Evaluates sharing policy and returns local advertisement if authorized.
        """
        audit = getattr(self.engine, "audit_logger", None)
        epoch = getattr(self.engine, "current_epoch", 1)

        if audit:
            audit.log(
                event_type=AuditEventType.RESOURCE_QUERY_RECEIVED,
                epoch=epoch,
                details={"sender": envelope.sender_peer_id},
            )

        sender_tenant = envelope.tenant_id or "default"
        scope = envelope.payload.get("scope", "ALLOW_CAPABILITY_METADATA")

        if not self.sharing_policy.can_share_with(sender_tenant, scope):
            return {"status": "DENIED", "advertisements": []}

        local_adv = self.create_local_advertisement()
        return {"status": "OK", "advertisements": [local_adv.to_dict()]}

    # ========================================================================
    # 4. Peer Invalidation & Liveness Hooks
    # ========================================================================

    def on_peer_disconnected(self, peer_id: str, reason: str = "disconnected") -> None:
        """Hook called when a peer disconnects or heartbeat times out."""
        self.registry.invalidate_peer(peer_id, reason=reason)

    def on_peer_revoked(self, peer_id: str) -> None:
        """Hook called when a peer is revoked."""
        self.registry.bar_peer(peer_id)
