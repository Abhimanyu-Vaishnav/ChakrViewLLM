"""
Bounded Federation Heartbeat & Liveness Monitor (Step 35).

Periodically verifies engine identity, protocol version, state digest, and session freshness
across enrolled federation members.

CRITICAL INVARIANTS:
- Heartbeat does NOT grant capabilities
- Heartbeat does NOT refresh trust indefinitely
- Heartbeat does NOT bypass session expiration
- Heartbeat does NOT mutate neural weights (ΔW = 0)
- Heartbeat failure (UNREACHABLE) != REVOCATION
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

from chakrview.cognition.peering.models import AuditEventType
from chakrview.cognition.federation.discovery.models import (
    FederationNodeMembership,
    MembershipState,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_MAX_MISSED_HEARTBEATS,
)
from chakrview.cognition.federation.discovery.errors import (
    HeartbeatError,
    HeartbeatTimeoutError,
    HeartbeatValidationError,
    UnknownPeerError,
    QuarantineError,
)
from chakrview.cognition.federation.discovery.membership import FederationMembershipManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeartbeatPayload:
    """Strongly typed bounded heartbeat payload."""
    sender_engine_id: str
    sender_zone_id: str
    epoch: int
    protocol_version: str
    state_version: int
    state_digest: str
    membership_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_engine_id": self.sender_engine_id,
            "sender_zone_id": self.sender_zone_id,
            "epoch": self.epoch,
            "protocol_version": self.protocol_version,
            "state_version": self.state_version,
            "state_digest": self.state_digest,
            "membership_id": self.membership_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeartbeatPayload":
        return cls(
            sender_engine_id=data["sender_engine_id"],
            sender_zone_id=data["sender_zone_id"],
            epoch=int(data["epoch"]),
            protocol_version=data["protocol_version"],
            state_version=int(data["state_version"]),
            state_digest=data["state_digest"],
            membership_id=data["membership_id"],
            timestamp=float(data.get("timestamp", time.time())),
        )


class FederationHeartbeatMonitor:
    """
    Manages deterministic heartbeat ping/pong validation and missed-deadline tracking.
    """

    def __init__(
        self,
        engine: Any = None,
        membership_manager: Optional[FederationMembershipManager] = None,
        interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
        heartbeat_timeout_seconds: Optional[float] = None,
        max_missed: int = DEFAULT_MAX_MISSED_HEARTBEATS,
    ) -> None:
        self.engine = engine or (getattr(membership_manager, "engine", None) if membership_manager else None)
        self.membership_manager = membership_manager or (getattr(engine, "membership_manager", None) if engine else None)
        self.interval_seconds = interval_seconds
        self.timeout_seconds = heartbeat_timeout_seconds if heartbeat_timeout_seconds is not None else timeout_seconds
        self.max_missed = max_missed

    def check_liveness(self, now: Optional[float] = None) -> List[str]:
        """
        Inspect heartbeat freshness across all member nodes.
        Returns list of node_ids that have exceeded the timeout deadline.
        """
        current_time = now or time.time()
        timed_out: List[str] = []
        if not self.membership_manager:
            return timed_out

        for m in self.membership_manager.list_members():
            if m.state in (MembershipState.MEMBER, MembershipState.AUTHENTICATED):
                if current_time - m.last_heartbeat_timestamp > self.timeout_seconds:
                    node_id = m.node_id or m.membership_id
                    timed_out.append(node_id)
                    self.record_missed_heartbeat(m.membership_id)
        return timed_out

    def create_heartbeat(self, membership_id: str) -> HeartbeatPayload:
        """Construct a verifiable heartbeat payload for an enrolled member."""
        coord = getattr(self.engine, "coordinator", None)
        state_ver = coord.current_version.version if coord else 1
        digest = self.engine.compute_security_state_digest().compute_composite_digest() if hasattr(self.engine, "compute_security_state_digest") else "NO_DIGEST"

        payload = HeartbeatPayload(
            sender_engine_id=self.engine.engine_identity.engine_id,
            sender_zone_id=self.engine.local_zone_id,
            epoch=self.engine.current_epoch,
            protocol_version=self.engine.engine_identity.protocol_version,
            state_version=state_ver,
            state_digest=digest,
            membership_id=membership_id,
            timestamp=time.time(),
        )

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.HEARTBEAT_SENT,
                epoch=self.engine.current_epoch,
                details={"target_membership_id": membership_id, "state_version": state_ver},
            )

        return payload

    def process_incoming_heartbeat(self, payload_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an inbound heartbeat request from a remote member.
        Enforces membership standing, revocation, quarantine, and state digest consistency.
        """
        payload = HeartbeatPayload.from_dict(payload_dict)
        sender_id = payload.sender_engine_id

        # 1. Lookup membership
        membership = self.membership_manager.get_membership_by_node_id(sender_id)
        if not membership:
            audit = getattr(self.engine, "audit_logger", None)
            if audit:
                audit.log(
                    event_type=AuditEventType.HEARTBEAT_FAILED,
                    epoch=self.engine.current_epoch,
                    details={"sender_id": sender_id, "reason": "Unknown peer"},
                )
            raise UnknownPeerError(f"Heartbeat rejected: sender node '{sender_id}' is not an enrolled member.")

        # 2. Check terminal revocation
        if membership.is_revoked():
            return {
                "status": "REVOKED",
                "responder_engine_id": self.engine.engine_identity.engine_id,
                "reason": "Node is permanently revoked in local registry",
            }

        # 3. Check quarantine
        if membership.is_quarantined():
            return {
                "status": "QUARANTINED",
                "responder_engine_id": self.engine.engine_identity.engine_id,
                "reason": f"Node is quarantined: {membership.quarantine_reason}",
            }

        # 4. Check protocol compatibility
        if payload.protocol_version != self.engine.engine_identity.protocol_version:
            self.membership_manager.quarantine_member(
                membership.membership_id,
                reason=f"Protocol mismatch: expected {self.engine.engine_identity.protocol_version}, got {payload.protocol_version}",
            )
            return {
                "status": "QUARANTINED",
                "responder_engine_id": self.engine.engine_identity.engine_id,
                "reason": "Protocol version mismatch",
            }

        # 5. Check state digest conflict
        coord = getattr(self.engine, "coordinator", None)
        local_ver = coord.current_version.version if coord else 1
        local_digest = self.engine.compute_security_state_digest().compute_composite_digest() if hasattr(self.engine, "compute_security_state_digest") else "NO_DIGEST"

        if payload.state_version == local_ver and payload.state_digest != local_digest:
            # Split-brain divergence detected during heartbeat!
            self.membership_manager.quarantine_member(
                membership.membership_id,
                reason=f"Split-brain state digest conflict at version {local_ver}",
            )
            return {
                "status": "DIGEST_CONFLICT",
                "responder_engine_id": self.engine.engine_identity.engine_id,
                "local_version": local_ver,
                "reason": "State digest divergence detected",
            }

        # 6. Update liveness
        membership.record_heartbeat(epoch=self.engine.current_epoch, timestamp=time.time())
        if membership.state == MembershipState.SUSPENDED:
            membership.transition_to(MembershipState.MEMBER, reason="Liveness restored via heartbeat")

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.HEARTBEAT_RECEIVED,
                epoch=self.engine.current_epoch,
                details={"sender_id": sender_id, "epoch": payload.epoch},
            )

        return {
            "status": "OK",
            "responder_engine_id": self.engine.engine_identity.engine_id,
            "epoch": self.engine.current_epoch,
            "state_version": local_ver,
            "state_digest": local_digest,
            "timestamp": time.time(),
        }

    def verify_response(self, membership_id: str, response: Dict[str, Any]) -> bool:
        """
        Validate responder's status in response to local heartbeat ping.
        """
        membership = self.membership_manager.get_membership(membership_id)
        if not membership:
            raise UnknownPeerError(f"Membership '{membership_id}' does not exist.")

        status = response.get("status")

        if status == "OK":
            membership.record_heartbeat(epoch=self.engine.current_epoch, timestamp=time.time())
            if membership.state == MembershipState.SUSPENDED:
                membership.transition_to(MembershipState.MEMBER, reason="Heartbeat responded successfully")
            return True

        if status == "DIGEST_CONFLICT":
            self.membership_manager.quarantine_member(
                membership_id,
                reason=f"Remote reported digest conflict: {response.get('reason')}",
            )
            return False

        if status == "QUARANTINED":
            self.membership_manager.quarantine_member(
                membership_id,
                reason=f"Remote indicates quarantine: {response.get('reason')}",
            )
            return False

        if status == "REVOKED":
            self.membership_manager.revoke_member(
                membership_id,
                reason=f"Remote reports revocation: {response.get('reason')}",
            )
            return False

        # Any unexpected status
        self.record_missed_heartbeat(membership_id)
        return False

    def record_missed_heartbeat(self, membership_id: str) -> bool:
        """
        Record a missed heartbeat deadline for a member.
        If missed count >= max_missed, transitions member to SUSPENDED.
        CRITICAL: SUSPENDED != REVOKED.
        """
        membership = self.membership_manager.get_membership(membership_id)
        if not membership:
            return False

        membership.missed_heartbeats += 1

        audit = getattr(self.engine, "audit_logger", None)
        if audit:
            audit.log(
                event_type=AuditEventType.HEARTBEAT_TIMEOUT,
                epoch=self.engine.current_epoch,
                details={
                    "membership_id": membership_id,
                    "missed_count": membership.missed_heartbeats,
                    "max_allowed": self.max_missed,
                },
            )

        if membership.missed_heartbeats >= self.max_missed and membership.is_active_member():
            self.membership_manager.suspend_member(
                membership_id,
                reason=f"Missed {membership.missed_heartbeats} consecutive heartbeats (timeout ceiling reached)",
            )
            return False

        return True
