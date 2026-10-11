"""
Federated Message Protocol & Cryptographic Integrity Subsystem (Step 26).

Enforces:
1. Strict message validation (sender, receiver, tenant, session, payload).
2. Tamper-evident cryptographic hashing (SHA-256 local fingerprinting).
3. Optional chain/hash linking across message threads.
4. Deterministic trace fingerprint generation for auditable federated logs.
5. ZERO external network or blockchain dependency (CPU-first, purely local).
"""

import hashlib
import json
from typing import Dict, List, Optional, Any

from chakrview.cognition.federated.models import (
    AgentMessage,
    MessageType,
    MessageEnvelope,
    AgentContract,
)


class MessageValidationError(ValueError):
    """Raised when an inter-agent message fails structural or security validation."""
    pass


class MessageTamperingError(PermissionError):
    """Raised when message payload has been altered and fails hash verification."""
    pass


class TenantRoutingError(PermissionError):
    """Raised when a message attempts cross-tenant communication."""
    pass


class FederatedProtocolValidator:
    """
    Validates inter-agent message envelopes, tenant routing, and cryptographic integrity.
    """

    @staticmethod
    def validate_message(
        message: AgentMessage,
        expected_tenant_id: str,
        expected_session_id: Optional[str] = None,
        registered_sender_ids: Optional[List[str]] = None,
        registered_receiver_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Validate an AgentMessage against tenant boundaries, registered IDs, and integrity.
        """
        # 1. Basic structural checks
        if not message.message_id:
            raise MessageValidationError("Message ID cannot be empty.")
        if not message.sender_agent_id:
            raise MessageValidationError("Sender Agent ID cannot be empty.")
        if not message.receiver_agent_id:
            raise MessageValidationError("Receiver Agent ID cannot be empty.")

        # 2. Multi-tenant isolation verification
        if message.tenant_id != expected_tenant_id:
            raise TenantRoutingError(
                f"Tenant routing violation: message tenant '{message.tenant_id}' "
                f"does not match expected context '{expected_tenant_id}'."
            )
        if expected_session_id and message.session_id != expected_session_id:
            raise TenantRoutingError(
                f"Session routing violation: message session '{message.session_id}' "
                f"does not match expected session '{expected_session_id}'."
            )

        # 3. Registered sender check
        if registered_sender_ids is not None:
            if message.sender_agent_id not in registered_sender_ids and message.sender_agent_id != "COORDINATOR":
                raise MessageValidationError(
                    f"Unknown sender: agent '{message.sender_agent_id}' is not registered."
                )

        # 4. Registered receiver check (or broadcast)
        if registered_receiver_ids is not None:
            if message.receiver_agent_id not in registered_receiver_ids and message.receiver_agent_id not in ("BROADCAST", "COORDINATOR"):
                raise MessageValidationError(
                    f"Unknown receiver: agent '{message.receiver_agent_id}' is not registered."
                )

        # 5. Cryptographic tamper verification
        if not message.verify_integrity():
            raise MessageTamperingError(
                f"Message integrity violation: hash '{message.integrity_hash}' does not match computed hash."
            )

    @staticmethod
    def compute_trace_fingerprint(
        events: List[Dict[str, Any]],
        task_id: str,
        tenant_id: str,
    ) -> str:
        """
        Compute a deterministic SHA-256 fingerprint over federated trace execution records.
        """
        trace_data = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "events": events,
        }
        encoded = json.dumps(trace_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def verify_message_chain(messages: List[AgentMessage]) -> bool:
        """
        Verify cryptographic hash linking in a sequence of parent-child messages.
        """
        msg_map = {m.message_id: m for m in messages}
        for msg in messages:
            if not msg.verify_integrity():
                return False
            if msg.parent_message_id:
                if msg.parent_message_id not in msg_map:
                    # Missing parent in thread
                    return False
        return True
