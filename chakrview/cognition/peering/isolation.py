"""
Cross-Zone and Tenant Isolation Guard for Peering Operations (Step 29).

CRITICAL ARCHITECTURAL AXIOMS:
1. STRICT TENANT ISOLATION:
   A peer from Zone A must NEVER access or operate on a different tenant in Zone B.
2. NO SENSITIVE LEAKAGE:
   Model weights, raw activations, logits, private scratchpads, internal chain-of-thought,
   and cryptographic secrets are strictly barred from cross-zone transmission.
"""

import re
from typing import Dict, Any, Optional, Set, Tuple

from chakrview.cognition.peering.models import PeerIdentity


class IsolationViolationError(PermissionError):
    """Raised when cross-tenant or cross-zone isolation boundary is breached."""
    pass


class CrossZoneIsolationGuard:
    """
    Enforces boundary isolation across zones, tenants, and sessions.
    Sanitizes inbound and outbound federation payloads.
    """

    # Prohibited payload keys that must never cross zone boundaries
    FORBIDDEN_KEYS: Set[str] = {
        "model_weights",
        "weights",
        "raw_logits",
        "logits",
        "raw_activations",
        "activations",
        "scratchpad",
        "private_thought",
        "chain_of_thought",
        "hidden_states",
        "private_key",
        "secret_key",
        "hmac_key",
        "auth_token",
        "token",
    }

    SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?")

    @classmethod
    def validate_tenant_boundary(
        cls,
        peer_zone_id: str,
        target_zone_id: str,
        peer_tenant_id: str,
        target_tenant_id: str,
    ) -> None:
        """
        Validate that a cross-zone request does not attempt cross-tenant crossover.
        """
        if peer_tenant_id != target_tenant_id:
            raise IsolationViolationError(
                f"Cross-tenant crossover denied: peer tenant '{peer_tenant_id}' cannot "
                f"access target tenant '{target_tenant_id}'."
            )

    @classmethod
    def sanitize_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize outbound or inbound payload.
        Fails closed with IsolationViolationError if prohibited artifacts are present.
        """
        sanitized: Dict[str, Any] = {}
        for key, value in payload.items():
            key_lower = key.lower()
            if key_lower in cls.FORBIDDEN_KEYS:
                raise IsolationViolationError(
                    f"Prohibited artifact key '{key}' detected in cross-zone payload. Transfer blocked."
                )

            # Check string values for sensitive credential patterns
            if isinstance(value, str):
                if cls.SECRET_PATTERN.search(value):
                    raise IsolationViolationError(
                        f"Potential credential pattern detected in payload key '{key}'. Transfer blocked."
                    )
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_payload(value)
            elif isinstance(value, list):
                sanitized_list = []
                for item in value:
                    if isinstance(item, dict):
                        sanitized_list.append(cls.sanitize_payload(item))
                    elif isinstance(item, str) and cls.SECRET_PATTERN.search(item):
                        raise IsolationViolationError("Credential pattern detected in list item.")
                    else:
                        sanitized_list.append(item)
                sanitized[key] = sanitized_list
            else:
                sanitized[key] = value

        return sanitized
