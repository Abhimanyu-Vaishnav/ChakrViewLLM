"""
Cross-Zone Federation Policy and Boundary Gate (Step 29).

CRITICAL ARCHITECTURAL AXIOM:
DEFAULT DENY:
Every cross-zone operation, federation scope, and task delegation is denied by default.
Explicit policy grants are required.
Prohibited scopes (weight access, private memory, tenant crossover, authority transfer)
are strictly non-negotiable and unconditionally denied.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from chakrview.cognition.peering.models import (
    FederationScope,
    ProhibitedScope,
    PeerIdentity,
    MAX_PEERS_TOTAL,
    MAX_PEERS_PER_ZONE,
    MAX_ACTIVE_PEERS,
    MAX_DELEGATION_DEPTH,
    DEFAULT_TRUST_TTL_EPOCHS,
)


class PolicyViolationError(PermissionError):
    """Raised when an operation violates federation policy or boundaries."""
    pass


@dataclass
class CrossZoneFederationPolicy:
    """
    Governed policy rules controlling cross-zone peer admission and scope authorization.
    """
    default_deny: bool = True
    allowed_scopes: Set[FederationScope] = field(default_factory=lambda: {
        FederationScope.ALLOW_EVIDENCE_EXCHANGE,
        FederationScope.ALLOW_VERIFICATION,
        FederationScope.ALLOW_CAPABILITY_METADATA,
        FederationScope.ALLOW_MODEL_METADATA,
    })
    prohibited_scopes: Set[ProhibitedScope] = field(default_factory=lambda: {
        ProhibitedScope.DENY_MODEL_WEIGHT_ACCESS,
        ProhibitedScope.DENY_PRIVATE_MEMORY,
        ProhibitedScope.DENY_TENANT_CROSSOVER,
        ProhibitedScope.DENY_AUTHORITY_TRANSFER,
    })
    allowed_zones: Optional[Set[str]] = None  # None means any zone allowed subject to policy
    blocked_zones: Set[str] = field(default_factory=set)
    allowed_organizations: Optional[Set[str]] = None
    max_peers_total: int = MAX_PEERS_TOTAL
    max_peers_per_zone: int = MAX_PEERS_PER_ZONE
    max_active_peers: int = MAX_ACTIVE_PEERS
    max_delegation_depth: int = MAX_DELEGATION_DEPTH
    default_ttl_epochs: int = DEFAULT_TRUST_TTL_EPOCHS

    def __post_init__(self) -> None:
        # Enforce hard ceilings
        if self.max_peers_total > MAX_PEERS_TOTAL:
            self.max_peers_total = MAX_PEERS_TOTAL
        if self.max_peers_per_zone > MAX_PEERS_PER_ZONE:
            self.max_peers_per_zone = MAX_PEERS_PER_ZONE
        if self.max_active_peers > MAX_ACTIVE_PEERS:
            self.max_active_peers = MAX_ACTIVE_PEERS
        if self.max_delegation_depth > MAX_DELEGATION_DEPTH:
            self.max_delegation_depth = MAX_DELEGATION_DEPTH

    def is_peer_admissible(self, peer: PeerIdentity) -> Tuple[bool, str]:
        """Verify whether a peer's identity is eligible for discovery and negotiation."""
        if peer.zone_id in self.blocked_zones:
            return False, f"Zone '{peer.zone_id}' is explicitly blocked by local policy."

        if self.allowed_zones is not None and peer.zone_id not in self.allowed_zones:
            return False, f"Zone '{peer.zone_id}' is not in the allowed zones whitelist."

        if self.allowed_organizations is not None and peer.organization_id not in self.allowed_organizations:
            return False, f"Organization '{peer.organization_id}' is not in the allowed organizations whitelist."

        return True, "Peer is admissible under policy."

    def evaluate_scope_request(self, peer: PeerIdentity, scope: FederationScope) -> Tuple[bool, str]:
        """
        Evaluate whether a specific federation scope may be granted to a peer.
        Enforces default-deny and unconditional prohibitions.
        """
        # 1. Check peer admissibility
        admissible, reason = self.is_peer_admissible(peer)
        if not admissible:
            return False, reason

        # 2. Check unconditional prohibitions
        scope_str = scope.value if isinstance(scope, FederationScope) else str(scope)
        for prohibited in self.prohibited_scopes:
            if prohibited.value in scope_str:
                return False, f"Scope '{scope_str}' violates prohibited boundary {prohibited.value}."

        # 3. Check allowed scopes whitelist (Default Deny)
        if scope not in self.allowed_scopes:
            return False, f"Scope '{scope_str}' is not permitted by local policy whitelist."

        return True, f"Scope '{scope_str}' authorized."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_deny": self.default_deny,
            "allowed_scopes": [s.value for s in self.allowed_scopes],
            "prohibited_scopes": [p.value for p in self.prohibited_scopes],
            "allowed_zones": list(self.allowed_zones) if self.allowed_zones is not None else None,
            "blocked_zones": list(self.blocked_zones),
            "allowed_organizations": list(self.allowed_organizations) if self.allowed_organizations is not None else None,
            "max_peers_total": self.max_peers_total,
            "max_peers_per_zone": self.max_peers_per_zone,
            "max_active_peers": self.max_active_peers,
            "max_delegation_depth": self.max_delegation_depth,
            "default_ttl_epochs": self.default_ttl_epochs,
        }
