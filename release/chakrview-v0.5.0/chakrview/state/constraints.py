"""
Constraint & Policy State Representation for ChakrView (Step 18).

Provides inspectable, auditable representations of active execution
boundaries, resource ceilings, safety interlocks, and policy rules.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid


@dataclass
class PolicyRestriction:
    """
    Explicit, auditable restriction rule within the active constraint state.
    """
    restriction_id: str
    category: str
    rule: str
    enforced_by: str = "CapabilityGate"
    is_active: bool = True
    severity: str = "ERROR"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRestriction":
        return cls(**data)


@dataclass
class ConstraintState:
    """
    Active constraint state defining hard execution limits and permitted capabilities.
    """
    max_memory_mb: int = 1024
    max_steps: int = 10
    max_depth: int = 5
    timeout_seconds: float = 60.0
    max_token_budget: int = 512
    allowed_capabilities: Set[str] = field(default_factory=set)
    blocked_capabilities: Set[str] = field(default_factory=set)
    allowed_risk_levels: Set[str] = field(default_factory=lambda: {"READ_ONLY", "COMPUTE"})
    safety_interlocks_active: bool = True
    restrictions: List[PolicyRestriction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (1 <= self.max_token_budget <= 512):
            raise ValueError(f"max_token_budget must be in [1, 512], got {self.max_token_budget}")

    def is_capability_permitted(self, capability_id: str, risk_level: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate whether a capability and risk level are permitted under active constraints.
        """
        if capability_id in self.blocked_capabilities:
            return False, f"Capability '{capability_id}' is explicitly in blocked_capabilities."

        if self.allowed_capabilities and capability_id not in self.allowed_capabilities:
            return False, f"Capability '{capability_id}' is not in allowed_capabilities whitelist."

        if risk_level not in self.allowed_risk_levels:
            return False, f"Risk level '{risk_level}' is not in permitted risk levels: {self.allowed_risk_levels}."

        return True, None

    def add_restriction(self, restriction: PolicyRestriction) -> None:
        """Add an auditable policy restriction."""
        self.restrictions.append(restriction)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["allowed_capabilities"] = list(self.allowed_capabilities)
        d["blocked_capabilities"] = list(self.blocked_capabilities)
        d["allowed_risk_levels"] = list(self.allowed_risk_levels)
        d["restrictions"] = [r.to_dict() for r in self.restrictions]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintState":
        d = dict(data)
        d["allowed_capabilities"] = set(d.get("allowed_capabilities", []))
        d["blocked_capabilities"] = set(d.get("blocked_capabilities", []))
        d["allowed_risk_levels"] = set(d.get("allowed_risk_levels", ["READ_ONLY", "COMPUTE"]))
        d["restrictions"] = [
            PolicyRestriction.from_dict(r) for r in d.get("restrictions", [])
        ]
        return cls(**d)
