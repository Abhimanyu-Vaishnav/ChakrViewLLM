"""
Capability State Representation for ChakrView (Step 18).

Provides observable status tracking for registered capabilities:
AVAILABLE, UNAVAILABLE, DISABLED, UNKNOWN, DEGRADED, BUSY, ERROR.

CRITICAL ARCHITECTURAL DISTINCTIONS:
- UNKNOWN != FALSE
  (e.g., capability status is UNKNOWN when it has not yet been probed).
- UNAVAILABLE != UNKNOWN
  (e.g., capability is UNAVAILABLE when its driver or hardware endpoint is missing).
- DISABLED != UNAVAILABLE
  (e.g., capability is DISABLED when shut down administratively or by policy).

Architectural Rule:
CapabilityState is strictly an OBSERVATION of registry and health states.
It does NOT grant authority and cannot bypass CapabilityGate.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any


class ObservedCapabilityStatus(str, Enum):
    """Observable operational status of a capability."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"
    DEGRADED = "DEGRADED"
    BUSY = "BUSY"
    ERROR = "ERROR"


@dataclass
class CapabilityObservation:
    """
    Observable diagnostic report for a specific capability.
    """
    capability_id: str
    observed_status: ObservedCapabilityStatus = ObservedCapabilityStatus.UNKNOWN
    risk_level: str = "READ_ONLY"
    last_checked_at: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    health_notes: str = ""
    is_authorized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["observed_status"] = self.observed_status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityObservation":
        d = dict(data)
        d["observed_status"] = ObservedCapabilityStatus(d["observed_status"])
        return cls(**d)


@dataclass
class CapabilityState:
    """
    Observable state tracking all known capabilities and their current statuses.
    """
    capabilities: Dict[str, CapabilityObservation] = field(default_factory=dict)

    def record_status(
        self,
        capability_id: str,
        status: ObservedCapabilityStatus,
        risk_level: str = "READ_ONLY",
        latency_ms: float = 0.0,
        health_notes: str = "",
        is_authorized: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CapabilityObservation:
        """Record or update capability status observation."""
        obs = CapabilityObservation(
            capability_id=capability_id,
            observed_status=status,
            risk_level=risk_level,
            last_checked_at=time.time(),
            latency_ms=latency_ms,
            health_notes=health_notes,
            is_authorized=is_authorized,
            metadata=metadata or {},
        )
        self.capabilities[capability_id] = obs
        return obs

    def get_status(self, capability_id: str) -> ObservedCapabilityStatus:
        """Retrieve observed status; returns UNKNOWN if never probed."""
        if capability_id not in self.capabilities:
            return ObservedCapabilityStatus.UNKNOWN
        return self.capabilities[capability_id].observed_status

    def is_available(self, capability_id: str) -> bool:
        """True only if observed status is AVAILABLE or DEGRADED."""
        st = self.get_status(capability_id)
        return st in (ObservedCapabilityStatus.AVAILABLE, ObservedCapabilityStatus.DEGRADED)

    def list_by_status(self, status: ObservedCapabilityStatus) -> List[CapabilityObservation]:
        """List all capabilities matching observed status."""
        return [c for c in self.capabilities.values() if c.observed_status == status]

    def sync_with_registry(self, registry: Any, policy: Optional[Any] = None) -> None:
        """
        Synchronize state observations with a Step 17 CapabilityRegistry.
        """
        if hasattr(registry, "list_capabilities"):
            descriptors = registry.list_capabilities()
            for desc in descriptors:
                cap_id = desc.capability_id
                # Map contract CapabilityStatus to ObservedCapabilityStatus
                raw_st = desc.status.value if hasattr(desc.status, "value") else str(desc.status)
                try:
                    obs_status = ObservedCapabilityStatus(raw_st)
                except ValueError:
                    obs_status = ObservedCapabilityStatus.UNKNOWN

                # Check policy authorization observation
                is_auth = False
                if policy is not None:
                    allowed_caps = getattr(policy, "allowed_capabilities", None)
                    if allowed_caps is None:
                        allowed_caps = getattr(policy, "supported_capability_ids", None)
                    if allowed_caps and cap_id in allowed_caps:
                        is_auth = True

                self.record_status(
                    capability_id=cap_id,
                    status=obs_status,
                    risk_level=desc.risk_level.value if hasattr(desc.risk_level, "value") else str(desc.risk_level),
                    is_authorized=is_auth,
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityState":
        caps = {
            k: CapabilityObservation.from_dict(v)
            for k, v in data.get("capabilities", {}).items()
        }
        return cls(capabilities=caps)
