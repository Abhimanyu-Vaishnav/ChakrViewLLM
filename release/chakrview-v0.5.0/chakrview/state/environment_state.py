"""
Environment State Representation for ChakrView (Step 18).

Provides inspectable observation of the execution environment:
platform, architecture, available hardware resources, network status,
safety interlocks, and operational modes.

Architectural Rule:
Environment state is an OBSERVATION of physical/digital reality.
It does NOT possess authority to grant permissions or bypass security gates.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any


class OperationalMode(str, Enum):
    """System-level operational posture."""
    NORMAL = "NORMAL"
    MAINTENANCE = "MAINTENANCE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    DEGRADED = "DEGRADED"
    DIAGNOSTIC = "DIAGNOSTIC"


class DeviceConnectionStatus(str, Enum):
    """Connectivity status of external hardware or network bus."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RESTRICTED = "RESTRICTED"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


@dataclass
class EnvironmentState:
    """
    Observable snapshot of host platform and external hardware environment.
    """
    environment_id: str = "env_desktop_x86_64"
    platform: str = "Windows/Linux/macOS"
    architecture: str = "x86_64"
    available_resources: Dict[str, Any] = field(default_factory=lambda: {
        "memory_limit_mb": 4096,
        "compute_limit_ms": 10000.0,
        "cpu_count": 8,
    })
    network_status: DeviceConnectionStatus = DeviceConnectionStatus.CONNECTED
    registered_capabilities: List[str] = field(default_factory=list)
    safety_interlock_status: Dict[str, bool] = field(default_factory=dict)
    operational_mode: OperationalMode = OperationalMode.NORMAL
    observed_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_operational_mode(self, mode: OperationalMode, reason: str = "") -> None:
        """Update active operational mode with audit record."""
        self.operational_mode = mode
        self.observed_at = time.time()
        if reason:
            self.metadata[f"mode_transition_{mode.value}_reason"] = reason

    def set_safety_interlock(self, interlock_name: str, is_active: bool) -> None:
        """Update hardware safety interlock state."""
        self.safety_interlock_status[interlock_name] = is_active
        self.observed_at = time.time()

    def update_resource_metric(self, key: str, value: Any) -> None:
        """Update dynamic resource observation."""
        self.available_resources[key] = value
        self.observed_at = time.time()

    def sync_from_profile(self, profile: Any) -> None:
        """
        Synchronize observable parameters from a Step 17 EnvironmentProfile.
        """
        if hasattr(profile, "environment_id"):
            self.environment_id = profile.environment_id
        if hasattr(profile, "platform"):
            self.platform = profile.platform
        if hasattr(profile, "architecture"):
            self.architecture = profile.architecture
        if hasattr(profile, "memory_limit_mb"):
            self.available_resources["memory_limit_mb"] = profile.memory_limit_mb
        if hasattr(profile, "compute_limit_ms"):
            self.available_resources["compute_limit_ms"] = profile.compute_limit_ms
        if hasattr(profile, "network_available"):
            self.network_status = (
                DeviceConnectionStatus.CONNECTED
                if profile.network_available
                else DeviceConnectionStatus.DISCONNECTED
            )
        if hasattr(profile, "supported_capability_ids"):
            self.registered_capabilities = list(profile.supported_capability_ids)
        self.observed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["network_status"] = self.network_status.value
        d["operational_mode"] = self.operational_mode.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentState":
        d = dict(data)
        d["network_status"] = DeviceConnectionStatus(d["network_status"])
        d["operational_mode"] = OperationalMode(d["operational_mode"])
        return cls(**d)
