"""
Sovereign Environment and Device Profiles for ChakrView (Step 17).

Provides environment descriptors decoupling the core cognitive brain from
specific execution environments (Desktop, Edge/ARM64, Robotics, Automotive).

Architectural Rule:
The brain reasons about capability contracts; the EnvironmentProfile dictates
what capabilities, risk levels, and resource budgets are available.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set

from chakrview.capability.contract import RiskClassification


@dataclass
class EnvironmentProfile:
    """
    Formal specification of an execution environment and its hardware constraints.
    """
    environment_id: str
    name: str
    platform: str = "generic"
    architecture: str = "x86_64"
    memory_limit_mb: int = 1024
    compute_limit_ms: float = 5000.0
    network_available: bool = False
    supported_capability_ids: Set[str] = field(default_factory=set)
    allowed_risk_levels: Set[RiskClassification] = field(
        default_factory=lambda: {RiskClassification.READ_ONLY, RiskClassification.COMPUTE}
    )
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def is_capability_supported(self, capability_id: str) -> bool:
        """Check whether capability_id is in the environment's supported set."""
        if not self.supported_capability_ids:
            # If empty set, by convention all registered capabilities are permitted subject to risk level
            return True
        return capability_id in self.supported_capability_ids

    def is_risk_allowed(self, risk_level: RiskClassification) -> bool:
        """Check whether risk classification is permitted in this environment."""
        return risk_level in self.allowed_risk_levels

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supported_capability_ids"] = list(self.supported_capability_ids)
        d["allowed_risk_levels"] = [r.value for r in self.allowed_risk_levels]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentProfile":
        d = dict(data)
        d["supported_capability_ids"] = set(d.get("supported_capability_ids", []))
        d["allowed_risk_levels"] = {
            RiskClassification(r) for r in d.get("allowed_risk_levels", [])
        }
        return cls(**d)


# =====================================================================
# Predefined Mock Environments for Testing & Deployment Verification
# =====================================================================

def get_desktop_environment() -> EnvironmentProfile:
    """Standard workstation environment with software tools and text processing."""
    return EnvironmentProfile(
        environment_id="env_desktop_x86_64",
        name="Desktop Workstation Environment",
        platform="Windows/Linux/macOS",
        architecture="x86_64",
        memory_limit_mb=4096,
        compute_limit_ms=10000.0,
        network_available=True,
        supported_capability_ids={
            "calculator",
            "text_transform",
            "system_clock",
        },
        allowed_risk_levels={
            RiskClassification.READ_ONLY,
            RiskClassification.COMPUTE,
            RiskClassification.EXTERNAL_WRITE,
        },
        description="High-memory workstation for analysis, text generation, and local compute.",
    )


def get_edge_environment() -> EnvironmentProfile:
    """Constrained ARM64 environment for Raspberry Pi / embedded single-board computers."""
    return EnvironmentProfile(
        environment_id="env_edge_arm64",
        name="Edge SBC Environment",
        platform="Linux/Embedded",
        architecture="ARM64",
        memory_limit_mb=256,
        compute_limit_ms=2000.0,
        network_available=False,
        supported_capability_ids={
            "calculator",
            "system_clock",
            "mock_environmental_sensor",
        },
        allowed_risk_levels={
            RiskClassification.READ_ONLY,
            RiskClassification.COMPUTE,
        },
        description="Low-memory embedded ARM64 environment with sensor monitoring.",
    )


def get_mock_robot_environment() -> EnvironmentProfile:
    """Simulated robotics platform supporting sensors and bounded actuation."""
    return EnvironmentProfile(
        environment_id="env_mock_robot",
        name="Mock Robotics Environment",
        platform="Robotics-RTOS-Mock",
        architecture="ARM64",
        memory_limit_mb=512,
        compute_limit_ms=3000.0,
        network_available=False,
        supported_capability_ids={
            "system_clock",
            "mock_environmental_sensor",
            "mock_motor_actuator",
        },
        allowed_risk_levels={
            RiskClassification.READ_ONLY,
            RiskClassification.COMPUTE,
            RiskClassification.PHYSICAL_ACTION,
        },
        constraints={"max_motor_speed_rpm": 100.0, "safety_interlock_required": True},
        description="Simulated robotic platform for autonomous sensing and motor control.",
    )


def get_mock_vehicle_environment() -> EnvironmentProfile:
    """Simulated automotive vehicle subsystem with telemetry and constrained actuation."""
    return EnvironmentProfile(
        environment_id="env_mock_vehicle",
        name="Mock Vehicle Subsystem Environment",
        platform="Automotive-ECU-Mock",
        architecture="ARM-Cortex-R",
        memory_limit_mb=128,
        compute_limit_ms=1000.0,
        network_available=False,
        supported_capability_ids={
            "system_clock",
            "mock_environmental_sensor",
        },
        allowed_risk_levels={
            RiskClassification.READ_ONLY,
            RiskClassification.COMPUTE,
        },
        constraints={"bus_type": "CAN-FD-Mock", "iso26262_asil_level": "ASIL-B-Mock"},
        description="Simulated vehicle telemetry controller with safety interlocks.",
    )
