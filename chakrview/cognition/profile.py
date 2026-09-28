"""
Hardware-Aware Deployment Profiles for ChakrView Cognitive Subsystem (Step 15).

Provides resource-bounded execution profiles suitable for CPU, Edge, Desktop,
and Server environments without coupling to specific hardware architectures.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

from chakrview.cognition.task import TaskConstraints


@dataclass
class DeploymentProfile:
    """
    Resource profile tailoring cognitive execution limits to target deployment environments.

    Attributes:
        profile_name: Identifier for profile (e.g. 'edge_arm64', 'desktop_x64', 'server').
        max_steps: Maximum plan steps permitted.
        max_depth: Maximum dependency depth permitted.
        max_retries: Maximum retries per failed step.
        timeout_seconds: Hard execution timeout in seconds.
        max_token_budget: Upper bound on prompt context (strictly <= 512).
        max_concurrent_operations: Concurrency throttle.
        max_memory_mb: Approximate memory budget guideline in megabytes.
        enable_detailed_traces: Whether full audit logs are persisted.
        description: Functional description of target environment.
    """
    profile_name: str
    max_steps: int = 10
    max_depth: int = 5
    max_retries: int = 2
    timeout_seconds: float = 60.0
    max_token_budget: int = 512
    max_concurrent_operations: int = 1
    max_memory_mb: int = 1024
    enable_detailed_traces: bool = True
    description: str = "Standard deployment profile"

    def __post_init__(self) -> None:
        if not (1 <= self.max_token_budget <= 512):
            raise ValueError(f"max_token_budget must be in [1, 512], got {self.max_token_budget}")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive.")

    def to_constraints(self) -> TaskConstraints:
        """Derive TaskConstraints matching this profile."""
        return TaskConstraints(
            max_depth=self.max_depth,
            max_steps=self.max_steps,
            max_retries=self.max_retries,
            timeout_seconds=self.timeout_seconds,
            max_token_budget=self.max_token_budget,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentProfile":
        return cls(**data)


# Predefined Standard Profiles

def get_edge_profile() -> DeploymentProfile:
    """Resource-constrained profile for Raspberry Pi, mobile, or low-memory edge devices."""
    return DeploymentProfile(
        profile_name="edge_arm64",
        max_steps=4,
        max_depth=3,
        max_retries=1,
        timeout_seconds=30.0,
        max_token_budget=256,
        max_concurrent_operations=1,
        max_memory_mb=256,
        enable_detailed_traces=False,
        description="Edge/Mobile profile for ARM64 and low-memory environments.",
    )


def get_desktop_profile() -> DeploymentProfile:
    """Standard profile for desktop workstations and laptops."""
    return DeploymentProfile(
        profile_name="desktop_standard",
        max_steps=10,
        max_depth=5,
        max_retries=2,
        timeout_seconds=60.0,
        max_token_budget=512,
        max_concurrent_operations=2,
        max_memory_mb=1024,
        enable_detailed_traces=True,
        description="Standard desktop profile for x86_64 / Apple Silicon workstations.",
    )


def get_server_profile() -> DeploymentProfile:
    """High-capacity profile for private servers and enterprise clusters."""
    return DeploymentProfile(
        profile_name="server_enterprise",
        max_steps=20,
        max_depth=8,
        max_retries=3,
        timeout_seconds=120.0,
        max_token_budget=512,
        max_concurrent_operations=4,
        max_memory_mb=4096,
        enable_detailed_traces=True,
        description="Enterprise server profile for multi-agent workflows.",
    )
