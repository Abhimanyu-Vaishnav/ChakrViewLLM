"""
System Identity Representation for ChakrView (Step 18).

Provides a formal, inspectable, machine-readable identity specification
for the running ChakrView system instance.

Architectural Rule:
The identity is deterministic software metadata describing the system,
its frozen neural core, tokenizer specifications, deployment environment,
and active policy profiles. It contains NO anthropomorphic or sentience claims.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class SystemIdentity:
    """
    Formal immutable identity specification of a ChakrView instance.
    
    Attributes:
        system_id: Unique instance or node identifier.
        system_name: Human-readable platform name.
        software_version: Application software version string.
        architecture_version: Architectural milestone version (e.g. 'Step 18').
        model_version: Frozen neural core identifier ('ChakrMicro-v0.1').
        model_parameters: Exact frozen parameter count (3,443,136).
        tokenizer_vocab: Exact tokenizer vocabulary size (4,096).
        context_length: Maximum sequence context horizon (512).
        deployment_environment: Active environment identifier.
        supported_capabilities: Tuple of capability identifiers supported by this instance.
        policy_profile: Active governance policy profile.
        created_at: Epoch timestamp of instance identity instantiation.
        metadata: Diagnostic or tenant-specific metadata.
    """
    system_id: str
    system_name: str = "ChakrView Indigenous AI Framework"
    software_version: str = "0.1.0"
    architecture_version: str = "Step 18"
    model_version: str = "ChakrMicro-v0.1"
    model_parameters: int = 3443136
    tokenizer_vocab: int = 4096
    context_length: int = 512
    deployment_environment: str = "desktop_x86_64"
    supported_capabilities: List[str] = field(default_factory=list)
    policy_profile: str = "governed_default"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_parameters != 3443136:
            raise ValueError(f"Invalid model_parameters: {self.model_parameters}. Frozen invariant is 3,443,136.")
        if self.tokenizer_vocab != 4096:
            raise ValueError(f"Invalid tokenizer_vocab: {self.tokenizer_vocab}. Frozen invariant is 4,096.")
        if self.context_length != 512:
            raise ValueError(f"Invalid context_length: {self.context_length}. Frozen invariant is 512.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemIdentity":
        d = dict(data)
        if "supported_capabilities" in d and isinstance(d["supported_capabilities"], (list, tuple)):
            d["supported_capabilities"] = list(d["supported_capabilities"])
        return cls(**d)


def get_current_system_identity(
    system_id: str = "chakrview-core-node-01",
    deployment_env: str = "desktop_x86_64",
    supported_capabilities: Optional[List[str]] = None,
    policy_profile: str = "governed_default",
    metadata: Optional[Dict[str, Any]] = None,
) -> SystemIdentity:
    """
    Factory constructing the verified standard SystemIdentity for ChakrView.
    """
    default_caps = [
        "calculator",
        "text_transform",
        "system_clock",
        "mock_environmental_sensor",
        "mock_motor_actuator",
    ]
    caps = supported_capabilities if supported_capabilities is not None else default_caps
    return SystemIdentity(
        system_id=system_id,
        deployment_environment=deployment_env,
        supported_capabilities=list(caps),
        policy_profile=policy_profile,
        metadata=metadata or {},
    )
