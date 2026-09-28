"""
Sovereign Capability Contracts and Data Models for ChakrView (Step 17).

Defines strongly typed, hardware-agnostic capability abstractions:
- RiskClassification: metadata risk tiering (READ_ONLY -> HIGH_IMPACT)
- CapabilityStatus: lifecycle & operational health
- CapabilityCategory: domain categories (SOFTWARE, EDGE_DEVICE, SENSOR, etc.)
- CapabilityDescriptor: formal capability manifest and metadata
- CapabilityRequest / CapabilityResult / CapabilityContext: governed execution boundary
- Capability: abstract base class for executable capabilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid


class RiskClassification(str, Enum):
    """
    Risk tiering for capability governance.
    
    Used by policy gates to constrain execution. Risk != Permission.
    """
    READ_ONLY = "READ_ONLY"            # Passive data read, sensor read, telemetry, clock
    COMPUTE = "COMPUTE"                # Mathematical calculation, text transformation, local pure logic
    EXTERNAL_WRITE = "EXTERNAL_WRITE"  # File generation, database update, network mutation
    PHYSICAL_ACTION = "PHYSICAL_ACTION"# Actuator movement, valve open/close, motor speed change
    HIGH_IMPACT = "HIGH_IMPACT"        # Critical system state change, firmware flash, power cycle


class CapabilityStatus(str, Enum):
    """Operational health and availability status of a capability."""
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityCategory(str, Enum):
    """High-level domain category of a capability."""
    SOFTWARE = "SOFTWARE"
    EDGE_DEVICE = "EDGE_DEVICE"
    PHYSICAL_DEVICE = "PHYSICAL_DEVICE"
    SENSOR = "SENSOR"
    ACTUATOR = "ACTUATOR"
    UTILITY = "UTILITY"


@dataclass
class CapabilityPermission:
    """
    Permission definition required to authorize a capability.
    """
    permission_id: str
    name: str
    resource: str = "*"
    action: str = "execute"
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityPermission":
        return cls(**data)


@dataclass
class ResourceLimits:
    """
    Resource execution bounds for a capability.
    """
    max_memory_mb: int = 256
    max_cpu_time_ms: float = 5000.0
    rate_limit_per_min: int = 60
    timeout_seconds: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLimits":
        return cls(**data)


@dataclass
class CapabilityDescriptor:
    """
    Formal metadata specification of a capability.
    
    The core brain only interacts with capabilities through this contract.
    """
    capability_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    category: CapabilityCategory = CapabilityCategory.SOFTWARE
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    risk_level: RiskClassification = RiskClassification.READ_ONLY
    provider_id: str = "default_provider"
    status: CapabilityStatus = CapabilityStatus.AVAILABLE
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["risk_level"] = self.risk_level.value
        d["status"] = self.status.value
        d["resource_limits"] = self.resource_limits.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityDescriptor":
        d = dict(data)
        d["category"] = CapabilityCategory(d["category"])
        d["risk_level"] = RiskClassification(d["risk_level"])
        d["status"] = CapabilityStatus(d["status"])
        if isinstance(d.get("resource_limits"), dict):
            d["resource_limits"] = ResourceLimits.from_dict(d["resource_limits"])
        return cls(**d)


@dataclass
class CapabilityRequest:
    """
    Deterministic request payload for capability execution.
    """
    capability_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parameters: Dict[str, Any] = field(default_factory=dict)
    caller_id: str = "agent"
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    timeout_seconds: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityRequest":
        return cls(**data)


@dataclass
class CapabilityResult:
    """
    Structured outcome of capability execution.
    """
    request_id: str
    capability_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    status: CapabilityStatus = CapabilityStatus.AVAILABLE
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityResult":
        d = dict(data)
        d["status"] = CapabilityStatus(d["status"])
        return cls(**d)


@dataclass
class CapabilityContext:
    """
    Execution context providing caller authorization and environment boundaries.
    """
    user_id: str = "default_user"
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    environment_id: str = "default_env"
    granted_permissions: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None


class Capability(ABC):
    """
    Abstract base class for an executable capability.
    
    Decouples implementation logic from the cognitive core.
    """

    @property
    @abstractmethod
    def descriptor(self) -> CapabilityDescriptor:
        """Formal manifest and metadata of this capability."""
        pass

    @abstractmethod
    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        """
        Execute the capability under governed constraints.
        """
        pass

    def validate_arguments(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate input parameters against descriptor schema.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        schema = self.descriptor.input_schema
        if not schema:
            return True, None

        required = schema.get("required", [])
        for field_name in required:
            if field_name not in parameters:
                return False, f"Missing required parameter '{field_name}'."

        properties = schema.get("properties", {})
        for param_name, param_val in parameters.items():
            if param_name in properties:
                spec = properties[param_name]
                expected_type = spec.get("type")
                if expected_type == "string" and not isinstance(param_val, str):
                    return False, f"Parameter '{param_name}' must be a string."
                elif expected_type == "number" and not isinstance(param_val, (int, float)):
                    return False, f"Parameter '{param_name}' must be a number."
                elif expected_type == "integer" and not isinstance(param_val, int):
                    return False, f"Parameter '{param_name}' must be an integer."
                elif expected_type == "boolean" and not isinstance(param_val, bool):
                    return False, f"Parameter '{param_name}' must be a boolean."
                elif expected_type == "array" and not isinstance(param_val, list):
                    return False, f"Parameter '{param_name}' must be a list."
                elif expected_type == "object" and not isinstance(param_val, dict):
                    return False, f"Parameter '{param_name}' must be a dictionary."

                # Range check
                if "minimum" in spec and isinstance(param_val, (int, float)):
                    if param_val < spec["minimum"]:
                        return False, f"Parameter '{param_name}' < minimum {spec['minimum']}."
                if "maximum" in spec and isinstance(param_val, (int, float)):
                    if param_val > spec["maximum"]:
                        return False, f"Parameter '{param_name}' > maximum {spec['maximum']}."

        return True, None

    def check_health(self) -> CapabilityStatus:
        """Query operational status of the underlying capability implementation."""
        return self.descriptor.status
