"""
ChakrView Sovereign Capability Subsystem (Step 17).

Exposes strongly typed, hardware-agnostic capability contracts, registries,
policy gates, providers, and environment profiles.
"""

from chakrview.capability.contract import (
    RiskClassification,
    CapabilityStatus,
    CapabilityCategory,
    CapabilityPermission,
    ResourceLimits,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    Capability,
)
from chakrview.capability.registry import (
    CapabilityRegistry,
    CapabilityAlreadyRegisteredError,
    CapabilityNotFoundError,
    CapabilityVersionConflictError,
)
from chakrview.capability.provider import (
    CapabilityProvider,
    CalculatorCapability,
    CalculatorCapabilityProvider,
    TextTransformCapability,
    TextTransformCapabilityProvider,
    ClockCapability,
    ClockCapabilityProvider,
    MockSensorCapability,
    MockSensorCapabilityProvider,
    MockActuatorCapability,
    MockActuatorCapabilityProvider,
)
from chakrview.capability.gate import (
    CapabilityGate,
    CapabilityAuthorizationError,
    CapabilityArgumentValidationError,
    CapabilityDisabledError,
    CapabilityExecutionTimeoutError,
)
from chakrview.capability.environment import (
    EnvironmentProfile,
    get_desktop_environment,
    get_edge_environment,
    get_mock_robot_environment,
    get_mock_vehicle_environment,
)
from chakrview.capability.bridge import (
    ToolCapabilityAdapter,
    capability_result_to_observation,
    get_standard_capability_registry,
)

__all__ = [
    # Contracts
    "RiskClassification",
    "CapabilityStatus",
    "CapabilityCategory",
    "CapabilityPermission",
    "ResourceLimits",
    "CapabilityDescriptor",
    "CapabilityRequest",
    "CapabilityResult",
    "CapabilityContext",
    "Capability",
    # Registry
    "CapabilityRegistry",
    "CapabilityAlreadyRegisteredError",
    "CapabilityNotFoundError",
    "CapabilityVersionConflictError",
    # Providers
    "CapabilityProvider",
    "CalculatorCapability",
    "CalculatorCapabilityProvider",
    "TextTransformCapability",
    "TextTransformCapabilityProvider",
    "ClockCapability",
    "ClockCapabilityProvider",
    "MockSensorCapability",
    "MockSensorCapabilityProvider",
    "MockActuatorCapability",
    "MockActuatorCapabilityProvider",
    # Gate & Security
    "CapabilityGate",
    "CapabilityAuthorizationError",
    "CapabilityArgumentValidationError",
    "CapabilityDisabledError",
    "CapabilityExecutionTimeoutError",
    # Environment Profiles
    "EnvironmentProfile",
    "get_desktop_environment",
    "get_edge_environment",
    "get_mock_robot_environment",
    "get_mock_vehicle_environment",
    # Bridges
    "ToolCapabilityAdapter",
    "capability_result_to_observation",
    "get_standard_capability_registry",
]
