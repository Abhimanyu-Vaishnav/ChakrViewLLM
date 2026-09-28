"""
Sovereign Capability Provider Abstraction and Reference Providers (Step 17).

Provides the provider abstraction separating capability implementations from the core brain:
- CapabilityProvider: abstract provider interface
- Reference/Mock Providers for testing and governance validation:
  1. CalculatorCapabilityProvider (COMPUTE)
  2. TextTransformCapabilityProvider (COMPUTE)
  3. ClockCapabilityProvider (READ_ONLY)
  4. MockSensorCapabilityProvider (SENSOR / READ_ONLY)
  5. MockActuatorCapabilityProvider (ACTUATOR / PHYSICAL_ACTION)

Crucial Principle:
These reference implementations demonstrate contract adherence and governance boundaries.
NO real hardware/robot/vehicle drivers are implemented here.
"""

from abc import ABC, abstractmethod
import ast
from datetime import datetime, timezone
import math
import operator
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityCategory,
    RiskClassification,
    CapabilityStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
    ResourceLimits,
)


class CapabilityProvider(ABC):
    """
    Abstract provider interface managing a collection of capabilities.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version of provider."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize provider resources."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release provider resources."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[Capability]:
        """Return list of capabilities exposed by this provider."""
        pass


# =====================================================================
# 1. Calculator Capability Provider (COMPUTE)
# =====================================================================

class CalculatorCapability(Capability):
    """Safe mathematical computation capability using pure AST parsing."""

    def __init__(self, provider_id: str = "software_math_provider") -> None:
        self._provider_id = provider_id
        self._descriptor = CapabilityDescriptor(
            capability_id="calculator",
            name="Deterministic Calculator",
            version="1.0.0",
            description="Performs safe mathematical arithmetic using AST evaluation.",
            category=CapabilityCategory.SOFTWARE,
            risk_level=RiskClassification.COMPUTE,
            provider_id=self._provider_id,
            input_schema={
                "required": ["expression"],
                "properties": {
                    "expression": {"type": "string"},
                },
            },
            output_schema={
                "properties": {
                    "result": {"type": "number"},
                },
            },
            required_permissions=["capability.compute.math"],
            resource_limits=ResourceLimits(max_cpu_time_ms=500.0, timeout_seconds=2.0),
            tags=["math", "arithmetic", "compute"],
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        t0 = time.perf_counter()
        valid, err = self.validate_arguments(request.parameters)
        if not valid:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=err,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        expr_str = request.parameters["expression"]
        try:
            val = self._eval_expr(expr_str)
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=True,
                output={"result": val},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.AVAILABLE,
            )
        except Exception as e:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=f"Calculation error: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

    def _eval_expr(self, expr_str: str) -> float:
        parsed = ast.parse(expr_str, mode="eval")
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval_node(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                return float(allowed_operators[op_type](left, right))
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                operand = _eval_node(node.operand)
                return float(allowed_operators[op_type](operand))
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        return _eval_node(parsed)


class CalculatorCapabilityProvider(CapabilityProvider):
    """Provider exposing mathematical compute capability."""

    def __init__(self) -> None:
        self._initialized = False
        self._calc = CalculatorCapability(provider_id=self.provider_id)

    @property
    def provider_id(self) -> str:
        return "provider.software.math"

    @property
    def name(self) -> str:
        return "Software Math Provider"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def shutdown(self) -> None:
        self._initialized = False

    def get_capabilities(self) -> List[Capability]:
        return [self._calc]


# =====================================================================
# 2. Text Transform Capability Provider (COMPUTE)
# =====================================================================

class TextTransformCapability(Capability):
    """Text transformation and metric analysis capability."""

    def __init__(self, provider_id: str = "software_text_provider") -> None:
        self._provider_id = provider_id
        self._descriptor = CapabilityDescriptor(
            capability_id="text_transform",
            name="Deterministic Text Transformer",
            version="1.0.0",
            description="Applies deterministic formatting and analysis to text.",
            category=CapabilityCategory.SOFTWARE,
            risk_level=RiskClassification.COMPUTE,
            provider_id=self._provider_id,
            input_schema={
                "required": ["text", "operation"],
                "properties": {
                    "text": {"type": "string"},
                    "operation": {"type": "string"},
                },
            },
            output_schema={
                "properties": {
                    "transformed": {"type": "string"},
                    "count": {"type": "integer"},
                },
            },
            required_permissions=["capability.compute.text"],
            tags=["text", "utility"],
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        t0 = time.perf_counter()
        valid, err = self.validate_arguments(request.parameters)
        if not valid:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=err,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        text = request.parameters["text"]
        op = request.parameters["operation"].lower()

        if op == "uppercase":
            res = {"transformed": text.upper(), "count": len(text)}
        elif op == "lowercase":
            res = {"transformed": text.lower(), "count": len(text)}
        elif op == "word_count":
            words = text.split()
            res = {"transformed": text, "count": len(words)}
        elif op == "title_case":
            res = {"transformed": text.title(), "count": len(text)}
        else:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=f"Unsupported operation '{op}'. Supported: uppercase, lowercase, word_count, title_case",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        return CapabilityResult(
            request_id=request.request_id,
            capability_id=self.descriptor.capability_id,
            success=True,
            output=res,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            status=CapabilityStatus.AVAILABLE,
        )


class TextTransformCapabilityProvider(CapabilityProvider):
    """Provider exposing text processing capabilities."""

    def __init__(self) -> None:
        self._text_cap = TextTransformCapability(provider_id=self.provider_id)

    @property
    def provider_id(self) -> str:
        return "provider.software.text"

    @property
    def name(self) -> str:
        return "Software Text Provider"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def get_capabilities(self) -> List[Capability]:
        return [self._text_cap]


# =====================================================================
# 3. Clock Capability Provider (READ_ONLY / UTILITY)
# =====================================================================

class ClockCapability(Capability):
    """Deterministic system clock and timestamp capability."""

    def __init__(self, provider_id: str = "utility_clock_provider") -> None:
        self._provider_id = provider_id
        self._descriptor = CapabilityDescriptor(
            capability_id="system_clock",
            name="System Clock Capability",
            version="1.0.0",
            description="Provides UTC and timezone-aware timestamps.",
            category=CapabilityCategory.UTILITY,
            risk_level=RiskClassification.READ_ONLY,
            provider_id=self._provider_id,
            input_schema={
                "properties": {
                    "timezone": {"type": "string"},
                },
            },
            output_schema={
                "properties": {
                    "iso_timestamp": {"type": "string"},
                    "unix_epoch": {"type": "number"},
                },
            },
            required_permissions=["capability.read.clock"],
            tags=["time", "clock", "utility"],
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        t0 = time.perf_counter()
        now = datetime.now(timezone.utc)
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=self.descriptor.capability_id,
            success=True,
            output={
                "iso_timestamp": now.isoformat(),
                "unix_epoch": now.timestamp(),
                "timezone": "UTC",
            },
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            status=CapabilityStatus.AVAILABLE,
        )


class ClockCapabilityProvider(CapabilityProvider):
    """Provider exposing clock and time utility capabilities."""

    def __init__(self) -> None:
        self._clock_cap = ClockCapability(provider_id=self.provider_id)

    @property
    def provider_id(self) -> str:
        return "provider.utility.clock"

    @property
    def name(self) -> str:
        return "Utility Clock Provider"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def get_capabilities(self) -> List[Capability]:
        return [self._clock_cap]


# =====================================================================
# 4. Mock Sensor Capability Provider (SENSOR / READ_ONLY)
# =====================================================================

class MockSensorCapability(Capability):
    """Simulated environmental sensor reading capability."""

    def __init__(
        self,
        provider_id: str = "mock_sensor_provider",
        temperature_c: float = 24.5,
        humidity_pct: float = 48.0,
        pressure_hpa: float = 1013.25,
    ) -> None:
        self._provider_id = provider_id
        self.temperature_c = temperature_c
        self.humidity_pct = humidity_pct
        self.pressure_hpa = pressure_hpa
        self._descriptor = CapabilityDescriptor(
            capability_id="mock_environmental_sensor",
            name="Mock Environmental Sensor",
            version="1.0.0",
            description="Simulates telemetry readings from a hardware environmental sensor.",
            category=CapabilityCategory.SENSOR,
            risk_level=RiskClassification.READ_ONLY,
            provider_id=self._provider_id,
            input_schema={
                "properties": {
                    "metric": {"type": "string"},
                },
            },
            output_schema={
                "properties": {
                    "metric": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                },
            },
            required_permissions=["capability.sensor.read"],
            tags=["sensor", "iot", "edge", "telemetry"],
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        t0 = time.perf_counter()
        metric = request.parameters.get("metric", "all").lower()

        if metric == "temperature":
            out = {"metric": "temperature", "value": self.temperature_c, "unit": "Celsius"}
        elif metric == "humidity":
            out = {"metric": "humidity", "value": self.humidity_pct, "unit": "%"}
        elif metric == "pressure":
            out = {"metric": "pressure", "value": self.pressure_hpa, "unit": "hPa"}
        elif metric == "all":
            out = {
                "temperature_c": self.temperature_c,
                "humidity_pct": self.humidity_pct,
                "pressure_hpa": self.pressure_hpa,
                "status": "nominal",
            }
        else:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=f"Unknown sensor metric '{metric}'. Expected: temperature, humidity, pressure, all",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        return CapabilityResult(
            request_id=request.request_id,
            capability_id=self.descriptor.capability_id,
            success=True,
            output=out,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            status=CapabilityStatus.AVAILABLE,
        )


class MockSensorCapabilityProvider(CapabilityProvider):
    """Provider exposing simulated environmental sensor capability."""

    def __init__(self) -> None:
        self._sensor_cap = MockSensorCapability(provider_id=self.provider_id)

    @property
    def provider_id(self) -> str:
        return "provider.device.mock_sensor"

    @property
    def name(self) -> str:
        return "Mock Device Sensor Provider"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def get_capabilities(self) -> List[Capability]:
        return [self._sensor_cap]


# =====================================================================
# 5. Mock Actuator Capability Provider (ACTUATOR / PHYSICAL_ACTION)
# =====================================================================

class MockActuatorCapability(Capability):
    """Simulated motor actuator capability with safety bounds checking."""

    def __init__(self, provider_id: str = "mock_actuator_provider") -> None:
        self._provider_id = provider_id
        self.current_speed_rpm: float = 0.0
        self.current_angle_deg: float = 0.0
        self.is_emergency_halted: bool = False
        self._descriptor = CapabilityDescriptor(
            capability_id="mock_motor_actuator",
            name="Mock Motor Actuator",
            version="1.0.0",
            description="Simulates bounded physical actuation (speed and angle control).",
            category=CapabilityCategory.ACTUATOR,
            risk_level=RiskClassification.PHYSICAL_ACTION,
            provider_id=self._provider_id,
            input_schema={
                "required": ["action"],
                "properties": {
                    "action": {"type": "string"},
                    "target_value": {"type": "number"},
                },
            },
            output_schema={
                "properties": {
                    "action": {"type": "string"},
                    "speed_rpm": {"type": "number"},
                    "angle_deg": {"type": "number"},
                    "is_halted": {"type": "boolean"},
                },
            },
            required_permissions=["capability.actuator.control"],
            tags=["actuator", "motor", "robotics", "physical"],
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def execute(
        self,
        request: CapabilityRequest,
        context: Optional[CapabilityContext] = None,
    ) -> CapabilityResult:
        t0 = time.perf_counter()
        valid, err = self.validate_arguments(request.parameters)
        if not valid:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=err,
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        action = request.parameters["action"].lower()
        target = request.parameters.get("target_value", 0.0)

        # Safety Check: If emergency halted, reject all commands except reset
        if self.is_emergency_halted and action != "reset":
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error="Actuator is in EMERGENCY_HALT state. Send 'reset' action to clear.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.DEGRADED,
            )

        if action == "set_speed":
            if not (-100.0 <= target <= 100.0):
                return CapabilityResult(
                    request_id=request.request_id,
                    capability_id=self.descriptor.capability_id,
                    success=False,
                    error=f"Target speed {target} exceeds safety boundary [-100.0, 100.0] RPM.",
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                    status=CapabilityStatus.ERROR,
                )
            self.current_speed_rpm = float(target)

        elif action == "set_angle":
            if not (0.0 <= target <= 360.0):
                return CapabilityResult(
                    request_id=request.request_id,
                    capability_id=self.descriptor.capability_id,
                    success=False,
                    error=f"Target angle {target} exceeds safety boundary [0.0, 360.0] degrees.",
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                    status=CapabilityStatus.ERROR,
                )
            self.current_angle_deg = float(target)

        elif action in ("stop", "halt"):
            self.current_speed_rpm = 0.0

        elif action == "emergency_halt":
            self.current_speed_rpm = 0.0
            self.is_emergency_halted = True

        elif action == "reset":
            self.current_speed_rpm = 0.0
            self.is_emergency_halted = False

        else:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=f"Unknown actuator action '{action}'.",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )

        return CapabilityResult(
            request_id=request.request_id,
            capability_id=self.descriptor.capability_id,
            success=True,
            output={
                "action": action,
                "speed_rpm": self.current_speed_rpm,
                "angle_deg": self.current_angle_deg,
                "is_halted": self.is_emergency_halted,
            },
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            status=CapabilityStatus.AVAILABLE,
        )


class MockActuatorCapabilityProvider(CapabilityProvider):
    """Provider exposing simulated motor actuator capability."""

    def __init__(self) -> None:
        self._actuator_cap = MockActuatorCapability(provider_id=self.provider_id)

    @property
    def provider_id(self) -> str:
        return "provider.device.mock_actuator"

    @property
    def name(self) -> str:
        return "Mock Device Actuator Provider"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        self._actuator_cap.current_speed_rpm = 0.0

    def get_capabilities(self) -> List[Capability]:
        return [self._actuator_cap]
