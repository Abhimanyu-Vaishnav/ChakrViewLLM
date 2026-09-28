"""
Sovereign Capability Bridges and Adapters for ChakrView (Step 17).

Provides seamless interoperability between existing Step 11/15 Tool abstractions
and the new Step 17 Capability subsystem:
- ToolCapabilityAdapter: Wraps any existing Tool as a governed Capability.
- get_standard_capability_registry(): Initializes a registry populated with default providers.
- CapabilityObservationAdapter: Converts a CapabilityResult into a StepObservation.
"""

import time
from typing import Dict, List, Optional, Any

from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityCategory,
    RiskClassification,
    CapabilityStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityContext,
)
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.provider import (
    CalculatorCapabilityProvider,
    TextTransformCapabilityProvider,
    ClockCapabilityProvider,
    MockSensorCapabilityProvider,
    MockActuatorCapabilityProvider,
)
from chakrview.runtime.tools import Tool, ToolRegistry, get_standard_tool_registry
from chakrview.cognition.observation import StepObservation


class ToolCapabilityAdapter(Capability):
    """
    Adapts an existing Step 11 Tool into a Step 17 Capability.
    """

    def __init__(self, tool: Tool) -> None:
        self.tool = tool
        self._descriptor = CapabilityDescriptor(
            capability_id=tool.tool_id,
            name=tool.name,
            version="1.0.0",
            description=tool.description,
            category=CapabilityCategory.SOFTWARE,
            risk_level=RiskClassification.COMPUTE,
            provider_id="provider.bridged_tool",
            input_schema=tool.parameters_schema,
            required_permissions=[f"capability.tool.{tool.tool_id}"],
            tags=["bridged", "tool"],
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

        try:
            tool_res = self.tool.execute(**request.parameters)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out_val = tool_res.output
            if tool_res.success and not isinstance(out_val, dict):
                out_val = {"result": out_val}
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=tool_res.success,
                output=out_val,
                error=tool_res.error,
                execution_time_ms=elapsed_ms,
                status=CapabilityStatus.AVAILABLE if tool_res.success else CapabilityStatus.ERROR,
            )
        except Exception as e:
            return CapabilityResult(
                request_id=request.request_id,
                capability_id=self.descriptor.capability_id,
                success=False,
                error=f"Bridged tool exception: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                status=CapabilityStatus.ERROR,
            )


def capability_result_to_observation(
    step_id: str,
    result: CapabilityResult,
    descriptor: Optional[CapabilityDescriptor] = None,
) -> StepObservation:
    """
    Convert a CapabilityResult into a StepObservation for cognitive execution graphs.
    """
    category = descriptor.category.value if descriptor else "UNKNOWN"
    risk = descriptor.risk_level.value if descriptor else "UNKNOWN"

    return StepObservation(
        step_id=step_id,
        success=result.success,
        output=result.output,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        provenance={
            "capability_id": result.capability_id,
            "category": category,
            "risk_level": risk,
            "request_id": result.request_id,
            "status": result.status.value,
        },
        verification_status=result.success,
        verification_notes=(
            f"Capability '{result.capability_id}' executed successfully."
            if result.success
            else f"Capability error: {result.error}"
        ),
    )


def get_standard_capability_registry(
    include_bridged_tools: bool = True,
) -> CapabilityRegistry:
    """
    Initialize a standard CapabilityRegistry populated with reference providers.
    
    Optionally bridges existing tools from the runtime ToolRegistry.
    """
    registry = CapabilityRegistry()

    # 1. Register Core Providers
    providers = [
        CalculatorCapabilityProvider(),
        TextTransformCapabilityProvider(),
        ClockCapabilityProvider(),
        MockSensorCapabilityProvider(),
        MockActuatorCapabilityProvider(),
    ]

    for p in providers:
        p.initialize()
        for cap in p.get_capabilities():
            registry.register(cap, overwrite=True)

    # 2. Bridge Existing Tools
    if include_bridged_tools:
        tool_reg = get_standard_tool_registry()
        for tool in tool_reg.list_tools():
            if not registry.has(tool.tool_id):
                registry.register(ToolCapabilityAdapter(tool), overwrite=False)

    return registry
