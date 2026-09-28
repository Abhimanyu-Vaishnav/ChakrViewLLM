"""
Sovereign Capability Registry for ChakrView (Step 17).

Provides deterministic registration, discovery, querying, and lifecycle tracking
of capabilities without granting autonomous execution authority.

Architectural Rule:
REGISTRATION != AUTHORIZATION
Registration merely establishes the existence, schemas, and provider contracts
of a capability. Authority to execute is governed strictly by CapabilityGate.
"""

from typing import Dict, List, Optional, Set
import re

from chakrview.capability.contract import (
    Capability,
    CapabilityDescriptor,
    CapabilityCategory,
    RiskClassification,
    CapabilityStatus,
)


class CapabilityAlreadyRegisteredError(ValueError):
    """Raised when attempting to register a duplicate capability ID without overwrite."""
    pass


class CapabilityNotFoundError(KeyError):
    """Raised when looking up an unregistered capability."""
    pass


class CapabilityVersionConflictError(ValueError):
    """Raised when capability version does not satisfy compatibility constraints."""
    pass


class CapabilityRegistry:
    """
    Central discovery registry for all available system capabilities.
    
    Maintains active descriptors and instances for software, edge device,
    and physical device capabilities.
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability, overwrite: bool = False) -> None:
        """
        Register a new capability instance.
        
        Args:
            capability: Capability instance implementing the Capability contract.
            overwrite: Whether to overwrite existing capability with same ID.
            
        Raises:
            CapabilityAlreadyRegisteredError: If already registered and overwrite is False.
        """
        desc = capability.descriptor
        cap_id = desc.capability_id

        if not cap_id or not isinstance(cap_id, str):
            raise ValueError("Capability ID must be a non-empty string.")

        if cap_id in self._capabilities and not overwrite:
            existing_desc = self._capabilities[cap_id].descriptor
            raise CapabilityAlreadyRegisteredError(
                f"Capability '{cap_id}' (version {existing_desc.version}) is already registered."
            )

        self._capabilities[cap_id] = capability

    def unregister(self, capability_id: str) -> Capability:
        """
        Remove a capability from the registry.
        
        Raises:
            CapabilityNotFoundError: If capability_id is not registered.
        """
        if capability_id not in self._capabilities:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' not found in registry.")
        return self._capabilities.pop(capability_id)

    def get(self, capability_id: str) -> Capability:
        """
        Retrieve a capability instance by ID.
        
        Raises:
            CapabilityNotFoundError: If capability_id is not registered.
        """
        if capability_id not in self._capabilities:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' not found in registry.")
        return self._capabilities[capability_id]

    def get_descriptor(self, capability_id: str) -> CapabilityDescriptor:
        """
        Retrieve descriptor metadata for a capability by ID.
        """
        return self.get(capability_id).descriptor

    def has(self, capability_id: str) -> bool:
        """Check whether capability_id is registered."""
        return capability_id in self._capabilities

    def is_available(self, capability_id: str) -> bool:
        """
        Check whether capability is registered and currently in an operational state.
        """
        if capability_id not in self._capabilities:
            return False
        cap = self._capabilities[capability_id]
        status = cap.check_health()
        return status in (CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED)

    def list_capabilities(
        self,
        category: Optional[CapabilityCategory] = None,
        risk_level: Optional[RiskClassification] = None,
        status: Optional[CapabilityStatus] = None,
        provider_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[CapabilityDescriptor]:
        """
        List capability descriptors matching query filters.
        """
        results: List[CapabilityDescriptor] = []
        for cap in self._capabilities.values():
            desc = cap.descriptor
            if category is not None and desc.category != category:
                continue
            if risk_level is not None and desc.risk_level != risk_level:
                continue
            if status is not None and desc.status != status:
                continue
            if provider_id is not None and desc.provider_id != provider_id:
                continue
            if tag is not None and tag not in desc.tags:
                continue
            results.append(desc)
        return results

    def verify_version_compatibility(
        self,
        capability_id: str,
        required_min_version: str,
    ) -> bool:
        """
        Check whether registered capability satisfies a minimum semantic version.
        """
        cap = self.get(capability_id)
        current = cap.descriptor.version
        
        def parse_version(v: str) -> List[int]:
            # Extract numbers: e.g. "1.2.3" -> [1, 2, 3]
            parts = re.findall(r"\d+", v)
            return [int(p) for p in parts] if parts else [0]

        curr_parts = parse_version(current)
        req_parts = parse_version(required_min_version)

        # Pad to equal length
        max_len = max(len(curr_parts), len(req_parts))
        curr_parts.extend([0] * (max_len - len(curr_parts)))
        req_parts.extend([0] * (max_len - len(req_parts)))

        if curr_parts < req_parts:
            raise CapabilityVersionConflictError(
                f"Capability '{capability_id}' version {current} is below required minimum {required_min_version}."
            )
        return True

    def count(self) -> int:
        """Total number of registered capabilities."""
        return len(self._capabilities)

    def clear(self) -> None:
        """Unregister all capabilities."""
        self._capabilities.clear()
