"""
Distributed Resource & Capability Advertisement Subsystem for ChakrView (Step 37).

Enables ChakrView nodes to securely and voluntarily describe hardware resources,
computational capabilities, and availability without compromising local authority
or granting ambient execution permissions.

Axioms:
- RESOURCE_DISCOVERY != RESOURCE_AUTHORIZATION
- RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION
- RESOURCE_ADVERTISEMENT != TRUST
- RESOURCE_ADVERTISEMENT != EXECUTION_AUTHORITY
- RESOURCE_VISIBILITY != RESOURCE_CONTROL
- PEER_RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
- LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
- Zero ambient execution rights
- ΔW = 0
"""

from chakrview.cognition.federation.resources.errors import (
    FederationResourceError,
    ResourceDiscoveryError,
    UnsupportedPlatformError,
    ResourceAdvertisementError,
    InvalidAdvertisementError,
    StaleAdvertisementError,
    AdvertisementTamperedError,
    AdvertisementReplayError,
    ProhibitedResourceDataError,
    ResourcePolicyError,
    UnauthorizedResourceAccessError,
    TenantResourceIsolationError,
    ResourceAuthorityViolationError,
    ResourceRegistryError,
    PeerResourceNotFoundError,
    DuplicateAdvertisementError,
)

from chakrview.cognition.federation.resources.models import (
    AcceleratorType,
    StorageClass,
    CPUResource,
    MemoryResource,
    AcceleratorResource,
    StorageResource,
    PlatformResource,
    NodeResourceProfile,
    ExecutionType,
    AdvertisedCapability,
    FederationCapabilityProfile,
    AdvertisementFreshness,
    ResourceAdvertisement,
    ResourceSharingPolicy,
    DEFAULT_ADVERTISEMENT_TTL_SECONDS,
    MAX_ADVERTISEMENT_TTL_SECONDS,
)

from chakrview.cognition.federation.resources.discovery import LocalResourceDetector

from chakrview.cognition.federation.resources.registry import FederationResourceRegistry

from chakrview.cognition.federation.resources.manager import FederationResourceManager

__all__ = [
    # Errors
    "FederationResourceError",
    "ResourceDiscoveryError",
    "UnsupportedPlatformError",
    "ResourceAdvertisementError",
    "InvalidAdvertisementError",
    "StaleAdvertisementError",
    "AdvertisementTamperedError",
    "AdvertisementReplayError",
    "ProhibitedResourceDataError",
    "ResourcePolicyError",
    "UnauthorizedResourceAccessError",
    "TenantResourceIsolationError",
    "ResourceAuthorityViolationError",
    "ResourceRegistryError",
    "PeerResourceNotFoundError",
    "DuplicateAdvertisementError",
    # Models
    "AcceleratorType",
    "StorageClass",
    "CPUResource",
    "MemoryResource",
    "AcceleratorResource",
    "StorageResource",
    "PlatformResource",
    "NodeResourceProfile",
    "ExecutionType",
    "AdvertisedCapability",
    "FederationCapabilityProfile",
    "AdvertisementFreshness",
    "ResourceAdvertisement",
    "ResourceSharingPolicy",
    "DEFAULT_ADVERTISEMENT_TTL_SECONDS",
    "MAX_ADVERTISEMENT_TTL_SECONDS",
    # Components
    "LocalResourceDetector",
    "FederationResourceRegistry",
    "FederationResourceManager",
]
