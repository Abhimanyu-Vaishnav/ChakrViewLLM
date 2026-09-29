"""
Strongly Typed Error Hierarchy for Distributed Resource & Capability Advertisement (Step 37).

Enforces fail-closed semantics across resource discovery, capability advertisement,
integrity verification, tenant isolation, and peer resource evaluation.
"""

from typing import Optional


class FederationResourceError(Exception):
    """Base exception for all federation resource and capability advertisement operations."""
    pass


# ============================================================================
# Discovery & Detection Errors
# ============================================================================

class ResourceDiscoveryError(FederationResourceError):
    """Raised when local hardware discovery or environment profiling fails."""
    pass


class UnsupportedPlatformError(ResourceDiscoveryError):
    """Raised when an operation is requested on an unsupported hardware platform."""
    pass


# ============================================================================
# Advertisement & Serialization Errors
# ============================================================================

class ResourceAdvertisementError(FederationResourceError):
    """Base exception for resource advertisement processing errors."""
    pass


class InvalidAdvertisementError(ResourceAdvertisementError):
    """Raised when a resource advertisement violates schema or contains invalid values."""
    pass


class StaleAdvertisementError(ResourceAdvertisementError):
    """Raised when an advertisement is rejected or access denied due to expired TTL."""
    pass


class AdvertisementTamperedError(ResourceAdvertisementError):
    """Raised when an advertisement fails cryptographic signature or digest verification."""
    pass


class AdvertisementReplayError(ResourceAdvertisementError):
    """Raised when an advertisement contains a regressive version or replayed identifier."""
    pass


class ProhibitedResourceDataError(ResourceAdvertisementError):
    """Raised when an advertisement attempts to leak private keys, secrets, model weights, or private paths."""
    pass


# ============================================================================
# Policy & Authorization Errors
# ============================================================================

class ResourcePolicyError(FederationResourceError):
    """Base exception for resource sharing and governance policy violations."""
    pass


class UnauthorizedResourceAccessError(ResourcePolicyError):
    """Raised when an unauthorized peer attempts to inspect or query private resources."""
    pass


class TenantResourceIsolationError(ResourcePolicyError):
    """Raised when a cross-tenant resource advertisement or query violation occurs."""
    pass


class ResourceAuthorityViolationError(ResourcePolicyError):
    """Raised when a remote peer attempts to claim authority or permission over local resources."""
    pass


# ============================================================================
# Registry Errors
# ============================================================================

class ResourceRegistryError(FederationResourceError):
    """Base exception for resource registry operations."""
    pass


class PeerResourceNotFoundError(ResourceRegistryError):
    """Raised when queried peer resource information does not exist in the registry."""
    pass


class DuplicateAdvertisementError(ResourceRegistryError):
    """Raised when a duplicate advertisement is registered within the same epoch/version."""
    pass
