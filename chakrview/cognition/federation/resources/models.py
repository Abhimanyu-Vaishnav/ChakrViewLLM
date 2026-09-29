"""
Domain Models for Distributed Resource & Capability Advertisement (Step 37).

Defines canonical representations for hardware resources, computational capabilities,
signed resource advertisements, freshness lifecycle, and sovereign local sharing policies.

Core Invariants:
- RESOURCE_DISCOVERY != RESOURCE_AUTHORIZATION
- RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION
- RESOURCE_ADVERTISEMENT != TRUST
- RESOURCE_ADVERTISEMENT != EXECUTION_AUTHORITY
- RESOURCE_VISIBILITY != RESOURCE_CONTROL
- PEER_RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
- LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
- ΔW = 0
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set

from chakrview.cognition.peering.crypto import (
    Ed25519PrivateKeyWrapper,
    Ed25519PublicKeyWrapper,
)
from chakrview.cognition.peering.models import FederationScope
from chakrview.cognition.federation.resources.errors import (
    InvalidAdvertisementError,
    AdvertisementTamperedError,
    ProhibitedResourceDataError,
)

# Prohibited sensitive keys that can NEVER be included in resource profiles or advertisements
PROHIBITED_RESOURCE_KEYS: Set[str] = {
    "private_key",
    "secret_key",
    "session_secret",
    "model_weights",
    "weight_tensor",
    "password",
    "credential",
    "mac_address",
    "username",
    "home_directory",
    "user_path",
    "serial_number",
}

DEFAULT_ADVERTISEMENT_TTL_SECONDS: float = 60.0
MAX_ADVERTISEMENT_TTL_SECONDS: float = 3600.0


# ============================================================================
# Hardware Resource Models (Neutral, Portable, Extensible)
# ============================================================================

class AcceleratorType(str, Enum):
    """Generic taxonomy for computational accelerators without vendor bias."""
    NONE = "NONE"
    GPU = "GPU"
    NPU = "NPU"
    TPU = "TPU"
    DSP = "DSP"
    CUSTOM = "CUSTOM"


class StorageClass(str, Enum):
    """Storage classification without private filesystem path exposure."""
    FAST_SSD = "FAST_SSD"
    HDD = "HDD"
    RAMDISK = "RAMDISK"
    NETWORK = "NETWORK"
    EPHEMERAL = "EPHEMERAL"


@dataclass
class CPUResource:
    """Hardware-neutral CPU topology and compute capacity description."""
    architecture: str
    logical_cores: int
    physical_cores: Optional[int] = None
    instruction_capabilities: List[str] = field(default_factory=list)
    total_capacity_mhz: Optional[float] = None
    available_cores: float = 1.0
    utilization_percent: Optional[float] = None  # Volatile metric

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CPUResource":
        return cls(**data)


@dataclass
class MemoryResource:
    """System RAM metrics in megabytes."""
    total_memory_mb: int
    available_memory_mb: int
    reservable_memory_mb: Optional[int] = None
    utilization_percent: Optional[float] = None  # Volatile metric

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryResource":
        return cls(**data)


@dataclass
class AcceleratorResource:
    """Generic hardware accelerator description without proprietary drivers."""
    accelerator_type: AcceleratorType = AcceleratorType.NONE
    device_count: int = 0
    model_name: Optional[str] = None
    total_memory_mb: Optional[int] = None
    available_memory_mb: Optional[int] = None
    compute_capabilities: List[str] = field(default_factory=list)
    is_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["accelerator_type"] = self.accelerator_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AcceleratorResource":
        d = dict(data)
        if "accelerator_type" in d and isinstance(d["accelerator_type"], str):
            d["accelerator_type"] = AcceleratorType(d["accelerator_type"])
        return cls(**d)


@dataclass
class StorageResource:
    """Voluntary coarse storage description without path leakage."""
    storage_class: StorageClass = StorageClass.FAST_SSD
    available_storage_mb: int = 0
    total_storage_mb: Optional[int] = None
    capability_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["storage_class"] = self.storage_class.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageResource":
        d = dict(data)
        if "storage_class" in d and isinstance(d["storage_class"], str):
            d["storage_class"] = StorageClass(d["storage_class"])
        return cls(**d)


@dataclass
class PlatformResource:
    """High-level platform and node runtime environment description."""
    os_family: str
    os_release: str
    python_version: str
    chakrview_version: str = "37.0"
    node_epoch: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformResource":
        return cls(**data)


@dataclass
class NodeResourceProfile:
    """
    Complete composite hardware and platform profile of a ChakrView node.
    Enforces privacy boundaries and zero secret leakage.
    """
    profile_id: str
    cpu: CPUResource
    memory: MemoryResource
    platform: PlatformResource
    accelerator: Optional[AcceleratorResource] = None
    storage: Optional[StorageResource] = None
    timestamp: float = field(default_factory=time.time)
    is_coarse: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "platform": self.platform.to_dict(),
            "accelerator": self.accelerator.to_dict() if self.accelerator else None,
            "storage": self.storage.to_dict() if self.storage else None,
            "timestamp": self.timestamp,
            "is_coarse": self.is_coarse,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeResourceProfile":
        d = dict(data)
        d["cpu"] = CPUResource.from_dict(d["cpu"])
        d["memory"] = MemoryResource.from_dict(d["memory"])
        d["platform"] = PlatformResource.from_dict(d["platform"])
        if d.get("accelerator"):
            d["accelerator"] = AcceleratorResource.from_dict(d["accelerator"])
        if d.get("storage"):
            d["storage"] = StorageResource.from_dict(d["storage"])
        return cls(**d)

    def sanitize_for_privacy(self) -> "NodeResourceProfile":
        """Generate a coarse-grained profile that hides fine-grained hardware details."""
        coarse_cpu = CPUResource(
            architecture=self.cpu.architecture,
            logical_cores=self.cpu.logical_cores,
            physical_cores=None,
            instruction_capabilities=[],
            total_capacity_mhz=None,
            available_cores=round(self.cpu.available_cores, 1),
            utilization_percent=None,
        )
        coarse_mem = MemoryResource(
            total_memory_mb=(self.memory.total_memory_mb // 1024) * 1024,
            available_memory_mb=(self.memory.available_memory_mb // 1024) * 1024,
            reservable_memory_mb=None,
            utilization_percent=None,
        )
        coarse_accel = None
        if self.accelerator and self.accelerator.is_available:
            coarse_accel = AcceleratorResource(
                accelerator_type=self.accelerator.accelerator_type,
                device_count=self.accelerator.device_count,
                model_name="Generic Accelerator",
                total_memory_mb=None,
                available_memory_mb=None,
                compute_capabilities=self.accelerator.compute_capabilities,
                is_available=True,
            )
        return NodeResourceProfile(
            profile_id=self.profile_id,
            cpu=coarse_cpu,
            memory=coarse_mem,
            platform=self.platform,
            accelerator=coarse_accel,
            storage=None,  # Suppress storage in coarse mode
            timestamp=self.timestamp,
            is_coarse=True,
        )


# ============================================================================
# Computational Capability Models
# ============================================================================

class ExecutionType(str, Enum):
    """Categorization of computational work supported by advertised capabilities."""
    INFERENCE = "INFERENCE"
    PREPROCESSING = "PREPROCESSING"
    TOKENIZATION = "TOKENIZATION"
    EMBEDDING = "EMBEDDING"
    DOCUMENT_PROCESSING = "DOCUMENT_PROCESSING"
    CRYPTOGRAPHIC_OPS = "CRYPTOGRAPHIC_OPS"
    GENERIC_COMPUTE = "GENERIC_COMPUTE"


@dataclass
class AdvertisedCapability:
    """
    Voluntarily advertised capability manifest.
    CRITICAL: ADVERTISEMENT != PERMISSION.
    """
    capability_id: str
    name: str
    version: str = "1.0.0"
    execution_type: ExecutionType = ExecutionType.GENERIC_COMPUTE
    supported_inputs: List[str] = field(default_factory=list)
    supported_outputs: List[str] = field(default_factory=list)
    concurrency_limit: int = 1
    requires_accelerator: bool = False
    is_exposed: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["execution_type"] = self.execution_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvertisedCapability":
        d = dict(data)
        if "execution_type" in d and isinstance(d["execution_type"], str):
            d["execution_type"] = ExecutionType(d["execution_type"])
        return cls(**d)


@dataclass
class FederationCapabilityProfile:
    """Collection of advertised capabilities supported by a node."""
    capabilities: Dict[str, AdvertisedCapability] = field(default_factory=dict)

    def add_capability(self, cap: AdvertisedCapability) -> None:
        self.capabilities[cap.capability_id] = cap

    def get_capability(self, cap_id: str) -> Optional[AdvertisedCapability]:
        return self.capabilities.get(cap_id)

    def list_capabilities(self, exposed_only: bool = True) -> List[AdvertisedCapability]:
        if exposed_only:
            return [c for c in self.capabilities.values() if c.is_exposed]
        return list(self.capabilities.values())

    def filter_by_execution_type(self, exec_type: ExecutionType) -> List[AdvertisedCapability]:
        return [c for c in self.capabilities.values() if c.execution_type == exec_type and c.is_exposed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederationCapabilityProfile":
        caps = {
            k: AdvertisedCapability.from_dict(v)
            for k, v in data.get("capabilities", {}).items()
        }
        return cls(capabilities=caps)


# ============================================================================
# Resource Advertisement & Freshness Models
# ============================================================================

class AdvertisementFreshness(str, Enum):
    """Lifecycle and freshness state of a resource advertisement."""
    FRESH = "FRESH"              # Within TTL window, active and authoritative
    AGING = "AGING"              # > 50% TTL elapsed, refresh recommended
    STALE = "STALE"              # Past TTL, unverified, cannot be used for matching
    EXPIRED = "EXPIRED"          # Past 2x TTL, scheduled for pruning
    UNAVAILABLE = "UNAVAILABLE"  # Underlying node disconnected or unreachable


@dataclass
class ResourceAdvertisement:
    """
    Cryptographically authenticated, signed statement of node resources & capabilities.
    
    Axiom:
    RESOURCE_ADVERTISEMENT != RESOURCE_PERMISSION
    RESOURCE_ADVERTISEMENT != EXECUTION_AUTHORITY
    PEER_RESOURCE_CLAIM != LOCAL_RESOURCE_FACT
    """
    advertisement_id: str
    node_id: str
    engine_id: str
    zone_id: str
    tenant_id: str
    version: int
    epoch: int
    resource_profile: NodeResourceProfile
    capabilities: List[AdvertisedCapability]
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = DEFAULT_ADVERTISEMENT_TTL_SECONDS
    federation_scope: str = "ALLOW_CAPABILITY_METADATA"
    payload_digest: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.payload_digest:
            self.payload_digest = self.compute_payload_digest()

    def compute_payload_digest(self) -> str:
        """Compute SHA-256 digest over canonical payload data."""
        payload_data = {
            "node_id": self.node_id,
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "federation_scope": self.federation_scope,
            "resource_profile": self.resource_profile.to_dict(),
            "capabilities": [c.to_dict() for c in self.capabilities],
        }
        _assert_no_prohibited_resource_keys(payload_data)
        canonical = json.dumps(payload_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def signing_bytes(self) -> bytes:
        """Deterministic bytes for Ed25519 signing."""
        canonical_str = (
            f"CHAKR_RES_ADV:{self.advertisement_id}:{self.node_id}:{self.version}:"
            f"{self.epoch}:{self.payload_digest}"
        )
        return canonical_str.encode("utf-8")

    def sign(self, private_key: Ed25519PrivateKeyWrapper) -> None:
        """Sign this advertisement using node's private key."""
        self.payload_digest = self.compute_payload_digest()
        self.signature = private_key.sign(self.signing_bytes()).hex()

    def verify_signature(self, public_key: Ed25519PublicKeyWrapper) -> bool:
        """Cryptographically verify Ed25519 signature."""
        if not self.signature:
            return False
        computed = self.compute_payload_digest()
        if computed != self.payload_digest:
            return False
        try:
            sig_bytes = bytes.fromhex(self.signature)
            return public_key.verify(sig_bytes, self.signing_bytes())
        except Exception:
            return False

    def verify_integrity(self) -> bool:
        """Verify that payload digest matches computed content digest."""
        return self.compute_payload_digest() == self.payload_digest

    def evaluate_freshness(self, current_time: Optional[float] = None) -> AdvertisementFreshness:
        """Evaluate current freshness lifecycle status based on wall-clock TTL."""
        now = current_time if current_time is not None else time.time()
        elapsed = now - self.timestamp
        if elapsed < 0:
            # Future timestamp anomaly
            return AdvertisementFreshness.STALE
        if elapsed <= (self.ttl_seconds * 0.5):
            return AdvertisementFreshness.FRESH
        if elapsed <= self.ttl_seconds:
            return AdvertisementFreshness.AGING
        if elapsed <= (self.ttl_seconds * 2.0):
            return AdvertisementFreshness.STALE
        return AdvertisementFreshness.EXPIRED

    def is_stale(self, current_time: Optional[float] = None) -> bool:
        """Returns True if advertisement has exceeded its valid TTL."""
        freshness = self.evaluate_freshness(current_time)
        return freshness in (AdvertisementFreshness.STALE, AdvertisementFreshness.EXPIRED, AdvertisementFreshness.UNAVAILABLE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "advertisement_id": self.advertisement_id,
            "node_id": self.node_id,
            "engine_id": self.engine_id,
            "zone_id": self.zone_id,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "federation_scope": self.federation_scope,
            "payload_digest": self.payload_digest,
            "signature": self.signature,
            "resource_profile": self.resource_profile.to_dict(),
            "capabilities": [c.to_dict() for c in self.capabilities],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceAdvertisement":
        d = dict(data)
        d["resource_profile"] = NodeResourceProfile.from_dict(d["resource_profile"])
        d["capabilities"] = [AdvertisedCapability.from_dict(c) for c in d.get("capabilities", [])]
        return cls(**d)


# ============================================================================
# Local Resource Sharing Policy Model
# ============================================================================

@dataclass
class ResourceSharingPolicy:
    """
    Sovereign local policy defining what resources and capabilities a node
    voluntarily exposes to the federation.
    
    Axiom:
    LOCAL_RESOURCE_POLICY > REMOTE_RESOURCE_CLAIM
    """
    allow_resource_advertisement: bool = True
    max_exposed_cores: Optional[float] = None
    max_exposed_memory_mb: Optional[int] = None
    allow_accelerator_sharing: bool = False
    coarse_privacy_enabled: bool = True
    allowed_tenants: Set[str] = field(default_factory=lambda: {"*"})
    allowed_scopes: Set[str] = field(default_factory=lambda: {"ALLOW_CAPABILITY_METADATA"})
    ttl_seconds: float = DEFAULT_ADVERTISEMENT_TTL_SECONDS
    exposed_capability_ids: Optional[Set[str]] = None  # None = all capabilities

    def can_share_with(self, tenant_id: str, scope: str) -> bool:
        """Verify whether local sharing policy allows sharing with target tenant & scope."""
        if not self.allow_resource_advertisement:
            return False
        tenant_match = "*" in self.allowed_tenants or tenant_id in self.allowed_tenants
        scope_match = scope in self.allowed_scopes
        return tenant_match and scope_match

    def filter_advertisement(self, raw_adv: ResourceAdvertisement) -> ResourceAdvertisement:
        """Sanitize and constrain raw advertisement per local policy limits."""
        prof = raw_adv.resource_profile
        if self.coarse_privacy_enabled:
            prof = prof.sanitize_for_privacy()

        # Constrain CPU cores
        if self.max_exposed_cores is not None:
            prof.cpu.available_cores = min(prof.cpu.available_cores, self.max_exposed_cores)

        # Constrain memory
        if self.max_exposed_memory_mb is not None:
            prof.memory.available_memory_mb = min(prof.memory.available_memory_mb, self.max_exposed_memory_mb)

        # Constrain accelerator
        if not self.allow_accelerator_sharing:
            prof.accelerator = None

        # Filter capabilities
        caps = raw_adv.capabilities
        if self.exposed_capability_ids is not None:
            caps = [c for c in caps if c.capability_id in self.exposed_capability_ids]

        return ResourceAdvertisement(
            advertisement_id=raw_adv.advertisement_id,
            node_id=raw_adv.node_id,
            engine_id=raw_adv.engine_id,
            zone_id=raw_adv.zone_id,
            tenant_id=raw_adv.tenant_id,
            version=raw_adv.version,
            epoch=raw_adv.epoch,
            timestamp=raw_adv.timestamp,
            ttl_seconds=min(raw_adv.ttl_seconds, self.ttl_seconds),
            federation_scope=raw_adv.federation_scope,
            resource_profile=prof,
            capabilities=caps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_resource_advertisement": self.allow_resource_advertisement,
            "max_exposed_cores": self.max_exposed_cores,
            "max_exposed_memory_mb": self.max_exposed_memory_mb,
            "allow_accelerator_sharing": self.allow_accelerator_sharing,
            "coarse_privacy_enabled": self.coarse_privacy_enabled,
            "allowed_tenants": list(self.allowed_tenants),
            "allowed_scopes": list(self.allowed_scopes),
            "ttl_seconds": self.ttl_seconds,
            "exposed_capability_ids": list(self.exposed_capability_ids) if self.exposed_capability_ids is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceSharingPolicy":
        d = dict(data)
        d["allowed_tenants"] = set(d.get("allowed_tenants", ["*"]))
        d["allowed_scopes"] = set(d.get("allowed_scopes", ["ALLOW_CAPABILITY_METADATA"]))
        if d.get("exposed_capability_ids") is not None:
            d["exposed_capability_ids"] = set(d["exposed_capability_ids"])
        return cls(**d)


# ============================================================================
# Security Helper
# ============================================================================

def _assert_no_prohibited_resource_keys(data: Any, path: str = "root") -> None:
    """Recursively verify no prohibited keys or private material exist in resource payload."""
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                raise ProhibitedResourceDataError(f"Dictionary key at {path} must be string, got {type(k)}")
            lower_k = k.lower()
            for prohibited in PROHIBITED_RESOURCE_KEYS:
                if prohibited in lower_k:
                    raise ProhibitedResourceDataError(
                        f"Prohibited sensitive keyword '{prohibited}' detected at key '{path}.{k}'."
                    )
            _assert_no_prohibited_resource_keys(v, f"{path}.{k}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _assert_no_prohibited_resource_keys(item, f"{path}[{idx}]")
