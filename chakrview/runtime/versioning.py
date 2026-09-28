"""
Version Hierarchy & Lineage Tracking for ChakrView (Step 9).

Maintains cryptographic provenance and lineage across specialized brain instances:
Universal Core -> Domain/Enterprise Profile -> User Customization.
Ensures derived versions retain clear ancestral chains without duplicating base model weights.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class BrainProfileType(str, Enum):
    """Classification of derived ChakrView brain profiles."""
    UNIVERSAL = "universal"
    CODING = "coding"
    REASONING = "reasoning"
    ENTERPRISE = "enterprise"
    USER_SPECIFIC = "user_specific"
    RESEARCH = "research"


@dataclass
class BrainVersionManifest:
    """
    Authoritative manifest describing a ChakrView instance and its exact dependencies.
    
    Attributes:
        version_id: Unique semantic identifier (e.g., "chakrview-v0.1.0-companyA-v0.1").
        profile_type: Functional profile type.
        base_model_version: Underlying neural model identifier ("chakrmicro-v0.1").
        base_model_hash: SHA-256 digest of frozen base weights.
        tokenizer_checksum: SHA-256 digest of frozen tokenizer vocabulary/merges.
        parent_version_id: Ancestral version ID (None for root universal instances).
        lineage: Complete chronological list of ancestral version IDs.
        adapter_manifests: Map of active adapter IDs to their respective SHA-256 hashes.
        skills: List of registered skill identifiers bound to this instance.
        knowledge_indexes: List of bound knowledge index identifiers.
        runtime_config_hash: Optional hash of runtime configuration.
        created_at: ISO timestamp of registration.
        metadata: Custom user, organization, or governance metadata.
    """
    version_id: str
    profile_type: BrainProfileType
    base_model_version: str
    base_model_hash: str
    tokenizer_checksum: str
    parent_version_id: Optional[str] = None
    lineage: List[str] = field(default_factory=list)
    adapter_manifests: Dict[str, str] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    knowledge_indexes: List[str] = field(default_factory=list)
    runtime_config_hash: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Guarantee version_id is in lineage
        if self.version_id not in self.lineage:
            self.lineage.append(self.version_id)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["profile_type"] = self.profile_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainVersionManifest":
        data = dict(data)
        if isinstance(data.get("profile_type"), str):
            data["profile_type"] = BrainProfileType(data["profile_type"])
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "BrainVersionManifest":
        return cls.from_dict(json.loads(json_str))

    def verify_parent(self, parent_manifest: "BrainVersionManifest") -> bool:
        """Verify that this manifest correctly inherits from parent."""
        if self.parent_version_id != parent_manifest.version_id:
            return False
        # Base model and tokenizer must match parent
        if self.base_model_hash != parent_manifest.base_model_hash:
            return False
        if self.tokenizer_checksum != parent_manifest.tokenizer_checksum:
            return False
        return True


class VersionLineageTracker:
    """
    Tracks and validates manifest chains across versions.
    """

    def __init__(self, storage_dir: Optional[Path | str] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: Dict[str, BrainVersionManifest] = {}

    def register_version(self, manifest: BrainVersionManifest) -> None:
        """Register a version manifest, verifying lineage continuity."""
        if manifest.version_id in self._manifests:
            raise ValueError(f"Version '{manifest.version_id}' is already registered.")

        if manifest.parent_version_id:
            parent = self._manifests.get(manifest.parent_version_id)
            if parent is None:
                raise ValueError(
                    f"Parent version '{manifest.parent_version_id}' not found in registry."
                )
            if not manifest.verify_parent(parent):
                raise ValueError(
                    f"Lineage verification failed: manifest invariants do not match parent '{parent.version_id}'."
                )

        self._manifests[manifest.version_id] = manifest
        if self.storage_dir:
            path = self.storage_dir / f"{manifest.version_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(manifest.to_json())

    def get_version(self, version_id: str) -> Optional[BrainVersionManifest]:
        return self._manifests.get(version_id)

    def get_lineage(self, version_id: str) -> List[BrainVersionManifest]:
        """Return complete chain of ancestor manifests from root to version_id."""
        manifest = self.get_version(version_id)
        if not manifest:
            return []

        chain: List[BrainVersionManifest] = []
        curr: Optional[BrainVersionManifest] = manifest
        while curr:
            chain.append(curr)
            curr = self.get_version(curr.parent_version_id) if curr.parent_version_id else None

        chain.reverse()
        return chain

    def count(self) -> int:
        return len(self._manifests)
