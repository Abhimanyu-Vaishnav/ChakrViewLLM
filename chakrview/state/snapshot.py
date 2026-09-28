"""
Cognitive State Snapshot and Comparison Engine for ChakrView (Step 18).

Provides immutable, strongly typed, serializable captures of full cognitive state:
- CognitiveStateSnapshot: point-in-time state capture
- SnapshotMetadata: lightweight audit descriptor
- SnapshotDiff: structural comparison between snapshots
- compare_snapshots: deterministic state delta calculator

Architectural Rules:
1. Snapshots are IMMUTABLE once created.
2. Serialization is strictly JSON-compatible (NO pickle).
3. Rollbacks NEVER erase audit history; they produce new versioned transitions.
"""

from dataclasses import dataclass, field, asdict
import json
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.state.identity import SystemIdentity
from chakrview.state.epistemic import KnowledgeState
from chakrview.state.uncertainty import UncertaintyState
from chakrview.state.task_state import TaskState
from chakrview.state.environment_state import EnvironmentState
from chakrview.state.capability_state import CapabilityState
from chakrview.state.constraints import ConstraintState


@dataclass(frozen=True)
class SnapshotMetadata:
    """
    Lightweight audit header for a state snapshot.
    """
    snapshot_id: str
    state_version: int
    timestamp: float
    owner_id: str
    session_id: str
    description: str = ""
    parent_snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMetadata":
        return cls(**data)


@dataclass
class CognitiveStateSnapshot:
    """
    Complete, immutable capture of the system's cognitive state at a point in time.
    """
    snapshot_id: str
    state_version: int
    timestamp: float
    owner_id: str
    session_id: str
    identity: SystemIdentity
    environment: EnvironmentState
    capabilities: CapabilityState
    tasks: Dict[str, TaskState]
    knowledge: KnowledgeState
    uncertainties: UncertaintyState
    constraints: ConstraintState
    description: str = ""
    parent_snapshot_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> SnapshotMetadata:
        """Extract lightweight metadata header."""
        return SnapshotMetadata(
            snapshot_id=self.snapshot_id,
            state_version=self.state_version,
            timestamp=self.timestamp,
            owner_id=self.owner_id,
            session_id=self.session_id,
            description=self.description,
            parent_snapshot_id=self.parent_snapshot_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to safe, JSON-serializable dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "timestamp": self.timestamp,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "description": self.description,
            "parent_snapshot_id": self.parent_snapshot_id,
            "provenance": self.provenance,
            "identity": self.identity.to_dict(),
            "environment": self.environment.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "knowledge": self.knowledge.to_dict(),
            "uncertainties": self.uncertainties.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to JSON string without pickle or unsafe code."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveStateSnapshot":
        """Deserialize from structured dictionary with validation."""
        tasks = {
            k: TaskState.from_dict(v)
            for k, v in data.get("tasks", {}).items()
        }
        return cls(
            snapshot_id=data["snapshot_id"],
            state_version=data["state_version"],
            timestamp=data["timestamp"],
            owner_id=data["owner_id"],
            session_id=data["session_id"],
            description=data.get("description", ""),
            parent_snapshot_id=data.get("parent_snapshot_id"),
            provenance=data.get("provenance", {}),
            identity=SystemIdentity.from_dict(data["identity"]),
            environment=EnvironmentState.from_dict(data["environment"]),
            capabilities=CapabilityState.from_dict(data["capabilities"]),
            tasks=tasks,
            knowledge=KnowledgeState.from_dict(data["knowledge"]),
            uncertainties=UncertaintyState.from_dict(data["uncertainties"]),
            constraints=ConstraintState.from_dict(data["constraints"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CognitiveStateSnapshot":
        """Safely deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class SnapshotDiff:
    """
    Structured comparison between two CognitiveStateSnapshot instances.
    """
    diff_id: str
    base_snapshot_id: str
    target_snapshot_id: str
    base_version: int
    target_version: int
    task_changes: Dict[str, Any]
    knowledge_changes: Dict[str, Any]
    capability_changes: Dict[str, Any]
    environment_changes: Dict[str, Any]
    uncertainty_changes: Dict[str, Any]
    has_differences: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_snapshots(
    base: CognitiveStateSnapshot,
    target: CognitiveStateSnapshot,
) -> SnapshotDiff:
    """
    Compute structured differences between two state snapshots.
    """
    # 1. Task differences
    task_changes: Dict[str, Any] = {
        "added_tasks": [],
        "removed_tasks": [],
        "modified_tasks": {},
    }
    base_task_ids = set(base.tasks.keys())
    target_task_ids = set(target.tasks.keys())
    task_changes["added_tasks"] = list(target_task_ids - base_task_ids)
    task_changes["removed_tasks"] = list(base_task_ids - target_task_ids)
    for tid in base_task_ids.intersection(target_task_ids):
        bt = base.tasks[tid]
        tt = target.tasks[tid]
        if bt.phase != tt.phase or bt.status != tt.status:
            task_changes["modified_tasks"][tid] = {
                "phase_change": f"{bt.phase.value} -> {tt.phase.value}",
                "status_change": f"{bt.status} -> {tt.status}",
            }

    # 2. Knowledge differences
    knowledge_changes: Dict[str, Any] = {
        "added_assertions": [],
        "removed_assertions": [],
        "status_changes": {},
    }
    base_assert_ids = set(base.knowledge.assertions.keys())
    target_assert_ids = set(target.knowledge.assertions.keys())
    knowledge_changes["added_assertions"] = list(target_assert_ids - base_assert_ids)
    knowledge_changes["removed_assertions"] = list(base_assert_ids - target_assert_ids)
    for aid in base_assert_ids.intersection(target_assert_ids):
        ba = base.knowledge.assertions[aid]
        ta = target.knowledge.assertions[aid]
        if ba.status != ta.status or ba.value != ta.value:
            knowledge_changes["status_changes"][aid] = {
                "old_status": ba.status.value,
                "new_status": ta.status.value,
                "value_changed": ba.value != ta.value,
            }

    # 3. Capability differences
    capability_changes: Dict[str, Any] = {
        "status_changes": {},
    }
    for cid, tobs in target.capabilities.capabilities.items():
        if cid in base.capabilities.capabilities:
            bobs = base.capabilities.capabilities[cid]
            if bobs.observed_status != tobs.observed_status:
                capability_changes["status_changes"][cid] = {
                    "old_status": bobs.observed_status.value,
                    "new_status": tobs.observed_status.value,
                }
        else:
            capability_changes["status_changes"][cid] = {
                "old_status": "NONE",
                "new_status": tobs.observed_status.value,
            }

    # 4. Environment differences
    environment_changes: Dict[str, Any] = {}
    if base.environment.operational_mode != target.environment.operational_mode:
        environment_changes["mode_change"] = {
            "from": base.environment.operational_mode.value,
            "to": target.environment.operational_mode.value,
        }
    if base.environment.network_status != target.environment.network_status:
        environment_changes["network_change"] = {
            "from": base.environment.network_status.value,
            "to": target.environment.network_status.value,
        }

    # 5. Uncertainty differences
    base_unc_keys = set(base.uncertainties.uncertainties.keys())
    target_unc_keys = set(target.uncertainties.uncertainties.keys())
    uncertainty_changes = {
        "added_uncertainties": list(target_unc_keys - base_unc_keys),
        "resolved_uncertainties": list(base_unc_keys - target_unc_keys),
    }

    has_diff = any([
        bool(task_changes["added_tasks"]),
        bool(task_changes["removed_tasks"]),
        bool(task_changes["modified_tasks"]),
        bool(knowledge_changes["added_assertions"]),
        bool(knowledge_changes["removed_assertions"]),
        bool(knowledge_changes["status_changes"]),
        bool(capability_changes["status_changes"]),
        bool(environment_changes),
        bool(uncertainty_changes["added_uncertainties"]),
        bool(uncertainty_changes["resolved_uncertainties"]),
    ])

    return SnapshotDiff(
        diff_id=f"diff_{uuid.uuid4().hex[:8]}",
        base_snapshot_id=base.snapshot_id,
        target_snapshot_id=target.snapshot_id,
        base_version=base.state_version,
        target_version=target.state_version,
        task_changes=task_changes,
        knowledge_changes=knowledge_changes,
        capability_changes=capability_changes,
        environment_changes=environment_changes,
        uncertainty_changes=uncertainty_changes,
        has_differences=has_diff,
    )
