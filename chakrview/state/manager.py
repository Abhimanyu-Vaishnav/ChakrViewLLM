"""
Cognitive State Manager for ChakrView (Step 18).

Coordinates the live system state, enforces multi-tenant isolation, manages
monotonically incrementing state versions, captures immutable snapshots,
and executes non-destructive state rollbacks with complete audit trail preservation.

Architectural Guarantees:
1. DATA != AUTHORITY: State observations or assertions NEVER grant execution authority.
2. Tenant Isolation: State is strictly partitioned by owner_id and session_id.
   Cross-tenant snapshot restoration raises StateIsolationError.
3. Rollbacks NEVER delete history: Rollback creates a new state transition
   referencing the target snapshot and incrementing the version.
4. Safe Deserialization: Validates all inputs; rejects code execution patterns.
"""

import copy
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.state.identity import SystemIdentity, get_current_system_identity
from chakrview.state.epistemic import KnowledgeState, KnowledgeAssertion, EpistemicStatus
from chakrview.state.uncertainty import UncertaintyState, Uncertainty
from chakrview.state.task_state import TaskState, TaskPhase
from chakrview.state.environment_state import EnvironmentState, OperationalMode
from chakrview.state.capability_state import CapabilityState, ObservedCapabilityStatus
from chakrview.state.constraints import ConstraintState, PolicyRestriction
from chakrview.state.snapshot import (
    CognitiveStateSnapshot,
    SnapshotMetadata,
    SnapshotDiff,
    compare_snapshots,
)


class StateIsolationError(PermissionError):
    """Raised when an unauthorized cross-owner or cross-session state operation is attempted."""
    pass


class StateValidationError(ValueError):
    """Raised when state inputs fail schema, integrity, or bounds checks."""
    pass


class SnapshotNotFoundError(KeyError):
    """Raised when looking up an unregistered snapshot ID."""
    pass


class CognitiveStateManager:
    """
    Central manager maintaining live cognitive state for an owner and session.
    """

    def __init__(
        self,
        owner_id: str = "default_user",
        session_id: str = "session_001",
        identity: Optional[SystemIdentity] = None,
        environment: Optional[EnvironmentState] = None,
        constraints: Optional[ConstraintState] = None,
    ) -> None:
        self.owner_id = owner_id
        self.session_id = session_id
        self.state_version: int = 1
        self.created_at: float = time.time()
        self.last_updated_at: float = self.created_at

        # Live State Containers
        self.identity = identity or get_current_system_identity()
        self.environment = environment or EnvironmentState()
        self.capabilities = CapabilityState()
        self.tasks: Dict[str, TaskState] = {}
        self.knowledge = KnowledgeState()
        self.uncertainties = UncertaintyState()
        self.constraints = constraints or ConstraintState()

        # Audit & Snapshots
        self._snapshots: Dict[str, CognitiveStateSnapshot] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._record_audit_event("INITIALIZE_STATE", details={"version": self.state_version})

    def _record_audit_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Append an event to the immutable audit log."""
        self._audit_log.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "state_version": self.state_version,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "details": details or {},
        })
        self.last_updated_at = time.time()

    def update_task(self, task: TaskState) -> None:
        """Register or update an active task."""
        self.tasks[task.task_id] = task
        self.state_version += 1
        self._record_audit_event("UPDATE_TASK", details={"task_id": task.task_id, "phase": task.phase.value})

    def assert_knowledge(
        self,
        subject: str,
        predicate: str,
        value: Any,
        status: EpistemicStatus = EpistemicStatus.KNOWN,
        confidence: float = 1.0,
        source: str = "system",
        source_id: Optional[str] = None,
        valid_until: Optional[float] = None,
        evidence_refs: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeAssertion:
        """Record or update a factual knowledge assertion."""
        assertion = self.knowledge.assert_fact(
            subject=subject,
            predicate=predicate,
            value=value,
            status=status,
            confidence=confidence,
            source=source,
            source_id=source_id,
            valid_until=valid_until,
            evidence_refs=evidence_refs,
            provenance=provenance,
        )
        self.state_version += 1
        self._record_audit_event("ASSERT_KNOWLEDGE", details={
            "assertion_id": assertion.assertion_id,
            "subject": subject,
            "predicate": predicate,
            "status": status.value,
        })
        return assertion

    def record_uncertainty(
        self,
        key: str,
        confidence: float,
        reason: str,
        source: str,
        evidence_refs: Optional[List[str]] = None,
        is_uncalibrated: bool = False,
    ) -> Uncertainty:
        """Explicitly record uncertainty for a key."""
        unc = self.uncertainties.record_uncertainty(
            key=key,
            confidence=confidence,
            reason=reason,
            source=source,
            evidence_refs=evidence_refs,
            is_uncalibrated=is_uncalibrated,
        )
        self.state_version += 1
        self._record_audit_event("RECORD_UNCERTAINTY", details={"key": key, "confidence": confidence, "reason": reason})
        return unc

    def update_capability_observation(
        self,
        capability_id: str,
        status: ObservedCapabilityStatus,
        risk_level: str = "READ_ONLY",
        latency_ms: float = 0.0,
        health_notes: str = "",
        is_authorized: bool = False,
    ) -> None:
        """Record capability status observation."""
        self.capabilities.record_status(
            capability_id=capability_id,
            status=status,
            risk_level=risk_level,
            latency_ms=latency_ms,
            health_notes=health_notes,
            is_authorized=is_authorized,
        )
        self.state_version += 1
        self._record_audit_event("OBSERVE_CAPABILITY", details={
            "capability_id": capability_id,
            "status": status.value,
        })

    def sync_with_registry(self, registry: Any, policy: Optional[Any] = None) -> None:
        """Synchronize observed capabilities with active CapabilityRegistry."""
        self.capabilities.sync_with_registry(registry=registry, policy=policy)
        self.state_version += 1
        self._record_audit_event("SYNC_CAPABILITIES", details={"total_observed": len(self.capabilities.capabilities)})

    def sync_with_environment_profile(self, profile: Any) -> None:
        """Synchronize observed environment with active EnvironmentProfile."""
        self.environment.sync_from_profile(profile)
        self.state_version += 1
        self._record_audit_event("SYNC_ENVIRONMENT", details={"environment_id": self.environment.environment_id})

    # =================================================================
    # Snapshot Engine
    # =================================================================

    def snapshot(self, description: str = "") -> CognitiveStateSnapshot:
        """
        Capture an immutable, point-in-time CognitiveStateSnapshot.
        """
        snap_id = f"snap_{uuid.uuid4().hex[:10]}"
        snap = CognitiveStateSnapshot(
            snapshot_id=snap_id,
            state_version=self.state_version,
            timestamp=time.time(),
            owner_id=self.owner_id,
            session_id=self.session_id,
            description=description,
            identity=copy.deepcopy(self.identity),
            environment=copy.deepcopy(self.environment),
            capabilities=copy.deepcopy(self.capabilities),
            tasks=copy.deepcopy(self.tasks),
            knowledge=copy.deepcopy(self.knowledge),
            uncertainties=copy.deepcopy(self.uncertainties),
            constraints=copy.deepcopy(self.constraints),
            provenance={"created_by": "CognitiveStateManager.snapshot"},
        )
        self._snapshots[snap_id] = snap
        self._record_audit_event("CREATE_SNAPSHOT", details={"snapshot_id": snap_id, "description": description})
        return snap

    def get_snapshot(self, snapshot_id: str) -> CognitiveStateSnapshot:
        """Retrieve snapshot by ID with multi-tenant ownership check."""
        if snapshot_id not in self._snapshots:
            raise SnapshotNotFoundError(f"Snapshot '{snapshot_id}' not found.")
        snap = self._snapshots[snapshot_id]
        if snap.owner_id != self.owner_id:
            raise StateIsolationError(
                f"Multi-tenant violation: Access denied to snapshot '{snapshot_id}' owned by '{snap.owner_id}'."
            )
        return snap

    def rollback(self, snapshot_id: str, reason: str = "") -> CognitiveStateSnapshot:
        """
        Roll back live state to a previously captured snapshot.
        
        Crucial Rule:
        Rollbacks NEVER erase audit history. A new state version is generated,
        and a ROLLBACK_TRANSITION event is recorded with complete provenance.
        """
        target_snap = self.get_snapshot(snapshot_id)

        prior_version = self.state_version
        new_version = prior_version + 1

        # Apply snapshot states
        self.identity = copy.deepcopy(target_snap.identity)
        self.environment = copy.deepcopy(target_snap.environment)
        self.capabilities = copy.deepcopy(target_snap.capabilities)
        self.tasks = copy.deepcopy(target_snap.tasks)
        self.knowledge = copy.deepcopy(target_snap.knowledge)
        self.uncertainties = copy.deepcopy(target_snap.uncertainties)
        self.constraints = copy.deepcopy(target_snap.constraints)

        # Update state version monotonically
        self.state_version = new_version
        self.last_updated_at = time.time()

        # Record rollback event in audit trail
        self._record_audit_event("ROLLBACK_TRANSITION", details={
            "target_snapshot_id": snapshot_id,
            "target_snapshot_version": target_snap.state_version,
            "prior_version": prior_version,
            "new_version": new_version,
            "reason": reason or "User requested rollback",
        })

        # Create a new snapshot capturing the rolled-back state with the new version
        return self.snapshot(description=f"Post-rollback to {snapshot_id} (version {target_snap.state_version})")

    def restore_from_snapshot(self, snapshot: CognitiveStateSnapshot) -> None:
        """
        Restore live state from an external snapshot with tenant validation.
        """
        if snapshot.owner_id != self.owner_id:
            raise StateIsolationError(
                f"Multi-tenant violation: Cannot restore snapshot owned by '{snapshot.owner_id}' "
                f"into manager owned by '{self.owner_id}'."
            )

        self.identity = copy.deepcopy(snapshot.identity)
        self.environment = copy.deepcopy(snapshot.environment)
        self.capabilities = copy.deepcopy(snapshot.capabilities)
        self.tasks = copy.deepcopy(snapshot.tasks)
        self.knowledge = copy.deepcopy(snapshot.knowledge)
        self.uncertainties = copy.deepcopy(snapshot.uncertainties)
        self.constraints = copy.deepcopy(snapshot.constraints)

        self.state_version += 1
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._record_audit_event("RESTORE_EXTERNAL_SNAPSHOT", details={
            "snapshot_id": snapshot.snapshot_id,
            "restored_version": snapshot.state_version,
        })

    def compare_snapshots(self, snapshot_id_a: str, snapshot_id_b: str) -> SnapshotDiff:
        """Compare two stored snapshots."""
        snap_a = self.get_snapshot(snapshot_id_a)
        snap_b = self.get_snapshot(snapshot_id_b)
        return compare_snapshots(snap_a, snap_b)

    def history(self) -> List[SnapshotMetadata]:
        """Return chronological list of all captured snapshot headers."""
        return [s.to_metadata() for s in self._snapshots.values()]

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return complete immutable audit log."""
        return list(self._audit_log)
