"""
ChakrView Cognitive Identity, Self-Model & System State Subsystem (Step 18).

Provides strongly typed, inspectable, deterministic state containers, epistemic qualification,
explicit uncertainty tracking, immutable snapshots, and multi-tenant state management.
"""

from chakrview.state.identity import (
    SystemIdentity,
    get_current_system_identity,
)
from chakrview.state.epistemic import (
    EpistemicStatus,
    KnowledgeAssertion,
    KnowledgeState,
)
from chakrview.state.uncertainty import (
    Uncertainty,
    UncertaintyState,
)
from chakrview.state.task_state import (
    TaskPhase,
    TaskState,
)
from chakrview.state.environment_state import (
    OperationalMode,
    DeviceConnectionStatus,
    EnvironmentState,
)
from chakrview.state.capability_state import (
    ObservedCapabilityStatus,
    CapabilityObservation,
    CapabilityState,
)
from chakrview.state.constraints import (
    PolicyRestriction,
    ConstraintState,
)
from chakrview.state.snapshot import (
    SnapshotMetadata,
    CognitiveStateSnapshot,
    SnapshotDiff,
    compare_snapshots,
)
from chakrview.state.manager import (
    CognitiveStateManager,
    StateIsolationError,
    StateValidationError,
    SnapshotNotFoundError,
)

__all__ = [
    # Identity
    "SystemIdentity",
    "get_current_system_identity",
    # Epistemic & Knowledge
    "EpistemicStatus",
    "KnowledgeAssertion",
    "KnowledgeState",
    # Uncertainty
    "Uncertainty",
    "UncertaintyState",
    # Task State
    "TaskPhase",
    "TaskState",
    # Environment State
    "OperationalMode",
    "DeviceConnectionStatus",
    "EnvironmentState",
    # Capability State
    "ObservedCapabilityStatus",
    "CapabilityObservation",
    "CapabilityState",
    # Constraints
    "PolicyRestriction",
    "ConstraintState",
    # Snapshot & Diff
    "SnapshotMetadata",
    "CognitiveStateSnapshot",
    "SnapshotDiff",
    "compare_snapshots",
    # Manager & Errors
    "CognitiveStateManager",
    "StateIsolationError",
    "StateValidationError",
    "SnapshotNotFoundError",
]
