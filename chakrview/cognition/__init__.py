"""
ChakrView Cognition Subsystem (Step 15).

Provides governed cognitive agent execution:
Understand -> Retrieve -> Plan -> Execute -> Observe -> Verify -> Recover -> Respond
"""

from chakrview.cognition.task import (
    CognitiveTask,
    TaskStatus,
    TaskConstraints,
    StateTransitionRecord,
    InvalidStateTransitionError,
    VALID_TRANSITIONS,
)
from chakrview.cognition.planner import (
    BoundedPlanner,
    CognitivePlan,
    PlanStep,
    PlanStepStatus,
    RetryPolicy,
    DeterministicRulePlanner,
    PlanningError,
)
from chakrview.cognition.graph import (
    ExecutionGraph,
    CycleDetectedError,
    GraphExecutionError,
)
from chakrview.cognition.observation import (
    StepObservation,
)
from chakrview.cognition.verifier import (
    StepVerifier,
    VerificationResult,
)
from chakrview.cognition.tool_gate import (
    GovernedToolGate,
    ToolAuthorizationError,
    ArgumentValidationError,
)
from chakrview.cognition.skill_selector import (
    CognitiveSkillSelector,
    SkillSelectionMatch,
)
from chakrview.cognition.recovery import (
    RecoveryManager,
    RollbackRecord,
)
from chakrview.cognition.artifacts import (
    CognitiveArtifact,
    ArtifactType,
    ArtifactManager,
)
from chakrview.cognition.trace import (
    ExecutionTrace,
    TraceEntry,
)
from chakrview.cognition.profile import (
    DeploymentProfile,
    get_edge_profile,
    get_desktop_profile,
    get_server_profile,
)
from chakrview.cognition.controller import (
    CognitiveController,
    CognitiveExecutionResult,
    MemoryCandidate,
)
from chakrview.cognition import critical, adaptation, diagnostics
from chakrview.cognition.critical import (
    Hypothesis as CriticalHypothesis,
    Evidence as CriticalEvidence,
    Assumption,
    CounterEvidence,
    AlternativeExplanation,
    Contradiction as CriticalContradiction,
    VerificationResult as CriticalVerificationResult,
    CriticalThinkingTrace,
    CriticalThinkingEngine,
    CriticalThinkingConfig,
)
from chakrview.cognition.adaptation import (
    ResourceProfile,
    HardwareProfileSnapshot,
    HardwareProfiler,
    AdaptiveExecutionPolicy,
)
from chakrview.cognition.diagnostics import (
    CoreIntegrityGuard,
    IntegrityCheckResult,
    SystemDiagnosticsEngine,
    SafeSelfHealingManager,
    DiagnosticStatus,
    DiagnosticReport,
)

__all__ = [
    "CognitiveTask",
    "TaskStatus",
    "TaskConstraints",
    "StateTransitionRecord",
    "InvalidStateTransitionError",
    "VALID_TRANSITIONS",
    "BoundedPlanner",
    "CognitivePlan",
    "PlanStep",
    "PlanStepStatus",
    "RetryPolicy",
    "DeterministicRulePlanner",
    "PlanningError",
    "ExecutionGraph",
    "CycleDetectedError",
    "GraphExecutionError",
    "StepObservation",
    "StepVerifier",
    "VerificationResult",
    "GovernedToolGate",
    "ToolAuthorizationError",
    "ArgumentValidationError",
    "CognitiveSkillSelector",
    "SkillSelectionMatch",
    "RecoveryManager",
    "RollbackRecord",
    "CognitiveArtifact",
    "ArtifactType",
    "ArtifactManager",
    "ExecutionTrace",
    "TraceEntry",
    "DeploymentProfile",
    "get_edge_profile",
    "get_desktop_profile",
    "get_server_profile",
    "CognitiveController",
    "CognitiveExecutionResult",
    "MemoryCandidate",
    "critical",
    "adaptation",
    "diagnostics",
    "CriticalHypothesis",
    "CriticalEvidence",
    "Assumption",
    "CounterEvidence",
    "AlternativeExplanation",
    "CriticalContradiction",
    "CriticalVerificationResult",
    "CriticalThinkingTrace",
    "CriticalThinkingEngine",
    "CriticalThinkingConfig",
    "ResourceProfile",
    "HardwareProfileSnapshot",
    "HardwareProfiler",
    "AdaptiveExecutionPolicy",
    "CoreIntegrityGuard",
    "IntegrityCheckResult",
    "SystemDiagnosticsEngine",
    "SafeSelfHealingManager",
    "DiagnosticStatus",
    "DiagnosticReport",
]

