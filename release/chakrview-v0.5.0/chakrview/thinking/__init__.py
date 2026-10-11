"""
ChakrView Thinking & Deliberation Foundation (Step 21).

Provides bounded, inspectable neuro-symbolic deliberation:
- Structured Thought representation (ThoughtStep, ThoughtPurpose)
- Bounded ThinkingWorkspace enforcing operational limits & tenant isolation
- Transparent heuristic attention (ThinkingAttention, FocusType)
- Structured CritiqueEngine (CritiqueResult, CritiqueVerdict)
- Multi-cycle RevisionEngine (RevisionPlan)
- Deterministic ThinkingStoppingPolicy (StoppingCondition)
- Auditable telemetry (ThinkingTrace)
- Orchestration layer (DeliberationEngine, DeliberationOutcome)
"""

from chakrview.thinking.thought import ThoughtPurpose, ThoughtStep
from chakrview.thinking.policy import (
    ThinkingPolicy,
    get_standard_policy,
    get_strict_policy,
    get_fast_policy,
)
from chakrview.thinking.workspace import (
    ThinkingWorkspace,
    WorkspaceBudgetExceededError,
    TenantIsolationError,
)
from chakrview.thinking.attention import (
    FocusType,
    AttentionFocus,
    ThinkingAttention,
)
from chakrview.thinking.critique import (
    CritiqueVerdict,
    CritiqueResult,
    CritiqueEngine,
)
from chakrview.thinking.revision import (
    RevisionPlan,
    RevisionEngine,
)
from chakrview.thinking.stopping import (
    StoppingCondition,
    ThinkingStoppingPolicy,
)
from chakrview.thinking.trace import (
    ThinkingTrace,
)
from chakrview.thinking.deliberation import (
    DeliberationOutcome,
    DeliberationEngine,
)

__all__ = [
    # Thought
    "ThoughtPurpose",
    "ThoughtStep",
    # Policy
    "ThinkingPolicy",
    "get_standard_policy",
    "get_strict_policy",
    "get_fast_policy",
    # Workspace
    "ThinkingWorkspace",
    "WorkspaceBudgetExceededError",
    "TenantIsolationError",
    # Attention
    "FocusType",
    "AttentionFocus",
    "ThinkingAttention",
    # Critique
    "CritiqueVerdict",
    "CritiqueResult",
    "CritiqueEngine",
    # Revision
    "RevisionPlan",
    "RevisionEngine",
    # Stopping
    "StoppingCondition",
    "ThinkingStoppingPolicy",
    # Trace
    "ThinkingTrace",
    # Deliberation
    "DeliberationOutcome",
    "DeliberationEngine",
]
