"""
Unified Cognitive Architecture Subsystem for ChakrView (Step 25).

Provides the complete 14-stage cognitive lifecycle connecting:
- Cognitive Task Classification & Decision Layer
- Bounded Unified Cognitive State & Context Compression
- Step 24 Continual Memory (Episodic, Semantic, Contradiction)
- Step 19 Governed Cognitive Reasoning
- Step 23 Critical Thinking & Anti-Confirmation-Bias
- Step 21 Neural Thinking & Deliberation
- Step 17 CapabilityGate & Authority Protection
- Step 23 Hardware Adaptation & Self-Diagnostics
- Safe Public Cognitive Tracing
- Governed Experience Capture & Step 22 Training Bridge
"""

from chakrview.cognition.unified.models import (
    CognitiveTaskType,
    DecisionState,
    UnifiedCognitiveState,
    SafePublicCognitiveTrace,
)
from chakrview.cognition.unified.policy import (
    UnifiedCognitivePolicy,
    HARD_CEILING_CONTEXT_TOKENS,
    HARD_CEILING_GENERATION_TOKENS,
    HARD_CEILING_THINKING_STEPS,
    HARD_CEILING_REVISIONS,
    HARD_CEILING_MEMORY_TOP_K,
    HARD_CEILING_REASONING_ITERATIONS,
)
from chakrview.cognition.unified.context import (
    CognitiveContextCompressor,
)
from chakrview.cognition.unified.decision import (
    CognitiveDecisionLayer,
)
from chakrview.cognition.unified.experience import (
    GovernedExperienceRecord,
    GovernedExperienceCapture,
)
from chakrview.cognition.unified.trace import (
    PublicTraceBuilder,
)
from chakrview.cognition.unified.engine import (
    UnifiedCognitiveEngine,
)

__all__ = [
    "CognitiveTaskType",
    "DecisionState",
    "UnifiedCognitiveState",
    "SafePublicCognitiveTrace",
    "UnifiedCognitivePolicy",
    "HARD_CEILING_CONTEXT_TOKENS",
    "HARD_CEILING_GENERATION_TOKENS",
    "HARD_CEILING_THINKING_STEPS",
    "HARD_CEILING_REVISIONS",
    "HARD_CEILING_MEMORY_TOP_K",
    "HARD_CEILING_REASONING_ITERATIONS",
    "CognitiveContextCompressor",
    "CognitiveDecisionLayer",
    "GovernedExperienceRecord",
    "GovernedExperienceCapture",
    "PublicTraceBuilder",
    "UnifiedCognitiveEngine",
]
