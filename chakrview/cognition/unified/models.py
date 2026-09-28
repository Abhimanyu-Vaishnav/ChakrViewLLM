"""
Unified Cognitive Architecture Data Models for ChakrView (Step 25).

Defines strongly typed, serializable structures for:
- Cognitive Task Classification (CognitiveTaskType)
- Bounded Decision States (DecisionState)
- Unified Cognitive State / Context Fusion (UnifiedCognitiveState)
- Safe Public Cognitive Trace (SafePublicCognitiveTrace)

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. DATA != AUTHORITY, MEMORY != AUTHORITY, REASONING != AUTHORITY,
   THINKING != AUTHORITY, CRITICAL THINKING != AUTHORITY, EXPERIENCE != AUTHORITY.
2. ZERO RUNTIME WEIGHT MUTATION: weights_modified is permanently False at runtime.
3. Model invariants remain strictly intact: 3,443,136 params, 4096 vocab, 512 context, BOS=0, EOS=1, PAD=2.
4. Private chain-of-thought is never exposed across public trace boundaries.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Set
import uuid


class CognitiveTaskType(str, Enum):
    """Classification of cognitive task objectives."""
    FACTUAL = "FACTUAL"                      # Direct information lookup / declarative recall
    ANALYTICAL = "ANALYTICAL"                # Multi-step decomposition, mathematical or logical analysis
    DECISION = "DECISION"                    # Evaluation of options against criteria and constraints
    CAPABILITY = "CAPABILITY"                # Requires external device / tool invocation via CapabilityGate
    EXPLORATORY = "EXPLORATORY"              # Open-ended hypothesis formulation under uncertainty
    GENERAL = "GENERAL"                      # General conversational or conversational reasoning task


class DecisionState(str, Enum):
    """
    Formal, bounded cognitive decision states.
    Prevents unconstrained autonomous drift by requiring explicit cognitive status.
    """
    ANSWER = "ANSWER"                                     # High confidence, verified, safe answer produced
    ANSWER_WITH_UNCERTAINTY = "ANSWER_WITH_UNCERTAINTY"   # Answer produced with explicit epistemic caveats
    NEED_CLARIFICATION = "NEED_CLARIFICATION"             # Input underspecified or ambiguous
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION" # Missing crucial evidence; refused fabrication
    REQUIRE_VERIFICATION = "REQUIRE_VERIFICATION"         # High-risk claim requires external validation
    REVISION_REQUIRED = "REVISION_REQUIRED"               # Intermediate critique failed; revision scheduled
    CAPABILITY_REQUIRED = "CAPABILITY_REQUIRED"           # External capability needed to fulfill objective
    SAFE_STOP = "SAFE_STOP"                               # Stopped due to invariant, policy, or safety hazard


@dataclass
class UnifiedCognitiveState:
    """
    Bounded, fused cognitive state representing the entire lifecycle of a cognitive cycle.
    """
    cycle_id: str
    tenant_id: str
    session_id: str
    user_prompt: str
    task_type: CognitiveTaskType = CognitiveTaskType.GENERAL
    active_objective: str = ""
    working_memory_snapshot: Optional[Dict[str, Any]] = None
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    counter_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_summary: Optional[str] = None
    critical_thinking_summary: Optional[str] = None
    deliberation_summary: Optional[str] = None
    decision_state: DecisionState = DecisionState.INSUFFICIENT_INFORMATION
    confidence: float = 0.5
    uncertainty_notes: Optional[str] = None
    revision_count: int = 0
    capability_requests: List[Dict[str, Any]] = field(default_factory=list)
    capability_results: List[Dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    experience_record: Optional[Dict[str, Any]] = None
    learning_candidate_id: Optional[str] = None
    weights_modified: bool = False  # Permanent core invariant: ALWAYS False
    truncated_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.tenant_id:
            raise ValueError("tenant_id cannot be empty.")
        if not self.session_id:
            raise ValueError("session_id cannot be empty.")

    def mark_completed(self, response: str, decision: DecisionState, confidence: float) -> None:
        """Mark cycle as completed with final response and decision."""
        self.final_response = response
        self.decision_state = decision
        self.confidence = max(0.0, min(1.0, confidence))
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cognitive state snapshot."""
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["decision_state"] = self.decision_state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedCognitiveState":
        d = dict(data)
        if "task_type" in d:
            d["task_type"] = CognitiveTaskType(d["task_type"])
        if "decision_state" in d:
            d["decision_state"] = DecisionState(d["decision_state"])
        return cls(**d)


@dataclass
class SafePublicCognitiveTrace:
    """
    Public, auditable cognitive trace.
    GUARANTEE: Does NOT expose internal chain-of-thought, intermediate private thoughts,
    or raw token scratchpads.
    """
    cycle_id: str
    tenant_id: str
    session_id: str
    task_type: str
    decision_state: str
    confidence: float
    retrieved_memory_count: int
    evidence_count: int
    counter_evidence_count: int
    contradiction_count: int
    revision_count: int
    uncertainty_acknowledged: bool
    capability_requested: bool
    capability_authorized: bool
    hardware_profile: str
    execution_time_ms: float
    weights_modified: bool = False  # Always False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
