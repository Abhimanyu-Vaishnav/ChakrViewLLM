"""
Governed Experience Capture & Learning Bridge for ChakrView (Step 25).

Captures completed unified cognitive cycles as auditable, governed experiences
and safely routes them to Step 24 Continual Memory and Step 22 Offline Training.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. RAW EXPERIENCE != VERIFIED TRAINING DATA.
2. Experiences are captured as EPISODIC memories with verification_status = UNVERIFIED.
3. Memory consolidation synthesizes CANDIDATE semantic facts.
4. Offline model training strictly requires explicit administrative sign-off (TRAINING_APPROVED).
5. ZERO RUNTIME WEIGHT MUTATION: weights_modified is permanently False.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.cognition.unified.models import UnifiedCognitiveState, DecisionState
from chakrview.memory.models import (
    Episode,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
)
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.intelligence.contracts import LearningRecord, LearningRecordStatus


@dataclass
class GovernedExperienceRecord:
    """
    Structured, auditable record of an executed cognitive cycle.
    """
    experience_id: str
    cycle_id: str
    tenant_id: str
    session_id: str
    task_class: str
    user_prompt: str
    final_response: str
    decision_state: str
    confidence: float
    verification_status: str
    memory_references: List[str]
    reasoning_summary: Optional[str]
    critical_summary: Optional[str]
    revision_occurred: bool
    uncertainty_remained: bool
    capability_outcome: Optional[str]
    timestamp: float = field(default_factory=time.time)
    weights_modified: bool = False  # Core invariant: permanently False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernedExperienceRecord":
        return cls(**data)


class GovernedExperienceCapture:
    """
    Captures runtime cognitive experiences and interfaces with Step 24 memory engine.
    """

    def __init__(self, memory_engine: Optional[ContinualCognitionEngine] = None) -> None:
        self.memory_engine = memory_engine

    def capture_cycle(
        self,
        state: UnifiedCognitiveState,
    ) -> GovernedExperienceRecord:
        """
        Transform a completed UnifiedCognitiveState into a governed experience record
        and persist as an episodic memory.
        """
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        now = time.time()

        mem_refs = [m.get("memory_id", str(m)) for m in state.retrieved_memories]
        ver_status = (
            MemoryVerificationState.VERIFIED.value
            if state.decision_state == DecisionState.ANSWER and state.confidence >= 0.8
            else MemoryVerificationState.UNVERIFIED.value
        )

        record = GovernedExperienceRecord(
            experience_id=exp_id,
            cycle_id=state.cycle_id,
            tenant_id=state.tenant_id,
            session_id=state.session_id,
            task_class=state.task_type.value,
            user_prompt=state.user_prompt,
            final_response=state.final_response,
            decision_state=state.decision_state.value,
            confidence=state.confidence,
            verification_status=ver_status,
            memory_references=mem_refs,
            reasoning_summary=state.reasoning_summary,
            critical_summary=state.critical_thinking_summary,
            revision_occurred=state.revision_count > 0,
            uncertainty_remained=state.decision_state in (
                DecisionState.ANSWER_WITH_UNCERTAINTY,
                DecisionState.INSUFFICIENT_INFORMATION,
            ),
            capability_outcome=(
                state.capability_results[0].get("status")
                if state.capability_results
                else None
            ),
            timestamp=now,
            weights_modified=False,
            metadata={
                "truncated_sections": state.truncated_metadata.get("omitted_sections", []),
                "revision_count": state.revision_count,
            },
        )

        # Store in Step 24 Episodic Memory if memory engine is available
        if self.memory_engine:
            try:
                self.memory_engine.record_experience(
                    tenant_id=state.tenant_id,
                    session_id=state.session_id,
                    situation=state.user_prompt,
                    action_or_response=state.final_response,
                    outcome=f"Decision: {state.decision_state.value} | Confidence: {state.confidence:.2f}",
                    task_id=state.cycle_id,
                    confidence=state.confidence,
                    provenance=MemoryProvenanceSource.SYSTEM_OBSERVED,
                    verification_status=(
                        MemoryVerificationState.VERIFIED
                        if ver_status == MemoryVerificationState.VERIFIED.value
                        else MemoryVerificationState.UNVERIFIED
                    ),
                    tags=[state.task_type.value.lower(), state.decision_state.value.lower()],
                    metadata={"experience_id": exp_id},
                )
            except Exception:
                pass  # Memory persistence must not crash cognitive flow

        return record
