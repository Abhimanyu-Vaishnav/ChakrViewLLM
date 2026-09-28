"""
Controlled Learning Feedback & Self-Improvement Foundation for ChakrView (Step 16).

Tracks:
Task -> Result -> Verification -> Feedback -> Improvement Signal -> Learning Candidate

Enforces the absolute architectural boundary:
USER DATA AND FEEDBACK DO NOT DIRECTLY MODIFY NEURAL MODEL WEIGHTS.
All learning signals are recorded as governed candidates for review and evaluation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid


class FeedbackSignal(str, Enum):
    """User or system feedback polarity."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    NEUTRAL = "neutral"


class LearningCategory(str, Enum):
    """Domain of proposed system improvement."""
    RETRIEVAL_IMPROVEMENT = "retrieval_improvement"
    SKILL_IMPROVEMENT = "skill_improvement"
    PLANNING_IMPROVEMENT = "planning_improvement"
    MEMORY_IMPROVEMENT = "memory_improvement"
    PERSONALIZATION_IMPROVEMENT = "personalization_improvement"


@dataclass
class LearningCandidate:
    """
    A proposed system improvement signal captured from task outcome or user feedback.

    Attributes:
        candidate_id: Unique candidate identifier.
        category: Subsystem targeted for improvement.
        signal: Polarity of feedback.
        task_id: ID of the cognitive task that generated the signal.
        observed_outcome: What the agent actually produced.
        feedback_text: Ground truth or user correction.
        proposed_adjustment: Suggested configuration or heuristic change.
        confidence: Assessment confidence score in [0.0, 1.0].
        created_at: Epoch timestamp of creation.
        status: Governance lifecycle ("PROPOSED", "REVIEWED", "APPROVED", "REJECTED").
        metadata: Diagnostic and provenance attributes.
    """
    candidate_id: str
    category: LearningCategory
    signal: FeedbackSignal
    observed_outcome: str
    feedback_text: str
    proposed_adjustment: str
    task_id: Optional[str] = None
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)
    status: str = "PROPOSED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "category": self.category.value,
            "signal": self.signal.value,
            "observed_outcome": self.observed_outcome,
            "feedback_text": self.feedback_text,
            "proposed_adjustment": self.proposed_adjustment,
            "task_id": self.task_id,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningCandidate":
        d = dict(data)
        d["category"] = LearningCategory(d["category"])
        d["signal"] = FeedbackSignal(d["signal"])
        return cls(**d)


class LearningFeedbackManager:
    """
    Manages collection, classification, and lifecycle of learning candidates.
    """

    def __init__(self) -> None:
        self._candidates: Dict[str, LearningCandidate] = {}

    def record_feedback(
        self,
        category: LearningCategory,
        signal: FeedbackSignal,
        observed_outcome: str,
        feedback_text: str,
        proposed_adjustment: str = "",
        task_id: Optional[str] = None,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningCandidate:
        """
        Record a governed learning candidate.
        """
        cid = f"learn_{uuid.uuid4().hex[:8]}"
        candidate = LearningCandidate(
            candidate_id=cid,
            category=category,
            signal=signal,
            observed_outcome=observed_outcome,
            feedback_text=feedback_text,
            proposed_adjustment=proposed_adjustment or f"Adjust heuristics for {category.value}",
            task_id=task_id,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._candidates[cid] = candidate
        return candidate

    def get_candidate(self, candidate_id: str) -> Optional[LearningCandidate]:
        """Look up candidate by ID."""
        return self._candidates.get(candidate_id)

    def list_candidates(
        self,
        category: Optional[LearningCategory] = None,
        status: Optional[str] = None,
    ) -> List[LearningCandidate]:
        """List learning candidates with optional filtering."""
        results = list(self._candidates.values())
        if category is not None:
            results = [c for c in results if c.category == category]
        if status is not None:
            results = [c for c in results if c.status == status]
        results.sort(key=lambda c: c.created_at, reverse=True)
        return results

    def clear(self) -> None:
        """Clear recorded candidates."""
        self._candidates.clear()
