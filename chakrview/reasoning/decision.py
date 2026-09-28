"""
Decision Layer for ChakrView Reasoning (Step 19).

Provides structured decision candidate generation, risk/utility evaluation,
and selection under the immutable principle:
DATA != AUTHORITY, REASONING != AUTHORITY, CAPABILITY EXISTENCE != AUTHORIZATION.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.reasoning.evidence import EvidenceStore


@dataclass
class DecisionCandidate:
    """
    Candidate action evaluated during reasoning.
    """
    candidate_id: str = field(default_factory=lambda: f"cand_{uuid.uuid4().hex[:8]}")
    action_type: str = "synthesize_answer"  # execute_capability, retrieve_knowledge, synthesize_answer, request_clarification
    description: str = ""
    expected_outcome: str = ""
    supporting_evidence_ids: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    uncertainty_score: float = 0.2
    required_capabilities: List[str] = field(default_factory=list)
    capability_arguments: Dict[str, Any] = field(default_factory=dict)
    utility_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "risks": list(self.risks),
            "constraints": list(self.constraints),
            "uncertainty_score": self.uncertainty_score,
            "required_capabilities": list(self.required_capabilities),
            "capability_arguments": dict(self.capability_arguments),
            "utility_score": self.utility_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionCandidate":
        return cls(**data)


@dataclass
class Decision:
    """
    Formally documented decision selecting an action from candidate options.
    """
    decision_id: str = field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    candidates: List[DecisionCandidate] = field(default_factory=list)
    selected_candidate_id: Optional[str] = None
    rationale: str = ""
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_selected_candidate(self) -> Optional[DecisionCandidate]:
        if not self.selected_candidate_id:
            return None
        for c in self.candidates:
            if c.candidate_id == self.selected_candidate_id:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "task_id": self.task_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        d = dict(data)
        d["candidates"] = [DecisionCandidate.from_dict(c) for c in d.get("candidates", [])]
        return cls(**d)


class DecisionEngine:
    """
    Evaluates candidate actions, scoring expected utility while penalizing risk and uncertainty.
    """

    def evaluate_and_select(
        self,
        task_id: str,
        candidates: List[DecisionCandidate],
        evidence_store: EvidenceStore,
    ) -> Decision:
        """
        Score candidates and select optimal action based on evidence and risk.
        """
        if not candidates:
            return Decision(
                task_id=task_id,
                candidates=[],
                selected_candidate_id=None,
                rationale="No decision candidates provided.",
                confidence=0.0,
            )

        best_candidate: Optional[DecisionCandidate] = None
        best_score = -1e9

        for cand in candidates:
            # Evidence backing score
            evidence_support = 0.0
            for ev_id in cand.supporting_evidence_ids:
                ev = evidence_store.get(ev_id)
                if ev is not None:
                    evidence_support += ev.effective_weight()

            # Net utility calculation
            # Utility = Base Utility + Evidence Support - (Risks + Uncertainty Penalty)
            risk_penalty = 0.25 * len(cand.risks)
            uncertainty_penalty = 0.5 * cand.uncertainty_score
            net_utility = cand.utility_score + 0.3 * evidence_support - risk_penalty - uncertainty_penalty

            cand.utility_score = round(net_utility, 3)

            if net_utility > best_score:
                best_score = net_utility
                best_candidate = cand

        selected_id = best_candidate.candidate_id if best_candidate else candidates[0].candidate_id
        selected_desc = best_candidate.description if best_candidate else "Default action"

        confidence = max(0.1, min(0.95, 0.5 + 0.1 * best_score))

        return Decision(
            task_id=task_id,
            candidates=candidates,
            selected_candidate_id=selected_id,
            rationale=f"Selected candidate '{selected_desc}' with net utility score {best_score:.3f}.",
            confidence=confidence,
        )
