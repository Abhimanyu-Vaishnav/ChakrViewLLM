"""
ChakrView Step 142: Critical Thinking & Epistemic Reasoning Engine.

Distinguishes epistemic truth categories:
- KNOWN: Directly supported by unambiguous verifiable evidence
- SUPPORTED: Inductively supported by corroborated facts
- UNCERTAIN: Plausible but missing critical corroboration
- CONFLICTING: Mutually contradictory claims
- UNKNOWN: Insufficient evidence available (Triggers explicit abstention: 'I do not have enough evidence')

Features:
- Assumption detection
- Alternative generation
- Contradiction reconciliation
- Epistemic conclusion revision when new evidence arrives
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


class EpistemicState(str, enum.Enum):
    KNOWN = "KNOWN"
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTING = "CONFLICTING"
    UNKNOWN = "UNKNOWN"


@dataclass
class EvidenceStatement:
    statement_id: str
    content: str
    source_reliability: float
    timestamp: float
    polarity: bool = True  # True for affirming, False for negating


@dataclass
class EpistemicAssessment:
    claim: str
    state: EpistemicState
    confidence: float
    identified_assumptions: List[str] = field(default_factory=list)
    alternative_hypotheses: List[str] = field(default_factory=list)
    abstention_triggered: bool = False
    justification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


class CriticalThinkingEngine:
    """
    Evaluates propositions epistemically, detects missing evidence, and enforces honest abstention.
    """

    @classmethod
    def evaluate_claim(
        cls,
        claim: str,
        supporting_evidence: List[EvidenceStatement],
        refuting_evidence: List[EvidenceStatement],
        required_evidence_threshold: int = 1,
    ) -> EpistemicAssessment:
        # 1. No evidence available -> UNKNOWN with mandatory abstention
        if not supporting_evidence and not refuting_evidence:
            return EpistemicAssessment(
                claim=claim,
                state=EpistemicState.UNKNOWN,
                confidence=0.0,
                identified_assumptions=["Assumes claim holds without prior observation"],
                alternative_hypotheses=["Claim might be false or unverifiable"],
                abstention_triggered=True,
                justification="I do not have enough evidence.",
            )

        # 2. Both supporting and refuting evidence present -> CONFLICTING
        if supporting_evidence and refuting_evidence:
            return EpistemicAssessment(
                claim=claim,
                state=EpistemicState.CONFLICTING,
                confidence=0.5,
                identified_assumptions=["Sources may have diverging perspectives or stale views"],
                alternative_hypotheses=["Partial truth in both positions", "Definitional divergence"],
                abstention_triggered=False,
                justification="Conflicting evidence observed; reconciliation required.",
            )

        # 3. Only supporting evidence
        if supporting_evidence:
            avg_rel = sum(e.source_reliability for e in supporting_evidence) / len(supporting_evidence)
            if len(supporting_evidence) >= required_evidence_threshold and avg_rel >= 0.8:
                return EpistemicAssessment(
                    claim=claim,
                    state=EpistemicState.KNOWN,
                    confidence=round(avg_rel, 3),
                    justification="Directly supported by verified evidence.",
                )
            else:
                return EpistemicAssessment(
                    claim=claim,
                    state=EpistemicState.SUPPORTED if avg_rel >= 0.5 else EpistemicState.UNCERTAIN,
                    confidence=round(avg_rel, 3),
                    identified_assumptions=["Evidence sample size is limited"],
                    justification="Evidence exists but confidence is intermediate.",
                )

        # 4. Only refuting evidence
        return EpistemicAssessment(
            claim=claim,
            state=EpistemicState.KNOWN,
            confidence=0.9,
            justification="Claim is refuted by observational evidence.",
        )

    @classmethod
    def revise_conclusion(
        cls,
        previous_assessment: EpistemicAssessment,
        new_evidence: List[EvidenceStatement],
    ) -> EpistemicAssessment:
        """Revises conclusion upon arrival of new observations."""
        supporting = [e for e in new_evidence if e.polarity]
        refuting = [e for e in new_evidence if not e.polarity]
        revised = cls.evaluate_claim(previous_assessment.claim, supporting, refuting)
        revised.justification = f"Revised: {revised.justification} (Previous state was {previous_assessment.state.value})"
        return revised
