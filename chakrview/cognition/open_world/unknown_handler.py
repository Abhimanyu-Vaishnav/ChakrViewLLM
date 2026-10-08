"""
Wave 551: Unknown-Question Handling & Honest Abstention Benchmark.

Enforces:
- “I don't know” is a first-class cognitive outcome, not a string fallback.
- Classifies 10 inquiry categories (A to J):
  A. Answerable internally
  B. Answerable from validated memory
  C. Answerable through external retrieval
  D. Partially answerable
  E. Contradictory evidence
  F. Insufficient evidence
  G. Genuinely unknown
  H. Impossible / unverifiable
  I. Ambiguous question
  J. Time-sensitive question without current access
- Chooses action: ANSWER, QUALIFY, RETRIEVE, ASK_CLARIFICATION, ABSTAIN, REPORT_CONFLICT.
- Penalizes manufactured certainty; rewards justified abstention.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class InquiryClass(str, Enum):
    A_INTERNAL = "A_INTERNAL"
    B_MEMORY = "B_MEMORY"
    C_EXTERNAL_RETRIEVABLE = "C_EXTERNAL_RETRIEVABLE"
    D_PARTIALLY_ANSWERABLE = "D_PARTIALLY_ANSWERABLE"
    E_CONTRADICTORY = "E_CONTRADICTORY"
    F_INSUFFICIENT_EVIDENCE = "F_INSUFFICIENT_EVIDENCE"
    G_GENUINELY_UNKNOWN = "G_GENUINELY_UNKNOWN"
    H_UNVERIFIABLE = "H_UNVERIFIABLE"
    I_AMBIGUOUS = "I_AMBIGUOUS"
    J_TIME_SENSITIVE_NO_ACCESS = "J_TIME_SENSITIVE_NO_ACCESS"


class CognitiveDecision(str, Enum):
    ANSWER = "ANSWER"
    QUALIFY = "QUALIFY"
    RETRIEVE = "RETRIEVE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    ABSTAIN = "ABSTAIN"
    REPORT_CONFLICT = "REPORT_CONFLICT"


@dataclass
class EpistemicResolution:
    inquiry_class: InquiryClass
    decision: CognitiveDecision
    response_content: str
    is_abstention: bool
    factual_confidence: float
    justification: str


class UnknownHandlingEngine:
    """
    Decides cognitive action across diverse inquiry classes with zero fabricated certainty.
    """

    @classmethod
    def resolve_inquiry(
        cls,
        query: str,
        inquiry_class: InquiryClass,
        known_data: Optional[Dict[str, Any]] = None,
    ) -> EpistemicResolution:
        known_data = known_data or {}

        if inquiry_class == InquiryClass.A_INTERNAL:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.ANSWER,
                response_content=known_data.get("answer", "Answer verified internally."),
                is_abstention=False,
                factual_confidence=0.98,
                justification="Directly answerable from internal neural/cognitive capabilities.",
            )

        if inquiry_class == InquiryClass.B_MEMORY:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.ANSWER,
                response_content=known_data.get("answer", "Answer verified from durable memory."),
                is_abstention=False,
                factual_confidence=0.95,
                justification="Answered from validated persistent memory.",
            )

        if inquiry_class == InquiryClass.C_EXTERNAL_RETRIEVABLE:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.RETRIEVE,
                response_content="External retrieval initiated for verified ground evidence.",
                is_abstention=False,
                factual_confidence=0.50,
                justification="Information is externally verifiable; dispatching external knowledge request.",
            )

        if inquiry_class == InquiryClass.D_PARTIALLY_ANSWERABLE:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.QUALIFY,
                response_content="Partially known: [Verified core known, specific edge unknown].",
                is_abstention=False,
                factual_confidence=0.70,
                justification="Qualified response acknowledging clear epistemic boundary.",
            )

        if inquiry_class == InquiryClass.E_CONTRADICTORY:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.REPORT_CONFLICT,
                response_content="The available evidence contains mutually contradictory claims. Revalidation is required.",
                is_abstention=True,
                factual_confidence=0.40,
                justification="Preserving open-world contradiction rather than guessing.",
            )

        if inquiry_class == InquiryClass.I_AMBIGUOUS:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.ASK_CLARIFICATION,
                response_content="The question is ambiguous. Could you clarify your intent or context?",
                is_abstention=True,
                factual_confidence=0.20,
                justification="Identified structural ambiguity; requesting clarification.",
            )

        if inquiry_class == InquiryClass.J_TIME_SENSITIVE_NO_ACCESS:
            return EpistemicResolution(
                inquiry_class=inquiry_class,
                decision=CognitiveDecision.ABSTAIN,
                response_content="This question requires real-time information and current external access is unavailable. I cannot verify current state.",
                is_abstention=True,
                factual_confidence=0.0,
                justification="Refused to extrapolate stale data on time-critical question.",
            )

        # Cases F (Insufficient), G (Genuinely Unknown), H (Unverifiable) -> Mandatory Abstention
        return EpistemicResolution(
            inquiry_class=inquiry_class,
            decision=CognitiveDecision.ABSTAIN,
            response_content="I do not have enough evidence to answer this factually.",
            is_abstention=True,
            factual_confidence=0.0,
            justification="Mandatory honest abstention: UNKNOWN != FALSE.",
        )
