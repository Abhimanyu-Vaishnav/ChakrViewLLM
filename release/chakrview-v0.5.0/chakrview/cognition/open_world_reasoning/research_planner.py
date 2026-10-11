"""
Wave 579: Open-World Research Planning & Unknown Handling Engine.

Architectural Guarantees:
- Structured Knowledge States:
  KNOWN, PARTIALLY_KNOWN, UNKNOWN, CONTESTED, INSUFFICIENT_EVIDENCE, AMBIGUOUS.
- Formulates an actionable research plan when information is incomplete:
  QUESTION -> KNOWN FACTS -> UNKNOWN FACTS -> REQUIRED EVIDENCE -> SEARCH TARGET -> VERIFICATION CRITERIA -> STOP CONDITION.
- Never treats the plan itself as completed evidence.
- Zero premature guessing or fabricated truth.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class EpistemicKnowledgeState(str, Enum):
    KNOWN = "KNOWN"
    PARTIALLY_KNOWN = "PARTIALLY_KNOWN"
    UNKNOWN = "UNKNOWN"
    CONTESTED = "CONTESTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class ResearchRequirement:
    target_question: str
    state: EpistemicKnowledgeState
    known_facts: List[str]
    unknown_facts: List[str]
    required_evidence_spans: List[str]
    search_targets: List[str]
    verification_criteria: List[str]
    stop_condition: str
    confidence: float
    is_ready_for_execution: bool


class OpenWorldResearchPlanner:
    """
    Formulates structured research requirements and verification plans
    when cognitive state is UNKNOWN or PARTIALLY_KNOWN.
    """

    @classmethod
    def plan_research(
        cls,
        question: str,
        target_entities: List[str],
        known_facts: List[str],
        expected_premises: List[str],
        is_contested: bool = False,
        is_ambiguous: bool = False,
    ) -> ResearchRequirement:
        known_clean = [f.strip() for f in known_facts if f.strip()]
        missing_premises = [p for p in expected_premises if p not in known_clean]

        # 1. Ambiguous query state
        if is_ambiguous:
            return ResearchRequirement(
                target_question=question,
                state=EpistemicKnowledgeState.AMBIGUOUS,
                known_facts=known_clean,
                unknown_facts=["Disambiguation context for target entities"],
                required_evidence_spans=["Contextual entity descriptors"],
                search_targets=[f"disambiguation for {e}" for e in target_entities],
                verification_criteria=["Unique entity match score >= 0.8"],
                stop_condition="Entity identity uniquely resolved or clarification prompted.",
                confidence=0.20,
                is_ready_for_execution=True,
            )

        # 2. Contested / Conflicting state
        if is_contested:
            return ResearchRequirement(
                target_question=question,
                state=EpistemicKnowledgeState.CONTESTED,
                known_facts=known_clean,
                unknown_facts=["Resolution of competing claims"],
                required_evidence_spans=["Primary authoritative corroboration"],
                search_targets=[f"authoritative source for {question}"],
                verification_criteria=["Independent primary publisher consensus >= 2"],
                stop_condition="Conflict classified as temporal change or verified consensus.",
                confidence=0.35,
                is_ready_for_execution=True,
            )

        # 3. Fully Known
        if not missing_premises and known_clean:
            return ResearchRequirement(
                target_question=question,
                state=EpistemicKnowledgeState.KNOWN,
                known_facts=known_clean,
                unknown_facts=[],
                required_evidence_spans=[],
                search_targets=[],
                verification_criteria=["Internal provenance check"],
                stop_condition="Complete premise coverage achieved.",
                confidence=0.95,
                is_ready_for_execution=False,
            )

        # 4. Partially Known
        if known_clean and missing_premises:
            return ResearchRequirement(
                target_question=question,
                state=EpistemicKnowledgeState.PARTIALLY_KNOWN,
                known_facts=known_clean,
                unknown_facts=missing_premises,
                required_evidence_spans=[f"Proof for {p}" for p in missing_premises],
                search_targets=[f"query: {p}" for p in missing_premises],
                verification_criteria=["Exact entity-relation match with valid timestamp"],
                stop_condition=f"All {len(missing_premises)} missing premises verified.",
                confidence=0.60,
                is_ready_for_execution=True,
            )

        # 5. Genuinely Unknown / Insufficient Evidence
        return ResearchRequirement(
            target_question=question,
            state=EpistemicKnowledgeState.UNKNOWN,
            known_facts=[],
            unknown_facts=expected_premises or [question],
            required_evidence_spans=[f"Ground evidence for {p}" for p in (expected_premises or [question])],
            search_targets=[f"search for {e}" for e in target_entities] or [f"search for {question}"],
            verification_criteria=["Multi-source independent verification"],
            stop_condition="Target evidence gathered or budget exhausted.",
            confidence=0.0,
            is_ready_for_execution=True,
        )
