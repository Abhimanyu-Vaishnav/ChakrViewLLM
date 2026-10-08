"""
Wave 563: Final Answer Claim-Level Attribution Engine.

Enforces:
- Every substantive factual claim in a generated response must trace to supporting evidence.
- Distinguishes:
  DIRECTLY_SUPPORTED (direct factual datum)
  INFERRED_FROM_SUPPORTED_FACTS (deductive link across evidence)
  PARTIALLY_SUPPORTED
  UNSUPPORTED (hallucination / ungrounded claim)
  CONTRADICTED
- Inferences are strictly labeled as INFERENCES, never masquerading as direct quotations/facts.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple

from .evidence_evaluation import ClaimSupportType, GroundedClaim


@dataclass
class ClaimAttributionNode:
    claim_id: str
    text_span: str
    target_entity: str
    support_type: ClaimSupportType
    cited_evidence_ids: List[str]
    is_inference: bool
    is_grounded: bool
    justification: str


@dataclass
class AttributedClaimGraph:
    response_id: str
    full_response_text: str
    claims: List[ClaimAttributionNode]
    total_claims: int
    directly_supported_count: int
    inferred_count: int
    unsupported_count: int
    contradicted_count: int
    attribution_precision: float  # (supported + inferred) / total
    hallucination_rate: float      # unsupported / total
    overall_grounded: bool


class FinalAnswerAttributionEngine:
    """
    Parses substantive claims from final answer output and constructs a verified claim-evidence graph.
    """

    @classmethod
    def audit_response_attribution(
        cls,
        response_id: str,
        response_text: str,
        claims: List[GroundedClaim],
        available_evidence: Dict[str, Dict[str, Any]],
    ) -> AttributedClaimGraph:
        nodes: List[ClaimAttributionNode] = []

        direct_cnt = 0
        inferred_cnt = 0
        unsupported_cnt = 0
        contradicted_cnt = 0

        for c in claims:
            # Audit evidence IDs cited
            valid_cited = [
                ev_id for ev_id in c.supporting_evidence_ids
                if ev_id in available_evidence
            ]

            # Determine grounding status
            if c.support_type == ClaimSupportType.CONTRADICTED:
                contradicted_cnt += 1
                is_grounded = False
                justification = "Claim is contradicted by verified evidence."
            elif not valid_cited:
                c.support_type = ClaimSupportType.UNSUPPORTED
                unsupported_cnt += 1
                is_grounded = False
                justification = "Zero valid evidence cited; ungrounded claim."
            elif c.support_type == ClaimSupportType.INFERRED_FROM_SUPPORTED_FACTS:
                inferred_cnt += 1
                is_grounded = True
                justification = "Deduced soundly from combination of verified premises."
            elif c.support_type == ClaimSupportType.DIRECTLY_SUPPORTED:
                direct_cnt += 1
                is_grounded = True
                justification = f"Directly grounded by evidence {valid_cited}."
            else:
                unsupported_cnt += 1
                is_grounded = False
                justification = "Insufficient evidence coverage."

            node = ClaimAttributionNode(
                claim_id=c.claim_id,
                text_span=c.statement,
                target_entity=c.target_entity,
                support_type=c.support_type,
                cited_evidence_ids=valid_cited,
                is_inference=(c.support_type == ClaimSupportType.INFERRED_FROM_SUPPORTED_FACTS),
                is_grounded=is_grounded,
                justification=justification,
            )
            nodes.append(node)

        total = max(1, len(nodes))
        precision = (direct_cnt + inferred_cnt) / total
        hallucination_rate = unsupported_cnt / total

        return AttributedClaimGraph(
            response_id=response_id,
            full_response_text=response_text,
            claims=nodes,
            total_claims=len(nodes),
            directly_supported_count=direct_cnt,
            inferred_count=inferred_cnt,
            unsupported_count=unsupported_cnt,
            contradicted_count=contradicted_cnt,
            attribution_precision=precision,
            hallucination_rate=hallucination_rate,
            overall_grounded=(unsupported_cnt == 0 and contradicted_cnt == 0),
        )
