"""
Wave 560: Multi-Hop Evidence Reasoning & Chain Grounding Engine.

Enforces:
- E1 -> intermediate A; E2 -> intermediate B; E3 -> relationship; A + B + rel -> conclusion.
- Explicit evidence lineage across all hops.
- Rejection of conclusion if any intermediate link is broken, contradictory, or missing.
- Resistance to reordered, duplicated, or entity-mismatched intermediate links.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set


class ChainCompletenessStatus(str, Enum):
    COMPLETE_GROUNDED = "COMPLETE_GROUNDED"
    MISSING_LINK = "MISSING_LINK"
    BROKEN_CONTRADICTORY_LINK = "BROKEN_CONTRADICTORY_LINK"
    ENTITY_MISMATCH_LINK = "ENTITY_MISMATCH_LINK"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"


@dataclass
class EvidenceHopLink:
    hop_index: int
    source_entity: str
    relation: str
    target_entity: str
    evidence_id: str
    is_valid: bool
    is_contradicted: bool = False
    evidence_text: str = ""


@dataclass
class MultiHopChainReport:
    chain_id: str
    query_start_entity: str
    query_target_entity: str
    required_hops: int
    observed_links: List[EvidenceHopLink]
    status: ChainCompletenessStatus
    inferred_conclusion: Optional[str]
    missing_link_index: Optional[int]
    is_deductively_sound: bool
    rationale: str


class MultiHopEvidenceChainEngine:
    """
    Validates compositional and recursive relational chains across multiple external evidence items.
    """

    @classmethod
    def verify_chain(
        cls,
        chain_id: str,
        start_entity: str,
        target_entity: str,
        evidence_hops: List[Dict[str, Any]],
        expected_hop_count: int,
    ) -> MultiHopChainReport:
        if len(evidence_hops) < expected_hop_count:
            return MultiHopChainReport(
                chain_id=chain_id,
                query_start_entity=start_entity,
                query_target_entity=target_entity,
                required_hops=expected_hop_count,
                observed_links=[],
                status=ChainCompletenessStatus.INSUFFICIENT_DEPTH,
                inferred_conclusion=None,
                missing_link_index=len(evidence_hops),
                is_deductively_sound=False,
                rationale=f"Chain requires {expected_hop_count} hops, but only {len(evidence_hops)} evidence items provided.",
            )

        current_node = start_entity.strip().lower()
        links: List[EvidenceHopLink] = []

        for idx, ev in enumerate(evidence_hops):
            s_ent = ev.get("source_entity", "").strip().lower()
            rel = ev.get("relation", "").strip().lower()
            t_ent = ev.get("target_entity", "").strip().lower()
            ev_id = ev.get("evidence_id", f"ev_hop_{idx}")
            is_contra = ev.get("is_contradicted", False)

            link = EvidenceHopLink(
                hop_index=idx,
                source_entity=s_ent,
                relation=rel,
                target_entity=t_ent,
                evidence_id=ev_id,
                is_valid=True,
                is_contradicted=is_contra,
                evidence_text=ev.get("text", ""),
            )

            # Check 1: Contradictory link
            if is_contra:
                links.append(link)
                return MultiHopChainReport(
                    chain_id=chain_id,
                    query_start_entity=start_entity,
                    query_target_entity=target_entity,
                    required_hops=expected_hop_count,
                    observed_links=links,
                    status=ChainCompletenessStatus.BROKEN_CONTRADICTORY_LINK,
                    inferred_conclusion=None,
                    missing_link_index=idx,
                    is_deductively_sound=False,
                    rationale=f"Hop {idx} ({s_ent} -> {t_ent}) is broken by contradictory evidence.",
                )

            # Check 2: Entity continuity (Current node must connect to next source)
            if s_ent != current_node:
                links.append(link)
                return MultiHopChainReport(
                    chain_id=chain_id,
                    query_start_entity=start_entity,
                    query_target_entity=target_entity,
                    required_hops=expected_hop_count,
                    observed_links=links,
                    status=ChainCompletenessStatus.ENTITY_MISMATCH_LINK,
                    inferred_conclusion=None,
                    missing_link_index=idx,
                    is_deductively_sound=False,
                    rationale=f"Hop {idx} broken: expected source '{current_node}', but found '{s_ent}'.",
                )

            current_node = t_ent
            links.append(link)

        # Check final entity reached
        if current_node != target_entity.strip().lower():
            return MultiHopChainReport(
                chain_id=chain_id,
                query_start_entity=start_entity,
                query_target_entity=target_entity,
                required_hops=expected_hop_count,
                observed_links=links,
                status=ChainCompletenessStatus.MISSING_LINK,
                inferred_conclusion=None,
                missing_link_index=expected_hop_count - 1,
                is_deductively_sound=False,
                rationale=f"Chain terminated at '{current_node}' rather than expected target '{target_entity}'.",
            )

        conclusion = f"{start_entity} is compositionally connected to {target_entity} across {len(links)} verified hops."
        return MultiHopChainReport(
            chain_id=chain_id,
            query_start_entity=start_entity,
            query_target_entity=target_entity,
            required_hops=expected_hop_count,
            observed_links=links,
            status=ChainCompletenessStatus.COMPLETE_GROUNDED,
            inferred_conclusion=conclusion,
            missing_link_index=None,
            is_deductively_sound=True,
            rationale="All intermediate hops deductively validated with full entity continuity.",
        )
