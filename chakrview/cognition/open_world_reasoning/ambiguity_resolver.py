"""
Wave 573: Ambiguous Language & Entity Disambiguation Engine.

Architectural Guarantees:
- Never assume the most popular/famous entity blindly (e.g. "Apple" company vs "apple" fruit).
- Disambiguates:
  same-name entities, acronyms, geographic overlap, role overlap, product vs org.
- States:
  RESOLVED, PARTIALLY_RESOLVED, AMBIGUOUS, UNRESOLVED.
- If ambiguous and context lacks discriminating evidence:
  triggers ASK_CLARIFICATION or ABSTAIN_WITH_REASON.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set


class AmbiguityType(str, Enum):
    HOMONYM_ENTITY = "HOMONYM_ENTITY"          # E.g. Apple Inc. vs Apple (botanical)
    GEOGRAPHIC_OVERLAP = "GEOGRAPHIC_OVERLAP"  # E.g. Springfield IL vs Springfield MA
    ACRONYM_COLLISION = "ACRONYM_COLLISION"    # E.g. NASA, MIT, or domain-specific acronyms
    TEMPORAL_ROLE = "TEMPORAL_ROLE"            # E.g. "The President" (which year?)
    NONE = "NONE"


class EntityResolutionState(str, Enum):
    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class CandidateEntity:
    entity_id: str
    canonical_name: str
    entity_type: str
    disambiguating_tokens: List[str]
    context_score: float = 0.0


@dataclass
class EntityDisambiguationResult:
    query_surface_form: str
    resolution_state: EntityResolutionState
    ambiguity_type: AmbiguityType
    selected_entity: Optional[CandidateEntity]
    candidate_entities: List[CandidateEntity]
    action: str                     # PROCEED, ASK_CLARIFICATION, ABSTAIN_WITH_REASON
    clarification_prompt: Optional[str]
    rationale: str


class AmbiguityResolutionEngine:
    """
    Detects lexical and entity-level ambiguity, resolves identity against context,
    and asks clarification rather than guessing.
    """

    @classmethod
    def resolve_entity(
        cls,
        surface_form: str,
        context_text: str,
        known_candidates: List[CandidateEntity],
        ambiguity_type: AmbiguityType = AmbiguityType.HOMONYM_ENTITY,
    ) -> EntityDisambiguationResult:
        if not known_candidates:
            return EntityDisambiguationResult(
                query_surface_form=surface_form,
                resolution_state=EntityResolutionState.UNRESOLVED,
                ambiguity_type=ambiguity_type,
                selected_entity=None,
                candidate_entities=[],
                action="ABSTAIN_WITH_REASON",
                clarification_prompt=f"No known entities registered for reference '{surface_form}'.",
                rationale="Unresolvable entity reference.",
            )

        if len(known_candidates) == 1:
            return EntityDisambiguationResult(
                query_surface_form=surface_form,
                resolution_state=EntityResolutionState.RESOLVED,
                ambiguity_type=AmbiguityType.NONE,
                selected_entity=known_candidates[0],
                candidate_entities=known_candidates,
                action="PROCEED",
                clarification_prompt=None,
                rationale="Single unambiguous candidate.",
            )

        # Multiple candidates: score against context
        context_clean = context_text.lower()
        scored: List[CandidateEntity] = []

        for cand in known_candidates:
            matches = sum(
                1 for tok in cand.disambiguating_tokens if tok.lower() in context_clean
            )
            score = matches / max(len(cand.disambiguating_tokens), 1)
            cand.context_score = score
            scored.append(cand)

        # Sort descending by score
        scored.sort(key=lambda x: x.context_score, reverse=True)
        top = scored[0]
        runner_up = scored[1]

        # Distinct threshold
        if top.context_score >= 0.5 and (top.context_score - runner_up.context_score) >= 0.3:
            return EntityDisambiguationResult(
                query_surface_form=surface_form,
                resolution_state=EntityResolutionState.RESOLVED,
                ambiguity_type=ambiguity_type,
                selected_entity=top,
                candidate_entities=scored,
                action="PROCEED",
                clarification_prompt=None,
                rationale=f"Context successfully disambiguated '{surface_form}' as {top.canonical_name} ({top.entity_type}).",
            )
        elif top.context_score > 0.2 and top.context_score == runner_up.context_score:
            # Tie between top candidates -> Ambiguous
            cand_names = [f"{c.canonical_name} ({c.entity_type})" for c in scored[:3]]
            return EntityDisambiguationResult(
                query_surface_form=surface_form,
                resolution_state=EntityResolutionState.AMBIGUOUS,
                ambiguity_type=ambiguity_type,
                selected_entity=None,
                candidate_entities=scored,
                action="ASK_CLARIFICATION",
                clarification_prompt=f"'{surface_form}' could refer to {', or '.join(cand_names)}. Which one do you mean?",
                rationale="Context contains equal or insufficient cues to separate candidates.",
            )
        elif top.context_score < 0.2:
            # Completely absent context
            cand_names = [f"{c.canonical_name} ({c.entity_type})" for c in scored[:3]]
            return EntityDisambiguationResult(
                query_surface_form=surface_form,
                resolution_state=EntityResolutionState.AMBIGUOUS,
                ambiguity_type=ambiguity_type,
                selected_entity=None,
                candidate_entities=scored,
                action="ASK_CLARIFICATION",
                clarification_prompt=f"Reference '{surface_form}' is ambiguous without context. Are you referring to {', or '.join(cand_names)}?",
                rationale="Zero discriminating context found for ambiguous reference.",
            )
        else:
            return EntityDisambiguationResult(
                query_surface_form=surface_form,
                resolution_state=EntityResolutionState.PARTIALLY_RESOLVED,
                ambiguity_type=ambiguity_type,
                selected_entity=top,
                candidate_entities=scored,
                action="PROCEED",
                clarification_prompt=None,
                rationale="Weak disambiguation supported by partial cues.",
            )
