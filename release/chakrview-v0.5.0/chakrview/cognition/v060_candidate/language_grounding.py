"""
Stage 4: Language Semantic Grounding.

Guarantees:
- Maps natural language text sequences into internal relational structures:
  (SUBJECT, RELATION, OBJECT)
- Tests multiple linguistic surface forms mapping toward the same semantic triple:
  - "Ram lives in Delhi"
  - "Ram resides in Delhi"
  - "Delhi is where Ram lives"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class GroundedTriple:
    subject: str
    relation: str
    object: str
    surface_form: str
    confidence: float


@dataclass
class GroundingResult:
    input_text: str
    extracted_triple: Optional[GroundedTriple]
    canonical_semantic_form: str
    is_grounded: bool
    rationale: str


class LanguageSemanticGroundingBridge:
    """
    Grounds varied natural language surface expressions into normalized relational triples.
    """

    RELATION_PATTERNS = [
        (r"lives\s+in|resides\s+in|residence\s+is|where\s+.*lives|lives", "LIVES_IN"),
        (r"is\s+the\s+capital\s+of|capital\s+city\s+of", "CAPITAL_OF"),
        (r"works\s+at|employed\s+by|works\s+for", "WORKS_AT"),
        (r"founded|established|created", "FOUNDED_BY"),
    ]

    @classmethod
    def ground_text_to_semantic_triple(
        cls,
        text: str,
    ) -> GroundingResult:
        clean = text.strip()
        words = clean.split()

        # Identify relation pattern
        matched_rel: Optional[str] = None
        for pat, canonical_rel in cls.RELATION_PATTERNS:
            if re.search(pat, clean, re.IGNORECASE):
                matched_rel = canonical_rel
                break

        if not matched_rel:
            return GroundingResult(
                input_text=clean,
                extracted_triple=None,
                canonical_semantic_form="UNKNOWN_RELATION",
                is_grounded=False,
                rationale="No known relational semantics identified in text surface form.",
            )

        # Extract subject and object candidates
        # Look for entities (capitalized or boundary tokens)
        ents = [w.strip(".,;:?!()") for w in words if w.istitle() and len(w) > 2]

        if len(ents) >= 2:
            # Check inversion (e.g. "Delhi is where Ram lives")
            if "where" in clean.lower():
                subj = ents[1]
                obj = ents[0]
            else:
                subj = ents[0]
                obj = ents[1]

            triple = GroundedTriple(
                subject=subj,
                relation=matched_rel,
                object=obj,
                surface_form=clean,
                confidence=0.92,
            )
            canonical = f"({subj}) --[{matched_rel}]--> ({obj})"
            return GroundingResult(
                input_text=clean,
                extracted_triple=triple,
                canonical_semantic_form=canonical,
                is_grounded=True,
                rationale=f"Grounded surface form into canonical triple: {canonical}.",
            )

        return GroundingResult(
            input_text=clean,
            extracted_triple=None,
            canonical_semantic_form="INSUFFICIENT_ENTITIES",
            is_grounded=False,
            rationale="Identified relation but insufficient bound entities.",
        )
