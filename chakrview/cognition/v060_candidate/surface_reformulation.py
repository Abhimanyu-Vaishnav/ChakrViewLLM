"""
Stage 6: Surface Reformulation Generalization.

Guarantees:
- Tests structurally equivalent natural language inputs with different phrasing:
  - Active: "Ram lives in Delhi."
  - Passive / Inverted: "Delhi is where Ram lives."
  - Synonym / Lexical shift: "Ram resides in Delhi."
- Verifies that underlying canonical semantic triples remain invariant across surface forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from chakrview.cognition.v060_candidate.language_grounding import (
    LanguageSemanticGroundingBridge,
    GroundingResult,
)


@dataclass
class ReformulationResult:
    total_surface_variants: int
    invariant_matches: int
    reformulation_invariance_rate: float
    canonical_form_achieved: str
    is_generalization_pass: bool
    rationale: str


class SurfaceReformulationTester:
    """
    Tests whether distinct surface expressions map consistently to identical semantic triples.
    """

    @classmethod
    def test_surface_invariance(
        cls,
        surface_variants: List[str],
        expected_canonical_form: str,
    ) -> ReformulationResult:
        matches = 0
        canonical_forms = []

        for variant in surface_variants:
            res = LanguageSemanticGroundingBridge.ground_text_to_semantic_triple(variant)
            canonical_forms.append(res.canonical_semantic_form)
            if res.is_grounded and res.canonical_semantic_form == expected_canonical_form:
                matches += 1

        total = len(surface_variants)
        rate = (matches / total) if total > 0 else 0.0
        passed = (rate == 1.0)

        return ReformulationResult(
            total_surface_variants=total,
            invariant_matches=matches,
            reformulation_invariance_rate=rate,
            canonical_form_achieved=canonical_forms[0] if canonical_forms else "NONE",
            is_generalization_pass=passed,
            rationale=f"{matches}/{total} surface variants mapped to target canonical triple '{expected_canonical_form}'.",
        )
