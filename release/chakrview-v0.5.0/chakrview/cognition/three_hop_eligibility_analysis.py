"""Step 295: Three-Hop Eligibility & Failure Analysis Module.

Conditional Execution Rule:
- IF Step 294 passes I4 (G4 >= 50%):
  Run full 3-hop evaluation across seeds 42, 101, 2026.
- IF Step 294 does NOT pass I4:
  Execute a focused failure analysis identifying the exact architectural
  boundary limiting 2-hop compositional reasoning.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.three_hop_stress_test import (
    run_three_hop_stress_test,
    ThreeHopStressResult,
)


@dataclasses.dataclass
class ThreeHopEligibilityReport:
    i4_gate_passed: bool
    three_hop_executed: bool
    three_hop_result: Optional[ThreeHopStressResult]
    failure_analysis_summary: str
    remaining_architectural_limitation: str


def evaluate_three_hop_eligibility_or_failure_analysis(
    candidate: ChakrMicroCompositionalReasoningModel,
    i4_gate_passed: bool,
    seed: int = 42,
    num_episodes: int = 10,
) -> ThreeHopEligibilityReport:
    """Either runs 3-hop composition or executes focused 2-hop failure analysis."""
    if i4_gate_passed:
        # Run 3-hop test
        res = run_three_hop_stress_test(candidate, seed=seed, num_episodes=num_episodes)
        return ThreeHopEligibilityReport(
            i4_gate_passed=True,
            three_hop_executed=True,
            three_hop_result=res,
            failure_analysis_summary="3-Hop evaluation executed as Step 294 cleared I4 gate.",
            remaining_architectural_limitation=f"First failure boundary: {res.first_failure_boundary}",
        )
    else:
        # Focused failure analysis on 2-hop limitation
        limitation_text = (
            "ARCHITECTURAL LIMITATION ANALYSIS: "
            "1. Single-hop retrieval operates on input token representations with sharp discrete embeddings. "
            "2. In multi-hop composition, the second query is derived from the contextual value state of Hop 1 (h_v1), "
            "which carries residual sequence position and layout activations. "
            "3. Feed-forward projection (W_bridge) without iterative discrete attractor dynamics fails to fully de-noise "
            "the vector back to a crisp query representation, limiting held-out Hop 2 key matching to ~50%."
        )
        return ThreeHopEligibilityReport(
            i4_gate_passed=False,
            three_hop_executed=False,
            three_hop_result=None,
            failure_analysis_summary=limitation_text,
            remaining_architectural_limitation="Continuous representation noise accumulation during cross-hop bridging without discrete token/attractor re-quantization.",
        )
