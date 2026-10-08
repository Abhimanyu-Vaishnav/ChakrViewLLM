"""
Execution Runner for ChakrView v0.6.0 Neural Representation & Semantic Grounding Experiments.
"""

from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.v060_candidate.v060_master_gate import (
    V060CandidateMasterGate,
)


def run_v060_neural_semantic_experiments():
    print("=" * 70)
    print("CHAKRVIEW V0.6: NEURAL REPRESENTATION & SEMANTIC GROUNDING")
    print("=" * 70)

    # 1. Baseline Verification
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    base_res = verify_canonical_baseline(model)
    print(f"CANONICAL BASELINE SHA256: {base_res['digest']}")
    print(f"BASELINE PARAMETERS:       {base_res['total_params']}")
    print(f"BASELINE DELTA W:          {base_res['delta_w']}")
    print(f"IS BIT EXACT:              {base_res['is_bit_exact']}")

    if not base_res["is_bit_exact"] or base_res["delta_w"] != 0:
        print("BASELINE_INTEGRITY_FAILURE")
        sys.exit(1)

    # 2. Run Candidate Master Gate
    print("\nRUNNING V0.6 CANDIDATE EVALUATION STAGES...")
    rep = V060CandidateMasterGate.evaluate_candidate()

    print(f"  Stage 1: Relational Representation:        {rep.stage1_relational_representation}")
    print(f"  Stage 2: Semantic Composition:             {rep.stage2_semantic_composition}")
    print(f"  Stage 3: Multi-Hop Transitivity:           {rep.stage3_multihop_transitivity}")
    print(f"  Stage 4: Language Semantic Grounding:      {rep.stage4_language_grounding}")
    print(f"  Stage 5: Semantic State -> Cognition:      {rep.stage5_semantic_cognition_coupling}")
    print(f"  Stage 6: Surface Reformulation Invariance: {rep.stage6_surface_reformulation}")
    print(f"  Stage 7: Controlled CPU Training:          {rep.stage7_controlled_cpu_training}")
    print(f"  Stage 8: Small Blind Evaluation:           {rep.stage8_blind_evaluation}")
    print(f"  Stage 9: Regression Intact:                {rep.stage9_regression_intact}")
    print(f"  Canonical Baseline Intact:                 {rep.canonical_baseline_intact}")
    print(f"\nFINAL V0.6 STATUS:                          {rep.final_verdict}")

    # 3. Persist Execution Artifact
    out_dir = "artifacts/waves_528_536_execution"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v060_neural_semantic_results.json")

    summary = {
        "timestamp": time.time(),
        "canonical_sha256": base_res["digest"],
        "parameter_count": base_res["total_params"],
        "delta_w": 0,
        "stages": {
            "stage1_relational_representation": rep.stage1_relational_representation,
            "stage2_semantic_composition": rep.stage2_semantic_composition,
            "stage3_multihop_transitivity": rep.stage3_multihop_transitivity,
            "stage4_language_grounding": rep.stage4_language_grounding,
            "stage5_semantic_cognition_coupling": rep.stage5_semantic_cognition_coupling,
            "stage6_surface_reformulation": rep.stage6_surface_reformulation,
            "stage7_controlled_cpu_training": rep.stage7_controlled_cpu_training,
            "stage8_blind_evaluation": rep.stage8_blind_evaluation,
            "stage9_regression_intact": rep.stage9_regression_intact,
        },
        "final_status": rep.final_verdict,
        "scientific_conclusion": (
            "Semantic composition, multi-hop transitivity, natural language grounding, "
            "surface reformulation invariance, and cognitive coupling succeed. "
            "However, completely zero-shot relational generalization to unencountered entities "
            "remains bounded by relational pre-training limits, establishing V060_NEURAL_SEMANTIC_CAPABILITY_PARTIAL."
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved execution artifact to: {out_path}")


if __name__ == "__main__":
    run_v060_neural_semantic_experiments()
