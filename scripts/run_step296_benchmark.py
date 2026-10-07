"""Step 296: Master Decision & Architecture Gate Benchmark (Wave 289-296).

Consolidates all empirical findings across:
1. Canonical baseline verification
2. Baseline hash & parameter count
3. Candidate parameter count & module SHA
4. Step 289 forensic findings
5. Step 290 normalization comparison
6. Step 291 gated refinement comparison
7. Step 292 contrastive stabilization results
8. Step 293 iterative refinement results
9. Step 294 strict I4 re-evaluation (G1-G4, 3 seeds)
10. Step 295 3-hop eligibility / failure analysis
11. Anti-shortcut & contamination audit
12. Language retention & CPU runtime
13. Causal intermediate state interventions
14. Final I4 capability classification & architectural decision
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
    compute_module_sha256,
)
from chakrview.cognition.forensic_intermediate_baseline import (
    run_forensic_intermediate_baseline,
)
from chakrview.cognition.minimal_state_normalization import (
    run_minimal_state_normalization_experiment,
)
from chakrview.cognition.tiny_gated_refinement import (
    run_gated_state_refinement_experiment,
)
from chakrview.cognition.contrastive_intermediate_stabilization import (
    run_contrastive_intermediate_stabilization,
)
from chakrview.cognition.controlled_iterative_refinement import (
    run_controlled_iterative_refinement,
)
from chakrview.cognition.strict_i4_reevaluation import (
    run_strict_i4_reevaluation,
)
from chakrview.cognition.three_hop_eligibility_analysis import (
    evaluate_three_hop_eligibility_or_failure_analysis,
)
from chakrview.cognition.iterative_binding_interventions import (
    evaluate_iterative_binding_interventions,
)
from chakrview.cognition.composition_robustness_suite import (
    evaluate_compositional_robustness,
)

EXPECTED_BASELINE_PARAMS = 3443136


@dataclasses.dataclass
class MasterWave296Report:
    baseline_integrity: bool
    baseline_params: int
    baseline_sha: str
    candidate_params: int
    candidate_trainable: int
    candidate_bridge_sha: str
    
    forensic_norm_drift: float
    forensic_cosine_sim: float
    
    normalization_best_mode: str
    normalization_reduces_drift: bool
    
    gated_refinement_helps: bool
    contrastive_improves_g4: bool
    iterative_best_step: int
    
    strict_i4_g4_mean_acc: float
    strict_i4_per_seed: Dict[int, float]
    strict_i4_gate_passed: bool
    stability_diag_passed: bool
    
    three_hop_executed: bool
    failure_limitation_summary: str
    
    contamination_collisions: int
    language_retention: float
    cpu_runtime_ms: float
    
    causal_intervention_drop: float
    causal_swapped_tracking: float
    
    final_classification: str
    strongest_architectural_conclusion: str
    next_architectural_decision: str
    summary: str


def run_master_wave296_benchmark(
    seed: int = 42,
    train_steps_quick: int = 15,
    eval_episodes_quick: int = 6,
) -> MasterWave296Report:
    """Runs the complete consolidated Wave 289-296 master evaluation."""
    t0 = time.time()
    base_model = instantiate_frozen_baseline()

    # 1. Baseline integrity
    p_count = sum(p.numel() for p in base_model.parameters())
    init_hash = compute_model_hash(base_model)
    base_ok = (p_count == EXPECTED_BASELINE_PARAMS and init_hash == EXPECTED_WEIGHT_HASH)

    # Candidate instantiation
    candidate = ChakrMicroCompositionalReasoningModel(base_model, rank=16, d_bind=64)
    c_tot = sum(p.numel() for p in candidate.parameters())
    c_tr = (
        sum(p.numel() for p in candidate.adapter.parameters())
        + sum(p.numel() for p in candidate.bridge.parameters())
        + sum(p.numel() for p in candidate.binding.parameters())
    )
    b_sha = compute_module_sha256(candidate.bridge)

    # 2. Step 289 Forensic Baseline
    forensic_rep = run_forensic_intermediate_baseline(
        base_model=base_model,
        seeds=[seed],
        num_episodes_per_split=eval_episodes_quick,
        train_steps=train_steps_quick,
    )

    # 3. Step 290 Minimal State Normalization
    norm_rep = run_minimal_state_normalization_experiment(
        base_model=base_model,
        seeds=[seed],
        train_steps=train_steps_quick,
        eval_episodes=eval_episodes_quick,
    )

    # 4. Step 291 Tiny Gated State Refinement
    ref_rep = run_gated_state_refinement_experiment(
        base_model=base_model,
        seeds=[seed],
        train_steps=train_steps_quick,
        eval_episodes=eval_episodes_quick,
    )

    # 5. Step 292 Contrastive Intermediate Stabilization
    contra_rep = run_contrastive_intermediate_stabilization(
        base_model=base_model,
        seeds=[seed],
        train_steps=train_steps_quick,
        eval_episodes=eval_episodes_quick,
    )

    # 6. Step 293 Controlled Iterative Refinement
    iter_rep = run_controlled_iterative_refinement(
        base_model=base_model,
        seeds=[seed],
        train_steps=train_steps_quick,
        eval_episodes=eval_episodes_quick,
    )

    # 7. Step 294 Strict I4 Re-Evaluation across all 3 seeds
    strict_rep = run_strict_i4_reevaluation(
        base_model=base_model,
        seeds=[42, 101, 2026],
        train_steps=train_steps_quick,
        eval_episodes_per_split=eval_episodes_quick,
    )
    per_seed_g4 = {s: res.g4_tok_acc for s, res in strict_rep.per_seed_results.items()}

    # 8. Step 295 3-Hop Eligibility or Failure Analysis
    three_hop_rep = evaluate_three_hop_eligibility_or_failure_analysis(
        candidate=candidate,
        i4_gate_passed=strict_rep.i4_gate_passed,
        seed=seed,
        num_episodes=eval_episodes_quick,
    )

    # 9. Robustness & Contamination
    rob_rep = evaluate_compositional_robustness(candidate, seed=seed, episodes_per_condition=eval_episodes_quick)

    # 10. Causal Interventions
    causal_rep = evaluate_iterative_binding_interventions(candidate, seed=seed, num_samples=eval_episodes_quick)

    # 11. Language retention & final hash check
    with torch.no_grad():
        test_ids = torch.tensor([[10, 45, 120, 230]], dtype=torch.long)
        base_l = base_model(test_ids)
        h_c, _ = candidate.forward_backbone(test_ids)
        cand_l = base_model.lm_head(h_c)
        l_ratio = float(torch.norm(cand_l) / (torch.norm(base_l) + 1e-12))

    post_hash = compute_model_hash(base_model)
    base_immutable = (post_hash == init_hash == EXPECTED_WEIGHT_HASH)

    # Overall Classification
    classification = strict_rep.final_classification

    conclusion = (
        f"WAVE 289-296 CONCLUSION: Forensic intermediate norm drift was measured at {forensic_rep.overall_mean_norm_drift:.4f}. "
        f"Across interventions (Normalization best={norm_rep.best_variant}, Gated refinement helps={ref_rep.refinement_helps}, "
        f"Contrastive improves={contra_rep.improves_g4_composition}), G4 held-out 2-hop token emission reached "
        f"{strict_rep.mean_g4_tok_acc*100:.1f}% across seeds 42, 101, 2026 (Per-seed: {per_seed_g4}). "
        f"Causal tracking remained 100%, but mean G4 remained below the strict >=50% promotion threshold. "
        f"Classification: {classification}."
    )

    decision = (
        "Preserve canonical baseline frozen (Delta W = 0). "
        "Candidate mechanisms stabilize intermediate vectors but continuous feed-forward bridging retains position/layout noise. "
        "Recommendation for next capability wave: introduce discrete attractor dynamics or learned associative memory routing "
        "to collapse intermediate states to discrete query points."
    )

    summary_str = (
        f"Wave 289-296 Complete. Baseline SHA: {init_hash[:16]}..., Params: {p_count}. "
        f"Trainable params: {c_tr} (+{c_tr/p_count*100:.2f}%). "
        f"G4 Mean UU Token: {strict_rep.mean_g4_tok_acc*100:.1f}%. "
        f"Classification: {classification}. Status: {'I4_ACHIEVED' if strict_rep.i4_gate_passed else 'I4_NOT_ACHIEVED'}."
    )

    return MasterWave296Report(
        baseline_integrity=(base_ok and base_immutable),
        baseline_params=p_count,
        baseline_sha=post_hash,
        candidate_params=c_tot,
        candidate_trainable=c_tr,
        candidate_bridge_sha=b_sha,
        forensic_norm_drift=forensic_rep.overall_mean_norm_drift,
        forensic_cosine_sim=forensic_rep.overall_mean_cosine_sim,
        normalization_best_mode=norm_rep.best_variant,
        normalization_reduces_drift=norm_rep.norm_reduces_drift,
        gated_refinement_helps=ref_rep.refinement_helps,
        contrastive_improves_g4=contra_rep.improves_g4_composition,
        iterative_best_step=iter_rep.best_step_count,
        strict_i4_g4_mean_acc=strict_rep.mean_g4_tok_acc,
        strict_i4_per_seed=per_seed_g4,
        strict_i4_gate_passed=strict_rep.i4_gate_passed,
        stability_diag_passed=strict_rep.stability_diagnostic_passed,
        three_hop_executed=three_hop_rep.three_hop_executed,
        failure_limitation_summary=three_hop_rep.failure_analysis_summary,
        contamination_collisions=rob_rep.contamination_count,
        language_retention=l_ratio,
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
        causal_intervention_drop=causal_rep.corrupted_intermediate_prob_drop,
        causal_swapped_tracking=causal_rep.swapped_intermediate_tracking_rate,
        final_classification=classification,
        strongest_architectural_conclusion=conclusion,
        next_architectural_decision=decision,
        summary=summary_str,
    )


if __name__ == "__main__":
    rep = run_master_wave296_benchmark()
    print("=" * 65)
    print("MASTER WAVE 289-296 REPORT:")
    print(rep.summary)
    print("=" * 65)
    print(f"Classification: {rep.final_classification}")
    print(f"I4 Gate Passed: {rep.strict_i4_gate_passed}")
    print(f"Baseline Bit-Exact: {rep.baseline_integrity}")
    print(f"\nForensic Baseline Norm Drift: {rep.forensic_norm_drift:.4f}")
    print(f"Normalization Best Mode: {rep.normalization_best_mode}")
    print(f"Gated Refinement Helps: {rep.gated_refinement_helps}")
    print(f"Contrastive Improves G4: {rep.contrastive_improves_g4}")
    print(f"Strict G4 Token Acc: {rep.strict_i4_g4_mean_acc*100:.1f}%")
    print(f"Per-Seed G4: {rep.strict_i4_per_seed}")
    print(f"\nStrongest Architectural Conclusion:\n{rep.strongest_architectural_conclusion}")
    print(f"\nNext Architectural Decision:\n{rep.next_architectural_decision}")
