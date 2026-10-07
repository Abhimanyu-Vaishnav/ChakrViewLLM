"""Master benchmark runner for Wave 377–384: Adaptive Recurrent Reasoning Core.

Executes all verification steps end-to-end:
- Baseline Invariant Pre-check (Parameters = 3,443,136, SHA-256 Bit-Exact)
- Step 377: Adaptive Recurrent Core Architecture & Parameter Audit (<600k budget)
- Step 378: Minimal Adaptive Learnability (Mixed 1/2/3 Hop Tasks)
- Step 379: Adaptive vs Fixed Depth Control Study (Variants A through F)
- Step 380: Adaptive Computation Objective & Pareto Tradeoff Study
- Step 381: State Causality & Stability under Adaptive Cycles
- Step 382: Compositional Generalization & Anti-Shortcut Suite
- Step 383: Harder Composition (4-Hop) & Resource Adaptation Study
- Step 384: Master Decision Gate across Strict Seeds (42, 101, 2026)
- Baseline Invariant Post-check (Delta W = 0)
"""

import time
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.adaptive_recurrent_reasoning_core import (
    AdaptiveRecurrentReasoningCore,
    inspect_adaptive_core,
)
from chakrview.cognition.adaptive_learnability import train_and_eval_adaptive_learnability
from chakrview.cognition.adaptive_depth_study import run_adaptive_vs_fixed_depth_study
from chakrview.cognition.adaptive_computation_objective import run_adaptive_computation_objective_study
from chakrview.cognition.adaptive_state_causality import run_adaptive_state_causality_study
from chakrview.cognition.adaptive_generalization_stress import evaluate_adaptive_compositional_generalization
from chakrview.cognition.adaptive_resource_adaptation import run_four_hop_and_budget_adaptation_study
from chakrview.cognition.adaptive_master_decision_gate import run_strict_adaptive_i4_evaluation


def main():
    t0 = time.time()
    print("=" * 75)
    print("CHAKRVIEW WAVE 377–384: ADAPTIVE RECURRENT REASONING CORE BENCHMARK")
    print("=" * 75)

    # 1. Baseline Invariant Pre-check
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Pre-Wave SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Pre-wave baseline SHA mismatch!"
    assert total_base_params == 3_443_136, "Pre-wave baseline param count mismatch!"

    # 2. Step 377 Architecture & Parameter Budget
    print("\n--- STEP 377: Adaptive Recurrent Core Architecture & Budget Audit ---")
    core = AdaptiveRecurrentReasoningCore()
    trainable_p = core.trainable_param_count
    total_p = core.total_param_count
    controller_p = sum(p.numel() for p in core.controller.parameters())
    print(f"Trainable Parameters: {trainable_p:,}")
    print(f"Total Parameters: {total_p:,}")
    print(f"Controller Parameters: {controller_p:,}")
    assert trainable_p < 600_000, f"Trainable params {trainable_p} exceeds 600k budget!"
    assert total_p < 1_000_000, f"Total params {total_p} exceeds 1M hard limit!"
    print("Parameter budget audit: PASS (<600k preferred, <1M hard limit)")

    # 3. Step 378 Minimal Adaptive Learnability
    print("\n--- STEP 378: Minimal Adaptive Learnability (Mixed 1/2/3 Hop Tasks) ---")
    core, step378_rep = train_and_eval_adaptive_learnability(core=core, seed=42, train_steps=20, eval_episodes_per_hop=4)
    print(f"Overall Accuracy: {step378_rep.overall_accuracy:.2%}")
    print(f"Mean Cycles Overall: {step378_rep.mean_cycles_overall:.2f}")
    for h, m in step378_rep.metrics_by_hop.items():
        print(f"  {h}-Hop: Acc={m.accuracy:.2%}, AvgC={m.avg_cycles:.2f}, Premature={m.premature_halts:.2%}")
    print(f"Step 378 Summary: {step378_rep.summary}")

    # 4. Step 379 Adaptive vs Fixed Depth Control Study
    print("\n--- STEP 379: Adaptive vs Fixed Depth Control Study ---")
    step379_rep = run_adaptive_vs_fixed_depth_study(seed=42, train_steps=15, eval_episodes_per_hop=4)
    for v_id, vr in step379_rep.variants.items():
        print(f"  [{v_id:20s}] G4={vr.g4_acc:.2%}, MeanC={vr.mean_cycles:.2f}, 1hC={vr.cycles_1hop:.1f}, 2hC={vr.cycles_2hop:.1f}")
    print(f"Best Variant: {step379_rep.best_variant} | Generalization Delta: {step379_rep.generalization_delta:+.2%}")

    # 5. Step 380 Adaptive Computation Objective & Pareto Study
    print("\n--- STEP 380: Adaptive Computation Objective & Pareto Study ---")
    step380_rep = run_adaptive_computation_objective_study(seed=42, train_steps=15, eval_episodes=4)
    for lam, pt in step380_rep.points.items():
        print(f"  Lambda={lam:5.2f} -> Acc={pt.accuracy:.2%}, AvgC={pt.mean_cycles:.2f}, Score={pt.pareto_score:.4f}")
    print(f"Best Lambda: {step380_rep.best_lambda}")

    # 6. Step 381 State Causality & Stability
    print("\n--- STEP 381: State Causality & Stability ---")
    step381_rep = run_adaptive_state_causality_study(core=core, seed=42, num_episodes=6)
    for c_id, cr in step381_rep.interventions.items():
        print(f"  [{c_id:12s}] Acc={cr.final_acc:.2%}, Prob={cr.target_prob:.4f}, Drop={cr.drop_from_normal:+.4f}")
    print(f"Inter-Cycle Cosine Sim: {step381_rep.stability.inter_cycle_cosine_sim:.4f}")
    print(f"Query-State Alignment: {step381_rep.stability.query_state_alignment:.4f}")

    # 7. Step 382 Compositional Generalization Stress
    print("\n--- STEP 382: Compositional Generalization & Anti-Shortcut Suite ---")
    step382_rep = evaluate_adaptive_compositional_generalization(core, seed=42, episodes_per_condition=6)
    print(f"G1={step382_rep.g1_acc:.2%}, G2={step382_rep.g2_acc:.2%}, G3={step382_rep.g3_acc:.2%}, G4={step382_rep.g4_acc:.2%}")
    print(f"Passed Permutations: {step382_rep.passed_conditions}/{step382_rep.total_conditions}")
    print(f"Positional Correlation: {step382_rep.positional_correlation:+.4f}")
    print(f"Contamination Zero: {step382_rep.contamination_zero}")

    # 8. Step 383 4-Hop Composition & Resource Adaptation
    print("\n--- STEP 383: 4-Hop Escalation & Resource Budget Adaptation ---")
    step383_rep = run_four_hop_and_budget_adaptation_study(core=core, seed=42, episodes=6)
    print(f"4-Hop Mean Accuracy: {step383_rep.four_hop_mean_acc:.2%}")
    for b, pt in step383_rep.budget_sweep.items():
        print(f"  Budget {b}: Acc={pt.final_acc:.2%}, MeanC={pt.mean_cycles:.2f}, Bounded={pt.strictly_bounded}")

    # 9. Step 384 Master Decision Gate across Seeds (42, 101, 2026)
    print("\n--- STEP 384: Master Decision Gate across Seeds (42, 101, 2026) ---")
    step384_rep = run_strict_adaptive_i4_evaluation(seeds=(42, 101, 2026), train_steps=20, eval_episodes=6)
    print(f"Final Classification: {step384_rep.final_classification}")
    print(f"Decision Rationale: {step384_rep.decision_rationale}")
    print(f"Mean G4: {step384_rep.mean_g4:.2%}")
    print(f"Mean H2: {step384_rep.mean_h2_routing:.2%}")
    print(f"Language Retention: {step384_rep.language_retention:.4f}")
    print(f"Trainable Parameters: {step384_rep.trainable_parameters:,}")
    print(f"Baseline Exact: {step384_rep.baseline_exact}")

    # 10. Baseline Post-check
    post_hash = compute_model_hash(base_model)
    print(f"\n[*] Canonical Baseline Post-Wave SHA-256: {post_hash}")
    assert post_hash == EXPECTED_WEIGHT_HASH, "Post-wave baseline SHA mismatch!"
    assert init_hash == post_hash, "Baseline altered during benchmarking!"
    print("[*] Delta W = 0 verified. Canonical baseline is strictly bit-exact.")

    dt = time.time() - t0
    print("\n" + "=" * 75)
    print(f"WAVE 377–384 BENCHMARK COMPLETE (Total CPU Runtime: {dt:.2f}s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
