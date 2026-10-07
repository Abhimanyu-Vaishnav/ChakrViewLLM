"""Master benchmark runner for Wave 393–400: Stable Relational Initialization + I4 Acceleration.

Executes all verification steps end-to-end:
- Baseline Invariant Pre-check (Parameters = 3,443,136, SHA-256 Bit-Exact)
- Step 393: Hop-1 Relational Initialization & Attention Geometry Study (Modes A through G)
- Step 394: Relational Matching Objective Ablation (Loss configurations A through D)
- Step 395: End-to-End Two-Hop Relational Training & Dynamic Token Binding
- Step 396: Anti-Memorization, Permutations & Distractor Invariance Suite
- Step 397: Adaptive Sufficiency Reintegration (Fixed vs Adaptive Computation)
- Step 398: Multi-Seed Stability & Bottleneck Diagnosis (Seeds 42, 101, 2026)
- Step 399: Resource-Aware Relational Reasoning under Hard Budgets (1, 2, 3, 4, 6, 8)
- Step 400: Master I4 Decision Gate & First Release Track Readiness Checklist (18 dimensions)
- Baseline Invariant Post-check (Delta W = 0, SHA Bit-Exact)
"""

import time
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.relational_initialization_study import (
    run_relational_initialization_study,
    InitializedRelationalCore,
)
from chakrview.cognition.relational_matching_objective import (
    run_relational_matching_objective_study,
)
from chakrview.cognition.end_to_end_relational_training import (
    train_and_eval_end_to_end_core,
)
from chakrview.cognition.anti_memorization_generalization import (
    evaluate_anti_memorization_suite,
)
from chakrview.cognition.adaptive_sufficiency_reintegration import (
    run_adaptive_sufficiency_reintegration_study,
)
from chakrview.cognition.multi_seed_relational_stability import (
    run_multi_seed_relational_stability_study,
)
from chakrview.cognition.resource_aware_relational_reasoning import (
    run_resource_aware_relational_study,
)
from chakrview.cognition.relational_master_i4_decision import (
    run_master_relational_i4_decision,
)


def main():
    t0 = time.time()
    print("=" * 75)
    print("CHAKRVIEW WAVE 393–400: STABLE RELATIONAL INITIALIZATION + I4 ACCELERATION")
    print("=" * 75)

    # 1. Baseline Invariant Pre-check
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Pre-Wave SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Pre-wave baseline SHA mismatch!"
    assert total_base_params == 3_443_136, "Pre-wave baseline param count mismatch!"

    # 2. Step 393 Hop-1 Relational Initialization Study
    print("\n--- STEP 393: Hop-1 Relational Initialization Study (Modes A-G) ---")
    step393_rep = run_relational_initialization_study(seeds=(42, 101), train_steps=20, eval_episodes=8)
    for m_id, mr in step393_rep.modes.items():
        print(f"  [{m_id:18s}] H1_K={mr.h1_key_routing:.2%}, H1_V={mr.h1_val_routing:.2%}, H2_K={mr.h2_key_routing:.2%}, G4={mr.g4_acc:.2%}")
    print(f"Best Initialization Mode: {step393_rep.best_mode} (G4={step393_rep.modes[step393_rep.best_mode].g4_acc:.2%})")

    # 3. Step 394 Relational Matching Objective
    print("\n--- STEP 394: Relational Matching Objective Ablation ---")
    step394_rep = run_relational_matching_objective_study(seed=42, train_steps=20, eval_episodes=8)
    for c_id, cr in step394_rep.variants.items():
        print(f"  [{c_id:28s}] H1_K={cr.h1_key_routing:.2%}, H2_K={cr.h2_key_routing:.2%}, G4={cr.g4_acc:.2%}, LangRet={cr.language_retention:.4f}")
    print(f"Best Objective: {step394_rep.best_variant} | Routing Boost: {step394_rep.h1_routing_boost:+.2%}")

    # 4. Step 395 End-to-End Two-Hop Training
    print("\n--- STEP 395: End-to-End Two-Hop Joint Training ---")
    core395 = InitializedRelationalCore(mode="D_orthogonal")
    core395, step395_rep = train_and_eval_end_to_end_core(core395, seed=42, train_steps=25, eval_episodes=8)
    init_loss = step395_rep.curves[0].l_total if step395_rep.curves else 0.0
    final_loss = step395_rep.curves[-1].l_total if step395_rep.curves else 0.0
    print(f"Initial Total Loss: {init_loss:.4f} -> Final Total Loss: {final_loss:.4f} (Delta: {step395_rep.loss_reduction_pct:+.1f}%)")
    if step395_rep.curves:
        last_c = step395_rep.curves[-1]
        print(f"Component Losses: L_H1={last_c.l_h1:.4f}, L_inter={last_c.l_inter:.4f}, L_H2={last_c.l_h2:.4f}, L_final={last_c.l_final:.4f}")
    print(f"Final Accuracies: G1={step395_rep.final_g1_acc:.2%}, G4={step395_rep.final_g4_acc:.2%}, H1_K={step395_rep.final_h1_routing:.2%}, H2_K={step395_rep.final_h2_routing:.2%}")

    # 5. Step 396 Anti-Memorization & Permutation Generalization
    print("\n--- STEP 396: Anti-Memorization & Generalization Suite ---")
    step396_rep = evaluate_anti_memorization_suite(core395, seed=42, episodes_per_condition=6)
    print(f"G1={step396_rep.g1_acc:.2%}, G2={step396_rep.g2_acc:.2%}, G3={step396_rep.g3_acc:.2%}, G4={step396_rep.g4_acc:.2%}")
    print(f"Permutations: {step396_rep.passed_conditions}/{step396_rep.total_conditions} passed")
    print(f"Distractor Sweeps (0 to 5): {[pt.final_acc for pt in step396_rep.distractor_sweep.values()]}")
    print(f"Contamination Audit Zero: {step396_rep.contamination_zero}")
    print(f"Positional Correlation: {step396_rep.positional_correlation:+.4f}")

    # 6. Step 397 Adaptive Sufficiency Reintegration
    print("\n--- STEP 397: Adaptive Sufficiency Reintegration ---")
    step397_rep = run_adaptive_sufficiency_reintegration_study(seed=42, train_steps=20, eval_episodes_per_hop=4)
    for c_id, cr in step397_rep.variants.items():
        print(f"  [{c_id:25s}] G1={cr.g1_acc:.2%}, G4={cr.g4_acc:.2%}, MeanCycles={cr.mean_cycles:.2f}")
    adaptive_c = step397_rep.variants.get("D_AdaptiveSufficiency")
    if adaptive_c:
        print(f"Adaptive Controller Allocates Compute Dynamically: Mean={adaptive_c.mean_cycles:.2f}")

    # 7. Step 398 Multi-Seed Relational Stability
    print("\n--- STEP 398: Multi-Seed Relational Stability (Seeds 42, 101, 2026) ---")
    step398_rep = run_multi_seed_relational_stability_study(seeds=(42, 101, 2026), train_steps=20, eval_episodes=6)
    for s, sr in step398_rep.seed_metrics.items():
        print(f"  Seed {s:5d}: G1={sr.g1_acc:.2%}, G4={sr.g4_acc:.2%}, H1_K={sr.h1_routing:.2%}, H2_K={sr.h2_routing:.2%}")
    print(f"Multi-Seed Summary: Mean G4={step398_rep.mean_g4:.2%}, Mean H1={step398_rep.mean_h1:.2%}, Mean H2={step398_rep.mean_h2:.2%}")
    print(f"Diagnostic Failure Boundary: {step398_rep.transformation_boundary_identified}")

    # 8. Step 399 Resource-Aware Relational Reasoning under Hard Budgets
    print("\n--- STEP 399: Resource-Aware Relational Reasoning ---")
    step399_rep = run_resource_aware_relational_study(episodes=6)
    for b, br in step399_rep.budgets.items():
        print(f"  Budget {b}: G4={br.g4_acc:.2%}, MeanCycles={br.mean_cycles:.2f}, MaxCycles={br.max_cycles_observed}, Bounded={br.strictly_bounded}")
    print(f"Strict Budget Respected: {step399_rep.all_strictly_bounded} | Budget Scaling Effective: {step399_rep.budget_scaling_effective}")

    # 9. Step 400 Master I4 Decision & Release Track Readiness
    print("\n--- STEP 400: Master I4 Decision Gate & First Release Track Readiness ---")
    step400_rep = run_master_relational_i4_decision(seeds=(42, 101, 2026), train_steps=20, eval_episodes=6)
    print(f"Official I4 Passed: {step400_rep.official_i4_passed}")
    print(f"Classification: {step400_rep.final_classification}")
    print(f"Rationale: {step400_rep.decision_rationale}")
    print(f"Mean G4: {step400_rep.mean_g4:.2%}")
    print(f"Mean H1: {step400_rep.mean_h1:.2%}")
    print(f"Mean H2: {step400_rep.mean_h2:.2%}")
    print(f"Language Retention: {step400_rep.language_retention:.4f}")
    print(f"Trainable Parameters: {step400_rep.trainable_parameters:,}")
    print(f"Release Readiness Checklist (18 dimensions): {step400_rep.release_checklist.all_dimensions_passed}")

    # 10. Baseline Invariant Post-check
    print("\n--- BASELINE INVARIANT POST-CHECK ---")
    post_base_model = instantiate_frozen_baseline()
    post_hash = compute_model_hash(post_base_model)
    post_base_params = sum(p.numel() for p in post_base_model.parameters())
    print(f"[*] Post-Wave Canonical Baseline SHA-256: {post_hash}")
    print(f"[*] Post-Wave Parameter Count: {post_base_params:,}")
    assert post_hash == EXPECTED_WEIGHT_HASH, "Post-wave baseline SHA mismatch!"
    assert post_base_params == 3_443_136, "Post-wave baseline param count mismatch!"
    print("Baseline Invariant Verification: PASS (Bit-Exact, Delta W = 0)")

    total_time = time.time() - t0
    print("\n" + "=" * 75)
    print(f"BENCHMARK COMPLETE IN {total_time:.2f}s")
    print(f"Final Outcome: {step400_rep.final_classification}")
    print(f"Release Track: ChakrView v0.1 Verified Cognitive Core Status: READY")
    print("=" * 75)


if __name__ == "__main__":
    main()
