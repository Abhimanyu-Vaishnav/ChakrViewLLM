"""Master benchmark runner for Wave 369–376: Unified Compositional Neural Core.

Executes all verification steps end-to-end:
- Baseline Invariant Pre-check
- Step 369: Unified Compositional Core Architecture & Parameter Audit (<500k budget)
- Step 370: Minimal Two-Hop Learnability & All 8 Intermediate Metrics
- Step 371: Causal State Interventions (7 conditions)
- Step 372: Compositional Curriculum Training (Levels 0 through 8)
- Step 373: Core Architectural Ablations (A through G)
- Step 374: Unified Generalization Stress & Anti-Shortcut Suite
- Step 375: Three-Hop Composition Escalation Diagnostic
- Step 376: Master Decision Gate across strict seeds (42, 101, 2026)
- Baseline Invariant Post-check
"""

import time
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.unified_compositional_core import UnifiedCompositionalCore
from chakrview.cognition.minimal_two_hop_learnability import train_and_eval_two_hop_core
from chakrview.cognition.causal_state_test import run_causal_state_interventions_on_core
from chakrview.cognition.compositional_curriculum_training import train_compositional_curriculum
from chakrview.cognition.core_ablations_study import run_core_ablations_study
from chakrview.cognition.unified_generalization_stress import evaluate_unified_generalization_stress
from chakrview.cognition.three_hop_escalation_diagnostic import run_three_hop_escalation_diagnostic
from chakrview.cognition.strict_unified_core_i4_evaluation import run_strict_unified_core_i4_evaluation


def main():
    t0 = time.time()
    print("=" * 75)
    print("CHAKRVIEW WAVE 369–376: UNIFIED COMPOSITIONAL NEURAL CORE BENCHMARK")
    print("=" * 75)

    # 1. Baseline Invariant Pre-check
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Pre-Wave SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Pre-wave baseline SHA mismatch!"

    # 2. Step 369 Architecture & Parameters
    print("\n--- STEP 369: Unified Compositional Core Architecture ---")
    core = UnifiedCompositionalCore()
    trainable_p = core.trainable_param_count
    total_p = core.total_param_count
    print(f"Trainable Parameters: {trainable_p:,}")
    print(f"Total Parameters: {total_p:,}")
    assert trainable_p < 500_000, f"Trainable params {trainable_p} exceeds 500k preferred budget!"
    assert total_p < 1_000_000, f"Total params {total_p} exceeds 1M hard limit!"
    print("Param budget audit: PASS (<500k preferred, <1M hard limit)")

    # 3. Step 370 Minimal Two-Hop Learnability
    print("\n--- STEP 370: Minimal Two-Hop Learnability & Intermediate Metrics ---")
    step370_rep = train_and_eval_two_hop_core(seed=42, train_steps=20, eval_episodes=6)
    print(f"Mean G4 (Unseen/Unseen): {step370_rep.mean_g4:.2%}")
    print(f"Hop-1 Key Routing: {step370_rep.mean_h1_k:.2%}")
    print(f"Hop-2 Key Routing: {step370_rep.mean_h2_k:.2%}")
    print(f"Step 370 Summary: {step370_rep.summary}")

    # 4. Step 371 Causal State Test
    print("\n--- STEP 371: Causal State Interventions ---")
    causal_rep = run_causal_state_interventions_on_core(seed=42, num_episodes=6)
    for c_id, cr in causal_rep.results.items():
        print(f"  [{c_id:12s}] Acc={cr.final_token_acc:.2%}, Prob={cr.target_token_prob:.4f}, H2_k={cr.h2_key_acc:.2%}")
    print(f"Causal State Required: {causal_rep.state_causally_dependent}")
    print(f"Mean Degradation: {causal_rep.mean_degradation:.4f}")

    # 5. Step 372 Compositional Curriculum
    print("\n--- STEP 372: Compositional Curriculum Training (9 Levels) ---")
    cur_rep = train_compositional_curriculum(core=core, seed=42, steps_per_level=4, eval_episodes=6)
    for l_id, lr in cur_rep.level_results.items():
        print(f"  Level {l_id} ({lr.level_name}): Loss={lr.loss:.4f}, G4={lr.g4_acc:.2%}, H2_k={lr.h2_key_acc:.2%}")
    print(f"All Levels Promoted: {cur_rep.all_promoted}, Contamination Zero: {cur_rep.contamination_zero}")

    # 6. Step 373 Core Ablations
    print("\n--- STEP 373: Core Architectural Ablations ---")
    abl_rep = run_core_ablations_study(seed=42)
    for v_id, vr in abl_rep.variants.items():
        print(f"  [{v_id:16s}] Loss={vr.train_loss:.4f}, G4={vr.g4_acc:.2%}, H2_k={vr.h2_key_acc:.2%}")
    print(f"Recurrence Causally Essential: {abl_rep.recurrent_state_contributes}")
    print(f"Best Variant: {abl_rep.best_variant_id}")

    # 7. Step 374 Generalization Stress
    print("\n--- STEP 374: Generalization Stress & Anti-Shortcut Suite ---")
    stress_rep = evaluate_unified_generalization_stress(core, seed=42, episodes_per_condition=6)
    print(f"Stress Passed Conditions: {stress_rep.passed_conditions}/{stress_rep.total_conditions}")
    print(f"Positional Correlation: {stress_rep.positional_correlation:.4f}")
    print(f"Distractor 0 Acc: {stress_rep.distractor_sweep[0].final_token_acc:.2%}")
    print(f"Distractor 5 Acc: {stress_rep.distractor_sweep[5].final_token_acc:.2%}")
    print(f"Stress Summary: {stress_rep.summary}")

    # 8. Step 375 Three-Hop Escalation Diagnostic
    print("\n--- STEP 375: Three-Hop Escalation Diagnostic ---")
    three_rep = run_three_hop_escalation_diagnostic(core, seed=42, episodes_per_split=6)
    print(f"Mean H1: {three_rep.mean_h1:.2%}, Mean H2: {three_rep.mean_h2:.2%}, Mean H3: {three_rep.mean_h3:.2%}")
    print(f"Final 3-Hop Acc: {three_rep.mean_final_acc:.2%}")
    print(f"Degradation Boundary: {three_rep.degradation_boundary}")

    # 9. Step 376 Master Decision Gate
    print("\n--- STEP 376: Master Decision Gate across Seeds (42, 101, 2026) ---")
    gate_rep = run_strict_unified_core_i4_evaluation(seeds=(42, 101, 2026), steps_per_level=4, eval_episodes=6)
    print(f"Final Classification: {gate_rep.final_classification}")
    print(f"Decision Rationale: {gate_rep.decision_rationale}")
    print(f"Mean G4: {gate_rep.mean_g4:.2%}")
    print(f"Mean H2 Key: {gate_rep.mean_h2_key:.2%}")
    print(f"Language Retention: {gate_rep.language_retention:.4f}")
    print(f"Trainable Parameters: {gate_rep.trainable_parameters:,}")
    print(f"Baseline Exact: {gate_rep.baseline_exact}")

    # 10. Baseline Post-check
    post_hash = compute_model_hash(base_model)
    print(f"\n[*] Canonical Baseline Post-Wave SHA-256: {post_hash}")
    assert post_hash == EXPECTED_WEIGHT_HASH, "Post-wave baseline SHA mismatch!"
    assert init_hash == post_hash, "Baseline altered during benchmarking!"
    print("[*] Delta W = 0 verified. Canonical baseline is strictly bit-exact.")

    dt = time.time() - t0
    print("\n" + "=" * 75)
    print(f"WAVE 369–376 BENCHMARK COMPLETE (Total CPU Runtime: {dt:.2f}s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
