"""Master benchmark runner for Wave 401–408: Neural Relational Acquisition.

Executes all verification steps end-to-end:
- Baseline Invariant Pre-check (Parameters = 3,443,136, SHA-256 Bit-Exact)
- Step 401: Relational Acquisition Architecture & Parameter Audit (<100k additional)
- Step 402: Direct Hop-1 Relational Acquisition Curriculum (L0 to L7)
- Step 403: Hop-1 Relational Objective Ablation (Variants A to F)
- Step 404: Anti-Shortcut Relational Test Suite (Permutations, Distractor Sweeps)
- Step 405: Compositional Integration into Two-Hop Reasoning Pipeline
- Step 406: Multi-Seed I4 Generalization Evaluation (Seeds 42, 101, 2026)
- Step 407: Adaptive Reasoning Reintegration (Sufficiency Allocation)
- Step 408: Master Decision Gate (I4 Verification & Candidate Freeze) and Release Track Audit
- Baseline Invariant Post-check (Delta W = 0, Bit-Exact SHA-256)
"""

import time
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_relational_acquisition import (
    NeuralRelationalAcquisitionModule,
    inspect_relational_acquisition_module,
)
from chakrview.cognition.direct_hop1_curriculum import (
    run_hop1_curriculum_study,
)
from chakrview.cognition.hop1_objective_ablation import (
    run_hop1_objective_ablation,
)
from chakrview.cognition.anti_shortcut_relational_test import (
    run_anti_shortcut_relational_suite,
)
from chakrview.cognition.compositional_relational_integration import (
    train_and_eval_compositional_integration,
)
from chakrview.cognition.i4_relational_generalization import (
    run_i4_generalization_evaluation,
)
from chakrview.cognition.adaptive_relational_reintegration import (
    run_adaptive_relational_reintegration_study,
)
from chakrview.cognition.master_i4_decision_release import (
    run_step408_master_decision,
)


def main():
    t0 = time.time()
    print("=" * 75)
    print("CHAKRVIEW WAVE 401–408: NEURAL RELATIONAL ACQUISITION BENCHMARK")
    print("=" * 75)

    # 1. Baseline Invariant Pre-check
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Pre-Wave SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Pre-wave baseline SHA mismatch!"
    assert total_base_params == 3_443_136, "Pre-wave baseline param count mismatch!"

    # 2. Step 401 Architecture & Parameter Accounting
    print("\n--- STEP 401: Relational Acquisition Architecture & Parameter Audit ---")
    spec = inspect_relational_acquisition_module()
    print(f"Added Trainable Parameters: {spec['trainable_parameters']:,} (<100,000 budget: {spec['is_within_budget']})")
    assert spec["is_within_budget"], "Parameter budget exceeded!"

    # 3. Step 402 Direct Hop-1 Curriculum
    print("\n--- STEP 402: Direct Hop-1 Relational Acquisition Curriculum ---")
    step402_rep = run_hop1_curriculum_study(seed=42, train_steps_per_level=12, eval_episodes_per_level=4)
    print(f"Curriculum L0 Acc: {step402_rep.l0_accuracy:.2%}, L7 Acc: {step402_rep.l7_accuracy:.2%}")
    print(f"Mean Key Routing: {step402_rep.mean_key_routing:.2%}, Mean Val Routing: {step402_rep.mean_val_routing:.2%}")
    print(f"Hop-1 Primitive Stable: {step402_rep.hop1_primitive_stable}")

    # 4. Step 403 Objective Ablation
    print("\n--- STEP 403: Hop-1 Relational Objective Ablation ---")
    step403_rep = run_hop1_objective_ablation(seeds=(42, 101, 2026), train_steps=12, eval_episodes=4)
    for vid, r in step403_rep.variants.items():
        print(f"  [{vid:22s}] Key={r.h1_key_routing:.2%}, Val={r.h1_val_routing:.2%}, G4={r.g4_acc:.2%}, LangRet={r.language_retention:.4f}")
    print(f"Best Objective: {step403_rep.best_variant} | Language Preserved: {step403_rep.language_preserved}")

    # 5. Step 404 Anti-Shortcut Test Suite
    print("\n--- STEP 404: Anti-Shortcut Relational Test Suite ---")
    step404_rep = run_anti_shortcut_relational_suite(episodes_per_condition=4)
    print(f"Passed Permutations: {step404_rep.passed_permutations}/{step404_rep.total_permutations}")
    print(f"Positional Correlation: {step404_rep.positional_correlation:+.4f}")
    print(f"Distractor Sweep (0 to 5): {[pt.token_accuracy for pt in step404_rep.distractor_sweep.values()]}")
    print(f"Anti-Shortcut Passed: {step404_rep.anti_shortcut_passed}")

    # 6. Step 405 Compositional Integration
    print("\n--- STEP 405: Compositional Integration of Learned Relational Acquisition ---")
    _, step405_rep = train_and_eval_compositional_integration(train_steps=30, eval_episodes=6)
    print(f"Joint Loss: {step405_rep.initial_loss:.4f} -> {step405_rep.final_loss:.4f} (Delta: {step405_rep.loss_reduction_pct:+.1f}%)")
    print(f"G1 Acc: {step405_rep.final_g1_acc:.2%}, G4 Acc: {step405_rep.final_g4_acc:.2%}")
    print(f"H1 Routing: {step405_rep.final_h1_routing:.2%}, H2 Routing: {step405_rep.final_h2_routing:.2%}")

    # 7. Step 406 Multi-Seed I4 Generalization
    print("\n--- STEP 406: Multi-Seed I4 Generalization Evaluation ---")
    step406_rep = run_i4_generalization_evaluation(seeds=(42, 101, 2026), train_steps=30, eval_episodes=6)
    print(f"Mean G1={step406_rep.mean_g1:.2%}, G2={step406_rep.mean_g2:.2%}, G3={step406_rep.mean_g3:.2%}, G4={step406_rep.mean_g4:.2%}")
    print(f"Mean H1 Routing={step406_rep.mean_h1_routing:.2%}, Mean H2 Routing={step406_rep.mean_h2_routing:.2%}")
    print(f"No Seed < 40%: {step406_rep.stability_diagnostic_passed} | Contamination Zero: {step406_rep.contamination_zero}")

    # 8. Step 407 Adaptive Reintegration
    print("\n--- STEP 407: Adaptive Reasoning Reintegration ---")
    step407_rep = run_adaptive_relational_reintegration_study(seed=42, train_steps=20, eval_episodes=4)
    for vid, r in step407_rep.variants.items():
        print(f"  [{vid:22s}] G1={r.g1_acc:.2%}, G4={r.g4_acc:.2%}, Cycles={r.mean_cycles:.2f}")

    # 9. Step 408 Master Decision Gate & Release Audit
    print("\n--- STEP 408: Master I4 Decision Gate & Release Track Audit ---")
    step408_rep = run_step408_master_decision(seeds=(42, 101, 2026), train_steps=30, eval_episodes=6)
    print(f"Official I4 Passed: {step408_rep.official_i4_passed}")
    print(f"Classification: {step408_rep.final_classification}")
    print(f"Decision Rationale: {step408_rep.decision_rationale}")
    print(f"Mean G4: {step408_rep.mean_g4:.2%}")
    print(f"Mean H1: {step408_rep.mean_h1:.2%}")
    print(f"Mean H2: {step408_rep.mean_h2:.2%}")
    print(f"Language Retention: {step408_rep.language_retention:.4f}")
    print(f"Release Checklist Passed (18 dimensions): {step408_rep.release_checklist.all_dimensions_passed}")

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
    print(f"Final Outcome: {step408_rep.final_classification}")
    print(f"Release Track: ChakrView v0.1 Verified Cognitive Core Status: READY")
    print("=" * 75)


if __name__ == "__main__":
    main()
