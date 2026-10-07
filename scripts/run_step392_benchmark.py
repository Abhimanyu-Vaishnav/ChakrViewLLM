"""Master benchmark runner for Wave 385–392: Need-Based Adaptive Reasoning.

Executes all verification steps end-to-end:
- Baseline Invariant Pre-check (Parameters = 3,443,136, SHA-256 Bit-Exact)
- Step 385: Sufficiency-Based Controller Architecture & Parameter Audit (<550k budget, <50k added)
- Step 386: Lazy-Halt Ablation Study (Variants A through E)
- Step 387: Counterfactual Continuation Test (Path A vs Path B)
- Step 388: Difficulty Generalization (1-hop through 4-hop)
- Step 389: Strict Compositional Generalization Suite & Permutations
- Step 390: Multi-Seed Controller Stability Investigation (Seeds 42, 101, 2026 forensics)
- Step 391: Resource-Aware Adaptive Reasoning Study (Budgets 1 to 8)
- Step 392: Master Decision Gate across Strict Seeds (42, 101, 2026)
- Baseline Invariant Post-check (Delta W = 0)
"""

import time
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.sufficiency_adaptive_reasoning import (
    SufficiencyAdaptiveReasoningCore,
    inspect_sufficiency_core,
)
from chakrview.cognition.lazy_halt_ablation import (
    run_lazy_halt_ablation_study,
    train_and_eval_sufficiency_core,
)
from chakrview.cognition.counterfactual_continuation import run_counterfactual_continuation_test
from chakrview.cognition.difficulty_generalization import run_difficulty_generalization_study
from chakrview.cognition.sufficiency_generalization_suite import evaluate_sufficiency_strict_generalization
from chakrview.cognition.multi_seed_stability_study import run_multi_seed_controller_stability_study
from chakrview.cognition.resource_aware_reasoning import run_resource_aware_reasoning_study
from chakrview.cognition.sufficiency_master_decision_gate import run_strict_sufficiency_i4_evaluation


def main():
    t0 = time.time()
    print("=" * 75)
    print("CHAKRVIEW WAVE 385–392: NEED-BASED ADAPTIVE REASONING BENCHMARK")
    print("=" * 75)

    # 1. Baseline Invariant Pre-check
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Pre-Wave SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Pre-wave baseline SHA mismatch!"
    assert total_base_params == 3_443_136, "Pre-wave baseline param count mismatch!"

    # 2. Step 385 Architecture & Parameter Budget
    print("\n--- STEP 385: Sufficiency-Based Controller Architecture & Budget Audit ---")
    spec = inspect_sufficiency_core()
    trainable_p = spec["trainable_parameters"]
    total_p = spec["total_parameters"]
    ctrl_p = spec["controller_parameters"]
    print(f"Trainable Parameters: {trainable_p:,}")
    print(f"Total Parameters: {total_p:,}")
    print(f"Added Sufficiency Controller Parameters: {ctrl_p:,}")
    assert trainable_p < 550_000, f"Trainable params {trainable_p} exceeds 550k budget!"
    assert ctrl_p < 50_000, f"Added controller params {ctrl_p} exceeds 50k target!"
    print("Parameter budget audit: PASS (<550k trainable, <50k controller)")

    # 3. Step 386 Lazy-Halt Ablation Study
    print("\n--- STEP 386: Lazy-Halt Ablation Study ---")
    step386_rep = run_lazy_halt_ablation_study(seed=42)
    for v_id, vr in step386_rep.variants.items():
        print(f"  [{v_id:25s}] G4={vr.g4_acc:.2%}, Acc={vr.overall_accuracy:.2%}, 1hC={vr.mean_cycles_1hop:.1f}, 2hC={vr.mean_cycles_2hop:.1f}")
    print(f"Best Variant: {step386_rep.best_variant} | Lazy-Halt Alleviated: {step386_rep.lazy_halt_alleviated}")

    # 4. Step 387 Counterfactual Continuation Test
    print("\n--- STEP 387: Counterfactual Continuation Test ---")
    core = SufficiencyAdaptiveReasoningCore()
    train_and_eval_sufficiency_core(core, seed=42, train_steps=15, eval_episodes_per_hop=4)
    step387_rep = run_counterfactual_continuation_test(core=core, seed=42, episodes_per_hop=4)
    print(f"Benefit 1h={step387_rep.mean_benefit_1hop:+.4f}, 2h={step387_rep.mean_benefit_2hop:+.4f}, 3h={step387_rep.mean_benefit_3hop:+.4f}")
    print(f"Correlation Continue with Benefit: {step387_rep.correlation_continue_with_benefit:+.4f}")
    print(f"Alignment Accuracy: {step387_rep.alignment_accuracy:.2%}")

    # 5. Step 388 Difficulty Generalization
    print("\n--- STEP 388: Difficulty Generalization ---")
    step388_rep = run_difficulty_generalization_study(core=core, seed=42, eval_episodes_per_level=4)
    for d, lr in step388_rep.levels.items():
        print(f"  {d}-Hop: Acc={lr.accuracy:.2%}, MeanC={lr.mean_cycles:.2f}, Premature={lr.premature_halt_rate:.2%}")
    print(f"Compute Scales With Difficulty: {step388_rep.compute_scales_with_difficulty}")

    # 6. Step 389 Strict Compositional Generalization Suite
    print("\n--- STEP 389: Strict Compositional Generalization Suite ---")
    step389_rep = evaluate_sufficiency_strict_generalization(core, seed=42, episodes_per_condition=4)
    print(f"G1={step389_rep.g1_acc:.2%}, G2={step389_rep.g2_acc:.2%}, G3={step389_rep.g3_acc:.2%}, G4={step389_rep.g4_acc:.2%}")
    print(f"Passed Permutations: {step389_rep.passed_conditions}/{step389_rep.total_conditions}")
    print(f"Positional Correlation: {step389_rep.positional_correlation:+.4f}")
    print(f"Contamination Zero: {step389_rep.contamination_zero}")

    # 7. Step 390 Multi-Seed Controller Stability Investigation
    print("\n--- STEP 390: Multi-Seed Controller Stability Investigation ---")
    step390_rep = run_multi_seed_controller_stability_study(seeds=(42, 101, 2026), train_steps=15, eval_episodes=4)
    for s, sm in step390_rep.seed_metrics.items():
        print(f"  Seed {s:5d}: G4={sm.g4_acc:.2%}, MeanC={sm.mean_cycles:.2f}, SuffScore={sm.mean_sufficiency_score:.4f}, Diag: {sm.seed2026_root_cause}")
    print(f"Mean G4: {step390_rep.mean_g4:.2%}")

    # 8. Step 391 Resource-Aware Adaptive Reasoning Study
    print("\n--- STEP 391: Resource-Aware Adaptive Reasoning Study ---")
    step391_rep = run_resource_aware_reasoning_study(core=core, seed=42, episodes=4)
    for b, pt in step391_rep.budgets.items():
        print(f"  Budget {b}: G4={pt.g4_acc:.2%}, MeanC={pt.mean_cycles:.2f}, MaxObs={pt.max_cycles_observed}, Bounded={pt.strictly_bounded}")
    print(f"All Strictly Bounded: {step391_rep.all_strictly_bounded} | Non-Greedy: {step391_rep.non_greedy_consumption_proven}")

    # 9. Step 392 Master Decision Gate across Seeds (42, 101, 2026)
    print("\n--- STEP 392: Master Decision Gate across Seeds (42, 101, 2026) ---")
    step392_rep = run_strict_sufficiency_i4_evaluation(seeds=(42, 101, 2026), train_steps=20, eval_episodes=6)
    print(f"Final Classification: {step392_rep.final_classification}")
    print(f"Decision Rationale: {step392_rep.decision_rationale}")
    print(f"Mean G4: {step392_rep.mean_g4:.2%}")
    print(f"Mean H2: {step392_rep.mean_h2:.2%}")
    print(f"Language Retention: {step392_rep.language_retention:.4f}")
    print(f"Trainable Parameters: {step392_rep.trainable_parameters:,}")
    print(f"Baseline Exact: {step392_rep.baseline_exact}")

    # 10. Baseline Post-check
    post_hash = compute_model_hash(base_model)
    print(f"\n[*] Canonical Baseline Post-Wave SHA-256: {post_hash}")
    assert post_hash == EXPECTED_WEIGHT_HASH, "Post-wave baseline SHA mismatch!"
    assert init_hash == post_hash, "Baseline altered during benchmarking!"
    print("[*] Delta W = 0 verified. Canonical baseline is strictly bit-exact.")

    dt = time.time() - t0
    print("\n" + "=" * 75)
    print(f"WAVE 385–392 BENCHMARK COMPLETE (Total CPU Runtime: {dt:.2f}s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
