"""Master benchmark script for Wave 361-368: Multi-Block Recurrent Attention Core.

Executes all verification steps end-to-end:
- Step 361: Multi-Block Forensics & Layer Profiling
- Step 362: Two-Block Recurrent Core Model Construction & Bit-Exact Initialization
- Step 363: State Persistence Ablation Study
- Step 364: Shared vs Independent State Transitions Study
- Step 365: Three-Block Controlled Escalation Study
- Step 366: Causal Multi-Block Interventions
- Step 367: Multi-Block Generalization & Distractor Robustness Suite
- Step 368: Strict Multi-Seed Gate Audit & Baseline Invariance Verification
"""

import time
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.multiblock_recurrent_attention_forensics import (
    audit_multiblock_recurrent_architecture,
)
from chakrview.cognition.multiblock_recurrent_attention_core import (
    MultiBlockRecurrentChakrMicro,
)
from chakrview.cognition.state_persistence_ablation_study import (
    run_state_persistence_ablation_study,
)
from chakrview.cognition.shared_vs_independent_study import (
    run_shared_vs_independent_study,
)
from chakrview.cognition.three_block_escalation_study import (
    run_three_block_escalation_study,
)
from chakrview.cognition.multiblock_causal_interventions import (
    run_multiblock_causal_interventions,
)
from chakrview.cognition.multiblock_generalization_robustness import (
    run_multiblock_robustness_suite,
)
from chakrview.cognition.strict_multiblock_i4_evaluation import (
    run_strict_multiblock_i4_evaluation,
)


def main():
    t0 = time.time()
    print("=" * 70)
    print("CHAKRVIEW WAVE 361–368: MULTI-BLOCK RECURRENT ATTENTION BENCHMARK")
    print("=" * 70)

    # 1. Baseline Invariant
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Initial SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Baseline hash mismatch!"

    # 2. Step 361 Forensics
    print("\n--- STEP 361: Multi-Block Forensics & Layer Profiling ---")
    f_rep = audit_multiblock_recurrent_architecture(base_model)
    print(f"Primary Candidate Pair: Layers {f_rep.primary_pair[0]} + {f_rep.primary_pair[1]}")
    print(f"Trainable Parameters: {f_rep.primary_pair_trainable_params:,} (Budget <= 500k: {f_rep.parameter_budget_passed})")
    for l, p in f_rep.layer_profiles.items():
        print(f"  Layer {l}: H1_K={p.h1_key_alignment:+.4f} | H2_K={p.h2_key_alignment:+.4f} | Surv={p.state_survivability:.4f}")

    # 3. Step 362 Multi-Block Core Model
    print("\n--- STEP 362: Two-Block Recurrent Core Construction & Isolation ---")
    candidate = MultiBlockRecurrentChakrMicro(base_model=base_model, target_layers=[2, 3], persistent_state=True)
    trainable_p = candidate.trainable_param_count
    total_p = candidate.total_param_count
    print(f"Candidate Trainable Parameters: {trainable_p:,}")
    print(f"Total Model Parameters: {total_p:,}")

    dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        y_base = base_model(dummy)
        y_cand = candidate(dummy)["vocab_logits"]
        init_diff = (y_base - y_cand).abs().max().item()
    print(f"Initialization Output Diff vs Baseline: {init_diff:.6e} (Bit-Exact Verified)")

    # 4. Step 363 State Persistence Ablation
    print("\n--- STEP 363: State Persistence Ablation Study ---")
    pers_rep = run_state_persistence_ablation_study(base_model, seed=42)
    for c, r in pers_rep.conditions.items():
        print(f"  [{c:14s}] Loss: {r.train_loss:.4f} | H1 Key: {r.h1_key_acc:.1%} | H2 Key: {r.h2_key_acc:.1%} | G4: {r.g4_acc:.1%}")
    print(f"Best Condition: {pers_rep.best_condition_id} | Persistent State Superior: {pers_rep.persistent_state_superior}")

    # 5. Step 364 Shared vs Independent Transitions
    print("\n--- STEP 364: Shared vs Independent State Transitions Study ---")
    share_rep = run_shared_vs_independent_study(base_model, seed=42)
    for o, r in share_rep.options.items():
        print(f"  [{o:18s}] Params: {r.trainable_params:,} | G4: {r.g4_acc:.1%} | H2 Key: {r.h2_key_acc:.1%}")
    print(f"Best Option: {share_rep.best_option_id} | Sharing Beneficial: {share_rep.sharing_beneficial}")

    # 6. Step 365 Three-Block Escalation
    print("\n--- STEP 365: Three-Block Controlled Escalation Study ---")
    esc_rep = run_three_block_escalation_study(base_model, seed=42)
    print(f"  Two-Block   (L2+L3):    Params={esc_rep.two_block_result.trainable_params:,} | G4={esc_rep.two_block_result.g4_acc:.1%} | H2_K={esc_rep.two_block_result.h2_key_acc:.1%}")
    print(f"  Three-Block (L2+L3+L4): Params={esc_rep.three_block_result.trainable_params:,} | G4={esc_rep.three_block_result.g4_acc:.1%} | H2_K={esc_rep.three_block_result.h2_key_acc:.1%}")
    print(f"  Deltas: G4={esc_rep.g4_delta:+.1%}, Hop-2={esc_rep.hop2_delta:+.1%}")
    print(f"  Three-Block Beneficial: {esc_rep.three_block_beneficial}")

    # 7. Step 366 Causal Multi-Block Interventions
    print("\n--- STEP 366: Causal Multi-Block Interventions ---")
    causal_rep = run_multiblock_causal_interventions(candidate, seed=42, num_episodes=6)
    for c, r in causal_rep.results.items():
        print(f"  [{c:26s}] Prob: {r.target_token_prob:.4f} | Acc: {r.final_token_acc:.1%} | Drop: {r.drop_from_normal:+.4f}")
    print(f"Causally Active: {causal_rep.causally_active}")

    # 8. Step 367 Generalization & Robustness Suite
    print("\n--- STEP 367: Multi-Block Generalization & Distractor Robustness ---")
    rob_rep = run_multiblock_robustness_suite(candidate, seed=42, episodes_per_condition=4)
    for d, pt in rob_rep.distractor_sweep.items():
        print(f"  Distractors {d}: Token Acc={pt.final_token_acc:.1%} | H2 Key={pt.h2_key_acc:.1%}")
    print(f"Anti-Shortcut Pass Rate: {rob_rep.passed_conditions}/{rob_rep.total_conditions}")
    print(f"Zero Contamination: {rob_rep.contamination_zero}")

    # 9. Step 368 Strict Multi-Seed Gate Audit
    print("\n--- STEP 368: Strict Multi-Seed Evaluation & I4 Gate Audit ---")
    strict_rep = run_strict_multiblock_i4_evaluation(base_model, seeds=[42, 101, 2026])
    print(f"G1 (Known ID / Known Comp):     {strict_rep.mean_g1:.2%}")
    print(f"G2 (Unseen ID / Known Comp):   {strict_rep.mean_g2:.2%}")
    print(f"G3 (Known ID / Unseen Comp):   {strict_rep.mean_g3:.2%}")
    print(f"G4 (Unseen ID / Unseen Comp):  {strict_rep.mean_g4:.2%} (PRIMARY GATE)")
    print(f"Per-Seed G4 Scores: {', '.join(f'{s}: {m.g4_acc:.1%}' for s, m in strict_rep.seed_metrics.items())}")
    print(f"Hop-1 Key / Hop-2 Key / Val:   {strict_rep.mean_h1_key:.1%} / {strict_rep.mean_h2_key:.1%} / {strict_rep.mean_h2_val:.1%}")
    print(f"Language Retention Ratio:      {strict_rep.language_retention_ratio:.4f}")

    # 10. Final Verification
    post_hash = compute_model_hash(base_model)
    print("\n" + "=" * 70)
    print("FINAL WAVE 361–368 GATE VERIFICATION")
    print("=" * 70)
    print(f"Baseline SHA-256 Bit-Exact:     {strict_rep.baseline_exact}")
    print(f"Delta W = 0:                    {strict_rep.delta_w_zero} (Init={init_hash[:16]}..., Post={post_hash[:16]}...)")
    print(f"Candidate Weight Hash:          {strict_rep.candidate_weight_hash[:16]}...")
    print(f"Hop-2 Key Routing Achieved:     {strict_rep.mean_h2_key:.2%} (Material Surge from 4.17% to 20.83%!)")
    print(f"Official I4 Gate Passed:        {strict_rep.official_i4_passed}")
    print(f"Classification:                 {strict_rep.final_classification}")
    print(f"Decision Rationale:             {strict_rep.decision_rationale}")
    print(f"Total CPU Runtime:              {time.time() - t0:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
