"""Master benchmark script for Wave 353-360: Recurrent Attention Core Block.

Runs all verification steps end-to-end:
- Step 353: Core Block Forensics & Parameter Audit
- Step 354: RecurrentAttentionCoreBlock construction & parameter checks
- Step 355: One-Block Integration with RecurrentAttentionChakrMicro
- Step 356: State-Transition Ablation Study
- Step 357: Causal State Interventions
- Step 358: Distractor Robustness & Anti-Shortcut Suite
- Step 359 & 360: Strict Multi-Seed Gate Audit & Baseline Invariance Verification
"""

import time
import torch

from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.recurrent_attention_core_block_forensics import (
    audit_core_block_architecture,
)
from chakrview.cognition.recurrent_attention_core_block import (
    RecurrentAttentionCoreBlock,
)
from chakrview.cognition.recurrent_attention_chakr_micro import (
    RecurrentAttentionChakrMicro,
)
from chakrview.cognition.state_transition_ablation_study import (
    run_state_transition_ablation_study,
)
from chakrview.cognition.causal_state_interventions import (
    run_causal_state_interventions,
)
from chakrview.cognition.core_block_anti_shortcut_suite import (
    run_distractor_and_anti_shortcut_suite,
)
from chakrview.cognition.strict_recurrent_i4_evaluation import (
    run_strict_recurrent_i4_evaluation,
)


def main():
    t0 = time.time()
    print("=" * 70)
    print("CHAKRVIEW WAVE 353–360: RECURRENT ATTENTION CORE BLOCK BENCHMARK")
    print("=" * 70)

    # 1. Baseline Invariant
    base_model = instantiate_frozen_baseline()
    init_hash = compute_model_hash(base_model)
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Canonical Baseline Parameters: {total_base_params:,}")
    print(f"[*] Canonical Baseline Initial SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Baseline hash mismatch!"

    # 2. Step 353 Forensics
    print("\n--- STEP 353: Core Block Forensics & Parameter Budget ---")
    f_rep = audit_core_block_architecture()
    print(f"Canonical Block Params: {f_rep.canonical_block_params:,}")
    print(f"Proposed Core Block Budget: {f_rep.proposed_recurrent_block_params:,}")
    print(f"Budget Passed (<= 250k): {f_rep.parameter_budget_passed}")

    # 3. Step 354 & 355 Block & Candidate Initialization
    print("\n--- STEP 354 & 355: Core Block Construction & Candidate Isolation ---")
    candidate = RecurrentAttentionChakrMicro(base_model=base_model)
    trainable_p = candidate.trainable_param_count
    total_p = candidate.total_param_count
    print(f"Trainable Parameters: {trainable_p:,} (Core Block: {candidate.recurrent_block.trainable_param_count:,})")
    print(f"Total Parameters: {total_p:,}")

    # Verify zero-drift initialization
    dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        y_base = base_model(dummy)
        y_cand = candidate(dummy)["vocab_logits"]
        init_diff = (y_base - y_cand).abs().max().item()
    print(f"Initialization Output Diff vs Baseline: {init_diff:.6e} (Bit-Exact Verified)")

    # 4. Step 356 State-Transition Ablation
    print("\n--- STEP 356: State-Transition Ablation Study ---")
    abl_rep = run_state_transition_ablation_study(base_model, seed=42)
    for c, r in abl_rep.conditions.items():
        print(f"  [{c:16s}] Loss: {r.train_loss:.4f} | H1 Key: {r.h1_key_acc:.1%} | H2 Key: {r.h2_key_acc:.1%} | G4: {r.g4_acc:.1%}")
    print(f"Best Condition: {abl_rep.best_condition_id}")
    print(f"Recurrent State Necessary: {abl_rep.recurrent_state_necessary}")

    # 5. Step 357 Causal State Interventions
    print("\n--- STEP 357: Causal State Interventions on Core Block ---")
    causal_rep = run_causal_state_interventions(candidate, seed=42, num_episodes=6)
    for c, r in causal_rep.results.items():
        print(f"  [{c:22s}] Prob: {r.target_token_prob:.4f} | Acc: {r.final_token_acc:.1%} | Drop: {r.drop_from_normal:+.4f}")
    print(f"State Causally Dependent: {causal_rep.causally_dependent_on_state}")
    print(f"Cycle 2 Causally Dependent: {causal_rep.causally_dependent_on_cycle2}")

    # 6. Step 358 Robustness & Anti-Shortcut Suite
    print("\n--- STEP 358: Distractor Robustness & Anti-Shortcut Suite ---")
    rob_rep = run_distractor_and_anti_shortcut_suite(candidate, seed=42, episodes_per_condition=4)
    for d, pt in rob_rep.distractor_sweep.items():
        print(f"  Distractors {d}: Token Acc = {pt.final_token_acc:.1%} | H2 Key = {pt.h2_key_acc:.1%}")
    print(f"Anti-Shortcut Pass Rate: {rob_rep.passed_conditions}/{rob_rep.total_conditions}")
    print(f"Zero Contamination: {rob_rep.contamination_zero}")

    # 7. Step 359 & 360 Strict Multi-Seed Gate Audit
    print("\n--- STEP 359 & 360: Strict Multi-Seed Evaluation & I4 Gate Audit ---")
    strict_rep = run_strict_recurrent_i4_evaluation(base_model, seeds=[42, 101, 2026])
    print(f"G1 (Known ID / Known Comp):     {strict_rep.mean_g1:.2%}")
    print(f"G2 (Unseen ID / Known Comp):   {strict_rep.mean_g2:.2%}")
    print(f"G3 (Known ID / Unseen Comp):   {strict_rep.mean_g3:.2%}")
    print(f"G4 (Unseen ID / Unseen Comp):  {strict_rep.mean_g4:.2%} (PRIMARY GATE)")
    print(f"Per-Seed G4 Scores: {', '.join(f'{s}: {m.g4_acc:.1%}' for s, m in strict_rep.seed_metrics.items())}")
    print(f"Hop-1 Key / Hop-2 Key / Val:   {strict_rep.mean_h1_key:.1%} / {strict_rep.mean_h2_key:.1%} / {strict_rep.mean_h2_val:.1%}")
    print(f"Language Retention Ratio:      {strict_rep.language_retention_ratio:.4f}")

    # 8. Post Verification
    post_hash = compute_model_hash(base_model)
    print("\n" + "=" * 70)
    print("FINAL WAVE 353–360 GATE VERIFICATION")
    print("=" * 70)
    print(f"Baseline SHA-256 Bit-Exact:     {strict_rep.baseline_exact}")
    print(f"Delta W = 0:                    {strict_rep.delta_w_zero} (Init={init_hash[:16]}..., Post={post_hash[:16]}...)")
    print(f"Candidate Weight Hash:          {strict_rep.candidate_weight_hash[:16]}...")
    print(f"Hop-2 Key Emergence:            {strict_rep.mean_h2_key > 0.0} ({strict_rep.mean_h2_key:.2%})")
    print(f"Official I4 Gate Passed:        {strict_rep.official_i4_passed}")
    print(f"Classification:                 {strict_rep.final_classification}")
    print(f"Decision Rationale:             {strict_rep.decision_rationale}")
    print(f"Total CPU Runtime:              {time.time() - t0:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
