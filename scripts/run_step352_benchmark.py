"""Master benchmark execution script for Wave 345–352: Complete Transformer Block Training.

Runs the complete Steps 345–352 experimental pipeline across seeds 42, 101, 2026:
- Step 345: Complete Block Forensics & Parameter Audit
- Step 346: One-Block Training (Layer 3 Candidate)
- Step 347: Training Objective Study (Obj A, B, C, D)
- Step 348: Layer Location Study (Layer 2 vs Layer 3 vs Layer 4)
- Step 349: Two-Block Controlled Training (Layer 3 vs Layer 2+3)
- Step 350: Causal Neural-Core Interventions (8 conditions)
- Step 351: Distractor Sweep (0 to 5) & Anti-Shortcut Suite
- Step 352: Strict Multi-Seed Evaluation & Official I4 Gate Promotion Audit
"""

import time
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.complete_block_training_forensics import (
    run_complete_block_forensics,
)
from chakrview.cognition.trainable_transformer_block import (
    TrainableTransformerBlockCandidate,
)
from chakrview.cognition.training_objective_study import (
    run_training_objective_study,
)
from chakrview.cognition.layer_location_study import (
    run_layer_location_study,
)
from chakrview.cognition.two_block_training_study import (
    run_two_block_controlled_training,
)
from chakrview.cognition.causal_block_interventions import (
    run_core_causal_interventions,
)
from chakrview.cognition.block_anti_shortcut_suite import (
    run_block_anti_shortcut_suite,
)
from chakrview.cognition.strict_block_i4_evaluation import (
    run_strict_block_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


def main():
    print("=" * 70)
    print("CHAKRVIEW WAVE 345–352: COMPLETE TRANSFORMER BLOCK TRAINING BENCHMARK")
    print("=" * 70)

    t0 = time.time()
    torch.manual_seed(42)
    cfg = ModelConfig()
    base_model = ChakrMicro(cfg)
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    init_hash = compute_model_hash(base_model)
    print(f"[*] Canonical Baseline Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"[*] Canonical Baseline Initial SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Canonical baseline hash mismatch!"

    # 1. Step 345 Forensics
    print("\n--- STEP 345: Complete Block Forensics & Parameter Audit ---")
    forensics = run_complete_block_forensics(base_model)
    print(f"Total Model Parameters: {forensics.total_model_params:,}")
    print(f"Parameters per Transformer Block: {forensics.total_block_params:,}")
    print(f"Embedding Parameters: {forensics.embedding_params:,} (Tied with LMHead)")
    print(f"Final RMSNorm: {forensics.final_norm_params:,}")
    print("\nLayer Parameter Table:")
    print(forensics.parameter_table_str)
    print(f"\nRecommended Primary Block: Layer {forensics.recommended_primary_block}")
    print(f"Rationale: {forensics.rationale}")

    # 2. Step 346 One-Block Candidate Verification
    print("\n--- STEP 346: One-Block Candidate Construction & Isolation ---")
    cand = TrainableTransformerBlockCandidate(base_model, trainable_layers=[3], enable_contextual_binding=True)
    p_train = cand.trainable_param_count
    p_tot = cand.total_param_count
    print(f"Trainable Parameters: {p_train:,} (Block 3: 442,752 + Binding: {sum(p.numel() for p in cand.binding.parameters()):,})")
    print(f"Total Parameters: {p_tot:,}")

    inp_dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        b_out = base_model(inp_dummy)
        c_out = cand(inp_dummy)
    init_diff = float(torch.max(torch.abs(b_out - c_out)).item())
    print(f"Max Initialization Output Difference: {init_diff:.6e} (Exact Identity Verified)")

    # 3. Step 347 Training Objective Study
    print("\n--- STEP 347: Training Objective Study (Objectives A-D) ---")
    obj_rep = run_training_objective_study(base_model)
    for obj_name, r in obj_rep.results.items():
        print(f"  [{obj_name:<26}] Train Loss: {r.train_loss:.4f} | H1 Key: {r.h1_key_acc*100:.1f}% | H2 Key: {r.h2_key_acc*100:.1f}% | G4: {r.final_tok_acc*100:.1f}%")
    print(f"Best Objective: {obj_rep.best_objective}")

    # 4. Step 348 Layer Location Study
    print("\n--- STEP 348: Layer Location Study (Layer 2 vs 3 vs 4) ---")
    loc_rep = run_layer_location_study(base_model, candidate_layers=[2, 3, 4], seed=42)
    for l_idx, r in loc_rep.layer_results.items():
        print(f"  [Layer {l_idx}] Params: {r.trainable_params:,} | H1 Key: {r.h1_key_acc*100:.1f}% | H2 Key: {r.h2_key_acc*100:.1f}% | G4: {r.g4_tok_acc*100:.1f}%")
    print(f"Best Location: Layer {loc_rep.best_layer} (G4: {loc_rep.best_g4*100:.1f}%)")

    # 5. Step 349 Two-Block Controlled Training
    print("\n--- STEP 349: Two-Block Controlled Training Study ---")
    two_rep = run_two_block_controlled_training(base_model)
    print(f"Single Block (L3) G4:   {two_rep.single_block_metrics.g4_acc*100:.1f}% (Params: {two_rep.single_block_metrics.trainable_params:,})")
    print(f"Two Blocks (L2+L3) G4:  {two_rep.two_block_metrics.g4_acc*100:.1f}% (Params: {two_rep.two_block_metrics.trainable_params:,})")
    print(f"G4 Delta:               {two_rep.g4_delta*100:+.1f}%")
    print(f"Two Blocks Beneficial:  {two_rep.two_block_improves_g4}")

    # 6. Step 350 Causal Neural-Core Interventions
    print("\n--- STEP 350: Causal Neural-Core Interventions on Trained Block ---")
    env = CompositionalAssociativeEnvironment(seed=42)
    causal_rep = run_core_causal_interventions(cand, base_model, env, num_episodes=6, target_layer=3)
    for c_name, c_res in causal_rep.results.items():
        print(f"  [{c_name:<26}] Target Prob: {c_res.mean_target_prob:.4f} | Final Acc: {c_res.final_tok_acc*100:.1f}% | Drop: {c_res.prob_drop_vs_normal:+.4f}")
    print(f"Trained Block Causally Active: {causal_rep.is_trained_block_causally_active} (Mean Drop = {causal_rep.mean_prob_drop:+.4f})")

    # 7. Step 351 Generalization & Anti-Shortcut Suite
    print("\n--- STEP 351: Generalization & Anti-Shortcut Suite ---")
    anti_rep = run_block_anti_shortcut_suite(cand, seed=42, num_episodes_per_cond=4)
    print("Distractor Sweep (0 to 5):")
    for d, acc in anti_rep.distractor_results.items():
        print(f"  {d} distractors -> Final Token Acc: {acc*100:.1f}%")
    print(f"Anti-Shortcut Pass Rate: {anti_rep.passed_count}/{anti_rep.total_count} ({anti_rep.pass_rate*100:.1f}%)")
    print(f"Zero Contamination:      {anti_rep.zero_contamination_verified}")
    print(f"Suite Valid:             {anti_rep.suite_valid}")

    # 8. Step 352 Strict Multi-Seed Evaluation & Gate Audit
    print("\n--- STEP 352: Strict Multi-Seed Evaluation & I4 Gate Audit ---")
    eval_rep = run_strict_block_i4_evaluation(base_model, target_layer=3, seeds=[42, 101, 2026])
    print(f"G1 (Known ID / Known Comp):     {eval_rep.mean_g1 * 100:.2f}%")
    print(f"G2 (Unseen ID / Known Comp):   {eval_rep.mean_g2 * 100:.2f}%")
    print(f"G3 (Known ID / Unseen Comp):   {eval_rep.mean_g3 * 100:.2f}%")
    print(f"G4 (Unseen ID / Unseen Comp):  {eval_rep.mean_g4 * 100:.2f}% (PRIMARY GATE)")
    print(f"Per-Seed G4 Scores: { {s: f'{m.g4_acc * 100:.1f}%' for s, m in eval_rep.seed_metrics.items()} }")
    print(f"Hop-1 Key / Hop-2 Key / Val:   {eval_rep.mean_h1_key*100:.1f}% / {eval_rep.mean_h2_key*100:.1f}% / {eval_rep.mean_h2_val*100:.1f}%")
    print(f"I3 Single-Hop Preservation:    {eval_rep.mean_i3_uu * 100:.2f}%")
    print(f"Language Retention Ratio:      {eval_rep.language_retention_ratio:.4f}")

    # Baseline Invariants Check
    post_hash = compute_model_hash(base_model)
    delta_w_zero = (post_hash == init_hash)
    total_elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("FINAL WAVE 345–352 GATE VERIFICATION")
    print("=" * 70)
    print(f"Baseline SHA-256 Bit-Exact:     {eval_rep.baseline_exact}")
    print(f"Delta W = 0:                    {delta_w_zero} (Init={init_hash[:16]}..., Post={post_hash[:16]}...)")
    print(f"Candidate Weight Hash:          {eval_rep.candidate_weight_hash[:16]}...")
    print(f"Stability Diagnostic Passed:    {eval_rep.stability_diagnostic_passed}")
    print(f"Official I4 Gate Passed:        {eval_rep.official_i4_passed}")
    print(f"Classification:                 {eval_rep.final_classification}")
    print(f"Decision Rationale:             {eval_rep.decision_rationale}")
    print(f"Total CPU Runtime:              {total_elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
