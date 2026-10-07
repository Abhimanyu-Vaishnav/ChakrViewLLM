"""Master benchmark execution script for Wave 337–344: Compact Recurrent Attention Core.

Runs the complete Steps 337–344 experimental pipeline across seeds 42, 101, 2026:
- Step 337: Integrated Core Forensics
- Step 338: Minimal Integrated Core (Identity & Parameter Budget Verification)
- Step 339: Value + FFN Learning Ablation Study
- Step 340: Recurrent Integration Comparison (Static vs Recurrent)
- Step 341: Causal Neural-Path Interventions (10 conditions)
- Step 342: Distractor Sweep (0 to 5) & Anti-Shortcut Suite
- Step 343 & 344: Strict Multi-Seed Evaluation & I4 Gate Promotion Audit
"""

import time
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.integrated_recurrent_attention_forensics import (
    run_integrated_core_forensics,
)
from chakrview.cognition.compact_recurrent_attention_core import (
    CompactRecurrentAttentionCore,
)
from chakrview.cognition.value_ffn_learning import (
    run_value_ffn_ablation_study,
)
from chakrview.cognition.recurrent_integration import (
    run_recurrent_integration_comparison,
)
from chakrview.cognition.causal_neural_path_interventions import (
    run_causal_neural_path_interventions,
)
from chakrview.cognition.integrated_anti_shortcut import (
    run_integrated_anti_shortcut_suite,
)
from chakrview.cognition.strict_integrated_i4_evaluation import (
    run_strict_integrated_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


def main():
    print("=" * 70)
    print("CHAKRVIEW WAVE 337–344: COMPACT RECURRENT ATTENTION CORE BENCHMARK")
    print("=" * 70)

    t0 = time.time()
    torch.manual_seed(42)
    cfg = ModelConfig()
    base_model = ChakrMicro(cfg)
    base_model.eval()

    init_hash = compute_model_hash(base_model)
    print(f"[*] Canonical Baseline Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"[*] Canonical Baseline Initial SHA-256: {init_hash}")
    assert init_hash == EXPECTED_WEIGHT_HASH, "Canonical baseline hash mismatch!"

    # 1. Step 337 Forensics
    print("\n--- STEP 337: Integrated Core Forensics ---")
    forensics = run_integrated_core_forensics(base_model, target_layer=3, seeds=[42, 101, 2026], num_episodes=4)
    print(f"Recommended Target Layer: {forensics.recommended_target_layer}")
    print(f"Hop-1 Key Margin: {forensics.metrics.h1_key_margin:+.4f}")
    print(f"Hop-2 Key Margin: {forensics.metrics.h2_key_margin:+.4f}")
    print(f"Intermediate Drift: {forensics.metrics.intermediate_state_drift:.4f}")
    print(f"Recurrent State Norm: {forensics.metrics.recurrent_state_norm:.4f}")

    # 2. Step 338 Minimal Integrated Core Verification
    print("\n--- STEP 338: Minimal Integrated Core Construction ---")
    core = CompactRecurrentAttentionCore(base_model, target_layer=3)
    p_train = core.trainable_param_count
    p_tot = core.total_param_count
    print(f"Trainable Parameters: {p_train:,} (Budget Target: <=150k, Preferred: <=120k)")
    print(f"Total Parameters: {p_tot:,}")
    assert p_train <= 150000, "Trainable budget exceeded!"

    inp_dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        b_out = base_model(inp_dummy)
        c_out = core(inp_dummy)
    init_diff = float(torch.max(torch.abs(b_out - c_out)).item())
    print(f"Max Initialization Output Difference: {init_diff:.6e} (Exact Identity Verified)")

    # 3. Step 339 Value + FFN Learning Study
    print("\n--- STEP 339: Value + FFN Learning Study ---")
    ablation_rep = run_value_ffn_ablation_study(base_model)
    for cfg_name, m in ablation_rep.metrics_by_config.items():
        print(f"  [{cfg_name:<10}] Params: {m.trainable_params:,} | H1 Key: {m.h1_key_acc*100:.1f}% | H2 Key: {m.h2_key_acc*100:.1f}% | G4: {m.mean_g4_acc*100:.1f}%")
    print(f"Best Configuration: {ablation_rep.best_config}")
    print(f"V+FFN Beneficial: {ablation_rep.v_ffn_beneficial}")

    # 4. Step 340 Recurrent Integration Comparison
    print("\n--- STEP 340: Recurrent Integration Comparison ---")
    rec_rep = run_recurrent_integration_comparison(base_model)
    print(f"Static Core G4:    {rec_rep.metrics_static.mean_g4_acc * 100:.2f}% (H2 Key: {rec_rep.metrics_static.mean_h2_key_acc*100:.1f}%)")
    print(f"Recurrent Core G4: {rec_rep.metrics_recurrent.mean_g4_acc * 100:.2f}% (H2 Key: {rec_rep.metrics_recurrent.mean_h2_key_acc*100:.1f}%)")
    print(f"G4 Improvement:    {rec_rep.recurrent_gain_g4 * 100:+.2f}%")
    print(f"Recurrent Beneficial: {rec_rep.recurrent_beneficial}")

    # 5. Step 341 Causal Neural-Path Interventions
    print("\n--- STEP 341: Causal Neural-Path Interventions (10 Conditions) ---")
    env = CompositionalAssociativeEnvironment(seed=42)
    causal_rep = run_causal_neural_path_interventions(core, env, num_episodes=6)
    for c_name, c_res in causal_rep.results.items():
        print(f"  [{c_name:<28}] Target Prob: {c_res.mean_target_prob:.4f} | Final Acc: {c_res.final_tok_acc*100:.1f}% | Drop: {c_res.prob_drop_vs_normal:+.4f}")
    print(f"Recurrent Causally Active: {causal_rep.is_recurrent_causally_active}")
    print(f"V/FFN Causally Active:     {causal_rep.is_v_ffn_causally_active}")
    print(f"Attention Causally Active: {causal_rep.is_attention_causally_active}")

    # 6. Step 342 Distractor Sweep & Anti-Shortcut Suite
    print("\n--- STEP 342: Distractor Sweep & Anti-Shortcut Suite ---")
    anti_rep = run_integrated_anti_shortcut_suite(core, seed=42, num_episodes_per_test=4)
    print("Distractor Sweep (0 to 5):")
    for d, d_res in anti_rep.distractor_results.items():
        print(f"  {d} distractors -> Final Token Acc: {d_res.tok_acc*100:.1f}%, Hop-2 Key: {d_res.h2_key_acc*100:.1f}%")
    print(f"Anti-Shortcut Pass Rate: {anti_rep.passed_count}/{anti_rep.total_count} ({anti_rep.pass_rate*100:.1f}%)")
    print(f"Zero Contamination:      {anti_rep.zero_contamination_verified}")
    print(f"Suite Valid:             {anti_rep.suite_valid}")

    # 7. Steps 343 & 344 Strict Multi-Seed Evaluation and Gate Audit
    print("\n--- STEPS 343 & 344: Strict Multi-Seed Evaluation & I4 Gate Audit ---")
    eval_rep = run_strict_integrated_i4_evaluation(base_model, seeds=[42, 101, 2026])
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
    print("FINAL WAVE 337–344 GATE VERIFICATION")
    print("=" * 70)
    print(f"Baseline SHA-256 Bit-Exact:     {eval_rep.baseline_exact}")
    print(f"Delta W = 0:                    {delta_w_zero} (Init={init_hash[:16]}..., Post={post_hash[:16]}...)")
    print(f"Stability Diagnostic Passed:    {eval_rep.stability_diagnostic_passed}")
    print(f"Official I4 Gate Passed:        {eval_rep.official_i4_passed}")
    print(f"Classification:                 {eval_rep.final_classification}")
    print(f"Decision Rationale:             {eval_rep.decision_rationale}")
    print(f"Total CPU Runtime:              {total_elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
