"""Master benchmark execution script for Wave 329–336: Trainable Compositional Attention.

Runs the complete Steps 329–336 experimental pipeline across seeds 42, 101, 2026:
- Step 329: Attention Mechanism Forensics
- Step 330: Q-Only low-rank adaptation
- Step 331: K-Only low-rank adaptation
- Step 332: Q+K low-rank adaptation
- Step 333: Dedicated Relational Attention Head
- Step 334: Causal Attention Interventions (8 conditions)
- Step 335: Anti-Shortcut & Generalization Suite (16 conditions)
- Step 336: Strict Multi-Seed I3/I4 Promotion Gate & Architecture Decision
"""

import time
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.compositional_attention_forensics import (
    run_compositional_attention_forensics,
)
from chakrview.cognition.trainable_attention_subset import (
    CompositionalAttentionSubsetModel,
)
from chakrview.cognition.dedicated_relational_head import (
    ChakrMicroWithDedicatedRelationalHead,
)
from chakrview.cognition.causal_attention_interventions import (
    run_causal_attention_interventions,
)
from chakrview.cognition.attention_anti_shortcut import (
    run_attention_anti_shortcut_suite,
)
from chakrview.cognition.strict_attention_i4_evaluation import (
    run_strict_attention_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


def main():
    print("=" * 70)
    print("CHAKRVIEW WAVE 329–336: TRAINABLE COMPOSITIONAL ATTENTION BENCHMARK")
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

    # 1. Step 329 Forensics
    print("\n--- STEP 329: Attention Mechanism Forensics ---")
    forensics = run_compositional_attention_forensics(base_model, seeds=[42, 101, 2026], num_episodes=4)
    print(f"Optimal Head: Layer {forensics.best_layer}, Head {forensics.best_head}")
    print(f"Highest Natural Key Discrimination Margin: {forensics.best_margin:+.4f}")
    print(f"Forensics Summary: {forensics.summary}")

    # 2. Step 336 Multi-Seed Evaluation of all 5 architectures (Baseline, Q-only, K-only, QK, RelHead)
    print("\n--- STEPS 330-333 & 336: Multi-Seed Architecture Evaluations ---")
    eval_report = run_strict_attention_i4_evaluation(base_model, seeds=[42, 101, 2026])

    for name, res in eval_report.arch_results.items():
        print(f"\nArchitecture: [{name}]")
        print(f"  Trainable Parameters: {res.trainable_params:,} (Total: {res.total_params:,})")
        print(f"  G1 (Known ID / Known Comp): {res.mean_g1 * 100:.2f}%")
        print(f"  G2 (Unseen ID / Known Comp): {res.mean_g2 * 100:.2f}%")
        print(f"  G3 (Known ID / Unseen Comp): {res.mean_g3 * 100:.2f}%")
        print(f"  G4 (Unseen ID / Unseen Comp): {res.mean_g4 * 100:.2f}%")
        print(f"  Per-Seed G4: { {s: f'{score * 100:.1f}%' for s, score in res.seed_g4_scores.items()} }")
        print(f"  Hop-1 Key / Hop-2 Key: {res.mean_h1_key * 100:.2f}% / {res.mean_h2_key * 100:.2f}%")
        print(f"  I3 Single-Hop UU Binding: {res.i3_uu_acc * 100:.2f}%")
        print(f"  Language Retention: {res.language_retention:.4f}")
        print(f"  Gate Passed: {res.passed_i4_gate} (Stability: {res.stability_passed})")

    # 3. Step 334 Causal Interventions on Best Candidate
    print("\n--- STEP 334: Causal Attention Interventions on Best Candidate ---")
    best_name = eval_report.best_architecture
    print(f"Evaluating best architecture: {best_name}")
    if best_name == "rel_head":
        best_cand = ChakrMicroWithDedicatedRelationalHead(base_model, target_layer=3, head_dim=32)
    elif best_name == "qk":
        best_cand = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="qk", rank=16)
    elif best_name == "q_only":
        best_cand = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="q_only", rank=16)
    else:
        best_cand = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="k_only", rank=16)

    env = CompositionalAssociativeEnvironment(seed=42)
    episodes = [env.generate_episode("disjoint_test", num_distractors=1, episode_idx=2500000 + i) for i in range(8)]
    causal_rep = run_causal_attention_interventions(best_cand, episodes)
    for cond, r in causal_rep.results.items():
        print(f"  Condition [{cond:<20}]: Target Prob={r.target_tok_prob:.4f}, Accuracy={r.tok_acc * 100:.1f}%, Drop={r.prob_drop_vs_normal:+.4f}")
    print(f"  Causally Active: {causal_rep.is_causally_active} (Mean prob drop = {causal_rep.mean_prob_drop_under_disruption:+.4f})")

    # 4. Step 335 Anti-Shortcut Suite
    print("\n--- STEP 335: Anti-Shortcut & Generalization Suite ---")
    anti_shortcut = run_attention_anti_shortcut_suite(best_cand, seed=42, num_episodes_per_cond=4)
    print(f"  Passed Conditions: {anti_shortcut.passed_count}/{anti_shortcut.total_count} ({anti_shortcut.pass_rate * 100:.1f}%)")
    print(f"  Zero Contamination: {anti_shortcut.zero_contamination_verified}")
    print(f"  Suite Valid: {anti_shortcut.anti_shortcut_suite_valid}")

    # 5. Final Invariants and Decision
    post_hash = compute_model_hash(base_model)
    delta_w_zero = (post_hash == init_hash)
    total_elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("FINAL WAVE 329–336 GATE VERIFICATION")
    print("=" * 70)
    print(f"Baseline SHA-256 Bit-Exact: {eval_report.base_sha256_exact}")
    print(f"Delta W = 0: {delta_w_zero} (Init={init_hash[:16]}..., Post={post_hash[:16]}...)")
    print(f"Official I4 Gate Passed: {eval_report.official_i4_passed}")
    print(f"Architecture Decision: {eval_report.architecture_decision}")
    print(f"Decision Rationale: {eval_report.decision_rationale}")
    print(f"Total CPU Runtime: {total_elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
