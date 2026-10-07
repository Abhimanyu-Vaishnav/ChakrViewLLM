"""Step 328: Master Architecture Decision Benchmark for Wave 321-328.

Consolidates all empirical evidence from Wave 321-328:
1. Canonical baseline immutability audit (3,443,136 params, hash c5571c..., Delta W = 0)
2. Step 321: Integrated state-transition forensics (margin degradation across layers)
3. Step 322: Minimal recurrent state-transition mechanism (GRU-like gate, zero-op identity initialization)
4. Step 323: State integration with attention comparison (Configurations A, B, C)
5. Step 324: Two-step recurrent composition & 5 causal interventions
6. Step 325: Distractor-robust state transition across 0, 1, 2, 3, 5 distractors
7. Step 326: Compositional generalization & 14-condition anti-shortcut audit
8. Step 327: Strict multi-seed I3 preservation & I4 evaluation across seeds 42, 101, 2026
9. Three-way architecture comparison
10. Final capability classification and architectural decision
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.recurrent_state_transition import (
    GatedRecurrentStateTransition,
    compute_module_sha256,
)
from chakrview.cognition.integrated_state_transition_forensics import (
    run_integrated_state_transition_forensics,
    IntegratedForensicsReport,
)
from chakrview.cognition.state_attention_integration import (
    StateAttentionIntegrationModel,
    run_state_integration_comparison,
    StateIntegrationComparisonReport,
)
from chakrview.cognition.recurrent_causal_interventions import (
    run_recurrent_causal_interventions,
    RecurrentCausalResults,
)
from chakrview.cognition.distractor_robust_state import (
    evaluate_distractor_robust_state,
    DistractorRobustStateReport,
)
from chakrview.cognition.recurrent_generalization_suite import (
    run_recurrent_anti_shortcut_suite,
    RecurrentAntiShortcutReport,
)
from chakrview.cognition.strict_recurrent_i4_evaluation import (
    run_strict_recurrent_i3_i4_evaluation,
    train_recurrent_core_candidate,
    StrictRecurrentI4Report,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)

EXPECTED_BASELINE_PARAMS = 3443136


@dataclasses.dataclass
class MasterWave328Report:
    baseline_integrity: bool
    baseline_params: int
    baseline_sha: str
    baseline_delta_w: int
    
    candidate_params: int
    candidate_trainable: int
    recurrent_core_params: int
    recurrent_core_sha256: str
    
    forensic_best_layer: int
    forensic_best_margin: float
    
    best_integration_mode: str
    best_integration_g4: float
    
    causal_normal_acc: float
    causal_zeroed_acc: float
    causal_corrupted_acc: float
    causal_swapped_acc: float
    causal_bypass_acc: float
    state_causally_active: bool
    
    distractor_resilient: bool
    distractor_mean_purity: float
    
    anti_shortcut_conditions_checked: int
    anti_shortcut_passed: bool
    
    i3_preserved: bool
    i3_uu_tok_acc: float
    language_retention_ratio: float
    
    g1_mean_acc: float
    g2_mean_acc: float
    g3_mean_acc: float
    g4_mean_acc: float
    per_seed_g4: Dict[int, float]
    
    i4_gate_passed: bool
    stability_diagnostic_passed: bool
    final_classification: str
    architecture_decision: str
    cpu_runtime_ms: float


def run_wave328_master_benchmark(
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> MasterWave328Report:
    """Executes the master benchmark for Wave 321-328."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t_start = time.perf_counter()

    # 1. Baseline verification
    base_model = instantiate_frozen_baseline()
    base_params = sum(p.numel() for p in base_model.parameters())
    base_sha = compute_model_hash(base_model)
    baseline_ok = (base_params == EXPECTED_BASELINE_PARAMS and base_sha == EXPECTED_WEIGHT_HASH)
    assert baseline_ok, f"Baseline verification failed: {base_params} params, sha={base_sha}"

    # 2. Candidate parameter footprint
    sample_model = StateAttentionIntegrationModel(base_model, integration_mode="A", d_state=64)
    rec_core_params = sum(p.numel() for p in sample_model.transition_core.parameters() if p.requires_grad)
    cand_trainable = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    cand_total = base_params + cand_trainable
    core_sha = compute_module_sha256(sample_model.transition_core)

    # 3. Step 321 Forensics
    forensic_rep = run_integrated_state_transition_forensics(base_model, seeds=seeds, num_episodes=eval_episodes)

    # 4. Step 323 Integration Comparison
    integ_rep = run_state_integration_comparison(base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes)

    # 5. Train model for causal & distractor tests
    trained_model, _ = train_recurrent_core_candidate(
        base_model, seed=42, train_steps=train_steps, eval_episodes=eval_episodes, integration_mode="A", use_binding=True,
    )
    env = CompositionalAssociativeEnvironment(seed=42)

    # 6. Step 324 Causal Interventions
    causal_rep = run_recurrent_causal_interventions(trained_model, env, num_episodes=eval_episodes, seed=42)

    # 7. Step 325 Distractor Resistance
    dist_rep = evaluate_distractor_robust_state(trained_model, env=env, seed=42, episodes_per_scale=eval_episodes)

    # 8. Step 326 Anti-Shortcut Suite
    anti_rep = run_recurrent_anti_shortcut_suite(trained_model, env=env, seed=42, episodes_per_cond=eval_episodes)

    # 9. Step 327 Strict Multi-Seed I3 & I4 Evaluation
    strict_rep = run_strict_recurrent_i3_i4_evaluation(
        base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes,
    )

    # Post-run baseline verification
    post_sha = compute_model_hash(base_model)
    delta_w = 0 if post_sha == EXPECTED_WEIGHT_HASH else 1
    assert delta_w == 0, "Canonical baseline modified!"

    t_end = time.perf_counter()
    cpu_ms = (t_end - t_start) * 1000.0

    per_seed_g4 = {s: strict_rep.per_seed_results[s].g4_tok_acc for s in seeds}

    # Architectural Decision
    if strict_rep.final_classification == "I4_ACHIEVED":
        decision = (
            "Freeze Integrated Recurrent Neural Core as candidate architecture. "
            "Integrated state transitions successfully achieve multi-hop compositional reasoning (>50% G4)."
        )
    elif strict_rep.final_classification == "I4_EMERGING":
        decision = (
            "Integrated Recurrent Core demonstrates improved query modulation and distractor stability, "
            "but multi-seed variance remains above the stability gate. Continue refining state conditioning."
        )
    else:
        decision = (
            "Integrated Recurrent State Transition does not satisfy I4 promotion. "
            "As mandated by the stop condition, transition to modifying the attention mechanism itself or training internal core weights."
        )

    return MasterWave328Report(
        baseline_integrity=baseline_ok,
        baseline_params=base_params,
        baseline_sha=base_sha,
        baseline_delta_w=delta_w,
        candidate_params=cand_total,
        candidate_trainable=cand_trainable,
        recurrent_core_params=rec_core_params,
        recurrent_core_sha256=core_sha,
        forensic_best_layer=forensic_rep.best_insertion_layer,
        forensic_best_margin=forensic_rep.per_layer_metrics[forensic_rep.best_insertion_layer].key_margin,
        best_integration_mode=integ_rep.best_mode,
        best_integration_g4=integ_rep.best_g4_acc,
        causal_normal_acc=causal_rep.normal_tok_acc,
        causal_zeroed_acc=causal_rep.zeroed_tok_acc,
        causal_corrupted_acc=causal_rep.corrupted_tok_acc,
        causal_swapped_acc=causal_rep.swapped_tok_acc,
        causal_bypass_acc=causal_rep.bypass_tok_acc,
        state_causally_active=causal_rep.state_causally_active,
        distractor_resilient=dist_rep.distractor_resilient,
        distractor_mean_purity=dist_rep.mean_state_purity,
        anti_shortcut_conditions_checked=len(anti_rep.conditions),
        anti_shortcut_passed=not anti_rep.shortcuts_detected,
        i3_preserved=strict_rep.i3_preserved,
        i3_uu_tok_acc=strict_rep.i3_uu_tok_acc,
        language_retention_ratio=strict_rep.language_retention_ratio,
        g1_mean_acc=strict_rep.mean_g1_tok_acc,
        g2_mean_acc=strict_rep.mean_g2_tok_acc,
        g3_mean_acc=strict_rep.mean_g3_tok_acc,
        g4_mean_acc=strict_rep.mean_g4_tok_acc,
        per_seed_g4=per_seed_g4,
        i4_gate_passed=strict_rep.i4_gate_passed,
        stability_diagnostic_passed=strict_rep.stability_diagnostic_passed,
        final_classification=strict_rep.final_classification,
        architecture_decision=decision,
        cpu_runtime_ms=cpu_ms,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING WAVE 321-328 MASTER BENCHMARK: INTEGRATED RECURRENT NEURAL CORE")
    print("=" * 70)
    rep = run_wave328_master_benchmark(train_steps=12, eval_episodes=6)
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Baseline Params: {rep.baseline_params} | SHA-256: {rep.baseline_sha[:16]}... | Delta W: {rep.baseline_delta_w}")
    print(f"Candidate Trainable Params: {rep.candidate_trainable} (Recurrent Core: {rep.recurrent_core_params})")
    print(f"Forensic Best Insertion: Layer {rep.forensic_best_layer} (Margin: {rep.forensic_best_margin:+.4f})")
    print(f"Best Integration Mode: Mode {rep.best_integration_mode} (G4 = {rep.best_integration_g4*100:.1f}%)")
    print(f"Causal Interventions: Normal={rep.causal_normal_acc*100:.1f}%, Zeroed={rep.causal_zeroed_acc*100:.1f}%, Corrupt={rep.causal_corrupted_acc*100:.1f}%, Swap={rep.causal_swapped_acc*100:.1f}%, Bypass={rep.causal_bypass_acc*100:.1f}%")
    print(f"State Causally Active: {rep.state_causally_active}")
    print(f"Distractor Resilience: {rep.distractor_resilient} (Mean Purity: {rep.distractor_mean_purity:.3f})")
    print(f"Anti-Shortcut: {rep.anti_shortcut_conditions_checked} conditions audited, Passed: {rep.anti_shortcut_passed}")
    print(f"I3 Preserved: {rep.i3_preserved} (UU Tok = {rep.i3_uu_tok_acc*100:.1f}%), Language Retention: {rep.language_retention_ratio:.5f}")
    print(f"G1={rep.g1_mean_acc*100:.1f}%, G2={rep.g2_mean_acc*100:.1f}%, G3={rep.g3_mean_acc*100:.1f}%, G4={rep.g4_mean_acc*100:.1f}%")
    print(f"Per-seed G4: {rep.per_seed_g4}")
    print(f"I4 Gate Passed: {rep.i4_gate_passed} | Stability Diag Passed: {rep.stability_diagnostic_passed}")
    print(f"Classification: {rep.final_classification}")
    print(f"Architecture Decision: {rep.architecture_decision}")
    print(f"CPU Runtime: {rep.cpu_runtime_ms:.1f} ms")
    print("=" * 70)
