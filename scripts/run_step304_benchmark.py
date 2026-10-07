"""Step 304: Master Decision & Architecture Gate Benchmark (Wave 297-304).

Consolidates all empirical findings for Wave 297-304:
1. Canonical baseline verification (params = 3,443,136, hash = c5571c..., Delta W = 0)
2. Candidate parameter count and module SHA-256
3. Step 297 forensic attractor baseline analysis
4. Step 298 minimal learned associative attractor codebook implementation
5. Step 299 associative attractor training study with ablation
6. Step 300 attractor -> Hop-2 query integration comparison (Variants A, B, C, D)
7. Step 301 attractor robustness & 14-condition anti-shortcut testing
8. Step 302 I3 preservation & language retention audit
9. Step 303 strict multi-seed I4 evaluation across seeds 42, 101, 2026
10. Final I4 capability classification & concrete architecture decision
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
from chakrview.cognition.learned_associative_attractor import (
    LearnedAssociativeAttractor,
    compute_module_sha256,
)
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.forensic_attractor_baseline import (
    run_attractor_forensic_baseline,
    AttractorForensicReport,
)
from chakrview.cognition.associative_attractor_training import (
    run_associative_attractor_training_study,
    train_attractor_compositional_model,
    AttractorTrainingReport,
)
from chakrview.cognition.attractor_query_integration import (
    run_attractor_query_integration_study,
    AttractorIntegrationReport,
)
from chakrview.cognition.attractor_anti_shortcut import (
    run_attractor_anti_shortcut_suite,
    AttractorAntiShortcutReport,
)
from chakrview.cognition.attractor_i3_preservation import (
    evaluate_i3_preservation_and_language_retention,
    I3PreservationReport,
)
from chakrview.cognition.strict_attractor_i4_evaluation import (
    run_strict_attractor_i4_evaluation,
    StrictAttractorI4Report,
)

EXPECTED_BASELINE_PARAMS = 3443136


@dataclasses.dataclass
class MasterWave304Report:
    baseline_integrity: bool
    baseline_params: int
    baseline_sha: str
    baseline_delta_w: int
    
    candidate_params: int
    candidate_trainable: int
    attractor_params: int
    attractor_sha256: str
    
    forensic_theoretically_plausible: bool
    forensic_between_within_ratio: float
    
    training_benefit_delta: float
    training_utilization: float
    
    best_integration_variant: str
    best_integration_g4: float
    
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


def run_wave304_master_benchmark(
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> MasterWave304Report:
    """Executes the master benchmark for Wave 297-304."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t_start = time.perf_counter()

    # 1. Canonical baseline audit
    base_model = instantiate_frozen_baseline()
    base_params = sum(p.numel() for p in base_model.parameters())
    base_sha = compute_model_hash(base_model)
    baseline_ok = (base_params == EXPECTED_BASELINE_PARAMS and base_sha == EXPECTED_WEIGHT_HASH)
    assert baseline_ok, f"Baseline integrity check failed: {base_params} params, sha={base_sha}"

    # 2. Candidate & Attractor Parameter Footprint
    sample_cand = ChakrMicroCompositionalReasoningModel(base_model, rank=16)
    sample_att = LearnedAssociativeAttractor(d_model=sample_cand.config.d_model, num_attractors=16, d_attractor=64)
    att_params = sum(p.numel() for p in sample_att.parameters() if p.requires_grad)
    cand_trainable = sum(p.numel() for p in sample_cand.parameters() if p.requires_grad) + att_params
    cand_total = base_params + cand_trainable
    att_sha = compute_module_sha256(sample_att)

    # 3. Step 297 Forensic Attractor Baseline
    forensic_rep = run_attractor_forensic_baseline(
        base_model, seeds=seeds, num_episodes_per_split=eval_episodes, train_steps=train_steps,
    )

    # 4. Step 299 Training Study
    train_rep = run_associative_attractor_training_study(
        base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes,
    )

    # 5. Step 300 Attractor Query Integration Study
    integ_rep = run_attractor_query_integration_study(
        base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes,
    )

    # 6. Step 301 Anti-Shortcut Suite
    trained_cand, trained_att, _ = train_attractor_compositional_model(
        base_model, seed=42, train_steps=train_steps, eval_episodes=eval_episodes, use_attractor_loss=True,
    )
    anti_rep = run_attractor_anti_shortcut_suite(
        trained_cand, trained_att, seed=42, episodes_per_cond=eval_episodes,
    )

    # 7. Step 302 I3 Preservation & Language Retention
    i3_rep = evaluate_i3_preservation_and_language_retention(
        base_model, trained_cand, trained_att, seed=42, num_episodes=eval_episodes,
    )

    # 8. Step 303 Strict Multi-Seed I4 Evaluation
    strict_rep = run_strict_attractor_i4_evaluation(
        base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes,
    )

    # Verify baseline immutability post evaluation
    post_sha = compute_model_hash(base_model)
    delta_w = 0 if post_sha == EXPECTED_WEIGHT_HASH else 1
    assert delta_w == 0, "Canonical baseline modified during evaluation!"

    t_end = time.perf_counter()
    cpu_ms = (t_end - t_start) * 1000.0

    per_seed_g4 = {s: strict_rep.per_seed_results[s].g4_tok_acc for s in seeds}

    # Final architecture decision rule
    if strict_rep.final_classification == "I4_ACHIEVED":
        arch_decision = "Freeze candidate architecture with learned associative attractor. Proceed to formal milestone freeze."
    elif strict_rep.final_classification == "I4_EMERGING":
        arch_decision = "Associative attractor yields incremental intermediate stability, but continuous bridge remains noisy. Evaluate trainable compositional backbone or explicit relational memory next."
    else:
        arch_decision = "Attractor codebook on frozen representations does not resolve cross-hop binding breakdown. Recommendation: Transition from frozen-adapter bridge to trainable compositional backbone or recurrent relational state space."

    return MasterWave304Report(
        baseline_integrity=baseline_ok,
        baseline_params=base_params,
        baseline_sha=base_sha,
        baseline_delta_w=delta_w,
        candidate_params=cand_total,
        candidate_trainable=cand_trainable,
        attractor_params=att_params,
        attractor_sha256=att_sha,
        forensic_theoretically_plausible=forensic_rep.attractor_theoretically_plausible,
        forensic_between_within_ratio=forensic_rep.overall_ratio_between_within,
        training_benefit_delta=train_rep.attractor_loss_benefit,
        training_utilization=sum(m.mean_attractor_utilization for m in train_rep.full_attractor_metrics) / len(train_rep.full_attractor_metrics),
        best_integration_variant=integ_rep.best_variant,
        best_integration_g4=integ_rep.best_g4_tok_acc,
        anti_shortcut_conditions_checked=len(anti_rep.conditions),
        anti_shortcut_passed=not anti_rep.shortcuts_detected,
        i3_preserved=i3_rep.i3_preserved,
        i3_uu_tok_acc=i3_rep.i3_uu_tok_acc,
        language_retention_ratio=i3_rep.language_retention_ratio,
        g1_mean_acc=strict_rep.mean_g1_tok_acc,
        g2_mean_acc=strict_rep.mean_g2_tok_acc,
        g3_mean_acc=strict_rep.mean_g3_tok_acc,
        g4_mean_acc=strict_rep.mean_g4_tok_acc,
        per_seed_g4=per_seed_g4,
        i4_gate_passed=strict_rep.i4_gate_passed,
        stability_diagnostic_passed=strict_rep.stability_diagnostic_passed,
        final_classification=strict_rep.final_classification,
        architecture_decision=arch_decision,
        cpu_runtime_ms=cpu_ms,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING WAVE 297-304 MASTER BENCHMARK: LEARNED ASSOCIATIVE ATTRACTOR")
    print("=" * 70)
    rep = run_wave304_master_benchmark(train_steps=12, eval_episodes=6)
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Baseline Params: {rep.baseline_params} | SHA-256: {rep.baseline_sha[:16]}... | Delta W: {rep.baseline_delta_w}")
    print(f"Candidate Trainable Params: {rep.candidate_trainable} (Attractor: {rep.attractor_params})")
    print(f"Forensic Theoretical Plausibility: {rep.forensic_theoretically_plausible} (Ratio: {rep.forensic_between_within_ratio:.2f})")
    print(f"Best Integration Variant: Variant {rep.best_integration_variant} (G4 = {rep.best_integration_g4*100:.1f}%)")
    print(f"Anti-Shortcut: {rep.anti_shortcut_conditions_checked} conditions audited, Passed: {rep.anti_shortcut_passed}")
    print(f"I3 Preserved: {rep.i3_preserved} (UU Tok = {rep.i3_uu_tok_acc*100:.1f}%), Language Retention: {rep.language_retention_ratio:.5f}")
    print(f"G1={rep.g1_mean_acc*100:.1f}%, G2={rep.g2_mean_acc*100:.1f}%, G3={rep.g3_mean_acc*100:.1f}%, G4={rep.g4_mean_acc*100:.1f}%")
    print(f"Per-seed G4: {rep.per_seed_g4}")
    print(f"I4 Gate Passed: {rep.i4_gate_passed} | Stability Diag Passed: {rep.stability_diagnostic_passed}")
    print(f"Classification: {rep.final_classification}")
    print(f"Architecture Decision: {rep.architecture_decision}")
    print(f"CPU Runtime: {rep.cpu_runtime_ms:.1f} ms")
    print("=" * 70)
