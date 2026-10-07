"""Step 312: Master Architecture Decision Benchmark for Wave 305-312.

Consolidates all empirical evidence from Wave 305-312:
1. Canonical baseline immutability audit (3,443,136 params, hash c5571c..., Delta W = 0)
2. Step 305 Internal Composition Forensics (layer-wise separability, margin degradation)
3. Step 306 Minimal Internal Trainable Pathway (parameter bound, no-op initialization)
4. Step 307 Internal Location Comparison (Configurations A, B, C, D)
5. Step 308 Internal Pathway + Dynamic Contextual Binding Integration (Modes A, B, C, D)
6. Step 309 Controlled Multi-Stage Training Objective & Ablation
7. Step 310 Anti-Shortcut & Generalization Auditing across 14 stress conditions
8. Step 311 Strict I3 Preservation and Multi-Seed I4 Evaluation (Seeds 42, 101, 2026)
9. Final capability classification (I4_ACHIEVED, I4_EMERGING, I4_NOT_ACHIEVED)
10. Explicit architecture-level decision rule and recommendation
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
from chakrview.cognition.internal_trainable_pathway import (
    ChakrMicroWithInternalPathway,
    compute_module_sha256,
)
from chakrview.cognition.internal_composition_forensics import (
    run_internal_composition_forensics,
    InternalCompositionForensicsReport,
)
from chakrview.cognition.internal_location_comparison import (
    run_internal_location_comparison,
    InternalLocationComparisonReport,
)
from chakrview.cognition.internal_integration_study import (
    run_internal_integration_study,
    InternalIntegrationReport,
)
from chakrview.cognition.pathway_training_objective import (
    run_compositional_training_objective_study,
    train_pathway_model,
    PathwayTrainingObjectiveReport,
)
from chakrview.cognition.pathway_anti_shortcut import (
    run_pathway_anti_shortcut_suite,
    PathwayAntiShortcutReport,
)
from chakrview.cognition.strict_pathway_i4_evaluation import (
    run_strict_pathway_i3_i4_evaluation,
    StrictPathwayEvaluationReport,
)

EXPECTED_BASELINE_PARAMS = 3443136


@dataclasses.dataclass
class MasterWave312Report:
    baseline_integrity: bool
    baseline_params: int
    baseline_sha: str
    baseline_delta_w: int
    
    candidate_params: int
    candidate_trainable: int
    pathway_params: int
    pathway_sha256: str
    
    forensic_critical_layer: int
    forensic_failure_mechanism: str
    
    best_internal_location: str
    best_location_g4: float
    
    best_integration_mode: str
    best_integration_g4: float
    
    objective_hop2_benefit: float
    
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


def run_wave312_master_benchmark(
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> MasterWave312Report:
    """Executes the master benchmark for Wave 305-312."""
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
    sample_pathway = ChakrMicroWithInternalPathway(base_model, target_layers=[3, 4, 5], bottleneck_dim=32)
    path_params = sum(p.numel() for p in sample_pathway.parameters() if p.requires_grad)
    cand_trainable = path_params + 25345  # pathway + dynamic binding
    cand_total = base_params + cand_trainable
    path_sha = compute_module_sha256(sample_pathway.adapters)

    # 3. Step 305 Forensics
    forensic_rep = run_internal_composition_forensics(base_model, seeds=seeds, num_episodes=eval_episodes)

    # 4. Step 307 Location Comparison
    loc_rep = run_internal_location_comparison(base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes)

    # 5. Step 308 Integration Study
    integ_rep = run_internal_integration_study(base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes)

    # 6. Step 309 Objective Study
    obj_rep = run_compositional_training_objective_study(base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes)

    # 7. Step 310 Anti-Shortcut Suite
    trained_model, _ = train_pathway_model(base_model, seed=42, train_steps=train_steps, eval_episodes=eval_episodes, use_hop2_objective=True)
    anti_rep = run_pathway_anti_shortcut_suite(trained_model, seed=42, episodes_per_cond=eval_episodes)

    # 8. Step 311 Strict I3 & I4 Evaluation
    strict_rep = run_strict_pathway_i3_i4_evaluation(base_model, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes)

    # Post-run baseline verification
    post_sha = compute_model_hash(base_model)
    delta_w = 0 if post_sha == EXPECTED_WEIGHT_HASH else 1
    assert delta_w == 0, "Canonical baseline modified!"

    t_end = time.perf_counter()
    cpu_ms = (t_end - t_start) * 1000.0

    per_seed_g4 = {s: strict_rep.per_seed_results[s].g4_tok_acc for s in seeds}

    # Architectural Decision
    if strict_rep.final_classification == "I4_ACHIEVED":
        decision = "Freeze internal trainable pathway architecture as successful candidate."
    elif strict_rep.final_classification == "I4_EMERGING":
        decision = "Internal bottleneck pathway provides localized transformation capacity, but single-stream residual updates remain noisy under multi-pair interference. Explore explicit recurrent/relational memory circuit next."
    else:
        decision = "Small internal bottleneck adapters inside frozen transformer layers do not resolve multi-hop relational collapse. As mandated by the stop condition, transition next architecture wave to EXPLICIT RECURRENT / STATE-SPACE RELATIONAL MEMORY."

    return MasterWave312Report(
        baseline_integrity=baseline_ok,
        baseline_params=base_params,
        baseline_sha=base_sha,
        baseline_delta_w=delta_w,
        candidate_params=cand_total,
        candidate_trainable=cand_trainable,
        pathway_params=path_params,
        pathway_sha256=path_sha,
        forensic_critical_layer=forensic_rep.mean_critical_layer,
        forensic_failure_mechanism=forensic_rep.primary_failure_mechanism,
        best_internal_location=loc_rep.best_location,
        best_location_g4=loc_rep.best_g4_tok_acc,
        best_integration_mode=integ_rep.best_mode,
        best_integration_g4=integ_rep.best_g4_acc,
        objective_hop2_benefit=obj_rep.hop2_objective_benefit,
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
    print("RUNNING WAVE 305-312 MASTER BENCHMARK: TRAINABLE COMPOSITIONAL PATHWAY")
    print("=" * 70)
    rep = run_wave312_master_benchmark(train_steps=12, eval_episodes=6)
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Baseline Params: {rep.baseline_params} | SHA-256: {rep.baseline_sha[:16]}... | Delta W: {rep.baseline_delta_w}")
    print(f"Candidate Trainable Params: {rep.candidate_trainable} (Internal Pathway: {rep.pathway_params})")
    print(f"Critical Forensic Layer: Layer {rep.forensic_critical_layer}")
    print(f"Best Internal Location: {rep.best_internal_location} (G4 = {rep.best_location_g4*100:.1f}%)")
    print(f"Best Integration Mode: Mode {rep.best_integration_mode} (G4 = {rep.best_integration_g4*100:.1f}%)")
    print(f"Anti-Shortcut: {rep.anti_shortcut_conditions_checked} conditions audited, Passed: {rep.anti_shortcut_passed}")
    print(f"I3 Preserved: {rep.i3_preserved} (UU Tok = {rep.i3_uu_tok_acc*100:.1f}%), Language Retention: {rep.language_retention_ratio:.5f}")
    print(f"G1={rep.g1_mean_acc*100:.1f}%, G2={rep.g2_mean_acc*100:.1f}%, G3={rep.g3_mean_acc*100:.1f}%, G4={rep.g4_mean_acc*100:.1f}%")
    print(f"Per-seed G4: {rep.per_seed_g4}")
    print(f"I4 Gate Passed: {rep.i4_gate_passed} | Stability Diag Passed: {rep.stability_diagnostic_passed}")
    print(f"Classification: {rep.final_classification}")
    print(f"Architecture Decision: {rep.architecture_decision}")
    print(f"CPU Runtime: {rep.cpu_runtime_ms:.1f} ms")
    print("=" * 70)
