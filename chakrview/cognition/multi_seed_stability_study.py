"""Step 390: Multi-Seed Controller Stability Investigation.

Audits SufficiencyAdaptiveReasoningCore across strict seeds (42, 101, 2026):
Focuses on investigating Seed 2026 failure modes:
- Lazy halt vs poor representation vs controller instability vs output binding.

Measures per seed:
- G1, G2, G3, G4 accuracies
- H1, H2, H3 routing accuracies
- Mean reasoning cycles executed
- Premature halt rate & unnecessary continuation rate
- Sufficiency score distribution
- State norm & drift metrics
- Root cause diagnosis for Seed 2026
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.sufficiency_adaptive_reasoning import SufficiencyAdaptiveReasoningCore
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)
from chakrview.cognition.lazy_halt_ablation import train_and_eval_sufficiency_core


@dataclasses.dataclass
class SeedInvestigationMetrics:
    seed: int
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_routing: float
    h2_routing: float
    h3_routing: float
    mean_cycles: float
    premature_halt_rate: float
    unnecessary_rate: float
    mean_sufficiency_score: float
    state_norm: float
    seed2026_root_cause: str


@dataclasses.dataclass
class Step390MultiSeedStabilityReport:
    seed_metrics: Dict[int, SeedInvestigationMetrics]
    mean_g4: float
    stability_passed: bool
    seed2026_failure_diagnosis: str
    summary: str


def run_multi_seed_controller_stability_study(
    seeds: Tuple[int, ...] = (42, 101, 2026),
    train_steps: int = 20,
    eval_episodes: int = 6,
) -> Step390MultiSeedStabilityReport:
    """Executes Step 390 multi-seed stability and Seed 2026 forensic investigation."""
    results: Dict[int, SeedInvestigationMetrics] = {}
    all_g4 = []

    for s in seeds:
        torch.manual_seed(s)
        env = CompositionalAssociativeEnvironment(seed=s)

        core = SufficiencyAdaptiveReasoningCore()
        acc_h, cyc_h, g4, h2, h3 = train_and_eval_sufficiency_core(
            core, seed=s, train_steps=train_steps, eval_episodes_per_hop=eval_episodes
        )

        all_g4.append(g4)

        # Inspect internal states & sufficiency distribution on 2-hop tasks
        suff_scores = []
        state_norms = []
        h1_cnt = 0
        with torch.no_grad():
            for ep_i in range(eval_episodes):
                ep = generate_mixed_hop_episode(env, hop_count=2, split="disjoint_test", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                out = core(seq, candidate_positions=c_pos, return_trace=True)
                traces = out["cycle_traces"]
                if traces:
                    suff_scores.append(traces[0].sufficiency_score.item())
                    state_norms.append(torch.norm(traces[0].state_s, p=2).item())
                    if traces[0].w_attn is not None and ep.key_positions[0] >= 0:
                        w1 = traces[0].w_attn[0].mean(dim=0)[-1, :]
                        if torch.argmax(w1).item() == ep.key_positions[0]:
                            h1_cnt += 1

        m_suff = sum(suff_scores) / max(1, len(suff_scores))
        m_norm = sum(state_norms) / max(1, len(state_norms))
        h1_acc = h1_cnt / max(1, eval_episodes)

        c1 = cyc_h[1]
        c2 = cyc_h[2]
        c3 = cyc_h[3]
        premature = 1.0 if (c2 < 2.0 or c3 < 3.0) else 0.0
        unnecessary = 1.0 if c1 > 1.0 else 0.0

        # Diagnosis for Seed 2026 or low-performing seeds
        if g4 < 0.30:
            if premature > 0.5:
                diagnosis = "Lazy-Halt Attractor: Controller terminated before intermediate hop"
            elif h1_acc < 0.20:
                diagnosis = "Initial Retrieval Collapse: Query failed to bind Hop-1 key"
            elif m_norm < 0.05:
                diagnosis = "State Vanishing: Recurrent state magnitude collapsed"
            else:
                diagnosis = "Contextual Binding Mismatch: Candidate projection misaligned"
        else:
            diagnosis = "Stable Compositional Execution"

        results[s] = SeedInvestigationMetrics(
            seed=s,
            g1_acc=acc_h.get(1, 0.0),
            g2_acc=acc_h.get(2, 0.0),
            g3_acc=acc_h.get(3, 0.0),
            g4_acc=g4,
            h1_routing=h1_acc,
            h2_routing=h2,
            h3_routing=h3,
            mean_cycles=c2,
            premature_halt_rate=premature,
            unnecessary_rate=unnecessary,
            mean_sufficiency_score=m_suff,
            state_norm=m_norm,
            seed2026_root_cause=diagnosis,
        )

    mean_g4 = sum(all_g4) / len(all_g4)
    stable = all(m.g4_acc >= 0.40 for m in results.values())
    s2026_diag = results[2026].seed2026_root_cause

    summary = (
        f"Multi-Seed Stability: Mean G4={mean_g4:.2%}. Stability (all >=40%)={stable}. "
        f"Seed 42 G4={results[42].g4_acc:.2%}, Seed 101 G4={results[101].g4_acc:.2%}, "
        f"Seed 2026 G4={results[2026].g4_acc:.2%}. Seed 2026 Diagnosis: {s2026_diag}."
    )

    return Step390MultiSeedStabilityReport(
        seed_metrics=results,
        mean_g4=mean_g4,
        stability_passed=stable,
        seed2026_failure_diagnosis=s2026_diag,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 390: Running Multi-Seed Controller Stability Investigation...")
    rep = run_multi_seed_controller_stability_study(seeds=(42, 101, 2026), train_steps=20, eval_episodes=6)
    print("Report Summary:", rep.summary)
    for s, m in rep.seed_metrics.items():
        print(f"  Seed {s:5d}: G4={m.g4_acc:.2%}, Cycles={m.mean_cycles:.2f}, SuffScore={m.mean_sufficiency_score:.4f}, Norm={m.state_norm:.4f}, Diag: {m.seed2026_root_cause}")
