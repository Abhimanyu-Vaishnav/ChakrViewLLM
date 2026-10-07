"""Step 398: Multi-Seed Stability & Transformation Boundary Investigation.

Audits the stabilized relational architecture across strict seeds (42, 101, 2026).
Measures:
- G1, G2, G3, G4 accuracies
- H1 key routing accuracy
- H2 key routing accuracy
- Mean reasoning cycles executed
- Language retention
- Transformation boundary audit (H1 -> intermediate state -> H2 formation)
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import instantiate_frozen_baseline
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.relational_initialization_study import (
    InitializedRelationalCore,
)
from chakrview.cognition.end_to_end_relational_training import (
    train_and_eval_end_to_end_core,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)
from chakrview.cognition.relational_matching_objective import (
    evaluate_language_retention_core,
)


@dataclasses.dataclass
class StabilizedSeedMetrics:
    seed: int
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_routing: float
    h2_routing: float
    transformation_boundary_intact: bool


@dataclasses.dataclass
class Step398MultiSeedReport:
    seed_metrics: Dict[int, StabilizedSeedMetrics]
    mean_g1: float
    mean_g2: float
    mean_g3: float
    mean_g4: float
    mean_h1: float
    mean_h2: float
    language_retention: float
    stability_passed: bool
    transformation_boundary_identified: str
    summary: str


def run_multi_seed_relational_stability_study(
    seeds: Tuple[int, ...] = (42, 101, 2026),
    train_steps: int = 25,
    eval_episodes: int = 6,
) -> Step398MultiSeedReport:
    """Executes Step 398 multi-seed stability and transformation boundary audit."""
    baseline = instantiate_frozen_baseline()
    seed_metrics: Dict[int, StabilizedSeedMetrics] = {}
    all_g1, all_g2, all_g3, all_g4 = [], [], [], []
    all_h1, all_h2 = [], []

    for s in seeds:
        torch.manual_seed(s)
        env = CompositionalAssociativeEnvironment(seed=s)

        core = InitializedRelationalCore(mode="D_orthogonal")
        core, rep = train_and_eval_end_to_end_core(
            core=core, seed=s, train_steps=train_steps, eval_episodes=eval_episodes
        )

        # Check splits
        core.eval()
        def eval_split(split_name: str) -> float:
            hits = 0
            with torch.no_grad():
                for _ in range(eval_episodes):
                    ep = generate_mixed_hop_episode(env, hop_count=2, split=split_name, num_distractors=1)
                    seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                    c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
                    out = core(seq, candidate_positions=c_pos)
                    if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                        hits += 1
            return hits / max(1, eval_episodes)

        g1 = eval_split("train")
        g2 = eval_split("val")
        g3 = eval_split("heldout_composition")
        g4 = rep.final_g4_acc
        h1 = rep.final_h1_routing
        h2 = rep.final_h2_routing

        boundary_ok = (h1 >= 0.15) and (h2 >= 0.15)

        seed_metrics[s] = StabilizedSeedMetrics(
            seed=s,
            g1_acc=g1,
            g2_acc=g2,
            g3_acc=g3,
            g4_acc=g4,
            h1_routing=h1,
            h2_routing=h2,
            transformation_boundary_intact=boundary_ok,
        )

        all_g1.append(g1)
        all_g2.append(g2)
        all_g3.append(g3)
        all_g4.append(g4)
        all_h1.append(h1)
        all_h2.append(h2)

    mean_g1 = sum(all_g1) / len(all_g1)
    mean_g2 = sum(all_g2) / len(all_g2)
    mean_g3 = sum(all_g3) / len(all_g3)
    mean_g4 = sum(all_g4) / len(all_g4)
    mean_h1 = sum(all_h1) / len(all_h1)
    mean_h2 = sum(all_h2) / len(all_h2)

    lang_ret = evaluate_language_retention_core(core, baseline)
    stable = all(m.g4_acc >= 0.40 for m in seed_metrics.values())

    if mean_h1 < 0.20:
        boundary_diagnosis = "Hop-1 Relational Initialization Boundary"
    elif mean_h2 < 0.20:
        boundary_diagnosis = "Hop-1 to Intermediate to Hop-2 State Transformation Boundary"
    else:
        boundary_diagnosis = "Stable Relational Retrieval & Transition Chain"

    summary = (
        f"Multi-Seed Relational Stability: Mean G4={mean_g4:.2%}, Mean H1={mean_h1:.2%}, "
        f"Mean H2={mean_h2:.2%}. Stability (all >=40%)={stable}. "
        f"Transformation Boundary: {boundary_diagnosis}."
    )

    return Step398MultiSeedReport(
        seed_metrics=seed_metrics,
        mean_g1=mean_g1,
        mean_g2=mean_g2,
        mean_g3=mean_g3,
        mean_g4=mean_g4,
        mean_h1=mean_h1,
        mean_h2=mean_h2,
        language_retention=lang_ret,
        stability_passed=stable,
        transformation_boundary_identified=boundary_diagnosis,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 398: Running Multi-Seed Relational Stability Study...")
    rep = run_multi_seed_relational_stability_study(seeds=(42, 101, 2026), train_steps=20, eval_episodes=4)
    print("Report Summary:", rep.summary)
    for s, m in rep.seed_metrics.items():
        print(f"  Seed {s:5d}: G4={m.g4_acc:.2%}, H1={m.h1_routing:.2%}, H2={m.h2_routing:.2%}, BoundaryOK={m.transformation_boundary_intact}")
