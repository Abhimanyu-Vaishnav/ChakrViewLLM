"""Step 413: Long-Horizon Knowledge Retention Study.

Investigates whether the I4 candidate retains its compositional
relational capability when exposed to additional training epochs beyond
the Wave 401-408 training window.

Retention Dimensions
--------------------
1. G4 compositional generalization retention across extended training.
2. H1/H2 routing stability under continued gradient pressure.
3. Language modelling retention (perplexity proxy).
4. Resistance to catastrophic forgetting on original training distribution.

The study uses a controlled protocol:
- Train for N standard epochs (warm-up, identical to Wave 408).
- Then apply M additional epochs of compositional curriculum.
- Evaluate G4, H1, H2, and language retention at each checkpoint.
- Report forgetting delta = score_at_peak - score_at_end.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_relational_acquisition import (
    NeuralRelationalAcquisitionModule,
)
from chakrview.cognition.compositional_relational_integration import (
    train_and_eval_compositional_integration,
)
from chakrview.cognition.adaptive_learnability import generate_mixed_hop_episode
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.hop1_objective_ablation import evaluate_language_retention


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RetentionCheckpoint:
    epoch: int
    g4_acc: float
    h1_routing: float
    h2_routing: float
    language_retention: float
    loss: float


@dataclasses.dataclass
class LongHorizonRetentionResult:
    seed: int
    checkpoints: List[RetentionCheckpoint]
    peak_g4: float
    final_g4: float
    forgetting_delta_g4: float
    peak_h2: float
    final_h2: float
    forgetting_delta_h2: float
    final_language_retention: float
    retention_sustained: bool  # True if forgetting_delta_g4 < 20%


@dataclasses.dataclass
class Step413RetentionReport:
    results: List[LongHorizonRetentionResult]
    mean_final_g4: float
    mean_forgetting_delta_g4: float
    retention_sustained_count: int
    total_seeds: int
    summary: str


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_model(
    model: NeuralRelationalAcquisitionModule,
    env: CompositionalAssociativeEnvironment,
    eval_episodes: int = 6,
    split: str = "disjoint_test",
) -> Tuple[float, float, float]:
    """Returns (g4_acc, h1_routing, h2_routing)."""
    model.eval()
    hits, h1_hits, h2_hits = 0, 0, 0
    with torch.no_grad():
        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(env, hop_count=2, split=split, num_distractors=1)
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = model(seq, candidate_positions=c_pos, max_hops=2)
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                hits += 1
            if torch.argmax(out["w1_key"][0]).item() == ep.key_positions[0]:
                h1_hits += 1
            if torch.argmax(out["w2_key"][0]).item() == ep.key_positions[1]:
                h2_hits += 1
    N = max(1, eval_episodes)
    return hits / N, h1_hits / N, h2_hits / N


# ---------------------------------------------------------------------------
# Core retention study
# ---------------------------------------------------------------------------

def _run_retention_study_seed(
    seed: int,
    warmup_steps: int = 30,
    extension_steps: int = 30,
    checkpoint_interval: int = 10,
    eval_episodes: int = 6,
) -> LongHorizonRetentionResult:
    """Runs the retention study for a single seed."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    baseline = instantiate_frozen_baseline()

    model = NeuralRelationalAcquisitionModule(d_input=96, d_model=96, vocab_size=4096)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    checkpoints: List[RetentionCheckpoint] = []

    # -----------
    # Warm-up phase (identical to Wave 408 training)
    # -----------
    total_steps = warmup_steps + extension_steps

    for step in range(1, total_steps + 1):
        model.train()
        ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        out = model(seq, candidate_positions=c_pos, max_hops=2)

        target = torch.tensor([ep.target_idx], dtype=torch.long)
        h1_target = torch.tensor([ep.key_positions[0]], dtype=torch.long)
        h2_target = torch.tensor([ep.key_positions[1]], dtype=torch.long)

        loss_final = F.cross_entropy(out["binding_logits"], target)
        loss_h1 = F.cross_entropy(out["w1_key"], h1_target)
        loss_h2 = F.cross_entropy(out["w2_key"], h2_target)
        loss = loss_final + 0.25 * loss_h1 + 0.25 * loss_h2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % checkpoint_interval == 0 or step == total_steps:
            g4, h1, h2 = _eval_model(model, env, eval_episodes=eval_episodes)
            lang_ret = evaluate_language_retention(model, baseline)
            checkpoints.append(RetentionCheckpoint(
                epoch=step,
                g4_acc=g4,
                h1_routing=h1,
                h2_routing=h2,
                language_retention=lang_ret,
                loss=loss.item(),
            ))

    # Compute retention metrics
    all_g4 = [c.g4_acc for c in checkpoints]
    all_h2 = [c.h2_routing for c in checkpoints]
    peak_g4 = max(all_g4)
    final_g4 = all_g4[-1]
    peak_h2 = max(all_h2)
    final_h2 = all_h2[-1]

    forgetting_g4 = peak_g4 - final_g4
    forgetting_h2 = peak_h2 - final_h2
    retention_ok = forgetting_g4 < 0.20  # Less than 20% absolute drop

    return LongHorizonRetentionResult(
        seed=seed,
        checkpoints=checkpoints,
        peak_g4=peak_g4,
        final_g4=final_g4,
        forgetting_delta_g4=forgetting_g4,
        peak_h2=peak_h2,
        final_h2=final_h2,
        forgetting_delta_h2=forgetting_h2,
        final_language_retention=checkpoints[-1].language_retention,
        retention_sustained=retention_ok,
    )


def run_step413_long_horizon_retention(
    seeds: Tuple[int, ...] = (42, 101),
    warmup_steps: int = 30,
    extension_steps: int = 30,
    checkpoint_interval: int = 10,
    eval_episodes: int = 6,
) -> Step413RetentionReport:
    """Executes Step 413: long-horizon knowledge retention study."""
    results: List[LongHorizonRetentionResult] = []
    for seed in seeds:
        r = _run_retention_study_seed(
            seed=seed,
            warmup_steps=warmup_steps,
            extension_steps=extension_steps,
            checkpoint_interval=checkpoint_interval,
            eval_episodes=eval_episodes,
        )
        results.append(r)

    mean_final_g4 = sum(r.final_g4 for r in results) / max(1, len(results))
    mean_forgetting = sum(r.forgetting_delta_g4 for r in results) / max(1, len(results))
    sustained_count = sum(1 for r in results if r.retention_sustained)

    summary = (
        f"Step 413 | Long-Horizon Retention | "
        f"Seeds={list(seeds)} | "
        f"MeanFinalG4={mean_final_g4:.2%} | "
        f"MeanForgettingDelta={mean_forgetting:.2%} | "
        f"RetentionSustained={sustained_count}/{len(results)}"
    )

    return Step413RetentionReport(
        results=results,
        mean_final_g4=mean_final_g4,
        mean_forgetting_delta_g4=mean_forgetting,
        retention_sustained_count=sustained_count,
        total_seeds=len(results),
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 413: Long-Horizon Knowledge Retention Study...")
    rep = run_step413_long_horizon_retention(seeds=(42,), warmup_steps=20, extension_steps=20)
    print(rep.summary)
    for r in rep.results:
        print(f"  Seed {r.seed}: peak_G4={r.peak_g4:.2%}, final_G4={r.final_g4:.2%}, forgetting={r.forgetting_delta_g4:.2%}, sustained={r.retention_sustained}")
        for c in r.checkpoints:
            print(f"    epoch={c.epoch:3d}: G4={c.g4_acc:.2%}, H2={c.h2_routing:.2%}, lang={c.language_retention:.4f}")
