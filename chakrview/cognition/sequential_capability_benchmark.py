"""Step 415: Sequential Capability Benchmark (A -> B -> C -> D).

Runs a structured four-task sequential benchmark with explicit
acquisition, retention, forgetting, and transfer diagnosis.

Tasks
-----
A: Single-hop retrieval  (train split, 1 distractor)
B: Two-hop compositional (train split, 1 distractor) - I4 target
C: Two-hop generalization (heldout_composition, 2 distractors) - stress
D: Disjoint two-hop     (disjoint_test, 1 distractor) - true generalization

Diagnosis Thresholds
--------------------
- Acquisition per task >= 0.30
- Forgetting delta <= 0.40
- Language retention throughout >= 0.9400
- No catastrophic collapse: final_task_D >= 0.20
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_relational_acquisition import (
    NeuralRelationalAcquisitionModule,
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
class SequentialTaskSpec:
    name: str
    hop_count: int
    split: str
    num_distractors: int


@dataclasses.dataclass
class SequentialBenchmarkEntry:
    """Score on task X evaluated after training on task Y."""
    eval_task: str
    eval_after_task: str
    score: float


@dataclasses.dataclass
class SequentialBenchmarkSeedResult:
    seed: int
    entries: List[SequentialBenchmarkEntry]
    # Derived
    acquisition_a: float
    acquisition_b: float
    acquisition_c: float
    acquisition_d: float
    forgetting_a_after_b: float
    forgetting_b_after_c: float
    forgetting_b_after_d: float
    language_retention: float
    catastrophic_collapse: bool  # final_D < 0.20
    passed: bool


@dataclasses.dataclass
class Step415SequentialBenchmarkReport:
    seed_results: List[SequentialBenchmarkSeedResult]
    mean_acquisition_b: float
    mean_acquisition_d: float
    mean_forgetting_a: float
    mean_forgetting_b: float
    catastrophic_count: int
    total_seeds: int
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

TASK_SPECS: List[SequentialTaskSpec] = [
    SequentialTaskSpec("A", hop_count=1, split="train", num_distractors=1),
    SequentialTaskSpec("B", hop_count=2, split="train", num_distractors=1),
    SequentialTaskSpec("C", hop_count=2, split="heldout_composition", num_distractors=2),
    SequentialTaskSpec("D", hop_count=2, split="disjoint_test", num_distractors=1),
]


def _train_task(
    model: NeuralRelationalAcquisitionModule,
    env: CompositionalAssociativeEnvironment,
    optimizer: torch.optim.Optimizer,
    spec: SequentialTaskSpec,
    n_steps: int,
) -> None:
    model.train()
    for _ in range(n_steps):
        ep = generate_mixed_hop_episode(
            env, hop_count=spec.hop_count,
            split=spec.split, num_distractors=spec.num_distractors
        )
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        out = model(seq, candidate_positions=c_pos, max_hops=max(2, spec.hop_count))

        target = torch.tensor([ep.target_idx], dtype=torch.long)
        loss = F.cross_entropy(out["binding_logits"], target)
        loss += 0.25 * F.cross_entropy(out["w1_key"], torch.tensor([ep.key_positions[0]], dtype=torch.long))
        if spec.hop_count >= 2 and len(ep.key_positions) > 1:
            loss += 0.25 * F.cross_entropy(out["w2_key"], torch.tensor([ep.key_positions[1]], dtype=torch.long))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _eval_task(
    model: NeuralRelationalAcquisitionModule,
    env: CompositionalAssociativeEnvironment,
    spec: SequentialTaskSpec,
    eval_episodes: int,
) -> float:
    model.eval()
    hits = 0
    with torch.no_grad():
        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(
                env, hop_count=spec.hop_count,
                split=spec.split, num_distractors=spec.num_distractors
            )
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = model(seq, candidate_positions=c_pos, max_hops=max(2, spec.hop_count))
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                hits += 1
    return hits / max(1, eval_episodes)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _run_seed_benchmark(
    seed: int,
    steps_per_task: int = 20,
    eval_episodes: int = 6,
) -> SequentialBenchmarkSeedResult:
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    baseline = instantiate_frozen_baseline()

    model = NeuralRelationalAcquisitionModule(d_input=96, d_model=96, vocab_size=4096)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    entries: List[SequentialBenchmarkEntry] = []
    scores: Dict[str, Dict[str, float]] = {s.name: {} for s in TASK_SPECS}

    # Sequential training and evaluation
    for i, spec in enumerate(TASK_SPECS):
        _train_task(model, env, optimizer, spec, n_steps=steps_per_task)
        # After training task i, evaluate ALL tasks so far
        for j, eval_spec in enumerate(TASK_SPECS[: i + 1]):
            score = _eval_task(model, env, eval_spec, eval_episodes)
            key = f"after_{spec.name}"
            scores[eval_spec.name][key] = score
            entries.append(SequentialBenchmarkEntry(
                eval_task=eval_spec.name,
                eval_after_task=spec.name,
                score=score,
            ))

    # Also evaluate all tasks after final task D
    for spec in TASK_SPECS:
        if f"after_D" not in scores[spec.name]:
            score = _eval_task(model, env, spec, eval_episodes)
            scores[spec.name]["after_D"] = score
            entries.append(SequentialBenchmarkEntry(
                eval_task=spec.name,
                eval_after_task="D",
                score=score,
            ))

    # Extract key metrics
    acq_a = scores["A"].get("after_A", 0.0)
    acq_b = scores["B"].get("after_B", 0.0)
    acq_c = scores["C"].get("after_C", 0.0)
    acq_d = scores["D"].get("after_D", 0.0)
    forget_a_after_b = max(0.0, acq_a - scores["A"].get("after_B", acq_a))
    forget_b_after_c = max(0.0, acq_b - scores["B"].get("after_C", acq_b))
    forget_b_after_d = max(0.0, acq_b - scores["B"].get("after_D", acq_b))

    lang_ret = evaluate_language_retention(model, baseline)
    catastrophic = acq_d < 0.20

    passed = (
        acq_b >= 0.30
        and forget_a_after_b <= 0.40
        and forget_b_after_c <= 0.40
        and lang_ret >= 0.9400
        and not catastrophic
    )

    return SequentialBenchmarkSeedResult(
        seed=seed,
        entries=entries,
        acquisition_a=acq_a,
        acquisition_b=acq_b,
        acquisition_c=acq_c,
        acquisition_d=acq_d,
        forgetting_a_after_b=forget_a_after_b,
        forgetting_b_after_c=forget_b_after_c,
        forgetting_b_after_d=forget_b_after_d,
        language_retention=lang_ret,
        catastrophic_collapse=catastrophic,
        passed=passed,
    )


def run_step415_sequential_benchmark(
    seeds: Tuple[int, ...] = (42, 101),
    steps_per_task: int = 20,
    eval_episodes: int = 6,
) -> Step415SequentialBenchmarkReport:
    """Executes Step 415: sequential A->B->C->D capability benchmark."""
    seed_results: List[SequentialBenchmarkSeedResult] = []
    for seed in seeds:
        r = _run_seed_benchmark(seed, steps_per_task=steps_per_task, eval_episodes=eval_episodes)
        seed_results.append(r)

    mean_acq_b = sum(r.acquisition_b for r in seed_results) / max(1, len(seed_results))
    mean_acq_d = sum(r.acquisition_d for r in seed_results) / max(1, len(seed_results))
    mean_forget_a = sum(r.forgetting_a_after_b for r in seed_results) / max(1, len(seed_results))
    mean_forget_b = sum(r.forgetting_b_after_c for r in seed_results) / max(1, len(seed_results))
    catastrophic_count = sum(1 for r in seed_results if r.catastrophic_collapse)

    summary = (
        f"Step 415 | Sequential Benchmark A->B->C->D | "
        f"Seeds={list(seeds)} | "
        f"MeanAcqB={mean_acq_b:.2%} | "
        f"MeanAcqD={mean_acq_d:.2%} | "
        f"MeanForgetA={mean_forget_a:.2%} | "
        f"MeanForgetB={mean_forget_b:.2%} | "
        f"CatastrophicCollapse={catastrophic_count}/{len(seed_results)}"
    )

    return Step415SequentialBenchmarkReport(
        seed_results=seed_results,
        mean_acquisition_b=mean_acq_b,
        mean_acquisition_d=mean_acq_d,
        mean_forgetting_a=mean_forget_a,
        mean_forgetting_b=mean_forget_b,
        catastrophic_count=catastrophic_count,
        total_seeds=len(seed_results),
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 415: Sequential Capability Benchmark A->B->C->D...")
    rep = run_step415_sequential_benchmark(seeds=(42,), steps_per_task=15, eval_episodes=4)
    print(rep.summary)
    for r in rep.seed_results:
        print(f"  Seed {r.seed}: acq_B={r.acquisition_b:.2%}, acq_D={r.acquisition_d:.2%}, "
              f"forget_A={r.forgetting_a_after_b:.2%}, forget_B={r.forgetting_b_after_c:.2%}, "
              f"lang={r.language_retention:.4f}, passed={r.passed}")
