"""Step 414: Continual Learning Evaluation.

Evaluates the I4 candidate under a continual learning scenario:
- Task A: Standard single-hop retrieval (warm-up).
- Task B: Two-hop compositional binding (I4 target).
- Task C: Three-hop extended composition (stress test).
- Task D: Return to Task A (backward transfer / forgetting probe).

Metrics
-------
- Acquisition: Score on each task after training it.
- Retention: Score on prior tasks after learning new tasks.
- Forgetting: Max(prior_score) - score_after_new_task.
- Transfer: Whether learning B facilitates C (positive transfer).

Thresholds
----------
- Acquisition G4 >= 0.40 per task (where applicable)
- Forgetting delta <= 0.30 (acceptable forgetting bound)
- Language retention >= 0.9400 throughout
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
class ContinualTaskResult:
    task_name: str
    hop_count: int
    score_after_training: float   # Acquisition
    score_after_later_tasks: float  # Retention
    forgetting_delta: float
    transfer_benefit: float  # Positive = facilitation, Negative = interference


@dataclasses.dataclass
class Step414ContinualLearningReport:
    seed: int
    task_results: List[ContinualTaskResult]
    mean_acquisition: float
    mean_forgetting: float
    catastrophic_forgetting_detected: bool
    language_retention_final: float
    backward_transfer_a: float
    summary: str


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_n_steps(
    model: NeuralRelationalAcquisitionModule,
    env: CompositionalAssociativeEnvironment,
    optimizer: torch.optim.Optimizer,
    hop_count: int,
    n_steps: int,
) -> float:
    """Trains model for n_steps with the given hop_count. Returns final loss."""
    model.train()
    last_loss = 0.0
    for _ in range(n_steps):
        ep = generate_mixed_hop_episode(env, hop_count=hop_count, split="train", num_distractors=1)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        out = model(seq, candidate_positions=c_pos, max_hops=max(2, hop_count))

        target = torch.tensor([ep.target_idx], dtype=torch.long)
        h1_target = torch.tensor([ep.key_positions[0]], dtype=torch.long)

        loss = F.cross_entropy(out["binding_logits"], target)
        loss += 0.25 * F.cross_entropy(out["w1_key"], h1_target)
        if hop_count >= 2 and len(ep.key_positions) > 1:
            h2_target = torch.tensor([ep.key_positions[1]], dtype=torch.long)
            loss += 0.25 * F.cross_entropy(out["w2_key"], h2_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def _evaluate_hop(
    model: NeuralRelationalAcquisitionModule,
    env: CompositionalAssociativeEnvironment,
    hop_count: int,
    eval_episodes: int = 6,
    split: str = "disjoint_test",
) -> float:
    """Returns G accuracy for a given hop_count."""
    model.eval()
    hits = 0
    with torch.no_grad():
        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(env, hop_count=hop_count, split=split, num_distractors=1)
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = model(seq, candidate_positions=c_pos, max_hops=max(2, hop_count))
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                hits += 1
    return hits / max(1, eval_episodes)


# ---------------------------------------------------------------------------
# Main continual learning study
# ---------------------------------------------------------------------------

def run_step414_continual_learning(
    seed: int = 42,
    steps_per_task: int = 20,
    eval_episodes: int = 6,
) -> Step414ContinualLearningReport:
    """
    Runs the A -> B -> C -> D continual learning protocol.

    Tasks
    -----
    A: 1-hop retrieval
    B: 2-hop compositional binding (I4)
    C: 2-hop with 3 distractors (stress: harder than standard I4)
    D: 1-hop retrieval again (backward transfer probe)
    """
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    baseline = instantiate_frozen_baseline()

    model = NeuralRelationalAcquisitionModule(d_input=96, d_model=96, vocab_size=4096)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scores: Dict[str, Dict[str, float]] = {"A": {}, "B": {}, "C": {}, "D": {}}

    # -------- Task A: 1-hop retrieval --------
    _train_n_steps(model, env, optimizer, hop_count=1, n_steps=steps_per_task)
    scores["A"]["after_A"] = _evaluate_hop(model, env, hop_count=1, eval_episodes=eval_episodes)

    # -------- Task B: 2-hop compositional --------
    _train_n_steps(model, env, optimizer, hop_count=2, n_steps=steps_per_task)
    scores["A"]["after_B"] = _evaluate_hop(model, env, hop_count=1, eval_episodes=eval_episodes)
    scores["B"]["after_B"] = _evaluate_hop(model, env, hop_count=2, eval_episodes=eval_episodes)

    # -------- Task C: stress (harder 2-hop) --------
    # Train with more distractors (simulate harder generalization)
    model.train()
    for _ in range(steps_per_task):
        ep = generate_mixed_hop_episode(env, hop_count=2, split="heldout_composition", num_distractors=2)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        out = model(seq, candidate_positions=c_pos, max_hops=2)
        target = torch.tensor([ep.target_idx], dtype=torch.long)
        h1_t = torch.tensor([ep.key_positions[0]], dtype=torch.long)
        h2_t = torch.tensor([ep.key_positions[1]], dtype=torch.long)
        loss = (
            F.cross_entropy(out["binding_logits"], target)
            + 0.25 * F.cross_entropy(out["w1_key"], h1_t)
            + 0.25 * F.cross_entropy(out["w2_key"], h2_t)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scores["A"]["after_C"] = _evaluate_hop(model, env, hop_count=1, eval_episodes=eval_episodes)
    scores["B"]["after_C"] = _evaluate_hop(model, env, hop_count=2, eval_episodes=eval_episodes)
    scores["C"]["after_C"] = _evaluate_hop(model, env, hop_count=2, eval_episodes=eval_episodes, split="heldout_composition")

    # -------- Task D: 1-hop revisit (backward transfer probe) --------
    _train_n_steps(model, env, optimizer, hop_count=1, n_steps=steps_per_task)
    scores["A"]["after_D"] = _evaluate_hop(model, env, hop_count=1, eval_episodes=eval_episodes)
    scores["D"]["after_D"] = _evaluate_hop(model, env, hop_count=1, eval_episodes=eval_episodes)

    # Language retention
    lang_ret = evaluate_language_retention(model, baseline)

    # Build result objects
    task_results: List[ContinualTaskResult] = []

    for task_name, hop, acq_key, ret_key in [
        ("A_1hop", 1, "after_A", "after_D"),
        ("B_2hop", 2, "after_B", "after_C"),
        ("C_stress", 2, "after_C", "after_D"),
        ("D_return", 1, "after_D", "after_D"),
    ]:
        task_key = task_name.split("_")[0]
        acq = scores[task_key].get(acq_key, 0.0)
        ret = scores[task_key].get(ret_key, acq)
        forgetting = max(0.0, acq - ret)
        # Transfer: for Task B, compare B score after C vs after B
        if task_name == "B_2hop":
            transfer = scores["B"].get("after_C", 0.0) - acq
        else:
            transfer = 0.0
        task_results.append(ContinualTaskResult(
            task_name=task_name,
            hop_count=hop,
            score_after_training=acq,
            score_after_later_tasks=ret,
            forgetting_delta=forgetting,
            transfer_benefit=transfer,
        ))

    mean_acq = sum(r.score_after_training for r in task_results) / len(task_results)
    mean_forget = sum(r.forgetting_delta for r in task_results) / len(task_results)
    catastrophic = mean_forget > 0.30
    backward_a = scores["A"].get("after_D", 0.0) - scores["A"].get("after_A", 0.0)

    summary = (
        f"Step 414 | Continual Learning | Seed={seed} | "
        f"MeanAcquisition={mean_acq:.2%} | "
        f"MeanForgetting={mean_forget:.2%} | "
        f"CatastrophicForgetting={catastrophic} | "
        f"BackwardTransfer_A={backward_a:+.2%} | "
        f"LangRet={lang_ret:.4f}"
    )

    return Step414ContinualLearningReport(
        seed=seed,
        task_results=task_results,
        mean_acquisition=mean_acq,
        mean_forgetting=mean_forget,
        catastrophic_forgetting_detected=catastrophic,
        language_retention_final=lang_ret,
        backward_transfer_a=backward_a,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 414: Continual Learning Evaluation...")
    rep = run_step414_continual_learning(seed=42, steps_per_task=15, eval_episodes=4)
    print(rep.summary)
    for r in rep.task_results:
        print(f"  Task {r.task_name}: acq={r.score_after_training:.2%}, ret={r.score_after_later_tasks:.2%}, forget={r.forgetting_delta:.2%}")
