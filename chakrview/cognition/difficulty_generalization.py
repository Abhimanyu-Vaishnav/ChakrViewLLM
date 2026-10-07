"""Step 388: Difficulty Generalization under Need-Based Reasoning.

Trains on mixed 1-hop, 2-hop, 3-hop problems without providing hop counts.
Evaluates across 1-hop, 2-hop, 3-hop, and 4-hop difficulties.

Tracks:
- Mean cycles executed by task depth (1h vs 2h vs 3h vs 4h)
- Monotonic compute scaling (does computation scale with difficulty?)
- Accuracy across each difficulty level
- Premature halt rate and unnecessary continuation rate
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
    AdaptiveEpisode,
)
from chakrview.cognition.lazy_halt_ablation import train_and_eval_sufficiency_core


@dataclasses.dataclass
class DifficultyLevelResult:
    hop_depth: int
    accuracy: float
    mean_cycles: float
    min_cycles: int
    max_cycles: int
    premature_halt_rate: float
    unnecessary_rate: float


@dataclasses.dataclass
class Step388DifficultyGeneralizationReport:
    levels: Dict[int, DifficultyLevelResult]
    overall_accuracy: float
    compute_scales_with_difficulty: bool
    monotonicity_metric: float
    summary: str


def run_difficulty_generalization_study(
    core: Optional[SufficiencyAdaptiveReasoningCore] = None,
    seed: int = 42,
    eval_episodes_per_level: int = 6,
) -> Step388DifficultyGeneralizationReport:
    """Executes Step 388 evaluation across 1-hop, 2-hop, 3-hop, and 4-hop tasks."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    if core is None:
        core = SufficiencyAdaptiveReasoningCore()
        train_and_eval_sufficiency_core(core, seed=seed, train_steps=20, eval_episodes_per_hop=4)
    core.eval()

    level_results: Dict[int, DifficultyLevelResult] = {}
    all_acc = []
    cycle_means = []

    for depth in [1, 2, 3, 4]:
        hits = 0
        cycles_list = []
        premature = 0
        unnecessary = 0

        with torch.no_grad():
            for ep_i in range(eval_episodes_per_level):
                if depth <= 3:
                    ep = generate_mixed_hop_episode(env, hop_count=depth, split="disjoint_test", num_distractors=1)
                    seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                    c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
                    tgt_idx = ep.target_idx
                else:
                    # 4-hop generation
                    pool = list(env.DISJOINT_KEYS_POOL)
                    if len(pool) < 5: pool = pool + list(env.DISJOINT_VALS_POOL)
                    sampled = env.rng.sample(pool, 5)
                    A, B, C, D, E = sampled[0], sampled[1], sampled[2], sampled[3], sampled[4]
                    pairs = [(A, B), (B, C), (C, D), (D, E)]
                    env.rng.shuffle(pairs)
                    prompt = env.render_prompt(pairs, query_key=A, layout="standard_map")
                    tokens = tok.encode(prompt)
                    seq = torch.tensor([tokens], dtype=torch.long)
                    cand_pos, cand_toks = [], []
                    for k, v in pairs:
                        v_enc = tok.encode(v)[0]
                        m = [j for j, t in enumerate(tokens[:-1]) if t == v_enc]
                        if m and m[0] not in cand_pos:
                            cand_pos.append(m[0])
                            cand_toks.append(v_enc)
                    if not cand_pos: cand_pos, cand_toks = [0], [tokens[0]]
                    c_pos = torch.tensor([cand_pos], dtype=torch.long)
                    E_enc = tok.encode(E)[0]
                    tgt_idx = cand_toks.index(E_enc) if E_enc in cand_toks else 0

                out = core(seq, candidate_positions=c_pos, max_reasoning_cycles=6)
                c_used = out["total_cycles_executed"]
                cycles_list.append(c_used)

                pred_idx = torch.argmax(out["binding_logits"][0]).item()
                if pred_idx == tgt_idx:
                    hits += 1

                if c_used < depth:
                    premature += 1
                elif c_used > depth:
                    unnecessary += 1

        N = max(1, eval_episodes_per_level)
        acc = hits / N
        m_c = sum(cycles_list) / N
        all_acc.append(acc)
        cycle_means.append(m_c)

        level_results[depth] = DifficultyLevelResult(
            hop_depth=depth,
            accuracy=acc,
            mean_cycles=m_c,
            min_cycles=min(cycles_list) if cycles_list else 0,
            max_cycles=max(cycles_list) if cycles_list else 0,
            premature_halt_rate=premature / N,
            unnecessary_rate=unnecessary / N,
        )

    # Monotonicity check
    monotonic = (cycle_means[0] <= cycle_means[1] <= cycle_means[2] <= cycle_means[3])
    mono_score = cycle_means[3] - cycle_means[0]

    mean_acc = sum(all_acc) / len(all_acc)

    summary = (
        f"Difficulty Generalization: Mean Acc={mean_acc:.2%}. Cycles: 1h={cycle_means[0]:.2f}, "
        f"2h={cycle_means[1]:.2f}, 3h={cycle_means[2]:.2f}, 4h={cycle_means[3]:.2f}. "
        f"Compute Scales With Difficulty={monotonic} (Delta={mono_score:+.2f})."
    )

    return Step388DifficultyGeneralizationReport(
        levels=level_results,
        overall_accuracy=mean_acc,
        compute_scales_with_difficulty=monotonic,
        monotonicity_metric=mono_score,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 388: Running Difficulty Generalization Study...")
    rep = run_difficulty_generalization_study(eval_episodes_per_level=6)
    print("Report Summary:", rep.summary)
    for d, r in rep.levels.items():
        print(f"  Level {d}-Hop: Acc={r.accuracy:.2%}, MeanC={r.mean_cycles:.2f}, Premature={r.premature_halt_rate:.2%}")
