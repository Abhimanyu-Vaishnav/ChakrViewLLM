"""Step 387: Counterfactual Continuation Test.

Evaluates causal utility of additional reasoning cycles from the same intermediate state S_t:
Path A: Answer immediately at cycle t
Path B: Execute one additional reasoning cycle (t + 1), then answer

Measures:
benefit_of_continuation = quality(Path B) - quality(Path A)

Evaluates whether the sufficiency controller's decision (continue vs halt)
positively correlates with the measured empirical benefit of continuation.
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


@dataclasses.dataclass
class CounterfactualPairResult:
    episode_idx: int
    hop_depth: int
    path_a_prob: float
    path_b_prob: float
    benefit_delta: float
    controller_continue_prob: float
    controller_decision_aligned: bool


@dataclasses.dataclass
class Step387CounterfactualReport:
    pairs: List[CounterfactualPairResult]
    mean_benefit_1hop: float
    mean_benefit_2hop: float
    mean_benefit_3hop: float
    correlation_continue_with_benefit: float
    alignment_accuracy: float
    summary: str


def run_counterfactual_continuation_test(
    core: Optional[SufficiencyAdaptiveReasoningCore] = None,
    seed: int = 42,
    episodes_per_hop: int = 6,
) -> Step387CounterfactualReport:
    """Executes Step 387 counterfactual continuation evaluation."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    if core is None:
        core = SufficiencyAdaptiveReasoningCore()
    core.eval()

    pairs: List[CounterfactualPairResult] = []
    benefits_by_hop = {1: [], 2: [], 3: []}
    all_benefits = []
    all_probs = []
    aligned_count = 0

    with torch.no_grad():
        pair_id = 0
        for h_k in [1, 2, 3]:
            for ep_i in range(episodes_per_hop):
                pair_id += 1
                ep = generate_mixed_hop_episode(env, hop_count=h_k, split="disjoint_test", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                # Path A: Fixed 1 cycle
                out_a = core(seq, candidate_positions=c_pos, fixed_cycles=1, return_trace=True)
                prob_a = out_a["binding_probs"][0, ep.target_idx].item()
                c_prob = out_a["p_continue_per_cycle"][0].item()

                # Path B: Fixed 2 cycles
                out_b = core(seq, candidate_positions=c_pos, fixed_cycles=2, return_trace=True)
                prob_b = out_b["binding_probs"][0, ep.target_idx].item()

                benefit = prob_b - prob_a
                benefits_by_hop[h_k].append(benefit)
                all_benefits.append(benefit)
                all_probs.append(c_prob)

                # Decision alignment:
                # If benefit > 0, controller should have continue_prob >= 0.5
                # If benefit <= 0, controller should have continue_prob < 0.5
                aligned = (benefit > 0.0 and c_prob >= 0.5) or (benefit <= 0.0 and c_prob < 0.5)
                if aligned:
                    aligned_count += 1

                pairs.append(
                    CounterfactualPairResult(
                        episode_idx=pair_id,
                        hop_depth=h_k,
                        path_a_prob=prob_a,
                        path_b_prob=prob_b,
                        benefit_delta=benefit,
                        controller_continue_prob=c_prob,
                        controller_decision_aligned=aligned,
                    )
                )

    m_b1 = sum(benefits_by_hop[1]) / max(1, len(benefits_by_hop[1]))
    m_b2 = sum(benefits_by_hop[2]) / max(1, len(benefits_by_hop[2]))
    m_b3 = sum(benefits_by_hop[3]) / max(1, len(benefits_by_hop[3]))

    # Pearson correlation
    b_t = torch.tensor(all_benefits, dtype=torch.float)
    p_t = torch.tensor(all_probs, dtype=torch.float)
    b_m = b_t.mean()
    p_m = p_t.mean()
    num = ((b_t - b_m) * (p_t - p_m)).sum()
    denom = torch.sqrt(((b_t - b_m) ** 2).sum() * ((p_t - p_m) ** 2).sum() + 1e-8)
    corr = float(num / denom)

    align_rate = aligned_count / max(1, len(pairs))

    summary = (
        f"Counterfactual Continuation: Benefit 1h={m_b1:+.4f}, 2h={m_b2:+.4f}, 3h={m_b3:+.4f}. "
        f"Continuation-Benefit Correlation={corr:+.4f}. Alignment Accuracy={align_rate:.2%}."
    )

    return Step387CounterfactualReport(
        pairs=pairs,
        mean_benefit_1hop=m_b1,
        mean_benefit_2hop=m_b2,
        mean_benefit_3hop=m_b3,
        correlation_continue_with_benefit=corr,
        alignment_accuracy=align_rate,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 387: Running Counterfactual Continuation Test...")
    rep = run_counterfactual_continuation_test(episodes_per_hop=6)
    print("Report Summary:", rep.summary)
    for p in rep.pairs[:6]:
        print(f"  Ep {p.episode_idx} ({p.hop_depth}h): PathA={p.path_a_prob:.4f}, PathB={p.path_b_prob:.4f}, Benefit={p.benefit_delta:+.4f}, ContProb={p.controller_continue_prob:.4f}")
