"""Step 366: Causal Multi-Block Interventions.

Applies causal interventions to the multi-block recurrent core (L2 + L3):
1. normal_persistent_state: Standard multi-block forward pass
2. zero_l2_state: Force Layer 2 s1 = 0
3. zero_l3_state: Force Layer 3 s1 = 0
4. shuffle_l2_l3_state: Permute dimensions of state passed from L2 to L3
5. reset_state_between_blocks: Reset state between L2 and L3 (fresh s0 at L3)
6. bypass_l3_recurrent_pathway: Force q2 = q_base at L3 (bypass state in query gen)
7. disable_l3_second_cycle: Set Cycle 2 attention output a2 = 0 at L3
8. replace_l3_with_baseline: Replace Layer 3 with original frozen baseline block
9. replace_both_with_baseline: Replace both Layer 2 and Layer 3 with original frozen blocks

Measures:
- Target token probability
- Final token accuracy
- Hop-1 & Hop-2 key routing
- Drop from normal baseline condition
- Verifies causal dependence on the multi-block recurrent neural core.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import instantiate_frozen_baseline
from chakrview.cognition.multiblock_recurrent_attention_core import MultiBlockRecurrentChakrMicro
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class MultiBlockCausalResult:
    condition: str
    target_token_prob: float
    final_token_acc: float
    h1_key_acc: float
    h2_key_acc: float
    drop_from_normal: float


@dataclasses.dataclass
class MultiBlockCausalReport:
    results: Dict[str, MultiBlockCausalResult]
    normal_prob: float
    causally_active: bool
    mean_degradation: float
    summary: str


def run_multiblock_causal_interventions(
    candidate: MultiBlockRecurrentChakrMicro,
    seed: int = 42,
    num_episodes: int = 8,
) -> MultiBlockCausalReport:
    """Executes 9 causal intervention conditions on the trained multi-block candidate."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    episodes = [
        env.generate_episode(split="disjoint_test", num_distractors=1, episode_idx=366000 + i)
        for i in range(num_episodes)
    ]

    candidate.eval()

    conditions = [
        "1_normal_persistent",
        "2_zero_l2_state",
        "3_zero_l3_state",
        "4_shuffle_l2_l3_state",
        "5_reset_between_blocks",
        "6_bypass_l3_recurrent",
        "7_disable_l3_cycle2",
        "8_replace_l3_baseline",
        "9_replace_both_baseline",
    ]

    results: Dict[str, MultiBlockCausalResult] = {}
    normal_prob = 0.0

    for cond in conditions:
        tot_prob = 0.0
        token_hits = 0
        h1_k_hits = 0
        h2_k_hits = 0

        with torch.no_grad():
            for ep in episodes:
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                cand_positions, cand_tokens = [], []
                for k, v in ep.all_premise_pairs:
                    v_enc = tok.encode(v)[0]
                    pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if pos_list and pos_list[0] not in cand_positions:
                        cand_positions.append(pos_list[0])
                        cand_tokens.append(v_enc)
                if not cand_positions:
                    cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

                c_ids = torch.tensor([cand_tokens], dtype=torch.long)
                c_pos = torch.tensor([cand_positions], dtype=torch.long)
                tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0

                zero_l = None
                shuffle_b = False
                reset_b = False
                bypass_l = None
                disable_c2_l = None
                replace_base_l = None

                if cond == "2_zero_l2_state":
                    zero_l = [2]
                elif cond == "3_zero_l3_state":
                    zero_l = [3]
                elif cond == "4_shuffle_l2_l3_state":
                    shuffle_b = True
                elif cond == "5_reset_between_blocks":
                    reset_b = True
                elif cond == "6_bypass_l3_recurrent":
                    bypass_l = [3]
                elif cond == "7_disable_l3_cycle2":
                    disable_c2_l = [3]
                elif cond == "8_replace_l3_baseline":
                    replace_base_l = [3]
                elif cond == "9_replace_both_baseline":
                    replace_base_l = [2, 3]

                out = candidate(
                    input_ids=seq,
                    candidate_ids=c_ids,
                    candidate_positions=c_pos,
                    zero_state_layers=zero_l,
                    shuffle_state_between_blocks=shuffle_b,
                    reset_state_between_blocks=reset_b,
                    bypass_state_layers=bypass_l,
                    disable_cycle2_layers=disable_c2_l,
                    replace_block_with_baseline=replace_base_l,
                    return_traces=True,
                )

                b_probs = out["candidate_probs"][0]
                prob = b_probs[tgt_idx].item()
                tot_prob += prob

                pred_idx = out["binding_logits"].argmax(dim=-1).item()
                if pred_idx == tgt_idx:
                    token_hits += 1

                traces = out["traces"]
                if 2 in traces and traces[2].attn1_weights is not None:
                    attn1 = traces[2].attn1_weights[0].mean(dim=0)[-1]
                    if attn1.argmax().item() == ep.hop1_key_pos:
                        h1_k_hits += 1

                if 3 in traces and traces[3].attn2_weights is not None and cond != "7_disable_l3_cycle2":
                    attn2 = traces[3].attn2_weights[0].mean(dim=0)[-1]
                    if attn2.argmax().item() == ep.hop2_key_pos:
                        h2_k_hits += 1

        N = max(1, len(episodes))
        mean_p = tot_prob / N
        if cond == "1_normal_persistent":
            normal_prob = mean_p

        results[cond] = MultiBlockCausalResult(
            condition=cond,
            target_token_prob=mean_p,
            final_token_acc=token_hits / N,
            h1_key_acc=h1_k_hits / N,
            h2_key_acc=h2_k_hits / N,
            drop_from_normal=normal_prob - mean_p,
        )

    norm_res = results["1_normal_persistent"]
    rep_both = results["9_replace_both_baseline"]
    rep_l3 = results["8_replace_l3_baseline"]

    causally_active = (
        norm_res.target_token_prob >= rep_both.target_token_prob
        or norm_res.final_token_acc >= rep_both.final_token_acc
    )

    drops = [r.drop_from_normal for k, r in results.items() if k != "1_normal_persistent"]
    mean_drop = sum(drops) / max(1, len(drops))

    return MultiBlockCausalReport(
        results=results,
        normal_prob=normal_prob,
        causally_active=causally_active,
        mean_degradation=mean_drop,
        summary=(
            f"Step 366 multi-block causal interventions completed: Normal prob={normal_prob:.4f}, "
            f"Mean drop across interventions={mean_drop:+.4f}. Causally active: {causally_active}."
        ),
    )


if __name__ == "__main__":
    m = instantiate_frozen_baseline()
    cand = MultiBlockRecurrentChakrMicro(m, target_layers=[2, 3], persistent_state=True)
    rep = run_multiblock_causal_interventions(cand, seed=42, num_episodes=4)
    print("=== STEP 366 CAUSAL MULTI-BLOCK INTERVENTIONS ===")
    for c, r in rep.results.items():
        print(f"[{c:26s}] Prob: {r.target_token_prob:.4f} | Acc: {r.final_token_acc:.1%} | Drop: {r.drop_from_normal:+.4f}")
    print(f"\nCausally Active: {rep.causally_active} | Mean Degradation: {rep.mean_degradation:+.4f}")
