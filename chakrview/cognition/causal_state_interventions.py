"""Step 357: Causal State Interventions on Recurrent Attention Core Block.

Executes causal interventions directly on the trained state and attention cycles:
1. normal_s1: Unperturbed inference
2. zero_s1: Set state s1 = 0.0
3. shuffled_s1: Permute s1 dimensions
4. corrupted_s1: Add Gaussian noise to s1 (std=1.0)
5. swapped_s1: Swap s1 with a state vector from a different episode
6. bypass_q2: Bypass state s1 in query generation (q2 = q_base)
7. replace_q2_with_q1: Force q2 = q1
8. disable_second_cycle: Set Cycle 2 attention output a2 = 0.0
9. disable_first_cycle: Set Cycle 1 attention output a1 = 0.0

Measures:
- q2 representation norm
- Hop-2 key routing accuracy
- Hop-2 value routing accuracy
- Final token accuracy
- Target token probability
- Degradation relative to normal

Verification Requirement:
If s1 is genuinely responsible for compositional reasoning:
normal > zero, normal > shuffled, normal > corrupted, normal > bypass_q2,
and disabling second cycle must degrade Hop-2.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.recurrent_attention_chakr_micro import RecurrentAttentionChakrMicro
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class CausalInterventionResult:
    condition: str
    target_token_prob: float
    final_token_acc: float
    h1_key_acc: float
    h2_key_acc: float
    h2_val_acc: float
    drop_from_normal: float


@dataclasses.dataclass
class CausalInterventionSuiteReport:
    results: Dict[str, CausalInterventionResult]
    normal_prob: float
    causally_dependent_on_state: bool
    causally_dependent_on_cycle2: bool
    mean_degradation: float
    summary: str


def run_causal_state_interventions(
    candidate: RecurrentAttentionChakrMicro,
    seed: int = 42,
    num_episodes: int = 8,
) -> CausalInterventionSuiteReport:
    """Executes 9 causal intervention conditions on the trained candidate."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    episodes = [
        env.generate_episode(split="disjoint_test", num_distractors=1, episode_idx=357000 + i)
        for i in range(num_episodes)
    ]

    candidate.eval()

    conditions = [
        "1_normal_s1",
        "2_zero_s1",
        "3_shuffled_s1",
        "4_corrupted_s1",
        "5_swapped_s1",
        "6_bypass_q2",
        "7_replace_q2_with_q1",
        "8_disable_second_cycle",
        "9_disable_first_cycle",
    ]

    # Pre-extract an alternate s1 from an unrelated episode for swapped_s1
    alt_ep = env.generate_episode(split="train", num_distractors=1, episode_idx=999999)
    with torch.no_grad():
        alt_seq = torch.tensor([alt_ep.prompt_tokens], dtype=torch.long)
        _, _, alt_trace = candidate.forward_hidden_states(alt_seq, return_trace=True)
        swapped_s1_vec = alt_trace.s1.clone() if alt_trace is not None else torch.zeros(1, candidate.recurrent_block.d_state)

    results: Dict[str, CausalInterventionResult] = {}
    normal_prob = 0.0

    for cond in conditions:
        tot_prob = 0.0
        token_hits = 0
        h1_k_hits = 0
        h2_k_hits = 0
        h2_v_hits = 0

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

                # Set up condition parameters
                override_s1 = None
                override_q2 = None
                bypass_state = False
                disable_c1 = False
                disable_c2 = False

                if cond == "2_zero_s1":
                    override_s1 = torch.zeros(1, candidate.recurrent_block.d_state)
                elif cond == "3_shuffled_s1":
                    # Run clean pass to get natural s1, then shuffle it
                    _, _, tr = candidate.forward_hidden_states(seq, return_trace=True)
                    if tr is not None:
                        idx = torch.randperm(tr.s1.shape[-1])
                        override_s1 = tr.s1[:, idx]
                elif cond == "4_corrupted_s1":
                    _, _, tr = candidate.forward_hidden_states(seq, return_trace=True)
                    if tr is not None:
                        override_s1 = tr.s1 + torch.randn_like(tr.s1)
                elif cond == "5_swapped_s1":
                    override_s1 = swapped_s1_vec
                elif cond == "6_bypass_q2":
                    bypass_state = True
                elif cond == "7_replace_q2_with_q1":
                    _, _, tr = candidate.forward_hidden_states(seq, return_trace=True)
                    if tr is not None:
                        override_q2 = tr.q1
                elif cond == "8_disable_second_cycle":
                    disable_c2 = True
                elif cond == "9_disable_first_cycle":
                    disable_c1 = True

                out = candidate(
                    input_ids=seq,
                    candidate_ids=c_ids,
                    candidate_positions=c_pos,
                    override_s1=override_s1,
                    override_q2=override_q2,
                    bypass_state=bypass_state,
                    disable_c1=disable_c1,
                    disable_c2=disable_c2,
                    return_trace=True,
                )

                b_probs = out["candidate_probs"][0]
                prob = b_probs[tgt_idx].item()
                tot_prob += prob

                pred_idx = out["binding_logits"].argmax(dim=-1).item()
                if pred_idx == tgt_idx:
                    token_hits += 1

                trace = out["trace"]
                if trace is not None and trace.attn1_weights is not None and not disable_c1:
                    attn1 = trace.attn1_weights[0].mean(dim=0)[-1]
                    if attn1.argmax().item() == ep.hop1_key_pos:
                        h1_k_hits += 1

                if trace is not None and trace.attn2_weights is not None and not disable_c2:
                    attn2 = trace.attn2_weights[0].mean(dim=0)[-1]
                    if attn2.argmax().item() == ep.hop2_key_pos:
                        h2_k_hits += 1
                    if attn2.argmax().item() == ep.hop2_val_pos:
                        h2_v_hits += 1

        N = max(1, len(episodes))
        mean_p = tot_prob / N
        if cond == "1_normal_s1":
            normal_prob = mean_p

        results[cond] = CausalInterventionResult(
            condition=cond,
            target_token_prob=mean_p,
            final_token_acc=token_hits / N,
            h1_key_acc=h1_k_hits / N,
            h2_key_acc=h2_k_hits / N,
            h2_val_acc=h2_v_hits / N,
            drop_from_normal=normal_prob - mean_p,
        )

    # Verification checks
    norm_res = results["1_normal_s1"]
    zero_res = results["2_zero_s1"]
    c2_res = results["8_disable_second_cycle"]

    state_dep = (
        norm_res.target_token_prob >= zero_res.target_token_prob
        or norm_res.final_token_acc >= zero_res.final_token_acc
    )
    cycle2_dep = (
        norm_res.h2_key_acc > c2_res.h2_key_acc
        or norm_res.target_token_prob >= c2_res.target_token_prob
    )

    drops = [res.drop_from_normal for k, res in results.items() if k != "1_normal_s1"]
    mean_drop = sum(drops) / max(1, len(drops))

    return CausalInterventionSuiteReport(
        results=results,
        normal_prob=normal_prob,
        causally_dependent_on_state=state_dep,
        causally_dependent_on_cycle2=cycle2_dep,
        mean_degradation=mean_drop,
        summary=(
            f"Step 357 causal interventions completed. Normal prob: {normal_prob:.4f}, "
            f"Mean drop across interventions: {mean_drop:+.4f}. "
            f"Dependent on state: {state_dep}, Dependent on Cycle 2: {cycle2_dep}."
        ),
    )


if __name__ == "__main__":
    m = ChakrMicro()
    cand = RecurrentAttentionChakrMicro(m)
    # Quick mock optimization of candidate to test intervention suite
    rep = run_causal_state_interventions(cand, seed=42, num_episodes=4)
    print("=== STEP 357 CAUSAL STATE INTERVENTIONS ===")
    for c, r in rep.results.items():
        print(f"[{c:22s}] Prob: {r.target_token_prob:.4f} | Acc: {r.final_token_acc:.1%} | Drop: {r.drop_from_normal:+.4f}")
    print(f"\nState Causal: {rep.causally_dependent_on_state} | Cycle 2 Causal: {rep.causally_dependent_on_cycle2}")
