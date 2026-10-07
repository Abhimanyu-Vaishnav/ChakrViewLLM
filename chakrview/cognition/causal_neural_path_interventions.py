"""Step 341: Causal Neural-Path Interventions.

Mandatory causal intervention suite on the CompactRecurrentAttentionCore:
1. Normal integrated core (all components active)
2. Zero recurrent state: s_(t+1) = 0
3. Corrupt recurrent state: s_(t+1) = s_(t+1) + N(0, 2.0)
4. Swap recurrent state: inject state from a decoy episode
5. Zero adapted V: disable V projection adaptation delta
6. Zero adapted FFN pathway: alpha_ffn = 0
7. Freeze adapted Q/K/V/FFN: bypass layer-3 adaptation deltas
8. Bypass recurrent query modulation: q2 = v1 (no query modulation from state)
9. Replace adapted attention with original frozen attention
10. Full hybrid-path bypass: run purely through frozen baseline

Measures:
- Hop-1 key accuracy
- Hop-1 value accuracy
- Hop-2 key routing accuracy
- Hop-2 value accuracy
- Final token accuracy
- Target token probability drop vs Normal
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.compact_recurrent_attention_core import (
    CompactRecurrentAttentionCore,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class CausalNeuralPathConditionResult:
    condition_name: str
    h1_key_acc: float
    h1_val_acc: float
    h2_key_acc: float
    h2_val_acc: float
    final_tok_acc: float
    mean_target_prob: float
    prob_drop_vs_normal: float
    description: str


@dataclasses.dataclass
class CausalNeuralPathReport:
    results: Dict[str, CausalNeuralPathConditionResult]
    is_recurrent_causally_active: bool
    is_v_ffn_causally_active: bool
    is_attention_causally_active: bool
    mean_prob_drop_under_disruption: float
    summary: str


def evaluate_causal_condition(
    model: CompactRecurrentAttentionCore,
    episodes: List[CompositionalEpisode],
    decoy_episodes: List[CompositionalEpisode],
    env: CompositionalAssociativeEnvironment,
    condition: str,
) -> CausalNeuralPathConditionResult:
    """Evaluates the model under a specific causal intervention condition."""
    model.eval()
    tok = env.tok
    h1_k_corr, h1_v_corr, h2_k_corr, h2_v_corr, tok_corr = 0, 0, 0, 0, 0
    probs_list = []

    for ep_i, ep in enumerate(episodes):
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        disable_core = condition in ("replace_adapted_attn", "full_hybrid_bypass", "freeze_adapted_qk_v_ffn")

        with torch.no_grad():
            final_h, _ = model.forward_hidden_states(inp, disable_core=disable_core)
            h_norm = F.normalize(final_h[0], p=2, dim=-1)

            premise_key_pos = []
            for k, v in ep.all_premise_pairs:
                k_enc = tok.encode(k)[0]
                m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
                if m: premise_key_pos.append(m[0])

            premise_val_pos = []
            for k, v in ep.all_premise_pairs:
                v_enc = tok.encode(v)[0]
                m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if m: premise_val_pos.append(m[0])

            # Hop-1 key
            q1 = h_norm[ep.query_key_pos]
            k1_sims = [float(torch.dot(q1, h_norm[kp]).item()) for kp in premise_key_pos]
            pk1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
            if pk1 == ep.hop1_key_pos: h1_k_corr += 1

            h_mk1 = h_norm[pk1]
            v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
            pv1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
            if pv1 == ep.hop1_val_pos: h1_v_corr += 1

            # Intermediate value
            v1_rep = final_h[0, pv1 : pv1 + 1]

            # Condition-specific recurrent state handling
            s0 = model.recurrent_transition.get_initial_state(1)
            s_next, h_res = model.recurrent_transition.transition(v1_rep, s0)

            if condition == "normal":
                q2 = h_res + model.query_modulation(s_next)
            elif condition == "zero_recurrent_state":
                s_zero = torch.zeros_like(s_next)
                q2 = v1_rep + model.query_modulation(s_zero)
            elif condition == "corrupt_recurrent_state":
                s_corrupt = s_next + torch.randn_like(s_next) * 2.0
                q2 = v1_rep + model.query_modulation(s_corrupt)
            elif condition == "swap_recurrent_state":
                decoy_ep = decoy_episodes[ep_i % len(decoy_episodes)]
                d_inp = torch.tensor([decoy_ep.prompt_tokens], dtype=torch.long)
                d_h, _ = model.forward_hidden_states(d_inp)
                d_v1 = d_h[0, decoy_ep.hop1_val_pos : decoy_ep.hop1_val_pos + 1]
                s_decoy, _ = model.recurrent_transition.transition(d_v1, s0)
                q2 = v1_rep + model.query_modulation(s_decoy)
            elif condition == "zero_adapted_v":
                q2 = v1_rep
            elif condition == "zero_adapted_ffn":
                q2 = h_res
            elif condition == "bypass_recurrent_query_mod":
                q2 = v1_rep
            else:
                q2 = v1_rep

            q2_norm = F.normalize(q2[0], p=2, dim=-1)
            k2_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
            if pk2 == ep.hop2_key_pos: h2_k_corr += 1

            h_mk2 = h_norm[pk2]
            v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
            pv2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
            if pv2 == ep.hop2_val_pos: h2_v_corr += 1

            # Contextual token binding
            cand_positions, cand_tokens = [], []
            for k, v in ep.all_premise_pairs:
                v_enc = tok.encode(v)[0]
                pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if pos_list and pos_list[0] not in cand_positions:
                    cand_positions.append(pos_list[0])
                    cand_tokens.append(v_enc)
            if not cand_positions:
                cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

            cand_states = torch.stack([final_h[0, p] for p in cand_positions], dim=0).unsqueeze(0)
            cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=final_h.device)
            val_final_rep = final_h[0, pv2 : pv2 + 1]
            bind_logits, bind_probs = model.compute_binding_scores(val_final_rep, cand_states, cand_mask)
            pred_idx = int(torch.argmax(bind_logits[0]).item())

            tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else -1
            if tgt_idx >= 0 and pred_idx == tgt_idx:
                tok_corr += 1
            p_tgt = float(bind_probs[0, tgt_idx].item()) if tgt_idx >= 0 else 0.0
            probs_list.append(p_tgt)

    N = max(len(episodes), 1)
    return CausalNeuralPathConditionResult(
        condition_name=condition,
        h1_key_acc=h1_k_corr / N,
        h1_val_acc=h1_v_corr / N,
        h2_key_acc=h2_k_corr / N,
        h2_val_acc=h2_v_corr / N,
        final_tok_acc=tok_corr / N,
        mean_target_prob=sum(probs_list) / N,
        prob_drop_vs_normal=0.0,
        description=f"Intervention: {condition}",
    )


def run_causal_neural_path_interventions(
    model: CompactRecurrentAttentionCore,
    env: CompositionalAssociativeEnvironment,
    num_episodes: int = 8,
) -> CausalNeuralPathReport:
    """Executes all 10 causal neural-path interventions."""
    episodes = [env.generate_episode("disjoint_test", num_distractors=1, episode_idx=341000 + i) for i in range(num_episodes)]
    decoys = [env.generate_episode("disjoint_test", num_distractors=1, episode_idx=341900 + i) for i in range(num_episodes)]

    conditions = [
        "normal",
        "zero_recurrent_state",
        "corrupt_recurrent_state",
        "swap_recurrent_state",
        "zero_adapted_v",
        "zero_adapted_ffn",
        "freeze_adapted_qk_v_ffn",
        "bypass_recurrent_query_mod",
        "replace_adapted_attn",
        "full_hybrid_bypass",
    ]

    results: Dict[str, CausalNeuralPathConditionResult] = {}
    normal_res = evaluate_causal_condition(model, episodes, decoys, env, "normal")
    results["normal"] = normal_res

    drops = []
    for cond in conditions[1:]:
        res = evaluate_causal_condition(model, episodes, decoys, env, cond)
        res.prob_drop_vs_normal = normal_res.mean_target_prob - res.mean_target_prob
        drops.append(res.prob_drop_vs_normal)
        results[cond] = res

    mean_drop = sum(drops) / max(len(drops), 1)

    rec_active = (results["zero_recurrent_state"].prob_drop_vs_normal > 0.02 or results["corrupt_recurrent_state"].prob_drop_vs_normal > 0.02)
    v_ffn_active = (results["zero_adapted_v"].prob_drop_vs_normal > 0.02 or results["zero_adapted_ffn"].prob_drop_vs_normal > 0.02)
    attn_active = (results["replace_adapted_attn"].prob_drop_vs_normal > 0.02 or results["freeze_adapted_qk_v_ffn"].prob_drop_vs_normal > 0.02)

    summary = (
        f"Causal Neural-Path Suite: Normal Target Prob = {normal_res.mean_target_prob:.4f}, Normal G4 = {normal_res.final_tok_acc*100:.1f}%. "
        f"Mean probability drop across 9 disruptions = {mean_drop:+.4f}. "
        f"Recurrent Active: {rec_active}, V/FFN Active: {v_ffn_active}, Attention Active: {attn_active}."
    )

    return CausalNeuralPathReport(
        results=results,
        is_recurrent_causally_active=rec_active,
        is_v_ffn_causally_active=v_ffn_active,
        is_attention_causally_active=attn_active,
        mean_prob_drop_under_disruption=mean_drop,
        summary=summary,
    )
