"""Step 350: Causal Neural-Core Interventions on Trained Block.

Executes 8 causal interventions directly on the trained TransformerBlock:
1. Normal trained block (unmodified forward pass)
2. Replace trained block with original frozen baseline block
3. Zero attention contribution in trained block (skip attention residual)
4. Zero V contribution (zero out v_proj output)
5. Zero FFN contribution (skip FFN residual)
6. Bypass trained block completely
7. Shuffle trained block output along hidden dimension
8. Restore baseline block

Measures:
- Hop-1 key accuracy
- Hop-1 value accuracy
- Hop-2 key routing accuracy
- Hop-2 value accuracy
- Final token probability on target
- Final token accuracy
- Target probability drop vs Normal
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.trainable_transformer_block import (
    TrainableTransformerBlockCandidate,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class CoreInterventionResult:
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
class CoreCausalInterventionReport:
    results: Dict[str, CoreInterventionResult]
    is_trained_block_causally_active: bool
    mean_prob_drop: float
    summary: str


def evaluate_core_intervention(
    candidate: TrainableTransformerBlockCandidate,
    base_model: ChakrMicro,
    episodes: List[CompositionalEpisode],
    env: CompositionalAssociativeEnvironment,
    condition: str,
    target_layer: int = 3,
) -> CoreInterventionResult:
    """Evaluates candidate model under specified block causal intervention."""
    candidate.eval()
    tok = env.tok
    h1_k_corr, h1_v_corr, h2_k_corr, h2_v_corr, tok_corr = 0, 0, 0, 0, 0
    probs_list = []

    block = candidate.model.layers[target_layer]
    base_block = base_model.layers[target_layer]

    for ep in episodes:
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        B, T = inp.shape

        with torch.no_grad():
            x = candidate.model.embedding(inp)
            for i, layer in enumerate(candidate.model.layers):
                if i == target_layer:
                    if condition == "normal":
                        x = layer(x, layer_idx=i)
                    elif condition in ("replace_with_baseline", "restore_baseline"):
                        x = base_block(x, layer_idx=i)
                    elif condition == "zero_attn_contribution":
                        # Skip attention residual, execute only FFN
                        x = x + layer.ffn(layer.norm_2(x))
                    elif condition == "zero_v_contribution":
                        # Run attention with zeroed V
                        norm_x = layer.norm_1(x)
                        attn = layer.attn
                        q = attn.rotary(attn.q_proj(norm_x).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2), T)
                        k = attn.rotary(attn.k_proj(norm_x).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2), T)
                        v_zero = torch.zeros_like(attn.v_proj(norm_x).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2))
                        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale + attn.causal_mask(T)
                        context = torch.matmul(torch.softmax(scores, dim=-1), v_zero).transpose(1, 2).contiguous().view(B, T, candidate.d_model)
                        x = x + attn.out_proj(context)
                        x = x + layer.ffn(layer.norm_2(x))
                    elif condition == "zero_ffn_contribution":
                        # Skip FFN residual
                        x = x + layer.attn(layer.norm_1(x), layer_idx=i)
                    elif condition == "bypass_trained_block":
                        # Skip entire block
                        pass
                    elif condition == "shuffle_block_output":
                        x_out = layer(x, layer_idx=i)
                        perm = torch.randperm(x_out.shape[-1])
                        x = x_out[..., perm]
                    else:
                        x = layer(x, layer_idx=i)
                else:
                    x = layer(x, layer_idx=i)

            final_h = candidate.model.final_norm(x)
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

            # Hop-1 val
            h_mk1 = h_norm[pk1]
            v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
            pv1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
            if pv1 == ep.hop1_val_pos: h1_v_corr += 1

            # Hop-2 key
            v1_rep = final_h[0, pv1 : pv1 + 1]
            v1_norm = F.normalize(v1_rep[0], p=2, dim=-1)
            k2_sims = [float(torch.dot(v1_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
            if pk2 == ep.hop2_key_pos: h2_k_corr += 1

            # Hop-2 val
            h_mk2 = h_norm[pk2]
            v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
            pv2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
            if pv2 == ep.hop2_val_pos: h2_v_corr += 1

            # Binding
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
            bind_logits, bind_probs = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)
            pred_idx = int(torch.argmax(bind_logits[0]).item())

            tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else -1
            if tgt_idx >= 0 and pred_idx == tgt_idx:
                tok_corr += 1
            p_tgt = float(bind_probs[0, tgt_idx].item()) if tgt_idx >= 0 else 0.0
            probs_list.append(p_tgt)

    N = max(len(episodes), 1)
    return CoreInterventionResult(
        condition_name=condition,
        h1_key_acc=h1_k_corr / N,
        h1_val_acc=h1_v_corr / N,
        h2_key_acc=h2_k_corr / N,
        h2_val_acc=h2_v_corr / N,
        final_tok_acc=tok_corr / N,
        mean_target_prob=sum(probs_list) / N,
        prob_drop_vs_normal=0.0,
        description=f"Core intervention: {condition}",
    )


def run_core_causal_interventions(
    candidate: TrainableTransformerBlockCandidate,
    base_model: ChakrMicro,
    env: CompositionalAssociativeEnvironment,
    num_episodes: int = 8,
    target_layer: int = 3,
) -> CoreCausalInterventionReport:
    """Executes all 8 causal interventions on the trained block."""
    episodes = [env.generate_episode("disjoint_test", num_distractors=1, episode_idx=350000 + i) for i in range(num_episodes)]

    conditions = [
        "normal",
        "replace_with_baseline",
        "zero_attn_contribution",
        "zero_v_contribution",
        "zero_ffn_contribution",
        "bypass_trained_block",
        "shuffle_block_output",
        "restore_baseline",
    ]

    results: Dict[str, CoreInterventionResult] = {}
    norm_res = evaluate_core_intervention(candidate, base_model, episodes, env, "normal", target_layer)
    results["normal"] = norm_res

    drops = []
    for cond in conditions[1:]:
        res = evaluate_core_intervention(candidate, base_model, episodes, env, cond, target_layer)
        res.prob_drop_vs_normal = norm_res.mean_target_prob - res.mean_target_prob
        drops.append(res.prob_drop_vs_normal)
        results[cond] = res

    mean_drop = sum(drops) / max(len(drops), 1)
    active = (results["replace_with_baseline"].prob_drop_vs_normal > 0.02 or results["bypass_trained_block"].prob_drop_vs_normal > 0.02)

    summary = (
        f"Core Causal Interventions: Normal Target Prob = {norm_res.mean_target_prob:.4f}, Acc = {norm_res.final_tok_acc*100:.1f}%. "
        f"Mean probability drop under disruptions = {mean_drop:+.4f}. "
        f"Trained Block Causally Active: {active}."
    )

    return CoreCausalInterventionReport(
        results=results,
        is_trained_block_causally_active=active,
        mean_prob_drop=mean_drop,
        summary=summary,
    )
