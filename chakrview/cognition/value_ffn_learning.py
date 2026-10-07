"""Step 339: Value + FFN Learning Ablation Study.

Isolates content transformation across three configurations:
Config A: Q/K only (baseline comparison from Wave 329-336)
Config B: Q/K/V adaptation (adding value projection adaptation)
Config C: Q/K/V + compact FFN bottleneck (full content transformation pathway)

Evaluates:
- Hop-1 key routing accuracy
- Hop-1 value routing accuracy
- Intermediate state separation / quality
- Hop-2 key routing accuracy
- Final token accuracy on G1-G4

Objective:
Empirically test whether adapting V and FFN enables true intermediate content
transformation before enabling full recurrence.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.compact_recurrent_attention_core import (
    CompactRecurrentAttentionCore,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class ValueFFNAblationMetrics:
    config_name: str
    trainable_params: int
    h1_key_acc: float
    h1_val_acc: float
    inter_separation: float
    h2_key_acc: float
    mean_g4_acc: float
    summary: str


@dataclasses.dataclass
class ValueFFNAblationReport:
    metrics_by_config: Dict[str, ValueFFNAblationMetrics]
    best_config: str
    v_ffn_beneficial: bool
    summary: str


def train_and_eval_config(
    base_model: ChakrMicro,
    config_name: str,
    seed: int = 42,
    train_steps: int = 15,
    eval_episodes: int = 6,
) -> ValueFFNAblationMetrics:
    """Trains and tests one content-transformation configuration."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    if config_name == "QK_only":
        model = CompactRecurrentAttentionCore(
            base_model, target_layer=3, rank=16,
            enable_v=False, enable_ffn=False, enable_recurrent=False,
        )
    elif config_name == "QKV":
        model = CompactRecurrentAttentionCore(
            base_model, target_layer=3, rank=16,
            enable_v=True, enable_ffn=False, enable_recurrent=False,
        )
    elif config_name == "QKV_FFN":
        model = CompactRecurrentAttentionCore(
            base_model, target_layer=3, rank=16,
            enable_v=True, enable_ffn=True, enable_recurrent=False,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=2.0e-3, weight_decay=0.01)
    loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    model.train()
    for st in range(train_steps):
        ep = env.generate_episode(split="train", num_distractors=1, episode_idx=339000 + st)
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        opt.zero_grad()

        final_h, _ = model.forward_hidden_states(inp)
        h_norm = F.normalize(final_h[0], p=2, dim=-1)

        premise_key_pos = []
        for k, v in ep.all_premise_pairs:
            k_enc = tok.encode(k)[0]
            m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
            if m: premise_key_pos.append(m[0])

        r_loss = loss_fn(
            adapted_hidden=final_h, logits=final_h,
            query_key_pos=ep.query_key_pos, matching_key_pos=ep.hop1_key_pos,
            distractor_key_positions=premise_key_pos, associated_val_pos=ep.hop1_val_pos,
            target_token=ep.intermediate_token,
        )

        # Hop-2 routing loss
        v1_rep = final_h[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
        v1_norm = F.normalize(v1_rep[0], p=2, dim=-1)
        k2_logits = torch.matmul(v1_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
        l_k2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

        # Dynamic binding loss
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
        tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0
        val_final_rep = final_h[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
        bind_logits, _ = model.compute_binding_scores(val_final_rep, cand_states, cand_mask)
        l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))

        total_loss = r_loss.total_loss + 1.5 * l_k2 + 2.0 * l_bind
        total_loss.backward()
        opt.step()

    # Evaluation on test split (G4)
    model.eval()
    h1_k_corr, h1_v_corr, h2_k_corr, tok_corr = 0, 0, 0, 0
    seps = []

    with torch.no_grad():
        for ev_i in range(eval_episodes):
            ep = env.generate_episode(split="disjoint_test", num_distractors=1, episode_idx=339500 + ev_i)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            final_h, _ = model.forward_hidden_states(inp)
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

            # Separation metric
            t_s = float(torch.dot(v1_norm, h_norm[ep.hop2_key_pos]).item())
            decoy_s = [float(torch.dot(v1_norm, h_norm[kp]).item()) for kp in premise_key_pos if kp != ep.hop2_key_pos]
            avg_d = sum(decoy_s) / max(len(decoy_s), 1) if decoy_s else 0.0
            seps.append(t_s - avg_d)

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
            b_logits, _ = model.compute_binding_scores(val_final_rep, cand_states, cand_mask)
            pred_idx = int(torch.argmax(b_logits[0]).item())
            if cand_tokens[pred_idx] == ep.target_token:
                tok_corr += 1

    N = max(eval_episodes, 1)
    acc = tok_corr / N

    return ValueFFNAblationMetrics(
        config_name=config_name,
        trainable_params=model.trainable_param_count,
        h1_key_acc=h1_k_corr / N,
        h1_val_acc=h1_v_corr / N,
        inter_separation=sum(seps) / len(seps),
        h2_key_acc=h2_k_corr / N,
        mean_g4_acc=acc,
        summary=f"{config_name}: Params={model.trainable_param_count:,}, G4 Acc={acc * 100:.1f}%, H2 Key={h2_k_corr/N * 100:.1f}%",
    )


def run_value_ffn_ablation_study(base_model: ChakrMicro) -> ValueFFNAblationReport:
    """Executes Step 339 ablation study comparing QK, QKV, and QKV+FFN."""
    configs = ["QK_only", "QKV", "QKV_FFN"]
    metrics_by_config = {}

    for cfg in configs:
        metrics_by_config[cfg] = train_and_eval_config(base_model, cfg, seed=42)

    best_cfg = max(configs, key=lambda c: (metrics_by_config[c].mean_g4_acc, metrics_by_config[c].h2_key_acc))
    v_ffn_beneficial = (metrics_by_config["QKV_FFN"].h2_key_acc >= metrics_by_config["QK_only"].h2_key_acc)

    summary = (
        f"Value + FFN Learning Study: Best configuration is '{best_cfg}'. "
        f"QKV+FFN achieved H2 Key Acc = {metrics_by_config['QKV_FFN'].h2_key_acc * 100:.1f}% vs "
        f"QK-only = {metrics_by_config['QK_only'].h2_key_acc * 100:.1f}%. "
        f"V+FFN Beneficial: {v_ffn_beneficial}."
    )

    return ValueFFNAblationReport(
        metrics_by_config=metrics_by_config,
        best_config=best_cfg,
        v_ffn_beneficial=v_ffn_beneficial,
        summary=summary,
    )
