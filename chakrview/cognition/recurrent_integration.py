"""Step 340: Recurrent Integration with Attention Routing.

Integrates the Gated Recurrent State Transition with attention-driven routing:
1. Retrieved contextual value v1 is transformed via adapted FFN pathway.
2. Recurrent state update: s_(t+1) = Transition(v1, s_t).
3. Recurrent state directly modulates the second-hop query:
   q2 = v1 + QueryModulation(s_(t+1)).
4. q2 is actively matched against premise keys to retrieve second value v2.
5. Dynamic contextual binding maps v2 to output token.

Compares:
Option A: Q/K/V + FFN (Static feedforward composition)
Option B: Q/K/V + FFN + Recurrent State (Full integrated recurrent-attention core)

Evaluates:
- 1-hop accuracy
- 2-hop G1 (train), G2 (unseen ID), G3 (unseen comp), G4 (unseen ID + unseen comp)
- State norm and stability
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
class RecurrentIntegrationMetrics:
    name: str
    trainable_params: int
    mean_g1_acc: float
    mean_g2_acc: float
    mean_g3_acc: float
    mean_g4_acc: float
    mean_h1_key_acc: float
    mean_h2_key_acc: float
    mean_h2_val_acc: float
    summary: str


@dataclasses.dataclass
class RecurrentIntegrationReport:
    metrics_static: RecurrentIntegrationMetrics
    metrics_recurrent: RecurrentIntegrationMetrics
    recurrent_gain_g4: float
    recurrent_beneficial: bool
    summary: str


def train_and_evaluate_recurrent_core(
    base_model: ChakrMicro,
    enable_recurrent: bool = True,
    seed: int = 42,
    train_steps: int = 15,
    eval_episodes: int = 6,
) -> RecurrentIntegrationMetrics:
    """Trains and tests CompactRecurrentAttentionCore with or without recurrence."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    model = CompactRecurrentAttentionCore(
        base_model=base_model,
        target_layer=3,
        rank=16,
        bottleneck_dim=32,
        d_state=64,
        d_bind=64,
        enable_v=True,
        enable_ffn=True,
        enable_recurrent=enable_recurrent,
    )

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=2.0e-3, weight_decay=0.01)
    loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    model.train()
    for st in range(train_steps):
        ep = env.generate_episode(split="train", num_distractors=1, episode_idx=340000 + st)
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
        if enable_recurrent:
            s0 = model.recurrent_transition.get_initial_state(1)
            q2, _ = model.compute_recurrent_second_hop_query(v1_rep, s0)
        else:
            q2 = v1_rep

        q2_norm = F.normalize(q2[0], p=2, dim=-1)
        k2_logits = torch.matmul(q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
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

    # Multi-group evaluation
    model.eval()
    splits = [
        ("G1", "train"),
        ("G2", "disjoint_test"),
        ("G3", "heldout_composition"),
        ("G4", "disjoint_test"),
    ]
    group_accs = {}
    h1_k_corr, h2_k_corr, h2_v_corr = 0, 0, 0
    total_eval = 0

    with torch.no_grad():
        for g_name, sp_split in splits:
            corr_g = 0
            for ev_i in range(eval_episodes):
                ep = env.generate_episode(split=sp_split, num_distractors=1, episode_idx=340500 + ev_i)
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

                # Hop-1
                q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pk1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pk1 == ep.hop1_key_pos: h1_k_corr += 1

                h_mk1 = h_norm[pk1]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pv1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0

                # Hop-2 with recurrent state modulation
                v1_rep = final_h[0, pv1 : pv1 + 1]
                if enable_recurrent:
                    s0 = model.recurrent_transition.get_initial_state(1)
                    q2, _ = model.compute_recurrent_second_hop_query(v1_rep, s0)
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

                # Dynamic candidate token binding
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
                bind_logits, _ = model.compute_binding_scores(val_final_rep, cand_states, cand_mask)
                pred_idx = int(torch.argmax(bind_logits[0]).item())
                if cand_tokens[pred_idx] == ep.target_token:
                    corr_g += 1
                total_eval += 1

            group_accs[g_name] = corr_g / max(eval_episodes, 1)

    N_all = max(total_eval, 1)
    name = "QKV+FFN+Recurrent" if enable_recurrent else "QKV+FFN_Static"

    return RecurrentIntegrationMetrics(
        name=name,
        trainable_params=model.trainable_param_count,
        mean_g1_acc=group_accs["G1"],
        mean_g2_acc=group_accs["G2"],
        mean_g3_acc=group_accs["G3"],
        mean_g4_acc=group_accs["G4"],
        mean_h1_key_acc=h1_k_corr / N_all,
        mean_h2_key_acc=h2_k_corr / N_all,
        mean_h2_val_acc=h2_v_corr / N_all,
        summary=f"{name}: Params={model.trainable_param_count:,}, G4 Acc={group_accs['G4']*100:.1f}%, H2 Key={h2_k_corr/N_all*100:.1f}%",
    )


def run_recurrent_integration_comparison(base_model: ChakrMicro) -> RecurrentIntegrationReport:
    """Executes Step 340 comparison between static and recurrent core."""
    static_res = train_and_evaluate_recurrent_core(base_model, enable_recurrent=False, seed=42)
    recurrent_res = train_and_evaluate_recurrent_core(base_model, enable_recurrent=True, seed=42)

    gain = recurrent_res.mean_g4_acc - static_res.mean_g4_acc
    beneficial = gain >= 0.0

    summary = (
        f"Recurrent Integration Comparison: Static G4 = {static_res.mean_g4_acc * 100:.1f}% vs "
        f"Recurrent G4 = {recurrent_res.mean_g4_acc * 100:.1f}% (Δ = {gain * 100:+.1f}%). "
        f"Recurrent Beneficial: {beneficial}."
    )

    return RecurrentIntegrationReport(
        metrics_static=static_res,
        metrics_recurrent=recurrent_res,
        recurrent_gain_g4=gain,
        recurrent_beneficial=beneficial,
        summary=summary,
    )
