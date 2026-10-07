"""Step 324: Two-Step Recurrent Composition & Causal Interventions.

Implements the complete sequential recurrent composition pipeline:
1. input_ids -> Frozen ChakrMicro -> neural representations h
2. Hop-1 key routing: q1 (query) -> k1 (premise key) -> v1 (premise value)
3. RECURRENT STATE TRANSITION: s_(t+1) = Transition(v1, s_t)
4. Hop-2 neural computation conditioned on s_(t+1) -> Hop-2 query q2
5. Hop-2 key routing: q2 -> k2 (premise key) -> v2 (premise value)
6. Dynamic contextual candidate binding: v2 binds candidate tokens to output.

Executes 5 Causal Interventions on the recurrent state s_(t+1):
1. Normal State: Standard s_(t+1) propagation.
2. Zero State: s_(t+1) is zeroed out before Hop-2 query formation.
3. Corrupted State: Large Gaussian noise is injected into s_(t+1).
4. Swapped State: State from a decoy episode is injected into s_(t+1).
5. Bypass State: Recurrent state is completely bypassed (raw v1 used directly).
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
from chakrview.cognition.state_attention_integration import (
    StateAttentionIntegrationModel,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class RecurrentCausalResults:
    normal_tok_acc: float
    zeroed_tok_acc: float
    corrupted_tok_acc: float
    swapped_tok_acc: float
    bypass_tok_acc: float
    state_causally_active: bool
    summary: str


def run_recurrent_causal_interventions(
    model: StateAttentionIntegrationModel,
    env: CompositionalAssociativeEnvironment,
    num_episodes: int = 10,
    seed: int = 42,
) -> RecurrentCausalResults:
    """Executes the 5 causal interventions on recurrent state across evaluation episodes."""
    model.eval()
    tok = env.tok

    corr_norm, corr_zero, corr_corrupt, corr_swap, corr_bypass = 0, 0, 0, 0, 0

    with torch.no_grad():
        for ep_i in range(num_episodes):
            ep = env.generate_episode("disjoint_test", num_distractors=1, episode_idx=1500000 + ep_i)
            ep_decoy = env.generate_episode("disjoint_test", num_distractors=1, episode_idx=1590000 + ep_i)

            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            h_ad, _ = model.forward_backbone(inp)
            h_norm = F.normalize(h_ad[0], p=2, dim=-1)

            # Premise key positions
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

            cand_positions, cand_tokens = [], []
            for k, v in ep.all_premise_pairs:
                v_enc = tok.encode(v)[0]
                pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if pos_list and pos_list[0] not in cand_positions:
                    cand_positions.append(pos_list[0])
                    cand_tokens.append(v_enc)
            if not cand_positions:
                cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

            cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
            cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

            # Hop-1 routing
            h_q1 = h_norm[ep.query_key_pos]
            k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
            h_mk1 = h_norm[pred_k1]
            v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
            pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0

            v1_rep = h_ad[0, pred_v1 : pred_v1 + 1]
            s0 = model.transition_core.get_initial_state(1)
            s_next, _ = model.transition_core.transition(v1_rep, s0)

            # 1. Normal state
            q2_norm = v1_rep + model.mode_proj(s_next)
            q2_norm_n = F.normalize(q2_norm[0], p=2, dim=-1)
            k2_sims = [float(torch.dot(q2_norm_n, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
            h_mk2 = h_norm[pred_k2]
            v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
            pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
            b_log, _ = model.binding(h_ad[0, pred_v2 : pred_v2 + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(b_log[0]).item())] == ep.target_token:
                corr_norm += 1

            # 2. Zeroed state
            s_zero = torch.zeros_like(s_next)
            q2_zero = v1_rep + model.mode_proj(s_zero)
            q2_z_n = F.normalize(q2_zero[0], p=2, dim=-1)
            k2_z = [float(torch.dot(q2_z_n, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_z = premise_key_pos[k2_z.index(max(k2_z))] if k2_z else 0
            pv2_z = premise_val_pos[[float(torch.dot(h_norm[pk2_z], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_z], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bz_log, _ = model.binding(h_ad[0, pv2_z : pv2_z + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bz_log[0]).item())] == ep.target_token:
                corr_zero += 1

            # 3. Corrupted state
            s_corrupt = s_next + torch.randn_like(s_next) * 2.0
            q2_c = v1_rep + model.mode_proj(s_corrupt)
            q2_c_n = F.normalize(q2_c[0], p=2, dim=-1)
            k2_c = [float(torch.dot(q2_c_n, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_c = premise_key_pos[k2_c.index(max(k2_c))] if k2_c else 0
            pv2_c = premise_val_pos[[float(torch.dot(h_norm[pk2_c], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_c], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bc_log, _ = model.binding(h_ad[0, pv2_c : pv2_c + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bc_log[0]).item())] == ep.target_token:
                corr_corrupt += 1

            # 4. Swapped state
            inp_decoy = torch.tensor([ep_decoy.prompt_tokens], dtype=torch.long)
            h_decoy, _ = model.forward_backbone(inp_decoy)
            v_decoy = h_decoy[0, ep_decoy.hop1_val_pos : ep_decoy.hop1_val_pos + 1]
            s_swap, _ = model.transition_core.transition(v_decoy, s0)
            q2_s = v1_rep + model.mode_proj(s_swap)
            q2_s_n = F.normalize(q2_s[0], p=2, dim=-1)
            k2_s = [float(torch.dot(q2_s_n, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_s = premise_key_pos[k2_s.index(max(k2_s))] if k2_s else 0
            pv2_s = premise_val_pos[[float(torch.dot(h_norm[pk2_s], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_s], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bs_log, _ = model.binding(h_ad[0, pv2_s : pv2_s + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bs_log[0]).item())] == ep.target_token:
                corr_swap += 1

            # 5. Bypass state
            q2_b = v1_rep
            q2_b_n = F.normalize(q2_b[0], p=2, dim=-1)
            k2_b = [float(torch.dot(q2_b_n, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_b = premise_key_pos[k2_b.index(max(k2_b))] if k2_b else 0
            pv2_b = premise_val_pos[[float(torch.dot(h_norm[pk2_b], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_b], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bb_log, _ = model.binding(h_ad[0, pv2_b : pv2_b + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bb_log[0]).item())] == ep.target_token:
                corr_bypass += 1

    norm_acc = corr_norm / num_episodes
    zero_acc = corr_zero / num_episodes
    corrupt_acc = corr_corrupt / num_episodes
    swap_acc = corr_swap / num_episodes
    bypass_acc = corr_bypass / num_episodes

    is_active = (norm_acc > zero_acc or norm_acc > corrupt_acc or norm_acc > swap_acc)

    summary = (
        f"Recurrent State Causal Interventions: "
        f"Normal={norm_acc*100:.1f}%, Zeroed={zero_acc*100:.1f}%, "
        f"Corrupted={corrupt_acc*100:.1f}%, Swapped={swap_acc*100:.1f}%, "
        f"Bypass={bypass_acc*100:.1f}%. "
        f"State Causally Active: {is_active}."
    )

    return RecurrentCausalResults(
        normal_tok_acc=norm_acc,
        zeroed_tok_acc=zero_acc,
        corrupted_tok_acc=corrupt_acc,
        swapped_tok_acc=swap_acc,
        bypass_tok_acc=bypass_acc,
        state_causally_active=is_active,
        summary=summary,
    )
