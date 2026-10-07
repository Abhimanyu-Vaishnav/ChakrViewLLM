"""Step 316: Sequential Two-Hop State Update with Relational Memory.

Integrates DifferentiableRelationalMemory into the complete compositional reasoning pipeline:
Causal Sequence:
1. input_ids -> Frozen ChakrMicro -> hidden states h
2. Hop-1 key routing: q1 (query) -> k1 (premise key) -> v1 (premise value)
3. MEMORY WRITE: v1 is written into Relational Memory M_t -> M_(t+1)
4. MEMORY READ: Readout from M_(t+1) produces Hop-2 query q2
5. Hop-2 key routing: q2 -> k2 (premise key) -> v2 (premise value)
6. Dynamic Contextual Token Binding: v2 binds candidate tokens to output answer.

Performs 5 Causal Interventions:
A. Normal Memory: Standard read/write flow.
B. Zeroed Memory: M_(t+1) is zeroed out before read.
C. Corrupted Memory: Gaussian noise injected into M_(t+1) before read.
D. Swapped Memory: State from a decoy episode is injected into M_(t+1).
E. Bypass Memory: Direct continuous bypass without relational memory.
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
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.relational_memory import (
    DifferentiableRelationalMemory,
    compute_module_sha256,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


class ChakrMicroWithRelationalMemory(nn.Module):
    """
    Compositional architecture pairing frozen ChakrMicro with DifferentiableRelationalMemory
    and dynamic contextual token binding.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        num_slots: int = 2,
        d_mem: int = 64,
        rank: int = 16,
        d_bind: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.d_model = base_model.config.d_model

        # Strictly freeze canonical base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Lightweight representation adapter for query-key routing
        from chakrview.cognition.neural_representation_adapter import GatedResidualAdapter
        self.adapter = GatedResidualAdapter(d_model=self.d_model, rank=rank)

        # Relational Memory
        self.memory = DifferentiableRelationalMemory(
            d_model=self.d_model,
            num_slots=num_slots,
            d_mem=d_mem,
        )

        # Dynamic contextual candidate binding
        self.binding = DynamicContextualTokenBinding(
            d_model=self.d_model,
            d_bind=d_bind,
        )

    def forward_backbone(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            h = layer(h, layer_idx=i)
        h = self.base_model.final_norm(h)
        h_adapted = self.adapter(h)
        logits = self.base_model.lm_head(h_adapted)
        return h_adapted, logits

    def compute_binding_scores(
        self,
        retrieved_val_rep: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.binding(retrieved_val_rep, candidate_states, candidate_mask)


@dataclasses.dataclass
class CausalInterventionResults:
    normal_tok_acc: float
    zeroed_tok_acc: float
    corrupted_tok_acc: float
    swapped_tok_acc: float
    bypass_tok_acc: float
    memory_is_causally_active: bool
    summary: str


def run_sequential_state_interventions(
    candidate: ChakrMicroWithRelationalMemory,
    env: CompositionalAssociativeEnvironment,
    num_episodes: int = 10,
    seed: int = 42,
) -> CausalInterventionResults:
    """Executes the 5 causal interventions on memory state across evaluation episodes."""
    candidate.eval()
    tok = env.tok

    corr_normal, corr_zero, corr_corrupt, corr_swap, corr_bypass = 0, 0, 0, 0, 0

    with torch.no_grad():
        for ep_i in range(num_episodes):
            ep = env.generate_episode("disjoint_test", num_distractors=1, episode_idx=950000 + ep_i)
            ep_decoy = env.generate_episode("disjoint_test", num_distractors=1, episode_idx=990000 + ep_i)

            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            h_ad, _ = candidate.forward_backbone(inp)
            h_norm = F.normalize(h_ad[0], p=2, dim=-1)

            # Extract premise positions
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

            # Candidate tokens for dynamic binding
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

            # Hop-1 value vector
            v1_rep = h_ad[0, pred_v1 : pred_v1 + 1]

            # Initial memory state & write
            mem_init = candidate.memory.get_initial_state(1)
            mem_updated, _ = candidate.memory.write(mem_init, v1_rep)

            # Interventions:
            # 1. Normal
            q2_norm_rep, _ = candidate.memory.read(mem_updated, v1_rep)
            q2_norm = F.normalize(q2_norm_rep[0], p=2, dim=-1)
            k2_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
            h_mk2 = h_norm[pred_k2]
            v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
            pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
            val_final = h_ad[0, pred_v2 : pred_v2 + 1]
            b_logits, _ = candidate.compute_binding_scores(val_final, cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(b_logits[0]).item())] == ep.target_token:
                corr_normal += 1

            # 2. Zeroed
            mem_zeroed = torch.zeros_like(mem_updated)
            q2_zero, _ = candidate.memory.read(mem_zeroed, v1_rep)
            q2_z_norm = F.normalize(q2_zero[0], p=2, dim=-1)
            k2_z = [float(torch.dot(q2_z_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_z = premise_key_pos[k2_z.index(max(k2_z))] if k2_z else 0
            vmk2_z = premise_val_pos[[float(torch.dot(h_norm[pk2_z], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_z], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bz_logits, _ = candidate.compute_binding_scores(h_ad[0, vmk2_z : vmk2_z + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bz_logits[0]).item())] == ep.target_token:
                corr_zero += 1

            # 3. Corrupted
            mem_corrupt = mem_updated + torch.randn_like(mem_updated) * 1.5
            q2_c, _ = candidate.memory.read(mem_corrupt, v1_rep)
            q2_c_norm = F.normalize(q2_c[0], p=2, dim=-1)
            k2_c = [float(torch.dot(q2_c_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_c = premise_key_pos[k2_c.index(max(k2_c))] if k2_c else 0
            vmk2_c = premise_val_pos[[float(torch.dot(h_norm[pk2_c], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_c], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bc_logits, _ = candidate.compute_binding_scores(h_ad[0, vmk2_c : vmk2_c + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bc_logits[0]).item())] == ep.target_token:
                corr_corrupt += 1

            # 4. Swapped
            inp_d = torch.tensor([ep_decoy.prompt_tokens], dtype=torch.long)
            h_d, _ = candidate.forward_backbone(inp_d)
            v_d = h_d[0, ep_decoy.hop1_val_pos : ep_decoy.hop1_val_pos + 1]
            mem_swap, _ = candidate.memory.write(mem_init, v_d)
            q2_s, _ = candidate.memory.read(mem_swap, v1_rep)
            q2_s_norm = F.normalize(q2_s[0], p=2, dim=-1)
            k2_s = [float(torch.dot(q2_s_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_s = premise_key_pos[k2_s.index(max(k2_s))] if k2_s else 0
            vmk2_s = premise_val_pos[[float(torch.dot(h_norm[pk2_s], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_s], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bs_logits, _ = candidate.compute_binding_scores(h_ad[0, vmk2_s : vmk2_s + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bs_logits[0]).item())] == ep.target_token:
                corr_swap += 1

            # 5. Bypass
            q2_b_norm = F.normalize(v1_rep[0], p=2, dim=-1)
            k2_b = [float(torch.dot(q2_b_norm, h_norm[kp]).item()) for kp in premise_key_pos]
            pk2_b = premise_key_pos[k2_b.index(max(k2_b))] if k2_b else 0
            vmk2_b = premise_val_pos[[float(torch.dot(h_norm[pk2_b], h_norm[vp]).item()) for vp in premise_val_pos].index(max([float(torch.dot(h_norm[pk2_b], h_norm[vp]).item()) for vp in premise_val_pos]))] if premise_val_pos else 0
            bb_logits, _ = candidate.compute_binding_scores(h_ad[0, vmk2_b : vmk2_b + 1], cand_states, cand_mask)
            if cand_tokens[int(torch.argmax(bb_logits[0]).item())] == ep.target_token:
                corr_bypass += 1

    norm_acc = corr_normal / num_episodes
    zero_acc = corr_zero / num_episodes
    corrupt_acc = corr_corrupt / num_episodes
    swap_acc = corr_swap / num_episodes
    bypass_acc = corr_bypass / num_episodes

    is_active = (norm_acc > zero_acc or norm_acc > corrupt_acc or norm_acc > swap_acc)

    summary = (
        f"Causal Interventions on Relational Memory: "
        f"Normal={norm_acc*100:.1f}%, Zeroed={zero_acc*100:.1f}%, "
        f"Corrupted={corrupt_acc*100:.1f}%, Swapped={swap_acc*100:.1f}%, "
        f"Bypass={bypass_acc*100:.1f}%. "
        f"Memory Causally Active: {is_active}."
    )

    return CausalInterventionResults(
        normal_tok_acc=norm_acc,
        zeroed_tok_acc=zero_acc,
        corrupted_tok_acc=corrupt_acc,
        swapped_tok_acc=swap_acc,
        bypass_tok_acc=bypass_acc,
        memory_is_causally_active=is_active,
        summary=summary,
    )
