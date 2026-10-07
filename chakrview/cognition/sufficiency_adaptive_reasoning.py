"""Step 385: Sufficiency-Based Adaptive Reasoning Core Architecture.

Replaces the blind step-count continuation penalty with an Answer-Sufficiency Neural Controller.
Instead of asking "How many steps have elapsed?", the controller asks:
"Is my current internal representation sufficient to answer?"

Inputs to Sufficiency Estimator:
1. Current recurrent reasoning state s_t [B, d_state]
2. Current active query representation q_rec [B, d_model]
3. Current retrieved contextual answer representation r_curr [B, d_model]
4. (Optional) Candidate distribution features (entropy, top-1 confidence, margin between top-1 and top-2)

Outputs:
- sufficiency_logit: logit indicating whether current representation is sufficient [B, 1]
- sufficiency_prob: sigmoid(sufficiency_logit) in [0, 1]
- continue_prob: 1.0 - sufficiency_prob (more sufficiency -> less need to continue)

Properties:
- Standalone architecture (<550k trainable parameters, <50k added by sufficiency controller)
- Zero answer label leakage
- Zero ground-truth hop-count input
- Zero symbolic lookup
- Differentiable end-to-end
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask
from chakrview.brain.output import LMHead


@dataclasses.dataclass
class SufficiencyCycleTrace:
    cycle_index: int
    w_attn: Optional[torch.Tensor]           # [B, n_heads, T, T]
    state_s: torch.Tensor                     # [B, d_state]
    query_q: torch.Tensor                     # [B, 1, d_model]
    context_r: torch.Tensor                   # [B, d_model]
    sufficiency_score: torch.Tensor           # [B, 1]
    continue_prob: torch.Tensor               # [B, 1]
    continue_logit: torch.Tensor              # [B, 1]
    halt_decision: torch.Tensor               # [B, 1] (bool)
    entropy_stat: Optional[float] = None
    confidence_stat: Optional[float] = None


@dataclasses.dataclass
class SufficiencyCoreOutput:
    vocab_logits: torch.Tensor                # [B, T, vocab_size]
    binding_logits: Optional[torch.Tensor]    # [B, num_candidates]
    binding_probs: Optional[torch.Tensor]     # [B, num_candidates]
    total_cycles_executed: int
    cycle_traces: List[SufficiencyCycleTrace]
    final_state: torch.Tensor                 # [B, d_state]
    p_continue_per_cycle: List[torch.Tensor]
    sufficiency_scores_per_cycle: List[torch.Tensor]
    computation_cost: torch.Tensor


class AnswerSufficiencyController(nn.Module):
    """
    Evaluates whether the internal representation at cycle t is sufficient to answer.
    Inputs:
    - state_t: recurrent state vector [B, d_state]
    - query_t: active query representation [B, d_model]
    - context_t: retrieved answer context [B, d_model]
    - cand_features: optional summary of candidate certainty [B, d_cand_feat] (e.g. top1_p, margin, entropy)

    Outputs:
    - sufficiency_logit: logit for answer sufficiency [B, 1]
    - sufficiency_score: sigmoid(sufficiency_logit) [B, 1]
    - continue_prob: 1.0 - sufficiency_score [B, 1]
    """
    def __init__(self, d_state: int = 48, d_model: int = 96, d_extra: int = 3, hidden_dim: int = 64):
        super().__init__()
        in_dim = d_state + d_model + d_model + d_extra  # 48 + 96 + 96 + 3 = 243
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize so that initial sufficiency is modest (~0.2), encouraging initial exploration
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, -1.0)

    def forward(
        self,
        state_t: torch.Tensor,
        query_t: torch.Tensor,
        context_t: torch.Tensor,
        cand_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = state_t.shape[0]
        if cand_features is None:
            cand_features = torch.zeros(B, 3, device=state_t.device)

        features = torch.cat([state_t, query_t, context_t, cand_features], dim=-1)
        h = self.act(self.fc1(self.norm(features)))
        suff_logit = self.fc2(h)
        suff_score = torch.sigmoid(suff_logit)
        cont_prob = 1.0 - suff_score
        return suff_logit, suff_score, cont_prob


class SufficiencyAdaptiveReasoningCore(nn.Module):
    """
    Unified Recurrent Sequence Core with Answer-Sufficiency Adaptive Reasoning.
    Parameter budget target: < 550k trainable parameters.
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 96,
        n_heads: int = 4,
        d_state: int = 48,
        d_bind: int = 32,
        max_seq_len: int = 512,
        max_cycles: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 24
        self.d_state = d_state
        self.d_bind = d_bind
        self.max_seq_len = max_seq_len
        self.max_cycles = max_cycles

        # 1. Embedding Layer & Pre-Norms
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm_input = RMSNorm(d_model, eps=rms_norm_eps)
        self.norm_cycle = RMSNorm(d_model, eps=rms_norm_eps)

        # 2. Rotary & Causal Masking
        self.rotary = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self.causal_mask = CausalMask(max_seq_len=max_seq_len)

        # 3. Base Input Attention Projections (Cycle 0)
        self.q0_proj = nn.Linear(d_model, d_model, bias=False)
        self.k0_proj = nn.Linear(d_model, d_model, bias=False)
        self.v0_proj = nn.Linear(d_model, d_model, bias=False)
        self.out0_proj = nn.Linear(d_model, d_model, bias=False)

        # 4. Shared Recurrent Reasoning Attention Projections
        self.k_rec_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_rec_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_rec_proj = nn.Linear(d_model, d_model, bias=False)

        # 5. Intermediate Value Extraction & GRU State Transition
        self.norm_r = nn.LayerNorm(d_model)
        self.val_to_state = nn.Linear(d_model, d_state, bias=False)

        self.w_z_h = nn.Linear(d_state, d_state)
        self.w_z_s = nn.Linear(d_state, d_state, bias=False)
        self.w_r_h = nn.Linear(d_state, d_state)
        self.w_r_s = nn.Linear(d_state, d_state, bias=False)
        self.w_n_h = nn.Linear(d_state, d_state)
        self.w_n_s = nn.Linear(d_state, d_state, bias=False)

        self.s0 = nn.Parameter(torch.zeros(1, d_state))

        # 6. State-Conditioned Query Formulation
        self.state_to_q = nn.Linear(d_state, d_model, bias=False)
        self.cycle_gamma = nn.Parameter(torch.tensor([1.0]))

        # 7. Answer-Sufficiency Controller (<20k parameters)
        self.sufficiency_controller = AnswerSufficiencyController(
            d_state=d_state, d_model=d_model, d_extra=3, hidden_dim=64
        )

        # 8. Dynamic Contextual Token Binding Module
        self.norm_bind_val = nn.LayerNorm(d_model)
        self.bind_query_proj = nn.Linear(d_model, d_bind, bias=False)
        self.norm_bind_cand = nn.LayerNorm(d_model)
        self.bind_cand_proj = nn.Linear(d_model, d_bind, bias=False)
        self.bind_scale = nn.Parameter(torch.tensor([4.0]))

        # 9. Standard LM Head (tied to embedding weight directly)
        self.lm_head = LMHead(self.embedding.weight)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initializes weights using truncated normal initialization."""
        std = 0.02
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    @property
    def total_param_count(self) -> int:
        seen = set()
        count = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                count += p.numel()
        return count

    @property
    def trainable_param_count(self) -> int:
        seen = set()
        count = 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                count += p.numel()
        return count

    def _causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head causal attention matching ChakrView standards."""
        B, T, _ = q.shape
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, T)
        k = self.rotary(k, T)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = self.causal_mask(T)
        scores = scores + mask

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -1e9)
            else:
                scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return context, weights

    def compute_dynamic_binding(
        self,
        val_rep: torch.Tensor,
        cand_states: torch.Tensor,
        cand_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes dynamic contextual token binding compatibility."""
        B, N, _ = cand_states.shape
        q = self.bind_query_proj(self.norm_bind_val(val_rep))  # [B, d_bind]
        q = F.normalize(q, p=2, dim=-1)

        k = self.bind_cand_proj(self.norm_bind_cand(cand_states))  # [B, N, d_bind]
        k = F.normalize(k, p=2, dim=-1)

        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.bind_scale.abs()  # [B, N]
        if cand_mask is not None:
            scores = scores.masked_fill(~cand_mask, -1e9)

        probs = torch.softmax(scores, dim=-1)
        return scores, probs

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_positions: Optional[torch.Tensor] = None,
        query_pos: Optional[int] = None,
        max_reasoning_cycles: Optional[int] = None,
        fixed_cycles: Optional[int] = None,
        sufficiency_threshold: float = 0.5,
        override_state: Optional[torch.Tensor] = None,
        override_cycle: Optional[int] = None,
        bypass_controller: bool = False,
        random_controller: bool = False,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass with sufficiency-guided adaptive reasoning cycles.
        """
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)
        max_cycles = max_reasoning_cycles if max_reasoning_cycles is not None else self.max_cycles

        # 1. Input Embedding & Pre-Norm
        x = self.embedding(input_ids)
        h0 = self.norm_input(x)

        # 2. Cycle 0: Base relational sequence representation
        q0 = self.q0_proj(h0)
        k0 = self.k0_proj(h0)
        v0 = self.v0_proj(h0)
        a0_ctx, w0 = self._causal_attention(q0, k0, v0)
        a0 = self.out0_proj(a0_ctx)

        h_curr = h0 + a0  # Current sequence representation [B, T, d_model]
        s_curr = self.s0.expand(B, -1)  # [B, d_state]

        cycle_traces: List[SufficiencyCycleTrace] = []
        p_continue_list: List[torch.Tensor] = []
        suff_score_list: List[torch.Tensor] = []

        active_mask = torch.ones(B, 1, dtype=torch.bool, device=input_ids.device)
        differentiable_cost = torch.zeros(B, 1, device=input_ids.device)

        cycles_to_run = fixed_cycles if fixed_cycles is not None else max_cycles
        executed_cycles = 0

        for cycle_idx in range(1, cycles_to_run + 1):
            executed_cycles += 1

            # A. Value extraction at query position
            h_q = h_curr[:, pos : pos + 1, :]
            r_curr = self.norm_r(h_q).squeeze(1)         # [B, d_model]
            r_feat = self.val_to_state(r_curr)           # [B, d_state]

            # B. GRU State Transition
            z_t = torch.sigmoid(self.w_z_h(r_feat) + self.w_z_s(s_curr))
            r_t = torch.sigmoid(self.w_r_h(r_feat) + self.w_r_s(s_curr))
            n_t = torch.tanh(self.w_n_h(r_feat) + self.w_n_s(r_t * s_curr))
            s_next = (1.0 - z_t) * s_curr + z_t * n_t

            # Apply intervention if requested
            if override_state is not None and (override_cycle is None or override_cycle == cycle_idx):
                s_next = override_state

            # C. State-Conditioned Query Formulation
            q_mod = self.state_to_q(s_next).unsqueeze(1) # [B, 1, d_model]
            q_rec = q0.clone()
            q_rec[:, pos : pos + 1, :] = q_rec[:, pos : pos + 1, :] + q_mod
            q_vec = q_rec[:, pos, :]

            # D. Dynamic Candidate Features (entropy, top-1 confidence, margin)
            cand_feat = None
            ent_val, conf_val = 0.0, 0.0
            if candidate_positions is not None:
                B_c, N_c = candidate_positions.shape
                c_states = torch.stack([h_curr[b, candidate_positions[b]] for b in range(B_c)], dim=0)
                _, p_cand = self.compute_dynamic_binding(r_curr, c_states)
                sorted_p, _ = torch.sort(p_cand, dim=-1, descending=True)
                top1 = sorted_p[:, 0:1]
                top2 = sorted_p[:, 1:2] if N_c > 1 else torch.zeros_like(top1)
                margin = top1 - top2
                entropy = -(p_cand * torch.log(p_cand + 1e-8)).sum(dim=-1, keepdim=True)
                cand_feat = torch.cat([top1, margin, entropy], dim=-1)
                ent_val = entropy.mean().item()
                conf_val = top1.mean().item()

            # E. Answer-Sufficiency Evaluation
            s_logit, s_score, c_prob = self.sufficiency_controller(s_next, q_vec, r_curr, cand_feat)

            if random_controller:
                c_prob = torch.rand_like(c_prob)
                s_score = 1.0 - c_prob
            elif bypass_controller:
                c_prob = torch.ones_like(c_prob)
                s_score = torch.zeros_like(s_score)

            p_continue_list.append(c_prob)
            suff_score_list.append(s_score)
            differentiable_cost = differentiable_cost + c_prob

            # Halt decision: halt if representation sufficiency >= threshold
            should_halt = (s_score >= sufficiency_threshold)
            halt_decision = should_halt & active_mask

            # F. Recurrent Attention Cycle
            h_norm = self.norm_cycle(h_curr)
            k_rec = self.k_rec_proj(h_norm)
            v_rec = self.v_rec_proj(h_norm)
            a_rec_ctx, w_rec = self._causal_attention(q_rec, k_rec, v_rec)
            a_rec = self.out_rec_proj(a_rec_ctx)

            h_next = h_curr + self.cycle_gamma * a_rec

            cycle_traces.append(
                SufficiencyCycleTrace(
                    cycle_index=cycle_idx,
                    w_attn=w_rec if return_trace else None,
                    state_s=s_next,
                    query_q=q_mod,
                    context_r=r_curr,
                    sufficiency_score=s_score,
                    continue_prob=c_prob,
                    continue_logit=s_logit,
                    halt_decision=halt_decision,
                    entropy_stat=ent_val,
                    confidence_stat=conf_val,
                )
            )

            s_curr = s_next
            h_curr = h_next

            if fixed_cycles is None and not bypass_controller and not random_controller:
                active_mask = active_mask & (~should_halt)
                if not active_mask.any():
                    break

        h_final = self.norm_cycle(h_curr)
        vocab_logits = self.lm_head(h_final)

        out = {
            "vocab_logits": vocab_logits,
            "final_hidden": h_final,
            "final_state": s_curr,
            "total_cycles_executed": executed_cycles,
            "cycle_traces": cycle_traces,
            "p_continue_per_cycle": p_continue_list,
            "sufficiency_scores_per_cycle": suff_score_list,
            "computation_cost": differentiable_cost.mean(),
            "binding_logits": None,
            "binding_probs": None,
            "trace": cycle_traces[-1] if cycle_traces else None,
        }

        if candidate_positions is not None:
            B_cand, N_cand = candidate_positions.shape
            cand_states = torch.stack([h_final[b, candidate_positions[b]] for b in range(B_cand)], dim=0)
            val_rep = h_final[:, pos, :]
            b_logits, b_probs = self.compute_dynamic_binding(val_rep, cand_states)
            out["binding_logits"] = b_logits
            out["binding_probs"] = b_probs

        return out


def inspect_sufficiency_core() -> Dict[str, Any]:
    """Prints architecture details and parameter counts."""
    core = SufficiencyAdaptiveReasoningCore()
    total_p = core.total_param_count
    trainable_p = core.trainable_param_count
    ctrl_p = sum(p.numel() for p in core.sufficiency_controller.parameters())

    return {
        "total_parameters": total_p,
        "trainable_parameters": trainable_p,
        "controller_parameters": ctrl_p,
        "d_model": core.d_model,
        "d_state": core.d_state,
        "max_cycles": core.max_cycles,
        "budget_passed": trainable_p < 550_000,
    }


if __name__ == "__main__":
    spec = inspect_sufficiency_core()
    print("=" * 60)
    print("STEP 385: SUFFICIENCY-BASED ADAPTIVE REASONING CORE SPEC")
    print("=" * 60)
    for k, v in spec.items():
        print(f"  {k:25s}: {v}")

    # Forward test
    x = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    core = SufficiencyAdaptiveReasoningCore()
    c_pos = torch.tensor([[1, 2]], dtype=torch.long)
    out = core(x, candidate_positions=c_pos, return_trace=True)
    print(f"\nForward executed: cycles={out['total_cycles_executed']}")
    print(f"Sufficiency scores: {[round(s.item(), 4) for s in out['sufficiency_scores_per_cycle']]}")
    print(f"Continue probs:     {[round(p.item(), 4) for p in out['p_continue_per_cycle']]}")
