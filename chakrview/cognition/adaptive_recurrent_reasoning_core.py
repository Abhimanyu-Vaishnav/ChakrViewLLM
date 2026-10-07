"""Step 377: Adaptive Recurrent Reasoning Core Architecture.

Standalone sequence core with dynamic computation depth controlled by a neural halting controller.
Does NOT wrap ChakrMicro blocks.
Inherits the proven parameter efficiency of UnifiedCompositionalCore (<500k-600k params),
extending it with:
1. Dynamic recurrent reasoning cycles (1 to MAX_CYCLES, default safety limit = 8)
2. Neural halting/continuation controller (MLP evaluating state_t + query_t + context_t)
3. State-conditioned query refinement across iterative hops
4. Dynamic contextual token binding
5. Full causal and gradient observability for each cycle
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
class AdaptiveCycleTrace:
    cycle_index: int
    w_attn: Optional[torch.Tensor]       # [B, n_heads, T, T]
    state_s: torch.Tensor                 # [B, d_state]
    query_q: torch.Tensor                 # [B, 1, d_model]
    context_r: torch.Tensor               # [B, d_model]
    continue_prob: torch.Tensor           # [B, 1]
    continue_logit: torch.Tensor          # [B, 1]
    halt_decision: torch.Tensor           # [B, 1] (bool)


@dataclasses.dataclass
class AdaptiveCoreOutput:
    vocab_logits: torch.Tensor            # [B, T, vocab_size]
    binding_logits: Optional[torch.Tensor]# [B, num_candidates]
    binding_probs: Optional[torch.Tensor] # [B, num_candidates]
    total_cycles_executed: int
    cycle_traces: List[AdaptiveCycleTrace]
    final_state: torch.Tensor             # [B, d_state]
    cumulative_halt_probs: torch.Tensor   # [B, num_cycles]
    p_continue_per_cycle: List[torch.Tensor]
    computation_cost: torch.Tensor        # scalar or [B] differentiable cycle usage


class NeuralHaltingController(nn.Module):
    """
    Evaluates whether additional reasoning cycles are required.
    Inputs:
    - state_t: recurrent state vector [B, d_state]
    - query_t: active query representation at the reasoning position [B, d_model]
    - context_t: retrieved relational context representation [B, d_model]

    Outputs:
    - continue_logit: logit for whether reasoning should continue [B, 1]
    - continue_prob: sigmoid(continue_logit) [B, 1]
    """
    def __init__(self, d_state: int = 48, d_model: int = 96, hidden_dim: int = 64):
        super().__init__()
        in_dim = d_state + d_model + d_model  # 48 + 96 + 96 = 240
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize with positive bias so network explores multiple cycles initially
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.2)

    def forward(
        self,
        state_t: torch.Tensor,
        query_t: torch.Tensor,
        context_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([state_t, query_t, context_t], dim=-1)
        h = self.act(self.fc1(self.norm(features)))
        logit = self.fc2(h)
        prob = torch.sigmoid(logit)
        return logit, prob


class AdaptiveRecurrentReasoningCore(nn.Module):
    """
    Standalone sequence core with adaptive recurrent reasoning cycles.
    Parameters budget target: < 600k trainable parameters.
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

        # 3. Base Input Attention Projections (Cycle 0 / Initial sequence processing)
        self.q0_proj = nn.Linear(d_model, d_model, bias=False)
        self.k0_proj = nn.Linear(d_model, d_model, bias=False)
        self.v0_proj = nn.Linear(d_model, d_model, bias=False)
        self.out0_proj = nn.Linear(d_model, d_model, bias=False)

        # 4. Shared Recurrent Reasoning Attention Projections (Cycles 1..T)
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

        # 6. State-Conditioned Query Formulation for next reasoning cycle
        self.state_to_q = nn.Linear(d_state, d_model, bias=False)
        self.cycle_gamma = nn.Parameter(torch.tensor([1.0]))

        # 7. Neural Halting / Continuation Controller
        self.controller = NeuralHaltingController(d_state=d_state, d_model=d_model, hidden_dim=64)

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
        halt_threshold: float = 0.5,
        override_state: Optional[torch.Tensor] = None,
        override_cycle: Optional[int] = None,
        bypass_controller: bool = False,
        random_controller: bool = False,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass with adaptive or fixed recurrent reasoning cycles.

        Args:
            input_ids: [B, T]
            candidate_positions: [B, N] token positions for dynamic binding
            query_pos: position in sequence to compute queries from (default: T - 1)
            max_reasoning_cycles: safety boundary on maximum cycles
            fixed_cycles: if set, forces exactly N cycles (disabling adaptive termination)
            halt_threshold: continue_prob threshold for halting (default: 0.5)
            override_state: intervention tensor to inject into state
            override_cycle: cycle index at which to apply override_state
            bypass_controller: if True, halting probability is ignored (runs to max_cycles)
            random_controller: if True, random halting probability used
            return_trace: return per-cycle attention weights and intermediate states
        """
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)
        max_cycles = max_reasoning_cycles if max_reasoning_cycles is not None else self.max_cycles

        # 1. Input Embedding & Pre-Norm
        x = self.embedding(input_ids)
        h0 = self.norm_input(x)

        # 2. Cycle 0: Initial relational context & sequence representation
        q0 = self.q0_proj(h0)
        k0 = self.k0_proj(h0)
        v0 = self.v0_proj(h0)
        a0_ctx, w0 = self._causal_attention(q0, k0, v0)
        a0 = self.out0_proj(a0_ctx)

        h_curr = h0 + a0  # Current sequence representation [B, T, d_model]

        # Initial recurrent state
        s_curr = self.s0.expand(B, -1)  # [B, d_state]

        # Tracking variables
        cycle_traces: List[AdaptiveCycleTrace] = []
        p_continue_list: List[torch.Tensor] = []
        cumulative_halt_probs = []

        active_mask = torch.ones(B, 1, dtype=torch.bool, device=input_ids.device)
        differentiable_cost = torch.zeros(B, 1, device=input_ids.device)

        cycles_to_run = fixed_cycles if fixed_cycles is not None else max_cycles

        # Track execution per cycle
        executed_cycles = 0

        for cycle_idx in range(1, cycles_to_run + 1):
            executed_cycles += 1

            # A. Value extraction at query position
            h_q = h_curr[:, pos : pos + 1, :]            # [B, 1, d_model]
            r_curr = self.norm_r(h_q).squeeze(1)         # [B, d_model]
            r_feat = self.val_to_state(r_curr)           # [B, d_state]

            # B. GRU State Transition
            z_t = torch.sigmoid(self.w_z_h(r_feat) + self.w_z_s(s_curr))
            r_t = torch.sigmoid(self.w_r_h(r_feat) + self.w_r_s(s_curr))
            n_t = torch.tanh(self.w_n_h(r_feat) + self.w_n_s(r_t * s_curr))
            s_next = (1.0 - z_t) * s_curr + z_t * n_t

            # Apply state intervention if requested
            if override_state is not None and (override_cycle is None or override_cycle == cycle_idx):
                s_next = override_state

            # D. State-Conditioned Query Formulation for this reasoning hop
            q_mod = self.state_to_q(s_next).unsqueeze(1) # [B, 1, d_model]
            q_rec = q0.clone()
            q_rec[:, pos : pos + 1, :] = q_rec[:, pos : pos + 1, :] + q_mod

            # C. Neural Halting / Continuation Controller
            q_vec = q_rec[:, pos, :]                     # [B, d_model] dynamic query representation
            c_logit, c_prob = self.controller(s_next, q_vec, r_curr)

            if random_controller:
                c_prob = torch.rand_like(c_prob)
            elif bypass_controller:
                c_prob = torch.ones_like(c_prob)

            p_continue_list.append(c_prob)
            differentiable_cost = differentiable_cost + c_prob

            # Halt decision for this cycle
            should_halt = (c_prob < halt_threshold)
            halt_decision = should_halt & active_mask

            # E. Recurrent Attention Cycle
            h_norm = self.norm_cycle(h_curr)
            k_rec = self.k_rec_proj(h_norm)
            v_rec = self.v_rec_proj(h_norm)
            a_rec_ctx, w_rec = self._causal_attention(q_rec, k_rec, v_rec)
            a_rec = self.out_rec_proj(a_rec_ctx)

            # Update sequence representation
            h_next = h_curr + self.cycle_gamma * a_rec

            # Record cycle trace
            if return_trace or True:
                cycle_traces.append(
                    AdaptiveCycleTrace(
                        cycle_index=cycle_idx,
                        w_attn=w_rec if return_trace else None,
                        state_s=s_next,
                        query_q=q_mod,
                        context_r=r_curr,
                        continue_prob=c_prob,
                        continue_logit=c_logit,
                        halt_decision=halt_decision,
                    )
                )

            # Update states for continuing items
            s_curr = s_next
            h_curr = h_next

            # If dynamic execution and all batch items decide to halt, break
            if fixed_cycles is None and not bypass_controller and not random_controller:
                active_mask = active_mask & (~should_halt)
                if not active_mask.any():
                    break

        # Final representation
        h_final = self.norm_cycle(h_curr)
        vocab_logits = self.lm_head(h_final)

        out = {
            "vocab_logits": vocab_logits,
            "final_hidden": h_final,
            "final_state": s_curr,
            "total_cycles_executed": executed_cycles,
            "cycle_traces": cycle_traces,
            "p_continue_per_cycle": p_continue_list,
            "computation_cost": differentiable_cost.mean(),
            "binding_logits": None,
            "binding_probs": None,
            "trace": cycle_traces[-1] if cycle_traces else None,
        }

        # Dynamic Contextual Token Binding
        if candidate_positions is not None:
            B_cand, N_cand = candidate_positions.shape
            cand_states = torch.stack(
                [h_final[b, candidate_positions[b]] for b in range(B_cand)],
                dim=0,
            )
            val_rep = h_final[:, pos, :]  # Final query position state
            b_logits, b_probs = self.compute_dynamic_binding(val_rep, cand_states)
            out["binding_logits"] = b_logits
            out["binding_probs"] = b_probs

        return out


def inspect_adaptive_core() -> Dict[str, Any]:
    """Prints architecture specification, parameter counts, and tensor dimensions."""
    core = AdaptiveRecurrentReasoningCore()
    total_p = core.total_param_count
    trainable_p = core.trainable_param_count
    controller_p = sum(p.numel() for p in core.controller.parameters())

    info = {
        "total_parameters": total_p,
        "trainable_parameters": trainable_p,
        "controller_parameters": controller_p,
        "d_model": core.d_model,
        "d_state": core.d_state,
        "n_heads": core.n_heads,
        "head_dim": core.head_dim,
        "d_bind": core.d_bind,
        "max_cycles": core.max_cycles,
        "budget_passed": trainable_p < 600_000,
    }
    return info


if __name__ == "__main__":
    spec = inspect_adaptive_core()
    print("=" * 60)
    print("STEP 377: ADAPTIVE RECURRENT REASONING CORE SPECIFICATION")
    print("=" * 60)
    for k, v in spec.items():
        print(f"  {k:25s}: {v}")
    
    # Sanity forward check
    x = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    core = AdaptiveRecurrentReasoningCore()
    out = core(x, return_trace=True)
    print(f"\nForward sanity executed: cycles={out['total_cycles_executed']}, logits={out['vocab_logits'].shape}")
    print(f"Controller continue probabilities: {[p.item() for p in out['p_continue_per_cycle']]}")
