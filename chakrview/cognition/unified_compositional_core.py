"""Step 369: Unified Compositional Neural Core Architecture.

Implements UnifiedCompositionalCore:
An end-to-end purpose-built sequence core specifically designed for multi-hop
relational reasoning and variable binding without relying on frozen ChakrMicro blocks.

Conceptual Flow:
tokens
  -> Token Embedding + Learned / RoPE representations
  -> Relational Attention Cycle 1 (Q1, K1, V1)
  -> Value representation r1 extracted at query position
  -> Compositional State Update (GRU / Gated Transition, d_state <= 64)
  -> State-conditioned Query Formation q2 = Q2(s1) + q_base
  -> Relational Attention Cycle 2 (q2, K2, V2)
  -> Value representation r2 extracted
  -> Dynamic Contextual Token Binding (neural dot-product compatibility against premise tokens)
  -> Output logits and prediction

Properties:
- Parameter budget: Preferred < 500k trainable parameters, hard max < 1M.
- Intermediate state s1 and second-hop query q2 are explicit first-class computational objects.
- Differentiable end-to-end.
- CPU-first, fast convergence, zero symbolic lookup, zero answer-label leakage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask
from chakrview.brain.output import LMHead


def compute_module_sha256(module: nn.Module) -> str:
    """Computes deterministic SHA-256 digest of named module parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@dataclasses.dataclass
class UnifiedCoreTrace:
    q1: torch.Tensor
    k1: torch.Tensor
    v1: torch.Tensor
    a1: torch.Tensor
    r1: torch.Tensor
    s1: torch.Tensor
    q2: torch.Tensor
    k2: torch.Tensor
    v2: torch.Tensor
    a2: torch.Tensor
    r2: torch.Tensor
    w1: Optional[torch.Tensor]
    w2: Optional[torch.Tensor]


class UnifiedCompositionalCore(nn.Module):
    """
    Unified Recurrent-Attention Neural Core for multi-hop compositional reasoning.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 96,
        n_heads: int = 4,
        d_state: int = 48,
        d_bind: int = 32,
        max_seq_len: int = 512,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 32
        self.d_state = d_state
        self.d_bind = d_bind
        self.max_seq_len = max_seq_len

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Pre-Norms
        self.norm_input = RMSNorm(d_model, eps=rms_norm_eps)
        self.norm_c1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.norm_c2 = RMSNorm(d_model, eps=rms_norm_eps)

        # RoPE and Causal Masking
        self.rotary = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self.causal_mask = CausalMask(max_seq_len=max_seq_len)

        # 3. Cycle 1 Projections
        self.q1_proj = nn.Linear(d_model, d_model, bias=False)
        self.k1_proj = nn.Linear(d_model, d_model, bias=False)
        self.v1_proj = nn.Linear(d_model, d_model, bias=False)
        self.out1_proj = nn.Linear(d_model, d_model, bias=False)

        # 4. Value Extraction & Gated State Transition
        self.norm_r1 = nn.LayerNorm(d_model)
        self.val_to_state = nn.Linear(d_model, d_state, bias=False)

        self.w_z_h = nn.Linear(d_state, d_state)
        self.w_z_s = nn.Linear(d_state, d_state, bias=False)

        self.w_r_h = nn.Linear(d_state, d_state)
        self.w_r_s = nn.Linear(d_state, d_state, bias=False)

        self.w_n_h = nn.Linear(d_state, d_state)
        self.w_n_s = nn.Linear(d_state, d_state, bias=False)

        self.s0 = nn.Parameter(torch.zeros(1, d_state))

        # 5. Query Formation q2 from state s1
        self.state_to_q2 = nn.Linear(d_state, d_model, bias=False)

        # 6. Cycle 2 Projections
        self.k2_proj = nn.Linear(d_model, d_model, bias=False)
        self.v2_proj = nn.Linear(d_model, d_model, bias=False)
        self.out2_proj = nn.Linear(d_model, d_model, bias=False)
        self.gamma_c2 = nn.Parameter(torch.tensor([1.0]))  # Learnable scaling for Cycle 2

        # 7. Dynamic Contextual Token Binding Module
        self.norm_bind_val = nn.LayerNorm(d_model)
        self.bind_query_proj = nn.Linear(d_model, d_bind, bias=False)

        self.norm_bind_cand = nn.LayerNorm(d_model)
        self.bind_cand_proj = nn.Linear(d_model, d_bind, bias=False)
        self.bind_scale = nn.Parameter(torch.tensor([4.0]))

        # 8. Standard LM Head (Strict weight tying: shares embedding weight directly)
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
        """
        Computes dynamic contextual token binding compatibility.
        val_rep: [B, d_model]
        cand_states: [B, N, d_model]
        Returns logits [B, N], probs [B, N]
        """
        B, N, _ = cand_states.shape
        q = self.bind_query_proj(self.norm_bind_val(val_rep))  # [B, d_bind]
        q = F.normalize(q, p=2, dim=-1)

        k = self.bind_cand_proj(self.norm_bind_cand(cand_states))  # [B, N, d_bind]
        k = F.normalize(k, p=2, dim=-1)

        # Dot-product compatibility
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
        override_s1: Optional[torch.Tensor] = None,
        override_q2: Optional[torch.Tensor] = None,
        disable_c1: bool = False,
        disable_c2: bool = False,
        bypass_state: bool = False,
        reset_state: bool = False,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass through UnifiedCompositionalCore.
        """
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)

        # 1. Embedding & Pre-norm
        x = self.embedding(input_ids)
        h0 = self.norm_input(x)

        # 2. Relational Attention Cycle 1
        q1 = self.q1_proj(h0)
        k1 = self.k1_proj(h0)
        v1 = self.v1_proj(h0)

        a1_ctx, w1 = self._causal_attention(q1, k1, v1)
        a1 = self.out1_proj(a1_ctx)
        if disable_c1:
            a1 = torch.zeros_like(a1)

        # Intermediate residual 1
        h1 = h0 + a1

        # 3. Value Representation Extraction & State Transition
        h_q = a1_ctx[:, pos : pos + 1, :]  # [B, 1, d_model]
        h_q_norm = self.norm_r1(h_q).squeeze(1)  # [B, d_model]
        r1 = self.val_to_state(h_q_norm)  # [B, d_state]

        # GRU State Transition
        s_prev = self.s0.expand(B, -1)
        if reset_state:
            s1 = s_prev
        else:
            z_t = torch.sigmoid(self.w_z_h(r1) + self.w_z_s(s_prev))
            r_t = torch.sigmoid(self.w_r_h(r1) + self.w_r_s(s_prev))
            n_t = torch.tanh(self.w_n_h(r1) + self.w_n_s(r_t * s_prev))
            s1 = (1.0 - z_t) * s_prev + z_t * n_t

        if override_s1 is not None:
            s1 = override_s1

        # 4. State-Conditioned Query Formation q2
        q2_base = q1
        if bypass_state:
            q2 = q2_base
        else:
            q2_mod = self.state_to_q2(s1).unsqueeze(1)  # [B, 1, d_model]
            q2 = q2_base.clone()
            q2[:, pos : pos + 1, :] = q2[:, pos : pos + 1, :] + q2_mod

        if override_q2 is not None:
            q2 = override_q2

        # 5. Relational Attention Cycle 2
        h1_norm = self.norm_c1(h1)
        k2 = self.k2_proj(h1_norm)
        v2 = self.v2_proj(h1_norm)

        a2_ctx, w2 = self._causal_attention(q2, k2, v2)
        a2 = self.out2_proj(a2_ctx)
        if disable_c2:
            a2 = torch.zeros_like(a2)

        # Final sequence representation
        h2 = h1 + self.gamma_c2 * a2
        h_final = self.norm_c2(h2)

        # Second-hop retrieved representation at query pos
        r2 = h_final[:, pos, :]  # [B, d_model]

        # 6. Vocab Logits
        vocab_logits = self.lm_head(h_final)

        out = {
            "vocab_logits": vocab_logits,
            "final_hidden": h_final,
            "r1": r1,
            "s1": s1,
            "q2": q2,
            "r2": r2,
            "trace": None,
        }

        # 7. Dynamic Contextual Token Binding
        if candidate_positions is not None:
            B, N = candidate_positions.shape
            cand_states = torch.stack(
                [h_final[b, candidate_positions[b]] for b in range(B)],
                dim=0,
            )  # [B, N, d_model]
            cand_mask = torch.ones((B, N), dtype=torch.bool, device=h_final.device)
            b_logits, b_probs = self.compute_dynamic_binding(
                val_rep=r2,
                cand_states=cand_states,
                cand_mask=cand_mask,
            )
            out["binding_logits"] = b_logits
            out["candidate_probs"] = b_probs

        if return_trace:
            out["trace"] = UnifiedCoreTrace(
                q1=q1,
                k1=k1,
                v1=v1,
                a1=a1,
                r1=r1,
                s1=s1,
                q2=q2,
                k2=k2,
                v2=v2,
                a2=a2,
                r2=r2,
                w1=w1,
                w2=w2,
            )

        return out


def audit_unified_core_architecture() -> Dict[str, Any]:
    """Audits parameters and architecture bounds of UnifiedCompositionalCore."""
    core = UnifiedCompositionalCore()
    total_p = core.total_param_count
    trainable_p = core.trainable_param_count
    
    # Check budget constraints: Preferred < 500k, hard max < 1M.
    # Note: Embedding is tied with LMHead. With vocab=4096 and d_model=192,
    # embedding = 786,432 parameters. Core projections = ~150k parameters.
    # Total = ~945k parameters (within the <1M hard limit).
    # If embedding is excluded/compacted (e.g. d_model=128), it's easily < 500k.
    return {
        "total_params": total_p,
        "trainable_params": trainable_p,
        "within_1m_budget": trainable_p < 1_000_000,
        "within_500k_preferred": trainable_p < 500_000,
        "summary": (
            f"UnifiedCompositionalCore contains {trainable_p:,} trainable parameters "
            f"(hard budget < 1,000,000 passed: {trainable_p < 1_000_000}). "
            f"Integrates 2 explicit relational attention cycles, GRU state transition (d_state=48), "
            f"state-to-query modulation, and dynamic contextual binding."
        ),
    }


if __name__ == "__main__":
    rep = audit_unified_core_architecture()
    print("=== STEP 369 UNIFIED COMPOSITIONAL CORE ===")
    print(f"Total Parameters:     {rep['total_params']:,}")
    print(f"Trainable Parameters: {rep['trainable_params']:,}")
    print(f"Hard Limit (< 1M):    {rep['within_1m_budget']}")
    print(rep["summary"])
