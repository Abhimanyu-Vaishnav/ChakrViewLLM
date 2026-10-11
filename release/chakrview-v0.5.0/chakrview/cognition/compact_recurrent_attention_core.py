"""Step 338: Compact Recurrent Attention Core Model.

Implements the unified, differentiable neural core that combines:
1. Low-Rank Q Adaptation: q_proj + (alpha_q / r) * B_q(A_q(x))
2. Low-Rank K Adaptation: k_proj + (alpha_k / r) * B_k(A_k(x))
3. Low-Rank V Adaptation: v_proj + (alpha_v / r) * B_v(A_v(x))
4. Compact FFN Bottleneck: x + tanh(alpha_ffn) * Up(GELU(Down(Norm(x))))
5. Gated Recurrent State Transition: s_(t+1) = GRU_Transition(h_val, s_t)
6. Dynamic Contextual Token Binding Head

Parameter Budget:
- Low-Rank Q (rank=16): 2 * 192 * 16 = 6,144 params
- Low-Rank K (rank=16): 2 * 192 * 16 = 6,144 params
- Low-Rank V (rank=16): 2 * 192 * 16 = 6,144 params
- Compact FFN (bottleneck=32): 2 * 192 * 32 + 192 = 12,480 params
- Recurrent State Core (d_state=64): ~37,000 params
- Dynamic Binding Head: ~25,000 params
Total Trainable Parameters: ~93,000 (< 100,000 preferred budget, << 150,000 max budget).

Initialization:
- All adaptation up-projections zero-initialized (B_q = 0, B_k = 0, B_v = 0, alpha_ffn = 0)
- Canonical baseline remains strictly bit-exact (Delta W = 0)
- Initial candidate forward pass is exact numerical identity with baseline.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.trainable_attention_subset import (
    LowRankProjectionDelta,
)
from chakrview.cognition.internal_trainable_pathway import (
    InternalBottleneckAdapter,
)
from chakrview.cognition.recurrent_state_transition import (
    GatedRecurrentStateTransition,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
)


def compute_module_sha256(module: nn.Module) -> str:
    """Computes deterministic SHA-256 digest of named module parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class CompactRecurrentAttentionCore(nn.Module):
    """
    Unified Recurrent-Attention Neural Core integrated into ChakrMicro at target layer.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        target_layer: int = 3,
        rank: int = 12,
        bottleneck_dim: int = 24,
        d_state: int = 48,
        d_bind: int = 32,
        enable_v: bool = True,
        enable_ffn: bool = True,
        enable_recurrent: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.d_model = base_model.config.d_model
        self.target_layer = target_layer
        self.rank = rank
        self.bottleneck_dim = bottleneck_dim
        self.d_state = d_state
        self.d_bind = d_bind

        self.enable_v = enable_v
        self.enable_ffn = enable_ffn
        self.enable_recurrent = enable_recurrent

        # Strictly freeze canonical base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 1. Trainable Attention Projections
        self.q_delta = LowRankProjectionDelta(self.d_model, self.d_model, rank=rank)
        self.k_delta = LowRankProjectionDelta(self.d_model, self.d_model, rank=rank)
        self.v_delta = LowRankProjectionDelta(self.d_model, self.d_model, rank=rank) if enable_v else None

        # 2. Trainable FFN Pathway
        self.ffn_adapter = InternalBottleneckAdapter(d_model=self.d_model, bottleneck_dim=bottleneck_dim) if enable_ffn else None

        # 3. Gated Recurrent State Transition
        if enable_recurrent:
            self.recurrent_transition = GatedRecurrentStateTransition(d_model=self.d_model, d_state=d_state)
            self.query_modulation = nn.Linear(d_state, self.d_model, bias=False)
            nn.init.zeros_(self.query_modulation.weight)  # Zero initialization
        else:
            self.recurrent_transition = None
            self.query_modulation = None

        # 4. Dynamic Contextual Token Binding
        self.binding = DynamicContextualTokenBinding(d_model=self.d_model, d_bind=d_bind)

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        disable_core: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Executes transformer forward pass with integrated attention adaptation & FFN at target layer.
        """
        B, T = input_ids.shape
        x = self.base_model.embedding(input_ids)
        states = [x]

        for i, layer in enumerate(self.base_model.layers):
            norm_x = layer.norm_1(x)
            attn = layer.attn

            if i == self.target_layer and not disable_core:
                # 1. Adapted Q/K/V Projections
                q = attn.q_proj(norm_x) + self.q_delta(norm_x)
                k = attn.k_proj(norm_x) + self.k_delta(norm_x)
                v = attn.v_proj(norm_x)
                if self.v_delta is not None:
                    v = v + self.v_delta(norm_x)

                # Reshape heads: [B, H, T, d_head]
                q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
                k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
                v = v.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

                q_rot = attn.rotary(q, T)
                k_rot = attn.rotary(k, T)

                scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * attn.scale
                mask = attn.causal_mask(T)
                scores = scores + mask
                probs = torch.softmax(scores, dim=-1)
                context = torch.matmul(probs, v)
                context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
                attn_out = attn.out_proj(context)

                # Pre-norm residual step
                x = x + attn_out

                # 2. Adapted FFN execution
                ffn_norm = layer.norm_2(x)
                ffn_out = layer.ffn(ffn_norm)
                if self.ffn_adapter is not None:
                    ffn_out = self.ffn_adapter(ffn_out)
                x = x + ffn_out

            else:
                x = layer(x, layer_idx=i)

            states.append(x)

        final_h = self.base_model.final_norm(x)
        return final_h, states

    def forward(
        self,
        input_ids: torch.Tensor,
        disable_core: bool = False,
    ) -> torch.Tensor:
        final_h, _ = self.forward_hidden_states(input_ids, disable_core=disable_core)
        return self.base_model.lm_head(final_h)

    def compute_recurrent_second_hop_query(
        self,
        v1_rep: torch.Tensor,
        s_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies recurrent state transition and query modulation:
        s_(t+1) = Transition(v1_rep, s_t)
        q2 = v1_rep + QueryModulation(s_(t+1))
        """
        if self.recurrent_transition is None or self.query_modulation is None:
            return v1_rep, s_t

        s_next, h_res = self.recurrent_transition.transition(v1_rep, s_t)
        q2 = h_res + self.query_modulation(s_next)
        return q2, s_next

    def compute_binding_scores(
        self,
        retrieved_val_rep: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Binds retrieved contextual value to candidate tokens."""
        return self.binding(retrieved_val_rep, candidate_states, candidate_mask)
