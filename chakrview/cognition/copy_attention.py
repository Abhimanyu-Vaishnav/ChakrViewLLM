"""Step 187: Neural Copy / Pointer Mechanism.

Implements a minimal, fully differentiable neural copy/pointer mechanism
operating on top of ChakrMicro's final hidden states:

P(w) = g * P_gen(w) + (1 - g) * sum_{i: x_i = w} P_copy(i)

Where:
- q_i = W_q * h_i
- k_j = W_k * h_j
- S_ij = (q_i . k_j) / sqrt(d) (with causal masking)
- P_copy(j) = softmax(S_i)[:]_j
- g = sigmoid(W_g * h_i) is P(gen)
- (1 - g) is P(copy)

Exposes:
- copy_prob: (1 - g)
- gen_prob: g
- copy_attention: distribution over source positions [0 .. T-1]
- selected_source_position: argmax position of copy attention
- selected_source_token: input_ids[selected_source_position]
- combined_logits: log(P(w)) for downstream loss and inference

STRICT PROJECT INVARIANTS:
- 100% neural (learned query/key projection, softmax, learned sigmoid gate)
- NO Python dictionary lookup
- NO string matching
- NO external answer lookup
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro


@dataclasses.dataclass
class CopyMechanismOutput:
    combined_logits: torch.Tensor           # [batch, seq_len, vocab_size]
    gen_logits: torch.Tensor                # [batch, seq_len, vocab_size]
    copy_attention: torch.Tensor            # [batch, seq_len, seq_len]
    copy_prob: torch.Tensor                 # [batch, seq_len, 1]
    gen_prob: torch.Tensor                  # [batch, seq_len, 1]
    selected_source_pos: torch.Tensor       # [batch, seq_len]
    selected_source_token: torch.Tensor     # [batch, seq_len]


class NeuralCopyPointerHead(nn.Module):
    """Minimal differentiable pointer-generator head."""

    def __init__(self, d_model: int, vocab_size: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Query and Key projections for copy attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # Learned copy-generation gate: g = sigmoid(W_g * h + b_g)
        # g is P(gen), (1-g) is P(copy)
        self.gate_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, T, d_model]
        gen_logits: torch.Tensor,         # [B, T, vocab_size]
        input_ids: torch.Tensor,          # [B, T]
        causal_mask: bool = True,
        force_mode: Optional[str] = None, # None, "copy_only", "gen_only"
    ) -> CopyMechanismOutput:
        B, T, D = hidden_states.shape

        # 1. Compute copy attention scores: S_ij = (q_i . k_j) / sqrt(d)
        q = self.q_proj(hidden_states)    # [B, T, D]
        k = self.k_proj(hidden_states)    # [B, T, D]

        scores = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)  # [B, T, T]

        if causal_mask:
            mask = torch.triu(torch.ones(T, T, device=hidden_states.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        copy_attn = F.softmax(scores, dim=-1)  # [B, T, T]

        # 2. Compute copy probability vs generation probability
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, T, 1] -> p_gen

        if force_mode == "copy_only":
            p_gen = torch.zeros_like(gate)
            p_copy = torch.ones_like(gate)
        elif force_mode == "gen_only":
            p_gen = torch.ones_like(gate)
            p_copy = torch.zeros_like(gate)
        else:
            p_gen = gate
            p_copy = 1.0 - gate

        # 3. Generation distribution: P_gen = softmax(gen_logits)
        gen_probs = F.softmax(gen_logits, dim=-1)  # [B, T, vocab_size]

        # 4. Scatter copy attention probabilities into vocabulary space
        # copy_dist[b, t, token_id] = sum_{j: input_ids[b, j] == token_id} copy_attn[b, t, j]
        copy_dist = torch.zeros(B, T, self.vocab_size, device=hidden_states.device, dtype=hidden_states.dtype)
        expanded_ids = input_ids.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
        copy_dist.scatter_add_(dim=2, index=expanded_ids, src=copy_attn)

        # 5. Combined probability distribution:
        # P_final = p_gen * P_gen + p_copy * P_copy
        p_final = p_gen * gen_probs + p_copy * copy_dist
        p_final = torch.clamp(p_final, min=1e-12)
        combined_logits = torch.log(p_final)

        sel_pos = torch.argmax(copy_attn, dim=-1)  # [B, T]
        sel_token = torch.gather(input_ids, dim=1, index=sel_pos)  # [B, T]

        return CopyMechanismOutput(
            combined_logits=combined_logits,
            gen_logits=gen_logits,
            copy_attention=copy_attn,
            copy_prob=p_copy,
            gen_prob=p_gen,
            selected_source_pos=sel_pos,
            selected_source_token=sel_token,
        )


class ChakrMicroWithCopy(nn.Module):
    """Wrapper that equips ChakrMicro with the NeuralCopyPointerHead."""

    def __init__(self, base_model: ChakrMicro, force_mode: Optional[str] = None):
        super().__init__()
        self.base_model = base_model
        self.force_mode = force_mode
        self.copy_head = NeuralCopyPointerHead(
            d_model=base_model.config.d_model,
            vocab_size=base_model.config.vocab_size,
        )

    @property
    def config(self):
        return self.base_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        force_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, CopyMechanismOutput]:
        mode = force_mode or self.force_mode
        B, T = input_ids.shape

        # Run base transformer backbone forward pass
        x = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        hidden_states = self.base_model.final_norm(x)
        gen_logits = self.base_model.lm_head(hidden_states)

        copy_out = self.copy_head(
            hidden_states=hidden_states,
            gen_logits=gen_logits,
            input_ids=input_ids,
            force_mode=mode,
        )
        return copy_out.combined_logits, copy_out
