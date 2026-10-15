"""
Batches 826-831: Neural Pointer-Generator Architecture for ChakrMicro.
Location: chakrview/brain/pointer_generator.py

Features:
1. ChakrPointerMicro:
   Combines base ChakrMicro causal transformer with:
   - A Pointer Attention Head over prompt positions.
   - A Learned Copy Gate p_gen in [0, 1].
2. Dual-distribution Output:
   P(w) = p_gen * P_vocab(w) + (1 - p_gen) * sum_{i: x_i = w} P_ptr(i)
3. Prompt/Output Boundary Awareness:
   Only tokens in the prompt context [0, prompt_len) can be pointed to.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


class PointerGeneratorHead(nn.Module):
    """
    Computes pointer attention over prompt tokens and the dynamic copy gate p_gen.
    """

    def __init__(self, d_model: int = 192):
        super().__init__()
        self.d_model = d_model
        # Query projection for current decoder state
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        # Key projection for prompt representations
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        # Copy gate projection: [h_t, context_vector] -> p_gen in (0, 1)
        self.W_gate = nn.Linear(d_model * 2, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, T, d_model]
        prompt_len: int,                  # Length of prompt prefix
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: All sequence hidden states from final RMSNorm [B, T, d_model]
            prompt_len: Number of prompt tokens (valid copy sources)
            
        Returns:
            pointer_probs: [B, T, prompt_len] probability distribution over prompt positions
            p_gen: [B, T, 1] probability of generating from vocabulary vs copying
        """
        B, T, D = hidden_states.shape
        prompt_states = hidden_states[:, :prompt_len, :]  # [B, P, D]

        # 1. Attention keys and queries
        queries = self.W_query(hidden_states)              # [B, T, D]
        keys = self.W_key(prompt_states)                  # [B, P, D]

        # Scaled dot-product pointer attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(D)  # [B, T, P]
        pointer_probs = F.softmax(scores, dim=-1)                         # [B, T, P]

        # 2. Context vector: weighted sum of prompt representations
        context = torch.bmm(pointer_probs, prompt_states)                 # [B, T, D]

        # 3. Generation gate: p_gen in (0, 1)
        gate_input = torch.cat([hidden_states, context], dim=-1)           # [B, T, 2*D]
        p_gen = torch.sigmoid(self.W_gate(gate_input))                     # [B, T, 1]

        return pointer_probs, p_gen


class ChakrPointerMicro(nn.Module):
    """
    ChakrMicro model extended with Pointer-Generator dual-distribution emission.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.core = ChakrMicro(config)
        self.pointer_head = PointerGeneratorHead(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes vocabulary logits, pointer distribution, copy gate, and combined distribution.
        
        Args:
            input_ids: [B, T]
            prompt_len: Number of prompt tokens
            
        Returns:
            vocab_logits: [B, T, vocab_size]
            pointer_probs: [B, T, prompt_len]
            p_gen: [B, T, 1]
            final_probs: [B, T, vocab_size] combined probability distribution
        """
        B, T = input_ids.shape
        # Get base model logits and hidden states
        vocab_logits, h = self.core(
            input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_hidden_states=True,
        )

        vocab_probs = F.softmax(vocab_logits, dim=-1)  # [B, T, V]
        pointer_probs, p_gen = self.pointer_head(h, prompt_len)  # [B, T, P], [B, T, 1]

        # Combine distributions:
        # final_probs = p_gen * vocab_probs + (1 - p_gen) * pointer_mapped_probs
        final_probs = p_gen * vocab_probs

        # Scatter pointer probabilities to corresponding token IDs in prompt
        # prompt_ids: [B, P]
        prompt_ids = input_ids[:, :prompt_len]
        # Expand prompt_ids to [B, T, P]
        expanded_prompt_ids = prompt_ids.unsqueeze(1).expand(B, T, prompt_len)
        pointer_contrib = (1.0 - p_gen) * pointer_probs  # [B, T, P]

        final_probs.scatter_add_(2, expanded_prompt_ids, pointer_contrib)

        return vocab_logits, pointer_probs, p_gen, final_probs
