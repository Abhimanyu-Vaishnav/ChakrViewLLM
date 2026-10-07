"""Step 346: Trainable Transformer Block Model.

Implements an isolated ChakrMicro candidate where ONE OR TWO complete TransformerBlock(s)
are trainable, while all other layers and embeddings remain strictly frozen.

Strict Properties:
- Clone-based parameter isolation: baseline weights remain completely frozen (Delta W = 0).
- NO external adapters, bridges, attractors, pointers, or external memories.
- Complete block training:
  * layer.norm_1 (RMSNorm weights)
  * layer.attn.q_proj, k_proj, v_proj, out_proj
  * layer.norm_2 (RMSNorm weights)
  * layer.ffn.gate_proj, up_proj, down_proj
  Total trainable parameters for 1 block = exactly 442,752 parameters.
  Total trainable parameters for 2 blocks = exactly 885,504 parameters.
- Exact identity at initialization: cloned model starts with bit-identical baseline weights.
- Includes minimal output binding readout to evaluate dynamic candidate token emission.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
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


class TrainableTransformerBlockCandidate(nn.Module):
    """
    ChakrMicro model wrapper where specified block indices (e.g. Layer 3)
    are fully trainable, while all other blocks remain frozen.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        trainable_layers: Optional[List[int]] = None,
        enable_contextual_binding: bool = True,
        d_bind: int = 32,
    ):
        super().__init__()
        # Deepcopy the base model to ensure strict isolation from canonical baseline
        self.model = copy.deepcopy(base_model)
        self.config = self.model.config
        self.d_model = self.config.d_model

        if trainable_layers is None:
            trainable_layers = [3]
        self.trainable_layers = sorted(trainable_layers)

        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze ONLY parameters in specified trainable_layers
        for l_idx in self.trainable_layers:
            block = self.model.layers[l_idx]
            for p in block.parameters():
                p.requires_grad = True

        # Minimal contextual binding readout for token emission benchmarking
        if enable_contextual_binding:
            self.binding = DynamicContextualTokenBinding(d_model=self.d_model, d_bind=d_bind)
        else:
            self.binding = None

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        bypass_trained_blocks: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Executes forward pass, optionally bypassing trained blocks."""
        B, T = input_ids.shape
        x = self.model.embedding(input_ids)
        states = [x]

        for i, layer in enumerate(self.model.layers):
            if bypass_trained_blocks and i in self.trainable_layers:
                # Bypass: skip block execution
                pass
            else:
                x = layer(x, layer_idx=i)
            states.append(x)

        final_h = self.model.final_norm(x)
        return final_h, states

    def forward(
        self,
        input_ids: torch.Tensor,
        bypass_trained_blocks: bool = False,
    ) -> torch.Tensor:
        final_h, _ = self.forward_hidden_states(input_ids, bypass_trained_blocks=bypass_trained_blocks)
        return self.model.lm_head(final_h)

    def compute_binding_scores(
        self,
        retrieved_val_rep: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.binding is None:
            raise ValueError("Dynamic contextual binding is not enabled in this model")
        return self.binding(retrieved_val_rep, candidate_states, candidate_mask)
