"""Step 355: One-Block Integration with RecurrentAttentionCoreBlock.

Implements RecurrentAttentionChakrMicro:
An isolated candidate model where:
- Layer 0: frozen original canonical
- Layer 1: frozen original canonical
- Layer 2: frozen original canonical
- Layer 3: REPLACED BY RecurrentAttentionCoreBlock
- Layer 4: frozen original canonical
- Layer 5: frozen original canonical
- Embeddings and Final Norm: frozen original canonical

Key Guarantees:
- Baseline model remains bit-exact and untouched (Delta W = 0).
- Candidate clones canonical weights at initialization.
- RecurrentAttentionCoreBlock Layer 3 initial attention projections Q1, K1, V1, Out1
  are copied directly from canonical Layer 3 weights, and gamma_c2 is initialized to 0.0,
  ensuring initial identity / zero disruption.
- Recurrent state exists INSIDE the replacement block (no external wrappers).
- Trainable parameter count is strictly <= 250,000 (217,537 block + minimal binding readout = 230,594).
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
from chakrview.cognition.recurrent_attention_core_block import (
    RecurrentAttentionCoreBlock,
    CoreBlockTrace,
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


class RecurrentAttentionChakrMicro(nn.Module):
    """
    ChakrMicro candidate where Layer 3 is replaced with RecurrentAttentionCoreBlock.
    All other layers remain completely frozen.
    """

    def __init__(
        self,
        base_model: ChakrMicro,
        target_layer: int = 3,
        d_state: int = 48,
        use_shared_kv: bool = True,
        freeze_ffn: bool = True,
        enable_contextual_binding: bool = True,
        d_bind: int = 32,
    ) -> None:
        super().__init__()
        # Clone base model to enforce strict baseline isolation
        self.base_clone = copy.deepcopy(base_model)
        self.config = self.base_clone.config
        self.d_model = self.config.d_model
        self.target_layer = target_layer

        # Freeze everything in the base clone
        for p in self.base_clone.parameters():
            p.requires_grad = False

        # Create the replacement RecurrentAttentionCoreBlock
        self.recurrent_block = RecurrentAttentionCoreBlock(
            config=self.config,
            d_state=d_state,
            use_shared_kv=use_shared_kv,
            freeze_ffn=freeze_ffn,
        )

        # Copy canonical weights for Layer 3 into recurrent_block to ensure seamless initialization
        with torch.no_grad():
            orig_block = self.base_clone.layers[target_layer]
            self.recurrent_block.norm_1.weight.copy_(orig_block.norm_1.weight)
            self.recurrent_block.norm_2.weight.copy_(orig_block.norm_2.weight)
            self.recurrent_block.q1_proj.weight.copy_(orig_block.attn.q_proj.weight)
            self.recurrent_block.k1_proj.weight.copy_(orig_block.attn.k_proj.weight)
            self.recurrent_block.v1_proj.weight.copy_(orig_block.attn.v_proj.weight)
            self.recurrent_block.out1_proj.weight.copy_(orig_block.attn.out_proj.weight)
            self.recurrent_block.ffn.gate_proj.weight.copy_(orig_block.ffn.gate_proj.weight)
            self.recurrent_block.ffn.up_proj.weight.copy_(orig_block.ffn.up_proj.weight)
            self.recurrent_block.ffn.down_proj.weight.copy_(orig_block.ffn.down_proj.weight)

        # Minimal contextual binding readout
        if enable_contextual_binding:
            self.binding = DynamicContextualTokenBinding(d_model=self.d_model, d_bind=d_bind)
        else:
            self.binding = None

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        # Base model frozen parameters (excluding layer 3) + replacement block + binding
        return sum(p.numel() for p in self.parameters())

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor,
        query_pos: Optional[int] = None,
        override_s1: Optional[torch.Tensor] = None,
        override_q2: Optional[torch.Tensor] = None,
        disable_c1: bool = False,
        disable_c2: bool = False,
        bypass_state: bool = False,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[CoreBlockTrace]]:
        """
        Passes inputs through layers 0-2 (frozen), recurrent_block at layer 3, and layers 4-5 (frozen).
        """
        B, T = input_ids.shape
        x = self.base_clone.embedding(input_ids)

        layer_states = [x]
        core_trace = None

        for idx in range(self.config.n_layers):
            if idx == self.target_layer:
                # Replace Layer 3 with RecurrentAttentionCoreBlock
                x, core_trace = self.recurrent_block(
                    x=x,
                    query_pos=query_pos,
                    override_s1=override_s1,
                    override_q2=override_q2,
                    disable_c1=disable_c1,
                    disable_c2=disable_c2,
                    bypass_state=bypass_state,
                    return_trace=return_trace,
                )
            else:
                x = self.base_clone.layers[idx](x)
            layer_states.append(x)

        final_hidden = self.base_clone.final_norm(x)
        return final_hidden, layer_states, core_trace

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
        candidate_positions: Optional[torch.Tensor] = None,
        query_pos: Optional[int] = None,
        override_s1: Optional[torch.Tensor] = None,
        override_q2: Optional[torch.Tensor] = None,
        disable_c1: bool = False,
        disable_c2: bool = False,
        bypass_state: bool = False,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        """Full forward pass returning vocabulary logits, contextual logits, and trace."""
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)

        final_hidden, layer_states, trace = self.forward_hidden_states(
            input_ids=input_ids,
            query_pos=pos,
            override_s1=override_s1,
            override_q2=override_q2,
            disable_c1=disable_c1,
            disable_c2=disable_c2,
            bypass_state=bypass_state,
            return_trace=return_trace,
        )

        # Standard language logits via tied embedding
        vocab_logits = self.base_clone.lm_head(final_hidden)

        out = {
            "vocab_logits": vocab_logits,
            "final_hidden": final_hidden,
            "layer_states": layer_states,
            "trace": trace,
        }

        # Dynamic contextual binding logits
        if self.binding is not None and candidate_ids is not None and candidate_positions is not None:
            # candidate_positions: [B, N]
            B, N = candidate_positions.shape
            cand_states = torch.stack(
                [final_hidden[b, candidate_positions[b]] for b in range(B)],
                dim=0,
            )  # [B, N, d_model]
            cand_mask = torch.ones((B, N), dtype=torch.bool, device=final_hidden.device)
            val_rep = final_hidden[:, pos : pos + 1, :]  # [B, 1, d_model]
            bind_logits, bind_probs = self.binding(
                retrieved_value_rep=val_rep,
                candidate_token_states=cand_states,
                candidate_mask=cand_mask,
            )
            out["binding_logits"] = bind_logits
            out["candidate_probs"] = bind_probs

        return out
